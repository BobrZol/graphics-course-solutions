#include "WorldRenderer.hpp"

#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/RenderTargetStates.hpp>
#include <etna/BlockingTransferHelper.hpp>
#include <etna/Profiling.hpp>
#include <glm/ext.hpp>
#include <imgui.h>
#include <stb_image.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>


namespace
{
void generateMipmaps(vk::CommandBuffer cmd, const etna::Image& image)
{
  auto extent = image.getExtent();
  for (uint32_t mip = 1; mip < image.getMipLevelCount(); ++mip)
  {
    etna::set_state(
      cmd, image.get(), vk::PipelineStageFlagBits2::eTransfer,
      vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite,
      vk::ImageLayout::eGeneral, vk::ImageAspectFlagBits::eColor, ForceSetState::eTrue);
    etna::flush_barriers(cmd);

    const auto width = std::max(1u, extent.width / 2);
    const auto height = std::max(1u, extent.height / 2);
    vk::ImageBlit blit{
      .srcSubresource = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .mipLevel = mip - 1,
        .layerCount = 4,
      },
      .dstSubresource = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .mipLevel = mip,
        .layerCount = 4,
      },
    };
    blit.srcOffsets[1] = vk::Offset3D{
      static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height), 1};
    blit.dstOffsets[1] = vk::Offset3D{
      static_cast<int32_t>(width), static_cast<int32_t>(height), 1};
    cmd.blitImage(
      image.get(), vk::ImageLayout::eGeneral, image.get(), vk::ImageLayout::eGeneral,
      {blit}, vk::Filter::eLinear);
    extent = vk::Extent3D{width, height, 1};
  }
}
}

WorldRenderer::WorldRenderer()
{
  auto& ctx = etna::get_context();
  heightmap = ctx.createImage(etna::Image::CreateInfo{
    .extent = {1024, 1024, 1},
    .name = "terrain_heightmap",
    .format = vk::Format::eR32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
  });
  heightSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .addressMode = vk::SamplerAddressMode::eClampToEdge,
    .name = "height_sampler",
    .maxLod = 0,
  });
  materialSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .addressMode = vk::SamplerAddressMode::eRepeat,
    .name = "material_sampler",
    .maxLod = 10,
  });

  auto commands = ctx.createOneShotCmdMgr();
  etna::BlockingTransferHelper transfer({.stagingSize = 1024 * 1024 * 4});
  std::vector<uint32_t> meshIndices;
  meshIndices.reserve(256 * 256 * 6);
  for (uint32_t z = 0; z < 256; ++z)
    for (uint32_t x = 0; x < 256; ++x)
    {
      const uint32_t a = z * 257 + x;
      meshIndices.insert(meshIndices.end(), {a, a + 257, a + 1, a + 1, a + 257, a + 258});
    }
  indices = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = meshIndices.size() * sizeof(uint32_t),
    .bufferUsage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "terrain_indices",
  });
  transfer.uploadBuffer<uint32_t>(*commands, indices, 0, meshIndices);

  const std::array names{"sand_03", "sparse_grass", "rock_06", "snow_01"};
  const std::array suffixes{"diff", "rough", "disp"};
  const std::array images{&albedoMaps, &roughnessMaps, &detailMaps};
  for (size_t map = 0; map < images.size(); ++map)
  {
    *images[map] = ctx.createImage(etna::Image::CreateInfo{
      .extent = {1024, 1024, 1},
      .name = suffixes[map],
      .format = map == 0 ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm,
      .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst |
        vk::ImageUsageFlagBits::eTransferSrc,
      .layers = 4,
      .mipLevels = 11,
    });
    for (uint32_t layer = 0; layer < names.size(); ++layer)
    {
      const std::string path = std::string(GRAPHICS_COURSE_RESOURCES_ROOT) +
        "/textures/terrain/" + names[layer] + "_" + suffixes[map] + "_1k.jpg";
      int width = 0;
      int height = 0;
      int channels = 0;
      std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(
        stbi_load(path.c_str(), &width, &height, &channels, 4), stbi_image_free);
      if (!pixels || width != 1024 || height != 1024)
        throw std::runtime_error("Cannot load 1024x1024 terrain texture: " + path);
      transfer.uploadImage(
        *commands, *images[map], 0, layer,
        {reinterpret_cast<const std::byte*>(pixels.get()), 1024 * 1024 * 4});
    }
  }
  auto cmd = commands->start();
  ETNA_CHECK_VK_RESULT(cmd.begin(vk::CommandBufferBeginInfo{}));
  for (const auto* image : images)
    generateMipmaps(cmd, *image);
  ETNA_CHECK_VK_RESULT(cmd.end());
  commands->submitAndWait(std::move(cmd));
}

void WorldRenderer::allocateResources(glm::uvec2 swapchain_resolution)
{
  resolution = swapchain_resolution;
  mainViewDepth = etna::get_context().createImage(etna::Image::CreateInfo{
    .extent = {resolution.x, resolution.y, 1},
    .name = "main_view_depth",
    .format = vk::Format::eD32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
  });
}

void WorldRenderer::loadShaders()
{
  etna::create_program(
    "terrain", {TERRAIN_SHADERS_ROOT "static_mesh.vert.spv",
                TERRAIN_SHADERS_ROOT "static_mesh.frag.spv"});
  etna::create_program(
    "heightmap", {TERRAIN_SHADERS_ROOT "heightmap.vert.spv",
                  TERRAIN_SHADERS_ROOT "heightmap.frag.spv"});
}

void WorldRenderer::setupPipelines(vk::Format swapchain_format)
{
  outputSrgb = swapchain_format == vk::Format::eB8G8R8A8Srgb ||
    swapchain_format == vk::Format::eR8G8B8A8Srgb;
  auto& manager = etna::get_context().getPipelineManager();
  terrainPipeline = {};
  terrainPipeline = manager.createGraphicsPipeline("terrain", etna::GraphicsPipeline::CreateInfo{
    .fragmentShaderOutput = {
      .colorAttachmentFormats = {swapchain_format},
      .depthAttachmentFormat = vk::Format::eD32Sfloat,
    },
  });
  heightmapPipeline = {};
  heightmapPipeline = manager.createGraphicsPipeline("heightmap", etna::GraphicsPipeline::CreateInfo{
    .depthConfig = {.depthTestEnable = vk::False, .depthWriteEnable = vk::False},
    .fragmentShaderOutput = {.colorAttachmentFormats = {vk::Format::eR32Sfloat}},
  });
}

void WorldRenderer::debugInput(const Keyboard& kb)
{
  if (kb[KeyboardKey::kB] == ButtonState::Falling)
    heightmapDirty = true;
}

void WorldRenderer::update(const FramePacket& packet)
{
  params.projView = packet.mainCam.projTm(float(resolution.x) / float(resolution.y)) *
    packet.mainCam.viewTm();
  params.cameraExposure = glm::vec4(packet.mainCam.position, params.cameraExposure.w);
}

void WorldRenderer::drawGui()
{
  ImGui::SetNextWindowSize({330, 0}, ImGuiCond_FirstUseEver);
  ImGui::Begin("Terrain");
  if (ImGui::CollapsingHeader("Noise / island", ImGuiTreeNodeFlags_DefaultOpen))
  {
    heightmapDirty |= ImGui::SliderFloat("Frequency", &noiseParams.noise.x, 1, 12);
    heightmapDirty |= ImGui::SliderFloat("Persistence", &noiseParams.noise.y, 0.2f, 0.8f);
    heightmapDirty |= ImGui::SliderFloat("Lacunarity", &noiseParams.noise.z, 1.5f, 3);
    int octaves = static_cast<int>(noiseParams.noise.w);
    heightmapDirty |= ImGui::SliderInt("Octaves", &octaves, 1, 10);
    noiseParams.noise.w = static_cast<float>(octaves);
    int seed = static_cast<int>(noiseParams.island.w);
    heightmapDirty |= ImGui::SliderInt("Seed", &seed, 0, 1000);
    noiseParams.island.w = static_cast<float>(seed);
    heightmapDirty |= ImGui::SliderFloat("Island radius", &noiseParams.island.x, 0.5f, 1);
    heightmapDirty |= ImGui::SliderFloat("Coast width", &noiseParams.island.y, 0.15f, 0.8f);
    heightmapDirty |= ImGui::SliderFloat("Height curve", &noiseParams.island.z, 0.7f, 3);
    ImGui::SliderFloat("Height", &params.terrainSettings.y, 10, 120);
  }
  if (ImGui::CollapsingHeader("Materials", ImGuiTreeNodeFlags_DefaultOpen))
  {
    ImGui::SliderFloat("Texture tiling", &params.terrainSettings.z, 4, 100);
    ImGui::SliderFloat("Height detail", &params.terrainSettings.w, 0, 1);
  }
  if (ImGui::CollapsingHeader("Lighting", ImGuiTreeNodeFlags_DefaultOpen))
  {
    ImGui::SliderFloat("Sun azimuth", &sunAzimuth, -180, 180);
    ImGui::SliderFloat("Sun elevation", &sunElevation, 5, 90);
    ImGui::SliderFloat("Sun intensity", &params.sun.w, 0, 10);
    ImGui::ColorEdit3("Sun color", &params.light.x);
    ImGui::SliderFloat("Ambient", &params.light.w, 0, 1);
    ImGui::SliderFloat("Exposure", &params.cameraExposure.w, 0.2f, 3);
  }
  ImGui::TextUnformatted("WASD: move | R/F: up/down\nRMB: mouse look | Shift: faster\nWheel: FOV | Esc: close");
  ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
  ImGui::End();
}

void WorldRenderer::generateHeightmap(vk::CommandBuffer cmd)
{
  etna::RenderTargetState targets(
    cmd, {{0, 0}, {1024, 1024}},
    {{.image = heightmap.get(), .view = heightmap.getView({})}}, {});
  cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, heightmapPipeline.getVkPipeline());
  cmd.pushConstants<NoiseParams>(
    heightmapPipeline.getVkPipelineLayout(), vk::ShaderStageFlagBits::eFragment, 0, {noiseParams});
  cmd.draw(3, 1, 0, 0);
  heightmapDirty = false;
}

void WorldRenderer::renderWorld(
  vk::CommandBuffer cmd, vk::Image target_image, vk::ImageView target_image_view)
{
  ETNA_PROFILE_GPU(cmd, renderWorld);
  if (heightmapDirty)
    generateHeightmap(cmd);

  const auto shader = etna::get_shader_program("terrain");
  auto set = etna::create_descriptor_set(shader.getDescriptorLayoutId(0), cmd, {
    etna::Binding{0, heightmap.genBinding(heightSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
    etna::Binding{1, albedoMaps.genBinding(materialSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal,
      {.type = vk::ImageViewType::e2DArray})},
    etna::Binding{2, roughnessMaps.genBinding(materialSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal,
      {.type = vk::ImageViewType::e2DArray})},
    etna::Binding{3, detailMaps.genBinding(materialSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal,
      {.type = vk::ImageViewType::e2DArray})},
  });
  const float elevation = glm::radians(sunElevation);
  const float azimuth = glm::radians(sunAzimuth);
  params.sun = glm::vec4(
    std::cos(elevation) * std::cos(azimuth), std::sin(elevation),
    std::cos(elevation) * std::sin(azimuth), params.sun.w);
  auto drawParams = params;
  drawParams.cameraExposure.w *= outputSrgb ? 1 : -1;

  etna::RenderTargetState targets(
    cmd, {{0, 0}, {resolution.x, resolution.y}},
    {{.image = target_image, .view = target_image_view,
      .clearColorValue = std::array<float, 4>{0.32f, 0.52f, 0.72f, 1}}},
    {.image = mainViewDepth.get(), .view = mainViewDepth.getView({})});
  cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, terrainPipeline.getVkPipeline());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
    terrainPipeline.getVkPipelineLayout(), 0, {set.getVkSet()}, {});
  cmd.bindIndexBuffer(indices.get(), 0, vk::IndexType::eUint32);
  cmd.pushConstants<PushConstants>(terrainPipeline.getVkPipelineLayout(),
    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, {drawParams});
  cmd.drawIndexed(256 * 256 * 6, 1, 0, 0, 0);
  drawParams.terrainSettings.y = -1;
  cmd.pushConstants<PushConstants>(terrainPipeline.getVkPipelineLayout(),
    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, {drawParams});
  cmd.draw(6, 1, 0, 0);
}
