#include "DeferredRenderer.hpp"

#include <etna/GlobalContext.hpp>
#include <etna/BlockingTransferHelper.hpp>
#include <etna/PipelineManager.hpp>
#include <glm/ext.hpp>
#include <imgui.h>
#include <cstdlib>
#include <random>
#include <stdexcept>


DeferredRenderer::DeferredRenderer()
{
  static_assert(sizeof(Parameters) == 544);
  static_assert(sizeof(LocalLight) == 32);

  auto& ctx = etna::get_context();
  const auto properties = ctx.getPhysicalDevice().getProperties();
  const auto queue = ctx.getPhysicalDevice().getQueueFamilyProperties()[ctx.getQueueFamilyIdx()];
  timestampsSupported = queue.timestampValidBits != 0 && properties.limits.timestampComputeAndGraphics;
  timestampPeriod = properties.limits.timestampPeriod;
  timestampMask = queue.timestampValidBits == 64 ? ~uint64_t{0} : (uint64_t{1} << queue.timestampValidBits) - 1;

  sky = ctx.createImage({
    .extent = {2048, 2048, 1},
    .name = "sky_cubemap",
    .format = vk::Format::eR16G16B16A16Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
    .layers = 6,
    .flags = vk::ImageCreateFlagBits::eCubeCompatible,
  });
  skySampler = etna::Sampler({.name = "sky_sampler", .maxLod = 0});

  exposureStatistics = ctx.createBuffer({
    .size = 103 * sizeof(uint32_t),
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .name = "exposure_statistics",
  });

  colorLut = ctx.createImage({
    .extent = {32, 32, 32},
    .name = "color_lut",
    .format = vk::Format::eR8G8B8A8Unorm,
    .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
    .type = vk::ImageType::e3D,
  });

  std::vector<std::byte> lut(32 * 32 * 32 * 4);
  std::ifstream lutFile(std::string(GRAPHICS_COURSE_RESOURCES_ROOT) + "/textures/grading/lut.rgba", std::ios::binary);
  if (!lutFile.read(reinterpret_cast<char*>(lut.data()), static_cast<std::streamsize>(lut.size())))
    throw std::runtime_error("Cannot read color LUT");

  auto commands = ctx.createOneShotCmdMgr();
  auto staging = ctx.createBuffer({
    .size = lut.size(),
    .bufferUsage = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
    .name = "lut_upload",
  });
  auto cmd = commands->start();
  ETNA_CHECK_VK_RESULT(cmd.begin(vk::CommandBufferBeginInfo{}));

  for (size_t offset = 0; offset < lut.size(); offset += 65536)
    cmd.updateBuffer(staging.get(), offset, std::min(size_t{65536}, lut.size() - offset), lut.data() + offset);

  etna::set_state(cmd, staging.get(), vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
  etna::flush_barriers(cmd);
  etna::set_state(cmd, staging.get(), vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
  etna::set_state(cmd, colorLut.get(), vk::PipelineStageFlagBits2::eTransfer,
    vk::AccessFlagBits2::eTransferWrite, vk::ImageLayout::eTransferDstOptimal, vk::ImageAspectFlagBits::eColor);
  etna::flush_barriers(cmd);
  cmd.copyBufferToImage(staging.get(), colorLut.get(), vk::ImageLayout::eTransferDstOptimal,
    {vk::BufferImageCopy{.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
      .imageExtent = {32, 32, 32}}});
  ETNA_CHECK_VK_RESULT(cmd.end());
  commands->submitAndWait(std::move(cmd));

  sampler = etna::Sampler({
    .filter = vk::Filter::eNearest,
    .mipmapMode = vk::SamplerMipmapMode::eNearest,
    .name = "gbuffer_sampler",
    .maxLod = 0,
  });

  frames.emplace(ctx.getMainWorkCount(), [&](size_t) {
    FrameResources frame;
    frame.parameters = ctx.createBuffer({
      .size = sizeof(Parameters),
      .bufferUsage = vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst,
      .name = "deferred_parameters",
    });
    frame.lights = ctx.createBuffer({
      .size = sizeof(lights),
      .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      .name = "local_lights",
    });
    frame.clusters = ctx.createBuffer({
      .size = 16 * 9 * 24 * 1025 * sizeof(uint32_t),
      .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer,
      .name = "cluster_light_indices",
    });

    if (timestampsSupported)
      frame.queries = etna::unwrap_vk_result(ctx.getDevice().createQueryPoolUnique({
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = 9,
      }));

    return frame;
  });

  std::mt19937 random(42);
  std::uniform_real_distribution<float> unit(0, 1);

  for (auto& light : lights)
  {
    float angle = unit(random) * glm::two_pi<float>();
    float radius = std::sqrt(unit(random)) * 105;
    light.positionRadius = {std::cos(angle) * radius, 4 + unit(random) * 45,
      std::sin(angle) * radius, lightRadius};
    light.colorIntensity = {0.25f + unit(random) * 0.75f, 0.25f + unit(random) * 0.75f,
      0.25f + unit(random) * 0.75f, lightIntensity};
  }

  const std::array positions{glm::vec3{32, 35, 28}, glm::vec3{-20, 35, 12},
    glm::vec3{15, 35, -30}, glm::vec3{55, 35, 50}};

  for (size_t i = 0; i < positions.size(); ++i)
  {
    auto transform = glm::translate(glm::mat4{1}, positions[i]);
    transform = glm::rotate(transform, float(i) * 0.65f, glm::vec3{0, 1, 0});
    transform = glm::scale(transform, glm::vec3{22, 130, 18});
    parameters.decalInverse[i] = glm::inverse(transform);
  }

  benchmark = std::getenv("DEFERRED_BENCHMARK") != nullptr;

  if (const auto* value = std::getenv("DEFERRED_COMPACT"))
    compact = std::string(value) != "0";
  if (const auto* value = std::getenv("DEFERRED_CLUSTERED"))
    clustered = std::string(value) != "0";
  if (const auto* value = std::getenv("DEFERRED_DECALS"))
    decalsEnabled = std::string(value) != "0";

  if (benchmark)
  {
    if (!timestampsSupported)
      throw std::runtime_error("GPU timestamps are required for the deferred benchmark");

    benchmarkFile.open("deferred_benchmark.csv");
    if (!benchmarkFile)
      throw std::runtime_error("Cannot write deferred_benchmark.csv");

    benchmarkFile << "compact,clustered,lights,depth_ms,gbuffer_ms,decals_ms,clusters_ms,lighting_ms,total_ms\n";
  }
}

std::vector<vk::Format> DeferredRenderer::formats(bool compact)
{
  if (compact)
    return {vk::Format::eR8G8B8A8Srgb, vk::Format::eR16G16Unorm, vk::Format::eR8G8B8A8Unorm};

  return {vk::Format::eR32G32B32A32Sfloat, vk::Format::eR32G32B32A32Sfloat,
    vk::Format::eR32G32B32A32Sfloat};
}

void DeferredRenderer::allocateResources(glm::uvec2 size)
{
  resolution = size;
  hdr = etna::get_context().createImage({
    .extent = {size.x, size.y, 1},
    .name = "hdr",
    .format = vk::Format::eB10G11R11UfloatPack32,
    .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
  });
  auto& ctx = etna::get_context();

  depth = ctx.createImage({
    .extent = {size.x, size.y, 1},
    .name = "gbuffer_depth",
    .format = vk::Format::eD32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
  });

  allocateGbuffer();
}

void DeferredRenderer::allocateGbuffer()
{
  for (auto& set : buffers)
    for (size_t i = 0; i < set.size(); ++i)
      set[i] = etna::get_context().createImage({
        .extent = {resolution.x, resolution.y, 1},
        .name = "gbuffer",
        .format = formats(compact)[i],
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
      });

  allocatedCompact = compact;
}

void DeferredRenderer::loadShaders()
{
  etna::create_program("exposure", {TERRAIN_SHADERS_ROOT "exposure.comp.spv"});
  etna::create_program("post", {TERRAIN_SHADERS_ROOT "heightmap.vert.spv", TERRAIN_SHADERS_ROOT "post.frag.spv"});
  etna::create_program("sky",
    {TERRAIN_SHADERS_ROOT "heightmap.vert.spv", TERRAIN_SHADERS_ROOT "sky.frag.spv"});
  etna::create_program("deferred_lighting",
    {TERRAIN_SHADERS_ROOT "heightmap.vert.spv", TERRAIN_SHADERS_ROOT "lighting.frag.spv"});
  etna::create_program("decals",
    {TERRAIN_SHADERS_ROOT "heightmap.vert.spv", TERRAIN_SHADERS_ROOT "decals.frag.spv"});
  etna::create_program("clusters", {TERRAIN_SHADERS_ROOT "clusters.comp.spv"});
}

void DeferredRenderer::setupPipelines(vk::Format output_format)
{
  auto& manager = etna::get_context().getPipelineManager();
  outputSrgb = output_format == vk::Format::eB8G8R8A8Srgb || output_format == vk::Format::eR8G8B8A8Srgb;

  skyPipeline = manager.createGraphicsPipeline("sky", {
    .depthConfig = {.depthTestEnable = vk::False, .depthWriteEnable = vk::False},
    .fragmentShaderOutput = {.colorAttachmentFormats = {vk::Format::eR16G16B16A16Sfloat}},
  });

  exposurePipeline = manager.createComputePipeline("exposure", {});
  postPipeline = manager.createGraphicsPipeline("post", {
    .depthConfig = {.depthTestEnable = vk::False, .depthWriteEnable = vk::False},
    .fragmentShaderOutput = {.colorAttachmentFormats = {output_format}},
  });

  lightingPipeline = manager.createGraphicsPipeline("deferred_lighting", {
    .depthConfig = {.depthTestEnable = vk::False, .depthWriteEnable = vk::False},
    .fragmentShaderOutput = {.colorAttachmentFormats = {vk::Format::eB10G11R11UfloatPack32}},
  });

  for (size_t mode = 0; mode < 2; ++mode)
  {
    etna::GraphicsPipeline::CreateInfo info;
    info.depthConfig.depthTestEnable = vk::False;
    info.depthConfig.depthWriteEnable = vk::False;
    info.blendingConfig.logicOp = vk::LogicOp::eCopy;
    info.blendingConfig.attachments.resize(3, info.blendingConfig.attachments.front());
    info.fragmentShaderOutput.colorAttachmentFormats = formats(mode != 0);

    decalPipelines[mode] = manager.createGraphicsPipeline("decals", info);
  }

  clusterPipeline = manager.createComputePipeline("clusters", {});
}

std::vector<etna::RenderTargetState::AttachmentParams> DeferredRenderer::attachments(bool decals)
{
  std::vector<etna::RenderTargetState::AttachmentParams> result;

  for (auto& image : buffers[decals])
    result.push_back({.image = image.get(), .view = image.getView({})});

  return result;
}

etna::RenderTargetState::AttachmentParams DeferredRenderer::depthAttachment(bool load)
{
  return {.image = depth.get(), .view = depth.getView({}),
    .loadOp = load ? vk::AttachmentLoadOp::eLoad : vk::AttachmentLoadOp::eClear};
}

void DeferredRenderer::update(const FramePacket& packet)
{
  deltaTime = std::clamp(packet.currentTime - previousTime, 0.0f, 0.25f);
  previousTime = packet.currentTime;
  const auto projection = packet.mainCam.projTm(float(resolution.x) / float(resolution.y));
  parameters.view = packet.mainCam.viewTm();
  parameters.inverseViewProjection = glm::inverse(projection * parameters.view);
  parameters.inverseProjection = glm::inverse(projection);
  parameters.screen = {resolution.x, resolution.y, packet.mainCam.zNear, packet.mainCam.zFar};
}

void DeferredRenderer::drawGui()
{
  if (ImGui::CollapsingHeader("Postprocessing", ImGuiTreeNodeFlags_DefaultOpen))
  {
    ImGui::Checkbox("Tone mapping", &toneMapping);
    ImGui::Checkbox("Color grading", &colorGrading);
    ImGui::Checkbox("Shared histogram", &sharedHistogram);
    ImGui::Text("Exposure GPU: %.3f ms", histogramMs);
    ImGui::SliderFloat("Exposure compensation", &exposureCompensation, -4, 4);
  }

  if (ImGui::CollapsingHeader("Sky / clouds"))
  {
    skyDirty |= ImGui::SliderFloat("Cloud coverage", &skyParameters.clouds.x, 0, 1);
    skyDirty |= ImGui::SliderFloat("Cloud density", &skyParameters.clouds.y, 0, 0.03f, "%.4f");
    skyDirty |= ImGui::SliderFloat("Cloud scale", &skyParameters.clouds.z, 50, 1500);
    skyDirty |= ImGui::SliderFloat("Layer bottom", &skyParameters.layer.x, 100, 2000);
    skyParameters.layer.y = std::max(skyParameters.layer.y, skyParameters.layer.x + 50);
    skyDirty |= ImGui::SliderFloat("Layer top", &skyParameters.layer.y,
      skyParameters.layer.x + 50, 4000);

    int steps = static_cast<int>(skyParameters.clouds.w);
    int shadowSteps = static_cast<int>(skyParameters.layer.z);
    skyDirty |= ImGui::SliderInt("Sky ray steps", &steps, 8, 96);
    skyDirty |= ImGui::SliderInt("Sky shadow steps", &shadowSteps, 1, 16);
    skyParameters.clouds.w = static_cast<float>(steps);
    skyParameters.layer.z = static_cast<float>(shadowSteps);
  }

  if (!ImGui::CollapsingHeader("Deferred", ImGuiTreeNodeFlags_DefaultOpen))
    return;

  ImGui::Checkbox("Compact G-buffer", &compact);
  ImGui::Checkbox("Clustered lighting", &clustered);
  ImGui::SliderInt("Local lights", &lightCount, 0, 1024);
  ImGui::SliderFloat("Light intensity", &lightIntensity, 0, 400);
  ImGui::SliderFloat("Light radius", &lightRadius, 5, 60);
  ImGui::Checkbox("Decals", &decalsEnabled);
  ImGui::SliderFloat("Decal opacity", &parameters.decal.x, 0, 1);
  ImGui::SliderFloat("Decal albedo", &parameters.decal.y, 0, 1);
  ImGui::SliderFloat("Decal normal", &parameters.decal.z, 0, 1);
  ImGui::SliderFloat("Decal roughness", &parameters.decal.w, 0, 1);
  ImGui::Text("G-buffer: %d bytes/pixel", compact ? 16 : 52);

  if (timestampsSupported)
  {
    ImGui::Text("Depth: %.3f ms | G-buffer: %.3f ms", timings[0], timings[1]);
    ImGui::Text("Decals: %.3f ms | Clusters: %.3f ms", timings[2], timings[3]);
    ImGui::Text("Lighting: %.3f ms | Total: %.3f ms", timings[4], timings[5]);
  }
}

void DeferredRenderer::beginFrame()
{
  auto& frame = frames->get();

  if (frame.hasQueries)
  {
    std::array<uint64_t, 9> values{};
    const auto result = etna::get_context().getDevice().getQueryPoolResults(frame.queries.get(),
      0, 9, sizeof(values), values.data(), sizeof(uint64_t), vk::QueryResultFlagBits::e64);

    if (result == vk::Result::eSuccess)
    {
      histogramMs = double((values[8] - values[7]) & timestampMask) * timestampPeriod / 1e6;
      if (std::getenv("EXPOSURE_BENCHMARK"))
      {
        static int samples = 0;
        static double total = 0;
        ++samples;

        if (samples > 30)
          total += histogramMs;

        if (samples == 150)
        {
          spdlog::info("Exposure benchmark: shared={}, mean_ms={}", sharedHistogram, total / 120);
          if (!sharedHistogram)
            benchmarkDone = true;
          sharedHistogram = false;
          samples = 0;
          total = 0;
        }
      }

      for (size_t i = 0; i < 5; ++i)
        timings[i] = double((values[i + 2] - values[i + 1]) & timestampMask) * timestampPeriod / 1e6;
      timings[5] = double((values[6] - values[0]) & timestampMask) * timestampPeriod / 1e6;

      if (benchmark && !benchmarkDone && frame.benchmarkPhase == benchmarkPhase)
      {
        ++benchmarkSamples;

        if (benchmarkSamples > 30)
          for (size_t i = 0; i < totals.size(); ++i)
            totals[i] += timings[i];

        if (benchmarkSamples == 150)
        {
          benchmarkFile << compact << ',' << clustered << ',' << lightCount;
          for (double value : totals)
            benchmarkFile << ',' << value / 120;
          benchmarkFile << '\n';
          benchmarkFile.flush();
          totals = {};
          benchmarkSamples = 0;
          benchmarkDone = ++benchmarkPhase == 12;
        }
      }
    }

    frame.hasQueries = false;
  }

  if (benchmark && !benchmarkDone)
  {
    compact = (benchmarkPhase / 2) % 2 != 0;
    clustered = benchmarkPhase % 2 != 0;
    lightCount = std::array{64, 256, 1024}[benchmarkPhase / 4];
  }

  if (allocatedCompact != compact)
  {
    ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());
    allocateGbuffer();
  }
}

void DeferredRenderer::timestamp(vk::CommandBuffer cmd, uint32_t index)
{
  if (timestampsSupported)
    cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, frames->get().queries.get(), index);
}

void DeferredRenderer::begin(vk::CommandBuffer cmd)
{
  if (timestampsSupported)
    cmd.resetQueryPool(frames->get().queries.get(), 0, 9);

  timestamp(cmd, 0);
}

void DeferredRenderer::finish(vk::CommandBuffer cmd, vk::Image output, vk::ImageView output_view,
  glm::vec4 camera, glm::vec4 sun, glm::vec4 light,
  const etna::Image& albedo, const etna::Image& roughness, const etna::Image& height,
  vk::Sampler material_sampler)
{
  auto& frame = frames->get();
  bakeSky(cmd, sun, light);
  parameters.cameraExposure = camera;
  parameters.cameraExposure.w *= outputSrgb ? 1 : -1;
  parameters.sun = sun;
  parameters.light = light;
  parameters.options = {lightCount, clustered ? 1 : 0, 0, 0};

  for (auto& source : lights)
  {
    source.positionRadius.w = lightRadius;
    source.colorIntensity.w = lightIntensity;
  }

  for (const auto* buffer : {&frame.parameters, &frame.lights})
    etna::set_state(cmd, buffer->get(), vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
  etna::flush_barriers(cmd);
  cmd.updateBuffer(frame.parameters.get(), 0, sizeof(parameters), &parameters);
  cmd.updateBuffer(frame.lights.get(), 0, sizeof(lights), lights.data());

  for (const auto* buffer : {&frame.parameters, &frame.lights})
    etna::set_state(cmd, buffer->get(),
      vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eComputeShader,
      buffer == &frame.parameters ? vk::AccessFlagBits2::eUniformRead : vk::AccessFlagBits2::eShaderStorageRead);
  etna::flush_barriers(cmd);

  const auto imageBinding = [&](const etna::Image& image) {
    return image.genBinding(sampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal);
  };

  if (decalsEnabled)
  {
    auto& source = buffers[0];
    const auto program = etna::get_shader_program("decals");
    auto set = etna::create_descriptor_set(program.getDescriptorLayoutId(0), cmd, {
      etna::Binding{0, imageBinding(source[0])},
      etna::Binding{1, imageBinding(source[1])},
      etna::Binding{2, imageBinding(source[2])},
      etna::Binding{3, imageBinding(depth)},
      etna::Binding{4, frame.parameters.genBinding()},
      etna::Binding{7, albedo.genBinding(material_sampler, vk::ImageLayout::eShaderReadOnlyOptimal,
        {.type = vk::ImageViewType::e2DArray})},
      etna::Binding{8, roughness.genBinding(material_sampler, vk::ImageLayout::eShaderReadOnlyOptimal,
        {.type = vk::ImageViewType::e2DArray})},
      etna::Binding{9, height.genBinding(material_sampler, vk::ImageLayout::eShaderReadOnlyOptimal,
        {.type = vk::ImageViewType::e2DArray})},
    });

    etna::RenderTargetState targets(cmd, {{0, 0}, {resolution.x, resolution.y}}, attachments(true), {});
    auto& pipeline = decalPipelines[compact];
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.getVkPipeline());
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline.getVkPipelineLayout(),
      0, {set.getVkSet()}, {});
    cmd.draw(3, 1, 0, 0);
  }

  timestamp(cmd, 4);

  if (clustered)
  {
    etna::set_state(cmd, frame.clusters.get(), vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderStorageWrite);
    etna::flush_barriers(cmd);
    const auto program = etna::get_shader_program("clusters");
    auto set = etna::create_descriptor_set(program.getDescriptorLayoutId(0), cmd, {
      etna::Binding{4, frame.parameters.genBinding()},
      etna::Binding{5, frame.lights.genBinding()},
      etna::Binding{6, frame.clusters.genBinding()},
    });
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, clusterPipeline.getVkPipeline());
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, clusterPipeline.getVkPipelineLayout(),
      0, {set.getVkSet()}, {});
    cmd.dispatch(16 * 9 * 24 / 64, 1, 1);
  }

  etna::set_state(cmd, frame.clusters.get(), vk::PipelineStageFlagBits2::eFragmentShader,
    vk::AccessFlagBits2::eShaderStorageRead);
  etna::flush_barriers(cmd);
  timestamp(cmd, 5);

  auto& source = buffers[decalsEnabled];
  const auto program = etna::get_shader_program("deferred_lighting");
  auto set = etna::create_descriptor_set(program.getDescriptorLayoutId(0), cmd, {
    etna::Binding{0, imageBinding(source[0])},
    etna::Binding{1, imageBinding(source[1])},
    etna::Binding{2, imageBinding(source[2])},
    etna::Binding{3, imageBinding(depth)},
    etna::Binding{4, frame.parameters.genBinding()},
    etna::Binding{5, frame.lights.genBinding()},
    etna::Binding{6, frame.clusters.genBinding()},
    etna::Binding{10, sky.genBinding(skySampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal,
      {.type = vk::ImageViewType::eCube})},
  });

  {
    etna::RenderTargetState targets(cmd, {{0, 0}, {resolution.x, resolution.y}},
      {{.image = hdr.get(), .view = hdr.getView({})}}, {});
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, lightingPipeline.getVkPipeline());
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, lightingPipeline.getVkPipelineLayout(),
      0, {set.getVkSet()}, {});
    cmd.draw(3, 1, 0, 0);
  }

  postprocess(cmd, output, output_view);
  timestamp(cmd, 6);
  frame.hasQueries = timestampsSupported;
  frame.benchmarkPhase = benchmarkPhase;
}


void DeferredRenderer::bakeSky(vk::CommandBuffer cmd, glm::vec4 sun, glm::vec4 color)
{
  color.w = 0;

  if (!skyDirty && skyParameters.sun == sun && skyParameters.color == color)
    return;

  skyParameters.sun = sun;
  skyParameters.color = color;

  for (uint32_t face = 0; face < 6; ++face)
  {
    skyParameters.layer.w = static_cast<float>(face);
    etna::RenderTargetState target(cmd, {{0, 0}, {2048, 2048}},
      {{.image = sky.get(),
        .view = sky.getView({.baseLayer = face, .layerCount = 1, .type = vk::ImageViewType::e2D})}}, {});
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, skyPipeline.getVkPipeline());
    cmd.pushConstants<SkyParameters>(skyPipeline.getVkPipelineLayout(),
      vk::ShaderStageFlagBits::eFragment, 0, {skyParameters});
    cmd.draw(3, 1, 0, 0);
  }

  skyDirty = false;
}


void DeferredRenderer::postprocess(vk::CommandBuffer cmd, vk::Image output, vk::ImageView view)
{
  etna::set_state(cmd, exposureStatistics.get(), vk::PipelineStageFlagBits2::eTransfer,
    vk::AccessFlagBits2::eTransferWrite);
  etna::flush_barriers(cmd);
  cmd.fillBuffer(exposureStatistics.get(), 4, exposureInitialized ? 404 : 408, 0);
  uint32_t minimum = 0x7f7fffff;
  cmd.updateBuffer(exposureStatistics.get(), 0, sizeof(minimum), &minimum);
  exposureInitialized = true;

  struct ComputeSettings
  {
    int phase;
    int shared;
    float dt;
    float compensation;
  } settings{0, sharedHistogram ? 1 : 0, deltaTime, 0};

  auto program = etna::get_shader_program("exposure");
  auto set = etna::create_descriptor_set(program.getDescriptorLayoutId(0), cmd, {
    etna::Binding{0, hdr.genBinding(sampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
    etna::Binding{1, exposureStatistics.genBinding()},
  });

  timestamp(cmd, 7);

  for (int phase = 0; phase < 3; ++phase)
  {
    etna::set_state(cmd, exposureStatistics.get(), vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);
    etna::flush_barriers(cmd);
    settings.phase = phase;
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, exposurePipeline.getVkPipeline());
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, exposurePipeline.getVkPipelineLayout(),
      0, {set.getVkSet()}, {});
    cmd.pushConstants<ComputeSettings>(exposurePipeline.getVkPipelineLayout(),
      vk::ShaderStageFlagBits::eCompute, 0, {settings});
    cmd.dispatch(phase == 2 ? 1 : (resolution.x + 15) / 16,
      phase == 2 ? 1 : (resolution.y + 15) / 16, 1);
  }

  etna::set_state(cmd, exposureStatistics.get(), vk::PipelineStageFlagBits2::eFragmentShader,
    vk::AccessFlagBits2::eShaderStorageRead);
  etna::flush_barriers(cmd);

  timestamp(cmd, 8);

  auto post = etna::get_shader_program("post");
  auto postSet = etna::create_descriptor_set(post.getDescriptorLayoutId(0), cmd, {
    etna::Binding{0, hdr.genBinding(sampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
    etna::Binding{1, exposureStatistics.genBinding()},
    etna::Binding{2, colorLut.genBinding(skySampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal,
      {.type = vk::ImageViewType::e3D})},
  });

  struct PostSettings
  {
    int toneMapping;
    int grading;
    float compensation;
    int srgb;
  } postSettings{toneMapping ? 1 : 0, colorGrading ? 1 : 0, exposureCompensation, outputSrgb ? 1 : 0};

  etna::RenderTargetState target(cmd, {{0, 0}, {resolution.x, resolution.y}},
    {{.image = output, .view = view}}, {});
  cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, postPipeline.getVkPipeline());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, postPipeline.getVkPipelineLayout(),
    0, {postSet.getVkSet()}, {});
  cmd.pushConstants<PostSettings>(postPipeline.getVkPipelineLayout(),
    vk::ShaderStageFlagBits::eFragment, 0, {postSettings});
  cmd.draw(3, 1, 0, 0);
}
