#include "DeferredRenderer.hpp"

#include <etna/GlobalContext.hpp>
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
        .queryCount = 7,
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

  lightingPipeline = manager.createGraphicsPipeline("deferred_lighting", {
    .depthConfig = {.depthTestEnable = vk::False, .depthWriteEnable = vk::False},
    .fragmentShaderOutput = {.colorAttachmentFormats = {output_format}},
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
  const auto projection = packet.mainCam.projTm(float(resolution.x) / float(resolution.y));
  parameters.view = packet.mainCam.viewTm();
  parameters.inverseViewProjection = glm::inverse(projection * parameters.view);
  parameters.inverseProjection = glm::inverse(projection);
  parameters.screen = {resolution.x, resolution.y, packet.mainCam.zNear, packet.mainCam.zFar};
}

void DeferredRenderer::drawGui()
{
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
    std::array<uint64_t, 7> values{};
    const auto result = etna::get_context().getDevice().getQueryPoolResults(frame.queries.get(),
      0, 7, sizeof(values), values.data(), sizeof(uint64_t), vk::QueryResultFlagBits::e64);

    if (result == vk::Result::eSuccess)
    {
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
    cmd.resetQueryPool(frames->get().queries.get(), 0, 7);

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
      {{.image = output, .view = output_view}}, {});
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, lightingPipeline.getVkPipeline());
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, lightingPipeline.getVkPipelineLayout(),
      0, {set.getVkSet()}, {});
    cmd.draw(3, 1, 0, 0);
  }

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
