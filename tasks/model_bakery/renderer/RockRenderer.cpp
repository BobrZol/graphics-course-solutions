#include "RockRenderer.hpp"

#include <etna/BlockingTransferHelper.hpp>
#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/Profiling.hpp>
#include <glm/ext.hpp>
#include <tiny_gltf.h>
#include <imgui.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>


namespace
{
uint32_t hash(uint32_t value)
{
  value = (value ^ (value >> 16)) * 2246822519u;
  value = (value ^ (value >> 13)) * 3266489917u;
  return value ^ (value >> 16);
}

float random(uint32_t seed, uint32_t channel)
{
  return static_cast<float>(hash(seed ^ (channel * 1597334677u)) >> 8) / 16777216.0f;
}

const unsigned char* element(const tinygltf::Model& model, const tinygltf::Accessor& accessor, size_t i)
{
  const auto& view = model.bufferViews.at(accessor.bufferView);
  const auto stride = accessor.ByteStride(view);
  if (accessor.sparse.isSparse || stride <= 0 || i >= accessor.count)
    throw std::runtime_error("Unsupported rock accessor");

  const size_t offset = view.byteOffset + accessor.byteOffset + i * static_cast<size_t>(stride);
  return model.buffers.at(view.buffer).data.data() + offset;
}

float sampleHeight(std::span<const float> heights, glm::vec2 grid, glm::vec2& slope)
{
  const auto texelHeight = [&](glm::vec2 p) {
    p *= 1023.0f / 256.0f;
    const auto cell = glm::ivec2(glm::floor(p));
    const auto f = glm::fract(p);
    const int x = std::min(cell.x + 1, 1023);
    const int z = std::min(cell.y + 1, 1023);
    return glm::mix(
      glm::mix(heights[cell.y * 1024 + cell.x], heights[cell.y * 1024 + x], f.x),
      glm::mix(heights[z * 1024 + cell.x], heights[z * 1024 + x], f.x), f.y);
  };

  const auto cell = glm::min(glm::floor(grid), glm::vec2(255));
  const auto f = grid - cell;
  const float b = texelHeight(cell + glm::vec2(1, 0));
  const float c = texelHeight(cell + glm::vec2(0, 1));
  if (f.x + f.y <= 1)
  {
    const float a = texelHeight(cell);
    slope = {b - a, c - a};
    return a * (1 - f.x - f.y) + b * f.x + c * f.y;
  }

  const float d = texelHeight(cell + glm::vec2(1));
  slope = {d - c, d - b};
  return d * (f.x + f.y - 1) + b * (1 - f.y) + c * (1 - f.x);
}

void generateMipmaps(vk::CommandBuffer cmd, const etna::Image& image)
{
  auto extent = image.getExtent();
  for (uint32_t mip = 1; mip < image.getMipLevelCount(); ++mip)
  {
    etna::set_state(cmd, image.get(), vk::PipelineStageFlagBits2::eTransfer,
      vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite,
      vk::ImageLayout::eGeneral, vk::ImageAspectFlagBits::eColor, ForceSetState::eTrue);
    etna::flush_barriers(cmd);

    vk::ImageBlit blit{
      .srcSubresource = {vk::ImageAspectFlagBits::eColor, mip - 1, 0, 1},
      .dstSubresource = {vk::ImageAspectFlagBits::eColor, mip, 0, 1},
    };
    blit.srcOffsets[1] = vk::Offset3D{static_cast<int>(extent.width), static_cast<int>(extent.height), 1};
    extent.width = std::max(1u, extent.width / 2);
    extent.height = std::max(1u, extent.height / 2);
    blit.dstOffsets[1] = vk::Offset3D{static_cast<int>(extent.width), static_cast<int>(extent.height), 1};
    cmd.blitImage(image.get(), vk::ImageLayout::eGeneral, image.get(), vk::ImageLayout::eGeneral,
      {blit}, vk::Filter::eLinear);
  }
}
}

RockRenderer::RockRenderer()
{
  static_assert(sizeof(Instance) == 80);
  static_assert(sizeof(ComputeParams) == 192);
  static_assert(sizeof(vk::DrawIndexedIndirectCommand) == 20);

  loadModel(models[0], "boulder_01");
  loadModel(models[1], "rock_09");

  auto& ctx = etna::get_context();
  const auto properties = ctx.getPhysicalDevice().getProperties();
  const auto queue = ctx.getPhysicalDevice().getQueueFamilyProperties()[ctx.getQueueFamilyIdx()];
  timestampsSupported = queue.timestampValidBits != 0 && properties.limits.timestampComputeAndGraphics;
  timestampPeriod = properties.limits.timestampPeriod;
  timestampMask = queue.timestampValidBits == 64 ? ~uint64_t{0} : (uint64_t{1} << queue.timestampValidBits) - 1;

  frameResources.emplace(ctx.getMainWorkCount(), [&](size_t) {
    FrameResources frame;
    frame.instances = ctx.createBuffer(etna::Buffer::CreateInfo{
      .size = 2 * 65536 * sizeof(Instance),
      .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
      .name = "rock_instances",
    });

    frame.commands = ctx.createBuffer(etna::Buffer::CreateInfo{
      .size = 2 * sizeof(vk::DrawIndexedIndirectCommand),
      .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer |
        vk::BufferUsageFlagBits::eTransferDst,
      .name = "rock_commands",
    });

    frame.parameters = ctx.createBuffer(etna::Buffer::CreateInfo{
      .size = sizeof(ComputeParams),
      .bufferUsage = vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst,
      .name = "rock_compute_parameters",
    });

    if (timestampsSupported)
      frame.queries = etna::unwrap_vk_result(ctx.getDevice().createQueryPoolUnique(vk::QueryPoolCreateInfo{
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = 4,
      }));
    return frame;
  });

  if (const char* mode = std::getenv("ROCK_GPU"))
    gpuMode = std::string(mode) != "0";

  benchmark = std::getenv("ROCK_BENCHMARK") != nullptr;
  if (benchmark)
  {
    if (!timestampsSupported)
      throw std::runtime_error("GPU timestamps are required for the rock benchmark");

    benchmarkFile.open("rock_benchmark.csv");
    if (!benchmarkFile)
      throw std::runtime_error("Cannot write rock_benchmark.csv");

    benchmarkFile << "mode,density,cpu_visible,cpu_prepare_ms,gpu_prepare_ms,gpu_draw_ms,gpu_scene_ms\n";
    selectBenchmarkCase();
  }

  sampler = etna::Sampler(etna::Sampler::CreateInfo{
    .addressMode = vk::SamplerAddressMode::eRepeat,
    .name = "rock_sampler",
  });
}

void RockRenderer::loadModel(Model& target, const char* name)
{
  tinygltf::TinyGLTF loader;
  tinygltf::Model model;
  std::string error;
  std::string warning;
  const std::string path = std::string(GRAPHICS_COURSE_RESOURCES_ROOT) + "/scenes/" + name + "/" + name + ".gltf";
  if (!loader.LoadASCIIFromFile(&model, &error, &warning, path))
    throw std::runtime_error("Cannot load " + path + ": " + error);
  if (model.meshes.size() != 1 || model.meshes[0].primitives.size() != 1)
    throw std::runtime_error("Expected one rock mesh with one material: " + path);

  const auto& primitive = model.meshes[0].primitives[0];
  if (primitive.mode != TINYGLTF_MODE_TRIANGLES)
    throw std::runtime_error("Expected a triangle rock mesh: " + path);

  const auto& positions = model.accessors.at(primitive.attributes.at("POSITION"));
  const auto& normals = model.accessors.at(primitive.attributes.at("NORMAL"));
  const auto& uvs = model.accessors.at(primitive.attributes.at("TEXCOORD_0"));
  if (positions.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT ||
      normals.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT ||
      uvs.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT ||
      positions.type != TINYGLTF_TYPE_VEC3 || normals.type != TINYGLTF_TYPE_VEC3 ||
      uvs.type != TINYGLTF_TYPE_VEC2 || normals.count != positions.count || uvs.count != positions.count)
    throw std::runtime_error("Unsupported rock vertex format: " + path);

  std::vector<Vertex> vertices(positions.count);
  glm::vec3 lower(std::numeric_limits<float>::max());
  glm::vec3 upper(std::numeric_limits<float>::lowest());
  for (size_t i = 0; i < vertices.size(); ++i)
  {
    auto& vertex = vertices[i];
    std::memcpy(&vertex.position, element(model, positions, i), sizeof(glm::vec3));
    std::memcpy(&vertex.normal, element(model, normals, i), sizeof(glm::vec3));
    std::memcpy(&vertex.uv, element(model, uvs, i), sizeof(glm::vec2));
    lower = glm::min(lower, vertex.position);
    upper = glm::max(upper, vertex.position);
  }

  const glm::vec3 extent = upper - lower;
  const float scale = 2.0f / std::max({extent.x, extent.y, extent.z});
  const glm::vec3 origin{(lower.x + upper.x) * 0.5f, lower.y, (lower.z + upper.z) * 0.5f};
  target.center = ((lower + upper) * 0.5f - origin) * scale;

  for (auto& vertex : vertices)
  {
    vertex.position = (vertex.position - origin) * scale;
    target.radius = std::max(target.radius, glm::length(vertex.position - target.center));
  }

  const auto& accessor = model.accessors.at(primitive.indices);
  std::vector<uint32_t> indices(accessor.count);
  for (size_t i = 0; i < indices.size(); ++i)
  {
    const auto* data = element(model, accessor, i);
    if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
      std::memcpy(&indices[i], data, sizeof(uint32_t));
    else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
    {
      uint16_t index;
      std::memcpy(&index, data, sizeof(index));
      indices[i] = index;
    }
    else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
      indices[i] = *data;
    else
      throw std::runtime_error("Unsupported rock indices: " + path);

    if (indices[i] >= vertices.size())
      throw std::runtime_error("Invalid rock index: " + path);
  }

  target.indexCount = static_cast<uint32_t>(indices.size());

  auto& ctx = etna::get_context();
  auto commands = ctx.createOneShotCmdMgr();
  etna::BlockingTransferHelper transfer({.stagingSize = 4 * 1024 * 1024});

  target.vertices = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = vertices.size() * sizeof(Vertex),
    .bufferUsage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .name = name,
  });

  target.indices = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = indices.size() * sizeof(uint32_t),
    .bufferUsage = vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .name = name,
  });

  transfer.uploadBuffer<Vertex>(*commands, target.vertices, 0, vertices);
  transfer.uploadBuffer<uint32_t>(*commands, target.indices, 0, indices);

  const auto& material = model.materials.at(primitive.material);
  const std::array textureIndices{
    material.pbrMetallicRoughness.baseColorTexture.index,
    material.pbrMetallicRoughness.metallicRoughnessTexture.index,
    material.normalTexture.index,
  };
  const std::array images{&target.albedo, &target.roughness, &target.normal};
  for (size_t i = 0; i < images.size(); ++i)
  {
    const auto& source = model.images.at(model.textures.at(textureIndices[i]).source);
    if (source.bits != 8 || source.component != 4 || source.width <= 0 || source.height <= 0)
      throw std::runtime_error("Unsupported rock texture: " + path);

    *images[i] = ctx.createImage(etna::Image::CreateInfo{
      .extent = {static_cast<uint32_t>(source.width), static_cast<uint32_t>(source.height), 1},
      .name = name,
      .format = i == 0 ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm,
      .imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc |
        vk::ImageUsageFlagBits::eTransferDst,
      .mipLevels = static_cast<size_t>(std::bit_width(static_cast<uint32_t>(std::max(source.width, source.height)))),
    });

    transfer.uploadImage(*commands, *images[i], 0, 0,
      {reinterpret_cast<const std::byte*>(source.image.data()), source.image.size()});
  }

  auto cmd = commands->start();
  ETNA_CHECK_VK_RESULT(cmd.begin(vk::CommandBufferBeginInfo{}));
  etna::set_state(cmd, target.vertices.get(), vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
  etna::set_state(cmd, target.indices.get(), vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
  etna::flush_barriers(cmd);
  etna::set_state(cmd, target.vertices.get(), vk::PipelineStageFlagBits2::eVertexInput, vk::AccessFlagBits2::eVertexAttributeRead);
  etna::set_state(cmd, target.indices.get(), vk::PipelineStageFlagBits2::eVertexInput, vk::AccessFlagBits2::eIndexRead);
  etna::flush_barriers(cmd);

  for (auto* image : images)
    generateMipmaps(cmd, *image);
  ETNA_CHECK_VK_RESULT(cmd.end());

  commands->submitAndWait(std::move(cmd));
}

void RockRenderer::loadShaders()
{
  etna::create_program("generate_rocks", {TERRAIN_SHADERS_ROOT "rocks.comp.spv"});
  etna::create_program("rocks", {TERRAIN_SHADERS_ROOT "rocks.vert.spv", TERRAIN_SHADERS_ROOT "rocks.frag.spv"});
}

void RockRenderer::setupPipeline(vk::Format format)
{
  computePipeline = {};
  computePipeline = etna::get_context().getPipelineManager().createComputePipeline("generate_rocks", {});

  pipeline = {};
  pipeline = etna::get_context().getPipelineManager().createGraphicsPipeline("rocks",
    etna::GraphicsPipeline::CreateInfo{
      .vertexShaderInput = {.bindings = {etna::VertexShaderInputDescription::Binding{.byteStreamDescription = {
        .stride = sizeof(Vertex),
        .attributes = {
          {.format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, position)},
          {.format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, normal)},
          {.format = vk::Format::eR32G32Sfloat, .offset = offsetof(Vertex, uv)},
        },
      }}}},
      .fragmentShaderOutput = {
        .colorAttachmentFormats = {format},
        .depthAttachmentFormat = vk::Format::eD32Sfloat,
      },
    });
}

void RockRenderer::drawGui()
{
  if (ImGui::CollapsingHeader("Rocks", ImGuiTreeNodeFlags_DefaultOpen))
  {
    ImGui::BeginDisabled(benchmark);
    ImGui::Checkbox("GPU generation", &gpuMode);
    ImGui::SliderFloat("Density", &density, 0, 1, "%.3f");
    ImGui::SliderFloat("Draw distance", &drawDistance, 20, 400);

    ImGui::SliderFloat("Min size", &sizeRange.x, 0.1f, 4);
    sizeRange.y = std::max(sizeRange.y, sizeRange.x);
    ImGui::SliderFloat("Max size", &sizeRange.y, sizeRange.x, 4);
    ImGui::EndDisabled();

    if (gpuMode)
      ImGui::TextUnformatted("GPU indirect rendering");
    else
      ImGui::Text("Visible: %zu + %zu", instances[0].size(), instances[1].size());

    ImGui::Text("CPU preparation: %.3f ms", cpuPreparationMs);

    if (timestampsSupported)
      ImGui::Text("GPU preparation / draw: %.3f / %.3f ms", gpuPreparationMs, gpuDrawMs);
  }
}

void RockRenderer::update(
  std::span<const float> heights, glm::vec2 terrain_size,
  glm::vec3 camera, const glm::mat4& proj_view)
{
  ZoneScoped;
  const auto start = std::chrono::steady_clock::now();
  for (auto& group : instances)
    group.clear();

  const auto matrix = glm::transpose(proj_view);
  std::array planes{matrix[3] + matrix[0], matrix[3] - matrix[0],
    matrix[3] + matrix[1], matrix[3] - matrix[1], matrix[2], matrix[3] - matrix[2]};
  for (auto& plane : planes)
    plane /= glm::length(glm::vec3(plane));

  const float halfSize = terrain_size.x * 0.5f;
  const float maxReach = std::max(models[0].radius + glm::length(models[0].center),
    models[1].radius + glm::length(models[1].center)) * sizeRange.y;
  const float range = drawDistance + maxReach;
  const int cells = static_cast<int>(std::ceil(terrain_size.x / 2));
  const int minX = std::max(0, static_cast<int>(std::floor((camera.x - range + halfSize) / 2)));
  const int maxX = std::min(cells - 1, static_cast<int>(std::floor((camera.x + range + halfSize) / 2)));
  const int minZ = std::max(0, static_cast<int>(std::floor((camera.z - range + halfSize) / 2)));
  const int maxZ = std::min(cells - 1, static_cast<int>(std::floor((camera.z + range + halfSize) / 2)));

  computeParams.planes = planes;
  computeParams.cameraDistance = glm::vec4(camera, drawDistance);
  computeParams.terrainDensity = glm::vec4(terrain_size, density, range);
  computeParams.sizes = glm::vec4(sizeRange, 0, 0);
  for (size_t i = 0; i < models.size(); ++i)
    computeParams.bounds[i] = glm::vec4(models[i].center, models[i].radius);
  computeParams.cells = {minX, minZ, std::max(0, maxX - minX + 1), std::max(0, maxZ - minZ + 1)};

  if (!gpuMode && density > 0 && !heights.empty())
  {
    for (int z = minZ; z <= maxZ; ++z)
      for (int x = minX; x <= maxX; ++x)
        for (uint32_t candidate = 0; candidate < 4; ++candidate)
        {
          const uint32_t seed = hash(static_cast<uint32_t>(x) * 73856093u ^
            static_cast<uint32_t>(z) * 19349663u ^ candidate * 83492791u ^ 42u);
          if (random(seed, 0) >= density)
            continue;

          const glm::vec2 point = (glm::vec2(x, z) + glm::vec2(random(seed, 1), random(seed, 2))) * 2.0f - halfSize;
          if (glm::dot(point - glm::vec2(camera.x, camera.z), point - glm::vec2(camera.x, camera.z)) > range * range)
            continue;

          glm::vec2 slope;
          const float height = sampleHeight(heights, (point / terrain_size.x + 0.5f) * 256.0f, slope);
          const float probability = glm::mix(0.05f, 1.0f, glm::smoothstep(0.025f, 0.15f, height)) *
            glm::mix(1.0f, 0.05f, glm::smoothstep(0.4f, 0.58f, height));
          if (height <= 0.01f || random(seed, 3) >= probability)
            continue;

          const size_t modelIndex = hash(seed ^ 0x9e3779b9u) % models.size();
          const auto& model = models[modelIndex];
          const float scale = glm::mix(sizeRange.x, sizeRange.y, random(seed, 4));
          const float angle = random(seed, 5) * glm::two_pi<float>();

          slope *= terrain_size.y * 256.0f / terrain_size.x;
          const glm::vec3 normal = glm::normalize(glm::vec3(-slope.x, 1, -slope.y));
          const glm::quat rotation = glm::quat(glm::vec3(0, 1, 0), normal) *
            glm::angleAxis(angle, glm::vec3(0, 1, 0));

          const glm::vec3 position = glm::vec3(point.x, height * terrain_size.y, point.y) - normal * (scale * 0.12f);
          const glm::vec3 center = position + rotation * (model.center * scale);
          const float radius = model.radius * scale;
          if (glm::distance(center, camera) > drawDistance + radius)
            continue;

          if (std::any_of(planes.begin(), planes.end(), [&](const glm::vec4& plane) {
                return glm::dot(glm::vec3(plane), center) + plane.w < -radius;
              }))
            continue;

          Instance instance;
          instance.transform = glm::scale(glm::translate(glm::mat4(1), position) *
            glm::mat4_cast(rotation), glm::vec3(scale));
          instance.color = glm::vec4(glm::mix(0.85f, 1.15f, random(seed, 6)) *
            glm::vec3(1.0f, glm::mix(0.95f, 1.05f, random(seed, 7)), glm::mix(0.93f, 1.05f, random(seed, 8))), 1);
          instances[modelIndex].push_back(instance);
        }
  }

  cpuPreparationMs = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - start).count();
}

void RockRenderer::upload(vk::CommandBuffer cmd, const etna::Image& heightmap, vk::Sampler height_sampler)
{
  const auto start = std::chrono::steady_clock::now();
  auto& frame = frameResources->get();

  if (timestampsSupported)
  {
    cmd.resetQueryPool(frame.queries.get(), 0, 4);
    cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, frame.queries.get(), 0);
  }

  if (gpuMode)
  {
    std::array<vk::DrawIndexedIndirectCommand, 2> commands{};
    for (size_t i = 0; i < models.size(); ++i)
      commands[i].indexCount = models[i].indexCount;
    etna::set_state(cmd, frame.commands.get(), vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
    etna::set_state(cmd, frame.parameters.get(), vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
    etna::flush_barriers(cmd);

    cmd.updateBuffer(frame.commands.get(), 0, sizeof(commands), commands.data());
    cmd.updateBuffer(frame.parameters.get(), 0, sizeof(computeParams), &computeParams);

    etna::set_state(cmd, frame.instances.get(), vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderStorageWrite);
    etna::set_state(cmd, frame.commands.get(), vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);
    etna::set_state(cmd, frame.parameters.get(), vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eUniformRead);

    const auto compute = etna::get_shader_program("generate_rocks");
    auto set = etna::create_descriptor_set(compute.getDescriptorLayoutId(0), cmd, {
      etna::Binding{0, frame.instances.genBinding()},
      etna::Binding{1, frame.commands.genBinding()},
      etna::Binding{2, heightmap.genBinding(height_sampler, vk::ImageLayout::eShaderReadOnlyOptimal)},
      etna::Binding{3, frame.parameters.genBinding()},
    });
    etna::flush_barriers(cmd);

    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline.getVkPipeline());
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, computePipeline.getVkPipelineLayout(),
      0, {set.getVkSet()}, {});

    const uint32_t candidates = static_cast<uint32_t>(computeParams.cells.z * computeParams.cells.w * 4);
    cmd.dispatch(density > 0 ? std::max(1u, (candidates + 63) / 64) : 1u, 1, 1);

    etna::set_state(cmd, frame.commands.get(), vk::PipelineStageFlagBits2::eDrawIndirect,
      vk::AccessFlagBits2::eIndirectCommandRead);
  }
  else
  {
    etna::set_state(cmd, frame.instances.get(), vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
    etna::flush_barriers(cmd);
    for (size_t i = 0; i < models.size(); ++i)
    {
      const size_t bytes = instances[i].size() * sizeof(Instance);
      const auto* data = reinterpret_cast<const std::byte*>(instances[i].data());
      for (size_t offset = 0; offset < bytes; offset += 65536)
        cmd.updateBuffer(frame.instances.get(), i * 65536 * sizeof(Instance) + offset,
          std::min(size_t{65536}, bytes - offset), data + offset);
    }
  }

  etna::set_state(cmd, frame.instances.get(), vk::PipelineStageFlagBits2::eVertexShader,
    vk::AccessFlagBits2::eShaderStorageRead);

  const auto shader = etna::get_shader_program("rocks");
  for (size_t i = 0; i < models.size(); ++i)
  {
    descriptorSets[i].reset();
    auto& model = models[i];
    descriptorSets[i].emplace(etna::create_descriptor_set(shader.getDescriptorLayoutId(0), cmd, {
      etna::Binding{0, frame.instances.genBinding(i * 65536 * sizeof(Instance), 65536 * sizeof(Instance))},
      etna::Binding{1, model.albedo.genBinding(sampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
      etna::Binding{2, model.roughness.genBinding(sampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
      etna::Binding{3, model.normal.genBinding(sampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
    }));
    etna::set_state(cmd, model.vertices.get(), vk::PipelineStageFlagBits2::eVertexInput, vk::AccessFlagBits2::eVertexAttributeRead);
    etna::set_state(cmd, model.indices.get(), vk::PipelineStageFlagBits2::eVertexInput, vk::AccessFlagBits2::eIndexRead);
  }
  etna::flush_barriers(cmd);

  if (timestampsSupported)
    cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, frame.queries.get(), 1);

  cpuPreparationMs += std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - start).count();

  frame.cpuMs = cpuPreparationMs;
  frame.gpuMode = gpuMode;
  frame.benchmarkPhase = benchmarkPhase;
  frame.visible = instances[0].size() + instances[1].size();
}

void RockRenderer::render(vk::CommandBuffer cmd, std::span<const std::byte> params)
{
  ETNA_PROFILE_GPU(cmd, renderRocks);
  auto& frame = frameResources->get();

  if (timestampsSupported)
    cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, frame.queries.get(), 2);

  cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.getVkPipeline());
  cmd.pushConstants(pipeline.getVkPipelineLayout(),
    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
    0, static_cast<uint32_t>(params.size()), params.data());
  for (size_t i = 0; i < models.size(); ++i)
  {
    if (!gpuMode && instances[i].empty())
      continue;

    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline.getVkPipelineLayout(),
      0, {descriptorSets[i]->getVkSet()}, {});
    cmd.bindVertexBuffers(0, {models[i].vertices.get()}, {0});
    cmd.bindIndexBuffer(models[i].indices.get(), 0, vk::IndexType::eUint32);

    if (gpuMode)
      cmd.drawIndexedIndirect(frame.commands.get(), i * sizeof(vk::DrawIndexedIndirectCommand),
        1, sizeof(vk::DrawIndexedIndirectCommand));
    else
      cmd.drawIndexed(models[i].indexCount, static_cast<uint32_t>(instances[i].size()), 0, 0, 0);
  }

  if (timestampsSupported)
  {
    cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, frame.queries.get(), 3);
    frame.hasQueries = true;
  }
}

void RockRenderer::beginFrame()
{
  auto& frame = frameResources->get();
  if (frame.hasQueries)
    readTimings(frame);
}

void RockRenderer::readTimings(FrameResources& frame)
{
  std::array<uint64_t, 4> values{};
  const auto result = etna::get_context().getDevice().getQueryPoolResults(
    frame.queries.get(), 0, 4, sizeof(values), values.data(), sizeof(uint64_t), vk::QueryResultFlagBits::e64);
  if (result != vk::Result::eSuccess)
    return;

  frame.hasQueries = false;
  const auto elapsed = [&](size_t start, size_t end) {
    return static_cast<double>((values[end] - values[start]) & timestampMask) * timestampPeriod * 1e-6;
  };

  gpuPreparationMs = elapsed(0, 1);
  gpuDrawMs = elapsed(2, 3);

  if (!benchmark || benchmarkDone || frame.benchmarkPhase != benchmarkPhase)
    return;

  ++benchmarkSamples;
  if (benchmarkSamples <= 30)
    return;

  benchmarkTotals[0] += frame.cpuMs;
  benchmarkTotals[1] += gpuPreparationMs;
  benchmarkTotals[2] += gpuDrawMs;
  benchmarkTotals[3] += elapsed(0, 3);
  if (benchmarkSamples < 150)
    return;

  benchmarkFile << (frame.gpuMode ? "GPU" : "CPU") << ',' << density << ',';
  if (!frame.gpuMode)
    benchmarkFile << frame.visible;
  for (double sum : benchmarkTotals)
    benchmarkFile << ',' << std::fixed << std::setprecision(6) << sum / 120;
  benchmarkFile << '\n';
  benchmarkFile.flush();

  ++benchmarkPhase;
  benchmarkSamples = 0;
  benchmarkTotals = {};
  if (benchmarkPhase == 6)
    benchmarkDone = true;
  else
    selectBenchmarkCase();
}

void RockRenderer::selectBenchmarkCase()
{
  const std::array densities{0.02f, 0.08f, 0.25f};
  density = densities[benchmarkPhase / 2];
  gpuMode = benchmarkPhase % 2 != 0;
  drawDistance = 120;
  sizeRange = {0.4f, 1.6f};
}
