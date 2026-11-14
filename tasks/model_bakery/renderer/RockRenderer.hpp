#pragma once

#include <etna/Buffer.hpp>
#include <etna/ComputePipeline.hpp>
#include <etna/DescriptorSet.hpp>
#include <etna/GpuSharedResource.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <etna/Image.hpp>
#include <etna/Sampler.hpp>
#include <glm/glm.hpp>
#include <array>
#include <chrono>
#include <fstream>
#include <optional>
#include <span>
#include <vector>


class RockRenderer
{
public:
  RockRenderer();

  void loadShaders();
  void setupPipeline();
  void drawGui();
  void beginFrame();

  bool isGpuMode() const { return gpuMode; }
  bool isBenchmarkDone() const { return benchmarkDone; }

  void update(
    std::span<const float> heights, glm::vec2 terrain_size,
    glm::vec3 camera, const glm::mat4& proj_view);
  void upload(vk::CommandBuffer cmd, const etna::Image& heightmap, vk::Sampler height_sampler);
  void render(vk::CommandBuffer cmd, std::span<const std::byte> params, bool depth_only, bool compact);

private:
  struct Vertex
  {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
  };

  struct Instance
  {
    glm::mat4 transform;
    glm::vec4 color;
  };

  struct Model
  {
    etna::Buffer vertices;
    etna::Buffer indices;
    etna::Image albedo;
    etna::Image roughness;
    etna::Image normal;

    uint32_t indexCount = 0;
    glm::vec3 center{};
    float radius = 0;
  };

  struct ComputeParams
  {
    std::array<glm::vec4, 6> planes;
    glm::vec4 cameraDistance;
    glm::vec4 terrainDensity;
    glm::vec4 sizes;
    std::array<glm::vec4, 2> bounds;
    glm::ivec4 cells;
  } computeParams{};

  struct FrameResources
  {
    etna::Buffer instances;
    etna::Buffer commands;
    etna::Buffer parameters;
    vk::UniqueQueryPool queries;

    bool hasQueries = false;
    bool gpuMode = false;
    int benchmarkPhase = -1;
    size_t visible = 0;
    double cpuMs = 0;
  };

  void loadModel(Model& model, const char* name);
  void readTimings(FrameResources& frame);
  void selectBenchmarkCase();

  std::array<Model, 2> models;
  std::array<std::vector<Instance>, 2> instances;
  std::optional<etna::GpuSharedResource<FrameResources>> frameResources;
  std::array<std::optional<etna::DescriptorSet>, 2> descriptorSets;

  etna::Sampler sampler;
  std::array<etna::GraphicsPipeline, 2> pipelines;
  etna::GraphicsPipeline depthPipeline;
  std::array<std::optional<etna::DescriptorSet>, 2> depthDescriptorSets;
  etna::ComputePipeline computePipeline;

  bool gpuMode = true;

  bool timestampsSupported = false;
  double timestampPeriod = 0;
  uint64_t timestampMask = 0;

  double cpuPreparationMs = 0;
  double gpuPreparationMs = 0;
  double gpuDrawMs = 0;

  bool benchmark = false;
  bool benchmarkDone = false;
  int benchmarkPhase = 0;
  int benchmarkSamples = 0;
  std::array<double, 4> benchmarkTotals{};
  std::ofstream benchmarkFile;

  float density = 0.08f;
  float drawDistance = 120;
  glm::vec2 sizeRange{0.4f, 1.6f};
};
