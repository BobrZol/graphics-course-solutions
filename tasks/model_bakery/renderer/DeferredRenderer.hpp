#pragma once

#include <etna/Buffer.hpp>
#include <etna/ComputePipeline.hpp>
#include <etna/GpuSharedResource.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <etna/Image.hpp>
#include <etna/RenderTargetStates.hpp>
#include <etna/Sampler.hpp>
#include <glm/glm.hpp>
#include <array>
#include <fstream>
#include <optional>

#include "FramePacket.hpp"


class DeferredRenderer
{
public:
  DeferredRenderer();

  void allocateResources(glm::uvec2 size);
  void loadShaders();
  void setupPipelines(vk::Format output_format);
  void update(const FramePacket& packet);
  void drawGui();
  void beginFrame();
  void begin(vk::CommandBuffer cmd);
  void timestamp(vk::CommandBuffer cmd, uint32_t index);
  void finish(vk::CommandBuffer cmd, vk::Image output, vk::ImageView output_view,
    glm::vec4 camera, glm::vec4 sun, glm::vec4 light,
    const etna::Image& albedo, const etna::Image& roughness, const etna::Image& height,
    vk::Sampler material_sampler);

  bool isCompact() const { return compact; }
  bool isBenchmarkDone() const { return benchmarkDone; }
  static std::vector<vk::Format> formats(bool compact);
  std::vector<etna::RenderTargetState::AttachmentParams> attachments(bool decals = false);
  etna::RenderTargetState::AttachmentParams depthAttachment(bool load = false);

private:
  void allocateGbuffer();

  struct Parameters
  {
    glm::mat4 inverseViewProjection{1};
    glm::mat4 view{1};
    glm::mat4 inverseProjection{1};
    glm::vec4 cameraExposure{};
    glm::vec4 sun{};
    glm::vec4 light{};
    glm::vec4 screen{};
    glm::ivec4 options{};
    glm::vec4 decal{0.8f, 1, 1, 1};
    std::array<glm::mat4, 4> decalInverse;
  } parameters;

  struct LocalLight
  {
    glm::vec4 positionRadius;
    glm::vec4 colorIntensity;
  };

  struct FrameResources
  {
    etna::Buffer parameters;
    etna::Buffer lights;
    etna::Buffer clusters;
    vk::UniqueQueryPool queries;
    bool hasQueries = false;
    int benchmarkPhase = -1;
  };

  std::optional<etna::GpuSharedResource<FrameResources>> frames;
  std::array<LocalLight, 1024> lights;
  std::array<std::array<etna::Image, 3>, 2> buffers;
  etna::Image depth;
  etna::Sampler sampler;
  std::array<etna::GraphicsPipeline, 2> decalPipelines;
  etna::GraphicsPipeline lightingPipeline;
  etna::ComputePipeline clusterPipeline;
  glm::uvec2 resolution{};

  bool compact = true;
  bool allocatedCompact = true;
  bool clustered = true;
  bool decalsEnabled = true;
  bool outputSrgb = true;
  int lightCount = 256;
  float lightIntensity = 150;
  float lightRadius = 25;

  bool timestampsSupported = false;
  double timestampPeriod = 0;
  uint64_t timestampMask = 0;
  std::array<double, 6> timings{};
  bool benchmark = false;
  bool benchmarkDone = false;
  int benchmarkPhase = 0;
  int benchmarkSamples = 0;
  std::array<double, 6> totals{};
  std::ofstream benchmarkFile;
};
