#pragma once

#include <etna/Image.hpp>
#include <etna/Sampler.hpp>
#include <etna/Buffer.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <glm/glm.hpp>

#include "wsi/Keyboard.hpp"
#include "FramePacket.hpp"


class WorldRenderer
{
public:
  WorldRenderer();

  void loadShaders();
  void allocateResources(glm::uvec2 swapchain_resolution);
  void setupPipelines(vk::Format swapchain_format);

  void debugInput(const Keyboard& kb);
  void update(const FramePacket& packet);
  void drawGui();
  void renderWorld(
    vk::CommandBuffer cmd_buf, vk::Image target_image, vk::ImageView target_image_view);

private:
  void generateHeightmap(vk::CommandBuffer cmd_buf);

  etna::Image mainViewDepth;
  etna::Image heightmap;
  etna::Image albedoMaps;
  etna::Image roughnessMaps;
  etna::Image detailMaps;
  etna::Sampler heightSampler;
  etna::Sampler materialSampler;
  etna::Buffer indices;

  struct PushConstants
  {
    glm::mat4 projView{1};
    glm::vec4 cameraExposure{0, 0, 0, 1.2f};
    glm::vec4 terrainSettings{256, 75, 32, 0.25f};
    glm::vec4 sun{0, 1, 0, 3.5f};
    glm::vec4 light{1, 0.95f, 0.85f, 0.25f};
  } params;

  struct NoiseParams
  {
    glm::vec4 noise{4, 0.5f, 2, 6};
    glm::vec4 island{0.95f, 0.55f, 1.4f, 7};
  } noiseParams;

  float sunAzimuth = 135;
  float sunElevation = 45;
  bool heightmapDirty = true;
  bool outputSrgb = true;
  etna::GraphicsPipeline terrainPipeline{};
  etna::GraphicsPipeline heightmapPipeline{};
  glm::uvec2 resolution{};
};
