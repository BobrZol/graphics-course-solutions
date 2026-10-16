#pragma once

#include <etna/Window.hpp>
#include <etna/PerFrameCmdMgr.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <etna/Image.hpp>
#include <etna/GlobalContext.hpp>
#include <etna/Sampler.hpp>

#include "wsi/OsWindowingManager.hpp"


class App
{
public:
  App();
  ~App();

  void run();

private:
  void drawFrame();
  void generateMipmaps(vk::CommandBuffer cmd, const etna::Image& image);

private:
  OsWindowingManager windowing;
  std::unique_ptr<OsWindow> osWindow;

  etna::GlobalContext* context;
  etna::GraphicsPipeline pipeline;
  etna::GraphicsPipeline materialPipeline;
  etna::GraphicsPipeline shieldPipeline;

  glm::uvec2 resolution;
  bool useVsync;
  etna::Image materialTarget;
  etna::Image shieldTarget;
  etna::Sampler sampler;

  std::unique_ptr<etna::Window> vkWindow;
  std::unique_ptr<etna::PerFrameCmdMgr> commandManager;
};
