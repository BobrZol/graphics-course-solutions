#pragma once

#include <etna/Window.hpp>
#include <etna/PerFrameCmdMgr.hpp>
#include <etna/ComputePipeline.hpp>
#include <etna/Image.hpp>
#include <etna/GlobalContext.hpp>
#include <etna/ComputePipeline.hpp>

#include "wsi/OsWindowingManager.hpp"
#include <etna/Image.hpp>


class App
{
public:
  App();
  ~App();

  void run();

private:
  void drawFrame();

private:
  OsWindowingManager windowing;
  std::unique_ptr<OsWindow> osWindow;

  etna::GlobalContext* context;
  etna::ComputePipeline pipeline;

  glm::uvec2 resolution;
  bool useVsync;
  etna::Image computeTarget;

  std::unique_ptr<etna::Window> vkWindow;
  std::unique_ptr<etna::PerFrameCmdMgr> commandManager;
};
