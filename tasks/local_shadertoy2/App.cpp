#include "App.hpp"

#include <etna/Etna.hpp>
#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/RenderTargetStates.hpp>
#include <algorithm>
#include <bit>


App::App()
  : resolution{1280, 720}
  , useVsync{true}
{
  // First, we need to initialize Vulkan, which is not trivial because
  // extensions are required for just about anything.
  {
    // GLFW tells us which extensions it needs to present frames to the OS window.
    // Actually rendering anything to a screen is optional in Vulkan, you can
    // alternatively save rendered frames into files, send them over network, etc.
    // Instance extensions do not depend on the actual GPU, only on the OS.
    auto glfwInstExts = windowing.getRequiredVulkanInstanceExtensions();

    std::vector<const char*> instanceExtensions{glfwInstExts.begin(), glfwInstExts.end()};

    // We also need the swapchain device extension to get access to the OS
    // window from inside of Vulkan on the GPU.
    // Device extensions require HW support from the GPU.
    // Generally, in Vulkan, we call the GPU a "device" and the CPU/OS combination a "host."
    std::vector<const char*> deviceExtensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    // Etna does all of the Vulkan initialization heavy lifting.
    // You can skip figuring out how it works for now.
    etna::initialize(etna::InitParams{
      .applicationName = "Local Shadertoy",
      .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
      .instanceExtensions = instanceExtensions,
      .deviceExtensions = deviceExtensions,
      // Replace with an index if etna detects your preferred GPU incorrectly
      .physicalDeviceIndexOverride = {},
      .numFramesInFlight = 1,
    });


    context = &etna::get_context();
  }

  // Next, we need a magical Etna helper to send commands to the GPU.
  // How it is actually performed is not trivial, but we can skip this for now.
  commandManager = etna::get_context().createPerFrameCmdMgr();

  // Now we can create an OS window
  osWindow = windowing.createWindow(OsWindow::CreateInfo{
    .resolution = resolution,
  });

  // But we also need to hook the OS window up to Vulkan manually!
  {
    // First, we ask GLFW to provide a "surface" for the window,
    // which is an opaque description of the area where we can actually render.
    auto surface = osWindow->createVkSurface(etna::get_context().getInstance());

    // Then we pass it to Etna to do the complicated work for us
    vkWindow = etna::get_context().createWindow(etna::Window::CreateInfo{
      .surface = std::move(surface),
    });

    // And finally ask Etna to create the actual swapchain so that we can
    // get (different) images each frame to render stuff into.
    // Here, we do not support window resizing, so we only need to call this once.
    auto [w, h] = vkWindow->recreateSwapchain(etna::Window::DesiredProperties{
      .resolution = {resolution.x, resolution.y},
      .vsync = useVsync,
      .numFramesInFlight = static_cast<uint32_t>(commandManager->getCmdBufferCount()),
    });

    // Technically, Vulkan might fail to initialize a swapchain with the requested
    // resolution and pick a different one. This, however, does not occur on platforms
    // we support. Still, it's better to follow the "intended" path.
    resolution = {w, h};
  }

  // TODO: Initialize any additional resources you require here!
  etna::create_program("toy",
    {LOCAL_SHADERTOY2_SHADERS_ROOT "toy.vert.spv", LOCAL_SHADERTOY2_SHADERS_ROOT "toy.frag.spv"});
  etna::create_program("toy_material",
    {LOCAL_SHADERTOY2_SHADERS_ROOT "toy.vert.spv", LOCAL_SHADERTOY2_SHADERS_ROOT "material.frag.spv"});
  etna::create_program("toy_shield",
    {LOCAL_SHADERTOY2_SHADERS_ROOT "toy.vert.spv", LOCAL_SHADERTOY2_SHADERS_ROOT "shield.frag.spv"});

  auto pipelineInfo = etna::GraphicsPipeline::CreateInfo{
    .depthConfig = {
      .depthTestEnable = vk::False,
      .depthWriteEnable = vk::False,
    },
    .fragmentShaderOutput = {
      .colorAttachmentFormats = {vkWindow->getCurrentFormat()},
    },
  };
  auto& pipelineManager = context->getPipelineManager();
  pipeline = pipelineManager.createGraphicsPipeline("toy", pipelineInfo);
  pipelineInfo.fragmentShaderOutput.colorAttachmentFormats = {vk::Format::eR8G8B8A8Unorm};
  materialPipeline = pipelineManager.createGraphicsPipeline("toy_material", pipelineInfo);
  shieldPipeline = pipelineManager.createGraphicsPipeline("toy_shield", pipelineInfo);

  const auto mipLevels = static_cast<std::size_t>(std::bit_width(std::max(resolution.x, resolution.y)));
  const auto imageUsage = vk::ImageUsageFlagBits::eColorAttachment |
    vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc |
    vk::ImageUsageFlagBits::eTransferDst;
  materialTarget = context->createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "material_target",
    .format = vk::Format::eR8G8B8A8Unorm,
    .imageUsage = imageUsage,
    .mipLevels = mipLevels,
  });
  shieldTarget = context->createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "shield_target",
    .format = vk::Format::eR8G8B8A8Unorm,
    .imageUsage = imageUsage,
    .mipLevels = mipLevels,
  });
  sampler = etna::Sampler(etna::Sampler::CreateInfo{
    .addressMode = vk::SamplerAddressMode::eRepeat,
    .mipmapMode = vk::SamplerMipmapMode::eLinear,
    .name = "procedural_sampler",
    .maxLod = static_cast<float>(mipLevels - 1),
  });
}

App::~App()
{
  ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());
}

void App::run()
{
  while (!osWindow->isBeingClosed())
  {
    windowing.poll();

    drawFrame();
  }

  // We need to wait for the GPU to execute the last frame before destroying
  // all resources and closing the application.
  ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());
}

void App::drawFrame()
{
  // First, get a command buffer to write GPU commands into.
  auto currentCmdBuf = commandManager->acquireNext();

  // Next, tell Etna that we are going to start processing the next frame.
  etna::begin_frame();

  // And now get the image we should be rendering the picture into.
  auto nextSwapchainImage = vkWindow->acquireNext();

  // When window is minimized, we can't render anything in Windows
  // because it kills the swapchain, so we skip frames in this case.
  if (nextSwapchainImage)
  {
    auto [backbuffer, backbufferView, backbufferAvailableSem, backbufferReadyForPresentSem] =
      *nextSwapchainImage;

    ETNA_CHECK_VK_RESULT(currentCmdBuf.begin(vk::CommandBufferBeginInfo{}));
    {
      struct PushConstants
      {
        glm::vec4 time_res;
      };
      const PushConstants pc{
        glm::vec4{
          static_cast<float>(windowing.getTime()),
          static_cast<float>(resolution.x),
          static_cast<float>(resolution.y),
          0.0f,
        },
      };

      auto drawProcedural = [&](etna::Image& target, const etna::GraphicsPipeline& passPipeline) {
        etna::RenderTargetState renderTargets(
          currentCmdBuf,
          {{0, 0}, {resolution.x, resolution.y}},
          {{.image = target.get(), .view = target.getView({.levelCount = 1})}},
          {});
        currentCmdBuf.bindPipeline(
          vk::PipelineBindPoint::eGraphics, passPipeline.getVkPipeline());
        currentCmdBuf.pushConstants(
          passPipeline.getVkPipelineLayout(),
          vk::ShaderStageFlagBits::eFragment, 0, sizeof(pc), &pc);
        currentCmdBuf.draw(3, 1, 0, 0);
      };

      drawProcedural(materialTarget, materialPipeline);
      drawProcedural(shieldTarget, shieldPipeline);
      generateMipmaps(currentCmdBuf, materialTarget);
      generateMipmaps(currentCmdBuf, shieldTarget);

      auto shaderInfo = etna::get_shader_program("toy");
      auto set = etna::create_descriptor_set(
        shaderInfo.getDescriptorLayoutId(0),
        currentCmdBuf,
        {
          etna::Binding{
            0, shieldTarget.genBinding(sampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
          etna::Binding{
            1, materialTarget.genBinding(sampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
        });

      {
        etna::RenderTargetState renderTargets(
          currentCmdBuf,
          {{0, 0}, {resolution.x, resolution.y}},
          {{.image = backbuffer, .view = backbufferView}},
          {});
        currentCmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.getVkPipeline());
        currentCmdBuf.bindDescriptorSets(
          vk::PipelineBindPoint::eGraphics,
          pipeline.getVkPipelineLayout(), 0, {set.getVkSet()}, {});
        currentCmdBuf.pushConstants(
          pipeline.getVkPipelineLayout(),
          vk::ShaderStageFlagBits::eFragment, 0, sizeof(pc), &pc);
        currentCmdBuf.draw(3, 1, 0, 0);
      }

      // At the end of "rendering", we are required to change how the pixels of the
      // swpchain image are laid out in memory to something that is appropriate
      // for presenting to the window (while preserving the content of the pixels!).
      etna::set_state(
        currentCmdBuf,
        backbuffer,
        // This looks weird, but is correct. Ask about it later.
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        {},
        vk::ImageLayout::ePresentSrcKHR,
        vk::ImageAspectFlagBits::eColor);
      // And of course flush the layout transition.
      etna::flush_barriers(currentCmdBuf);
    }
    ETNA_CHECK_VK_RESULT(currentCmdBuf.end());

    // We are done recording GPU commands now and we can send them to be executed by the GPU.
    // Note that the GPU won't start executing our commands before the backbufferAvailableSem
    // semaphore is signalled, which will happen when the OS says that the next swapchain image
    // is ready, and the result image will be ready for present after backbufferReadyForPresent
    // is signalled by GPU
    auto renderingDone = commandManager->submit(
      std::move(currentCmdBuf),
      std::move(backbufferAvailableSem),
      std::move(backbufferReadyForPresentSem));

    // Finally, present the backbuffer the screen, but only after the GPU tells the OS
    // that it is done executing the command buffer via the renderingDone semaphore.
    const bool presented = vkWindow->present(std::move(renderingDone), backbufferView);

    if (!presented)
      nextSwapchainImage = std::nullopt;
  }

  etna::end_frame();

  // After a window us un-minimized, we need to restore the swapchain to continue rendering.
  if (!nextSwapchainImage && osWindow->getResolution() != glm::uvec2{0, 0})
  {
    auto [w, h] = vkWindow->recreateSwapchain(etna::Window::DesiredProperties{
      .resolution = {resolution.x, resolution.y},
      .vsync = useVsync,
      .numFramesInFlight = static_cast<uint32_t>(commandManager->getCmdBufferCount()),
    });
    ETNA_VERIFY((resolution == glm::uvec2{w, h}));
  }
}

void App::generateMipmaps(vk::CommandBuffer cmd, const etna::Image& image)
{
  auto extent = image.getExtent();
  for (uint32_t mip = 1; mip < image.getMipLevelCount(); ++mip)
  {
    etna::set_state(
      cmd, image.get(),
      vk::PipelineStageFlagBits2::eTransfer,
      vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite,
      vk::ImageLayout::eGeneral, vk::ImageAspectFlagBits::eColor,
      ForceSetState::eTrue);
    etna::flush_barriers(cmd);

    const auto width = std::max(1u, extent.width / 2);
    const auto height = std::max(1u, extent.height / 2);
    vk::ImageBlit blit{
      .srcSubresource = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .mipLevel = mip - 1,
        .layerCount = 1,
      },
      .dstSubresource = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .mipLevel = mip,
        .layerCount = 1,
      },
    };
    blit.srcOffsets[1] = vk::Offset3D{
      static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height), 1};
    blit.dstOffsets[1] = vk::Offset3D{
      static_cast<int32_t>(width), static_cast<int32_t>(height), 1};
    cmd.blitImage(
      image.get(), vk::ImageLayout::eGeneral,
      image.get(), vk::ImageLayout::eGeneral, {blit}, vk::Filter::eLinear);
    extent = vk::Extent3D{width, height, 1};
  }
}
