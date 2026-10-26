#include "App.hpp"

#include <tracy/Tracy.hpp>
#include <imgui.h>
#include <algorithm>


App::App()
{
  glm::uvec2 initialRes = {1280, 720};
  mainWindow = windowing.createWindow(OsWindow::CreateInfo{
    .resolution = initialRes,
  });

  renderer.reset(new Renderer(initialRes));

  auto instExts = windowing.getRequiredVulkanInstanceExtensions();
  renderer->initVulkan(instExts);

  auto surface = mainWindow->createVkSurface(etna::get_context().getInstance());

  renderer->initFrameDelivery(std::move(surface), [this]() { return mainWindow->getResolution(); });

  ImGuiRenderer::enableImGuiForWindow(mainWindow->native());
  mainCam.lookAt({180, 145, 180}, {0, 15, 0}, {0, 1, 0});
  mainCam.zNear = 0.1f;
  mainCam.zFar = 2000;
}

void App::run()
{
  double lastTime = windowing.getTime();
  while (!mainWindow->isBeingClosed())
  {
    const double currTime = windowing.getTime();
    const float diffTime = static_cast<float>(currTime - lastTime);
    lastTime = currTime;

    windowing.poll();

    processInput(std::min(diffTime, 0.1f));

    drawFrame();

    FrameMark;
  }
}

void App::processInput(float dt)
{
  ZoneScoped;

  if (mainWindow->keyboard[KeyboardKey::kEscape] == ButtonState::Falling)
    mainWindow->askToClose();

  if (is_held_down(mainWindow->keyboard[KeyboardKey::kLeftShift]))
    camMoveSpeed = 100;
  else
    camMoveSpeed = 25;

  if (!ImGui::GetIO().WantCaptureMouse &&
      mainWindow->mouse[MouseButton::mbRight] == ButtonState::Rising)
    mainWindow->captureMouse = !mainWindow->captureMouse;

  if (!ImGui::GetIO().WantCaptureKeyboard)
    moveCam(mainCam, mainWindow->keyboard, dt);
  if (mainWindow->captureMouse)
    rotateCam(mainCam, mainWindow->mouse, dt);

  renderer->debugInput(mainWindow->keyboard);
}

void App::drawFrame()
{
  ZoneScoped;

  renderer->update(FramePacket{
    .mainCam = mainCam,
    .currentTime = static_cast<float>(windowing.getTime()),
  });
  renderer->drawFrame();
}

void App::moveCam(Camera& cam, const Keyboard& kb, float dt)
{
  // Move position of camera based on WASD keys, and FR keys for up and down

  glm::vec3 dir = {0, 0, 0};

  if (is_held_down(kb[KeyboardKey::kS]))
    dir -= cam.forward();

  if (is_held_down(kb[KeyboardKey::kW]))
    dir += cam.forward();

  if (is_held_down(kb[KeyboardKey::kA]))
    dir -= cam.right();

  if (is_held_down(kb[KeyboardKey::kD]))
    dir += cam.right();

  if (is_held_down(kb[KeyboardKey::kF]))
    dir -= cam.up();

  if (is_held_down(kb[KeyboardKey::kR]))
    dir += cam.up();

  // NOTE: This is how you make moving diagonally not be faster than
  // in a straight line.
  cam.move(dt * camMoveSpeed * (length(dir) > 1e-9 ? normalize(dir) : dir));
}

void App::rotateCam(Camera& cam, const Mouse& ms, float /*dt*/)
{
  // Rotate camera based on mouse movement
  cam.rotate(-camRotateSpeed * ms.capturedPosDelta.y, -camRotateSpeed * ms.capturedPosDelta.x);

  // Increase or decrease field of view based on mouse wheel
  cam.fov -= zoomSensitivity * ms.scrollDelta.y;
  if (cam.fov < 1.0f)
    cam.fov = 1.0f;
  if (cam.fov > 120.0f)
    cam.fov = 120.0f;
}
