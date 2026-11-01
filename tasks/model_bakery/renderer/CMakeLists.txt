
add_executable(terrain
  main.cpp
  App.cpp
  Renderer.cpp
  WorldRenderer.cpp
  RockRenderer.cpp
)

target_link_libraries(terrain
  PRIVATE glfw etna glm::glm wsi gui scene render_utils)

target_add_shaders(terrain
  shaders/static_mesh.frag
  shaders/static_mesh.vert
  shaders/heightmap.vert
  shaders/heightmap.frag
  shaders/rocks.vert
  shaders/rocks.frag
  shaders/rocks.comp
)
