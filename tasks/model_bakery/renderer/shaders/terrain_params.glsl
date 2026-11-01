#include "scene_params.glsl"

layout(set = 0, binding = 0) uniform sampler2D heightmap;

vec2 heightUV(vec2 uv)
{
  return (uv * 1023.0 + 0.5) / 1024.0;
}
