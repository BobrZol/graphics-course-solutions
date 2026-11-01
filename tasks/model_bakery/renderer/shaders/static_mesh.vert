#version 450
#extension GL_GOOGLE_include_directive : require

#include "terrain_params.glsl"

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec2 terrainUV;

void main()
{
  if (params.terrain.y < 0.0)
  {
    const vec2 corners[6] = vec2[6](
      vec2(-1, -1), vec2(-1, 1), vec2(1, -1),
      vec2(1, -1), vec2(-1, 1), vec2(1, 1));
    vec2 p = corners[gl_VertexIndex] * params.terrain.x * 20.0;
    terrainUV = p / params.terrain.x + 0.5;
    worldPosition = vec3(p.x, 0.0, p.y);
  }
  else
  {
    terrainUV = vec2(gl_VertexIndex % 257, gl_VertexIndex / 257) / 256.0;
    float height = textureLod(heightmap, heightUV(terrainUV), 0.0).r * params.terrain.y;
    vec2 p = (terrainUV - 0.5) * params.terrain.x;
    worldPosition = vec3(p.x, height, p.y);
  }
  gl_Position = params.projView * vec4(worldPosition, 1.0);
}
