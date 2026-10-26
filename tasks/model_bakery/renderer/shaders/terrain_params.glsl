layout(push_constant) uniform TerrainParams
{
  mat4 projView;
  vec4 cameraExposure;
  vec4 terrain;
  vec4 sun;
  vec4 light;
} params;

layout(set = 0, binding = 0) uniform sampler2D heightmap;

vec2 heightUV(vec2 uv)
{
  return (uv * 1023.0 + 0.5) / 1024.0;
}
