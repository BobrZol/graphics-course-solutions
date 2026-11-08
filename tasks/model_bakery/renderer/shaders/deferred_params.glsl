layout(set = 0, binding = 4, std140) uniform DeferredParams
{
  mat4 inverseViewProjection;
  mat4 view;
  mat4 inverseProjection;
  vec4 cameraExposure;
  vec4 sun;
  vec4 light;
  vec4 screen;
  ivec4 options;
  vec4 decal;
  mat4 decalInverse[4];
} scene;

vec3 reconstructPosition(ivec2 pixel, float depth)
{
  vec2 ndc = (vec2(pixel) + 0.5) / scene.screen.xy * 2.0 - 1.0;
  vec4 position = scene.inverseViewProjection * vec4(ndc, depth, 1.0);

  return position.xyz / position.w;
}

uint clusterIndex(ivec2 pixel, float viewDepth)
{
  uvec2 tile = min(uvec2((vec2(pixel) + 0.5) / scene.screen.xy * vec2(16, 9)), uvec2(15, 8));
  uint slice = uint(clamp(floor(log(max(viewDepth, scene.screen.z) / scene.screen.z) /
    log(scene.screen.w / scene.screen.z) * 24.0), 0.0, 23.0));

  return (slice * 9 + tile.y) * 16 + tile.x;
}
