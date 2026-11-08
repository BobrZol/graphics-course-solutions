#version 450
#extension GL_GOOGLE_include_directive : require

#include "deferred_params.glsl"
#include "gbuffer.glsl"
#include "pbr.glsl"
#include "local_lights.glsl"

layout(set = 0, binding = 0) uniform sampler2D albedoBuffer;
layout(set = 0, binding = 1) uniform sampler2D normalBuffer;
layout(set = 0, binding = 2) uniform sampler2D materialBuffer;
layout(set = 0, binding = 3) uniform sampler2D depthBuffer;

layout(set = 0, binding = 6, std430) readonly buffer ClusterLights
{
  uint entries[];
};

layout(location = 0) out vec4 outColor;

void main()
{
  ivec2 pixel = ivec2(gl_FragCoord.xy);
  float depth = texelFetch(depthBuffer, pixel, 0).r;

  if (depth >= 1.0)
  {
    outColor = vec4(0.32, 0.52, 0.72, 1.0);
    return;
  }

  vec3 position = reconstructPosition(pixel, depth);
  vec3 albedo = texelFetch(albedoBuffer, pixel, 0).rgb;
  vec3 n = decodeNormal(texelFetch(normalBuffer, pixel, 0).rg);
  vec3 material = texelFetch(materialBuffer, pixel, 0).rgb;
  vec3 v = normalize(scene.cameraExposure.xyz - position);
  float roughness = max(material.r, 0.045);
  float metallic = material.g;

  vec3 color = evaluateBRDF(albedo, roughness, metallic, n, v, normalize(scene.sun.xyz)) *
    scene.light.rgb * scene.sun.w;
  color += albedo * (1.0 - metallic) * material.b * scene.light.w *
    mix(vec3(0.35, 0.4, 0.5), vec3(0.8, 0.9, 1.0), n.y * 0.5 + 0.5);

  uint offset = clusterIndex(pixel, -(scene.view * vec4(position, 1)).z) * 1025;
  uint count = scene.options.y != 0 ? entries[offset] : uint(scene.options.x);

  for (uint i = 0; i < count; ++i)
  {
    uint index = scene.options.y != 0 ? entries[offset + 1 + i] : i;
    LocalLight source = lights[index];
    vec3 delta = source.positionRadius.xyz - position;
    float distanceSquared = dot(delta, delta);
    float radiusSquared = source.positionRadius.w * source.positionRadius.w;

    if (distanceSquared >= radiusSquared)
      continue;

    float falloff = pow(1.0 - distanceSquared / radiusSquared, 2.0) / max(distanceSquared, 1.0);
    vec3 l = delta * inversesqrt(max(distanceSquared, 0.0001));
    color += evaluateBRDF(albedo, roughness, metallic, n, v, l) *
      source.colorIntensity.rgb * source.colorIntensity.w * falloff;
  }

  color = vec3(1) - exp(-color * abs(scene.cameraExposure.w));

  if (scene.cameraExposure.w < 0.0)
    color = pow(color, vec3(1.0 / 2.2));

  outColor = vec4(color, 1);
}
