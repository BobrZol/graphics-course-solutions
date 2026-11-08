#version 450
#extension GL_GOOGLE_include_directive : require

#include "scene_params.glsl"

layout(set = 0, binding = 1) uniform sampler2D albedoMap;
layout(set = 0, binding = 2) uniform sampler2D roughnessMap;
layout(set = 0, binding = 3) uniform sampler2D normalMap;

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec2 texCoord;
layout(location = 3) in vec3 colorVariation;
layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec2 outNormal;
layout(location = 2) out vec4 outMaterial;

#include "gbuffer.glsl"

void main()
{
  vec3 normal = normalize(worldNormal);

  vec3 dpdx = dFdx(worldPosition);
  vec3 dpdy = dFdy(worldPosition);
  vec2 duvdx = dFdx(texCoord);
  vec2 duvdy = dFdy(texCoord);

  float determinant = duvdx.x * duvdy.y - duvdx.y * duvdy.x;
  vec3 tangent = (dpdx * duvdy.y - dpdy * duvdx.y) * sign(determinant);
  tangent -= normal * dot(normal, tangent);

  if (dot(tangent, tangent) > 1e-12 && abs(determinant) > 1e-12)
  {
    tangent = normalize(tangent);
    vec3 sourceBitangent = (dpdy * duvdx.x - dpdx * duvdy.x) * sign(determinant);
    vec3 bitangent = cross(normal, tangent) * sign(dot(cross(normal, tangent), sourceBitangent));
    vec3 detail = texture(normalMap, texCoord).xyz * 2.0 - 1.0;
    normal = normalize(mat3(tangent, bitangent, normal) * detail);
  }

  vec3 albedo = texture(albedoMap, texCoord).rgb * colorVariation;
  vec3 arm = texture(roughnessMap, texCoord).rgb;

  outAlbedo = vec4(albedo, 1.0);
  outNormal = encodeNormal(normal);
  outMaterial = vec4(clamp(arm.g, 0.045, 1.0), arm.b, arm.r, 1.0);
}
