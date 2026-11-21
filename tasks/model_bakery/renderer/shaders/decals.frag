#version 450
#extension GL_GOOGLE_include_directive : require

#include "deferred_params.glsl"
#include "gbuffer.glsl"

layout(set = 0, binding = 0) uniform sampler2D albedoBuffer;
layout(set = 0, binding = 1) uniform sampler2D normalBuffer;
layout(set = 0, binding = 2) uniform sampler2D materialBuffer;
layout(set = 0, binding = 3) uniform sampler2D depthBuffer;
layout(set = 0, binding = 7) uniform sampler2DArray decalAlbedo;
layout(set = 0, binding = 8) uniform sampler2DArray decalRoughness;
layout(set = 0, binding = 9) uniform sampler2DArray decalHeight;

layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec2 outNormal;
layout(location = 2) out vec4 outMaterial;

void main()
{
  ivec2 pixel = ivec2(gl_FragCoord.xy);
  outAlbedo = texelFetch(albedoBuffer, pixel, 0);
  outNormal = texelFetch(normalBuffer, pixel, 0).rg;
  outMaterial = texelFetch(materialBuffer, pixel, 0);
  float depth = texelFetch(depthBuffer, pixel, 0).r;

  if (depth >= 1.0)
    return;

  vec3 position = reconstructPosition(pixel, depth);
  vec3 normal = decodeNormal(outNormal);

  for (int i = 0; i < 4; ++i)
  {
    vec3 local = (scene.decalInverse[i] * vec4(position, 1)).xyz;

    if (any(greaterThan(abs(local), vec3(0.5))))
      continue;

    vec2 uv = local.xz + 0.5;
    vec3 coordinates = vec3(uv * 2.0, float(i % 3));
    vec4 paint = textureLod(decalAlbedo, coordinates, 0);
    float border = 1.0 - smoothstep(0.35, 0.5, max(abs(local.x), abs(local.z)));
    float alpha = scene.decal.x * paint.a * border;
    outAlbedo.rgb = mix(outAlbedo.rgb, paint.rgb, alpha * scene.decal.y);
    float roughness = textureLod(decalRoughness, coordinates, 0).r;
    outMaterial.r = mix(outMaterial.r, roughness, alpha * scene.decal.w);

    vec2 texel = 1.0 / vec2(textureSize(decalHeight, 0).xy);
    float left = textureLod(decalHeight, coordinates - vec3(texel.x, 0, 0), 0).r;
    float right = textureLod(decalHeight, coordinates + vec3(texel.x, 0, 0), 0).r;
    float down = textureLod(decalHeight, coordinates - vec3(0, texel.y, 0), 0).r;
    float up = textureLod(decalHeight, coordinates + vec3(0, texel.y, 0), 0).r;
    vec3 tangent = normalize(vec3(scene.decalInverse[i][0][0], 0, scene.decalInverse[i][2][0]));
    vec3 bitangent = normalize(vec3(scene.decalInverse[i][0][2], 0, scene.decalInverse[i][2][2]));
    vec3 gradient = (tangent * (right - left) + bitangent * (up - down)) * 8.0;
    vec3 detailNormal = normalize(normal - gradient + normal * dot(normal, gradient));
    normal = normalize(mix(normal, detailNormal, alpha * scene.decal.z));
  }

  outNormal = encodeNormal(normal);
}
