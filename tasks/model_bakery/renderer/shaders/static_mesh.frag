#version 450
#extension GL_GOOGLE_include_directive : require

#include "terrain_params.glsl"

layout(set = 0, binding = 1) uniform sampler2DArray albedoMaps;
layout(set = 0, binding = 2) uniform sampler2DArray roughnessMaps;
layout(set = 0, binding = 3) uniform sampler2DArray detailMaps;

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec2 terrainUV;
layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

vec3 lighting(vec3 albedo, float roughness, vec3 n)
{
  vec3 v = normalize(params.cameraExposure.xyz - worldPosition);
  vec3 l = normalize(params.sun.xyz);
  vec3 h = normalize(v + l);
  float nv = max(dot(n, v), 0.001);
  float nl = max(dot(n, l), 0.0);
  float nh = max(dot(n, h), 0.0);
  float hv = max(dot(h, v), 0.0);
  float a = roughness * roughness;
  float a2 = a * a;
  float d = nh * nh * (a2 - 1.0) + 1.0;
  float distribution = a2 / max(PI * d * d, 0.000001);
  float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
  float geometry = nv / (nv * (1.0 - k) + k) * nl / (nl * (1.0 - k) + k);
  vec3 fresnel = vec3(0.04) + 0.96 * pow(1.0 - hv, 5.0);
  vec3 specular = distribution * geometry * fresnel / max(4.0 * nv * nl, 0.001);
  vec3 direct = ((1.0 - fresnel) * albedo / PI + specular) * params.light.rgb * params.sun.w * nl;
  vec3 ambient = albedo * params.light.w * mix(vec3(0.35, 0.4, 0.5), vec3(0.8, 0.9, 1.0), n.y * 0.5 + 0.5);
  return direct + ambient;
}

void main()
{
  vec3 albedo;
  float roughness;
  vec3 normal;
  if (params.terrain.y < 0.0)
  {
    albedo = vec3(0.025, 0.18, 0.24);
    roughness = 0.24;
    normal = vec3(0, 1, 0);
  }
  else
  {
    vec2 texel = 1.0 / vec2(textureSize(heightmap, 0));
    vec2 uv = heightUV(terrainUV);
    float left = textureLod(heightmap, uv - vec2(texel.x, 0), 0.0).r;
    float right = textureLod(heightmap, uv + vec2(texel.x, 0), 0.0).r;
    float down = textureLod(heightmap, uv - vec2(0, texel.y), 0.0).r;
    float up = textureLod(heightmap, uv + vec2(0, texel.y), 0.0).r;
    float spacing = params.terrain.x / 1023.0;
    vec2 slope = vec2(right - left, up - down) * params.terrain.y / (2.0 * spacing);
    normal = normalize(vec3(-slope.x, 1.0, -slope.y));

    float height = worldPosition.y / params.terrain.y;
    float grass = smoothstep(0.035, 0.11, height);
    float rock = smoothstep(0.24, 0.38, height);
    float snow = smoothstep(0.43, 0.57, height);
    vec4 weights = vec4(1.0 - grass, grass * (1.0 - rock), grass * rock * (1.0 - snow), grass * rock * snow);
    vec2 materialUV = terrainUV * params.terrain.z;
    vec2 dx = dFdx(materialUV);
    vec2 dy = dFdy(materialUV);
    vec2 detailTexel = 1.0 / vec2(textureSize(detailMaps, 0).xy);
    vec2 detailSlope = vec2(0);
    albedo = vec3(0);
    roughness = 0.0;
    for (int layer = 0; layer < 4; ++layer)
    {
      vec3 p = vec3(materialUV, float(layer));
      albedo += weights[layer] * textureGrad(albedoMaps, p, dx, dy).rgb;
      roughness += weights[layer] * textureGrad(roughnessMaps, p, dx, dy).r;
      float hl = textureGrad(detailMaps, p - vec3(detailTexel.x, 0, 0), dx, dy).r;
      float hr = textureGrad(detailMaps, p + vec3(detailTexel.x, 0, 0), dx, dy).r;
      float hd = textureGrad(detailMaps, p - vec3(0, detailTexel.y, 0), dx, dy).r;
      float hu = textureGrad(detailMaps, p + vec3(0, detailTexel.y, 0), dx, dy).r;
      detailSlope += weights[layer] * vec2(hr - hl, hu - hd) / (2.0 * detailTexel);
    }
    vec3 detailGradient = vec3(detailSlope.x, 0, detailSlope.y) *
      params.terrain.w * params.terrain.z / params.terrain.x;
    normal = normalize(normal - (detailGradient - normal * dot(normal, detailGradient)));
  }
  vec3 color = lighting(albedo, clamp(roughness, 0.045, 1.0), normal);
  color = vec3(1) - exp(-color * abs(params.cameraExposure.w));
  if (params.cameraExposure.w < 0.0)
    color = pow(color, vec3(1.0 / 2.2));
  outColor = vec4(color, 1.0);
}
