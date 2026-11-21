#version 450

layout(set = 0, binding = 0) uniform sampler2D hdr;
layout(set = 0, binding = 1, std430) readonly buffer Statistics
{
  uint minimum;
  uint maximum;
  uint bins[100];
  float exposure;
} stats;
layout(set = 0, binding = 2) uniform sampler3D colorLut;

layout(push_constant) uniform Settings
{
  int tonemap;
  int grading;
  float compensation;
  int srgb;
} settings;

layout(location = 0) out vec4 outColor;

vec3 encode(vec3 c)
{
  return mix(12.92 * c, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, greaterThan(c, vec3(0.0031308)));
}

vec3 decode(vec3 c)
{
  return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), greaterThan(c, vec3(0.04045)));
}

void main()
{
  vec3 color = texelFetch(hdr, ivec2(gl_FragCoord.xy), 0).rgb;

  if (settings.tonemap != 0)
  {
    color *= stats.exposure * exp2(settings.compensation);
    color = color / (vec3(1) + color);
  }

  color = encode(clamp(color, 0.0, 1.0));

  if (settings.grading != 0)
    color = texture(colorLut, (color * 31.0 + 0.5) / 32.0).rgb;

  if (settings.srgb != 0)
    color = decode(color);

  outColor = vec4(color, 1);
}
