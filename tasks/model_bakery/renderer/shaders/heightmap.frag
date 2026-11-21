#version 450

layout(push_constant) uniform NoiseParams
{
  vec4 noise;
  vec4 island;
} params;

layout(location = 0) out float outHeight;

vec2 gradient(ivec2 p)
{
  uint h = uint(p.x) * 1597334677u ^ uint(p.y) * 3812015801u ^ uint(params.island.w) * 2798796415u;
  h = (h ^ (h >> 16)) * 2246822519u;
  h = (h ^ (h >> 13)) * 3266489917u;
  h ^= h >> 16;
  float angle = float(h & 65535u) * (6.2831853 / 65536.0);
  return vec2(cos(angle), sin(angle));
}

float perlin(vec2 p)
{
  ivec2 cell = ivec2(floor(p));
  vec2 f = fract(p);
  vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  return mix(
    mix(dot(gradient(cell), f), dot(gradient(cell + ivec2(1, 0)), f - vec2(1, 0)), u.x),
    mix(dot(gradient(cell + ivec2(0, 1)), f - vec2(0, 1)),
        dot(gradient(cell + ivec2(1, 1)), f - vec2(1, 1)), u.x), u.y);
}

void main()
{
  vec2 uv = (gl_FragCoord.xy - 0.5) / 1023.0;
  vec2 p = uv * params.noise.x + vec2(13.7, 29.1);
  float amplitude = 1.0;
  float total = 0.0;
  float weight = 0.0;
  for (int octave = 0; octave < int(params.noise.w); ++octave)
  {
    total += amplitude * perlin(p);
    weight += amplitude;
    amplitude *= params.noise.y;
    p = p * params.noise.z + vec2(7.3, 5.2);
  }
  float radius = length(uv * 2.0 - 1.0);
  float mask = 1.0 - smoothstep(params.island.x * (1.0 - params.island.y), params.island.x, radius);
  float height = pow(clamp(0.5 + total / weight, 0.0, 1.0), params.island.z);
  outHeight = height * mask - 0.035;
}
