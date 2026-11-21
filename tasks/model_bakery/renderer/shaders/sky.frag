#version 450

layout(push_constant) uniform SkyParameters
{
  vec4 sun;
  vec4 color;
  vec4 clouds;
  vec4 layer;
} sky;

layout(location = 0) out vec4 outColor;

float hash(vec3 p)
{
  p = fract(p * 0.1031);
  p += dot(p, p.yzx + 33.33);
  return fract((p.x + p.y) * p.z);
}

float noise(vec3 p)
{
  vec3 i = floor(p);
  vec3 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);

  return mix(
    mix(mix(hash(i), hash(i + vec3(1, 0, 0)), f.x),
        mix(hash(i + vec3(0, 1, 0)), hash(i + vec3(1, 1, 0)), f.x), f.y),
    mix(mix(hash(i + vec3(0, 0, 1)), hash(i + vec3(1, 0, 1)), f.x),
        mix(hash(i + vec3(0, 1, 1)), hash(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}

float density(vec3 p)
{
  float h = (p.y - sky.layer.x) / (sky.layer.y - sky.layer.x);

  if (h <= 0.0 || h >= 1.0)
    return 0.0;

  float distanceFade = 1.0 - smoothstep(15000.0, 60000.0, length(p.xz));
  p /= sky.clouds.z;
  float sum = 0.0;
  float amplitude = 0.533333;

  for (int i = 0; i < 4; ++i)
  {
    sum += amplitude * noise(p);
    p = p * 2.03 + vec3(17.1, 9.2, 13.4);
    amplitude *= 0.5;
  }

  float shape = smoothstep(1.0 - sky.clouds.x, 1.15 - sky.clouds.x, sum);
  return shape * sky.clouds.y * distanceFade * smoothstep(0.0, 0.15, h) * (1.0 - smoothstep(0.75, 1.0, h));
}

void main()
{
  vec2 uv = gl_FragCoord.xy / 2048.0 * 2.0 - 1.0;
  int face = int(sky.layer.w);
  vec3 directions[6] = vec3[6](
    vec3(1, -uv.y, -uv.x), vec3(-1, -uv.y, uv.x),
    vec3(uv.x, 1, uv.y), vec3(uv.x, -1, -uv.y),
    vec3(uv.x, -uv.y, 1), vec3(-uv.x, -uv.y, -1));
  vec3 rayDir = normalize(directions[face]);
  vec3 sunDir = normalize(sky.sun.xyz);
  float t = clamp(rayDir.y * 0.5 + 0.5, 0.0, 1.0);
  vec3 background = mix(vec3(0.55, 0.75, 1.0), vec3(0.025, 0.12, 0.38), t);
  float theta = acos(clamp(dot(rayDir, sunDir), -1.0, 1.0));
  background += sky.color.rgb * sky.sun.w *
    (exp(-theta * theta * 9000.0) + exp(-theta * 42.0));

  vec3 accumulated = vec3(0);
  float transmission = 1.0;

  if (rayDir.y > 0.0001 && sky.clouds.x > 0.0 && sky.clouds.y > 0.0)
  {
    float start = sky.layer.x / rayDir.y;
    float end = min(sky.layer.y / rayDir.y, 60000.0);
    float stepLength = max(end - start, 0.0) / sky.clouds.w;
    float shadowLength = (sky.layer.y - sky.layer.x) / max(abs(sunDir.y), 0.1);
    float shadowStep = shadowLength / sky.layer.z;

    for (int i = 0; i < int(sky.clouds.w) && transmission > 0.005; ++i)
    {
      vec3 p = rayDir * (start + (float(i) + 0.5) * stepLength);
      float d = density(p);

      if (d < 0.00001)
        continue;

      float opticalDepth = 0.0;

      for (int j = 0; j < int(sky.layer.z); ++j)
        opticalDepth += density(p + sunDir * ((float(j) + 0.5) * shadowStep)) * shadowStep;

      vec3 illumination = vec3(0.25, 0.32, 0.42) +
        sky.color.rgb * sky.sun.w * exp(-opticalDepth);
      float alpha = 1.0 - exp(-d * stepLength);
      accumulated += transmission * alpha * illumination;
      transmission *= 1.0 - alpha;
    }
  }

  outColor = vec4(accumulated + transmission * background, transmission);
}
