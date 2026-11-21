vec2 encodeNormal(vec3 n)
{
  n /= abs(n.x) + abs(n.y) + abs(n.z);
  vec2 encoded = n.xy;

  if (n.z < 0.0)
    encoded = (1.0 - abs(encoded.yx)) * mix(vec2(-1), vec2(1), greaterThanEqual(encoded, vec2(0)));

  return encoded * 0.5 + 0.5;
}

vec3 decodeNormal(vec2 encoded)
{
  vec2 f = encoded * 2.0 - 1.0;
  vec3 n = vec3(f, 1.0 - abs(f.x) - abs(f.y));
  float t = clamp(-n.z, 0.0, 1.0);
  n.xy += mix(vec2(t), vec2(-t), greaterThanEqual(n.xy, vec2(0)));

  return normalize(n);
}
