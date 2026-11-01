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

