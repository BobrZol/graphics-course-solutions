struct LocalLight
{
  vec4 positionRadius;
  vec4 colorIntensity;
};

layout(set = 0, binding = 5, std430) readonly buffer Lights
{
  LocalLight lights[];
};
