#version 450
#extension GL_GOOGLE_include_directive : require

#include "scene_params.glsl"

struct Instance
{
  mat4 transform;
  vec4 color;
};

layout(std430, set = 0, binding = 0) readonly buffer Instances
{
  Instance instances[];
};

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec2 texCoord;
layout(location = 3) out vec3 colorVariation;

invariant gl_Position;

void main()
{
  Instance instance = instances[gl_InstanceIndex];
  worldPosition = (instance.transform * vec4(position, 1.0)).xyz;
  worldNormal = normalize(mat3(instance.transform) * normal);
  texCoord = uv;
  colorVariation = instance.color.rgb;

  gl_Position = params.projView * vec4(worldPosition, 1.0);
}
