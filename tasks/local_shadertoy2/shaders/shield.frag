#version 450

layout(push_constant) uniform PC {
    vec4 time_res;
} pc;

layout(location = 0) out vec4 outColor;

void main() {
    vec2 uv = gl_FragCoord.xy / pc.time_res.yz;
    float shieldCenter = fract(pc.time_res.x * 0.25);
    float dist = abs(uv.y - shieldCenter);
    float waveWidth = 0.06;

    if (dist < waveWidth) {
        float strength = 1.0 - smoothstep(0.0, waveWidth, dist);
        vec3 edgeColor = vec3(0.1, 0.6, 1.0);
        vec3 coreColor = vec3(0.8, 0.85, 1.0);
        vec3 shieldRgb = mix(edgeColor, coreColor, pow(strength, 2.5));
        outColor = vec4(shieldRgb, strength);
    } else {
        outColor = vec4(0.0);
    }
}
