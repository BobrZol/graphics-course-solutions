#version 450

layout(push_constant) uniform PC {
    vec4 time_res;
} pc;

layout(location = 0) out vec4 outColor;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

void main() {
    vec2 uv = gl_FragCoord.xy / pc.time_res.yz;
    vec2 grid = uv * vec2(60.0, 30.0);
    vec2 plateId = floor(grid);
    vec2 plateFract = fract(grid);
    vec2 seam = smoothstep(0.0, 0.06, plateFract)
        * (1.0 - smoothstep(0.94, 1.0, plateFract));
    float isPlate = seam.x * seam.y;
    float rnd = hash(plateId);
    float plateRoughness = mix(0.2, 0.5, rnd);
    float roughness = mix(0.85, plateRoughness, isPlate);
    float plateMetallic = mix(0.1, 0.35, hash(plateId + 17.0));
    float metallic = mix(0.0, plateMetallic, isPlate);
    outColor = vec4(roughness, metallic, 0.0, 1.0);
}
