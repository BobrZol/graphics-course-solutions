#version 450

layout(push_constant) uniform PC {
    vec4 time_res;
} pc;

layout(set = 0, binding = 0) uniform sampler2D iChannel0;
layout(set = 0, binding = 1) uniform sampler2D iChannel1;
layout(location = 0) out vec4 outColor;

#define iTime pc.time_res.x
#define iResolution vec3(pc.time_res.yz, 1.0)

const float PI = 3.14159265359;

float sdSphere(vec3 p, float r) {
    return length(p) - r;
}

float sdBox(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0);
}

float mapLedge(vec3 p) {
    float ledge = max(sdSphere(p, 1.015), -sdBox(p - vec3(2, 0.34, 0.25), vec3(2, 0.3, 0.3)));
    ledge = max(ledge, -sdBox(p - vec3(1, 0.7, -0.4), vec3(2, 4, 0.1)));
    ledge = max(ledge, -sdBox(p - vec3(1, 0.7, 0.8), vec3(2, 4, 0.05)));
    ledge = max(ledge, -sdBox(p - vec3(1, 0.7, 0.8), vec3(2, 0.09, 4)));
    ledge = max(ledge, -sdBox(p - vec3(1, -0.65, 0.8), vec3(2, 0.04, 4)));
    ledge = max(ledge, -sdBox(p - vec3(1, -0.3, 0.8), vec3(2, 0.04, 4)));
    ledge = max(ledge, -sdBox(p, vec3(0.07, 4, 4)));
    ledge = max(ledge, -sdBox(p - vec3(1, 0.7, 0.2), vec3(4, 4, 0.1)));
    
    float equatorial_trough = sdBox(p, vec3(5, 0.05, 5));
    ledge = max(ledge, -equatorial_trough);
    return ledge;
}

float map(vec3 p) {
    float death_star = sdSphere(p, 1.0);
    float core = sdSphere(p, 0.94);
    float laser_focusing_lens = sdSphere(p - vec3(1.7, 0.75, 0.5), 1.0);
    float equatorial_trough = sdBox(p, vec3(5, 0.05, 5));
    float ledge = mapLedge(p);
    
    return min(min(max(max(death_star, -equatorial_trough), -laser_focusing_lens), core), ledge);
}

vec3 calcNormal(vec3 p) {
    vec2 e = vec2(0.001, 0.0);
    return normalize(vec3(
        map(p + e.xyy) - map(p - e.xyy),
        map(p + e.yxy) - map(p - e.yxy),
        map(p + e.yyx) - map(p - e.yyx)
    ));
}

vec2 rayMarch(vec3 ro, vec3 rd) {
    float dO = 0.0;
    for(int i = 0; i < 100; i++) {
        vec3 p = ro + rd * dO;
        float dS = map(p);
        dO += dS;
        if(abs(dS) < 0.001)
            return vec2(dO, 1.0);
        if(dO > 100.0)
            return vec2(dO, 0.0);
    }
    return vec2(dO, 0.0);
}

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float denom = (NdotH * NdotH * (a2 - 1.0) + 1.0);
    return a2 / (PI * denom * denom);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float k = pow(roughness + 1.0, 2.0) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return GeometrySchlickGGX(NdotV, roughness) * GeometrySchlickGGX(NdotL, roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 calcPBR(vec3 p, vec3 n, vec3 ro, vec3 albedo, float roughness, float metallic) {
    vec3 lightPos = vec3(1.0, 1.0, 3.0);
    vec3 lightColor = vec3(2.5);

    vec3 v = normalize(ro - p);
    vec3 l = normalize(lightPos - p);
    vec3 h = normalize(l + v);

    float NdotL = max(dot(n, l), 0.0);
    float NdotV = max(dot(n, v), 0.0);
    float VdotH = max(dot(v, h), 0.0);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    float NDF = DistributionGGX(n, h, roughness);
    float G   = GeometrySmith(n, v, l, roughness);
    vec3  F   = fresnelSchlick(VdotH, F0);

    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * NdotV * NdotL + 0.0001;
    vec3 specular = numerator / denominator;

    vec3 kS = F;
    vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);
    vec3 diffuse = kD * albedo / PI;

    vec3 directLight = (diffuse + specular) * lightColor * NdotL;
    vec3 ambient = vec3(0.03) * albedo;

    return ambient + directLight;
}

float getPlateHeight(vec2 uv) {
    vec2 grid = fract(uv * vec2(60.0, 30.0));
    vec2 seam = smoothstep(0.0, 0.05, grid) * (1.0 - smoothstep(0.95, 1.0, grid));
    return seam.x * seam.y;
}

vec3 applyNormalMap(vec3 p, vec3 n, vec2 uv) {
    vec2 e = vec2(0.002, 0.0);
    float hCenter = getPlateHeight(uv);
    float hRight  = getPlateHeight(uv + e.xy);
    float hUp     = getPlateHeight(uv + e.yx);
    
    vec2 grad = vec2(hRight - hCenter, hUp - hCenter) / e.x;
    vec3 tangent   = normalize(vec3(-p.z, 0.0, p.x));
    vec3 bitangent = cross(n, tangent);
    float bumpStrength = 0.15;

    return normalize(n - (tangent * grad.x + bitangent * grad.y) * bumpStrength);
}

float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

vec3 drawStars(vec3 rd) {
    vec2 uv = vec2(atan(rd.x, rd.z), asin(rd.y)) * 20.0;
    vec2 grid = floor(uv);
    float rnd = random(grid);
    
    if (rnd > 0.75) {
        vec2 cellPos = fract(uv) - 0.5;
        vec2 offset = vec2(random(grid + 13.0), random(grid + 78.0)) - 0.5;
        cellPos -= offset * 0.4;
        
        float dist = length(cellPos);
        float size = 0.02 + rnd * 0.04;
        float star = (1.0 - smoothstep(0.0, size, dist));
        vec3 starColor = mix(vec3(1.0, 0.9, 0.8), vec3(0.8, 0.9, 1.0), random(grid + 42.0));
        return star * starColor;
    }
    return vec3(0.0);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;

    float radius = 4.0;
    float height = 0.0;
    float speed = 1.0;
    
    vec3 ro = vec3(
        cos(iTime * speed) * radius,
        height,
        sin(iTime * speed) * radius
    );
    
    vec3 target = vec3(0.0, 0.0, 0.0);
    vec3 forward = normalize(target - ro);
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward));
    vec3 up = cross(forward, right);
    vec3 rd = normalize(forward + uv.x * right + uv.y * up);
    
    vec2 result = rayMarch(ro, rd);
    vec3 p = ro + rd * result.x;
    vec3 sp = normalize(p);
    vec2 sphereUV = vec2(
        atan(sp.z, sp.x) / (2.0 * PI) + 0.5,
        asin(clamp(sp.y, -1.0, 1.0)) / PI + 0.5
    );
    vec2 uvDx = dFdx(sphereUV);
    vec2 uvDy = dFdy(sphereUV);
    uvDx.x -= round(uvDx.x);
    uvDy.x -= round(uvDy.x);

    if(result.y > 0.5) {
        vec3 n = calcNormal(p);
        n = applyNormalMap(p, n, sphereUV);

        vec2 pbrParams = textureGrad(iChannel1, sphereUV, uvDx, uvDy).rg;
        float roughness = clamp(pbrParams.r, 0.05, 1.0);
        float metallic  = clamp(pbrParams.g, 0.0, 1.0);

        float ledgeDist = mapLedge(p);
        float sphereDist = sdSphere(p, 1.0);
        vec3 baseAlbedo = (ledgeDist < sphereDist + 0.01) ? vec3(0.4) : vec3(0.65);

        vec4 shieldTex = textureGrad(iChannel0, sphereUV, uvDx, uvDy);
        vec3 finalAlbedo = mix(baseAlbedo, shieldTex.rgb, shieldTex.a);

        vec3 color = calcPBR(p, n, ro, finalAlbedo, roughness, metallic);

        color += shieldTex.rgb * shieldTex.a * 1.5;

        fragColor = vec4(color, 1.0);
    } else {
        vec3 stars = drawStars(rd);
        vec3 skyColor = vec3(0.02, 0.02, 0.05) + stars;
        fragColor = vec4(skyColor, 1.0);
    }
}

void main() {
    mainImage(outColor, vec2(gl_FragCoord.x, iResolution.y - gl_FragCoord.y));
}
