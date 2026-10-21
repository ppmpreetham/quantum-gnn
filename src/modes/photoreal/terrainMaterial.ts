import * as THREE from 'three'

export interface GroundTextures { color: THREE.Texture; normal: THREE.Texture; arh: THREE.Texture; tileMeters: number }

const VERTEX_DECL = /* glsl */ `
varying vec3 vTerrainWorld;
varying vec3 vTerrainNormal;
`

const VERTEX_BODY = /* glsl */ `
vTerrainWorld = (modelMatrix * vec4(transformed, 1.0)).xyz;
vTerrainNormal = normalize(mat3(modelMatrix) * objectNormal);
`

/*
 * Two scanned ground sets (gravel and sand) blended by their own height
 * maps under a low-frequency mask, each sampled at two scales that cross-fade
 * with distance to hide tiling. AO, roughness and height share one packed
 * texture (R, G, B). The tangent frame is world-aligned (u along +x, image up
 * along -z), so no tangent attribute is needed on the terrain mesh.
 */
const FRAGMENT_DECL = /* glsl */ `
uniform sampler2D tColorA, tNormalA, tArhA, tColorB, tNormalB, tArhB;
uniform vec2 uTile;
uniform float uDetail;
varying vec3 vTerrainWorld;
varying vec3 vTerrainNormal;

float terrainHash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float terrainNoise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(terrainHash(i), terrainHash(i + vec2(1.0, 0.0)), u.x),
             mix(terrainHash(i + vec2(0.0, 1.0)), terrainHash(i + vec2(1.0, 1.0)), u.x), u.y) * 2.0 - 1.0;
}

struct GroundSample { vec3 color; vec3 normal; vec3 arh; };

// uv = rot * xz / tile; the sampled normal's xy turns by the same rotation into the shared world-aligned frame
GroundSample sampleGround(sampler2D c, sampler2D n, sampler2D a, vec2 uv, mat2 rot, float farMix) {
  vec2 farUv = uv * 0.23 + vec2(0.31, 0.67);
  GroundSample s;
  s.color = mix(texture(c, uv).rgb, texture(c, farUv).rgb, farMix * 0.55);
  vec3 nearN = texture(n, uv).xyz * 2.0 - 1.0;
  vec3 farN = texture(n, farUv).xyz * 2.0 - 1.0;
  s.normal = normalize(mix(nearN, farN, farMix * 0.6) * vec3(1.0, 1.0, 1.0 + farMix * 1.5));
  s.normal.xy = rot * s.normal.xy;
  s.arh = mix(texture(a, uv).rgb, texture(a, farUv).rgb, farMix * 0.55);
  return s;
}

vec3 groundColor;
float groundRoughness;
float groundOcclusion;
vec3 groundNormalTS;
`

const FRAGMENT_GROUND = /* glsl */ `
{
  vec2 xz = vTerrainWorld.xz;
  float farMix = smoothstep(6.0, 45.0, length(vViewPosition)) * uDetail;
  // set B is turned and scaled against set A: equal, aligned tilings beat into visible bands
  mat2 identity = mat2(1.0);
  mat2 turn = mat2(0.7648, 0.6442, -0.6442, 0.7648);
  GroundSample a = sampleGround(tColorA, tNormalA, tArhA, xz * uTile.x, identity, farMix);
  GroundSample b = sampleGround(tColorB, tNormalB, tArhB, turn * xz * uTile.y + vec2(0.43, 0.19), turn, farMix);
  float mask = terrainNoise(xz * 0.07) * 0.6 + terrainNoise(xz * 0.23) * 0.3 + terrainNoise(xz * 1.1) * 0.1;
  float wB = smoothstep(-0.12, 0.12, (b.arh.b - a.arh.b) + mask * 1.3);
  groundColor = mix(a.color, b.color, wB);
  groundRoughness = mix(a.arh.g, b.arh.g, wB);
  groundOcclusion = mix(a.arh.r, b.arh.r, wB);
  groundNormalTS = normalize(mix(a.normal, b.normal, wB));
  float macro = terrainNoise(xz * 0.018) * 0.6 + terrainNoise(xz * 0.061) * 0.4;
  groundColor *= 1.0 + macro * 0.11;
  groundColor = mix(groundColor, groundColor * vec3(1.07, 0.98, 0.9), clamp(macro, 0.0, 1.0) * 0.6);
}
`

const FRAGMENT_NORMAL = /* glsl */ `
{
  // gentle metre-scale undulation (tyre ruts, wind scour) from the gradient of a noise field
  vec2 p = vTerrainWorld.xz * 0.55;
  vec2 slope = vec2(terrainNoise(p + vec2(0.04, 0.0)) - terrainNoise(p - vec2(0.04, 0.0)),
                    terrainNoise(p + vec2(0.0, 0.04)) - terrainNoise(p - vec2(0.0, 0.04))) * 0.45;
  vec3 N = normalize(normalize(vTerrainNormal) - vec3(slope.x, 0.0, slope.y));
  vec3 T = normalize(vec3(1.0, 0.0, 0.0) - N * N.x);
  vec3 B = cross(N, T);
  vec3 worldNormal = normalize(T * groundNormalTS.x + B * groundNormalTS.y + N * groundNormalTS.z);
  normal = normalize((viewMatrix * vec4(worldNormal, 0.0)).xyz);
}
`

const FRAGMENT_AO = /* glsl */ `
{
  float ambientOcclusion = groundOcclusion;
  reflectedLight.indirectDiffuse *= ambientOcclusion;
  #if defined( USE_ENVMAP ) && defined( STANDARD )
    float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
    reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
  #endif
}
`

/**
 * MeshStandardMaterial with the field-site ground spliced into its shader,
 * so it keeps three.js lighting, shadows, fog and image-based light.
 * detail = false (low tier) drops the far-scale samples.
 */
export function createTerrainMaterial(a: GroundTextures, b: GroundTextures, detail: boolean): THREE.MeshStandardMaterial {
  const material = new THREE.MeshStandardMaterial({ roughness: 1, metalness: 0, envMapIntensity: 0.9 })
  const uniforms = {
    tColorA: { value: a.color }, tNormalA: { value: a.normal }, tArhA: { value: a.arh },
    tColorB: { value: b.color }, tNormalB: { value: b.normal }, tArhB: { value: b.arh },
    uTile: { value: new THREE.Vector2(1 / a.tileMeters, 1 / b.tileMeters) },
    uDetail: { value: detail ? 1 : 0 },
  }
  material.onBeforeCompile = shader => {
    Object.assign(shader.uniforms, uniforms)
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\n${VERTEX_DECL}`)
      .replace('#include <fog_vertex>', `#include <fog_vertex>\n${VERTEX_BODY}`)
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>\n${FRAGMENT_DECL}`)
      .replace('#include <map_fragment>', `${FRAGMENT_GROUND}\ndiffuseColor.rgb *= groundColor;`)
      .replace('#include <roughnessmap_fragment>', 'float roughnessFactor = roughness * groundRoughness;')
      .replace('#include <normal_fragment_maps>', FRAGMENT_NORMAL)
      .replace('#include <aomap_fragment>', FRAGMENT_AO)
  }
  material.customProgramCacheKey = () => `field-terrain-${detail ? 'detail' : 'flat'}`
  return material
}
