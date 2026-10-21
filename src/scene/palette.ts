import * as THREE from 'three'

/** Scene colours. CSS mirrors the same names in styles/tokens.css. */
export const SCENE = {
  horizon: '#141821',
  basalt: '#252a33',
  grid: '#3a414d',
  ceramic: '#e7eaed',
  foil: '#e2a93b',
  signal: '#ff5a1f',
  mint: '#4fd1b5',
  baseline: '#6f86ae',
  fogNear: 16,
  fogFar: 46,
} as const

const LOW = new THREE.Color(SCENE.signal)
const MID = new THREE.Color(SCENE.foil)
const HIGH = new THREE.Color(SCENE.mint)

/** Trust ramp: signal orange (0) through foil gold (0.5) to mint (1). */
export function trustColor(trust: number, out: THREE.Color): THREE.Color {
  const t = Math.min(1, Math.max(0, Number.isFinite(trust) ? trust : 0.5))
  return t < 0.5 ? out.copy(LOW).lerp(MID, t * 2) : out.copy(MID).lerp(HIGH, (t - 0.5) * 2)
}

export const trustCss = (trust: number) => `#${trustColor(trust, new THREE.Color()).getHexString()}`

/**
 * Shared GLSL for the custom shaders: linear fog like THREE.Fog, or
 * exponential-squared fog like THREE.FogExp2 when uFogDensity > 0.
 */
export const FOG_GLSL = /* glsl */ `
uniform vec3 uFogColor;
uniform vec2 uFogRange;
uniform float uFogDensity;
float fogFactor(float depth) {
  if (uFogDensity > 0.0) return 1.0 - exp(-uFogDensity * uFogDensity * depth * depth);
  return clamp((depth - uFogRange.x) / (uFogRange.y - uFogRange.x), 0.0, 1.0);
}
`
export const fogUniforms = () => ({
  uFogColor: { value: new THREE.Color(SCENE.horizon) },
  uFogRange: { value: new THREE.Vector2(SCENE.fogNear, SCENE.fogFar) },
  uFogDensity: { value: 0 },
})
type FogUniforms = ReturnType<typeof fogUniforms>

/** Copies the scene's fog into a custom shader's fog uniforms, so every mode's atmosphere reaches them. */
export function syncFog(uniforms: FogUniforms, fog: THREE.Fog | THREE.FogExp2 | null): void {
  if (fog instanceof THREE.FogExp2) {
    uniforms.uFogColor.value.copy(fog.color)
    uniforms.uFogDensity.value = fog.density
  } else if (fog instanceof THREE.Fog) {
    uniforms.uFogColor.value.copy(fog.color)
    uniforms.uFogRange.value.set(fog.near, fog.far)
    uniforms.uFogDensity.value = 0
  } else {
    uniforms.uFogRange.value.set(1e6, 2e6)
    uniforms.uFogDensity.value = 0
  }
}
