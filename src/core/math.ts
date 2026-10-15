export const TAU = Math.PI * 2

export const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value))
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t
export const saturate = (value: number) => clamp(value, 0, 1)

/** Maps any finite angle into (-π, π]. */
export function wrapAngle(angle: number): number {
  const wrapped = angle - TAU * Math.floor((angle + Math.PI) / TAU)
  return wrapped === -Math.PI ? Math.PI : wrapped
}

/** Interpolates along the shorter arc, so 179° to -179° turns 2°, not 358°. */
export const lerpAngle = (a: number, b: number, t: number) => wrapAngle(a + wrapAngle(b - a) * t)

/** sin(x)/x with a Taylor branch near zero, where the quotient loses precision. */
export const sinc = (x: number) => (Math.abs(x) < 1e-4 ? 1 - (x * x) / 6 : Math.sin(x) / x)

/** Frame-rate independent exponential smoothing factor for a half-life in seconds. */
export const dampFactor = (halfLife: number, dt: number) => (halfLife <= 0 ? 1 : 1 - Math.pow(2, -dt / halfLife))

/** Deterministic 32-bit PRNG (mulberry32) returning floats in [0, 1). */
export function createRng(seed: number): () => number {
  let state = (seed >>> 0) || 0x9e3779b9
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** FNV-1a string hash, used to give planner task ids stable map positions. */
export function hashString(text: string): number {
  let hash = 0x811c9dc5
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193)
  }
  return hash >>> 0
}
