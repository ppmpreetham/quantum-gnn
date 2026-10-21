import { ARENA } from '../../config/fleet.ts'
import { createRng } from '../../core/math.ts'

/**
 * The field site's shape and what grows on it, as pure functions so they
 * are deterministic and testable. The drivable arena is exactly flat
 * (rovers ride at y = 0); the ground eases into low dunes beyond it and
 * rises to a rim of hills that meets the HDRI horizon.
 */
export const FLAT_RADIUS = ARENA.radius + 1
const DUNE_RADIUS = FLAT_RADIUS + 26
export const TERRAIN_RADIUS = 420

const smoothstep = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)))
  return t * t * (3 - 2 * t)
}

/** Integer-lattice hash in [0, 1). */
function hash(ix: number, iz: number): number {
  let h = Math.imul(ix, 374761393) ^ Math.imul(iz, 668265263)
  h = Math.imul(h ^ (h >>> 13), 1274126177)
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296
}

/** Smooth value noise in [-1, 1]. */
export function valueNoise(x: number, z: number): number {
  const ix = Math.floor(x), iz = Math.floor(z)
  const fx = x - ix, fz = z - iz
  const ux = fx * fx * (3 - 2 * fx), uz = fz * fz * (3 - 2 * fz)
  const a = hash(ix, iz), b = hash(ix + 1, iz), c = hash(ix, iz + 1), d = hash(ix + 1, iz + 1)
  return (a + (b - a) * ux + (c - a) * uz + (a - b - c + d) * ux * uz) * 2 - 1
}

function fbm(x: number, z: number, octaves: number): number {
  let sum = 0, amplitude = 0.5, frequency = 1, norm = 0
  for (let o = 0; o < octaves; o++) {
    sum += valueNoise(x * frequency + o * 17.3, z * frequency - o * 9.1) * amplitude
    norm += amplitude
    amplitude *= 0.5
    frequency *= 2.03
  }
  return sum / norm
}

/** Ground height in metres at a point; 0 everywhere a rover can reach. */
export function terrainHeight(x: number, z: number): number {
  const r = Math.hypot(x, z)
  if (r <= FLAT_RADIUS) return 0
  const near = smoothstep(FLAT_RADIUS, DUNE_RADIUS, r)
  const dunes = fbm(x * 0.045, z * 0.045, 4) * 0.9 + Math.abs(fbm(x * 0.11, z * 0.11, 2)) * 0.25
  const rim = smoothstep(DUNE_RADIUS, TERRAIN_RADIUS * 0.6, r) ** 1.4 * (16 + 10 * fbm(x * 0.008, z * 0.008, 3))
  return near * (dunes * (0.6 + r * 0.02)) + rim
}

export interface PropPlacement { x: number; y: number; z: number; yaw: number; tilt: number; scale: number; variant: number }

export interface ScatterSpec {
  count: number
  /** Annulus the props occupy, in metres from the arena centre. */
  minRadius: number
  maxRadius: number
  scale: [number, number]
  variants: number
  /** Minimum centre spacing relative to scale, so large rocks do not interpenetrate. */
  spacing: number
  /** How strongly placement favours the inner edge (0 uniform by area, 1 uniform by radius). */
  inward: number
}

/**
 * Deterministic rejection-sampled placements. Every prop stays outside the
 * arena's footprint (minRadius is clamped past the drivable disc plus a
 * rover body) and sits slightly sunk into the terrain.
 */
export function scatter(spec: ScatterSpec, seed: number): PropPlacement[] {
  const rng = createRng(seed)
  const minRadius = Math.max(spec.minRadius, ARENA.radius + 0.25)
  const out: PropPlacement[] = []
  for (let attempt = 0; attempt < spec.count * 30 && out.length < spec.count; attempt++) {
    const u = rng()
    const area = Math.sqrt(minRadius ** 2 + u * (spec.maxRadius ** 2 - minRadius ** 2))
    const linear = minRadius + u * (spec.maxRadius - minRadius)
    const r = area + (linear - area) * spec.inward
    const a = rng() * Math.PI * 2
    const x = r * Math.cos(a), z = r * Math.sin(a)
    const scale = spec.scale[0] + rng() * (spec.scale[1] - spec.scale[0])
    if (out.some(p => Math.hypot(p.x - x, p.z - z) < spec.spacing * (p.scale + scale) / 2)) continue
    out.push({ x, z, y: terrainHeight(x, z) - 0.03 * scale, yaw: rng() * Math.PI * 2, tilt: (rng() - 0.5) * 0.25, scale, variant: Math.floor(rng() * spec.variants) })
  }
  return out
}
