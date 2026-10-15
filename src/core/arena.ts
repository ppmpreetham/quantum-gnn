import type { ArenaSpec } from '../config/fleet.ts'
import { TAU, clamp, createRng, hashString } from './math.ts'

/** Centre-to-centre spacing that keeps two circular footprints apart with a small margin. */
export const spawnSpacing = (arena: ArenaSpec) => 2 * arena.bodyRadius + 0.35

/** Largest radius a rover centre may reach while its whole footprint stays on the arena. */
export const drivableRadius = (arena: ArenaSpec) => arena.radius - arena.bodyRadius

/**
 * Radius of the region the fleet patrols. It grows with √n so density stays
 * roughly constant, is sized against radio range so the mesh stays mostly
 * connected, and never exceeds the drivable disc.
 */
export function patrolRadius(count: number, range: number, arena: ArenaSpec): number {
  const wanted = 0.55 * Math.min(range, 5) * Math.sqrt(Math.max(1, count) / 2.2)
  return clamp(wanted, 2 * arena.bodyRadius, drivableRadius(arena) - 0.25)
}

/** Hexagonal lattice points inside a disc, nearest the centre first. */
function hexLattice(radius: number, spacing: number): [number, number][] {
  const points: [number, number][] = []
  const rows = Math.ceil(radius / (spacing * Math.sqrt(3) / 2))
  for (let r = -rows; r <= rows; r++) {
    const z = r * spacing * Math.sqrt(3) / 2
    const shift = (r & 1) * spacing / 2
    for (let x = -radius - spacing; x <= radius + spacing; x += spacing) {
      if (Math.hypot(x + shift, z) <= radius) points.push([x + shift, z])
    }
  }
  return points.sort((p, q) => Math.hypot(...p) - Math.hypot(...q))
}

/** Upper bound on rovers that fit, from the hexagonal packing of the drivable disc. */
export const fleetCapacity = (arena: ArenaSpec) => hexLattice(drivableRadius(arena), spawnSpacing(arena)).length

/**
 * Deterministic, non-overlapping spawn points. Dart throwing inside the
 * patrol disc first; if that fails (dense fleets), fall back to a hex
 * lattice, which is guaranteed to hold fleetCapacity rovers. Rover 0, the
 * planner's root, takes the point nearest the centre.
 */
export function spawnLayout(count: number, seed: number, range: number, arena: ArenaSpec) {
  const rng = createRng(seed * 7919 + count)
  const spacing = spawnSpacing(arena)
  const limit = patrolRadius(count, range, arena)
  let points: [number, number][] = []
  for (let attempt = 0; attempt < count * 80 && points.length < count; attempt++) {
    const r = limit * Math.sqrt(rng())
    const a = rng() * TAU
    const p: [number, number] = [r * Math.cos(a), r * Math.sin(a)]
    if (points.every(q => Math.hypot(p[0] - q[0], p[1] - q[1]) >= spacing)) points.push(p)
  }
  if (points.length < count) {
    const lattice = hexLattice(drivableRadius(arena), spacing)
    if (lattice.length < count) throw new RangeError(`Arena holds at most ${lattice.length} rovers`)
    points = lattice.slice(0, count)
  }
  points.sort((p, q) => Math.hypot(...p) - Math.hypot(...q))
  return points.map(([x, z]) => ({ x, z, yaw: Math.atan2(-z, -x) + (rng() - 0.5) }))
}

/** A random point in the disc of the given radius, uniform by area. */
export function pointInDisc(rng: () => number, radius: number) {
  const r = radius * Math.sqrt(rng())
  const a = rng() * TAU
  return { x: r * Math.cos(a), z: r * Math.sin(a) }
}

/** Stable map position for a planner task id inside the patrol region. */
export function taskSite(taskId: string, seed: number, radius: number) {
  return pointInDisc(createRng(hashString(taskId) ^ seed), radius * 0.92)
}
