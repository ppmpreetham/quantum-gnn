import type * as THREE from 'three'

/** A static piece of the rover, already in model space (x forward, y up, z right). */
export interface BodyPart { geometry: THREE.BufferGeometry; material: THREE.Material | THREE.Material[] }

/**
 * A wheel spins about the lateral (z) axis through its pivot. side is -1 for
 * the left track (-z), +1 for the right, 0 for a centred caster. radius is
 * the rolling radius in model units, used to turn ground speed into spin.
 * yaw turns a shared wheel mesh to face the other side (0 or π).
 */
export interface WheelPart extends BodyPart { side: -1 | 0 | 1; pivot: THREE.Vector3; radius: number; yaw?: number }

/** One level of detail, used while the camera is closer than maxDistance (the last level has Infinity). */
export interface RoverLod { body: BodyPart[]; wheels: WheelPart[]; maxDistance: number }

export interface RoverModel {
  name: string
  source: 'sdf' | 'glb' | 'asset'
  /** Nearest first; a single entry means no LOD switching. */
  lods: RoverLod[]
  /** Points on the body that glow with the rover's status colour. */
  lights: THREE.BufferGeometry
  /** Triangles of the most detailed level. */
  triangles: number
  dispose: () => void
}

export type ForwardAxis = '+x' | '-x' | '+z' | '-z' | 'auto'

export const singleLod = (body: BodyPart[], wheels: WheelPart[]): RoverLod[] => [{ body, wheels, maxDistance: Infinity }]
