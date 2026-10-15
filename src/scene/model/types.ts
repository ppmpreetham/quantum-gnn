import type * as THREE from 'three'

/** A static piece of the rover, already in model space (x forward, y up, z right). */
export interface BodyPart { geometry: THREE.BufferGeometry; material: THREE.Material | THREE.Material[] }

/**
 * A wheel spins about the lateral (z) axis through its pivot. side is -1 for
 * the left track (-z), +1 for the right, 0 for a centred caster. radius is
 * the rolling radius in model units, used to turn ground speed into spin.
 */
export interface WheelPart extends BodyPart { side: -1 | 0 | 1; pivot: THREE.Vector3; radius: number }

export interface RoverModel {
  name: string
  source: 'sdf' | 'glb'
  body: BodyPart[]
  wheels: WheelPart[]
  /** Points on the body that glow with the rover's status colour. */
  lights: THREE.BufferGeometry
  triangles: number
  dispose: () => void
}

export type ForwardAxis = '+x' | '-x' | '+z' | '-z' | 'auto'
