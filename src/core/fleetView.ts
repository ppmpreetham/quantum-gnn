import type { Beacon, RoverActivity } from './fleet.ts'

/** Highlight marks a view may carry for rovers and links; renderers ignore views without them. */
export const MARK_NONE = 0
export const MARK_GOOD = 1
export const MARK_BAD = 2
export type Mark = typeof MARK_NONE | typeof MARK_GOOD | typeof MARK_BAD

export interface PoseSample { x: number; z: number; yaw: number; left: number; right: number }

/**
 * Everything the 3D renderers read about a fleet. The live FleetEngine
 * implements it, and so does the showcase's scripted fleet, so the same
 * scene components draw both without knowing which one they are given.
 */
export interface FleetView {
  readonly count: number
  readonly trust: Float32Array
  readonly activity: readonly RoverActivity[]
  /** Pairwise link state (LINK_NONE / LINK_CANDIDATE / LINK_KEPT), indexed a * count + b with a < b. */
  readonly planState: Uint8Array
  readonly planQuality: Float32Array
  /** Delivery-tree parent of each rover, -1 for the root and unreachable rovers. */
  readonly parent: Int32Array
  readonly beacons: readonly Beacon[]
  /** 1 while a rover is under an injected fault. */
  readonly faulted: Uint8Array
  readonly elapsed: number
  readonly planVersion: number
  readonly radioRange: number
  /** Optional highlight marks per rover and per link pair (same indexing as planState). */
  readonly roverMarks?: Uint8Array
  readonly linkMarks?: Uint8Array
  renderPose(i: number, out: PoseSample): PoseSample
  advance(realDelta: number): void
}
