import { ARENA } from '../../config/fleet.ts'
import type { Beacon, RoverActivity } from '../../core/fleet.ts'
import { LINK_CANDIDATE, LINK_KEPT } from '../../core/fleet.ts'
import type { FleetView, PoseSample } from '../../core/fleetView.ts'
import { pairIndex, widestPathTree } from '../../core/graph.ts'
import type { Link } from '../../core/graph.ts'
import { wrapAngle } from '../../core/math.ts'
import type { SidePlan } from './compare.ts'
import { positionAt } from './scenarios.ts'
import type { Scenario } from './scenarios.ts'

/** Seconds a task column stays "in progress" before its outcome shows. */
const OUTCOME_DELAY = 0.9
const BEACON_LINGER = 3.5

/**
 * Scripted motion shared by both halves of the split screen: rovers glide
 * between the scenario's recorded positions, so the planner on both sides
 * sees exactly the positions the verification run used.
 */
export class ScriptedMotion {
  /** Fractional cycle being rendered. */
  t = 0
  /** The cycle the motion is heading for. */
  target = 0
  elapsed = 0
  readonly yaw: Float64Array
  readonly travel: Float64Array
  readonly scenario: Scenario
  private readonly x: Float64Array
  private readonly z: Float64Array

  constructor(scenario: Scenario) {
    this.scenario = scenario
    const n = scenario.params.roverCount
    this.yaw = new Float64Array(n); this.travel = new Float64Array(n)
    this.x = new Float64Array(n); this.z = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      const [x, z] = positionAt(scenario, i, 0)
      this.x[i] = x; this.z[i] = z
      this.yaw[i] = Math.atan2(-z, -x)
    }
  }

  get count(): number { return this.x.length }

  /** Moves toward the target cycle at `rate` cycles per second, updating headings and wheel travel. */
  advance(dt: number, rate: number): void {
    if (!Number.isFinite(dt) || dt <= 0) return
    this.elapsed += dt
    this.t = Math.min(this.target, this.t + dt * rate)
    for (let i = 0; i < this.count; i++) {
      const [x, z] = positionAt(this.scenario, i, this.t)
      const dx = x - this.x[i], dz = z - this.z[i]
      const step = Math.hypot(dx, dz)
      if (step > 1e-5) {
        const heading = Math.atan2(dz, dx)
        // short hops can point backwards; a rover reverses rather than spinning on the spot
        const forward = Math.abs(wrapAngle(heading - this.yaw[i])) < Math.PI / 2
        const want = forward ? heading : wrapAngle(heading + Math.PI)
        this.yaw[i] = wrapAngle(this.yaw[i] + wrapAngle(want - this.yaw[i]) * Math.min(1, dt * 6))
        this.travel[i] += forward ? step : -step
      }
      this.x[i] = x; this.z[i] = z
    }
  }

  pose(i: number, out: PoseSample): PoseSample {
    out.x = this.x[i]; out.z = this.z[i]; out.yaw = this.yaw[i]
    out.left = out.right = this.travel[i] / ARENA.wheelRadius
    return out
  }
}

/** One half of the split screen: shared motion, this side's trust, links, tasks and highlight marks. */
export class ScriptedSide implements FleetView {
  readonly count: number
  readonly trust: Float32Array
  readonly activity: RoverActivity[]
  readonly planState: Uint8Array
  readonly planQuality: Float32Array
  readonly faulted: Uint8Array
  readonly roverMarks: Uint8Array
  readonly linkMarks: Uint8Array
  parent: Int32Array
  beacons: Beacon[] = []
  planVersion = 0
  readonly radioRange: number
  private readonly motion: ScriptedMotion

  // explicit fields rather than parameter properties: Node's type stripping (the unit tests) rejects those
  constructor(motion: ScriptedMotion) {
    this.motion = motion
    const n = motion.count
    this.count = n
    this.trust = new Float32Array(n).fill(0.5)
    this.activity = Array<RoverActivity>(n).fill('driving')
    this.planState = new Uint8Array(n * n)
    this.planQuality = new Float32Array(n * n)
    this.faulted = new Uint8Array(n)
    this.roverMarks = new Uint8Array(n)
    this.linkMarks = new Uint8Array(n * n)
    this.parent = new Int32Array(n).fill(-1)
    this.radioRange = motion.scenario.params.communicationRange
  }

  get elapsed(): number { return this.motion.elapsed }

  renderPose(i: number, out: PoseSample): PoseSample { return this.motion.pose(i, out) }

  /** Resolves task columns once their outcome is due and drops old ones; the shared motion advances elsewhere. */
  advance(): void {
    const now = this.motion.elapsed
    for (const b of this.beacons) {
      if (b.state === 'enroute' && now - b.issuedAt >= OUTCOME_DELAY) { b.state = 'resolved'; b.resolvedAt = now }
    }
    if (this.beacons.some(b => b.state === 'resolved' && now - b.resolvedAt > BEACON_LINGER)) {
      this.beacons = this.beacons.filter(b => b.state !== 'resolved' || now - b.resolvedAt <= BEACON_LINGER)
    }
  }

  /** Applies this side's plan for the cycle; tasks run where their rover stands (the simulator's tasks have no site). */
  apply(plan: SidePlan, faultedRovers: readonly (number | null)[]): void {
    const n = this.count
    plan.trust.forEach((t, i) => { if (i < n && Number.isFinite(t)) this.trust[i] = t })
    this.planState.fill(0); this.planQuality.fill(0); this.linkMarks.fill(0); this.roverMarks.fill(0); this.faulted.fill(0)
    const links: Link[] = []
    for (const e of plan.edges) {
      const a = Math.min(e.from, e.to), b = Math.max(e.from, e.to)
      if (a < 0 || b >= n || a === b) continue
      const k = pairIndex(a, b, n)
      this.planState[k] = e.active ? LINK_KEPT : LINK_CANDIDATE
      this.planQuality[k] = e.quality
      if (e.active) links.push({ a, b, quality: e.quality })
    }
    this.parent = widestPathTree(n, 0, links).parent
    for (const [a, b, mark] of plan.linkMarks) if (b < n) this.linkMarks[pairIndex(a, b, n)] = mark
    plan.roverMarks.forEach((mark, i) => { if (i < n) this.roverMarks[i] = mark })
    for (const rover of faultedRovers) if (rover !== null && rover >= 0 && rover < n) this.faulted[rover] = 1
    const sample: PoseSample = { x: 0, z: 0, yaw: 0, left: 0, right: 0 }
    for (const t of plan.tasks) {
      if (t.rover < 0 || t.rover >= n) continue
      const p = this.motion.pose(t.rover, sample)
      this.beacons.push({
        id: t.id, rover: t.rover, x: p.x, z: p.z, success: t.success, executed: t.executed, delivered: t.delivered,
        explored: t.explored, state: 'enroute', issuedAt: this.motion.elapsed, resolvedAt: 0,
      })
    }
    this.planVersion++
  }

  /** Back to the scenario's starting state (a replay). */
  reset(): void {
    this.trust.fill(0.5); this.planState.fill(0); this.planQuality.fill(0); this.faulted.fill(0)
    this.roverMarks.fill(0); this.linkMarks.fill(0); this.parent.fill(-1); this.beacons = []
    this.planVersion++
  }
}
