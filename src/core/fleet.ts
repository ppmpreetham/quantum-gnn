import type { ArenaSpec } from '../config/fleet.ts'
import { drivableRadius, patrolRadius, pointInDisc, spawnLayout, taskSite } from './arena.ts'
import type { FleetView } from './fleetView.ts'
import { pairIndex, reachableCount, widestPathTree } from './graph.ts'
import type { Link } from './graph.ts'
import { approach, integratePose, saturateTwist } from './kinematics.ts'
import type { DriveLimits, Pose, Twist } from './kinematics.ts'
import { clamp, createRng, lerp, lerpAngle, wrapAngle } from './math.ts'

export const STEP = 1 / 120
const MAX_FRAME = 0.1
const MAX_BACKLOG = 40 * STEP
const ARRIVE = 0.3
const SLOW_RADIUS = 1.1
const K_YAW = 3.2
const WORK_TIME = 1.4
const BEACON_TIMEOUT = 14
const BEACON_LINGER = 4
const MAX_BEACONS = 32

export type RoverActivity = 'driving' | 'turning' | 'yielding' | 'detour' | 'working' | 'parked'
export const LINK_NONE = 0
export const LINK_CANDIDATE = 1
export const LINK_KEPT = 2

export interface PlannerRover {
  index: number; trust: number; battery: number; compute: number; capability: number
  reliability: number; pathFactor: number; task: string | null
}
export interface PlannerEdge { from: number; to: number; active: boolean; quality: number; utility: number }
/** site is set for tasks placed on the map by the user; planner-generated tasks get a stable hashed site. */
export interface PlannerTask {
  id: string; rover: number; executed: boolean; delivered: boolean; success: boolean; explored: boolean
  site?: [number, number]
}
export interface PlanUpdate {
  cycle: number; rovers: PlannerRover[]; edges: PlannerEdge[]; tasks: PlannerTask[]
  faults?: { rover: number | null }[]
}

export interface Beacon {
  id: string; rover: number; x: number; z: number
  success: boolean; executed: boolean; delivered: boolean; explored: boolean
  state: 'enroute' | 'resolved'; issuedAt: number; resolvedAt: number
}
export interface EngineOptions { count: number; seed: number; range: number; cruise: number; timeScale: number; tether: boolean }

const finite = (value: unknown, fallback: number) => (typeof value === 'number' && Number.isFinite(value) ? value : fallback)

const isIndex = (i: unknown, n: number): i is number => Number.isInteger(i) && (i as number) >= 0 && (i as number) < n

/** A user-placed site from the planner, if it is a finite point inside the arena. */
function userSite(site: unknown, arena: ArenaSpec): { x: number; z: number } | null {
  if (!Array.isArray(site) || site.length !== 2) return null
  const [x, z] = site
  if (typeof x !== 'number' || typeof z !== 'number' || !Number.isFinite(x) || !Number.isFinite(z)) return null
  return Math.hypot(x, z) <= arena.radius ? { x, z } : null
}

/**
 * Fixed-step kinematic fleet. Each tick commits rovers one at a time and only
 * accepts a translation that keeps every pair at least two body radii apart
 * (or not closer than before). For a pair i < j, i is checked against j's old
 * position and j against i's new one, so after a tick
 * d_ij >= min(2R, previous d_ij): footprints that start apart never overlap.
 * A rejected rover still rotates in place, which cannot move its centre.
 */
export class FleetEngine implements FleetView {
  readonly count: number
  readonly seed: number
  readonly x: Float64Array; readonly z: Float64Array; readonly yaw: Float64Array
  readonly v: Float64Array; readonly omega: Float64Array
  readonly prevX: Float64Array; readonly prevZ: Float64Array; readonly prevYaw: Float64Array
  readonly spinLeft: Float64Array; readonly spinRight: Float64Array
  readonly prevSpinLeft: Float64Array; readonly prevSpinRight: Float64Array
  readonly goalX: Float64Array; readonly goalZ: Float64Array
  readonly activity: RoverActivity[]
  readonly trust: Float32Array; readonly battery: Float32Array; readonly compute: Float32Array
  readonly reliability: Float32Array; readonly pathFactor: Float32Array
  readonly task: (string | null)[]
  readonly planState: Uint8Array
  readonly planQuality: Float32Array
  readonly faulted: Uint8Array
  parent: Int32Array
  links: Link[] = []
  beacons: Beacon[] = []
  cycle = 0
  elapsed = 0
  planVersion = 0
  private accumulator = 0
  private cruise: number
  private timeScale: number
  private range: number
  private tether: boolean
  private patrol: number
  private readonly arena: ArenaSpec
  private readonly limits: DriveLimits
  private readonly rngs: (() => number)[]
  private readonly goalTask: (string | null)[]
  private readonly stuck: Float64Array; private readonly detour: Float64Array
  private readonly detourX: Float64Array; private readonly detourZ: Float64Array
  private readonly detours: Uint8Array; private readonly turnSign: Int8Array
  private readonly progressRef: Float64Array; private readonly progressClock: Float64Array
  private readonly working: Float64Array
  private readonly pose: Pose = { x: 0, z: 0, yaw: 0 }
  private readonly next: Pose = { x: 0, z: 0, yaw: 0 }
  private readonly twist: Twist = { v: 0, omega: 0 }

  constructor(options: EngineOptions, arena: ArenaSpec) {
    const n = options.count
    this.count = n
    this.seed = options.seed
    this.arena = arena
    this.limits = arena
    this.cruise = options.cruise
    this.timeScale = options.timeScale
    this.range = options.range
    this.tether = options.tether
    this.patrol = patrolRadius(n, options.range, arena)
    const f64 = () => new Float64Array(n)
    this.x = f64(); this.z = f64(); this.yaw = f64(); this.v = f64(); this.omega = f64()
    this.prevX = f64(); this.prevZ = f64(); this.prevYaw = f64()
    this.spinLeft = f64(); this.spinRight = f64(); this.prevSpinLeft = f64(); this.prevSpinRight = f64()
    this.goalX = f64(); this.goalZ = f64()
    this.stuck = f64(); this.detour = f64(); this.detourX = f64(); this.detourZ = f64()
    this.progressRef = f64(); this.progressClock = f64(); this.working = f64()
    this.detours = new Uint8Array(n)
    this.turnSign = new Int8Array(n)
    this.trust = new Float32Array(n).fill(0.5)
    this.battery = new Float32Array(n).fill(1)
    this.compute = new Float32Array(n).fill(1)
    this.reliability = new Float32Array(n).fill(0.5)
    this.pathFactor = new Float32Array(n)
    this.task = Array<string | null>(n).fill(null)
    this.goalTask = Array<string | null>(n).fill(null)
    this.activity = Array<RoverActivity>(n).fill('driving')
    this.planState = new Uint8Array(n * n)
    this.planQuality = new Float32Array(n * n)
    this.faulted = new Uint8Array(n)
    this.parent = new Int32Array(n).fill(-1)
    this.rngs = Array.from({ length: n }, (_, i) => createRng(options.seed * 104729 + i * 7907 + 1))
    spawnLayout(n, options.seed, options.range, arena).forEach((p, i) => {
      this.x[i] = this.prevX[i] = p.x
      this.z[i] = this.prevZ[i] = p.z
      this.yaw[i] = this.prevYaw[i] = wrapAngle(p.yaw)
      this.pickPatrolGoal(i)
    })
  }

  configure(options: Pick<EngineOptions, 'cruise' | 'timeScale' | 'range' | 'tether'>): void {
    this.cruise = Math.max(0, options.cruise)
    this.timeScale = Math.max(0, options.timeScale)
    this.tether = options.tether
    if (options.range !== this.range) {
      this.range = options.range
      this.patrol = patrolRadius(this.count, options.range, this.arena)
    }
  }

  /** Interpolation weight between the last two fixed steps. */
  get alpha(): number { return this.accumulator / STEP }

  advance(realDelta: number): void {
    if (!Number.isFinite(realDelta) || realDelta <= 0) return
    this.accumulator = Math.min(this.accumulator + Math.min(realDelta, MAX_FRAME) * this.timeScale, MAX_BACKLOG)
    while (this.accumulator >= STEP) {
      this.prevX.set(this.x); this.prevZ.set(this.z); this.prevYaw.set(this.yaw)
      this.prevSpinLeft.set(this.spinLeft); this.prevSpinRight.set(this.spinRight)
      this.tick(STEP)
      this.accumulator -= STEP
    }
    const expired = (b: Beacon) => b.state === 'resolved' && this.elapsed - b.resolvedAt > BEACON_LINGER
    if (this.beacons.some(expired)) this.beacons = this.beacons.filter(b => !expired(b))
  }

  /** Render pose blended between the previous and current fixed steps. */
  renderPose(i: number, out: { x: number; z: number; yaw: number; left: number; right: number }) {
    const t = this.alpha
    out.x = lerp(this.prevX[i], this.x[i], t)
    out.z = lerp(this.prevZ[i], this.z[i], t)
    out.yaw = lerpAngle(this.prevYaw[i], this.yaw[i], t)
    out.left = lerp(this.prevSpinLeft[i], this.spinLeft[i], t)
    out.right = lerp(this.prevSpinRight[i], this.spinRight[i], t)
    return out
  }

  /** Current positions for the planner, inside the arena by construction. */
  positions(): [number, number][] {
    const limit = this.arena.radius
    return Array.from({ length: this.count }, (_, i) => {
      const r = Math.hypot(this.x[i], this.z[i])
      const s = r > limit ? limit / r : 1
      return [Math.round(this.x[i] * s * 1e4) / 1e4, Math.round(this.z[i] * s * 1e4) / 1e4]
    })
  }

  applyPlan(plan: PlanUpdate): void {
    const n = this.count
    this.cycle = finite(plan.cycle, this.cycle)
    for (const r of plan.rovers ?? []) {
      const i = r.index
      if (!isIndex(i, n)) continue
      this.trust[i] = clamp(finite(r.trust, this.trust[i]), 0, 1)
      this.battery[i] = clamp(finite(r.battery, this.battery[i]), 0, 1)
      this.compute[i] = clamp(finite(r.compute, this.compute[i]), 0, 1)
      this.reliability[i] = clamp(finite(r.reliability, this.reliability[i]), 0, 1)
      this.pathFactor[i] = clamp(finite(r.pathFactor, 0), 0, 1)
      this.task[i] = typeof r.task === 'string' ? r.task : null
    }
    this.planState.fill(LINK_NONE)
    this.planQuality.fill(0)
    const links: Link[] = []
    for (const e of plan.edges ?? []) {
      const a = Math.min(e.from, e.to), b = Math.max(e.from, e.to)
      if (!Number.isInteger(a) || !Number.isInteger(b) || a < 0 || b >= n || a === b) continue
      const k = pairIndex(a, b, n)
      const quality = clamp(finite(e.quality, 0), 0, 1)
      this.planState[k] = e.active ? LINK_KEPT : LINK_CANDIDATE
      this.planQuality[k] = quality
      if (e.active) links.push({ a, b, quality })
    }
    this.links = links
    this.parent = widestPathTree(n, 0, links).parent
    this.faulted.fill(0)
    for (const { rover } of plan.faults ?? []) if (isIndex(rover, n)) this.faulted[rover] = 1
    for (const t of plan.tasks ?? []) this.dispatch(t)
    this.planVersion++
  }

  /** Kept links whose endpoints are still in radio range right now. */
  liveSummary() {
    const n = this.count
    let candidates = 0, kept = 0, breaking = 0
    for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) {
      const inRange = Math.hypot(this.x[a] - this.x[b], this.z[a] - this.z[b]) <= this.range
      const state = this.planState[a * n + b]
      if (inRange) candidates++
      if (state === LINK_KEPT) inRange ? kept++ : breaking++
    }
    const reachable = reachableCount(n, 0, this.links,
      l => Math.hypot(this.x[l.a] - this.x[l.b], this.z[l.a] - this.z[l.b]) <= this.range)
    return { candidates, kept, breaking, reachable }
  }

  inRange(a: number, b: number): boolean {
    return Math.hypot(this.x[a] - this.x[b], this.z[a] - this.z[b]) <= this.range
  }

  get radioRange(): number { return this.range }
  get patrolRadius(): number { return this.patrol }

  private dispatch(t: PlannerTask): void {
    const i = t.rover
    if (!isIndex(i, this.count) || typeof t.id !== 'string') return
    // a reassigned rover abandons its old errand; that beacon resolves now
    for (const b of this.beacons) if (b.rover === i && b.state === 'enroute') this.resolve(b)
    const site = userSite(t.site, this.arena) ?? taskSite(t.id, this.seed, this.patrol)
    this.beacons.push({
      id: t.id, rover: i, x: site.x, z: site.z, success: !!t.success, executed: !!t.executed,
      delivered: !!t.delivered, explored: !!t.explored, state: 'enroute', issuedAt: this.elapsed, resolvedAt: 0,
    })
    if (this.beacons.length > MAX_BEACONS) this.beacons.splice(0, this.beacons.length - MAX_BEACONS)
    this.setGoal(i, site.x, site.z, t.id)
    this.working[i] = 0
  }

  private setGoal(i: number, x: number, z: number, taskId: string | null): void {
    const limit = drivableRadius(this.arena) - 0.2
    const r = Math.hypot(x, z)
    const s = r > limit ? limit / r : 1
    this.goalX[i] = x * s
    this.goalZ[i] = z * s
    this.goalTask[i] = taskId
    this.detour[i] = 0
    this.detours[i] = 0
    this.progressRef[i] = Math.hypot(this.goalX[i] - this.x[i], this.goalZ[i] - this.z[i])
    this.progressClock[i] = 0
  }

  private pickPatrolGoal(i: number): void {
    const p = pointInDisc(this.rngs[i], this.patrol)
    this.setGoal(i, p.x, p.z, null)
  }

  private arrive(i: number): void {
    const id = this.goalTask[i]
    if (id) {
      const beacon = this.beacons.find(b => b.id === id && b.state === 'enroute')
      if (beacon) this.resolve(beacon)
      this.working[i] = WORK_TIME
    }
    this.pickPatrolGoal(i)
  }

  private resolve(beacon: Beacon): void {
    beacon.state = 'resolved'
    beacon.resolvedAt = this.elapsed
  }

  private startDetour(i: number): void {
    const side = this.detours[i] % 2 === 0 ? 1 : -1
    const heading = Math.atan2(this.goalZ[i] - this.z[i], this.goalX[i] - this.x[i]) + side * 1.1
    const reach = 1.4 + 0.3 * this.detours[i]
    let dx = this.x[i] + reach * Math.cos(heading)
    let dz = this.z[i] + reach * Math.sin(heading)
    const limit = drivableRadius(this.arena) - 0.5
    const r = Math.hypot(dx, dz)
    if (r > limit) { dx *= limit / r; dz *= limit / r }
    this.detourX[i] = dx
    this.detourZ[i] = dz
    this.detour[i] = 2.4
    this.stuck[i] = 0
    if (++this.detours[i] > 5) {
      // persistent blockage: the errand is unreachable for now, so give it up
      const id = this.goalTask[i]
      if (id) for (const b of this.beacons) if (b.id === id && b.state === 'enroute') this.resolve(b)
      this.pickPatrolGoal(i)
    }
  }

  private tick(dt: number): void {
    this.elapsed += dt
    for (const beacon of this.beacons) {
      if (beacon.state === 'enroute' && this.elapsed - beacon.issuedAt > BEACON_TIMEOUT) this.resolve(beacon)
    }
    for (let i = 0; i < this.count; i++) this.drive(i, dt)
  }

  private drive(i: number, dt: number): void {
    const { bodyRadius, track, wheelRadius } = this.arena
    const px = this.x[i], pz = this.z[i], yaw = this.yaw[i]
    const contact = 2 * bodyRadius
    if (this.detour[i] > 0) this.detour[i] -= dt
    let tx = this.detour[i] > 0 ? this.detourX[i] : this.goalX[i]
    let tz = this.detour[i] > 0 ? this.detourZ[i] : this.goalZ[i]
    let dist = Math.hypot(tx - px, tz - pz)
    const blockedNearGoal = this.stuck[i] > 0.5 && dist < contact + 0.4
    if (dist < ARRIVE || blockedNearGoal) {
      if (this.detour[i] > 0) this.detour[i] = 0
      else this.arrive(i)
      tx = this.goalX[i]; tz = this.goalZ[i]
      dist = Math.hypot(tx - px, tz - pz)
    }

    const hx = Math.cos(yaw), hz = Math.sin(yaw)
    let gx = dist > 1e-6 ? (tx - px) / dist : hx
    let gz = dist > 1e-6 ? (tz - pz) / dist : hz
    const sense = contact + 1.1
    let nearest = Infinity, nearestGap = Infinity, nx = 0, nz = 0
    for (let j = 0; j < this.count; j++) {
      if (j === i) continue
      const ox = px - this.x[j], oz = pz - this.z[j]
      const d = Math.hypot(ox, oz)
      if (d < nearest) { nearest = d; nx = this.x[j]; nz = this.z[j] }
      if (d >= sense || d < 1e-9) continue
      const ahead = -(ox * hx + oz * hz) / d
      if (ahead > 0) nearestGap = Math.min(nearestGap, d - contact)
      const w = ((sense - d) / (sense - contact)) ** 2
      gx += (ox / d) * w * 1.5
      gz += (oz / d) * w * 1.5
      // keep right: both rovers in a head-on meeting veer the same way and pass
      if (ahead > 0.2) { gx += -hz * w * ahead; gz += hx * w * ahead }
    }
    if (this.tether && this.count > 1 && nearest > 0.85 * this.range && this.goalTask[i] === null) {
      const pull = clamp((nearest - 0.85 * this.range) / (0.15 * this.range), 0, 1.5)
      gx += ((nx - px) / nearest) * pull
      gz += ((nz - pz) / nearest) * pull
    }
    const r = Math.hypot(px, pz)
    const soft = drivableRadius(this.arena) - 0.9
    if (r > soft) {
      const k = ((r - soft) / 0.9) ** 2 * 2
      gx -= (px / r) * k
      gz -= (pz / r) * k
    }

    const desired = Math.hypot(gx, gz) > 1e-6 ? Math.atan2(gz, gx) : yaw
    let error = wrapAngle(desired - yaw)
    // near ±π the sign of the error flips with tiny noise; hold the turn direction
    if (Math.abs(error) > 0.9 * Math.PI && this.turnSign[i] !== 0 && Math.sign(error) !== this.turnSign[i]) {
      error = this.turnSign[i] * Math.abs(error)
    }
    this.turnSign[i] = Math.abs(error) > 0.05 ? (Math.sign(error) as -1 | 1) : 0

    const parked = this.battery[i] < 0.05
    const facing = Math.max(0, Math.cos(error))
    let vTarget = this.cruise * facing * facing
    vTarget *= clamp(dist / SLOW_RADIUS, 0.3, 1) * clamp(nearestGap / 0.8, 0.15, 1)
    if (this.working[i] > 0) { this.working[i] -= dt; vTarget = 0 }
    if (parked) vTarget = 0
    const omegaTarget = parked || this.working[i] > 0 ? 0 : K_YAW * error
    this.v[i] = approach(this.v[i], vTarget, this.limits.maxAccel, this.limits.maxBrake, dt)
    this.omega[i] = approach(this.omega[i], omegaTarget, this.limits.maxYawAccel, this.limits.maxYawAccel, dt)
    const twist = saturateTwist(this.v[i], this.omega[i], this.limits, this.twist)
    this.v[i] = twist.v
    this.omega[i] = twist.omega

    this.pose.x = px; this.pose.z = pz; this.pose.yaw = yaw
    const next = integratePose(this.pose, twist.v, twist.omega, dt, this.next)
    const free = this.isFree(i, next.x, next.z)
    if (free) {
      this.x[i] = next.x
      this.z[i] = next.z
      this.stuck[i] = Math.max(0, this.stuck[i] - dt)
    } else {
      this.v[i] = 0
      this.stuck[i] += dt
    }
    this.yaw[i] = next.yaw
    const v = this.v[i]
    const half = (this.omega[i] * track) / 2
    this.spinLeft[i] += ((v + half) / wheelRadius) * dt
    this.spinRight[i] += ((v - half) / wheelRadius) * dt

    this.progressClock[i] += dt
    if (this.progressClock[i] >= 2) {
      const remaining = Math.hypot(this.goalX[i] - this.x[i], this.goalZ[i] - this.z[i])
      const idle = parked || this.working[i] > 0 || this.cruise < 0.05
      if (!idle && this.detour[i] <= 0 && this.progressRef[i] - remaining < 0.15 && remaining > ARRIVE) this.startDetour(i)
      this.progressRef[i] = remaining
      this.progressClock[i] = 0
    }
    if (this.stuck[i] > 1.2 && this.detour[i] <= 0) this.startDetour(i)

    this.activity[i] = parked ? 'parked'
      : this.working[i] > 0 ? 'working'
      : !free ? 'yielding'
      : this.detour[i] > 0 ? 'detour'
      : Math.abs(error) > 0.7 && Math.abs(v) < 0.15 ? 'turning'
      : 'driving'
  }

  private isFree(i: number, nx: number, nz: number): boolean {
    const limit = drivableRadius(this.arena)
    const r = Math.hypot(nx, nz)
    if (r > limit && r > Math.hypot(this.x[i], this.z[i])) return false
    const contact = 2 * this.arena.bodyRadius
    for (let j = 0; j < this.count; j++) {
      if (j === i) continue
      const after = Math.hypot(nx - this.x[j], nz - this.z[j])
      if (after < contact && after < Math.hypot(this.x[i] - this.x[j], this.z[i] - this.z[j])) return false
    }
    return true
  }
}
