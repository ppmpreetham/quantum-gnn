import { test } from 'node:test'
import assert from 'node:assert/strict'
import { ARENA, DEFAULT_PARAMS, PRESETS, advisories, sanitizeParams, snapParam } from '../src/config/fleet.ts'
import { drivableRadius, fleetCapacity, spawnLayout, spawnSpacing } from '../src/core/arena.ts'
import { FleetEngine, STEP } from '../src/core/fleet.ts'
import { pathToRoot, reachableCount, widestPathTree } from '../src/core/graph.ts'
import { integratePose, saturateTwist, twistFromWheels, wheelSpeeds } from '../src/core/kinematics.ts'
import { createRng, lerpAngle, wrapAngle } from '../src/core/math.ts'
import { bootstrapMeanCI, rate } from '../src/core/stats.ts'

const close = (a: number, b: number, eps = 1e-9) => assert.ok(Math.abs(a - b) <= eps, `${a} != ${b}`)

test('wrapAngle maps into (-π, π] and keeps the angle', () => {
  for (const a of [0, Math.PI, -Math.PI, 3 * Math.PI, -3 * Math.PI, 7.5, -100.25, 1e6]) {
    const w = wrapAngle(a)
    assert.ok(w > -Math.PI && w <= Math.PI, `${a} -> ${w}`)
    close(Math.cos(w), Math.cos(a), 1e-6)
    close(Math.sin(w), Math.sin(a), 1e-6)
  }
})

test('lerpAngle takes the short arc across ±π', () => {
  const from = Math.PI - 0.1, to = -Math.PI + 0.1
  const mid = lerpAngle(from, to, 0.5)
  close(Math.abs(mid), Math.PI, 1e-9)
})

test('integratePose matches the closed-form circular arc', () => {
  const v = 0.8, omega = 1.3, dt = 0.4
  const end = integratePose({ x: 1, z: -2, yaw: 0.3 }, v, omega, dt)
  const R = v / omega
  // centre of curvature sits to the right (+90° from heading) for omega > 0
  const cx = 1 - R * Math.sin(0.3), cz = -2 + R * Math.cos(0.3)
  close(Math.hypot(end.x - cx, end.z - cz), R, 1e-9)
  close(end.yaw, wrapAngle(0.3 + omega * dt))
})

test('integratePose is exact for straight lines and has no singularity at omega = 0', () => {
  const end = integratePose({ x: 0, z: 0, yaw: Math.PI / 4 }, 2, 0, 0.5)
  close(end.x, Math.SQRT1_2, 1e-12)
  close(end.z, Math.SQRT1_2, 1e-12)
  const tiny = integratePose({ x: 0, z: 0, yaw: 0 }, 1, 1e-12, 1)
  close(tiny.x, 1, 1e-9)
})

test('wheel speeds: a right turn (omega > 0) drives the left wheel faster', () => {
  const { left, right } = wheelSpeeds(0.5, 1, ARENA.track)
  assert.ok(left > right)
  const back = twistFromWheels(left, right, ARENA.track)
  close(back.v, 0.5)
  close(back.omega, 1)
})

test('wheel contact points travel exactly their rim distance on an arc', () => {
  const v = 0.6, omega = 0.9, dt = 0.01, steps = 200
  let pose = { x: 0, z: 0, yaw: 0 }
  let leftPath = 0, rightPath = 0
  const wheel = (p: typeof pose, side: number) => ({ x: p.x - side * Math.sin(p.yaw) * ARENA.track / 2, z: p.z + side * Math.cos(p.yaw) * ARENA.track / 2 })
  for (let k = 0; k < steps; k++) {
    const next = integratePose(pose, v, omega, dt)
    const l0 = wheel(pose, -1), l1 = wheel(next, -1), r0 = wheel(pose, 1), r1 = wheel(next, 1)
    leftPath += Math.hypot(l1.x - l0.x, l1.z - l0.z)
    rightPath += Math.hypot(r1.x - r0.x, r1.z - r0.z)
    pose = next
  }
  const { left, right } = wheelSpeeds(v, omega, ARENA.track)
  close(leftPath, left * dt * steps, 1e-4)
  close(rightPath, right * dt * steps, 1e-4)
})

test('saturateTwist respects wheel limits and keeps curvature', () => {
  const out = saturateTwist(2, 1, ARENA)
  const { left, right } = wheelSpeeds(out.v, out.omega, ARENA.track)
  assert.ok(Math.max(Math.abs(left), Math.abs(right)) <= ARENA.maxWheelSpeed + 1e-12)
  close(out.omega / out.v, 0.5, 1e-12)
  const spin = saturateTwist(0, 50, ARENA)
  close(spin.v, 0)
  assert.ok(Math.abs(spin.omega) <= ARENA.maxYawRate)
})

test('spawn layout is collision free, inside the arena, and deterministic', () => {
  for (let n = 2; n <= 16; n++) for (const seed of [0, 17, 99999]) {
    const points = spawnLayout(n, seed, 4.6, ARENA)
    assert.equal(points.length, n)
    for (const p of points) assert.ok(Math.hypot(p.x, p.z) <= drivableRadius(ARENA) + 1e-9)
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      assert.ok(Math.hypot(points[i].x - points[j].x, points[i].z - points[j].z) >= spawnSpacing(ARENA) - 1e-9)
    }
    assert.deepEqual(spawnLayout(n, seed, 4.6, ARENA), points)
  }
  assert.ok(fleetCapacity(ARENA) >= 16)
})

test('fleet never overlaps footprints or leaves the arena, at any speed', () => {
  for (const [count, seed, cruise] of [[16, 3, 1], [12, 11, 1], [8, 17, 0.5], [6, 5, 1]] as const) {
    const engine = new FleetEngine({ count, seed, range: 3.6, cruise, timeScale: 3, tether: true }, ARENA)
    const contact = 2 * ARENA.bodyRadius
    for (let frame = 0; frame < 1800; frame++) {
      engine.advance(1 / 60)
      for (let i = 0; i < count; i++) {
        assert.ok(Number.isFinite(engine.x[i]) && Number.isFinite(engine.z[i]) && Number.isFinite(engine.yaw[i]))
        assert.ok(Math.hypot(engine.x[i], engine.z[i]) <= drivableRadius(ARENA) + 1e-9)
        for (let j = i + 1; j < count; j++) {
          assert.ok(Math.hypot(engine.x[i] - engine.x[j], engine.z[i] - engine.z[j]) >= contact - 1e-9, `overlap ${i}/${j} at frame ${frame}`)
        }
      }
    }
  }
})

test('fleet keeps moving: no permanent deadlock in a dense swarm', () => {
  const engine = new FleetEngine({ count: 16, seed: 3, range: 3.6, cruise: 0.6, timeScale: 1, tether: true }, ARENA)
  const start = Array.from({ length: 16 }, (_, i) => [engine.x[i], engine.z[i]])
  for (let frame = 0; frame < 60 * 40; frame++) engine.advance(1 / 60)
  const moved = start.filter(([x, z], i) => Math.hypot(engine.x[i] - x, engine.z[i] - z) > 0.5).length
  assert.ok(moved >= 14, `only ${moved} of 16 rovers moved`)
})

test('advance ignores bad frame deltas and caps catch-up work', () => {
  const engine = new FleetEngine({ count: 4, seed: 1, range: 4, cruise: 0.5, timeScale: 1, tether: false }, ARENA)
  const before = engine.elapsed
  for (const bad of [Number.NaN, -1, 0, Number.POSITIVE_INFINITY]) engine.advance(bad)
  assert.equal(engine.elapsed, before)
  engine.advance(60)
  assert.ok(engine.elapsed <= 0.1 + STEP, 'a stalled tab must not replay a minute of motion')
})

test('applyPlan rejects malformed planner data without corrupting state', () => {
  const engine = new FleetEngine({ count: 3, seed: 1, range: 4, cruise: 0.5, timeScale: 1, tether: false }, ARENA)
  engine.applyPlan({
    cycle: 4,
    rovers: [{ index: 7, trust: 0.9, battery: 1, compute: 1, capability: 1, reliability: 1, pathFactor: 1, task: null },
      { index: 1, trust: Number.NaN, battery: 2, compute: -1, capability: 1, reliability: 0.4, pathFactor: 0.5, task: 'task-4-0' }],
    edges: [{ from: 0, to: 1, active: true, quality: 0.8, utility: 0.5 }, { from: 2, to: 2, active: true, quality: 1, utility: 1 },
      { from: 1, to: 9, active: true, quality: 1, utility: 1 }],
    tasks: [{ id: 'task-4-0', rover: 1, executed: true, delivered: true, success: true, explored: false }, { id: 'x', rover: 42, executed: true, delivered: true, success: true, explored: false }],
  })
  assert.equal(engine.trust[1], 0.5)
  assert.equal(engine.battery[1], 1)
  assert.equal(engine.compute[1], 0)
  assert.equal(engine.links.length, 1)
  assert.equal(engine.parent[1], 0)
  assert.equal(engine.beacons.length, 1)
})

test('widest path tree picks the maximum-bottleneck route', () => {
  const links = [{ a: 0, b: 1, quality: 0.9 }, { a: 1, b: 3, quality: 0.2 }, { a: 0, b: 2, quality: 0.6 }, { a: 2, b: 3, quality: 0.5 }]
  const { parent, bottleneck } = widestPathTree(5, 0, links)
  assert.deepEqual(pathToRoot(parent, 3, 0), [3, 2, 0])
  close(bottleneck[3], 0.5, 1e-6)
  assert.equal(parent[4], -1)
  assert.deepEqual(pathToRoot(parent, 4, 0), [])
  assert.equal(reachableCount(5, 0, links, () => true), 4)
  assert.equal(reachableCount(5, 0, links, l => l.quality > 0.55), 3)
})

test('params snap to their grid and fall back on garbage', () => {
  assert.equal(snapParam('roverCount', 3.6), 4)
  assert.equal(snapParam('roverCount', 1e9), 16)
  assert.equal(snapParam('coolingRate', 0.99049), 0.99)
  assert.equal(snapParam('speed', Number.NaN), DEFAULT_PARAMS.speed)
  const params = sanitizeParams({ roverCount: '12' as unknown as number, budget: -5 })
  assert.equal(params.roverCount, DEFAULT_PARAMS.roverCount)
  assert.equal(params.budget, 0)
})

test('presets are valid and on-grid', () => {
  for (const preset of PRESETS) assert.deepEqual(sanitizeParams(preset.params), preset.params, preset.id)
})

test('graph-theory advisories catch impossible constraints', () => {
  const impossible = advisories({ ...DEFAULT_PARAMS, roverCount: 5, maxDegree: 1 })
  assert.ok(impossible.some(a => a.level === 'warn'))
  const broke = advisories({ ...DEFAULT_PARAMS, roverCount: 10, budget: 5 })
  assert.ok(broke.some(a => a.text.includes('budget under 9')))
  assert.equal(advisories(DEFAULT_PARAMS).length, 0)
})

test('bootstrap CI brackets the mean and is deterministic', () => {
  const rng = createRng(3)
  const diffs = Array.from({ length: 60 }, () => (rng() < 0.6 ? 1 : 0) - (rng() < 0.4 ? 1 : 0))
  const ci = bootstrapMeanCI(diffs)!
  const m = diffs.reduce((a, b) => a + b, 0) / diffs.length
  assert.ok(ci[0] <= m && m <= ci[1])
  assert.deepEqual(bootstrapMeanCI(diffs), ci)
  assert.equal(bootstrapMeanCI([1]), null)
  assert.ok(Number.isNaN(rate(0, 0)))
})
