import { test } from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, readFileSync, statSync } from 'node:fs'
import { ARENA } from '../src/config/fleet.ts'
import { MARK_BAD, MARK_GOOD } from '../src/core/fleetView.ts'
import { LINK_CANDIDATE, LINK_KEPT } from '../src/core/fleet.ts'
import { DEFAULT_MODE, MODES, hashForMode, isModeId, modeFromHash } from '../src/modes/modes.ts'
import { equirectDirection, findSun, measureSun, rotateY, solarPosition, sunDirection } from '../src/modes/photoreal/environment.ts'
import type { PixelSampler, Vec3 } from '../src/modes/photoreal/environment.ts'
import { validateManifest } from '../src/modes/photoreal/manifest.ts'
import { FLAT_RADIUS, scatter, terrainHeight } from '../src/modes/photoreal/terrainField.ts'
import { compareCycle } from '../src/modes/showcase/compare.ts'
import { SCENARIOS, faultsAt, positionAt, traceVerdict } from '../src/modes/showcase/scenarios.ts'
import type { AssignmentSummary, SolverSummary } from '../src/modes/showcase/scenarios.ts'
import { ScriptedMotion, ScriptedSide } from '../src/modes/showcase/ScriptedFleet.ts'

const close = (a: number, b: number, eps = 1e-6) => assert.ok(Math.abs(a - b) <= eps, `${a} != ${b}`)
const length = ([x, y, z]: Vec3) => Math.hypot(x, y, z)

// ---------------------------------------------------------------- modes

test('mode registry: unique ids, console first and default', () => {
  assert.equal(new Set(MODES.map(m => m.id)).size, MODES.length)
  assert.equal(MODES[0].id, 'console')
  assert.equal(DEFAULT_MODE, 'console')
  assert.ok(MODES.every(m => m.label && m.summary && (m.mission === 'live' || m.mission === 'scripted')))
  assert.ok(isModeId('photoreal') && isModeId('showcase') && !isModeId('nope') && !isModeId(undefined))
})

test('mode hash: parses, falls back, and round-trips', () => {
  assert.equal(modeFromHash(''), 'console')
  assert.equal(modeFromHash('#mode=photoreal'), 'photoreal')
  assert.equal(modeFromHash('#x=1&mode=showcase'), 'showcase')
  assert.equal(modeFromHash('#mode=../../etc'), 'console')
  assert.equal(modeFromHash('#mode=unknown'), 'console')
  assert.equal(hashForMode('console'), '')
  for (const m of MODES) assert.equal(modeFromHash(hashForMode(m.id) || '#'), m.id)
})

// ---------------------------------------------------------------- showcase scenarios

test('every showcase scenario is replayable: positions in the arena, rovers apart, faults on real rovers', () => {
  assert.ok(SCENARIOS.length >= 4)
  assert.equal(new Set(SCENARIOS.map(s => s.id)).size, SCENARIOS.length)
  for (const s of SCENARIOS) {
    const n = s.params.roverCount
    for (const frame of s.positions) {
      assert.equal(frame.length, n)
      for (const [x, z] of frame) assert.ok(Math.hypot(x, z) <= ARENA.radius, s.id)
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        assert.ok(Math.hypot(frame[i][0] - frame[j][0], frame[i][1] - frame[j][1]) >= 2 * ARENA.bodyRadius, `${s.id}: ${i}/${j} overlap`)
      }
    }
    for (let c = 0; c < s.cycles; c++) for (const f of faultsAt(s, c)) assert.ok(f.rover === null || (f.rover >= 0 && f.rover < n))
  }
})

test('recorded showcase outcomes are wins, and match their own per-cycle traces', () => {
  for (const s of SCENARIOS) {
    if (s.comparison.startsWith('assign:')) {
      const m = s.measured as AssignmentSummary
      const trace = s.trace as number[][]
      assert.deepEqual(m.ours, [trace.reduce((a, t) => a + t[0], 0), trace.reduce((a, t) => a + t[1], 0)], s.id)
      assert.deepEqual(m.baseline, [trace.reduce((a, t) => a + t[2], 0), trace.reduce((a, t) => a + t[3], 0)], s.id)
      assert.equal(m.cyclesWon, trace.filter(t => traceVerdict(t) === 'win').length)
      assert.ok(m.ours[0] / m.ours[1] > m.baseline[0] / m.baseline[1], `${s.id} is not a win`)
    } else {
      const m = s.measured as SolverSummary
      for (const v of ['infeasible', 'worse', 'matched', 'better'] as const) assert.equal(m[v], s.trace.filter(t => t[0] === v).length, `${s.id} ${v}`)
      assert.equal(m.better, 0, `${s.id}: the sampler beat us on some cycle`)
      assert.ok(m.infeasible + m.worse > 0, `${s.id} is not a win`)
    }
  }
})

test('scripted positions: exact at whole cycles, eased between, held after the last', () => {
  const s = SCENARIOS.find(x => x.id === 'fragile-bridge')!
  assert.deepEqual(positionAt(s, 0, 3), s.positions[3][0])
  assert.deepEqual(positionAt(s, 0, s.cycles + 5), s.positions[s.cycles - 1][0])
  const [, zMid] = positionAt(s, 0, 2.5)
  const [z2, z3] = [s.positions[2][0][1], s.positions[3][0][1]]
  close(zMid, (z2 + z3) / 2, 1e-9)
})

test('scripted motion never overshoots its target and keeps rovers on the recorded path', () => {
  const s = SCENARIOS[0]
  const motion = new ScriptedMotion(s)
  motion.target = 2
  for (let k = 0; k < 400; k++) motion.advance(1 / 60, 1)
  assert.equal(motion.t, 2)
  const pose = motion.pose(3, { x: 0, z: 0, yaw: 0, left: 0, right: 0 })
  close(pose.x, s.positions[2][3][0], 1e-9)
  close(pose.z, s.positions[2][3][1], 1e-9)
})

// ---------------------------------------------------------------- showcase comparison

const rover = (index: number, trust = 0.5) => ({ index, trust, battery: 1, compute: 1, capability: 1, reliability: 0.6, pathFactor: 1, task: null })
const task = (id: string, r: number, success: boolean, executed = success) => ({ id, rover: r, executed, delivered: true, success, explored: false })

function assignmentStep(ours: ReturnType<typeof task>[], base: ReturnType<typeof task>[]) {
  return {
    sessionId: 's', cycle: 5, rovers: [0, 1, 2, 3].map(i => rover(i)), unassigned: [], faults: [],
    edges: [{ from: 0, to: 1, active: true, quality: 0.8, utility: 0.6 }, { from: 1, to: 2, active: false, quality: 0.5, utility: 0.3 }],
    tasks: ours,
    metrics: { cycleSuccesses: ours.filter(t => t.success).length, cycleTasks: ours.length } as never,
    assignmentBaselines: { trust_only: { successes: 0, tasks: 0, cycleSuccesses: base.filter(t => t.success).length, cycleTasks: base.length, trust: [0.9, 0.4, 0.4, 0.4], outcomes: base } },
    solver: null,
  }
}

test('assignment comparison marks our good pick and the baseline failure, with the real reason', () => {
  const result = compareCycle(assignmentStep([task('t-5-0', 2, true)], [task('t-5-0', 1, false)]), 'assign:trust_only', 'Trust-only')
  assert.equal(result.verdict, 'win')
  assert.equal(result.ours.roverMarks[2], MARK_GOOD)
  assert.equal(result.baseline.roverMarks[1], MARK_BAD)
  assert.deepEqual(result.focus, [2, 1])
  assert.match(result.headline, /RV-02.*work itself failed.*RV-03/)
  assert.deepEqual(result.baseline.trust, [0.9, 0.4, 0.4, 0.4])
})

test('assignment comparison reports a baseline win honestly, and ties without marks', () => {
  const loss = compareCycle(assignmentStep([task('a', 2, false)], [task('a', 1, true)]), 'assign:trust_only', 'Trust-only')
  assert.equal(loss.verdict, 'loss')
  assert.match(loss.headline, /Trust-only won this cycle/)
  const tie = compareCycle(assignmentStep([task('a', 2, true)], [task('a', 1, true)]), 'assign:trust_only', 'Trust-only')
  assert.equal(tie.verdict, 'tie')
  assert.ok(tie.ours.roverMarks.every(m => m === 0) && tie.baseline.roverMarks.every(m => m === 0))
})

function solverStep(run: object) {
  return { ...assignmentStep([], []), solver: { baseline: 'bitsa', oursMs: 10, latest: run, totals: { runs: 1, feasible: 0, matched: 0, gapSum: 0, msSum: 0 } } }
}

test('solver comparison: an over-cap, disconnected answer is marked on the baseline side', () => {
  const run = { feasible: false, energy: 900, oursEnergy: -2, gap: 450, ms: 50, matched: false, oursCost: 20,
    edges: [[0, 1], [1, 2], [1, 3]], violations: { overCap: [[1, 3, 2]], unreachable: [3], cost: 31, budget: 30 } }
  const result = compareCycle(solverStep(run) as never, 'solver:bitsa', 'Bit-flip SA')
  assert.equal(result.verdict, 'win')
  assert.equal(result.solver!.verdict, 'infeasible')
  assert.equal(result.baseline.roverMarks[1], MARK_BAD)
  assert.equal(result.baseline.roverMarks[3], MARK_BAD)
  assert.ok(result.baseline.linkMarks.some(([a, b, m]) => a === 1 && b === 2 && m === MARK_BAD))
  assert.match(result.headline, /RV-02 carries 3 links against a cap of 2.*RV-04 is cut off.*cost 31 against a budget of 30/)
  assert.deepEqual(result.baseline.edges.map(e => e.active), [true, true])
})

test('solver comparison: a lower energy than ours counts as a loss', () => {
  const run = { feasible: true, energy: -3, oursEnergy: -2, gap: -0.5, ms: 50, matched: true, oursCost: 20, edges: [[0, 1]], violations: { overCap: [], unreachable: [], cost: 5, budget: 30 } }
  const result = compareCycle(solverStep(run) as never, 'solver:bitsa', 'Bit-flip SA')
  assert.equal(result.verdict, 'loss')
  assert.equal(result.solver!.verdict, 'better')
})

test('a scripted side applies links, marks, faults and in-place task columns that resolve after a beat', () => {
  const s = SCENARIOS[0]
  const motion = new ScriptedMotion(s)
  const side = new ScriptedSide(motion)
  const n = side.count
  const result = compareCycle(assignmentStep([task('t', 2, true)], [task('t', 1, false)]) as never, 'assign:trust_only', 'Trust-only')
  side.apply(result.ours, [3])
  assert.equal(side.planState[0 * n + 1], LINK_KEPT)
  assert.equal(side.planState[1 * n + 2], LINK_CANDIDATE)
  assert.equal(side.roverMarks[2], MARK_GOOD)
  assert.equal(side.faulted[3], 1)
  assert.equal(side.parent[1], 0)
  assert.equal(side.beacons.length, 1)
  assert.equal(side.beacons[0].x, s.positions[0][2][0])
  assert.equal(side.beacons[0].state, 'enroute')
  motion.advance(1.0, 0)
  side.advance()
  assert.equal(side.beacons[0].state, 'resolved')
})

// ---------------------------------------------------------------- photoreal light

test('solar position at the site: a north-facing noon sun, a horizon sunrise, a symmetric afternoon', () => {
  const noon = solarPosition(12)
  assert.ok(noon.elevation > 0.8 && noon.elevation < 1.2, `noon elevation ${noon.elevation}`)
  assert.ok(Math.min(noon.azimuth, 2 * Math.PI - noon.azimuth) < 0.01, 'southern-hemisphere noon sun is due north')
  assert.ok(Math.abs(solarPosition(6.15).elevation) < 0.08)
  close(solarPosition(9).elevation, solarPosition(15).elevation, 1e-9)
  close(length(sunDirection(noon)), 1, 1e-12)
  const east = sunDirection({ elevation: 0, azimuth: Math.PI / 2 })
  close(east[0], 1); close(east[2], 0)
})

test('rotateY follows three.js: +90 degrees takes +x to -z', () => {
  const [x, y, z] = rotateY([1, 0, 0], Math.PI / 2)
  close(x, 0); close(y, 0); close(z, -1)
})

test('equirect directions: the middle row is the horizon, the top row looks up', () => {
  const [, y] = equirectDirection(10, 512, 2048, 1024)
  assert.ok(Math.abs(y) < 0.01)
  assert.ok(equirectDirection(0, 0, 2048, 1024)[1] > 0.99)
})

function syntheticSky(width: number, height: number, sun: [number, number] | null): PixelSampler {
  return (column, row) => {
    if (sun && Math.abs(column - sun[0]) <= 2 && Math.abs(row - sun[1]) <= 2) return [5000, 4800, 4500]
    return row < height / 2 ? [0.6, 0.8, 1.1] : [0.3, 0.25, 0.2]
  }
}

test('the sun is found in an HDRI, measured, and absent from an overcast sky', () => {
  const w = 512, h = 256
  const found = findSun(syntheticSky(w, h, [100, 40]), w, h, 1)!
  const expected = equirectDirection(100, 40, w, h)
  assert.ok(found.direction.every((v, i) => Math.abs(v - expected[i]) < 0.02))
  const measured = measureSun(syntheticSky(w, h, [100, 40]), w, h)!
  const solidAngle = 25 * (2 * Math.PI / w) * (Math.PI / h) * Math.cos((0.5 - 40.5 / h) * Math.PI)
  assert.ok(Math.abs(measured.irradiance / (4800 * solidAngle) - 1) < 0.1, `irradiance ${measured.irradiance}`)
  assert.equal(findSun(syntheticSky(w, h, null), w, h), null)
})

// ---------------------------------------------------------------- photoreal site

test('terrain is exactly flat wherever a rover can drive and rises to a rim far out', () => {
  for (let k = 0; k < 200; k++) {
    const a = k * 2.399, r = (k / 200) * FLAT_RADIUS
    assert.equal(terrainHeight(r * Math.cos(a), r * Math.sin(a)), 0)
  }
  assert.ok(Math.abs(terrainHeight(FLAT_RADIUS + 0.05, 0)) < 0.01, 'continuous at the arena edge')
  assert.ok(terrainHeight(300, 0) > 5)
})

test('scattered props are deterministic, outside the arena and spaced', () => {
  const spec = { count: 60, minRadius: 2, maxRadius: 40, scale: [1, 2] as [number, number], variants: 3, spacing: 1.5, inward: 0.5 }
  const a = scatter(spec, 7), b = scatter(spec, 7)
  assert.deepEqual(a, b)
  for (const p of a) assert.ok(Math.hypot(p.x, p.z) > ARENA.radius && p.variant >= 0 && p.variant < 3)
  for (let i = 0; i < a.length; i++) for (let j = i + 1; j < a.length; j++) {
    assert.ok(Math.hypot(a[i].x - a[j].x, a[i].z - a[j].z) >= spec.spacing * (a[i].scale + a[j].scale) / 2 - 1e-9)
  }
})

test('the shipped photoreal manifest is complete, credited, and its files exist within budget', () => {
  const root = new URL('../public/photoreal/', import.meta.url)
  const manifest = validateManifest(JSON.parse(readFileSync(new URL('manifest.json', root), 'utf8')))
  const files = [manifest.rover.file, ...manifest.props.map(p => p.file), ...manifest.grounds.flatMap(g => [g.color, g.normal, g.arh]), ...manifest.skies.map(s => s.file)]
  let bytes = 0
  for (const file of files) {
    assert.ok(existsSync(new URL(file, root)), `${file} is missing`)
    bytes += statSync(new URL(file, root)).size
  }
  assert.ok(bytes < 60 * 1024 * 1024, `assets total ${(bytes / 1048576).toFixed(1)} MB`)
  for (const credit of Object.values(manifest.credits)) assert.ok(credit.title && credit.author && credit.license && credit.url.startsWith('https://'))
  assert.ok(existsSync(new URL('CREDITS.md', root)))
  assert.throws(() => validateManifest({ ...manifest, credits: {} }))
})
