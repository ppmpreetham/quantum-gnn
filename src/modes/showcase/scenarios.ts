import raw from '../../../shared/showcase-scenarios.json' with { type: 'json' }
import { sanitizeParams } from '../../config/fleet.ts'
import type { FleetParams } from '../../config/fleet.ts'

/**
 * The showcase scenarios generated and verified by web/showcase_scenarios.py:
 * fixed parameters, the rover positions of every cycle and the faults of
 * every cycle, plus the outcome measured when they were replayed through
 * the planner. Pure data and maths, unit-tested in Node.
 */
export type FaultKind = 'reliability_drop' | 'battery_drain' | 'packet_loss_spike'
export interface ScenarioFault { kind: FaultKind; rover: number | null }
export type SolverVerdict = 'infeasible' | 'worse' | 'matched' | 'better' | 'skipped'
export interface AssignmentSummary { ours: [number, number]; baseline: [number, number]; cyclesWon: number; cyclesLost: number }
export type SolverSummary = Record<SolverVerdict, number>

export interface Scenario {
  id: string
  title: string
  comparison: string
  watch: string
  mechanism: string
  params: FleetParams
  cycles: number
  positions: [number, number][][]
  faults: Record<string, ScenarioFault[]>
  measured: AssignmentSummary | SolverSummary
  /** Per-cycle outcome of the verification run: [ours ok, ours tasks, base ok, base tasks] or [verdict]. */
  trace: (number[] | [SolverVerdict])[]
}

interface ScenarioFile { version: 1; note: string; scenarios: Scenario[] }

/** Rejects a scenario the planner would refuse or the renderer could not place. */
export function validateScenario(s: Scenario): Scenario {
  const n = s.params.roverCount
  const ok = typeof s.id === 'string' && typeof s.comparison === 'string' && s.cycles > 0
    && s.positions.length === s.cycles && s.positions.every(frame => frame.length === n && frame.every(p => p.length === 2 && p.every(Number.isFinite)))
    && s.trace.length === s.cycles
    && Object.values(s.faults).flat().every(f => f.rover === null || (Number.isInteger(f.rover) && f.rover >= 0 && f.rover < n))
  if (!ok) throw new Error(`Showcase scenario ${s.id} is malformed; regenerate it with web/showcase_scenarios.py`)
  const params = sanitizeParams(s.params)
  if (Object.keys(params).some(key => params[key as keyof FleetParams] !== s.params[key as keyof FleetParams])) {
    throw new Error(`Showcase scenario ${s.id} has parameters outside the planner's limits`)
  }
  return s
}

const file = raw as unknown as ScenarioFile
export const SCENARIO_NOTE = file.note
export const SCENARIOS: readonly Scenario[] = file.scenarios.map(validateScenario)

export const isAssignment = (s: Scenario) => s.comparison.startsWith('assign:')
export const baselineOf = (s: Scenario) => s.comparison.split(':')[1]

export function faultsAt(s: Scenario, cycle: number): ScenarioFault[] {
  return s.faults[String(cycle)] ?? []
}

/**
 * Render position of a rover at a fractional cycle: smooth (eased) motion
 * from frame floor(t) to the next, holding the last frame. Heading follows
 * the direction of travel and keeps its last value when a rover is still.
 */
export function positionAt(s: Scenario, rover: number, t: number): [number, number] {
  const last = s.cycles - 1
  const k = Math.max(0, Math.min(last, Math.floor(t)))
  const next = Math.min(last, k + 1)
  const f = Math.max(0, Math.min(1, t - k))
  const e = f * f * (3 - 2 * f)
  const [ax, az] = s.positions[k][rover], [bx, bz] = s.positions[next][rover]
  return [ax + (bx - ax) * e, az + (bz - az) * e]
}

/** Verdict of one recorded cycle, from our side: did we win, lose or tie it? */
export function traceVerdict(entry: Scenario['trace'][number]): 'win' | 'loss' | 'tie' {
  if (typeof entry[0] === 'string') {
    const v = entry[0]
    return v === 'infeasible' || v === 'worse' ? 'win' : v === 'better' ? 'loss' : 'tie'
  }
  const [ours, , base] = entry as number[]
  return ours > base ? 'win' : ours < base ? 'loss' : 'tie'
}
