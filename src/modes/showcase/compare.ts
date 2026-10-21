import type { SolverRun, StepResponse } from '../../api/planner.ts'
import type { PlannerEdge, PlannerTask } from '../../core/fleet.ts'
import { MARK_BAD, MARK_GOOD } from '../../core/fleetView.ts'
import type { Mark } from '../../core/fleetView.ts'

/**
 * Turns one planner answer into what each half of the split screen shows:
 * our plan on the left, the baseline's on the right (same positions, same
 * faults, same seed), plus the highlight that explains the cycle. Every
 * mark and sentence is derived from the real outcomes in the response.
 */
export interface SidePlan {
  trust: number[]
  edges: PlannerEdge[]
  tasks: PlannerTask[]
  roverMarks: Mark[]
  /** [a, b, mark] with a < b. */
  linkMarks: [number, number, Mark][]
}

export type Verdict = 'win' | 'loss' | 'tie'

export interface CycleComparison {
  cycle: number
  verdict: Verdict
  headline: string
  /** Rovers the camera should frame for this highlight (empty: overview). */
  focus: number[]
  ours: SidePlan
  baseline: SidePlan
  /** Assignment arms: [delivered, dispatched] this cycle. */
  score?: { ours: [number, number]; baseline: [number, number] }
  /** Solver arms: the sampler's run on our Q matrix. */
  solver?: { verdict: 'infeasible' | 'worse' | 'matched' | 'better'; run: SolverRun }
}

export const roverName = (i: number) => `RV-${String(i + 1).padStart(2, '0')}`

const blankMarks = (n: number): Mark[] => Array<Mark>(n).fill(0 as Mark)
const pair = (a: number, b: number): [number, number] => (a < b ? [a, b] : [b, a])

function failure(task: PlannerTask): string {
  return !task.executed ? 'the work itself failed' : 'the result was lost on its way to the root'
}

function compareAssignment(step: StepResponse, baseline: string, label: string): CycleComparison {
  const n = step.rovers.length
  const tally = step.assignmentBaselines[baseline]
  const ours: SidePlan = { trust: step.rovers.map(r => r.trust), edges: step.edges, tasks: step.tasks, roverMarks: blankMarks(n), linkMarks: [] }
  const base: SidePlan = { trust: tally.trust, edges: step.edges, tasks: tally.outcomes, roverMarks: blankMarks(n), linkMarks: [] }
  const okOurs = step.metrics.cycleSuccesses, okBase = tally.cycleSuccesses
  const verdict: Verdict = okOurs > okBase ? 'win' : okOurs < okBase ? 'loss' : 'tie'
  const byId = new Map(tally.outcomes.map(t => [t.id, t]))
  const decisive = step.tasks.find(t => t.success && byId.get(t.id) && !byId.get(t.id)!.success)
  const reverse = step.tasks.find(t => !t.success && byId.get(t.id)?.success)
  let headline: string, focus: number[] = []
  if (verdict === 'win' && decisive) {
    const other = byId.get(decisive.id)!
    ours.roverMarks[decisive.rover] = MARK_GOOD
    base.roverMarks[other.rover] = MARK_BAD
    focus = [...new Set([decisive.rover, other.rover])]
    headline = other.rover === decisive.rover
      ? `Both picked ${roverName(other.rover)}, but under ${label} ${failure(other)}; ours delivered.`
      : `${label} sent a task to ${roverName(other.rover)} and ${failure(other)}. Ours chose ${roverName(decisive.rover)} and delivered.`
  } else if (verdict === 'loss' && reverse) {
    const other = byId.get(reverse.id)!
    base.roverMarks[other.rover] = MARK_GOOD
    ours.roverMarks[reverse.rover] = MARK_BAD
    focus = [...new Set([reverse.rover, other.rover])]
    headline = `${label} won this cycle: its pick ${roverName(other.rover)} delivered while ours, ${roverName(reverse.rover)}, did not.`
  } else {
    headline = `Both delivered ${okOurs} of ${step.metrics.cycleTasks}.`
  }
  return { cycle: step.cycle, verdict, headline, focus, ours, baseline: base, score: { ours: [okOurs, step.metrics.cycleTasks], baseline: [okBase, tally.cycleTasks] } }
}

function compareSolver(step: StepResponse, label: string): CycleComparison {
  const latest = step.solver?.latest
  const n = step.rovers.length
  const trust = step.rovers.map(r => r.trust)
  if (!latest || !('feasible' in latest)) {
    const plan = (): SidePlan => ({ trust, edges: step.edges, tasks: step.tasks, roverMarks: blankMarks(n), linkMarks: [] })
    return { cycle: step.cycle, verdict: 'tie', headline: 'No QUBO was solved this cycle, so there was nothing to compare.', focus: [], ours: plan(), baseline: plan() }
  }
  const chosen = new Set(latest.edges.map(([a, b]) => pair(a, b).join(':')))
  const ours: SidePlan = { trust, edges: step.edges, tasks: step.tasks, roverMarks: blankMarks(n), linkMarks: [] }
  const base: SidePlan = {
    trust, tasks: [], roverMarks: blankMarks(n), linkMarks: [],
    edges: step.edges.map(e => ({ ...e, active: chosen.has(pair(e.from, e.to).join(':')) })),
  }
  const scale = Math.max(1, Math.abs(latest.oursEnergy))
  const verdict = !latest.feasible ? 'infeasible' : latest.energy < latest.oursEnergy - 1e-9 * scale ? 'better' : latest.matched ? 'matched' : 'worse'
  const v = latest.violations
  const focus: number[] = []
  const reasons: string[] = []
  for (const [rover, degree, cap] of v.overCap) {
    base.roverMarks[rover] = MARK_BAD
    focus.push(rover)
    reasons.push(`${roverName(rover)} carries ${degree} links against a cap of ${cap}`)
    for (const [a, b] of latest.edges) if (a === rover || b === rover) base.linkMarks.push([...pair(a, b), MARK_BAD])
  }
  for (const rover of v.unreachable) {
    base.roverMarks[rover] = MARK_BAD
    focus.push(rover)
  }
  if (v.unreachable.length) reasons.push(`${v.unreachable.map(roverName).join(', ')} ${v.unreachable.length === 1 ? 'is' : 'are'} cut off from the root`)
  if (v.budget !== null && v.cost > v.budget) reasons.push(`its links cost ${v.cost} against a budget of ${v.budget}`)
  // links we kept that the sampler dropped, and the ones it kept instead
  for (const e of step.edges) {
    const key = pair(e.from, e.to)
    const inBase = chosen.has(key.join(':'))
    if (e.active && !inBase && verdict !== 'matched') ours.linkMarks.push([...key, MARK_GOOD])
  }
  let headline: string
  if (verdict === 'infeasible') {
    headline = `${label} returned an edge set that breaks a hard constraint: ${reasons.join('; ') || 'its flow and slack bits are inconsistent'}. Ours meets every constraint.`
  } else if (verdict === 'worse') {
    const gap = latest.gap
    headline = `${label} found a feasible plan, but its energy is ${gap > 10 ? 'dominated by active penalty terms' : `${(gap * 100).toFixed(1)}% above ours`}.`
  } else if (verdict === 'better') {
    headline = `${label} beat our annealer this cycle (energy ${(latest.gap * 100).toFixed(2)}% lower).`
  } else {
    headline = `${label} matched our energy on this cycle's Q matrix.`
  }
  return {
    cycle: step.cycle, verdict: verdict === 'infeasible' || verdict === 'worse' ? 'win' : verdict === 'better' ? 'loss' : 'tie',
    headline, focus: [...new Set(focus)], ours, baseline: base, solver: { verdict, run: latest },
  }
}

/** Splits a step into the two sides of the showcase. */
export function compareCycle(step: StepResponse, comparison: string, label: string): CycleComparison {
  const [kind, baseline] = comparison.split(':')
  return kind === 'assign' ? compareAssignment(step, baseline, label) : compareSolver(step, label)
}
