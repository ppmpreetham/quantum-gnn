import type { FleetParams } from '../config/fleet'
import type { PlannerEdge, PlannerRover, PlannerTask } from '../core/fleet'

export type PruneMode =
  | 'full' | 'no-budget' | 'budget-only' | 'unconstrained' | 'keep-all-fallback'
  | 'stale-reuse' | 'none' | 'paired' | 'threshold' | 'topk'

export interface PlannerMetrics {
  successes: number; tasks: number; cycleSuccesses: number; cycleTasks: number
  buildMs: number; solveMs: number; gnnMs: number; pruneMode: PruneMode
  variables: number; energy: number | null; keptCount: number; candidateCount: number
  reachable: number; fleet: number; minSpanningCost: number | null; budget: number; degreeCap: number
}
/** A follower's tallies plus its own trust estimates and this cycle's outcomes (it shares our topology, not our choices). */
export interface BaselineTally {
  successes: number; tasks: number; cycleSuccesses: number; cycleTasks: number
  trust: number[]; outcomes: PlannerTask[]
}
/** Hard constraints an edge set breaks: [rover, degree, cap] triples, rovers cut off from the root, and its cost. */
export interface ConstraintReport { overCap: [number, number, number][]; unreachable: number[]; cost: number; budget: number | null }
export interface SolverRun {
  feasible: boolean; energy: number; oursEnergy: number; gap: number; ms: number; matched: boolean
  edges: [number, number][]; violations: ConstraintReport; oursCost: number
}
export interface SolverComparison {
  baseline: string; oursMs: number
  latest: SolverRun | { error: string } | null
  totals: { runs: number; feasible: number; matched: number; gapSum: number; msSum: number }
}
export interface ActiveFault { kind: FaultKind; rover: number | null; remaining: number }
export type FaultKind = 'reliability_drop' | 'battery_drain' | 'packet_loss_spike'
export interface FaultRequest { kind: FaultKind; rover?: number }
/** A task the user placed on the map; sent with the next cycle, which then dispatches only user tasks. */
export interface UserTask { id: string; x: number; z: number }

export interface StepResponse {
  sessionId: string; cycle: number
  rovers: PlannerRover[]; edges: PlannerEdge[]; tasks: PlannerTask[]; faults: ActiveFault[]
  /** User task ids no rover could take this cycle. */
  unassigned: string[]
  metrics: PlannerMetrics
  assignmentBaselines: Record<string, BaselineTally>
  solver: SolverComparison | null
}
export interface StepRequest {
  sessionId?: string; params: Partial<FleetParams>; positions: [number, number][]
  faults: FaultRequest[]; comparison: string
  /** Omitted: the planner draws its own tasks. A list (even empty): exactly these tasks this cycle. */
  tasks?: UserTask[]
}
export interface Health { status: string; engine: string; solvers: Record<string, boolean> }

export class PlannerError extends Error {
  readonly retryable: boolean
  readonly busy: boolean
  constructor(message: string, retryable: boolean, busy = false) {
    super(message)
    this.retryable = retryable
    this.busy = busy
  }
}

async function readError(response: Response): Promise<string> {
  try {
    const body = await response.json() as { error?: unknown }
    if (typeof body.error === 'string') return body.error
  } catch { /* fall through to the status text */ }
  return `Planner returned HTTP ${response.status}`
}

function isStepResponse(value: unknown): value is StepResponse {
  const v = value as StepResponse
  return !!v && typeof v.sessionId === 'string' && Array.isArray(v.rovers) && Array.isArray(v.edges)
    && Array.isArray(v.tasks) && typeof v.metrics === 'object' && v.metrics !== null
}

export async function requestStep(body: StepRequest, signal: AbortSignal): Promise<StepResponse> {
  let response: Response
  try {
    response = await fetch('/api/step', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal,
    })
  } catch (error) {
    if (signal.aborted) throw error
    throw new PlannerError('Cannot reach the planner. Start it with npm run dev.', true)
  }
  if (!response.ok) {
    // other 4xx mean this request is wrong and will stay wrong; 429 (sessions full) and 5xx can recover
    const status = response.status
    throw new PlannerError(await readError(response), status >= 500 || status === 429, status === 503)
  }
  const data: unknown = await response.json()
  if (!isStepResponse(data)) throw new PlannerError('The planner sent an unexpected response.', true)
  return data
}

export async function fetchHealth(signal: AbortSignal): Promise<Health | null> {
  try {
    const response = await fetch('/api/health', { signal })
    return response.ok ? await response.json() as Health : null
  } catch {
    return null
  }
}
