import type { PruneMode } from '../api/planner'
import type { ParamSpec } from '../config/fleet'
import type { RoverActivity } from '../core/fleet'

export const pct = (value: number, digits = 0) => (Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : '–')
export const signedPts = (value: number, digits = 1) =>
  (Number.isFinite(value) ? `${value >= 0 ? '+' : '−'}${Math.abs(value * 100).toFixed(digits)} pts` : '–')
export const ms = (value: number) => (!Number.isFinite(value) ? '–' : value < 1 ? '<1 ms' : value < 1000 ? `${value.toFixed(0)} ms` : `${(value / 1000).toFixed(2)} s`)

export function paramValue(spec: ParamSpec, value: number): string {
  if (spec.unit === '%') return `${Math.round(value * 100)}%`
  const digits = spec.integer ? 0 : Math.max(0, -Math.floor(Math.log10(spec.step) + 1e-9))
  const text = value.toFixed(digits)
  if (!spec.unit) return text
  return spec.unit === '×' ? `${text}×` : `${text} ${spec.unit}`
}

/** What each relaxation level of the planner guarantees, in plain words. */
export const PRUNE_MODES: Record<PruneMode, { label: string; detail: string; tone: 'good' | 'warn' | 'bad' | 'idle' }> = {
  full: { label: 'All constraints met', detail: 'Link caps, the link budget and full connectivity to the root all hold.', tone: 'good' },
  'no-budget': { label: 'Budget relaxed', detail: 'No connected plan fits the budget, so the planner kept link caps and connectivity and dropped the budget.', tone: 'warn' },
  'budget-only': { label: 'Connectivity relaxed', detail: 'Link caps and connectivity conflict, so connectivity was dropped; some rovers may not reach the root.', tone: 'warn' },
  unconstrained: { label: 'Link caps only', detail: 'Only per-rover link caps could be enforced this cycle.', tone: 'bad' },
  'keep-all-fallback': { label: 'Kept every link', detail: 'Even the link caps were infeasible, so no pruning happened this cycle.', tone: 'bad' },
  'stale-reuse': { label: 'Reused last plan', detail: 'The previous solution was reused on the live graph.', tone: 'warn' },
  none: { label: 'No links in range', detail: 'No two rovers are within radio range, so there is nothing to prune.', tone: 'idle' },
  paired: { label: 'Paired topology', detail: 'Baseline arm sharing the primary plan.', tone: 'idle' },
  threshold: { label: 'Threshold pruner', detail: 'Heuristic pruning.', tone: 'idle' },
  topk: { label: 'Top-K pruner', detail: 'Heuristic pruning.', tone: 'idle' },
}

export const ACTIVITY_LABEL: Record<RoverActivity, string> = {
  driving: 'Driving', turning: 'Turning', yielding: 'Yielding', detour: 'Detouring', working: 'On task', parked: 'Low battery',
}

export const FAULT_LABEL = {
  reliability_drop: 'Reliability drop',
  battery_drain: 'Battery drain',
  packet_loss_spike: 'Radio storm',
} as const
