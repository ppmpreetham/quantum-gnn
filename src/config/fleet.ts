import raw from '../../shared/fleet-config.json' with { type: 'json' }

export type ParamKey =
  | 'roverCount' | 'seed' | 'speed' | 'timeScale'
  | 'communicationRange' | 'packetLoss'
  | 'maxDegree' | 'budget' | 'restarts' | 'coolingRate'
  | 'tasksPerCycle' | 'exploration' | 'trustMemory' | 'reliabilityDrift'
  | 'plannerInterval'
export type FleetParams = Record<ParamKey, number>
export type ParamScope = 'session' | 'planner' | 'client'

export interface ParamSpec {
  key: ParamKey; label: string; group: string; scope: ParamScope
  min: number; max: number; step: number; integer?: boolean; unit?: string; help: string
}
export interface Preset { id: string; name: string; description: string; measuredCycleMs: number; params: FleetParams }
export interface Reference {
  metric: string; ours: number; baseline: number; delta?: number
  ci?: [number, number]; p?: number; significant?: boolean; source: string
}
export interface Comparison {
  id: string; kind: 'none' | 'assignment' | 'solver'; baseline?: string
  label: string; short: string; summary?: string; reference?: Reference
}
export interface ArenaSpec {
  radius: number; bodyRadius: number; wheelRadius: number; track: number
  maxWheelSpeed: number; maxAccel: number; maxBrake: number; maxYawRate: number; maxYawAccel: number
}
interface FleetConfig {
  arena: ArenaSpec; params: ParamSpec[]; groups: { id: string; label: string }[]
  defaults: FleetParams; presets: Preset[]; comparisons: Comparison[]
}

const config = raw as unknown as FleetConfig

export const ARENA: Readonly<ArenaSpec> = config.arena
export const PARAM_SPECS: readonly ParamSpec[] = config.params
export const PARAM_GROUPS = config.groups
export const DEFAULT_PARAMS: Readonly<FleetParams> = config.defaults
export const PRESETS: readonly Preset[] = config.presets
export const COMPARISONS: readonly Comparison[] = config.comparisons
export const SPEC_BY_KEY = Object.fromEntries(PARAM_SPECS.map(s => [s.key, s])) as Record<ParamKey, ParamSpec>

const decimals = (step: number) => Math.max(0, -Math.floor(Math.log10(step) + 1e-9))

/** Snap to the spec's grid inside [min, max]; non-finite input falls back to the default. */
export function snapParam(key: ParamKey, value: number): number {
  const spec = SPEC_BY_KEY[key]
  if (!Number.isFinite(value)) return DEFAULT_PARAMS[key]
  const bounded = Math.min(spec.max, Math.max(spec.min, value))
  const snapped = spec.min + Math.round((bounded - spec.min) / spec.step) * spec.step
  return Number(Math.min(spec.max, snapped).toFixed(decimals(spec.step)))
}

export function sanitizeParams(input: Partial<Record<ParamKey, unknown>>): FleetParams {
  const out = { ...DEFAULT_PARAMS }
  for (const spec of PARAM_SPECS) {
    const value = input[spec.key]
    out[spec.key] = snapParam(spec.key, typeof value === 'number' ? value : Number.NaN)
  }
  return out
}

export const plannerParams = (params: FleetParams) =>
  Object.fromEntries(PARAM_SPECS.filter(s => s.scope !== 'client').map(s => [s.key, params[s.key]])) as Partial<FleetParams>

export const sessionKey = (params: FleetParams) =>
  PARAM_SPECS.filter(s => s.scope === 'session').map(s => params[s.key]).join('|')

export const plannerKey = (params: FleetParams) =>
  PARAM_SPECS.filter(s => s.scope === 'planner').map(s => params[s.key]).join('|')

export interface Advisory { level: 'info' | 'warn'; text: string }

/** Constraint checks that follow from graph theory before any solve is attempted. */
export function advisories(params: FleetParams): Advisory[] {
  const notes: Advisory[] = []
  const n = params.roverCount
  if (n >= 3 && params.maxDegree < 2) {
    notes.push({ level: 'warn', text: `A tree on ${n} rovers needs a rover with at least 2 links, so a cap of 1 makes connectivity impossible. The planner will drop connectivity and say so.` })
  } else if (n >= 4 && params.maxDegree === 2) {
    notes.push({ level: 'info', text: 'A cap of 2 links allows only a chain through every rover (a Hamiltonian path). Expect relaxations whenever the radio graph has no such chain.' })
  }
  const minBudget = n - 1
  if (params.budget < minBudget) {
    notes.push({ level: 'warn', text: `Any spanning tree uses ${n - 1} links of at least 1 budget unit each, so a budget under ${minBudget} can never be met.` })
  }
  if (params.communicationRange < 2 * ARENA.bodyRadius + 0.2) {
    notes.push({ level: 'warn', text: 'The radio range is barely wider than two rover bodies, so links only form between rovers that are nearly touching.' })
  }
  if (params.tasksPerCycle > n) {
    notes.push({ level: 'info', text: `Only ${n} rovers can take tasks, so tasks per cycle is capped at ${n}.` })
  }
  return notes
}
