import { create } from 'zustand'
import { COMPARISONS, DEFAULT_PARAMS, PRESETS, sanitizeParams, snapParam } from '../config/fleet'
import type { FleetParams, ParamKey } from '../config/fleet'
import type { ActiveFault, FaultRequest, PlannerMetrics, PruneMode, SolverComparison, SolverRun, StepResponse } from '../api/planner'
import type { RoverActivity } from '../core/fleet'

export type CameraMode = 'orbit' | 'top' | 'follow' | 'chase'
export type QualityMode = 'auto' | 'high' | 'balanced' | 'low'
export type Tier = 'high' | 'balanced' | 'low'
export type OverlayKey = 'labels' | 'candidates' | 'ranges' | 'flow' | 'beacons' | 'path'
export type Overlays = Record<OverlayKey, boolean>
export type PlannerState = 'connecting' | 'online' | 'offline' | 'paused'

export interface CycleRecord {
  cycle: number
  ours: [successes: number, tasks: number]
  baselines: Record<string, [successes: number, tasks: number]>
  solveMs: number; kept: number; candidates: number; mode: PruneMode
  solver: (SolverRun & { baseline: string }) | null
}
export interface RoverTelemetry {
  index: number; id: string; trust: number; reliability: number; battery: number; compute: number
  pathFactor: number; task: string | null; activity: RoverActivity; speed: number
  x: number; z: number; yaw: number; parent: number; fault: ActiveFault['kind'] | null
}
export interface Telemetry {
  rovers: RoverTelemetry[]
  live: { candidates: number; kept: number; breaking: number; reachable: number }
}
export interface Perf { fps: number; calls: number; triangles: number; tier: Tier }

const HISTORY = 240
export const roverId = (i: number) => `RV-${String(i + 1).padStart(2, '0')}`

interface ConsoleState {
  params: FleetParams
  presetId: string | null
  comparison: string
  paused: boolean
  selected: number
  camera: CameraMode
  cameraNonce: number
  overlays: Overlays
  quality: QualityMode
  tether: boolean
  planner: { state: PlannerState; error: string | null; latencyMs: number }
  solvers: Record<string, boolean>
  metrics: PlannerMetrics | null
  solver: SolverComparison | null
  faults: ActiveFault[]
  pendingFaults: FaultRequest[]
  history: CycleRecord[]
  telemetry: Telemetry | null
  perf: Perf
  /** Tiers whose post-processing presented a blank frame on this GPU. */
  blankTiers: Tier[]
  modelName: string
  modelError: string | null

  setParam: (key: ParamKey, value: number) => void
  applyPreset: (id: string) => void
  setComparison: (id: string) => void
  togglePause: () => void
  select: (index: number) => void
  setCamera: (mode: CameraMode) => void
  recenter: () => void
  toggleOverlay: (key: OverlayKey) => void
  setQuality: (quality: QualityMode) => void
  setTether: (on: boolean) => void
  queueFault: (fault: FaultRequest) => void
  takeFaults: () => FaultRequest[]
  setPlanner: (state: PlannerState, error?: string | null, latencyMs?: number) => void
  setSolvers: (solvers: Record<string, boolean>) => void
  recordStep: (step: StepResponse) => void
  resetMission: () => void
  setTelemetry: (telemetry: Telemetry) => void
  setPerf: (perf: Perf) => void
  markBlank: (tier: Tier) => void
  setModel: (name: string, error: string | null) => void
}

export const useConsole = create<ConsoleState>()((set, get) => ({
  params: sanitizeParams(PRESETS[0]?.params ?? DEFAULT_PARAMS),
  presetId: PRESETS[0]?.id ?? null,
  comparison: COMPARISONS.find(c => c.kind === 'assignment')?.id ?? 'none',
  paused: false,
  selected: 0,
  camera: 'orbit',
  cameraNonce: 0,
  overlays: { labels: true, candidates: true, ranges: false, flow: true, beacons: true, path: true },
  quality: 'auto',
  tether: true,
  planner: { state: 'connecting', error: null, latencyMs: 0 },
  solvers: { bitsa: true },
  metrics: null,
  solver: null,
  faults: [],
  pendingFaults: [],
  history: [],
  telemetry: null,
  perf: { fps: 0, calls: 0, triangles: 0, tier: 'high' },
  blankTiers: [],
  modelName: 'Gazebo rover (SDF)',
  modelError: null,

  setParam: (key, value) => set(state => {
    const next = snapParam(key, value)
    if (state.params[key] === next) return state
    const params = { ...state.params, [key]: next }
    return { params, presetId: null, selected: Math.min(state.selected, params.roverCount - 1) }
  }),
  applyPreset: id => set(state => {
    const preset = PRESETS.find(p => p.id === id)
    if (!preset) return state
    const params = sanitizeParams(preset.params)
    return { params, presetId: id, selected: Math.min(state.selected, params.roverCount - 1) }
  }),
  setComparison: id => set(state => (COMPARISONS.some(c => c.id === id) ? { comparison: id, solver: null } : state)),
  togglePause: () => set(state => ({ paused: !state.paused })),
  select: index => set(state => ({ selected: Math.max(0, Math.min(state.params.roverCount - 1, Math.round(index))) })),
  setCamera: camera => set(state => ({ camera, cameraNonce: state.cameraNonce + 1 })),
  recenter: () => set(state => ({ camera: 'orbit', cameraNonce: state.cameraNonce + 1 })),
  toggleOverlay: key => set(state => ({ overlays: { ...state.overlays, [key]: !state.overlays[key] } })),
  setQuality: quality => set({ quality }),
  setTether: tether => set({ tether }),
  queueFault: fault => set(state => ({ pendingFaults: [...state.pendingFaults, fault].slice(-8) })),
  takeFaults: () => {
    const faults = get().pendingFaults
    if (faults.length) set({ pendingFaults: [] })
    return faults
  },
  setPlanner: (state, error = null, latencyMs) => set(current => ({
    planner: { state, error, latencyMs: latencyMs ?? current.planner.latencyMs },
  })),
  setSolvers: solvers => set({ solvers }),
  recordStep: step => set(state => {
    const m = step.metrics
    const latest = step.solver?.latest
    const record: CycleRecord = {
      cycle: step.cycle,
      ours: [m.cycleSuccesses, m.cycleTasks],
      baselines: Object.fromEntries(Object.entries(step.assignmentBaselines ?? {})
        .map(([name, tally]) => [name, [tally.cycleSuccesses, tally.cycleTasks]])),
      solveMs: m.solveMs, kept: m.keptCount, candidates: m.candidateCount, mode: m.pruneMode,
      solver: latest && 'feasible' in latest ? { ...latest, baseline: step.solver!.baseline } : null,
    }
    return {
      metrics: m, solver: step.solver, faults: step.faults ?? [],
      history: [...state.history, record].slice(-HISTORY),
    }
  }),
  resetMission: () => set({ metrics: null, solver: null, faults: [], pendingFaults: [], history: [], telemetry: null }),
  setTelemetry: telemetry => set({ telemetry }),
  setPerf: perf => set({ perf }),
  markBlank: tier => set(state => (state.blankTiers.includes(tier) ? state : { blankTiers: [...state.blankTiers, tier] })),
  setModel: (modelName, modelError) => set({ modelName, modelError }),
}))
