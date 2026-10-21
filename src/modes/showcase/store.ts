import { create } from 'zustand'
import type { CycleComparison } from './compare'
import { SCENARIOS } from './scenarios'

export type RunStatus = 'running' | 'paused' | 'done' | 'error'
/** Planning cycles per second of animation; highlights hold the frame longer. */
export type Speed = 1 | 2 | 4

interface ShowcaseState {
  scenarioId: string
  /** Bumped to restart the current scenario from cycle 0 with a fresh planner session. */
  runId: number
  status: RunStatus
  speed: Speed
  /** Glide the camera to each highlight. */
  directed: boolean
  results: CycleComparison[]
  /** Whether each live cycle reproduced the recorded verification run. */
  reproduced: boolean[]
  error: string | null
  /** Called as a run starts: clears the previous run's results. */
  begin: () => void
  select: (id: string) => void
  restart: () => void
  togglePlay: () => void
  setSpeed: (speed: Speed) => void
  setDirected: (on: boolean) => void
  record: (result: CycleComparison, reproduced: boolean) => void
  finish: () => void
  fail: (message: string) => void
}

export const useShowcase = create<ShowcaseState>()(set => ({
  scenarioId: SCENARIOS[0]?.id ?? '',
  runId: 0,
  status: 'running',
  speed: 1,
  directed: true,
  results: [],
  reproduced: [],
  error: null,
  begin: () => set({ status: 'running', results: [], reproduced: [], error: null }),
  select: scenarioId => set(state => ({ scenarioId, runId: state.runId + 1 })),
  restart: () => set(state => ({ runId: state.runId + 1 })),
  togglePlay: () => set(state => ({ status: state.status === 'running' ? 'paused' : state.status === 'paused' ? 'running' : state.status })),
  setSpeed: speed => set({ speed }),
  setDirected: directed => set({ directed }),
  record: (result, reproduced) => set(state => ({ results: [...state.results, result], reproduced: [...state.reproduced, reproduced] })),
  finish: () => set({ status: 'done' }),
  fail: error => set({ status: 'error', error }),
}))
