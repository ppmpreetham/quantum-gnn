import { createContext, useContext, useEffect, useMemo } from 'react'
import type { ReactNode } from 'react'
import { ARENA } from '../config/fleet'
import { FleetEngine } from '../core/fleet'
import type { FleetView, PoseSample } from '../core/fleetView'
import { useConsole } from './store'

/** Interpolated poses for the current frame, computed once and shared by every renderer. */
export class FramePoses {
  readonly x: Float32Array; readonly z: Float32Array; readonly yaw: Float32Array
  readonly left: Float32Array; readonly right: Float32Array
  private readonly scratch: PoseSample = { x: 0, z: 0, yaw: 0, left: 0, right: 0 }
  constructor(readonly count: number) {
    this.x = new Float32Array(count); this.z = new Float32Array(count); this.yaw = new Float32Array(count)
    this.left = new Float32Array(count); this.right = new Float32Array(count)
  }
  update(view: Pick<FleetView, 'renderPose'>): void {
    for (let i = 0; i < this.count; i++) {
      const p = view.renderPose(i, this.scratch)
      this.x[i] = p.x; this.z[i] = p.z; this.yaw[i] = p.yaw; this.left[i] = p.left; this.right[i] = p.right
    }
  }
}

/** Camera and pointer state the 2D views read and the scene consumes, without React renders. */
export interface ViewBus {
  footprint: Float32Array
  camera: { x: number; z: number; tx: number; tz: number }
  version: number
  request: { x: number; z: number } | null
  labels: (HTMLElement | null)[]
  /** Ground point under the cursor while placing tasks by click. */
  pointer: { x: number; z: number; active: boolean }
}

export const createViewBus = (): ViewBus => ({
  footprint: new Float32Array(8), camera: { x: 0, z: 0, tx: 0, tz: 0 }, version: 0, request: null, labels: [],
  pointer: { x: 0, z: 0, active: false },
})

/** What the 3D renderers draw: any FleetView with its interpolated poses. */
interface SceneValue { view: FleetView; poses: FramePoses; bus: ViewBus }
const SceneContext = createContext<SceneValue | null>(null)

export function SceneProvider({ value, children }: { value: SceneValue; children: ReactNode }) {
  return <SceneContext.Provider value={value}>{children}</SceneContext.Provider>
}

export function useScene(): SceneValue {
  const value = useContext(SceneContext)
  if (!value) throw new Error('useScene must be used inside a SceneProvider')
  return value
}

/** The live mission: the kinematic engine the planner loop drives, plus the panels that inspect it. */
interface FleetContextValue { engine: FleetEngine; poses: FramePoses; bus: ViewBus }
const FleetContext = createContext<FleetContextValue | null>(null)

export function FleetProvider({ children }: { children: ReactNode }) {
  const count = useConsole(s => s.params.roverCount)
  const seed = useConsole(s => s.params.seed)
  const engine = useMemo(() => {
    const { params, tether } = useConsole.getState()
    return new FleetEngine({ count, seed, range: params.communicationRange, cruise: params.speed, timeScale: params.timeScale, tether }, ARENA)
  }, [count, seed])
  const poses = useMemo(() => {
    const p = new FramePoses(engine.count)
    p.update(engine)
    return p
  }, [engine])
  const bus = useMemo(createViewBus, [])

  useEffect(() => {
    useConsole.getState().resetMission()
    const sync = () => {
      const { params, tether } = useConsole.getState()
      engine.configure({ cruise: params.speed, timeScale: params.timeScale, range: params.communicationRange, tether })
    }
    sync()
    return useConsole.subscribe(sync)
  }, [engine])

  const fleet = useMemo(() => ({ engine, poses, bus }), [engine, poses, bus])
  const scene = useMemo(() => ({ view: engine, poses, bus }), [engine, poses, bus])
  return (
    <FleetContext.Provider value={fleet}>
      <SceneProvider value={scene}>{children}</SceneProvider>
    </FleetContext.Provider>
  )
}

export function useFleet(): FleetContextValue {
  const value = useContext(FleetContext)
  if (!value) throw new Error('useFleet must be used inside FleetProvider')
  return value
}
