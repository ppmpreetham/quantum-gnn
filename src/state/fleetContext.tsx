import { createContext, useContext, useEffect, useMemo } from 'react'
import type { ReactNode } from 'react'
import { ARENA } from '../config/fleet'
import { FleetEngine } from '../core/fleet'
import { useConsole } from './store'

/** Interpolated poses for the current frame, computed once and shared by every renderer. */
export class FramePoses {
  readonly x: Float32Array; readonly z: Float32Array; readonly yaw: Float32Array
  readonly left: Float32Array; readonly right: Float32Array
  private readonly scratch = { x: 0, z: 0, yaw: 0, left: 0, right: 0 }
  constructor(readonly count: number) {
    this.x = new Float32Array(count); this.z = new Float32Array(count); this.yaw = new Float32Array(count)
    this.left = new Float32Array(count); this.right = new Float32Array(count)
  }
  update(engine: FleetEngine): void {
    for (let i = 0; i < this.count; i++) {
      const p = engine.renderPose(i, this.scratch)
      this.x[i] = p.x; this.z[i] = p.z; this.yaw[i] = p.yaw; this.left[i] = p.left; this.right[i] = p.right
    }
  }
}

/** Camera state the 2D views read and the camera rig consumes, without React renders. */
export interface ViewBus {
  footprint: Float32Array
  camera: { x: number; z: number; tx: number; tz: number }
  version: number
  request: { x: number; z: number } | null
  labels: (HTMLElement | null)[]
}

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
  const bus = useMemo<ViewBus>(() => ({
    footprint: new Float32Array(8), camera: { x: 0, z: 0, tx: 0, tz: 0 }, version: 0, request: null, labels: [],
  }), [])

  useEffect(() => {
    useConsole.getState().resetMission()
    const sync = () => {
      const { params, tether } = useConsole.getState()
      engine.configure({ cruise: params.speed, timeScale: params.timeScale, range: params.communicationRange, tether })
    }
    sync()
    return useConsole.subscribe(sync)
  }, [engine])

  const value = useMemo(() => ({ engine, poses, bus }), [engine, poses, bus])
  return <FleetContext.Provider value={value}>{children}</FleetContext.Provider>
}

export function useFleet(): FleetContextValue {
  const value = useContext(FleetContext)
  if (!value) throw new Error('useFleet must be used inside FleetProvider')
  return value
}
