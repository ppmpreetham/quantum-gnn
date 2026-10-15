import { useEffect } from 'react'
import type { FleetEngine } from '../core/fleet'
import { roverId, useConsole } from '../state/store'
import type { Telemetry } from '../state/store'

const TELEMETRY_HZ = 5

function snapshot(engine: FleetEngine): Telemetry {
  const faults = useConsole.getState().faults
  return {
    live: engine.liveSummary(),
    rovers: Array.from({ length: engine.count }, (_, i) => ({
      index: i, id: roverId(i), trust: engine.trust[i], reliability: engine.reliability[i],
      battery: engine.battery[i], compute: engine.compute[i], pathFactor: engine.pathFactor[i],
      task: engine.task[i], activity: engine.activity[i], speed: engine.v[i],
      x: engine.x[i], z: engine.z[i], yaw: engine.yaw[i], parent: engine.parent[i],
      fault: faults.find(f => f.rover === i)?.kind ?? null,
    })),
  }
}

/** Copies engine state into the store at 5 Hz: enough for panels, far below frame rate. */
export function useTelemetry(engine: FleetEngine): void {
  useEffect(() => {
    let last = Number.NaN
    const publish = () => {
      // unchanged state (paused, hidden tab) would only wake React and the paused render loop
      const key = engine.elapsed + engine.planVersion * 1e7 + useConsole.getState().faults.length * 1e9
      if (document.hidden || key === last) return
      last = key
      useConsole.getState().setTelemetry(snapshot(engine))
    }
    publish()
    const timer = setInterval(publish, 1000 / TELEMETRY_HZ)
    return () => clearInterval(timer)
  }, [engine])
}
