import type { ReactNode } from 'react'
import { useKeyboard } from '../hooks/useKeyboard'
import { usePlannerLoop } from '../hooks/usePlannerLoop'
import { useTelemetry } from '../hooks/useTelemetry'
import { FleetProvider, useFleet } from '../state/fleetContext'
import { useConsole } from '../state/store'

function MissionLoops() {
  const { engine } = useFleet()
  usePlannerLoop(engine)
  useTelemetry(engine)
  useKeyboard()
  return null
}

function PlannerNotice() {
  const planner = useConsole(s => s.planner)
  if (planner.state !== 'offline') return null
  return (
    <div className="planner-notice" role="status">
      <b>The planner is not answering</b>
      <p>{planner.error} Rovers keep driving; links, trust and tasks resume once it is back. Run <code>npm run dev</code> from the project root to start both servers.</p>
    </div>
  )
}

/**
 * The running mission shared by the live modes: the engine, the planner
 * loop and telemetry stay mounted while the user switches between them.
 */
export function LiveMission({ children }: { children: ReactNode }) {
  return (
    <FleetProvider>
      <MissionLoops />
      {children}
      <PlannerNotice />
    </FleetProvider>
  )
}
