import { Component, useState } from 'react'
import type { DragEvent, ReactNode } from 'react'
import { useKeyboard } from './hooks/useKeyboard'
import { usePlannerLoop } from './hooks/usePlannerLoop'
import { useRoverModel } from './hooks/useRoverModel'
import { useTelemetry } from './hooks/useTelemetry'
import { Stage } from './scene/Stage'
import { FleetProvider, useFleet } from './state/fleetContext'
import { useConsole } from './state/store'
import { LabelLayer } from './ui/LabelLayer'
import { Minimap } from './ui/Minimap'
import { MissionStrip } from './ui/MissionStrip'
import { MissionPanel } from './ui/panels/MissionPanel'
import { Roster } from './ui/Roster'
import { TopBar } from './ui/TopBar'

class Boundary extends Component<{ children: ReactNode; title: string; advice: string }, { error: string | null }> {
  state = { error: null as string | null }
  static getDerivedStateFromError(error: unknown) { return { error: error instanceof Error ? error.message : 'Unknown error' } }
  render() {
    if (!this.state.error) return this.props.children
    return (
      <div className="stage-notice" role="alert">
        <b>{this.props.title}</b>
        <p>{this.state.error}. {this.props.advice}</p>
        <button type="button" className="primary" onClick={() => window.location.reload()}>Reload</button>
      </div>
    )
  }
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

function Console() {
  const { engine } = useFleet()
  usePlannerLoop(engine)
  useTelemetry(engine)
  useKeyboard()
  const { model, controls } = useRoverModel()
  const [contextLost, setContextLost] = useState(false)
  const [dragging, setDragging] = useState(false)

  const onDrop = (event: DragEvent) => {
    event.preventDefault()
    setDragging(false)
    const file = event.dataTransfer.files[0]
    if (file) controls.onFile(file)
  }

  return (
    <div className="console">
      <main className="stage" onDragOver={event => { event.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)} onDrop={onDrop}>
        <Boundary title="The 3D view stopped" advice="Check that hardware acceleration is on, then reload. The planner and panels keep running.">
          <Stage model={model} onContextLost={setContextLost} />
        </Boundary>
        <LabelLayer />
        {contextLost && (
          <div className="stage-notice" role="alert">
            <b>Graphics context lost</b>
            <p>The browser reclaimed the GPU. The view comes back on its own when it is restored, or reload now.</p>
            <button type="button" className="primary" onClick={() => window.location.reload()}>Reload</button>
          </div>
        )}
        {dragging && <div className="drop-hint">Drop a .glb to replace the rover model</div>}
      </main>
      <TopBar />
      <MissionPanel model={controls} />
      <aside className="panel fleet-column" aria-label="Fleet map and roster">
        <Minimap />
        <Roster />
      </aside>
      <MissionStrip />
      <PlannerNotice />
    </div>
  )
}

export function App() {
  return (
    <Boundary title="The console stopped" advice="Reload the page; if it happens again, the browser console has the details.">
      <FleetProvider>
        <Console />
      </FleetProvider>
    </Boundary>
  )
}
