import { useConsole } from '../state/store'
import { Segmented } from './controls'
import { ms } from './format'
import { CAMERA_OPTIONS } from './panels/ViewTab'

const STATUS = {
  connecting: 'Connecting to planner',
  online: 'Planner online',
  offline: 'Planner offline',
  paused: 'Planner paused',
} as const

function Mark() {
  return (
    <svg className="mark" viewBox="0 0 32 32" aria-hidden="true">
      <path d="M6 22 L16 8 L26 22 Z" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
      <path d="M16 8 L16 22" stroke="currentColor" strokeWidth="1.6" opacity="0.45" />
      <circle cx="16" cy="8" r="3.2" fill="var(--foil)" />
      <circle cx="6" cy="22" r="2.6" fill="currentColor" />
      <circle cx="26" cy="22" r="2.6" fill="currentColor" />
      <circle cx="16" cy="22" r="2.2" fill="var(--signal)" />
    </svg>
  )
}

export function TopBar() {
  const planner = useConsole(s => s.planner)
  const paused = useConsole(s => s.paused)
  const togglePause = useConsole(s => s.togglePause)
  const camera = useConsole(s => s.camera)
  const setCamera = useConsole(s => s.setCamera)
  const cycle = useConsole(s => s.history.at(-1)?.cycle ?? 0)

  return (
    <header className="panel topbar">
      <div className="brand">
        <Mark />
        <div>
          <h1>Fleet console</h1>
          <p>QUBO link pruning and trust-aware task allocation, live</p>
        </div>
      </div>
      <div className="topbar-center">
        <Segmented label="Camera" value={camera} options={CAMERA_OPTIONS} onChange={setCamera} />
      </div>
      <div className="topbar-right">
        <span className={`status ${planner.state}`} title={planner.error ?? undefined} role="status">
          <i aria-hidden="true" />
          {STATUS[planner.state]}
          {planner.state === 'online' && <small>{ms(planner.latencyMs)}</small>}
        </span>
        <span className="cycle" title="Planner cycles completed">Cycle <b>{cycle}</b></span>
        <button type="button" className="primary" onClick={togglePause} aria-pressed={paused} title="Space">
          {paused ? 'Resume' : 'Pause'}
        </button>
      </div>
    </header>
  )
}
