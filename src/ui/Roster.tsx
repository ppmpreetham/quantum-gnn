import { memo, useRef } from 'react'
import { LINK_KEPT } from '../core/fleet'
import { SCENE, trustCss } from '../scene/palette'
import { useFleet } from '../state/fleetContext'
import { useConsole } from '../state/store'
import type { RoverTelemetry } from '../state/store'
import { drawChevron, fitCanvas, usePainter } from './canvas2d'
import { Meter } from './controls'
import { ACTIVITY_LABEL, FAULT_LABEL, pct } from './format'

/** Heading-up radar centred on one rover, scaled to its radio range: what that rover can reach. */
function LocalRadar({ index }: { index: number }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const { engine, poses, bus } = useFleet()
  usePainter(() => {
    const el = canvas.current
    const ctx = el?.getContext('2d')
    if (!el || !ctx || index >= engine.count) return
    const size = fitCanvas(el, ctx)
    if (!size) return
    const { width, height } = size
    const c = width / 2
    const range = engine.radioRange
    const s = (c - 3) / (range * 1.15)
    const n = engine.count
    const ox = poses.x[index], oz = poses.z[index], yaw = poses.yaw[index]
    // rotate the world so this rover's heading points up the canvas
    const rot = -Math.PI / 2 - yaw
    const cos = Math.cos(rot), sin = Math.sin(rot)
    const X = (x: number, z: number) => c + ((x - ox) * cos - (z - oz) * sin) * s
    const Y = (x: number, z: number) => height / 2 + ((x - ox) * sin + (z - oz) * cos) * s

    ctx.clearRect(0, 0, width, height)
    ctx.fillStyle = SCENE.basalt
    ctx.beginPath(); ctx.arc(c, height / 2, c - 1, 0, Math.PI * 2); ctx.fill()
    ctx.save(); ctx.clip()
    ctx.strokeStyle = 'rgba(226,169,59,0.4)'
    ctx.setLineDash([2, 3])
    ctx.beginPath(); ctx.arc(c, height / 2, range * s, 0, Math.PI * 2); ctx.stroke()
    ctx.setLineDash([])
    for (let j = 0; j < n; j++) {
      if (j === index) continue
      const a = Math.min(index, j), b = Math.max(index, j)
      const kept = engine.planState[a * n + b] === LINK_KEPT
      const live = engine.inRange(index, j)
      const x = X(poses.x[j], poses.z[j]), y = Y(poses.x[j], poses.z[j])
      if (kept || live) {
        ctx.strokeStyle = kept ? (live ? SCENE.foil : SCENE.signal) : 'rgba(111,134,174,0.45)'
        ctx.lineWidth = kept ? 1.5 : 1
        ctx.beginPath(); ctx.moveTo(c, height / 2); ctx.lineTo(x, y); ctx.stroke()
      }
      ctx.fillStyle = live ? trustCss(engine.trust[j]) : 'rgba(236,238,240,0.25)'
      ctx.beginPath(); ctx.arc(x, y, 2.4, 0, Math.PI * 2); ctx.fill()
    }
    ctx.restore()
    drawChevron(ctx, c, height / 2, -Math.PI / 2, 5.5, trustCss(engine.trust[index]))
  }, () => bus.version + engine.planVersion * 1e6, 12)
  return <canvas ref={canvas} className="radar" aria-hidden="true" />
}

const RoverCard = memo(function RoverCard({ rover, selected }: { rover: RoverTelemetry; selected: boolean }) {
  const select = useConsole(s => s.select)
  const setCamera = useConsole(s => s.setCamera)
  const tone = rover.fault ? 'fault' : rover.activity === 'parked' ? 'parked' : rover.activity === 'yielding' ? 'yielding' : ''
  return (
    <li>
      <button type="button" className={`rover-card ${selected ? 'selected' : ''}`} aria-pressed={selected}
        onClick={() => select(rover.index)} onDoubleClick={() => { select(rover.index); setCamera('chase') }}
        title="Click to select, double-click to ride along">
        <LocalRadar index={rover.index} />
        <span className="rover-main">
          <span className="rover-head">
            <b>{rover.id}</b>
            {rover.index === 0 && <span className="tag root">Root</span>}
            {rover.fault && <span className="tag fault">{FAULT_LABEL[rover.fault]}</span>}
            <span className={`activity ${tone}`}>{ACTIVITY_LABEL[rover.activity]}</span>
          </span>
          <span className="rover-meter">
            <span>Trust</span>
            <Meter label={`Trust ${pct(rover.trust)}`} value={rover.trust} color={trustCss(rover.trust)} marker={rover.reliability} />
            <em>{rover.trust.toFixed(2)}</em>
          </span>
          <span className="rover-meter">
            <span>Battery</span>
            <Meter label={`Battery ${pct(rover.battery)}`} value={rover.battery} color={rover.battery < 0.15 ? 'var(--signal)' : 'var(--ink-2)'} />
            <em>{pct(rover.battery)}</em>
          </span>
          <span className="rover-task">{rover.task ? `Task ${rover.task.replace('task-', '')}` : 'No task'}{rover.pathFactor > 0 ? `, link ${rover.pathFactor.toFixed(2)}` : ', no route to root'}</span>
        </span>
      </button>
    </li>
  )
})

export function Roster() {
  const rovers = useConsole(s => s.telemetry?.rovers)
  const selected = useConsole(s => s.selected)
  return (
    <section className="roster" aria-label="Fleet roster">
      <h2>Fleet <span>{rovers?.length ?? 0} rovers</span></h2>
      <p className="roster-hint">Trust bar; the tick marks hidden reliability from the simulator.</p>
      <ul>
        {rovers?.map(r => <RoverCard key={r.index} rover={r} selected={r.index === selected} />)}
      </ul>
    </section>
  )
}
