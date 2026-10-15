import { useRef } from 'react'
import type { PointerEvent } from 'react'
import { ARENA } from '../config/fleet'
import { LINK_KEPT } from '../core/fleet'
import { trustCss, SCENE } from '../scene/palette'
import { useFleet } from '../state/fleetContext'
import { useConsole } from '../state/store'
import { drawChevron, fitCanvas, usePainter } from './canvas2d'

const WORLD_RADIUS = ARENA.radius + 0.6
const PICK_PX = 14

/** North-up overview in the top camera's orientation: +x right, +z down. */
export function Minimap() {
  const canvas = useRef<HTMLCanvasElement>(null)
  const { engine, poses, bus } = useFleet()
  const drag = useRef(false)
  const view = useRef({ cx: 0, cy: 0, scale: 1 })

  usePainter(() => {
    const el = canvas.current
    const ctx = el?.getContext('2d')
    if (!el || !ctx) return
    const size = fitCanvas(el, ctx)
    if (!size) return
    const { width, height } = size
    const cx = width / 2, cy = height / 2
    const s = Math.min(width, height) / 2 / WORLD_RADIUS
    view.current = { cx, cy, scale: s }
    const X = (x: number) => cx + x * s, Y = (z: number) => cy + z * s
    const { selected, overlays } = useConsole.getState()
    const n = engine.count

    ctx.clearRect(0, 0, width, height)
    ctx.fillStyle = SCENE.basalt
    ctx.beginPath(); ctx.arc(cx, cy, ARENA.radius * s, 0, Math.PI * 2); ctx.fill()
    ctx.strokeStyle = 'rgba(226,169,59,0.45)'
    ctx.lineWidth = 1
    ctx.stroke()
    ctx.strokeStyle = 'rgba(255,255,255,0.06)'
    for (let r = 2; r < ARENA.radius; r += 2) { ctx.beginPath(); ctx.arc(cx, cy, r * s, 0, Math.PI * 2); ctx.stroke() }
    ctx.beginPath(); ctx.moveTo(X(-ARENA.radius), cy); ctx.lineTo(X(ARENA.radius), cy); ctx.moveTo(cx, Y(-ARENA.radius)); ctx.lineTo(cx, Y(ARENA.radius)); ctx.stroke()

    // the camera's ground footprint, like an RTS minimap
    const f = bus.footprint
    ctx.beginPath()
    ctx.moveTo(X(f[0]), Y(f[1]))
    for (let k = 1; k < 4; k++) ctx.lineTo(X(f[k * 2]), Y(f[k * 2 + 1]))
    ctx.closePath()
    ctx.fillStyle = 'rgba(236,238,240,0.07)'
    ctx.strokeStyle = 'rgba(236,238,240,0.5)'
    ctx.fill(); ctx.stroke()

    const range = engine.radioRange
    ctx.setLineDash([3, 4])
    ctx.strokeStyle = 'rgba(226,169,59,0.5)'
    ctx.beginPath(); ctx.arc(X(poses.x[selected]), Y(poses.z[selected]), range * s, 0, Math.PI * 2); ctx.stroke()
    ctx.setLineDash([])

    for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) {
      const kept = engine.planState[a * n + b] === LINK_KEPT
      const live = engine.inRange(a, b)
      if (!kept && !(live && overlays.candidates)) continue
      ctx.strokeStyle = kept ? (live ? SCENE.foil : SCENE.signal) : 'rgba(111,134,174,0.35)'
      ctx.lineWidth = kept ? 1.6 : 1
      ctx.beginPath(); ctx.moveTo(X(poses.x[a]), Y(poses.z[a])); ctx.lineTo(X(poses.x[b]), Y(poses.z[b])); ctx.stroke()
    }
    if (overlays.beacons) {
      for (const beacon of engine.beacons) {
        ctx.fillStyle = beacon.state === 'enroute' ? 'rgba(236,238,240,0.8)' : beacon.success ? SCENE.mint : SCENE.signal
        ctx.beginPath(); ctx.arc(X(beacon.x), Y(beacon.z), 2.6, 0, Math.PI * 2); ctx.fill()
      }
    }
    for (let i = 0; i < n; i++) {
      const x = X(poses.x[i]), y = Y(poses.z[i])
      if (i === 0) { ctx.strokeStyle = SCENE.foil; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.stroke() }
      drawChevron(ctx, x, y, poses.yaw[i], i === selected ? 8 : 6.5, trustCss(engine.trust[i]), i === selected ? SCENE.signal : undefined)
    }
  }, () => bus.version + engine.planVersion * 1e6 + useConsole.getState().selected * 1e9, 30)

  const toWorld = (event: { clientX: number; clientY: number; currentTarget: HTMLCanvasElement }) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const { cx, cy, scale } = view.current
    return { x: (event.clientX - rect.left - cx) / scale, z: (event.clientY - rect.top - cy) / scale }
  }
  const nearest = (x: number, z: number) => {
    let best = -1, bestD = PICK_PX / view.current.scale
    for (let i = 0; i < engine.count; i++) {
      const d = Math.hypot(poses.x[i] - x, poses.z[i] - z)
      if (d < bestD) { bestD = d; best = i }
    }
    return best
  }
  const onDown = (event: PointerEvent<HTMLCanvasElement>) => {
    const p = toWorld(event)
    const hit = nearest(p.x, p.z)
    if (hit >= 0) { useConsole.getState().select(hit); return }
    drag.current = true
    event.currentTarget.setPointerCapture(event.pointerId)
    bus.request = p
  }
  const onMove = (event: PointerEvent<HTMLCanvasElement>) => { if (drag.current) bus.request = toWorld(event) }
  const onUp = () => { drag.current = false }

  return (
    <figure className="minimap">
      <canvas ref={canvas} aria-label="Overhead map of the fleet. Click a rover to select it, or click and drag to move the camera."
        role="img" onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}
        onDoubleClick={event => { const p = toWorld(event); if (nearest(p.x, p.z) >= 0) useConsole.getState().setCamera('chase') }} />
    </figure>
  )
}
