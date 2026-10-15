import { useEffect, useRef } from 'react'

/** Matches the backing store to CSS size × devicePixelRatio; returns false while hidden. */
export function fitCanvas(canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D): { width: number; height: number } | null {
  const width = canvas.clientWidth, height = canvas.clientHeight
  if (width === 0 || height === 0) return null
  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  const w = Math.round(width * dpr), h = Math.round(height * dpr)
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  return { width, height }
}

/**
 * Shared requestAnimationFrame loop for 2D views. Each view redraws at most
 * fps times a second and only when its version key changes, so idle panels
 * cost nothing and every view shares a single rAF callback.
 */
type Painter = { draw: () => void; key: () => number; interval: number; last: number; lastKey: number }
const painters = new Set<Painter>()
let frame = 0
function tick(now: number) {
  for (const p of painters) {
    if (now - p.last < p.interval) continue
    const key = p.key()
    if (key === p.lastKey) continue
    p.last = now
    p.lastKey = key
    p.draw()
  }
  frame = painters.size ? requestAnimationFrame(tick) : 0
}

export function usePainter(draw: () => void, key: () => number, fps: number): void {
  const latest = useRef({ draw, key })
  latest.current = { draw, key }
  useEffect(() => {
    const painter: Painter = {
      draw: () => latest.current.draw(), key: () => latest.current.key(),
      interval: 1000 / fps, last: 0, lastKey: Number.NaN,
    }
    painters.add(painter)
    if (!frame) frame = requestAnimationFrame(tick)
    return () => { painters.delete(painter) }
  }, [fps])
}

/** A rover as an oriented chevron; heading uses the scene convention (yaw turns +x toward +z). */
export function drawChevron(ctx: CanvasRenderingContext2D, x: number, y: number, yaw: number, size: number, fill: string, stroke?: string) {
  ctx.save()
  ctx.translate(x, y)
  ctx.rotate(yaw)
  ctx.beginPath()
  ctx.moveTo(size, 0)
  ctx.lineTo(-size * 0.7, size * 0.72)
  ctx.lineTo(-size * 0.35, 0)
  ctx.lineTo(-size * 0.7, -size * 0.72)
  ctx.closePath()
  ctx.fillStyle = fill
  ctx.fill()
  if (stroke) { ctx.lineWidth = 1.5; ctx.strokeStyle = stroke; ctx.stroke() }
  ctx.restore()
}
