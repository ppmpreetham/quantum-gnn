import { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { useFleet } from '../state/fleetContext'
import { useConsole } from '../state/store'
import type { Tier } from '../state/store'

/** Advances the fleet and interpolates poses once per frame, before anything draws. */
export function FleetDriver() {
  const { engine, poses } = useFleet()
  useFrame((_, delta) => {
    if (!useConsole.getState().paused) engine.advance(delta)
    poses.update(engine)
  }, -3)
  return null
}

/** Places the DOM labels over their rovers by writing transforms only (no React renders). */
export function LabelProjector() {
  const { poses, bus } = useFleet()
  const size = useThree(s => s.size)
  const point = useMemo(() => new THREE.Vector3(), [])
  useFrame(({ camera }) => {
    const visible = useConsole.getState().overlays.labels
    for (let i = 0; i < poses.count; i++) {
      const el = bus.labels[i]
      if (!el) continue
      point.set(poses.x[i], 1.02, poses.z[i]).project(camera)
      const shown = visible && point.z < 1 && Math.abs(point.x) < 1.1 && Math.abs(point.y) < 1.1
      el.style.visibility = shown ? 'visible' : 'hidden'
      if (!shown) continue
      const x = (point.x * 0.5 + 0.5) * size.width
      const y = (-point.y * 0.5 + 0.5) * size.height
      el.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0)`
      el.style.zIndex = String(Math.round((1 - point.z) * 1e5))
    }
  })
  return null
}

/** Frame-wide draw calls and triangles, summed over every render pass, published once a second. */
export function PerfProbe({ tier }: { tier: Tier }) {
  const gl = useThree(s => s.gl)
  const tally = useRef({ frames: 0, time: 0 })
  useEffect(() => {
    gl.info.autoReset = false
    return () => { gl.info.autoReset = true }
  }, [gl])
  useFrame((_, delta) => {
    const t = tally.current
    t.frames++
    t.time += delta
    if (t.time >= 1) {
      useConsole.getState().setPerf({ fps: Math.round(t.frames / t.time), calls: gl.info.render.calls, triangles: gl.info.render.triangles, tier })
      t.frames = 0
      t.time = 0
    }
    gl.info.reset()
  }, -10)
  return null
}

/** With frameloop="demand" (paused), store changes such as a new selection must request a frame. */
export function InvalidateOnStore() {
  const invalidate = useThree(s => s.invalidate)
  useEffect(() => useConsole.subscribe(() => invalidate()), [invalidate])
  return null
}
