import { useEffect, useMemo } from 'react'
import type { ThreeEvent } from '@react-three/fiber'
import { ARENA } from '../config/fleet'
import { drivableRadius } from '../core/arena'
import { useScene } from '../state/fleetContext'
import { useConsole } from '../state/store'

/** A press that moves further than this many pixels is an orbit drag, not a click. */
const CLICK_SLOP = 4
const REACH = drivableRadius(ARENA) - 0.2

/**
 * An invisible disc over the drivable arena that turns clicks into tasks
 * while placement is set to click. R3F raycasts it like any mesh, so rovers
 * in front of it still take the click (they stop propagation), and the hover
 * point drives the ground reticle through the view bus without a re-render.
 */
export function TaskPicker() {
  const { bus } = useScene()
  const placement = useConsole(s => s.taskPlacement)
  const active = placement === 'click'
  const handlers = useMemo(() => ({
    onClick: (event: ThreeEvent<MouseEvent>) => {
      if (event.delta > CLICK_SLOP) return
      event.stopPropagation()
      useConsole.getState().queueTask(round(event.point.x), round(event.point.z))
    },
    onPointerMove: (event: ThreeEvent<PointerEvent>) => {
      bus.pointer.x = event.point.x
      bus.pointer.z = event.point.z
      bus.pointer.active = true
    },
    onPointerOut: () => { bus.pointer.active = false },
  }), [bus])
  useEffect(() => {
    if (!active) return
    return () => { bus.pointer.active = false }
  }, [active, bus])
  if (!active) return null
  return (
    <mesh rotation-x={-Math.PI / 2} position-y={0.001} visible={false} {...handlers}>
      <circleGeometry args={[REACH, 64]} />
    </mesh>
  )
}

const round = (v: number) => Math.round(v * 1e4) / 1e4
