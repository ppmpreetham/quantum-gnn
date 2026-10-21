import { useEffect, useMemo, useRef } from 'react'
import { createPortal, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { dampFactor } from '../../core/math'
import { FleetLayer } from '../../scene/FleetLayer'
import type { RoverModel } from '../../scene/model/types'
import { SceneCanvas } from '../../scene/SceneCanvas'
import { ArenaBackdrop } from '../../scene/Stage'
import { FramePoses, SceneProvider, createViewBus } from '../../state/fleetContext'
import type { ScriptedMotion, ScriptedSide } from './ScriptedFleet'
import { useShowcase } from './store'
import { GLIDE_SECONDS } from './useShowcaseRun'

/** DOM label elements for both halves, positioned every frame without React renders. */
export interface SplitLabels { left: (HTMLElement | null)[]; right: (HTMLElement | null)[] }

const OVERVIEW = { position: new THREE.Vector3(0, 13.5, 13), target: new THREE.Vector3(0, 0, 0) }
const FOCUS_OFFSET = new THREE.Vector3(3.8, 5.2, 5.4)

/** Advances the shared scripted motion and both sides' task columns, then interpolates poses once for both. */
function Driver({ motion, sides, poses }: { motion: ScriptedMotion; sides: ScriptedSide[]; poses: FramePoses }) {
  useFrame((_, delta) => {
    const { status, speed } = useShowcase.getState()
    if (status !== 'paused') motion.advance(Math.min(delta, 0.1), speed / GLIDE_SECONDS)
    for (const side of sides) side.advance()
    poses.update(sides[0])
  }, -3)
  return null
}

/** Each half gets the camera at half the canvas width. */
function HalfAspect() {
  useFrame(({ camera, size }) => {
    const perspective = camera as THREE.PerspectiveCamera
    const aspect = size.width / 2 / size.height
    if (Math.abs(perspective.aspect - aspect) > 1e-6) { perspective.aspect = aspect; perspective.updateProjectionMatrix() }
  }, -1)
  return null
}

/** Renders the two scenes side by side with one camera, so both halves frame the arena identically. */
function SplitRenderer({ scenes }: { scenes: [THREE.Scene, THREE.Scene] }) {
  useFrame(({ gl, camera, size }) => {
    const half = Math.floor(size.width / 2)
    gl.setScissorTest(true)
    gl.setViewport(0, 0, half, size.height); gl.setScissor(0, 0, half, size.height)
    gl.render(scenes[0], camera)
    gl.setViewport(half, 0, size.width - half, size.height); gl.setScissor(half, 0, size.width - half, size.height)
    gl.render(scenes[1], camera)
    gl.setScissorTest(false)
  }, 1)
  return null
}

/** Places the rover labels of both halves (the camera aspect is already the half-width one). */
function LabelProjector({ poses, labels }: { poses: FramePoses; labels: SplitLabels }) {
  const point = useMemo(() => new THREE.Vector3(), [])
  useFrame(({ camera, size }) => {
    const half = size.width / 2
    for (let i = 0; i < poses.count; i++) {
      point.set(poses.x[i], 1.02, poses.z[i]).project(camera)
      const shown = point.z < 1 && Math.abs(point.x) < 1.05 && Math.abs(point.y) < 1.1
      const x = (point.x * 0.5 + 0.5) * half, y = (-point.y * 0.5 + 0.5) * size.height
      for (const [list, offset] of [[labels.left, 0], [labels.right, half]] as const) {
        const el = list[i]
        if (!el) continue
        el.style.visibility = shown ? 'visible' : 'hidden'
        if (shown) el.style.transform = `translate3d(${(x + offset).toFixed(1)}px, ${y.toFixed(1)}px, 0)`
      }
    }
  })
  return null
}

/**
 * Directed camera: glides to frame the rovers a highlight is about, then
 * back to the overview. Touching the controls hands the camera to the user
 * until the next highlight.
 */
function DirectedCamera({ poses }: { poses: FramePoses }) {
  const controls = useRef<OrbitControlsImpl>(null)
  const camera = useThree(s => s.camera)
  const glide = useRef<{ position: THREE.Vector3; target: THREE.Vector3 } | null>({ position: OVERVIEW.position.clone(), target: OVERVIEW.target.clone() })
  const seen = useRef(0)
  useEffect(() => {
    const c = controls.current
    if (!c) return
    const cancel = () => { glide.current = null }
    c.addEventListener('start', cancel)
    return () => c.removeEventListener('start', cancel)
  }, [])
  useFrame((_, delta) => {
    const c = controls.current
    if (!c) return
    const { results, directed } = useShowcase.getState()
    if (results.length !== seen.current) {
      seen.current = results.length
      const latest = results.at(-1)
      if (!directed) glide.current = null
      else if (latest && latest.verdict !== 'tie' && latest.focus.length) {
        let x = 0, z = 0
        for (const i of latest.focus) { x += poses.x[i]; z += poses.z[i] }
        const target = new THREE.Vector3(x / latest.focus.length, 0.2, z / latest.focus.length)
        glide.current = { position: target.clone().add(FOCUS_OFFSET), target }
      } else {
        glide.current = { position: OVERVIEW.position.clone(), target: OVERVIEW.target.clone() }
      }
    }
    if (glide.current) {
      const k = dampFactor(0.45, Math.min(delta, 0.1))
      camera.position.lerp(glide.current.position, k)
      c.target.lerp(glide.current.target, k)
      if (camera.position.distanceToSquared(glide.current.position) < 1e-4) glide.current = null
    }
  }, -2)
  return <OrbitControls ref={controls} makeDefault enableDamping dampingFactor={0.1} minDistance={3} maxDistance={34} maxPolarAngle={Math.PI / 2 - 0.08} />
}

interface SplitStageProps {
  model: RoverModel
  motion: ScriptedMotion
  ours: ScriptedSide
  baseline: ScriptedSide
  labels: SplitLabels
  onContextLost: (lost: boolean) => void
}

/** One canvas, two scenes: our pipeline on the left, the baseline on the right, identical rovers and camera. */
export function SplitStage({ model, motion, ours, baseline, labels, onContextLost }: SplitStageProps) {
  const scenes = useMemo<[THREE.Scene, THREE.Scene]>(() => [new THREE.Scene(), new THREE.Scene()], [])
  const poses = useMemo(() => {
    const p = new FramePoses(motion.count)
    p.update(ours)
    return p
  }, [motion, ours])
  const bus = useMemo(createViewBus, [])
  const left = useMemo(() => ({ view: ours, poses, bus }), [ours, poses, bus])
  const right = useMemo(() => ({ view: baseline, poses, bus }), [baseline, poses, bus])
  return (
    <SceneCanvas onContextLost={onContextLost} paused={false} camera={{ position: OVERVIEW.position.toArray(), fov: 42, near: 0.1, far: 140 }}>
      <Driver motion={motion} sides={[ours, baseline]} poses={poses} />
      <HalfAspect />
      <SceneProvider value={left}>{createPortal(<><ArenaBackdrop /><FleetLayer model={model} selection={false} /></>, scenes[0])}</SceneProvider>
      <SceneProvider value={right}>{createPortal(<><ArenaBackdrop /><FleetLayer model={model} selection={false} /></>, scenes[1])}</SceneProvider>
      <SplitRenderer scenes={scenes} />
      <LabelProjector poses={poses} labels={labels} />
      <DirectedCamera poses={poses} />
    </SceneCanvas>
  )
}
