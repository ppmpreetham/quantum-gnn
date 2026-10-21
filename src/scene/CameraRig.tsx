import { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { ARENA } from '../config/fleet'
import { dampFactor } from '../core/math'
import { useScene } from '../state/fleetContext'
import { useConsole } from '../state/store'

export interface CameraHome { position: THREE.Vector3; target: THREE.Vector3 }
const HOME: CameraHome = { position: new THREE.Vector3(9.6, 8.4, 10.8), target: new THREE.Vector3(0, 0, 0) }
const TOP = { position: new THREE.Vector3(0, 19.5, 0.01), target: new THREE.Vector3(0, 0, 0) }
const FOLLOW_OFFSET = new THREE.Vector3(-3.2, 3.1, 3.8)
const CORNERS = [[-1, -1], [1, -1], [1, 1], [-1, 1]] as const
const MAX_TARGET_RADIUS = ARENA.radius + 2

/**
 * Orbit, top, follow and chase views over one OrbitControls. View changes
 * glide with frame-rate independent damping; touching the controls cancels
 * the glide so the user is never fought for the camera.
 */
export interface CameraRigProps {
  /** Where the orbit view returns to. */
  home?: CameraHome
  minDistance?: number
  /** Slow turntable orbit for unattended demos. */
  autoRotate?: boolean
}

export function CameraRig({ home = HOME, minDistance = 2.2, autoRotate = false }: CameraRigProps) {
  const controls = useRef<OrbitControlsImpl>(null)
  const camera = useThree(s => s.camera)
  const invalidate = useThree(s => s.invalidate)
  const { poses, bus } = useScene()
  const mode = useConsole(s => s.camera)
  const nonce = useConsole(s => s.cameraNonce)
  const selected = useConsole(s => s.selected)
  const glide = useRef<{ position: THREE.Vector3; target: THREE.Vector3 } | null>(null)
  const anchor = useRef<{ x: number; z: number } | null>(null)
  const scratch = useMemo(() => ({
    ndc: new THREE.Vector3(), dir: new THREE.Vector3(), heading: new THREE.Vector3(),
    chasePos: new THREE.Vector3(), chaseTarget: new THREE.Vector3(), shift: new THREE.Vector3(),
  }), [])

  useEffect(() => {
    if (mode === 'orbit') glide.current = { position: home.position.clone(), target: home.target.clone() }
    else if (mode === 'top') glide.current = { position: TOP.position.clone(), target: TOP.target.clone() }
    else glide.current = null
    invalidate()
  }, [mode, nonce, invalidate, home])

  useEffect(() => {
    const i = Math.min(selected, poses.count - 1)
    anchor.current = { x: poses.x[i], z: poses.z[i] }
    if (mode !== 'follow') return
    const target = controls.current?.target
    const focus = new THREE.Vector3(poses.x[i], 0, poses.z[i])
    const offset = target ? camera.position.clone().sub(target) : FOLLOW_OFFSET.clone()
    offset.setLength(THREE.MathUtils.clamp(offset.length(), 4, 7))
    if (offset.y < 1.5) offset.copy(FOLLOW_OFFSET)
    glide.current = { position: focus.clone().add(offset), target: focus }
    invalidate()
  }, [mode, nonce, selected, poses, camera, invalidate])

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
    const dt = Math.min(delta, 0.1)
    const i = Math.min(useConsole.getState().selected, poses.count - 1)
    const rx = poses.x[i], rz = poses.z[i]

    if (bus.request) {
      scratch.shift.set(bus.request.x - c.target.x, 0, bus.request.z - c.target.z)
      glide.current = { position: camera.position.clone().add(scratch.shift), target: c.target.clone().add(scratch.shift) }
      bus.request = null
      if (mode === 'follow' || mode === 'chase') useConsole.getState().setCamera('orbit')
    }

    if (mode === 'follow' && anchor.current) {
      // translate camera and target with the rover; the user's orbit offset survives
      const dx = rx - anchor.current.x, dz = rz - anchor.current.z
      camera.position.x += dx; camera.position.z += dz
      c.target.x += dx; c.target.z += dz
      if (glide.current) { glide.current.position.x += dx; glide.current.position.z += dz; glide.current.target.x += dx; glide.current.target.z += dz }
      anchor.current.x = rx; anchor.current.z = rz
    } else if (mode === 'chase') {
      const yaw = poses.yaw[i]
      scratch.heading.set(Math.cos(yaw), 0, Math.sin(yaw))
      scratch.chasePos.set(rx, 1.75, rz).addScaledVector(scratch.heading, -3.4)
      scratch.chaseTarget.set(rx, 0.35, rz).addScaledVector(scratch.heading, 1.8)
      const k = dampFactor(0.18, dt)
      camera.position.lerp(scratch.chasePos, k)
      c.target.lerp(scratch.chaseTarget, k)
    }

    if (glide.current) {
      const k = dampFactor(0.16, dt)
      camera.position.lerp(glide.current.position, k)
      c.target.lerp(glide.current.target, k)
      if (camera.position.distanceToSquared(glide.current.position) < 1e-4) glide.current = null
      invalidate()
    }

    const r = Math.hypot(c.target.x, c.target.z)
    if (r > MAX_TARGET_RADIUS) {
      const s = MAX_TARGET_RADIUS / r
      camera.position.x -= c.target.x * (1 - s); camera.position.z -= c.target.z * (1 - s)
      c.target.x *= s; c.target.z *= s
    }

    // ground footprint of the view frustum, for the minimap
    for (let k = 0; k < 4; k++) {
      scratch.ndc.set(CORNERS[k][0], CORNERS[k][1], 0.5).unproject(camera)
      scratch.dir.copy(scratch.ndc).sub(camera.position).normalize()
      let t = scratch.dir.y < -1e-3 ? -camera.position.y / scratch.dir.y : Infinity
      // corners above the horizon never meet the ground; clip them at the fog distance
      if (!Number.isFinite(t) || t > 60) {
        scratch.dir.y = 0
        scratch.dir.normalize()
        t = 60
      }
      bus.footprint[k * 2] = camera.position.x + scratch.dir.x * t
      bus.footprint[k * 2 + 1] = camera.position.z + scratch.dir.z * t
    }
    bus.camera.x = camera.position.x; bus.camera.z = camera.position.z
    bus.camera.tx = c.target.x; bus.camera.tz = c.target.z
    bus.version++
  }, -2)

  return (
    <OrbitControls ref={controls} makeDefault enableDamping dampingFactor={0.1}
      minDistance={minDistance} maxDistance={34} maxPolarAngle={Math.PI / 2 - 0.06}
      autoRotate={autoRotate && mode === 'orbit'} autoRotateSpeed={0.35}
      enableRotate={mode !== 'chase'} enablePan={mode === 'orbit' || mode === 'top'} />
  )
}
