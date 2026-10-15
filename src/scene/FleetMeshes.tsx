import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import type { ThreeEvent } from '@react-three/fiber'
import * as THREE from 'three'
import { ARENA } from '../config/fleet'
import { useFleet } from '../state/fleetContext'
import { useConsole } from '../state/store'
import type { RoverModel, WheelPart } from './model/types'
import { SCENE, trustColor } from './palette'

interface WheelGroup { geometry: THREE.BufferGeometry; material: THREE.Material | THREE.Material[]; parts: WheelPart[] }

/** Wheels sharing geometry and material render as one instanced mesh with parts × rovers instances. */
function groupWheels(wheels: WheelPart[]): WheelGroup[] {
  const groups = new Map<string, WheelGroup>()
  for (const part of wheels) {
    const materials = Array.isArray(part.material) ? part.material : [part.material]
    const key = `${part.geometry.uuid}:${materials.map(m => m.uuid).join(',')}`
    const group = groups.get(key) ?? { geometry: part.geometry, material: part.material, parts: [] }
    group.parts.push(part)
    groups.set(key, group)
  }
  return [...groups.values()]
}

const Y_AXIS = new THREE.Vector3(0, 1, 0)
const DIM = new THREE.Color(SCENE.baseline).multiplyScalar(0.35)
const FAULT = new THREE.Color(SCENE.signal)

/**
 * The whole fleet in (body parts + wheel groups + 1) draw calls, independent
 * of fleet size. Matrices are rebuilt from the shared interpolated poses.
 */
export function FleetMeshes({ model }: { model: RoverModel }) {
  const { engine, poses } = useFleet()
  const n = engine.count
  const select = useConsole(s => s.select)
  const bodies = useRef<(THREE.InstancedMesh | null)[]>([])
  const wheelMeshes = useRef<(THREE.InstancedMesh | null)[]>([])
  const lights = useRef<THREE.InstancedMesh>(null)
  const wheelGroups = useMemo(() => groupWheels(model.wheels), [model])
  const lightMaterial = useMemo(() => new THREE.MeshBasicMaterial({ toneMapped: false }), [])
  useEffect(() => () => lightMaterial.dispose(), [lightMaterial])
  const scratch = useMemo(() => ({
    root: new THREE.Matrix4(), part: new THREE.Matrix4(), spin: new THREE.Matrix4(),
    q: new THREE.Quaternion(), p: new THREE.Vector3(), s: new THREE.Vector3(1, 1, 1), color: new THREE.Color(),
  }), [])

  useEffect(() => {
    // instanceColor must exist before the first draw or the shader compiles without it
    const mesh = lights.current
    if (!mesh) return
    for (let i = 0; i < n; i++) mesh.setColorAt(i, scratch.color.set(SCENE.foil))
    mesh.instanceColor!.setUsage(THREE.DynamicDrawUsage)
  }, [n, scratch, model])

  useFrame(({ clock }) => {
    const { selected, faults } = useConsole.getState()
    const t = clock.elapsedTime
    for (let i = 0; i < n; i++) {
      scratch.p.set(poses.x[i], 0, poses.z[i])
      scratch.q.setFromAxisAngle(Y_AXIS, -poses.yaw[i])
      scratch.root.compose(scratch.p, scratch.q, scratch.s)
      for (const mesh of bodies.current) mesh?.setMatrixAt(i, scratch.root)
      lights.current?.setMatrixAt(i, scratch.root)
      for (let g = 0; g < wheelGroups.length; g++) {
        const mesh = wheelMeshes.current[g]
        const parts = wheelGroups[g].parts
        if (!mesh) continue
        for (let k = 0; k < parts.length; k++) {
          const part = parts[k]
          const travel = part.side < 0 ? poses.left[i] : part.side > 0 ? poses.right[i] : (poses.left[i] + poses.right[i]) / 2
          // engine spin is ground distance / SDF wheel radius; rescale to this wheel's radius.
          // Rolling toward +x turns the wheel clockwise about +z, hence the minus sign.
          const angle = (travel * ARENA.wheelRadius) / part.radius
          scratch.part.makeTranslation(part.pivot.x, part.pivot.y, part.pivot.z)
          scratch.spin.makeRotationZ(-angle)
          scratch.part.premultiply(scratch.root).multiply(scratch.spin)
          mesh.setMatrixAt(k * n + i, scratch.part)
        }
      }
      if (lights.current) {
        const fault = faults.some(f => f.rover === i)
        const color = scratch.color
        if (fault) color.copy(FAULT).multiplyScalar(1.5 + 1.5 * Math.max(0, Math.sin(t * 8)))
        else if (engine.activity[i] === 'parked') color.copy(DIM)
        else trustColor(engine.trust[i], color).multiplyScalar(i === selected ? 3.2 : 2.2)
        lights.current.setColorAt(i, color)
      }
    }
    for (const mesh of bodies.current) if (mesh) mesh.instanceMatrix.needsUpdate = true
    for (const mesh of wheelMeshes.current) if (mesh) mesh.instanceMatrix.needsUpdate = true
    if (lights.current) {
      lights.current.instanceMatrix.needsUpdate = true
      lights.current.instanceColor!.needsUpdate = true
    }
  })

  const onClick = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation()
    if (event.instanceId !== undefined) select(event.instanceId)
  }
  const dynamic = (mesh: THREE.InstancedMesh | null) => mesh?.instanceMatrix.setUsage(THREE.DynamicDrawUsage)

  return (
    <group>
      {model.body.map((part, k) => (
        <instancedMesh key={`body-${k}`} args={[part.geometry, part.material, n]} frustumCulled={false} onClick={onClick}
          ref={mesh => { bodies.current[k] = mesh; dynamic(mesh) }} />
      ))}
      {wheelGroups.map((group, g) => (
        <instancedMesh key={`wheel-${g}`} args={[group.geometry, group.material, n * group.parts.length]} frustumCulled={false}
          ref={mesh => { wheelMeshes.current[g] = mesh; dynamic(mesh) }} />
      ))}
      <instancedMesh ref={mesh => { (lights as { current: THREE.InstancedMesh | null }).current = mesh; dynamic(mesh) }}
        args={[model.lights, lightMaterial, n]} frustumCulled={false} />
    </group>
  )
}
