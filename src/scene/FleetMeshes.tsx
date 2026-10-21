import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import type { ThreeEvent } from '@react-three/fiber'
import * as THREE from 'three'
import { ARENA } from '../config/fleet'
import { useScene } from '../state/fleetContext'
import type { FramePoses } from '../state/fleetContext'
import { useConsole } from '../state/store'
import type { RoverLod, RoverModel, WheelPart } from './model/types'
import { SCENE, trustColor } from './palette'

interface WheelGroup { geometry: THREE.BufferGeometry; material: THREE.Material | THREE.Material[]; parts: WheelPart[] }

/** Wheels sharing geometry and material render as one instanced mesh with rovers × parts instances. */
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

const createScratch = () => ({
  root: new THREE.Matrix4(), part: new THREE.Matrix4(), spin: new THREE.Matrix4(), turn: new THREE.Matrix4(),
  q: new THREE.Quaternion(), p: new THREE.Vector3(), s: new THREE.Vector3(1, 1, 1), color: new THREE.Color(),
})
type Scratch = ReturnType<typeof createScratch>

/** Instanced meshes of one detail level; slots are packed from 0 so count trims the unused tail. */
class LevelMeshes {
  readonly bodies: (THREE.InstancedMesh | null)[] = []
  readonly wheels: (THREE.InstancedMesh | null)[] = []
  readonly groups: WheelGroup[]
  readonly rover: Int32Array
  used = 0
  constructor(readonly lod: RoverLod, capacity: number) {
    this.groups = groupWheels(lod.wheels)
    this.rover = new Int32Array(capacity)
  }
}

const Y_AXIS = new THREE.Vector3(0, 1, 0)
const DIM = new THREE.Color(SCENE.baseline).multiplyScalar(0.35)
const FAULT = new THREE.Color(SCENE.signal)
/** LOD switches only once the camera is this fraction past a threshold, so levels do not flicker at the boundary. */
const HYSTERESIS = 0.08

/**
 * The whole fleet in (body parts + wheel groups) × active detail levels + 1
 * draw calls, independent of fleet size. Each frame every rover is assigned
 * a detail level by camera distance and written into that level's next
 * slot; matrices come from the shared interpolated poses.
 */
export function FleetMeshes({ model, selectable = true, shadows = false }: { model: RoverModel; selectable?: boolean; shadows?: boolean }) {
  const { view, poses } = useScene()
  const n = view.count
  const select = useConsole(s => s.select)
  const levels = useMemo(() => model.lods.map(lod => new LevelMeshes(lod, n)), [model, n])
  // with several levels, shadows come from an invisible copy of the coarsest one: the shadow map needs no detail,
  // so the shadow pass draws about 6k triangles per rover instead of up to 80k
  const proxy = useMemo(() => (shadows && model.lods.length > 1 ? new LevelMeshes(model.lods.at(-1)!, n) : null), [model, n, shadows])
  const castFromLevels = shadows && !proxy
  const levelOf = useMemo(() => new Uint8Array(n), [n])
  const lights = useRef<THREE.InstancedMesh | null>(null)
  const lightMaterial = useMemo(() => new THREE.MeshBasicMaterial({ toneMapped: false }), [])
  // the shadow proxy draws nothing on screen; three.js only casts shadows from objects the camera renders
  const hiddenMaterial = useMemo(() => new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: false }), [])
  useEffect(() => () => { lightMaterial.dispose(); hiddenMaterial.dispose() }, [lightMaterial, hiddenMaterial])
  const scratch = useMemo(createScratch, [])

  useFrame(({ clock, camera }) => {
    const { selected } = useConsole.getState()
    const t = clock.elapsedTime
    for (const level of levels) level.used = 0
    for (let i = 0; i < n; i++) {
      scratch.p.set(poses.x[i], 0, poses.z[i])
      scratch.q.setFromAxisAngle(Y_AXIS, -poses.yaw[i])
      scratch.root.compose(scratch.p, scratch.q, scratch.s)
      levelOf[i] = pickLevel(levels, levelOf[i], camera.position.distanceTo(scratch.p))
      const level = levels[levelOf[i]]
      writeRover(level, level.used++, i, poses, scratch)
      if (proxy) writeRover(proxy, i, i, poses, scratch)
      const light = lights.current
      if (light) {
        light.setMatrixAt(i, scratch.root)
        const color = scratch.color
        if (view.faulted[i]) color.copy(FAULT).multiplyScalar(1.5 + 1.5 * Math.max(0, Math.sin(t * 8)))
        else if (view.activity[i] === 'parked') color.copy(DIM)
        else trustColor(view.trust[i], color).multiplyScalar(selectable && i === selected ? 3.2 : 2.2)
        light.setColorAt(i, color)
      }
    }
    if (proxy) proxy.used = n
    for (const level of proxy ? [...levels, proxy] : levels) commit(level)
    const light = lights.current
    if (light) {
      light.instanceMatrix.needsUpdate = true
      light.instanceColor!.needsUpdate = true
    }
  })

  // stable callbacks: a new ref callback would make React release and re-register the mesh on every render
  const proxyHandles = useMemo(() => proxy && {
    bodies: proxy.lod.body.map((_, k) => own(mesh => { proxy.bodies[k] = mesh })),
    wheels: proxy.groups.map((_, g) => own(mesh => { proxy.wheels[g] = mesh })),
  }, [proxy])
  const handles = useMemo(() => levels.map(level => ({
    bodies: level.lod.body.map((_, k) => own(mesh => { level.bodies[k] = mesh })),
    wheels: level.groups.map((_, g) => own(mesh => { level.wheels[g] = mesh })),
    click: (event: ThreeEvent<MouseEvent>) => {
      event.stopPropagation()
      if (event.instanceId !== undefined) select(level.rover[event.instanceId])
    },
  })), [levels, select])
  const lightRef = useMemo(() => own(mesh => {
    lights.current = mesh
    // instanceColor must exist before the first draw or the shader compiles without it
    if (!mesh) return
    for (let i = 0; i < n; i++) mesh.setColorAt(i, scratch.color.set(SCENE.foil))
    mesh.instanceColor!.setUsage(THREE.DynamicDrawUsage)
  }), [n, scratch])

  return (
    <group>
      {levels.map((level, l) => (
        <group key={l}>
          {level.lod.body.map((part, k) => (
            <instancedMesh key={`body-${k}`} args={[part.geometry, part.material, n]} frustumCulled={false}
              castShadow={castFromLevels} receiveShadow={shadows} onClick={selectable ? handles[l].click : undefined}
              ref={handles[l].bodies[k]} />
          ))}
          {level.groups.map((group, g) => (
            <instancedMesh key={`wheel-${g}`} args={[group.geometry, group.material, n * group.parts.length]} frustumCulled={false}
              castShadow={castFromLevels} receiveShadow={shadows} ref={handles[l].wheels[g]} />
          ))}
        </group>
      ))}
      {proxy && proxyHandles && (
        <group>
          {proxy.lod.body.map((part, k) => (
            <instancedMesh key={`shadow-body-${k}`} args={[part.geometry, hiddenMaterial, n]} frustumCulled={false} castShadow ref={proxyHandles.bodies[k]} />
          ))}
          {proxy.groups.map((group, g) => (
            <instancedMesh key={`shadow-wheel-${g}`} args={[group.geometry, hiddenMaterial, n * group.parts.length]} frustumCulled={false} castShadow ref={proxyHandles.wheels[g]} />
          ))}
        </group>
      )}
      <instancedMesh args={[model.lights, lightMaterial, n]} frustumCulled={false} ref={lightRef} />
    </group>
  )
}

/** Writes one rover's body and spinning wheels into a level's slot; scratch.root holds its pose. */
function writeRover(level: LevelMeshes, slot: number, i: number, poses: FramePoses, scratch: Scratch): void {
  level.rover[slot] = i
  for (const mesh of level.bodies) mesh?.setMatrixAt(slot, scratch.root)
  for (let g = 0; g < level.groups.length; g++) {
    const mesh = level.wheels[g]
    if (!mesh) continue
    const parts = level.groups[g].parts
    for (let k = 0; k < parts.length; k++) {
      const part = parts[k]
      const travel = part.side < 0 ? poses.left[i] : part.side > 0 ? poses.right[i] : (poses.left[i] + poses.right[i]) / 2
      // engine spin is ground distance / SDF wheel radius; rescale to this wheel's radius.
      // Rolling toward +x turns the wheel clockwise about +z, hence the minus sign.
      const angle = (travel * ARENA.wheelRadius) / part.radius
      scratch.part.makeTranslation(part.pivot.x, part.pivot.y, part.pivot.z)
      scratch.spin.makeRotationZ(-angle)
      scratch.part.premultiply(scratch.root).multiply(scratch.spin)
      if (part.yaw) scratch.part.multiply(scratch.turn.makeRotationY(part.yaw))
      mesh.setMatrixAt(slot * parts.length + k, scratch.part)
    }
  }
}

function commit(level: LevelMeshes): void {
  for (const mesh of level.bodies) if (mesh) { mesh.count = level.used; mesh.instanceMatrix.needsUpdate = true }
  level.wheels.forEach((mesh, g) => {
    if (!mesh) return
    mesh.count = level.used * level.groups[g].parts.length
    mesh.instanceMatrix.needsUpdate = true
  })
}

/** Nearest level whose range covers the distance, with hysteresis around the current level's edges. */
function pickLevel(levels: readonly LevelMeshes[], current: number, distance: number): number {
  if (levels.length === 1) return 0
  let level = Math.min(current, levels.length - 1)
  while (level < levels.length - 1 && distance > levels[level].lod.maxDistance * (1 + HYSTERESIS)) level++
  while (level > 0 && distance < levels[level - 1].lod.maxDistance * (1 - HYSTERESIS)) level--
  return level
}

/**
 * Ref callback that marks an instanced mesh's matrices dynamic and registers
 * it with the frame loop. The geometry and material belong to the rover
 * model; R3F frees only the per-instance buffers when the mesh unmounts.
 */
function own(register: (mesh: THREE.InstancedMesh | null) => void) {
  return (mesh: THREE.InstancedMesh | null) => {
    if (!mesh) return
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    register(mesh)
    return () => register(null)
  }
}
