import * as THREE from 'three'
import type { GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js'
import type { RoverLod, RoverModel, WheelPart } from '../../scene/model/types'
import { bakedGeometry, meshesByName } from './gltfGeometry'

/** Camera distance (m) up to which each level is used; the asset pipeline writes three levels. */
const LOD_RANGES = [5, 13, Infinity]

interface RoverExtras {
  wheels: { side: -1 | 1; pivot: [number, number, number]; yaw: number }[]
  wheelRadius: number
  beacon: [number, number, number]
}

/**
 * Turns the processed NASA Perseverance asset into the fleet renderer's
 * rover model: three levels of detail of one atlased body plus a single
 * wheel mesh placed at six pivots (the right-hand wheels turned half a turn).
 */
export function roverFromGltf(gltf: GLTF): RoverModel {
  gltf.scene.updateMatrixWorld(true)
  const extras = gltf.scene.userData as RoverExtras
  const meshes = meshesByName(gltf.scene)
  const created: THREE.BufferGeometry[] = []
  const bake = (name: string) => {
    const mesh = meshes.get(name)
    if (!mesh) throw new Error(`The rover asset has no ${name} mesh; rebuild it with scripts/photoreal-assets`)
    const geometry = bakedGeometry(mesh)
    created.push(geometry)
    return { geometry, material: mesh.material as THREE.MeshStandardMaterial }
  }
  const lods: RoverLod[] = LOD_RANGES.map((maxDistance, k) => {
    const body = bake(`body_lod${k}`)
    const wheel = bake(`wheel_lod${k}`)
    const wheels: WheelPart[] = extras.wheels.map(w => ({
      geometry: wheel.geometry, material: wheel.material, side: w.side, yaw: w.yaw,
      pivot: new THREE.Vector3(...w.pivot), radius: extras.wheelRadius,
    }))
    return { body: [body], wheels, maxDistance }
  })
  for (const { body } of lods) for (const part of body) (part.material as THREE.MeshStandardMaterial).envMapIntensity = 1
  const [bx, by, bz] = extras.beacon
  const lights = new THREE.SphereGeometry(0.011, 10, 8).translate(bx, by + 0.014, bz)
  const triangles = (lods[0].body[0].geometry.index!.count + extras.wheels.length * lods[0].wheels[0].geometry.index!.count) / 3
  return {
    name: 'NASA Perseverance (Mars 2020)', source: 'asset', lods, lights, triangles,
    dispose: () => { created.forEach(g => g.dispose()); lights.dispose() },
  }
}
