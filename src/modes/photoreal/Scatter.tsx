import { use, useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { bakedGeometry, meshesByName } from './gltfGeometry'
import { useLibrary } from './library'
import { scatter } from './terrainField'
import type { ScatterSpec } from './terrainField'

/**
 * Placement per prop. near: props closer than this (metres) use the detailed
 * mesh and may cast shadows; the rest use the light one.
 */
const SPECS: Record<string, ScatterSpec & { seed: number; near: number }> = {
  namaqualand_boulder_02: { seed: 11, near: 30, count: 26, minRadius: 8.5, maxRadius: 120, scale: [1.0, 3.2], variants: 1, spacing: 2.6, inward: 0.55 },
  namaqualand_boulder_03: { seed: 12, near: 30, count: 14, minRadius: 11, maxRadius: 140, scale: [1.0, 2.8], variants: 1, spacing: 2.6, inward: 0.5 },
  namaqualand_boulder_05: { seed: 13, near: 22, count: 40, minRadius: 7, maxRadius: 70, scale: [0.8, 2.4], variants: 1, spacing: 1.8, inward: 0.7 },
  namaqualand_stones_01: { seed: 14, near: 11, count: 200, minRadius: 6.3, maxRadius: 30, scale: [2.0, 5.0], variants: 3, spacing: 0.25, inward: 0.8 },
  wild_rooibos_bush: { seed: 15, near: 16, count: 80, minRadius: 7.2, maxRadius: 90, scale: [1.0, 2.6], variants: 2, spacing: 1.2, inward: 0.6 },
}

interface Batch { key: string; geometry: THREE.BufferGeometry; material: THREE.Material; matrices: THREE.Matrix4[]; near: boolean }

/**
 * Photoscanned Namaqualand boulders, stones and shrubs scattered around the
 * arena with deterministic placements: one instanced draw per prop variant
 * and detail level, with static matrices written once.
 */
export function Scatter({ density, shadows }: { density: number; shadows: boolean }) {
  const { library, manifest } = useLibrary()
  const loaded = use(library.cached('props', () => Promise.all(manifest.props.map(async asset => ({ asset, gltf: await library.gltf(asset.file) })))))
  const batches = useMemo(() => {
    const out: Batch[] = []
    for (const { asset, gltf } of loaded) {
      const spec = SPECS[asset.id]
      if (!spec) continue
      const meshes = meshesByName(gltf.scene)
      const placements = scatter({ ...spec, count: Math.round(spec.count * density) }, spec.seed)
      for (let v = 0; v < asset.variants; v++) {
        for (const near of [true, false]) {
          const mesh = meshes.get(`v${v}_lod${near ? 0 : 1}`)
          if (!mesh) continue
          const matrices = placements.filter(p => p.variant === v && Math.hypot(p.x, p.z) < spec.near === near).map(p =>
            new THREE.Matrix4().compose(
              new THREE.Vector3(p.x, p.y, p.z),
              new THREE.Quaternion().setFromEuler(new THREE.Euler(p.tilt, p.yaw, p.tilt * 0.6)),
              new THREE.Vector3(p.scale, p.scale, p.scale),
            ))
          if (matrices.length) out.push({ key: `${asset.id}-${v}-${near}`, geometry: bakedGeometry(mesh), material: mesh.material as THREE.Material, matrices, near })
        }
      }
    }
    return out
  }, [loaded, density])
  useEffect(() => () => batches.forEach(b => b.geometry.dispose()), [batches])

  return <group>{batches.map(batch => <StaticInstances key={batch.key} batch={batch} castShadow={shadows && batch.near} />)}</group>
}

function StaticInstances({ batch, castShadow }: { batch: Batch; castShadow: boolean }) {
  const ref = useRef<THREE.InstancedMesh>(null)
  useLayoutEffect(() => {
    const mesh = ref.current!
    batch.matrices.forEach((m, i) => mesh.setMatrixAt(i, m))
    mesh.instanceMatrix.needsUpdate = true
    // bounds over every instance, so the whole batch is culled when it is off screen
    mesh.computeBoundingSphere()
  }, [batch])
  return <instancedMesh ref={ref} args={[batch.geometry, batch.material, batch.matrices.length]} castShadow={castShadow} receiveShadow />
}
