import { use, useEffect, useMemo } from 'react'
import * as THREE from 'three'
import { TERRAIN_RADIUS, terrainHeight } from './terrainField'
import { createTerrainMaterial } from './terrainMaterial'
import type { GroundTextures } from './terrainMaterial'
import { useLibrary } from './library'

/**
 * Polar grid: rings grow geometrically from the centre, so the arena gets
 * centimetre-scale tessellation and the distant hills a few metres, about
 * 37k triangles in all. Normals come from central differences of the height
 * field rather than the mesh, so they stay smooth across ring sizes.
 */
function terrainGeometry(segments = 192, firstRing = 0.25, growth = 1.045): THREE.BufferGeometry {
  const radii = [0]
  for (let step = firstRing; radii.at(-1)! < TERRAIN_RADIUS; step *= growth) radii.push(Math.min(TERRAIN_RADIUS, radii.at(-1)! + step))
  const count = 1 + (radii.length - 1) * segments
  const position = new Float32Array(count * 3), normal = new Float32Array(count * 3)
  const eps = 0.05
  const write = (v: number, x: number, z: number) => {
    position.set([x, terrainHeight(x, z), z], v * 3)
    const dx = terrainHeight(x + eps, z) - terrainHeight(x - eps, z)
    const dz = terrainHeight(x, z + eps) - terrainHeight(x, z - eps)
    const length = Math.hypot(dx, 2 * eps, dz)
    normal.set([-dx / length, (2 * eps) / length, -dz / length], v * 3)
  }
  write(0, 0, 0)
  for (let ring = 1; ring < radii.length; ring++) {
    for (let s = 0; s < segments; s++) {
      const a = (s / segments) * Math.PI * 2
      write(1 + (ring - 1) * segments + s, radii[ring] * Math.cos(a), radii[ring] * Math.sin(a))
    }
  }
  const index: number[] = []
  const at = (ring: number, s: number) => 1 + (ring - 1) * segments + (s % segments)
  for (let s = 0; s < segments; s++) index.push(0, at(1, s + 1), at(1, s))
  for (let ring = 1; ring < radii.length - 1; ring++) {
    for (let s = 0; s < segments; s++) {
      const a = at(ring, s), b = at(ring, s + 1), c = at(ring + 1, s), d = at(ring + 1, s + 1)
      index.push(a, b, c, b, d, c)
    }
  }
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(position, 3))
  geometry.setAttribute('normal', new THREE.BufferAttribute(normal, 3))
  geometry.setIndex(index)
  geometry.computeBoundingSphere()
  return geometry
}

/**
 * The scans cover 2.5 m of real ground; the arena rover stands in for the
 * 3 m Perseverance at 0.3 scale. Rendering the grains 2.2 times larger than
 * strict scale reads as the coarse gravel a rover that size drives on.
 */
const GRAIN_SCALE = 2.2

/** The field site's ground: scanned Namaqualand gravel and sand on a flat arena that rolls into dunes. */
export function Terrain({ detail }: { detail: boolean }) {
  const { library, manifest } = useLibrary()
  const [a, b] = use(library.cached('grounds', () => Promise.all(manifest.grounds.slice(0, 2).map(async (g, k): Promise<GroundTextures> => {
    const [color, normal, arh] = await Promise.all([
      library.texture(g.color, THREE.SRGBColorSpace),
      library.texture(g.normal, THREE.NoColorSpace),
      library.texture(g.arh, THREE.NoColorSpace),
    ])
    return { color, normal, arh, tileMeters: g.tileMeters * GRAIN_SCALE * (k === 0 ? 1 : 1.31) }
  }))))
  const geometry = useMemo(() => terrainGeometry(), [])
  const material = useMemo(() => createTerrainMaterial(a, b, detail), [a, b, detail])
  useEffect(() => () => geometry.dispose(), [geometry])
  useEffect(() => () => material.dispose(), [material])
  return <mesh geometry={geometry} material={material} receiveShadow renderOrder={-1} />
}
