import * as THREE from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'

/**
 * Bakes colour, roughness and metalness into vertex attributes so a whole
 * multi-material rover merges into one geometry and renders in one draw call.
 */
export function paint(geometry: THREE.BufferGeometry, hex: string, roughness: number, metalness: number): THREE.BufferGeometry {
  const count = geometry.attributes.position.count
  const color = new THREE.Color(hex)
  const colors = new Float32Array(count * 3)
  const pbr = new Float32Array(count * 2)
  for (let i = 0; i < count; i++) {
    colors.set([color.r, color.g, color.b], i * 3)
    pbr.set([roughness, metalness], i * 2)
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  geometry.setAttribute('pbr', new THREE.BufferAttribute(pbr, 2))
  return geometry
}

export function mergePainted(parts: THREE.BufferGeometry[]): THREE.BufferGeometry {
  // RoundedBoxGeometry is non-indexed while primitives are indexed; merge needs one layout
  const flat = parts.map(p => (p.index ? p.toNonIndexed() : p))
  const merged = mergeGeometries(flat, false)
  new Set([...parts, ...flat]).forEach(p => p.dispose())
  if (!merged) throw new Error('Rover parts have incompatible attributes')
  merged.computeBoundingSphere()
  return merged
}

/** MeshStandardMaterial that reads roughness and metalness from the pbr attribute. */
export function createVertexPbrMaterial(envMapIntensity = 1): THREE.MeshStandardMaterial {
  const material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1, metalness: 1, envMapIntensity })
  material.onBeforeCompile = shader => {
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nattribute vec2 pbr;\nvarying vec2 vPbr;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvPbr = pbr;')
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying vec2 vPbr;')
      .replace('#include <roughnessmap_fragment>', 'float roughnessFactor = roughness * vPbr.x;')
      .replace('#include <metalnessmap_fragment>', 'float metalnessFactor = metalness * vPbr.y;')
  }
  material.customProgramCacheKey = () => 'vertex-pbr'
  return material
}
