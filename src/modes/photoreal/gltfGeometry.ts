import * as THREE from 'three'

/** Quantized, normalized or interleaved glTF attributes as plain floats, so geometry can be baked and merged. */
function floatAttribute(attribute: THREE.BufferAttribute | THREE.InterleavedBufferAttribute): THREE.BufferAttribute {
  const { count, itemSize } = attribute
  const out = new Float32Array(count * itemSize)
  for (let i = 0; i < count; i++) for (let k = 0; k < itemSize; k++) out[i * itemSize + k] = attribute.getComponent(i, k)
  return new THREE.BufferAttribute(out, itemSize)
}

/** A loaded mesh's geometry in model space: dequantized, with its node transform baked in. */
export function bakedGeometry(mesh: THREE.Mesh): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry()
  for (const [name, attribute] of Object.entries(mesh.geometry.attributes)) geometry.setAttribute(name, floatAttribute(attribute))
  if (mesh.geometry.index) geometry.setIndex(mesh.geometry.index.clone())
  mesh.updateWorldMatrix(true, false)
  geometry.applyMatrix4(mesh.matrixWorld)
  geometry.computeBoundingSphere()
  return geometry
}

/** Named meshes of a loaded scene. */
export function meshesByName(root: THREE.Object3D): Map<string, THREE.Mesh> {
  const meshes = new Map<string, THREE.Mesh>()
  root.traverse(object => { if ((object as THREE.Mesh).isMesh) meshes.set(object.name, object as THREE.Mesh) })
  return meshes
}
