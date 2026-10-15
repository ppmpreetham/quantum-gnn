import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import { ARENA } from '../../config/fleet'
import type { BodyPart, ForwardAxis, RoverModel, WheelPart } from './types'

const MAX_BYTES = 24 * 1024 * 1024
const MAX_TRIANGLES = 250_000
const MAX_MESHES = 96
const WHEEL_NAME = /wheel|tire|tyre|rim/i
const ROTATION: Record<Exclude<ForwardAxis, 'auto'>, number> = { '+x': 0, '-x': Math.PI, '+z': Math.PI / 2, '-z': -Math.PI / 2 }

/** glTF is y-up; Blender's exporter turns its -y forward into +z, which three.js models also use. */
function resolveForward(size: THREE.Vector3, forward: ForwardAxis): Exclude<ForwardAxis, 'auto'> {
  if (forward !== 'auto') return forward
  return size.z > size.x * 1.05 ? '+z' : '+x'
}

const triangleCount = (g: THREE.BufferGeometry) => (g.index ? g.index.count : g.attributes.position.count) / 3

/** Same attribute layout for every geometry so a material's meshes can merge into one. */
function normalizeAttributes(geometry: THREE.BufferGeometry, withUv: boolean): THREE.BufferGeometry {
  const g = geometry.index ? geometry.toNonIndexed() : geometry.clone()
  for (const name of Object.keys(g.attributes)) if (!['position', 'normal', 'uv'].includes(name)) g.deleteAttribute(name)
  g.morphAttributes = {}
  if (!g.attributes.normal) g.computeVertexNormals()
  if (withUv && !g.attributes.uv) g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(g.attributes.position.count * 2), 2))
  if (!withUv && g.attributes.uv) g.deleteAttribute('uv')
  return g
}

export async function loadGlbRover(file: File, forward: ForwardAxis): Promise<RoverModel> {
  if (!/\.glb$/i.test(file.name)) throw new Error('Choose a binary .glb file. In Blender: File, Export, glTF 2.0, format glTF Binary.')
  if (file.size > MAX_BYTES) throw new Error(`The file is ${(file.size / 1048576).toFixed(1)} MB; the limit is 24 MB.`)
  const manager = new THREE.LoadingManager()
  manager.setURLModifier(url => {
    if (!url.startsWith('blob:') && !url.startsWith('data:')) throw new Error('Textures must be embedded in the .glb.')
    return url
  })
  const loader = new GLTFLoader(manager).setMeshoptDecoder(MeshoptDecoder)
  let gltf
  try {
    gltf = await loader.parseAsync(await file.arrayBuffer(), '')
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    if (/draco/i.test(message)) throw new Error('Draco-compressed meshes are not supported. Re-export without Draco, or use Meshopt compression.')
    if (/ktx|basis/i.test(message)) throw new Error('KTX2 textures are not supported. Re-export with PNG or JPEG textures.')
    throw new Error(`Could not read the model: ${message}`)
  }

  const meshes: THREE.Mesh[] = []
  let skinned = false
  gltf.scene.updateMatrixWorld(true)
  gltf.scene.traverse(object => {
    if ((object as THREE.SkinnedMesh).isSkinnedMesh) skinned = true
    else if ((object as THREE.Mesh).isMesh) meshes.push(object as THREE.Mesh)
  })
  const triangles = meshes.reduce((sum, m) => sum + triangleCount(m.geometry), 0)
  const owned = new Set<THREE.Material>()
  meshes.forEach(m => (Array.isArray(m.material) ? m.material : [m.material]).forEach(mat => owned.add(mat)))
  const fail = (message: string): never => {
    meshes.forEach(m => m.geometry.dispose())
    owned.forEach(m => m.dispose())
    throw new Error(message)
  }
  if (skinned) fail('Skinned (rigged) meshes are not supported; apply the armature in Blender before exporting.')
  if (!meshes.length) fail('The file has no meshes.')
  if (meshes.length > MAX_MESHES) fail(`The model has ${meshes.length} meshes; join them in Blender to stay under ${MAX_MESHES}.`)
  if (triangles > MAX_TRIANGLES) fail(`The model has ${Math.round(triangles).toLocaleString()} triangles; decimate it below ${MAX_TRIANGLES.toLocaleString()}.`)

  const bounds = new THREE.Box3()
  meshes.forEach(m => { m.geometry.computeBoundingBox(); bounds.union(m.geometry.boundingBox!.clone().applyMatrix4(m.matrixWorld)) })
  const size = bounds.getSize(new THREE.Vector3())
  if (![size.x, size.y, size.z].every(Number.isFinite) || Math.max(size.x, size.z) < 1e-6) fail('The model has empty or invalid bounds.')
  const axis = resolveForward(size, forward)
  const center = bounds.getCenter(new THREE.Vector3())

  // centre on the ground, face +x, and scale so the footprint fits the collision circle
  const normalize = new THREE.Matrix4().makeRotationY(ROTATION[axis])
    .multiply(new THREE.Matrix4().makeTranslation(-center.x, -bounds.min.y, -center.z))
  const rotated = bounds.clone().applyMatrix4(normalize)
  const halfDiagonal = Math.hypot(rotated.max.x - rotated.min.x, rotated.max.z - rotated.min.z) / 2
  const scale = (ARENA.bodyRadius * 0.97) / halfDiagonal
  normalize.premultiply(new THREE.Matrix4().makeScale(scale, scale, scale))
  const width = (rotated.max.z - rotated.min.z) * scale

  const byMaterial = new Map<THREE.Material, THREE.BufferGeometry[]>()
  const body: BodyPart[] = []
  const wheels: WheelPart[] = []
  for (const mesh of meshes) {
    const geometry = mesh.geometry.clone().applyMatrix4(new THREE.Matrix4().multiplyMatrices(normalize, mesh.matrixWorld))
    if (WHEEL_NAME.test(mesh.name) || WHEEL_NAME.test(mesh.parent?.name ?? '')) {
      geometry.computeBoundingBox()
      const box = geometry.boundingBox!
      const pivot = box.getCenter(new THREE.Vector3())
      const radius = Math.max(1e-3, (box.max.y - box.min.y) / 2)
      const side = Math.abs(pivot.z) < width * 0.08 ? 0 : pivot.z < 0 ? -1 : 1
      wheels.push({ geometry: geometry.translate(-pivot.x, -pivot.y, -pivot.z), material: mesh.material, side, pivot, radius })
    } else if (Array.isArray(mesh.material)) {
      body.push({ geometry, material: mesh.material })
    } else {
      byMaterial.set(mesh.material, [...(byMaterial.get(mesh.material) ?? []), geometry])
    }
  }
  for (const [material, list] of byMaterial) {
    if (list.length === 1) { body.push({ geometry: list[0], material }); continue }
    const withUv = list.every(g => g.attributes.uv)
    const normalized = list.map(g => normalizeAttributes(g, withUv))
    const merged = mergeGeometries(normalized, false)
    list.forEach(g => g.dispose())
    normalized.forEach(g => g.dispose())
    if (!merged) fail('Some meshes could not be merged.')
    body.push({ geometry: merged!, material })
  }
  meshes.forEach(m => m.geometry.dispose())

  const top = rotated.max.y * scale
  const lights = new THREE.SphereGeometry(0.03, 12, 8).translate(0, top + 0.06, 0)
  const textures = new Set<THREE.Texture>()
  owned.forEach(m => Object.values(m).forEach(v => { if (v instanceof THREE.Texture) textures.add(v) }))
  return {
    name: file.name, source: 'glb', body, wheels, lights, triangles,
    dispose: () => {
      body.forEach(p => p.geometry.dispose())
      wheels.forEach(p => p.geometry.dispose())
      lights.dispose()
      owned.forEach(m => m.dispose())
      textures.forEach(t => t.dispose())
    },
  }
}
