import * as THREE from 'three'
import type { GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { HDRLoader } from 'three/examples/jsm/loaders/HDRLoader.js'
import { KTX2Loader } from 'three/examples/jsm/loaders/KTX2Loader.js'
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js'
import { validateManifest } from './manifest'
import type { Manifest } from './manifest'

export type { Manifest } from './manifest'

/** Where scripts/photoreal-assets/build.mjs writes the assets. */
export const ASSET_ROOT = '/photoreal/'

let manifestRequest: Promise<Manifest> | null = null

/** The asset manifest, fetched once per page (it is plain data, so the panels can read it outside the canvas). */
export function loadManifest(): Promise<Manifest> {
  manifestRequest ??= fetch(`${ASSET_ROOT}manifest.json`)
    .then(async response => {
      if (!response.ok) throw new Error(`Photoreal assets are missing (HTTP ${response.status} for manifest.json)`)
      return validateManifest(await response.json())
    })
    .catch(error => { manifestRequest = null; throw error })
  return manifestRequest
}

/**
 * Loads and owns every GPU resource of the photoreal mode. Requests are
 * cached as promises, so React's use() can suspend on them and repeated
 * reads cost nothing; dispose() frees textures, geometry, materials and the
 * transcoder workers when the mode is left.
 */
export class AssetLibrary {
  private readonly ktx2: KTX2Loader
  private readonly gltfLoader: GLTFLoader
  private readonly hdrLoader = new HDRLoader()
  private readonly cache = new Map<string, Promise<unknown>>()
  private readonly owned = new Set<{ dispose: () => void }>()
  private disposed = false

  constructor(renderer: THREE.WebGLRenderer) {
    // three.js resolves its bundled Basis transcoder itself (new URL(..., import.meta.url)), so Vite ships it
    this.ktx2 = new KTX2Loader().detectSupport(renderer)
    this.gltfLoader = new GLTFLoader().setKTX2Loader(this.ktx2).setMeshoptDecoder(MeshoptDecoder)
  }

  manifest(): Promise<Manifest> {
    return loadManifest()
  }

  gltf(file: string): Promise<GLTF> {
    return this.once(`gltf:${file}`, async () => {
      const gltf = await this.gltfLoader.loadAsync(ASSET_ROOT + file)
      gltf.scene.traverse(object => {
        const mesh = object as THREE.Mesh
        if (!mesh.isMesh) return
        this.own(mesh.geometry)
        for (const material of [mesh.material].flat()) {
          this.own(material)
          for (const value of Object.values(material)) if (value instanceof THREE.Texture) this.own(value)
        }
      })
      return gltf
    })
  }

  texture(file: string, colorSpace: THREE.ColorSpace): Promise<THREE.Texture> {
    return this.once(`ktx2:${file}`, async () => {
      const texture = this.own(await this.ktx2.loadAsync(ASSET_ROOT + file))
      texture.colorSpace = colorSpace
      texture.wrapS = texture.wrapT = THREE.RepeatWrapping
      texture.anisotropy = 8
      return texture
    })
  }

  /** Equirectangular HDRI, kept as half floats so it can also be sampled on the CPU for the sun and horizon. */
  hdr(file: string): Promise<THREE.DataTexture> {
    return this.once(`hdr:${file}`, async () => {
      const texture = this.own(await this.hdrLoader.loadAsync(ASSET_ROOT + file))
      texture.mapping = THREE.EquirectangularReflectionMapping
      return texture
    })
  }

  /** Registers a resource created from loaded assets so it is freed with them. */
  own<T extends { dispose: () => void }>(resource: T): T {
    if (this.disposed) resource.dispose()
    else this.owned.add(resource)
    return resource
  }

  dispose(): void {
    this.disposed = true
    for (const resource of this.owned) resource.dispose()
    this.owned.clear()
    this.cache.clear()
    this.ktx2.dispose()
  }

  /** A promise cached under key, so a component can suspend on derived async work without restarting it. */
  cached<T>(key: string, load: () => Promise<T>): Promise<T> {
    return this.once(`derived:${key}`, load)
  }

  private once<T>(key: string, load: () => Promise<T>): Promise<T> {
    let promise = this.cache.get(key) as Promise<T> | undefined
    if (!promise) {
      promise = load()
      this.cache.set(key, promise)
    }
    return promise
  }
}
