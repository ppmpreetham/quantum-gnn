/**
 * Rebuilds public/photoreal/ (the photoreal mode's assets) from the original
 * public-domain and CC0 sources, so every transform is reproducible:
 *
 *   cd scripts/photoreal-assets && npm install
 *   KTX_BIN=/path/to/KTX-Software/bin/ktx node build.mjs [--cache DIR]
 *
 * The rover becomes one atlased material (a single draw call per LOD) with
 * three meshoptimizer LODs and one shared wheel mesh; props get two LODs;
 * every texture is KTX2 (UASTC for colour and normals, ETC1S for packed
 * masks) and every mesh is Meshopt-compressed. A manifest with attribution
 * is written next to the assets for the client and for CREDITS.md.
 */
import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { Document, NodeIO } from '@gltf-transform/core'
import { ALL_EXTENSIONS, EXTMeshoptCompression, KHRMeshQuantization, KHRTextureBasisu } from '@gltf-transform/extensions'
import { meshopt, prune, quantize, reorder } from '@gltf-transform/functions'
import { MeshoptEncoder, MeshoptSimplifier } from 'meshoptimizer'
import sharp from 'sharp'

const HERE = dirname(fileURLToPath(import.meta.url))
const OUT = resolve(HERE, '../../public/photoreal')
const flag = name => { const i = process.argv.indexOf(name); return i > 0 ? process.argv[i + 1] : undefined }
const CACHE = resolve(flag('--cache') ?? join(HERE, '.cache'))
const KTX = process.env.KTX_BIN ?? 'ktx'
const TMP = join(CACHE, 'tmp')

/** Arena constants mirrored from shared/fleet-config.json. */
const FLEET = JSON.parse(readFileSync(resolve(HERE, '../../shared/fleet-config.json'), 'utf8'))
const FOOTPRINT = FLEET.arena.bodyRadius * 0.97

const POLY = 'https://dl.polyhaven.org/file/ph-assets'
const CREDITS = {
  perseverance: {
    title: 'Mars Perseverance Rover, 3D Model', author: 'NASA/JPL-Caltech',
    license: 'NASA media usage guidelines (not copyrighted in the US; credit NASA/JPL-Caltech)',
    url: 'https://science.nasa.gov/resource/mars-perseverance-rover-3d-model/',
  },
  goegap: { title: 'Goegap HDRI', author: 'Greg Zaal (Poly Haven)', license: 'CC0 1.0', url: 'https://polyhaven.com/a/goegap' },
  rogland_sunset: { title: 'Rogland Sunset HDRI', author: 'Greg Zaal (Poly Haven)', license: 'CC0 1.0', url: 'https://polyhaven.com/a/rogland_sunset' },
  rogland_overcast: { title: 'Rogland Overcast HDRI', author: 'Greg Zaal (Poly Haven)', license: 'CC0 1.0', url: 'https://polyhaven.com/a/rogland_overcast' },
  gravelly_sand: { title: 'Gravelly Sand texture', author: 'Poly Haven', license: 'CC0 1.0', url: 'https://polyhaven.com/a/gravelly_sand' },
  sandy_gravel_02: { title: 'Sandy Gravel 02 texture', author: 'Poly Haven', license: 'CC0 1.0', url: 'https://polyhaven.com/a/sandy_gravel_02' },
  namaqualand_boulder_02: { title: 'Namaqualand Boulder 02', author: 'Poly Haven', license: 'CC0 1.0', url: 'https://polyhaven.com/a/namaqualand_boulder_02' },
  namaqualand_boulder_03: { title: 'Namaqualand Boulder 03', author: 'Poly Haven', license: 'CC0 1.0', url: 'https://polyhaven.com/a/namaqualand_boulder_03' },
  namaqualand_boulder_05: { title: 'Namaqualand Boulder 05', author: 'Poly Haven', license: 'CC0 1.0', url: 'https://polyhaven.com/a/namaqualand_boulder_05' },
  namaqualand_stones_01: { title: 'Namaqualand Stones 01', author: 'Poly Haven', license: 'CC0 1.0', url: 'https://polyhaven.com/a/namaqualand_stones_01' },
  wild_rooibos_bush: { title: 'Wild Rooibos Bush', author: 'Poly Haven', license: 'CC0 1.0', url: 'https://polyhaven.com/a/wild_rooibos_bush' },
}

const SKIES = [
  { id: 'goegap', label: 'Midday, clear desert' },
  { id: 'rogland_sunset', label: 'Low sun, clear evening' },
  { id: 'rogland_overcast', label: 'Overcast, soft light' },
]
const GROUNDS = ['gravelly_sand', 'sandy_gravel_02']
/** Props are scaled to the rover: the 0.9 m arena rover stands in for the 3 m Perseverance. */
const WORLD_SCALE = 0.3
const PROPS = ['namaqualand_boulder_02', 'namaqualand_boulder_03', 'namaqualand_boulder_05', 'namaqualand_stones_01', 'wild_rooibos_bush']

const io = new NodeIO().registerExtensions(ALL_EXTENSIONS)
  .registerDependencies({ 'meshopt.encoder': MeshoptEncoder })

// ---------------------------------------------------------------- utilities

async function download(url, file) {
  const path = join(CACHE, file)
  if (existsSync(path)) return path
  mkdirSync(dirname(path), { recursive: true })
  for (let attempt = 1; ; attempt++) {
    try {
      const response = await fetch(url)
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      writeFileSync(path, Buffer.from(await response.arrayBuffer()))
      return path
    } catch (error) {
      if (attempt >= 4) throw new Error(`${url}: ${error.message}`)
      await new Promise(r => setTimeout(r, 1500 * attempt))
    }
  }
}

async function polyHavenModel(id) {
  const files = await (await fetch(`https://api.polyhaven.com/files/${id}`)).json()
  const entry = files.gltf['1k'].gltf
  const gltf = await download(entry.url, `${id}/${id}.gltf`)
  for (const [rel, include] of Object.entries(entry.include)) await download(include.url, `${id}/${rel}`)
  return gltf
}

/** Encodes an image buffer to KTX2. Colour data is sRGB; masks and normals are linear. */
function toKtx2(png, { srgb, mode }) {
  mkdirSync(TMP, { recursive: true })
  const input = join(TMP, 'in.png'), output = join(TMP, 'out.ktx2')
  writeFileSync(input, png)
  const args = ['create', '--format', srgb ? 'R8G8B8A8_SRGB' : 'R8G8B8A8_UNORM', '--generate-mipmap', '--assign-tf', srgb ? 'srgb' : 'linear']
  // normals stay plain RGB: --normal-mode would repack them as X in RGB and Y in alpha, which three.js does not decode
  if (mode === 'normal') args.push('--encode', 'uastc', '--uastc-quality', '2', '--uastc-rdo', '--uastc-rdo-l', '0.5', '--zstd', '18')
  else if (mode === 'uastc') args.push('--encode', 'uastc', '--uastc-quality', '2', '--uastc-rdo', '--uastc-rdo-l', '1', '--zstd', '18')
  else args.push('--encode', 'basis-lz', '--clevel', '4', '--qlevel', '255')
  args.push(input, output)
  execFileSync(KTX, args, { stdio: ['ignore', 'ignore', 'inherit'] })
  return readFileSync(output)
}

function attachTexture(doc, name, ktx2) {
  return doc.createTexture(name).setImage(new Uint8Array(ktx2)).setMimeType('image/ktx2').setURI(`${name}.ktx2`)
}

/** glTF baseColorFactor is linear; atlas pixels are sRGB. */
const toSrgb = c => Math.round(255 * (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055))

function writeGlb(doc, file) {
  mkdirSync(dirname(file), { recursive: true })
  return io.writeBinary(doc).then(bytes => { writeFileSync(file, bytes); return bytes.byteLength })
}

async function compressGeometry(doc) {
  doc.createExtension(KHRTextureBasisu).setRequired(true)
  await doc.transform(prune(), reorder({ encoder: MeshoptEncoder }), quantize(), meshopt({ encoder: MeshoptEncoder, level: 'medium' }))
  doc.createExtension(KHRMeshQuantization).setRequired(true)
  doc.createExtension(EXTMeshoptCompression).setRequired(true)
}

/** Simplifies an indexed triangle list (positions xyz, normals xyz, uv) to about ratio of its triangles. */
function simplify(mesh, ratio, error, flags = []) {
  if (ratio >= 1) return mesh
  const count = mesh.position.length / 3
  const attributes = new Float32Array(count * 5)
  for (let i = 0; i < count; i++) attributes.set([mesh.normal[i * 3], mesh.normal[i * 3 + 1], mesh.normal[i * 3 + 2], mesh.uv[i * 2], mesh.uv[i * 2 + 1]], i * 5)
  const target = Math.max(3, Math.floor((mesh.index.length * ratio) / 3) * 3)
  const [index] = flags.includes('Sloppy')
    ? MeshoptSimplifier.simplifySloppy(mesh.index, mesh.position, 3, null, target, error)
    : MeshoptSimplifier.simplifyWithAttributes(mesh.index, mesh.position, 3, attributes, 5, [0.4, 0.4, 0.4, 1, 1], null, target, error, flags)
  return compact({ ...mesh, index })
}

/** Drops vertices no triangle references. */
function compact(mesh) {
  const remap = new Int32Array(mesh.position.length / 3).fill(-1)
  let next = 0
  for (const v of mesh.index) if (remap[v] < 0) remap[v] = next++
  const out = { position: new Float32Array(next * 3), normal: new Float32Array(next * 3), uv: new Float32Array(next * 2), index: new Uint32Array(mesh.index.length) }
  remap.forEach((to, from) => {
    if (to < 0) return
    out.position.set(mesh.position.subarray(from * 3, from * 3 + 3), to * 3)
    out.normal.set(mesh.normal.subarray(from * 3, from * 3 + 3), to * 3)
    out.uv.set(mesh.uv.subarray(from * 2, from * 2 + 2), to * 2)
  })
  mesh.index.forEach((v, k) => { out.index[k] = remap[v] })
  return out
}

function addMesh(doc, buffer, name, mesh, material) {
  const prim = doc.createPrimitive()
    .setAttribute('POSITION', doc.createAccessor().setType('VEC3').setArray(mesh.position).setBuffer(buffer))
    .setAttribute('NORMAL', doc.createAccessor().setType('VEC3').setArray(mesh.normal).setBuffer(buffer))
    .setAttribute('TEXCOORD_0', doc.createAccessor().setType('VEC2').setArray(mesh.uv).setBuffer(buffer))
    .setIndices(doc.createAccessor().setType('SCALAR').setArray(mesh.index).setBuffer(buffer))
    .setMaterial(material)
  const node = doc.createNode(name).setMesh(doc.createMesh(name).addPrimitive(prim))
  return node
}

const triangles = mesh => mesh.index.length / 3

// ---------------------------------------------------------------- rover

/**
 * Packs every material of the NASA model into one atlas. Textured materials
 * whose UVs stay inside [0, 1] are cropped to the UV rectangle they use and
 * scaled into a cell; constant materials become small solid cells. Returns
 * a remap from material to cell and the three atlas images.
 */
async function buildAtlas(materials, size) {
  const cells = []
  for (const entry of materials) {
    const { material, uv } = entry
    const color = material.getBaseColorTexture()
    const inRange = uv[0] >= -0.01 && uv[1] <= 1.01 && uv[2] >= -0.01 && uv[3] <= 1.01
    if (color && inRange) {
      const meta = await sharp(Buffer.from(color.getImage())).metadata()
      const crop = {
        left: Math.max(0, Math.floor(Math.max(0, uv[0]) * meta.width) - 2),
        top: Math.max(0, Math.floor(Math.max(0, uv[2]) * meta.height) - 2),
      }
      crop.width = Math.min(meta.width - crop.left, Math.ceil(Math.min(1, uv[1]) * meta.width) + 2 - crop.left)
      crop.height = Math.min(meta.height - crop.top, Math.ceil(Math.min(1, uv[3]) * meta.height) + 2 - crop.top)
      const fit = n => Math.min(512, Math.max(32, 2 ** Math.round(Math.log2(Math.max(1, n) / 2))))
      cells.push({ entry, textured: true, crop, meta, w: fit(crop.width), h: fit(crop.height) })
    } else {
      cells.push({ entry, textured: false, w: 8, h: 8 })
    }
  }
  // shelf packing, tallest first
  cells.sort((a, b) => b.h - a.h || b.w - a.w)
  let x = 0, y = 0, shelf = 0
  for (const cell of cells) {
    if (x + cell.w > size) { x = 0; y += shelf; shelf = 0 }
    cell.x = x; cell.y = y
    x += cell.w; shelf = Math.max(shelf, cell.h)
  }
  if (y + shelf > size) throw new Error(`Atlas overflow: needs ${y + shelf} px rows`)

  const layers = { color: [], normal: [], orm: [] }
  for (const cell of cells) {
    const m = cell.entry.material
    const pad = Math.max(2, cell.w / 64)
    const inner = { width: cell.w - 2 * pad, height: cell.h - 2 * pad }
    const rough = m.getRoughnessFactor(), metal = m.getMetallicFactor()
    const orm = { r: 255, g: Math.round(rough * 255), b: Math.round(metal * 255) }
    const solid = (rgb) => sharp({ create: { width: cell.w, height: cell.h, channels: 3, background: rgb } }).png().toBuffer()
    const place = async (image, crop) => sharp(Buffer.from(image)).removeAlpha().extract(crop).resize(inner.width, inner.height, { fit: 'fill' })
      .extend({ top: pad, bottom: pad, left: pad, right: pad, extendWith: 'copy' }).png().toBuffer()
    if (cell.textured) {
      const factor = m.getBaseColorFactor()
      let color = await place(m.getBaseColorTexture().getImage(), cell.crop)
      if (factor.slice(0, 3).some(f => Math.abs(f - 1) > 1e-3)) {
        color = await sharp(color).linear(factor.slice(0, 3).map(f => f ** (1 / 2.2)), [0, 0, 0]).png().toBuffer()
      }
      layers.color.push({ input: color, left: cell.x, top: cell.y })
      const normal = m.getNormalTexture()
      if (normal) {
        const nmeta = await sharp(Buffer.from(normal.getImage())).metadata()
        const sx = nmeta.width / cell.meta.width, sy = nmeta.height / cell.meta.height
        const ncrop = { left: Math.floor(cell.crop.left * sx), top: Math.floor(cell.crop.top * sy), width: Math.max(1, Math.floor(cell.crop.width * sx)), height: Math.max(1, Math.floor(cell.crop.height * sy)) }
        layers.normal.push({ input: await place(normal.getImage(), ncrop), left: cell.x, top: cell.y })
      } else {
        layers.normal.push({ input: await solid({ r: 128, g: 128, b: 255 }), left: cell.x, top: cell.y })
      }
    } else {
      const f = m.getBaseColorFactor()
      let rgb = { r: toSrgb(f[0]), g: toSrgb(f[1]), b: toSrgb(f[2]) }
      const texture = m.getBaseColorTexture()
      if (texture) {
        const { channels } = await sharp(Buffer.from(texture.getImage())).stats()
        rgb = { r: Math.round(channels[0].mean), g: Math.round(channels[1].mean), b: Math.round(channels[2].mean) }
      }
      layers.color.push({ input: await solid(rgb), left: cell.x, top: cell.y })
      layers.normal.push({ input: await solid({ r: 128, g: 128, b: 255 }), left: cell.x, top: cell.y })
    }
    layers.orm.push({ input: await solid(orm), left: cell.x, top: cell.y })
    cell.pad = pad
  }
  const canvas = (background, list) => sharp({ create: { width: size, height: size, channels: 3, background } }).composite(list).png().toBuffer()
  const images = {
    color: await canvas({ r: 128, g: 128, b: 128 }, layers.color),
    normal: await canvas({ r: 128, g: 128, b: 255 }, layers.normal),
    orm: await sharp(await canvas({ r: 255, g: 128, b: 0 }, layers.orm)).resize(size / 2, size / 2).png().toBuffer(),
  }
  const remap = new Map(cells.map(cell => [cell.entry.material, (u, v) => {
    if (!cell.textured) return [(cell.x + cell.w / 2) / size, (cell.y + cell.h / 2) / size]
    const { uv } = cell.entry
    const su = (Math.min(Math.max(u, uv[0]), uv[1]) * cell.meta.width - cell.crop.left) / cell.crop.width
    const sv = (Math.min(Math.max(v, uv[2]), uv[3]) * cell.meta.height - cell.crop.top) / cell.crop.height
    return [(cell.x + cell.pad + su * (cell.w - 2 * cell.pad)) / size, (cell.y + cell.pad + sv * (cell.h - 2 * cell.pad)) / size]
  }]))
  return { remap, images }
}

function transformPoint(m, x, y, z) {
  return [m[0] * x + m[4] * y + m[8] * z + m[12], m[1] * x + m[5] * y + m[9] * z + m[13], m[2] * x + m[6] * y + m[10] * z + m[14]]
}
function transformNormal(m, x, y, z) {
  // inverse-transpose of the upper 3×3; scales here are uniform or axis-aligned so the adjugate suffices
  const a = m[0], b = m[1], c = m[2], d = m[4], e = m[5], f = m[6], g = m[8], h = m[9], i = m[10]
  const nx = (e * i - f * h) * x + (c * h - b * i) * y + (b * f - c * e) * z
  const ny = (f * g - d * i) * x + (a * i - c * g) * y + (c * d - a * f) * z
  const nz = (d * h - e * g) * x + (b * g - a * h) * y + (a * e - b * d) * z
  const len = Math.hypot(nx, ny, nz) || 1
  return [nx / len, ny / len, nz / len]
}
const determinant = m => m[0] * (m[5] * m[10] - m[6] * m[9]) - m[4] * (m[1] * m[10] - m[2] * m[9]) + m[8] * (m[1] * m[6] - m[2] * m[5])

/** Indexed triangle list that welds corners sharing position, normal and uv, so the simplifier sees a connected mesh. */
class MeshBuilder {
  position = []; normal = []; uv = []; index = []
  #seen = new Map()
  add(x, y, z, nx, ny, nz, u, v) {
    const key = `${x.toFixed(5)},${y.toFixed(5)},${z.toFixed(5)},${nx.toFixed(3)},${ny.toFixed(3)},${nz.toFixed(3)},${u.toFixed(5)},${v.toFixed(5)}`
    let i = this.#seen.get(key)
    if (i === undefined) {
      i = this.position.length / 3
      this.position.push(x, y, z); this.normal.push(nx, ny, nz); this.uv.push(u, v)
      this.#seen.set(key, i)
    }
    return i
  }
  build() {
    return { position: new Float32Array(this.position), normal: new Float32Array(this.normal), uv: new Float32Array(this.uv), index: new Uint32Array(this.index) }
  }
}

async function buildRover() {
  const src = await io.read(await download('https://assets.science.nasa.gov/content/dam/science/psd/mars/resources/gltf_files/25042_Perseverance.glb', 'nasa/perseverance.glb'))
  const root = src.getRoot()
  // gather primitives with their world matrices; 'transparent' helper geometry is invisible in the source
  const prims = []
  for (const node of root.listNodes()) {
    const mesh = node.getMesh()
    if (!mesh) continue
    for (const prim of mesh.listPrimitives()) {
      if (prim.getMaterial()?.getName() === 'transparent') continue
      prims.push({ node, prim, matrix: node.getWorldMatrix(), wheel: node.getName() === 'Wheels_objs' })
    }
  }
  const usage = new Map()
  for (const { prim } of prims) {
    const material = prim.getMaterial()
    const uv = prim.getAttribute('TEXCOORD_0')?.getArray()
    const entry = usage.get(material) ?? { material, uv: [Infinity, -Infinity, Infinity, -Infinity] }
    if (uv) for (let k = 0; k < uv.length; k += 2) {
      entry.uv[0] = Math.min(entry.uv[0], uv[k]); entry.uv[1] = Math.max(entry.uv[1], uv[k])
      entry.uv[2] = Math.min(entry.uv[2], uv[k + 1]); entry.uv[3] = Math.max(entry.uv[3], uv[k + 1])
    }
    usage.set(material, entry)
  }
  const { remap, images } = await buildAtlas([...usage.values()], 2048)
  if (process.env.DUMP_ATLAS) for (const [name, png] of Object.entries(images)) writeFileSync(join(CACHE, `atlas_${name}.png`), png)

  // body and wheels in source space; the wheel template is the front wheel on the source +x side
  const body = new MeshBuilder(), wheel = new MeshBuilder()
  const wheelCenters = new Map()
  for (const { prim, matrix, wheel: isWheel } of prims) {
    const pos = prim.getAttribute('POSITION').getArray()
    const nor = prim.getAttribute('NORMAL').getArray()
    const uv = prim.getAttribute('TEXCOORD_0')?.getArray()
    const idx = prim.getIndices().getArray()
    const flip = determinant(matrix) < 0
    const toUv = remap.get(prim.getMaterial())
    for (let t = 0; t < idx.length; t += 3) {
      const corners = [idx[t], flip ? idx[t + 2] : idx[t + 1], flip ? idx[t + 1] : idx[t + 2]]
      const world = corners.map(v => transformPoint(matrix, pos[v * 3], pos[v * 3 + 1], pos[v * 3 + 2]))
      let target = body
      if (isWheel) {
        const cx = (world[0][0] + world[1][0] + world[2][0]) / 3, cz = (world[0][2] + world[1][2] + world[2][2]) / 3
        const key = `${cx < 0 ? 'R' : 'L'}${cz > 0.6 ? 'F' : cz < -0.6 ? 'B' : 'M'}`
        const box = wheelCenters.get(key) ?? [Infinity, -Infinity, Infinity, -Infinity, Infinity, -Infinity]
        for (const p of world) { box[0] = Math.min(box[0], p[0]); box[1] = Math.max(box[1], p[0]); box[2] = Math.min(box[2], p[1]); box[3] = Math.max(box[3], p[1]); box[4] = Math.min(box[4], p[2]); box[5] = Math.max(box[5], p[2]) }
        wheelCenters.set(key, box)
        if (key !== 'LF') continue
        target = wheel
      }
      for (let k = 0; k < 3; k++) {
        const v = corners[k]
        const n = transformNormal(matrix, nor[v * 3], nor[v * 3 + 1], nor[v * 3 + 2])
        const [u, w] = toUv(uv ? uv[v * 2] : 0.5, uv ? uv[v * 2 + 1] : 0.5)
        target.index.push(target.add(world[k][0], world[k][1], world[k][2], n[0], n[1], n[2], u, w))
      }
    }
  }

  // arena frame: forward +z (source) becomes +x, ground at y = 0, footprint inside the collision circle
  const bodyMesh = body.build(), wheelMesh = wheel.build()
  const all = [bodyMesh.position, ...[...wheelCenters.values()].map(b => new Float32Array([b[0], b[2], b[4], b[1], b[3], b[5]]))]
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity]
  for (const array of all) for (let k = 0; k < array.length; k += 3) for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], array[k + a]); hi[a] = Math.max(hi[a], array[k + a]) }
  const center = [(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2]
  const scale = FOOTPRINT / (Math.hypot(hi[0] - lo[0], hi[2] - lo[2]) / 2)
  // R_y(+90°): x' = z, z' = -x, so source +z (front) maps to +x and source +x (rover's left) to -z
  const toArena = (x, y, z, translate = true) => {
    const dx = translate ? x - center[0] : x, dy = translate ? y - center[1] : y, dz = translate ? z - center[2] : z
    return [dz * scale, dy * scale, -dx * scale]
  }
  const reframe = mesh => {
    for (let k = 0; k < mesh.position.length; k += 3) {
      mesh.position.set(toArena(mesh.position[k], mesh.position[k + 1], mesh.position[k + 2]), k)
      mesh.normal.set([mesh.normal[k + 2], mesh.normal[k + 1], -mesh.normal[k]], k)
    }
    return mesh
  }
  const template = wheelCenters.get('LF')
  const templateCenter = [(template[0] + template[1]) / 2, (template[2] + template[3]) / 2, (template[4] + template[5]) / 2]
  reframe(bodyMesh)
  // wheel geometry is centred on its own hub so the client can spin it about z
  for (let k = 0; k < wheelMesh.position.length; k += 3) {
    wheelMesh.position[k] -= templateCenter[0]; wheelMesh.position[k + 1] -= templateCenter[1]; wheelMesh.position[k + 2] -= templateCenter[2]
  }
  for (let k = 0; k < wheelMesh.position.length; k += 3) {
    const p = toArena(wheelMesh.position[k], wheelMesh.position[k + 1], wheelMesh.position[k + 2], false)
    wheelMesh.position.set(p, k)
    wheelMesh.normal.set([wheelMesh.normal[k + 2], wheelMesh.normal[k + 1], -wheelMesh.normal[k]], k)
  }
  const wheels = [...wheelCenters.entries()].sort().map(([key, b]) => {
    const pivot = toArena((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2)
    // the template is a left wheel (source +x); right wheels are the same mesh turned half a turn
    return { key, side: key[0] === 'L' ? -1 : 1, pivot: pivot.map(v => +v.toFixed(5)), yaw: key[0] === 'L' ? 0 : Math.PI }
  })
  const radius = ((template[3] - template[2]) / 2) * scale
  let mastTop = [0, 0, 0]
  for (let k = 0; k < bodyMesh.position.length; k += 3) if (bodyMesh.position[k + 1] > mastTop[1]) mastTop = [bodyMesh.position[k], bodyMesh.position[k + 1], bodyMesh.position[k + 2]]

  await MeshoptSimplifier.ready
  const doc = new Document()
  const buffer = doc.createBuffer()
  const material = doc.createMaterial('perseverance_atlas').setRoughnessFactor(1).setMetallicFactor(1)
    .setBaseColorTexture(attachTexture(doc, 'rover_color', toKtx2(images.color, { srgb: true, mode: 'uastc' })))
    .setNormalTexture(attachTexture(doc, 'rover_normal', toKtx2(images.normal, { srgb: false, mode: 'normal' })))
    .setMetallicRoughnessTexture(attachTexture(doc, 'rover_orm', toKtx2(images.orm, { srgb: false, mode: 'etc1s' })))
  const lods = [
    { body: [0.42, 0.002, []], wheel: [0.25, 0.02, ['Prune']] },
    { body: [0.1, 0.01, ['Prune']], wheel: [0.06, 0.05, ['Prune']] },
    { body: [0.025, 0.04, ['Prune']], wheel: [0.015, 0.2, ['Sloppy']] },
  ]
  const report = []
  const scene = doc.createScene('rover')
  lods.forEach((lod, k) => {
    const b = simplify(bodyMesh, ...lod.body), w = simplify(wheelMesh, ...lod.wheel)
    scene.addChild(addMesh(doc, buffer, `body_lod${k}`, b, material))
    scene.addChild(addMesh(doc, buffer, `wheel_lod${k}`, w, material))
    report.push(`LOD${k}: body ${triangles(b)} + 6 × wheel ${triangles(w)} triangles`)
  })
  scene.setExtras({ wheels, wheelRadius: +radius.toFixed(5), beacon: mastTop.map(v => +v.toFixed(5)), sourceTriangles: triangles(bodyMesh) + 6 * triangles(wheelMesh) })
  await compressGeometry(doc)
  const bytes = await writeGlb(doc, join(OUT, 'rover.glb'))
  console.log(`rover.glb ${(bytes / 1048576).toFixed(2)} MB; ${report.join('; ')}`)
  return { file: 'rover.glb', credit: 'perseverance', lods: report.length }
}

// ---------------------------------------------------------------- props

/** One mesh per variant (its nodes merged), sitting on y = 0, centred in x/z and scaled to the rover. */
function collectVariant(root, names) {
  const builder = new MeshBuilder()
  for (const node of root.listNodes().filter(n => names.includes(n.getName()) && n.getMesh())) {
    const matrix = node.getWorldMatrix()
    const flip = determinant(matrix) < 0
    for (const prim of node.getMesh().listPrimitives()) {
      const pos = prim.getAttribute('POSITION').getArray(), nor = prim.getAttribute('NORMAL').getArray()
      const uv = prim.getAttribute('TEXCOORD_0').getArray(), idx = prim.getIndices().getArray()
      for (let t = 0; t < idx.length; t += 3) {
        for (const v of [idx[t], flip ? idx[t + 2] : idx[t + 1], flip ? idx[t + 1] : idx[t + 2]]) {
          const p = transformPoint(matrix, pos[v * 3], pos[v * 3 + 1], pos[v * 3 + 2])
          const n = transformNormal(matrix, nor[v * 3], nor[v * 3 + 1], nor[v * 3 + 2])
          builder.index.push(builder.add(p[0], p[1], p[2], n[0], n[1], n[2], uv[v * 2], uv[v * 2 + 1]))
        }
      }
    }
  }
  const mesh = builder.build()
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity]
  for (let k = 0; k < mesh.position.length; k += 3) for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], mesh.position[k + a]); hi[a] = Math.max(hi[a], mesh.position[k + a]) }
  const origin = [(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2]
  for (let k = 0; k < mesh.position.length; k += 3) for (let a = 0; a < 3; a++) mesh.position[k + a] = (mesh.position[k + a] - origin[a]) * WORLD_SCALE
  return { mesh, size: [(hi[0] - lo[0]) * WORLD_SCALE, (hi[1] - lo[1]) * WORLD_SCALE, (hi[2] - lo[2]) * WORLD_SCALE] }
}

const PROP_RECIPES = {
  namaqualand_boulder_02: { variants: [['boulder_02']], lod: [3000, 500] },
  namaqualand_boulder_03: { variants: [['boulder_03_LOD0']], lod: [3000, 500] },
  namaqualand_boulder_05: { variants: [['namaqualand_boulder_05']], lod: [2400, 400] },
  namaqualand_stones_01: { variants: [['namaqualand_stones_01_a'], ['namaqualand_stones_01_c'], ['namaqualand_stones_01_e']], lod: [420, 90] },
  wild_rooibos_bush: { variants: [['wild_rooibos_bush_a'], ['wild_rooibos_bush_c']], lod: [1600, 400], alpha: true },
}

async function buildProp(id) {
  const recipe = PROP_RECIPES[id]
  const src = await io.read(await polyHavenModel(id))
  const root = src.getRoot()
  const material = root.listMaterials()[0]
  const image = t => Buffer.from(t.getImage())
  let color = await sharp(image(material.getBaseColorTexture())).removeAlpha().png().toBuffer()
  if (recipe.alpha) {
    const files = await (await fetch(`https://api.polyhaven.com/files/${id}`)).json()
    const alpha = await download(files.Alpha['1k'].jpg.url, `${id}/alpha_1k.jpg`)
    const mask = await sharp(alpha).greyscale().toColourspace('b-w').raw().toBuffer({ resolveWithObject: true })
    color = await sharp(color).joinChannel(mask.data, { raw: { width: mask.info.width, height: mask.info.height, channels: 1 } }).png().toBuffer()
  }
  const doc = new Document()
  const buffer = doc.createBuffer()
  const out = doc.createMaterial(id).setRoughnessFactor(1).setMetallicFactor(material.getMetallicFactor()).setDoubleSided(!!recipe.alpha)
    .setBaseColorTexture(attachTexture(doc, `${id}_color`, toKtx2(color, { srgb: true, mode: 'etc1s' })))
    .setNormalTexture(attachTexture(doc, `${id}_normal`, toKtx2(await sharp(image(material.getNormalTexture())).png().toBuffer(), { srgb: false, mode: 'normal' })))
    .setMetallicRoughnessTexture(attachTexture(doc, `${id}_orm`, toKtx2(await sharp(image(material.getMetallicRoughnessTexture())).png().toBuffer(), { srgb: false, mode: 'etc1s' })))
  if (recipe.alpha) out.setAlphaMode('MASK').setAlphaCutoff(0.5)
  const scene = doc.createScene(id)
  const variants = []
  recipe.variants.forEach((names, v) => {
    const { mesh, size } = collectVariant(root, names)
    recipe.lod.forEach((target, k) => {
      const lod = simplify(mesh, Math.min(1, target / triangles(mesh)), k === 0 ? 0.004 : 0.03, k === 0 ? [] : ['Prune'])
      scene.addChild(addMesh(doc, buffer, `v${v}_lod${k}`, lod, out))
    })
    variants.push({ size: size.map(x => +x.toFixed(4)) })
  })
  scene.setExtras({ variants })
  await compressGeometry(doc)
  const file = `props/${id}.glb`
  const bytes = await writeGlb(doc, join(OUT, file))
  console.log(`${file} ${(bytes / 1048576).toFixed(2)} MB, ${variants.length} variant(s)`)
  return { id, file, credit: id, variants: variants.length, alpha: !!recipe.alpha }
}

// ---------------------------------------------------------------- terrain and skies

/** Ground texture set: colour, normal, and AO / roughness / height packed in one linear map. */
async function buildGround(id) {
  const files = await (await fetch(`https://api.polyhaven.com/files/${id}`)).json()
  const get = (key, res) => download(files[key][res].jpg.url, `${id}/${key}_${res}.jpg`)
  const color = await sharp(await get('Diffuse', '2k')).png().toBuffer()
  const normal = await sharp(await get('nor_gl', '2k')).resize(1024, 1024).png().toBuffer()
  const [arm, height] = await Promise.all([get('arm', '1k'), get('Displacement', '1k')])
  const armRaw = await sharp(arm).removeAlpha().raw().toBuffer({ resolveWithObject: true })
  const heightRaw = await sharp(height).greyscale().resize(armRaw.info.width, armRaw.info.height).raw().toBuffer()
  const packed = Buffer.alloc(armRaw.info.width * armRaw.info.height * 3)
  for (let i = 0; i < heightRaw.length; i++) {
    packed[i * 3] = armRaw.data[i * 3]
    packed[i * 3 + 1] = armRaw.data[i * 3 + 1]
    packed[i * 3 + 2] = heightRaw[i]
  }
  const arh = await sharp(packed, { raw: { width: armRaw.info.width, height: armRaw.info.height, channels: 3 } }).png().toBuffer()
  const info = await (await fetch(`https://api.polyhaven.com/info/${id}`)).json()
  const entry = { id, credit: id, tileMeters: +((info.dimensions?.[0] ?? 2000) / 1000 * WORLD_SCALE).toFixed(3) }
  for (const [name, png, options] of [['color', color, { srgb: true, mode: 'etc1s' }], ['normal', normal, { srgb: false, mode: 'normal' }], ['arh', arh, { srgb: false, mode: 'uastc' }]]) {
    const file = `ground/${id}_${name}.ktx2`
    mkdirSync(join(OUT, 'ground'), { recursive: true })
    writeFileSync(join(OUT, file), toKtx2(png, options))
    entry[name] = file
  }
  console.log(`ground ${id}: tile ${entry.tileMeters} m`)
  return entry
}

async function buildSky({ id, label }) {
  const files = await (await fetch(`https://api.polyhaven.com/files/${id}`)).json()
  const file = `sky/${id}_2k.hdr`
  const path = await download(files.hdri['2k'].hdr.url, file)
  mkdirSync(join(OUT, 'sky'), { recursive: true })
  writeFileSync(join(OUT, file), readFileSync(path))
  return { id, label, file, credit: id }
}

function writeCredits(manifest) {
  const used = new Set([manifest.rover.credit, ...manifest.props.map(p => p.credit), ...manifest.grounds.map(g => g.credit), ...manifest.skies.map(s => s.credit)])
  const lines = ['# Photoreal mode assets', '', 'Generated by `scripts/photoreal-assets/build.mjs` from the sources below. Meshes were re-topologised into LODs, textures re-encoded as KTX2, and the rover materials atlased; the content is otherwise unchanged.', '']
  for (const key of used) {
    const c = CREDITS[key]
    lines.push(`- **${c.title}** by ${c.author}. License: ${c.license}. Source: <${c.url}>`)
  }
  lines.push('', 'NASA material is used under the NASA media usage guidelines (https://www.nasa.gov/nasa-brand-center/images-and-media/); it does not imply endorsement by NASA.', '')
  writeFileSync(join(OUT, 'CREDITS.md'), lines.join('\n'))
}

main().catch(error => { console.error(error); process.exitCode = 1 })

async function main() {
  mkdirSync(OUT, { recursive: true })
  const manifestPath = join(OUT, 'manifest.json')
  const previous = existsSync(manifestPath) ? JSON.parse(readFileSync(manifestPath, 'utf8')) : {}
  const only = flag('--only')
  const want = part => !only || only === part
  const manifest = {
    version: 1,
    worldScale: WORLD_SCALE,
    rover: want('rover') ? await buildRover() : previous.rover,
    props: want('props') ? await sequential(PROPS, buildProp) : previous.props,
    grounds: want('ground') ? await sequential(GROUNDS, buildGround) : previous.grounds,
    skies: want('sky') ? await sequential(SKIES, buildSky) : previous.skies,
  }
  manifest.credits = Object.fromEntries(Object.entries(CREDITS).filter(([key]) =>
    [manifest.rover?.credit, ...(manifest.props ?? []).map(p => p.credit), ...(manifest.grounds ?? []).map(g => g.credit), ...(manifest.skies ?? []).map(s => s.credit)].includes(key)))
  writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`)
  writeCredits(manifest)
  rmSync(TMP, { recursive: true, force: true })
}

async function sequential(items, build) {
  const out = []
  for (const item of items) out.push(await build(item))
  return out
}
