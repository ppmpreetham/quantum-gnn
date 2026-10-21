import * as THREE from 'three'
import { horizonColor, measureSun } from './environment'
import type { PixelSampler, SunEstimate, Vec3 } from './environment'

export interface SkyAnalysis { sun: SunEstimate | null; horizon: Vec3 }

/**
 * Reads an HDRI's sun and horizon colour on the CPU, once per sky, then
 * clamps the solar disc in the texture so image-based light carries the sky
 * only. The directional light restores the sun's measured irradiance, with
 * shadows, instead of the sun being counted twice.
 */
export function analyzeSky(texture: THREE.DataTexture): SkyAnalysis {
  const { data, width, height } = texture.image as { data: Uint16Array | Float32Array; width: number; height: number }
  const channels = data.length / (width * height)
  const half = data instanceof Uint16Array
  const read = (i: number) => (half ? THREE.DataUtils.fromHalfFloat(data[i]) : data[i])
  const write = (i: number, v: number) => { data[i] = half ? THREE.DataUtils.toHalfFloat(v) : v }
  const sample: PixelSampler = (column, row) => {
    const i = (row * width + column) * channels
    return [read(i), read(i + 1), read(i + 2)]
  }
  const horizon = horizonColor(sample, width, height)
  const sun = measureSun(sample, width, height)
  if (sun) {
    for (let row = 0; row < height / 2; row++) for (let column = 0; column < width; column++) {
      const i = (row * width + column) * channels
      const l = 0.2126 * read(i) + 0.7152 * read(i + 1) + 0.0722 * read(i + 2)
      if (l <= sun.threshold) continue
      const k = sun.threshold / l
      for (let c = 0; c < 3; c++) write(i + c, read(i + c) * k)
    }
    texture.needsUpdate = true
  }
  return { sun, horizon }
}
