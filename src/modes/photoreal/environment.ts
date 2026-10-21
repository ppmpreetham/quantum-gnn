/**
 * Photoreal environment model: the settings the panel edits, their
 * defaults, and the pure light maths behind them (solar position for the
 * procedural sky, sun and horizon estimates from an HDRI). No three.js or
 * React here, so it is unit-tested in Node.
 */

export type SkySource = 'procedural' | string
export type ShadowQuality = 'off' | 'soft' | 'crisp' | 'ultra'
export type ToneCurve = 'agx' | 'aces' | 'neutral'

export interface EnvironmentSettings {
  /** An HDRI id from the asset manifest, or the procedural sky. */
  sky: SkySource
  /** Local solar time in hours, used by the procedural sky. */
  hour: number
  /** Turns the HDRI (and the sun found in it) about the vertical axis, in degrees. */
  skyRotation: number
  /** Exposure in stops. */
  exposure: number
  sunIntensity: number
  ambient: number
  /** Haze density; 0 clears the air. */
  haze: number
  shadows: ShadowQuality
  bloom: number
  ambientOcclusion: boolean
  depthOfField: boolean
  filmGrain: boolean
  toneCurve: ToneCurve
  /** Scattered rocks and shrubs, 0 to 1. */
  scatter: number
  /** Dust motes drifting in the light, 0 to 1. */
  dust: number
  /** Links, task columns and labels from the planner. */
  overlays: boolean
  autoOrbit: boolean
}

export const DEFAULT_ENVIRONMENT: Readonly<EnvironmentSettings> = {
  sky: 'goegap',
  hour: 15.5,
  skyRotation: 0,
  exposure: 0,
  sunIntensity: 1,
  ambient: 1,
  haze: 0.35,
  shadows: 'crisp',
  bloom: 0.35,
  ambientOcclusion: true,
  depthOfField: false,
  filmGrain: true,
  toneCurve: 'agx',
  scatter: 0.7,
  dust: 0.45,
  overlays: true,
  autoOrbit: false,
}

/** Shadow map edge length by quality; 'off' disables the sun's shadow. */
export const SHADOW_MAP_SIZE: Record<ShadowQuality, number> = { off: 0, soft: 1024, crisp: 2048, ultra: 4096 }

/** Goegap, Namaqualand, where the HDRIs and ground scans were captured. */
export const SITE_LATITUDE = -29.67
/** Early spring at the site (day of year), which gives a mid-height noon sun. */
export const SITE_DAY = 245

export interface SunPosition { elevation: number; azimuth: number }
export type Vec3 = [number, number, number]

const DEG = Math.PI / 180

/**
 * Approximate solar position (Cooper's declination and the standard
 * hour-angle formula). Azimuth is measured clockwise from north, in radians;
 * elevation is above the horizon.
 */
export function solarPosition(hour: number, latitude = SITE_LATITUDE, day = SITE_DAY): SunPosition {
  const declination = -23.44 * DEG * Math.cos((2 * Math.PI / 365) * (day + 10))
  const hourAngle = (hour - 12) * 15 * DEG
  const phi = latitude * DEG
  const sinElevation = Math.sin(phi) * Math.sin(declination) + Math.cos(phi) * Math.cos(declination) * Math.cos(hourAngle)
  const elevation = Math.asin(Math.max(-1, Math.min(1, sinElevation)))
  const azimuth = Math.atan2(Math.sin(hourAngle), Math.cos(hourAngle) * Math.sin(phi) - Math.tan(declination) * Math.cos(phi)) + Math.PI
  return { elevation, azimuth: ((azimuth % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI) }
}

/** Scene frame: y up, north is -z and east is +x. */
export function sunDirection({ elevation, azimuth }: SunPosition): Vec3 {
  const c = Math.cos(elevation)
  return [Math.sin(azimuth) * c, Math.sin(elevation), -Math.cos(azimuth) * c]
}

/** Rotates a direction about +y by the angle three.js applies through scene.environmentRotation. */
export function rotateY([x, y, z]: Vec3, angle: number): Vec3 {
  const c = Math.cos(angle), s = Math.sin(angle)
  return [x * c + z * s, y, -x * s + z * c]
}

/** Direction of an equirectangular pixel centre; row 0 is the top of the image, as HDR files store it. */
export function equirectDirection(column: number, row: number, width: number, height: number): Vec3 {
  const lon = ((column + 0.5) / width - 0.5) * 2 * Math.PI
  const lat = (0.5 - (row + 0.5) / height) * Math.PI
  return [Math.cos(lat) * Math.cos(lon), Math.sin(lat), Math.cos(lat) * Math.sin(lon)]
}

export type PixelSampler = (column: number, row: number) => Vec3

const luminance = ([r, g, b]: Vec3) => 0.2126 * r + 0.7152 * g + 0.0722 * b

/**
 * Finds the sun in an HDRI: the brightest region of the upper hemisphere,
 * refined to the luminance-weighted centroid of the pixels within a factor
 * of the peak so a clipped disc does not bias the direction. Returns null
 * for skies without a dominant source (overcast), where a soft fill light
 * fits better than a hard sun.
 */
export function findSun(sample: PixelSampler, width: number, height: number, step = 2): { direction: Vec3; strength: number } | null {
  let peak = 0, peakRow = 0, peakColumn = 0, sum = 0, count = 0
  for (let row = 0; row < height / 2; row += step) {
    for (let column = 0; column < width; column += step) {
      const l = luminance(sample(column, row))
      sum += l; count++
      if (l > peak) { peak = l; peakRow = row; peakColumn = column }
    }
  }
  const mean = sum / Math.max(1, count)
  if (peak < mean * 20) return null
  let wx = 0, wy = 0, wz = 0
  const window = Math.ceil(width / 64)
  for (let dr = -window; dr <= window; dr++) {
    for (let dc = -window; dc <= window; dc++) {
      const row = peakRow + dr
      if (row < 0 || row >= height) continue
      const column = (peakColumn + dc + width) % width
      const l = luminance(sample(column, row))
      if (l < peak * 0.25) continue
      const [x, y, z] = equirectDirection(column, row, width, height)
      wx += x * l; wy += y * l; wz += z * l
    }
  }
  const length = Math.hypot(wx, wy, wz) || 1
  return { direction: [wx / length, wy / length, wz / length], strength: peak / mean }
}

export interface SunEstimate { direction: Vec3; irradiance: number; color: Vec3; threshold: number }

/**
 * Measures the sun in an HDRI and reports what to remove from it. Pixels
 * brighter than the threshold form the solar disc; their radiance times
 * solid angle is the sun's irradiance, which a directional light can then
 * deliver with real shadows while the clamped sky provides the fill.
 * Without a dominant disc (overcast) it returns null.
 */
export function measureSun(sample: PixelSampler, width: number, height: number): SunEstimate | null {
  const found = findSun(sample, width, height)
  if (!found) return null
  let sum = 0, count = 0
  for (let row = 0; row < height / 2; row += 4) for (let column = 0; column < width; column += 4) { sum += luminance(sample(column, row)); count++ }
  const threshold = (sum / count) * 12
  const pixelSolidAngle = (2 * Math.PI / width) * (Math.PI / height)
  let irradiance = 0, r = 0, g = 0, b = 0
  for (let row = 0; row < height / 2; row++) {
    const cosLat = Math.cos((0.5 - (row + 0.5) / height) * Math.PI)
    for (let column = 0; column < width; column++) {
      const rgb = sample(column, row)
      const l = luminance(rgb)
      if (l <= threshold) continue
      const excess = (l - threshold) / l
      const weight = pixelSolidAngle * cosLat * excess
      irradiance += l * weight
      r += rgb[0] * weight; g += rgb[1] * weight; b += rgb[2] * weight
    }
  }
  const peak = Math.max(r, g, b) || 1
  return { direction: found.direction, irradiance, color: [r / peak, g / peak, b / peak], threshold }
}

/** Mean colour of a band just above the horizon: the colour distant haze takes on. */
export function horizonColor(sample: PixelSampler, width: number, height: number, bandDegrees = 6, step = 4): Vec3 {
  const rows = Math.max(1, Math.round((bandDegrees / 180) * height))
  const top = Math.floor(height / 2) - rows
  let r = 0, g = 0, b = 0, count = 0
  for (let row = Math.max(0, top); row < height / 2; row++) {
    for (let column = 0; column < width; column += step) {
      const [pr, pg, pb] = sample(column, row)
      r += pr; g += pg; b += pb; count++
    }
  }
  return count ? [r / count, g / count, b / count] : [0.6, 0.65, 0.7]
}

/** Warm, weak light near the horizon, white and strong high up (a simple airmass ramp). */
export function sunLight(elevation: number): { color: Vec3; intensity: number } {
  const e = Math.max(0, elevation)
  const t = Math.min(1, e / (35 * DEG))
  const color: Vec3 = [1, 0.62 + 0.33 * t, 0.38 + 0.52 * t]
  return { color, intensity: Math.sin(e) ** 0.6 * 6.5 }
}
