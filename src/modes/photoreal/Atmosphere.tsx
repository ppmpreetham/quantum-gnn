import { use, useEffect, useLayoutEffect, useMemo } from 'react'
import { Environment } from '@react-three/drei'
import * as THREE from 'three'
import { Sky } from 'three/examples/jsm/objects/Sky.js'
import type { Tier } from '../../state/store'
import { SHADOW_MAP_SIZE, rotateY, solarPosition, sunDirection, sunLight } from './environment'
import type { EnvironmentSettings, Vec3 } from './environment'
import { useLibrary } from './library'
import { analyzeSky } from './skyAnalysis'

const DEG = Math.PI / 180
/** Overcast skies have no dominant source: a high, soft fill stands in for the sun. */
const DIFFUSE_SUN: Vec3 = [0.25, 0.92, 0.3]
const FOG_SEED = '#b8b2a8'
const PROCEDURAL_SKY_SCALE = 0.25
const TIER_SHADOW_CAP: Record<Tier, number> = { high: 4096, balanced: 2048, low: 1024 }

export interface SkyState {
  /** Unit vector toward the sun in the scene frame. */
  sun: Vec3
  elevation: number
  /** False for skies without a dominant source (overcast HDRIs). */
  direct: boolean
  /** Measured solar irradiance and colour for HDRI skies; null means use the airmass model. */
  measured: { irradiance: number; color: Vec3 } | null
  /** Linear colour of the horizon haze. */
  horizon: THREE.Color
  hdr: THREE.DataTexture | null
}

/** Clear-air horizon colour for the procedural sky: blue-grey by day, amber near sunset. */
function proceduralHorizon(elevation: number): THREE.Color {
  const t = Math.min(1, Math.max(0, elevation / (25 * DEG)))
  return new THREE.Color(0.95, 0.62, 0.42).lerp(new THREE.Color(0.62, 0.7, 0.82), t).multiplyScalar(0.35 + 0.65 * t)
}

/** The sky the settings ask for: an HDRI with the sun and haze measured from its pixels, or the procedural sky. */
export function useSkyState(settings: EnvironmentSettings): SkyState {
  const { library, manifest } = useLibrary()
  const rotation = settings.skyRotation * DEG
  const sky = manifest.skies.find(s => s.id === settings.sky)
  const hdr = sky ? use(library.hdr(sky.file)) : null
  const analysis = sky && hdr ? use(library.cached(`sky:${sky.id}`, async () => analyzeSky(hdr))) : null
  return useMemo(() => {
    if (!hdr || !analysis) {
      const position = solarPosition(settings.hour)
      const sun = rotateY(sunDirection(position), rotation)
      return { sun, elevation: position.elevation, direct: position.elevation > 0, measured: null, horizon: proceduralHorizon(position.elevation), hdr: null }
    }
    const sun = rotateY(analysis.sun?.direction ?? DIFFUSE_SUN, rotation)
    const measured = analysis.sun && { irradiance: analysis.sun.irradiance, color: analysis.sun.color }
    return { sun, elevation: Math.asin(sun[1]), direct: analysis.sun !== null, measured, horizon: new THREE.Color(...analysis.horizon), hdr }
  }, [hdr, analysis, settings.hour, rotation])
}

/** Image-based light and backdrop from the HDRI, or from the procedural sky rendered into a cube map. */
export function SkyLight({ sky, settings }: { sky: SkyState; settings: EnvironmentSettings }) {
  const rotation: [number, number, number] = [0, settings.skyRotation * DEG, 0]
  if (sky.hdr) {
    return (
      <Environment map={sky.hdr} background backgroundRotation={rotation} environmentRotation={rotation}
        environmentIntensity={settings.ambient} backgroundIntensity={1} />
    )
  }
  return <ProceduralSky sun={sky.sun} intensity={settings.ambient} />
}

function ProceduralSky({ sun, intensity }: { sun: Vec3; intensity: number }) {
  const [backdrop, probe] = useMemo(() => {
    const backdrop = new Sky()
    backdrop.scale.setScalar(1000)
    const u = backdrop.material.uniforms
    u.turbidity.value = 3.2
    u.rayleigh.value = 1.4
    u.mieCoefficient.value = 0.004
    u.mieDirectionalG.value = 0.82
    u.cloudCoverage.value = 0.32
    u.cloudDensity.value = 0.35
    // Sky.js radiance is calibrated for an exposure of about 0.5; scale it to the HDRIs' units so one exposure fits all skies
    const material = backdrop.material
    material.uniforms.skyIntensity = { value: PROCEDURAL_SKY_SCALE }
    material.fragmentShader = material.fragmentShader
      .replace('uniform float time;', 'uniform float time;\nuniform float skyIntensity;')
      .replace('gl_FragColor = vec4( texColor, 1.0 );', 'gl_FragColor = vec4( texColor * skyIntensity, 1.0 );')
    const probe = new THREE.Mesh(backdrop.geometry, material)
    probe.scale.setScalar(1000)
    return [backdrop, probe]
  }, [])
  useEffect(() => () => { backdrop.geometry.dispose(); backdrop.material.dispose() }, [backdrop])
  useLayoutEffect(() => { backdrop.material.uniforms.sunPosition.value.set(...sun) }, [backdrop, sun])
  // the lighting probe re-renders only when the sun moves by about a degree or more
  const key = sun.map(v => Math.round(v * 60)).join(':')
  return (
    <>
      <primitive object={backdrop} />
      <Environment key={key} resolution={128} frames={1} environmentIntensity={intensity}>
        <primitive object={probe} />
      </Environment>
    </>
  )
}

/** The sun: a directional light aimed at the arena whose shadow frustum just covers it. */
export function SunLight({ sky, settings, tier }: { sky: SkyState; settings: EnvironmentSettings; tier: Tier }) {
  const size = Math.min(SHADOW_MAP_SIZE[settings.shadows], TIER_SHADOW_CAP[tier])
  const model = sunLight(sky.elevation)
  const color = sky.measured?.color ?? model.color
  const strength = (sky.measured ? sky.measured.irradiance : model.intensity) * settings.sunIntensity * (sky.direct ? 1 : 0.3)
  const position = sky.sun.map(v => v * 40) as Vec3
  const tint = useMemo(() => new THREE.Color(...color), [color[0], color[1], color[2]])
  return (
    // a new shadow map size needs a new light: three.js allocates the map once
    <directionalLight key={size} position={position} intensity={strength} color={tint}
      castShadow={size > 0} shadow-mapSize={[size, size]} shadow-bias={-0.0005} shadow-normalBias={0.04}
      shadow-radius={sky.direct ? 2.5 : 8} shadow-camera-left={-9} shadow-camera-right={9}
      shadow-camera-top={9} shadow-camera-bottom={-9} shadow-camera-near={1} shadow-camera-far={90} />
  )
}

/** Exponential haze tinted by the measured horizon, so the ground fades into the sky behind it. */
export function Haze({ sky, settings }: { sky: SkyState; settings: EnvironmentSettings }) {
  // constant args keep one fog object; colour and density update in place
  return <fogExp2 attach="fog" args={[FOG_SEED]} color={sky.horizon} density={settings.haze * 0.012} />
}
