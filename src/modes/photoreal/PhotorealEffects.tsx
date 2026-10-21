import { useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Bloom, DepthOfField, EffectComposer, N8AO, Noise, SMAA, ToneMapping, Vignette } from '@react-three/postprocessing'
import { BlendFunction, ToneMappingMode } from 'postprocessing'
import type { DepthOfFieldEffect } from 'postprocessing'
import * as THREE from 'three'
import { BlankFrameGuard } from '../../scene/Effects'
import { useTier } from '../../scene/SceneCanvas'
import { useScene } from '../../state/fleetContext'
import { useConsole } from '../../state/store'
import type { EnvironmentSettings, ToneCurve } from './environment'

const RENDERER_CURVE: Record<ToneCurve, THREE.ToneMapping> = {
  agx: THREE.AgXToneMapping, aces: THREE.ACESFilmicToneMapping, neutral: THREE.NeutralToneMapping,
}
const PASS_CURVE: Record<ToneCurve, ToneMappingMode> = {
  agx: ToneMappingMode.AGX, aces: ToneMappingMode.ACES_FILMIC, neutral: ToneMappingMode.NEUTRAL,
}

/** Exposure (stops) and tone curve on the renderer; the composer's tone mapping reads the same exposure. */
export function ExposureControl({ exposure, curve }: { exposure: number; curve: ToneCurve }) {
  const gl = useThree(s => s.gl)
  const invalidate = useThree(s => s.invalidate)
  useEffect(() => {
    const previous = { exposure: gl.toneMappingExposure, mapping: gl.toneMapping }
    gl.toneMappingExposure = 2 ** exposure
    gl.toneMapping = RENDERER_CURVE[curve]
    invalidate()
    return () => { gl.toneMappingExposure = previous.exposure; gl.toneMapping = previous.mapping }
  }, [gl, exposure, curve, invalidate])
  return null
}

/** Keeps a depth-of-field effect focused on the selected rover. */
function useFocusOnSelection() {
  const { poses } = useScene()
  const focus = useMemo(() => new THREE.Vector3(), [])
  const effect = useRef<DepthOfFieldEffect>(null)
  useLayoutEffect(() => { if (effect.current) effect.current.target = focus })
  useFrame(() => {
    const i = Math.min(useConsole.getState().selected, poses.count - 1)
    focus.set(poses.x[i], 0.35, poses.z[i])
  })
  return effect
}

/**
 * Post-processing by tier. High: ambient occlusion at full resolution,
 * bloom, optional depth of field, film grain and SMAA. Balanced: half-res
 * AO. Low: none; the renderer tone maps directly. Multisampling stays off:
 * AO must read a single-sampled depth buffer, the failure that once blacked
 * out the console, and the blank-frame guard still watches every tier.
 */
export function PhotorealEffects({ settings }: { settings: EnvironmentSettings }) {
  const tier = useTier()
  const markBlank = useConsole(s => s.markBlank)
  const dof = useFocusOnSelection()
  if (tier === 'low') return null
  const high = tier === 'high'
  return (
    <>
      <EffectComposer multisampling={0} enableNormalPass={false} frameBufferType={THREE.HalfFloatType}>
        <N8AO enabled={settings.ambientOcclusion} halfRes={!high} quality={high ? 'medium' : 'performance'}
          aoRadius={0.45} distanceFalloff={0.5} intensity={2.4} color="black" />
        {settings.depthOfField ? <DepthOfField ref={dof} worldFocusRange={3.2} bokehScale={3.5} /> : <></>}
        <Bloom mipmapBlur luminanceThreshold={1.1} luminanceSmoothing={0.3} intensity={settings.bloom} radius={0.75} />
        <ToneMapping mode={PASS_CURVE[settings.toneCurve]} />
        {settings.filmGrain ? <Noise premultiply blendFunction={BlendFunction.SOFT_LIGHT} opacity={0.22} /> : <></>}
        <Vignette offset={0.3} darkness={0.45} />
        <SMAA />
      </EffectComposer>
      <BlankFrameGuard key={tier} onBlank={() => markBlank(tier)} />
    </>
  )
}
