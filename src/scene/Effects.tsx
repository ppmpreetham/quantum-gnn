import { useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Bloom, EffectComposer, SMAA, ToneMapping, Vignette } from '@react-three/postprocessing'
import { ToneMappingMode } from 'postprocessing'
import type { Tier } from '../state/store'

/** Frames at which the guard samples the screen: once after warm-up, once after the first resize settles. */
const CHECK_FRAMES = new Set([24, 120])
/** Sample points as fractions of the drawing buffer; the scene never renders pure black at any of them. */
const SAMPLES = [[0.5, 0.5], [0.3, 0.35], [0.7, 0.35], [0.3, 0.65], [0.7, 0.65]] as const

/**
 * Some GPU drivers silently fail a post-processing setup and present an
 * all-black frame. The ground and sky are never black, so if every sample
 * point reads (0, 0, 0) right after the composer draws, this tier is broken
 * here. Runs after the composer (priority 2) in the same frame, before the
 * browser clears the drawing buffer.
 */
export function BlankFrameGuard({ onBlank }: { onBlank: () => void }) {
  const gl = useThree(s => s.gl)
  const frame = useRef(0)
  const pixel = useMemo(() => new Uint8Array(4), [])
  useFrame(() => {
    if (!CHECK_FRAMES.has(++frame.current)) return
    const ctx = gl.getContext()
    // a lost context also reads as zeros; that is a different failure with its own notice
    if (ctx.isContextLost()) return
    gl.setRenderTarget(null)
    const width = ctx.drawingBufferWidth, height = ctx.drawingBufferHeight
    const blank = SAMPLES.every(([fx, fy]) => {
      ctx.readPixels(Math.floor(width * fx), Math.floor(height * fy), 1, 1, ctx.RGBA, ctx.UNSIGNED_BYTE, pixel)
      return pixel[0] + pixel[1] + pixel[2] <= 3
    })
    if (blank) onBlank()
  }, 2)
  return null
}

/**
 * Post-processing by quality tier. High: 4× MSAA + bloom. Balanced: SMAA +
 * bloom. Low: no composer at all; the renderer tone maps directly, saving
 * every extra full-screen pass. Screen-space AO was removed: it read the
 * multisampled depth buffer and went fully black on some drivers, and the
 * instanced contact shadows already ground the rovers.
 */
export function Effects({ tier, onBlank }: { tier: Tier; onBlank: () => void }) {
  if (tier === 'low') return null
  const high = tier === 'high'
  return (
    <>
      <EffectComposer multisampling={high ? 4 : 0} enableNormalPass={false}>
        <Bloom mipmapBlur luminanceThreshold={0.92} luminanceSmoothing={0.25} intensity={0.75} radius={0.72} />
        <ToneMapping mode={ToneMappingMode.AGX} />
        <Vignette offset={0.28} darkness={0.5} />
        {high ? <></> : <SMAA />}
      </EffectComposer>
      <BlankFrameGuard key={tier} onBlank={onBlank} />
    </>
  )
}
