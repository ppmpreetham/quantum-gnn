import { createContext, useContext, useState } from 'react'
import type { ReactNode } from 'react'
import { Canvas } from '@react-three/fiber'
import type { CanvasProps } from '@react-three/fiber'
import { PerformanceMonitor } from '@react-three/drei'
import * as THREE from 'three'
import { useConsole } from '../state/store'
import type { Tier } from '../state/store'
import { InvalidateOnStore, PerfProbe } from './Probes'

const DPR: Record<Tier, number | [number, number]> = { high: [1, 2], balanced: [1, 1.5], low: 1 }
export const DOWNGRADE: Record<Tier, Tier> = { high: 'balanced', balanced: 'low', low: 'low' }
const DEFAULT_CAMERA: CanvasProps['camera'] = { position: [9.6, 8.4, 10.8], fov: 42, near: 0.1, far: 140 }

const TierContext = createContext<Tier>('high')
/** The quality tier the canvas is rendering at; effects and shadows scale with it. */
export const useTier = () => useContext(TierContext)

/** Chosen quality, stepped down past any tier that drew a blank frame on this GPU (even when picked by hand). */
function useResolvedTier(autoTier: Tier): Tier {
  const quality = useConsole(s => s.quality)
  const blankTiers = useConsole(s => s.blankTiers)
  let tier: Tier = quality === 'auto' ? autoTier : quality
  while (tier !== 'low' && blankTiers.includes(tier)) tier = DOWNGRADE[tier]
  return tier
}

interface SceneCanvasProps {
  children: ReactNode
  onContextLost: (lost: boolean) => void
  camera?: CanvasProps['camera']
  shadows?: CanvasProps['shadows']
  toneMapping?: THREE.ToneMapping
  /** Overrides the console's pause (a scripted mode keeps its own clock). */
  paused?: boolean
}

/**
 * The canvas every mode renders into: device pixel ratio and post-processing
 * by quality tier, automatic tier stepping on slow frames, a paused render
 * loop on demand, context-loss reporting and the frame statistics probe.
 */
export function SceneCanvas({ children, onContextLost, camera = DEFAULT_CAMERA, shadows, toneMapping = THREE.AgXToneMapping, paused: pausedProp }: SceneCanvasProps) {
  const quality = useConsole(s => s.quality)
  const consolePaused = useConsole(s => s.paused)
  const paused = pausedProp ?? consolePaused
  const [autoTier, setAutoTier] = useState<Tier>('high')
  const tier = useResolvedTier(autoTier)

  return (
    <Canvas
      dpr={DPR[tier]}
      frameloop={paused ? 'demand' : 'always'}
      camera={camera}
      shadows={shadows}
      gl={{ antialias: false, powerPreference: 'high-performance', stencil: false, alpha: false }}
      onCreated={({ gl }) => {
        gl.toneMapping = toneMapping
        gl.domElement.addEventListener('webglcontextlost', event => { event.preventDefault(); onContextLost(true) })
        gl.domElement.addEventListener('webglcontextrestored', () => onContextLost(false))
      }}
    >
      <TierContext.Provider value={tier}>{children}</TierContext.Provider>
      <PerfProbe tier={tier} />
      <InvalidateOnStore />
      {!paused && quality === 'auto' && (
        <PerformanceMonitor bounds={() => [45, 120]} flipflops={2}
          onDecline={() => setAutoTier(t => DOWNGRADE[t])} onFallback={() => setAutoTier('low')} />
      )}
    </Canvas>
  )
}
