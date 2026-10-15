import { useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { Environment, Lightformer, PerformanceMonitor } from '@react-three/drei'
import * as THREE from 'three'
import { useConsole } from '../state/store'
import type { Tier } from '../state/store'
import { Beacons } from './Beacons'
import { CameraRig } from './CameraRig'
import { Effects } from './Effects'
import { FleetMeshes } from './FleetMeshes'
import { Ground } from './Ground'
import { GroundDecals } from './GroundDecals'
import { LinkArcs } from './LinkArcs'
import type { RoverModel } from './model/types'
import { SCENE } from './palette'
import { FleetDriver, InvalidateOnStore, LabelProjector, PerfProbe } from './Probes'

const DPR: Record<Tier, number | [number, number]> = { high: [1, 2], balanced: [1, 1.5], low: 1 }
const DOWNGRADE: Record<Tier, Tier> = { high: 'balanced', balanced: 'low', low: 'low' }

/** Studio reflections from a few emissive cards, rendered once into a cube map: no HDR download. */
function StudioEnvironment() {
  return (
    <Environment resolution={128} frames={1}>
      <Lightformer form="rect" intensity={2.4} color="#fff3e0" position={[0, 6, -4]} scale={[10, 3, 1]} />
      <Lightformer form="rect" intensity={1.2} color="#bcd0f0" position={[-6, 3, 4]} rotation-y={Math.PI / 2} scale={[8, 2, 1]} />
      <Lightformer form="ring" intensity={1.6} color="#ffd9a0" position={[6, 4, 2]} scale={2} />
      <Lightformer form="rect" intensity={0.5} color="#2b3140" position={[0, -2, 0]} rotation-x={Math.PI / 2} scale={[20, 20, 1]} />
    </Environment>
  )
}

export function Stage({ model, onContextLost }: { model: RoverModel; onContextLost: (lost: boolean) => void }) {
  const quality = useConsole(s => s.quality)
  const paused = useConsole(s => s.paused)
  const [autoTier, setAutoTier] = useState<Tier>('high')
  const blankTiers = useConsole(s => s.blankTiers)
  const markBlank = useConsole(s => s.markBlank)
  let tier: Tier = quality === 'auto' ? autoTier : quality
  // a tier that rendered blank on this GPU is skipped, even when chosen by hand
  while (tier !== 'low' && blankTiers.includes(tier)) tier = DOWNGRADE[tier]

  return (
    <Canvas
      dpr={DPR[tier]}
      frameloop={paused ? 'demand' : 'always'}
      camera={{ position: [9.6, 8.4, 10.8], fov: 42, near: 0.1, far: 140 }}
      gl={{ antialias: false, powerPreference: 'high-performance', stencil: false, alpha: false }}
      onCreated={({ gl }) => {
        gl.toneMapping = THREE.AgXToneMapping
        gl.domElement.addEventListener('webglcontextlost', event => { event.preventDefault(); onContextLost(true) })
        gl.domElement.addEventListener('webglcontextrestored', () => onContextLost(false))
      }}
    >
      <color attach="background" args={[SCENE.horizon]} />
      <fog attach="fog" args={[SCENE.horizon, SCENE.fogNear, SCENE.fogFar]} />
      <hemisphereLight args={['#cbd7ea', '#2a2723', 0.85]} />
      <directionalLight position={[-6, 10, 4]} intensity={2.3} color="#fff1dc" />
      <directionalLight position={[8, 3.5, -6]} intensity={0.7} color="#9fb4d8" />
      <StudioEnvironment />
      <FleetDriver />
      <Ground />
      <GroundDecals />
      <FleetMeshes model={model} />
      <LinkArcs />
      <Beacons />
      <CameraRig />
      <LabelProjector />
      <PerfProbe tier={tier} />
      <InvalidateOnStore />
      {!paused && quality === 'auto' && (
        <PerformanceMonitor bounds={() => [45, 120]} flipflops={2}
          onDecline={() => setAutoTier(t => DOWNGRADE[t])} onFallback={() => setAutoTier('low')} />
      )}
      <Effects tier={tier} onBlank={() => markBlank(tier)} />
    </Canvas>
  )
}
