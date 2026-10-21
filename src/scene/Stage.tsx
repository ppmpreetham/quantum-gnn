import { Environment, Lightformer } from '@react-three/drei'
import { useConsole } from '../state/store'
import { Effects } from './Effects'
import { FleetLayer, LiveRig } from './FleetLayer'
import { Ground } from './Ground'
import type { RoverModel } from './model/types'
import { SCENE } from './palette'
import { useTier } from './SceneCanvas'

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

/** The matte dusk arena: background, fog, studio light and the ground disc (shared by the console and showcase). */
export function ArenaBackdrop() {
  return (
    <>
      <color attach="background" args={[SCENE.horizon]} />
      <fog attach="fog" args={[SCENE.horizon, SCENE.fogNear, SCENE.fogFar]} />
      <hemisphereLight args={['#cbd7ea', '#2a2723', 0.85]} />
      <directionalLight position={[-6, 10, 4]} intensity={2.3} color="#fff1dc" />
      <directionalLight position={[8, 3.5, -6]} intensity={0.7} color="#9fb4d8" />
      <StudioEnvironment />
      <Ground />
    </>
  )
}

/** The console's scene: a matte dusk arena, studio light and the live fleet, in about seven draw calls. */
export function ConsoleScene({ model }: { model: RoverModel }) {
  const tier = useTier()
  const markBlank = useConsole(s => s.markBlank)
  return (
    <>
      <ArenaBackdrop />
      <FleetLayer model={model} />
      <LiveRig />
      <Effects tier={tier} onBlank={() => markBlank(tier)} />
    </>
  )
}
