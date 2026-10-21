import { use, useDeferredValue, useMemo } from 'react'
import * as THREE from 'three'
import { FleetLayer, LiveRig } from '../../scene/FleetLayer'
import type { CameraHome } from '../../scene/CameraRig'
import type { RoverModel } from '../../scene/model/types'
import { useTier } from '../../scene/SceneCanvas'
import { Haze, SkyLight, SunLight, useSkyState } from './Atmosphere'
import { Dust } from './Dust'
import { sunLight } from './environment'
import { AssetLibraryProvider, useLibrary } from './library'
import { ExposureControl, PhotorealEffects } from './PhotorealEffects'
import { roverFromGltf } from './roverAsset'
import { Scatter } from './Scatter'
import { usePhotoreal } from './store'
import { SurveyStakes } from './SurveyStakes'
import { Terrain } from './Terrain'

/** A low three-quarter view, closer to the ground than the console's survey camera. */
const HOME: CameraHome = { position: new THREE.Vector3(7.4, 2.9, 8.2), target: new THREE.Vector3(0, 0.25, 0) }

function useRoverModel(): RoverModel {
  const { library, manifest } = useLibrary()
  return use(library.cached('rover', async () => library.own(roverFromGltf(await library.gltf(manifest.rover.file)))))
}

function FieldSite() {
  const live = usePhotoreal(s => s.settings)
  // a new sky is loaded in the background while the current one stays on screen
  const settings = useDeferredValue(live)
  const tier = useTier()
  const sky = useSkyState(settings)
  const rover = useRoverModel()
  const sunColor = useMemo(() => new THREE.Color(...sunLight(sky.elevation).color), [sky.elevation])
  const shadows = settings.shadows !== 'off'
  return (
    <>
      <ExposureControl exposure={settings.exposure} curve={settings.toneCurve} />
      <SkyLight sky={sky} settings={settings} />
      <SunLight sky={sky} settings={settings} tier={tier} />
      <Haze sky={sky} settings={settings} />
      <Terrain detail={tier !== 'low'} />
      <Scatter density={settings.scatter * (tier === 'low' ? 0.5 : 1)} shadows={shadows} />
      <SurveyStakes />
      {settings.dust > 0 && <Dust amount={settings.dust} sun={sky.sun} sunColor={sunColor} />}
      <FleetLayer model={rover} shadows={shadows} overlays={settings.overlays} overlayStrength={0.35} shadowAlpha={0.3} />
      <LiveRig home={HOME} minDistance={1.2} autoRotate={settings.autoOrbit} />
      <PhotorealEffects settings={settings} />
    </>
  )
}

/** The photoreal field site. Assets stream in behind a DOM progress overlay; everything is freed on exit. */
export function PhotorealScene() {
  return (
    <AssetLibraryProvider>
      <FieldSite />
    </AssetLibraryProvider>
  )
}
