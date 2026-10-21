import { Beacons } from './Beacons'
import { CameraRig } from './CameraRig'
import type { CameraRigProps } from './CameraRig'
import { FleetMeshes } from './FleetMeshes'
import { GroundDecals } from './GroundDecals'
import { LinkArcs } from './LinkArcs'
import type { RoverModel } from './model/types'
import { FleetDriver, LabelProjector } from './Probes'
import { TaskPicker } from './TaskPicker'

export interface FleetLayerProps {
  model: RoverModel
  /** Rovers cast and receive real shadows (photoreal) instead of relying on the contact decal alone. */
  shadows?: boolean
  /** Selection ring, radio range and delivery path of the selected rover; off in the scripted showcase. */
  selection?: boolean
  shadowAlpha?: number
  /** Planner overlays: links and task columns. */
  overlays?: boolean
  /** Opacity of the links and task columns; the photoreal site keeps them subtle. */
  overlayStrength?: number
}

/** Everything drawn for one fleet view: ground markers, rovers, radio links and task columns. */
export function FleetLayer({ model, shadows = false, selection = true, shadowAlpha, overlays = true, overlayStrength = 1 }: FleetLayerProps) {
  return (
    <>
      <GroundDecals selection={selection} shadowAlpha={shadowAlpha} />
      <FleetMeshes model={model} selectable={selection} shadows={shadows} />
      {overlays && <LinkArcs selection={selection} strength={overlayStrength} />}
      {overlays && <Beacons strength={overlayStrength} />}
    </>
  )
}

/** Per-canvas machinery of the live mission: the fixed-step driver, camera rig, DOM labels and the ground picker. */
export function LiveRig(camera: CameraRigProps) {
  return (
    <>
      <FleetDriver />
      <CameraRig {...camera} />
      <LabelProjector />
      <TaskPicker />
    </>
  )
}
