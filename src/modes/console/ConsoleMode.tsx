import { useRoverModel } from '../../hooks/useRoverModel'
import { SceneCanvas } from '../../scene/SceneCanvas'
import { ConsoleScene } from '../../scene/Stage'
import { LabelLayer } from '../../ui/LabelLayer'
import { Minimap } from '../../ui/Minimap'
import { MissionStrip } from '../../ui/MissionStrip'
import { MissionPanel } from '../../ui/panels/MissionPanel'
import { Roster } from '../../ui/Roster'
import { StageFrame } from '../../ui/shell/StageFrame'
import { TaskHint } from '../../ui/TaskHint'
import { CameraSwitch, LiveStatus, TopBar } from '../../ui/TopBar'

/** The original console: the dusk arena, mission panel, minimap, roster and results strip. */
export function ConsoleMode() {
  const { model, controls } = useRoverModel()
  return (
    <div className="console">
      <StageFrame onFile={controls.onFile} dropHint="Drop a .glb to replace the rover model"
        overlay={<><LabelLayer /><TaskHint /></>}>
        {onContextLost => (
          <SceneCanvas onContextLost={onContextLost}>
            <ConsoleScene model={model} />
          </SceneCanvas>
        )}
      </StageFrame>
      <TopBar center={<CameraSwitch />}><LiveStatus /></TopBar>
      <MissionPanel model={controls} />
      <aside className="panel fleet-column" aria-label="Fleet map and roster">
        <Minimap />
        <Roster />
      </aside>
      <MissionStrip />
    </div>
  )
}
