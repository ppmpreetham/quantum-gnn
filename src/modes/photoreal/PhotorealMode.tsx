import { Suspense, useEffect, useState } from 'react'
import { useProgress } from '@react-three/drei'
import { SceneCanvas } from '../../scene/SceneCanvas'
import { useConsole } from '../../state/store'
import { LabelLayer } from '../../ui/LabelLayer'
import { MissionStrip } from '../../ui/MissionStrip'
import { MissionTab } from '../../ui/panels/MissionTab'
import { CameraField, OverlaysField, QualityField } from '../../ui/panels/ViewTab'
import { StageFrame } from '../../ui/shell/StageFrame'
import { TabPanel } from '../../ui/TabPanel'
import type { PanelTab } from '../../ui/TabPanel'
import { TaskHint } from '../../ui/TaskHint'
import { CameraSwitch, LiveStatus, TopBar } from '../../ui/TopBar'
import { EnvironmentPanel } from './EnvironmentPanel'
import { PhotorealScene } from './PhotorealScene'
import { usePhotoreal } from './store'

const CAMERA = { position: [7.4, 2.9, 8.2] as [number, number, number], fov: 38, near: 0.05, far: 900 }
function LoadingOverlay() {
  const { active, progress, item } = useProgress()
  const [shown, setShown] = useState(true)
  useEffect(() => {
    if (active) { setShown(true); return }
    const timer = setTimeout(() => setShown(false), 400)
    return () => clearTimeout(timer)
  }, [active])
  if (!shown) return null
  const file = item?.split('/').pop()
  return (
    <div className={`field-loading ${active ? '' : 'done'}`} role="status" aria-live="polite">
      <b>Preparing the field site</b>
      <span className="field-loading-bar"><i style={{ width: `${Math.round(progress)}%` }} /></span>
      <small>{active && file ? `Streaming ${file}` : 'Compiling shaders'}</small>
    </div>
  )
}

const TABS: readonly PanelTab[] = [
  {
    id: 'environment', label: 'Environment',
    content: () => <Suspense fallback={<p className="tab-body field-hint">Loading the asset list…</p>}><EnvironmentPanel /></Suspense>,
  },
  { id: 'mission', label: 'Mission', content: () => <MissionTab /> },
  { id: 'view', label: 'View', content: () => <div className="tab-body"><CameraField /><OverlaysField /><QualityField /></div> },
]

/** H hides every panel for an unobstructed view. */
function useHudKey() {
  const toggle = usePhotoreal(s => s.toggleHud)
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey || (event.target as HTMLElement | null)?.closest('input, select, textarea')) return
      if (event.key === 'h' || event.key === 'H') toggle()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [toggle])
}

/**
 * The live mission on a photoreal desert test site: same rovers, planner
 * and results as the console, rendered with scanned terrain, captured skies
 * and a NASA rover model.
 */
export function PhotorealMode() {
  useHudKey()
  const hudHidden = usePhotoreal(s => s.hudHidden)
  const toggleHud = usePhotoreal(s => s.toggleHud)
  const labels = useConsole(s => s.overlays.labels)
  const overlays = usePhotoreal(s => s.settings.overlays)
  return (
    <div className={`console photoreal ${hudHidden ? 'hud-hidden' : ''}`}>
      <StageFrame overlay={<>{labels && overlays && <LabelLayer />}<TaskHint /><LoadingOverlay /></>}>
        {onContextLost => (
          <SceneCanvas onContextLost={onContextLost} camera={CAMERA} shadows="percentage">
            <PhotorealScene />
          </SceneCanvas>
        )}
      </StageFrame>
      {!hudHidden && (
        <>
          <TopBar center={<CameraSwitch />}><LiveStatus /></TopBar>
          <TabPanel label="Photoreal controls" tabs={TABS} />
          <MissionStrip />
        </>
      )}
      <button type="button" className="hud-toggle" onClick={toggleHud} title="H">{hudHidden ? 'Show panels' : 'Hide panels'}</button>
      <p className="attribution">Rover: NASA/JPL-Caltech. Terrain, rocks and skies: Poly Haven (CC0).</p>
    </div>
  )
}
