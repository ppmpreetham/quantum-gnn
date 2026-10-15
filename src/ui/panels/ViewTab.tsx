import { useRef } from 'react'
import type { ForwardAxis } from '../../scene/model/types'
import { useConsole } from '../../state/store'
import type { CameraMode, OverlayKey, QualityMode } from '../../state/store'
import { Segmented, Select, Toggle } from '../controls'

const OVERLAYS: { key: OverlayKey; label: string; hint: string }[] = [
  { key: 'labels', label: 'Rover labels', hint: 'Name tags above each rover.' },
  { key: 'candidates', label: 'Unused links', hint: 'Links in radio range that the planner pruned.' },
  { key: 'ranges', label: 'Radio range for every rover', hint: 'The selected rover always shows its range.' },
  { key: 'flow', label: 'Data flow', hint: 'Packets moving along the delivery tree toward the root.' },
  { key: 'path', label: 'Selected delivery path', hint: 'The widest path from the selected rover to the root.' },
  { key: 'beacons', label: 'Task sites', hint: 'Columns where tasks were dispatched and how they ended.' },
]

export const CAMERA_OPTIONS: { value: CameraMode; label: string; title: string }[] = [
  { value: 'orbit', label: 'Orbit', title: 'Free orbit (1)' },
  { value: 'top', label: 'Top', title: 'Overhead map view (2)' },
  { value: 'follow', label: 'Follow', title: 'Track the selected rover (3)' },
  { value: 'chase', label: 'Chase', title: 'Ride behind the selected rover (4)' },
]

export interface ModelControls {
  name: string; error: string | null; loading: boolean; forward: ForwardAxis; isCustom: boolean
  onFile: (file: File) => void; onForward: (axis: ForwardAxis) => void; onReset: () => void
}

export function ViewTab({ model }: { model: ModelControls }) {
  const camera = useConsole(s => s.camera)
  const setCamera = useConsole(s => s.setCamera)
  const overlays = useConsole(s => s.overlays)
  const toggleOverlay = useConsole(s => s.toggleOverlay)
  const quality = useConsole(s => s.quality)
  const setQuality = useConsole(s => s.setQuality)
  const perf = useConsole(s => s.perf)
  const paused = useConsole(s => s.paused)
  const blankTiers = useConsole(s => s.blankTiers)
  const file = useRef<HTMLInputElement>(null)

  return (
    <div className="tab-body">
      <div className="field">
        <span className="field-label">Camera</span>
        <Segmented label="Camera" value={camera} options={CAMERA_OPTIONS} onChange={setCamera} />
        <p className="field-hint">Drag to orbit, right-drag to pan, scroll to zoom. Click a rover to select it.</p>
      </div>

      <fieldset className="param-group">
        <legend>Overlays</legend>
        {OVERLAYS.map(o => <Toggle key={o.key} label={o.label} hint={o.hint} checked={overlays[o.key]} onChange={() => toggleOverlay(o.key)} />)}
      </fieldset>

      <div className="field">
        <span className="field-label">Graphics quality</span>
        <Segmented<QualityMode> label="Graphics quality" value={quality} onChange={setQuality} options={[
          { value: 'auto', label: 'Auto', title: 'Steps down when the frame rate drops' },
          { value: 'high', label: 'High', title: 'MSAA, ambient occlusion, bloom' },
          { value: 'balanced', label: 'Balanced', title: 'SMAA and bloom' },
          { value: 'low', label: 'Low', title: 'No post-processing' },
        ]} />
        <dl className="perf">
          <div><dt>Frame rate</dt><dd>{paused ? 'Idle' : `${perf.fps} fps`}</dd></div>
          <div><dt>Draw calls</dt><dd>{perf.calls}</dd></div>
          <div><dt>Triangles</dt><dd>{perf.triangles.toLocaleString()}</dd></div>
          <div><dt>Tier</dt><dd>{perf.tier}</dd></div>
        </dl>
        {blankTiers.length > 0 && (
          <p className="field-hint" role="status">
            Post-processing at {blankTiers.join(' and ')} quality drew a blank frame on this GPU, so the console uses {perf.tier} instead.
          </p>
        )}
      </div>

      <fieldset className="param-group">
        <legend>Rover model</legend>
        <p className="model-name">{model.name}</p>
        <div className="row">
          <button type="button" className="ghost" disabled={model.loading} onClick={() => file.current?.click()}>
            {model.loading ? 'Loading…' : 'Import Blender .glb'}
          </button>
          {model.isCustom && <button type="button" className="ghost" onClick={model.onReset}>Use SDF rover</button>}
        </div>
        <input ref={file} type="file" accept=".glb,model/gltf-binary" hidden
          onChange={event => { const f = event.target.files?.[0]; if (f) model.onFile(f); event.target.value = '' }} />
        {model.isCustom && (
          <Select label="Model faces" value={model.forward} onChange={v => model.onForward(v as ForwardAxis)} options={[
            { value: 'auto', label: 'Detect from shape' }, { value: '+x', label: '+X' }, { value: '-x', label: '−X' },
            { value: '+z', label: '+Z (Blender −Y)' }, { value: '-z', label: '−Z' },
          ]} />
        )}
        {model.error && <p className="error" role="alert">{model.error}</p>}
        <p className="field-hint">You can also drop a .glb onto the scene. Name wheel meshes with “wheel” so they spin; the model is scaled to the 0.9 m SDF footprint.</p>
      </fieldset>
    </div>
  )
}
