import { use } from 'react'
import { Segmented, Select, Slider, Toggle } from '../../ui/controls'
import { loadManifest } from './assets'
import type { Manifest } from './assets'
import { solarPosition } from './environment'
import type { EnvironmentSettings, ShadowQuality, ToneCurve } from './environment'
import { usePhotoreal } from './store'

const SHADOWS: { value: ShadowQuality; label: string; title: string }[] = [
  { value: 'off', label: 'Off', title: 'No sun shadows' },
  { value: 'soft', label: 'Soft', title: '1024 px shadow map' },
  { value: 'crisp', label: 'Crisp', title: '2048 px shadow map' },
  { value: 'ultra', label: 'Ultra', title: '4096 px shadow map (high tier only)' },
]
const CURVES: { value: ToneCurve; label: string; title: string }[] = [
  { value: 'agx', label: 'AgX', title: 'Filmic, desaturates highlights gracefully' },
  { value: 'aces', label: 'ACES', title: 'Punchier contrast, the classic film curve' },
  { value: 'neutral', label: 'Neutral', title: 'Khronos PBR Neutral: true-to-material colour' },
]

const clock = (hour: number) => `${String(Math.floor(hour)).padStart(2, '0')}:${String(Math.round((hour % 1) * 60) % 60).padStart(2, '0')}`
const degrees = (radians: number) => `${Math.round((radians * 180) / Math.PI)}°`
const stops = (ev: number) => `${ev >= 0 ? '+' : '−'}${Math.abs(ev).toFixed(1)} EV`
const pct = (v: number) => `${Math.round(v * 100)}%`

function useSetting<K extends keyof EnvironmentSettings>(key: K): [EnvironmentSettings[K], (value: EnvironmentSettings[K]) => void] {
  const value = usePhotoreal(s => s.settings[key])
  const set = usePhotoreal(s => s.set)
  return [value, next => set(key, next)]
}

function SkyFields({ manifest }: { manifest: Manifest }) {
  const [sky, setSky] = useSetting('sky')
  const [hour, setHour] = useSetting('hour')
  const [rotation, setRotation] = useSetting('skyRotation')
  const procedural = !manifest.skies.some(s => s.id === sky)
  const sun = solarPosition(hour)
  return (
    <fieldset className="param-group">
      <legend>Sky and sun</legend>
      <Select label="Sky" value={procedural ? 'procedural' : sky} onChange={setSky} options={[
        ...manifest.skies.map(s => ({ value: s.id, label: s.label, group: 'Captured HDRI (Namaqualand)' })),
        { value: 'procedural', label: 'Physical sky, any time of day', group: 'Simulated' },
      ]} hint={procedural
        ? `Sun at ${degrees(sun.elevation)} elevation. Atmospheric scattering and clouds are computed, and the scene's light is re-captured from the sky as the sun moves.`
        : 'A real sky photographed at the site the ground was scanned. The sun light is aimed at the brightest point of the image, so shadows match it.'} />
      {procedural && <Slider label="Time of day" value={hour} min={5.5} max={18.6} step={0.05} display={clock(hour)} onChange={setHour}
        help="Local solar time at Goegap, Namaqualand (29.7° S) in early spring." />}
      <Slider label="Site heading" value={rotation} min={-180} max={180} step={1} display={`${rotation}°`} onChange={setRotation}
        help="Turns the sky and the sun together around the arena." />
    </fieldset>
  )
}

function LightFields() {
  const [exposure, setExposure] = useSetting('exposure')
  const [sun, setSun] = useSetting('sunIntensity')
  const [ambient, setAmbient] = useSetting('ambient')
  const [haze, setHaze] = useSetting('haze')
  const [shadows, setShadows] = useSetting('shadows')
  const [curve, setCurve] = useSetting('toneCurve')
  return (
    <fieldset className="param-group">
      <legend>Light and atmosphere</legend>
      <Slider label="Exposure" value={exposure} min={-2} max={2} step={0.1} display={stops(exposure)} onChange={setExposure} />
      <Slider label="Sun strength" value={sun} min={0} max={2} step={0.05} display={pct(sun)} onChange={setSun} />
      <Slider label="Sky light" value={ambient} min={0} max={2} step={0.05} display={pct(ambient)} onChange={setAmbient}
        help="Image-based light from the sky: fills shadows and drives reflections." />
      <Slider label="Haze" value={haze} min={0} max={1} step={0.01} display={pct(haze)} onChange={setHaze}
        help="Aerial perspective: distant ground fades into the measured horizon colour." />
      <div className="field">
        <span className="field-label">Sun shadows</span>
        <Segmented label="Sun shadows" value={shadows} options={SHADOWS} onChange={setShadows} />
      </div>
      <div className="field">
        <span className="field-label">Tone curve</span>
        <Segmented label="Tone curve" value={curve} options={CURVES} onChange={setCurve} />
      </div>
    </fieldset>
  )
}

function LensFields() {
  const [bloom, setBloom] = useSetting('bloom')
  const [ao, setAo] = useSetting('ambientOcclusion')
  const [dof, setDof] = useSetting('depthOfField')
  const [grain, setGrain] = useSetting('filmGrain')
  return (
    <fieldset className="param-group">
      <legend>Camera and lens</legend>
      <Slider label="Bloom" value={bloom} min={0} max={1.5} step={0.05} display={pct(bloom)} onChange={setBloom} />
      <Toggle label="Ambient occlusion" checked={ao} onChange={() => setAo(!ao)} hint="Screen-space contact shading (balanced and high tiers)." />
      <Toggle label="Depth of field on the selected rover" checked={dof} onChange={() => setDof(!dof)} />
      <Toggle label="Film grain" checked={grain} onChange={() => setGrain(!grain)} />
    </fieldset>
  )
}

function SceneFields() {
  const [scatter, setScatter] = useSetting('scatter')
  const [dust, setDust] = useSetting('dust')
  const [overlays, setOverlays] = useSetting('overlays')
  const [orbit, setOrbit] = useSetting('autoOrbit')
  return (
    <fieldset className="param-group">
      <legend>Scene</legend>
      <Slider label="Rocks and shrubs" value={scatter} min={0} max={1} step={0.05} display={pct(scatter)} onChange={setScatter} />
      <Slider label="Dust in the air" value={dust} min={0} max={1} step={0.05} display={pct(dust)} onChange={setDust} />
      <Toggle label="Planner overlays" checked={overlays} onChange={() => setOverlays(!overlays)} hint="Radio links, task columns and markers from the live planner." />
      <Toggle label="Slow auto-orbit" checked={orbit} onChange={() => setOrbit(!orbit)} />
    </fieldset>
  )
}

/** Everything that shapes the photoreal image, grouped the way a photographer thinks about it. */
export function EnvironmentPanel() {
  const manifest = use(loadManifest())
  const reset = usePhotoreal(s => s.reset)
  return (
    <div className="tab-body">
      <SkyFields manifest={manifest} />
      <LightFields />
      <LensFields />
      <SceneFields />
      <button type="button" className="ghost wide" onClick={reset}>Restore default look</button>
      <Credits manifest={manifest} />
    </div>
  )
}

/** Attribution for every asset in the scene, as their licences ask. */
export function Credits({ manifest }: { manifest: Manifest }) {
  return (
    <section className="credits" aria-label="Asset credits">
      <h3>Credits</h3>
      <ul>
        {Object.entries(manifest.credits).map(([id, c]) => (
          <li key={id}><a href={c.url} target="_blank" rel="noreferrer">{c.title}</a>, {c.author}. {c.license}.</li>
        ))}
      </ul>
      <p className="field-hint">NASA material is used under NASA's media guidelines and does not imply endorsement.</p>
    </section>
  )
}
