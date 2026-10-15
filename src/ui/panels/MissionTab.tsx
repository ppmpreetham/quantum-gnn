import { COMPARISONS, OMITTED_NOTE, PRESETS } from '../../config/fleet'
import type { Comparison } from '../../config/fleet'
import type { FaultKind } from '../../api/planner'
import { roverId, useConsole } from '../../state/store'
import { Select } from '../controls'
import type { Option } from '../controls'
import { FAULT_LABEL, ms } from '../format'

const GROUP = { assignment: 'Task assignment', solver: 'QUBO solver', none: '' } as const

function ReferenceCard({ comparison }: { comparison: Comparison }) {
  const ref = comparison.reference
  if (!ref) return null
  const unit = ref.metric === 'Mission success' ? '%' : '% of instances'
  return (
    <figure className="reference">
      <figcaption>Published result</figcaption>
      <div className="reference-bars" role="img" aria-label={`${ref.metric}: ours ${ref.ours}${unit}, baseline ${ref.baseline}${unit}`}>
        <span className="reference-row"><b>Ours</b><span className="bar ours" style={{ width: `${ref.ours}%` }} /><em>{ref.ours}%</em></span>
        <span className="reference-row"><b>{comparison.short}</b><span className="bar base" style={{ width: `${ref.baseline}%` }} /><em>{ref.baseline}%</em></span>
      </div>
      <p>
        {ref.metric}
        {ref.delta !== undefined && <>, {ref.delta >= 0 ? '+' : '−'}{Math.abs(ref.delta)} pts</>}
        {ref.ci && <>, 95% CI [{ref.ci[0]}, {ref.ci[1]}]</>}
        {ref.p !== undefined && <>, p = {ref.p}</>}
        .{' '}{ref.significant === false && <strong>Not statistically significant. </strong>}
        {ref.source}
      </p>
    </figure>
  )
}

const FAULTS: { kind: FaultKind; targeted: boolean; detail: string }[] = [
  { kind: 'reliability_drop', targeted: true, detail: 'Hidden reliability falls by 0.6 for 4 cycles. Trust should follow.' },
  { kind: 'battery_drain', targeted: true, detail: 'Drains 25% battery per cycle for 2 cycles; below 15% a rover is not eligible for tasks.' },
  { kind: 'packet_loss_spike', targeted: false, detail: 'Adds 40% packet loss fleet-wide for 3 cycles, so link quality goes stale.' },
]

export function MissionTab() {
  const presetId = useConsole(s => s.presetId)
  const applyPreset = useConsole(s => s.applyPreset)
  const comparison = useConsole(s => s.comparison)
  const setComparison = useConsole(s => s.setComparison)
  const solvers = useConsole(s => s.solvers)
  const selected = useConsole(s => s.selected)
  const queueFault = useConsole(s => s.queueFault)
  const pending = useConsole(s => s.pendingFaults)
  const faults = useConsole(s => s.faults)
  const preset = PRESETS.find(p => p.id === presetId)
  const current = COMPARISONS.find(c => c.id === comparison) ?? COMPARISONS[0]

  const presetOptions: Option[] = [
    ...(preset ? [] : [{ value: '', label: 'Custom settings', disabled: true }]),
    ...PRESETS.map(p => ({ value: p.id, label: p.name })),
  ]
  const comparisonOptions: Option[] = COMPARISONS.map(c => {
    const missing = c.kind === 'solver' && c.baseline && solvers[c.baseline] === false
    return { value: c.id, label: missing ? `${c.label} (not installed)` : c.label, disabled: !!missing, group: GROUP[c.kind] || 'Off' }
  })

  return (
    <div className="tab-body">
      <Select label="Preset" value={presetId ?? ''} options={presetOptions} onChange={applyPreset}
        hint={preset ? <>{preset.description} Measured planning time about {ms(preset.measuredCycleMs)} per cycle.</> : 'You changed a setting. Pick a preset to start over.'} />

      <Select label="Compare our pipeline against" value={comparison} options={comparisonOptions} onChange={setComparison}
        hint={current.summary ?? 'Pick a baseline to run it side by side with the live mission.'} />
      {current.kind !== 'none' && <ReferenceCard comparison={current} />}
      <p className="footnote">{OMITTED_NOTE}</p>

      <section className="faults" aria-label="Fault injection">
        <h3>Inject a fault</h3>
        <p className="field-hint">Targets apply to the selected rover, {roverId(selected)}. Baselines receive the same fault, so comparisons stay paired.</p>
        <div className="fault-buttons">
          {FAULTS.map(f => (
            <button key={f.kind} type="button" className="ghost" title={f.detail}
              onClick={() => queueFault(f.targeted ? { kind: f.kind, rover: selected } : { kind: f.kind })}>
              {FAULT_LABEL[f.kind]}{f.targeted ? ` on ${roverId(selected)}` : ''}
            </button>
          ))}
        </div>
        {(pending.length > 0 || faults.length > 0) && (
          <ul className="fault-list">
            {pending.map((f, k) => <li key={`p${k}`}>{FAULT_LABEL[f.kind]}{f.rover !== undefined ? `, ${roverId(f.rover)}` : ''}: sent with the next cycle</li>)}
            {faults.map((f, k) => <li key={`a${k}`} className="active">{FAULT_LABEL[f.kind]}{f.rover !== null ? `, ${roverId(f.rover)}` : ''}: {f.remaining} cycle{f.remaining === 1 ? '' : 's'} left</li>)}
          </ul>
        )}
      </section>
    </div>
  )
}
