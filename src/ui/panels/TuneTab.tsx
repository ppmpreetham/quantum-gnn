import { useMemo } from 'react'
import { PARAM_GROUPS, PARAM_SPECS, advisories } from '../../config/fleet'
import { useConsole } from '../../state/store'
import { NumberField, Slider, Toggle } from '../controls'
import { paramValue } from '../format'

const SCOPE_NOTE = {
  session: 'Restarts the mission',
  planner: 'Applies next cycle',
  client: 'Applies instantly',
} as const

export function TuneTab() {
  const params = useConsole(s => s.params)
  const setParam = useConsole(s => s.setParam)
  const applyPreset = useConsole(s => s.applyPreset)
  const tether = useConsole(s => s.tether)
  const setTether = useConsole(s => s.setTether)
  const notes = useMemo(() => advisories(params), [params])

  return (
    <div className="tab-body">
      {notes.length > 0 && (
        <ul className="advisories" aria-live="polite">
          {notes.map(note => <li key={note.text} className={note.level}>{note.text}</li>)}
        </ul>
      )}
      {PARAM_GROUPS.map(group => (
        <fieldset key={group.id} className="param-group">
          <legend>{group.label}</legend>
          {PARAM_SPECS.filter(spec => spec.group === group.id).map(spec => spec.key === 'seed' ? (
            <NumberField key={spec.key} label={spec.label} value={params.seed} min={spec.min} max={spec.max}
              onChange={value => setParam('seed', value)} help={`${spec.help} ${SCOPE_NOTE[spec.scope]}.`}
              action={{ label: 'New scenario', onClick: () => setParam('seed', Math.floor(Math.random() * (spec.max + 1))) }} />
          ) : (
            <Slider key={spec.key} label={spec.label} value={params[spec.key]} min={spec.min} max={spec.max} step={spec.step}
              display={paramValue(spec, params[spec.key])} onChange={value => setParam(spec.key, value)}
              help={`${spec.help} ${SCOPE_NOTE[spec.scope]}.`} />
          ))}
          {group.id === 'fleet' && (
            <Toggle label="Keep radio contact while patrolling" checked={tether} onChange={() => setTether(!tether)}
              hint="Idle rovers steer back toward their nearest neighbour when it drifts near the edge of radio range." />
          )}
        </fieldset>
      ))}
      <button type="button" className="ghost wide" onClick={() => applyPreset('field')}>Restore field survey defaults</button>
    </div>
  )
}
