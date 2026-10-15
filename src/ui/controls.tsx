import { useId } from 'react'
import type { CSSProperties, ReactNode } from 'react'

export function Field({ label, hint, children, htmlFor }: { label: string; hint?: ReactNode; children: ReactNode; htmlFor?: string }) {
  return (
    <div className="field">
      <label className="field-label" htmlFor={htmlFor}>{label}</label>
      {children}
      {hint && <p className="field-hint">{hint}</p>}
    </div>
  )
}

export interface Option { value: string; label: string; disabled?: boolean; group?: string }

/** Native select for keyboard and screen-reader support, with optgroups when options carry a group. */
export function Select({ label, value, options, onChange, hint }: {
  label: string; value: string; options: Option[]; onChange: (value: string) => void; hint?: ReactNode
}) {
  const id = useId()
  const groups = [...new Set(options.map(o => o.group ?? ''))]
  const render = (list: Option[]) => list.map(o => <option key={o.value} value={o.value} disabled={o.disabled}>{o.label}</option>)
  return (
    <Field label={label} hint={hint} htmlFor={id}>
      <div className="select">
        <select id={id} value={value} onChange={event => onChange(event.target.value)}>
          {groups.length === 1 && !groups[0] ? render(options)
            : groups.map(g => <optgroup key={g} label={g}>{render(options.filter(o => (o.group ?? '') === g))}</optgroup>)}
        </select>
      </div>
    </Field>
  )
}

export function Slider({ label, value, min, max, step, display, onChange, help }: {
  label: string; value: number; min: number; max: number; step: number
  display: string; onChange: (value: number) => void; help?: string
}) {
  const id = useId()
  const fill = max > min ? ((value - min) / (max - min)) * 100 : 0
  return (
    <div className="slider" title={help}>
      <div className="slider-head">
        <label htmlFor={id}>{label}</label>
        <output htmlFor={id}>{display}</output>
      </div>
      <input id={id} type="range" min={min} max={max} step={step} value={value}
        style={{ '--fill': `${fill}%` } as CSSProperties}
        onChange={event => onChange(Number(event.target.value))} />
    </div>
  )
}

export function Toggle({ label, checked, onChange, hint }: { label: string; checked: boolean; onChange: () => void; hint?: string }) {
  return (
    <label className="toggle" title={hint}>
      <input type="checkbox" checked={checked} onChange={onChange} />
      <span className="toggle-track" aria-hidden="true"><span /></span>
      <span>{label}</span>
    </label>
  )
}

export function Segmented<T extends string>({ label, value, options, onChange }: {
  label: string; value: T; options: { value: T; label: string; title?: string }[]; onChange: (value: T) => void
}) {
  return (
    <div className="segmented" role="radiogroup" aria-label={label}>
      {options.map(o => (
        <button key={o.value} type="button" role="radio" aria-checked={o.value === value} title={o.title}
          className={o.value === value ? 'on' : ''} onClick={() => onChange(o.value)}>{o.label}</button>
      ))}
    </div>
  )
}

export function Meter({ value, color, marker, label }: { value: number; color: string; marker?: number; label: string }) {
  const v = Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0
  return (
    <span className="meter" role="meter" aria-label={label} aria-valuemin={0} aria-valuemax={1} aria-valuenow={Number(v.toFixed(2))}>
      <span className="meter-fill" style={{ width: `${v * 100}%`, background: color }} />
      {marker !== undefined && Number.isFinite(marker) && <span className="meter-marker" style={{ left: `${Math.min(1, Math.max(0, marker)) * 100}%` }} />}
    </span>
  )
}

/** Integer entry that commits on blur or Enter, so typing "1" on the way to "17" does not restart a mission. */
export function NumberField({ label, value, min, max, onChange, action, help }: {
  label: string; value: number; min: number; max: number; onChange: (value: number) => void
  action?: { label: string; onClick: () => void }; help?: string
}) {
  const id = useId()
  const commit = (input: HTMLInputElement) => {
    const next = Math.round(Number(input.value))
    if (Number.isFinite(next) && next >= min && next <= max) onChange(next)
    else input.value = String(value)
  }
  return (
    <div className="number-field" title={help}>
      <label htmlFor={id}>{label}</label>
      <div className="row">
        <input id={id} key={value} type="number" inputMode="numeric" min={min} max={max} step={1} defaultValue={value}
          onBlur={event => commit(event.currentTarget)}
          onKeyDown={event => { if (event.key === 'Enter') commit(event.currentTarget) }} />
        {action && <button type="button" className="ghost" onClick={action.onClick}>{action.label}</button>}
      </div>
    </div>
  )
}
