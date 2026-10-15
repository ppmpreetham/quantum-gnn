import { useState } from 'react'
import { MissionTab } from './MissionTab'
import { TuneTab } from './TuneTab'
import { ViewTab } from './ViewTab'
import type { ModelControls } from './ViewTab'

const TABS = [
  { id: 'mission', label: 'Mission' },
  { id: 'tune', label: 'Tune' },
  { id: 'view', label: 'View' },
] as const
type TabId = typeof TABS[number]['id']

export function MissionPanel({ model }: { model: ModelControls }) {
  const [tab, setTab] = useState<TabId>('mission')
  return (
    <aside className="panel mission-panel" aria-label="Mission controls">
      <div className="tabs" role="tablist" aria-label="Panel sections">
        {TABS.map(t => (
          <button key={t.id} type="button" role="tab" id={`tab-${t.id}`} aria-selected={tab === t.id}
            aria-controls={`pane-${t.id}`} className={tab === t.id ? 'on' : ''} onClick={() => setTab(t.id)}>{t.label}</button>
        ))}
      </div>
      <div className="tab-scroll" role="tabpanel" id={`pane-${tab}`} aria-labelledby={`tab-${tab}`}>
        {tab === 'mission' && <MissionTab />}
        {tab === 'tune' && <TuneTab />}
        {tab === 'view' && <ViewTab model={model} />}
      </div>
    </aside>
  )
}
