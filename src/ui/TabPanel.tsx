import { useState } from 'react'
import type { ReactNode } from 'react'

export interface PanelTab { id: string; label: string; content: () => ReactNode }

/** The left-hand panel shell: an accessible tab list over one scrolling pane. */
export function TabPanel({ tabs, label }: { tabs: readonly PanelTab[]; label: string }) {
  const [active, setActive] = useState(tabs[0].id)
  const current = tabs.find(t => t.id === active) ?? tabs[0]
  return (
    <aside className="panel mission-panel" aria-label={label}>
      <div className="tabs" role="tablist" aria-label="Panel sections">
        {tabs.map(t => (
          <button key={t.id} type="button" role="tab" id={`tab-${t.id}`} aria-selected={t.id === current.id}
            aria-controls={`pane-${t.id}`} className={t.id === current.id ? 'on' : ''} onClick={() => setActive(t.id)}>{t.label}</button>
        ))}
      </div>
      <div className="tab-scroll" role="tabpanel" id={`pane-${current.id}`} aria-labelledby={`tab-${current.id}`}>
        {current.content()}
      </div>
    </aside>
  )
}
