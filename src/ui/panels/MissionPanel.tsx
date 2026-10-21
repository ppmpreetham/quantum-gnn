import { TabPanel } from '../TabPanel'
import { MissionTab } from './MissionTab'
import { TuneTab } from './TuneTab'
import { ViewTab } from './ViewTab'
import type { ModelControls } from './ViewTab'

export function MissionPanel({ model }: { model: ModelControls }) {
  return (
    <TabPanel label="Mission controls" tabs={[
      { id: 'mission', label: 'Mission', content: () => <MissionTab /> },
      { id: 'tune', label: 'Tune', content: () => <TuneTab /> },
      { id: 'view', label: 'View', content: () => <ViewTab model={model} /> },
    ]} />
  )
}
