import { ModeHost } from './modes/ModeHost'
import { Boundary } from './ui/shell/Boundary'

export function App() {
  return (
    <Boundary title="The console stopped" advice="Reload the page; if it happens again, the browser console has the details.">
      <ModeHost />
    </Boundary>
  )
}
