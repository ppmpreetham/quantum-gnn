import { useEffect } from 'react'
import { useConsole } from '../state/store'
import type { CameraMode } from '../state/store'

const CAMERA_KEYS: Record<string, CameraMode> = { 1: 'orbit', 2: 'top', 3: 'follow', 4: 'chase' }

/** Space pauses, 1 to 4 pick a camera, [ and ] cycle the selected rover, Escape returns to orbit. */
export function useKeyboard(): void {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      if (event.metaKey || event.ctrlKey || event.altKey) return
      if (target?.closest('input, select, textarea, [contenteditable="true"]')) return
      const state = useConsole.getState()
      const n = state.params.roverCount
      if (event.code === 'Space' && !target?.closest('button')) { event.preventDefault(); state.togglePause() }
      else if (CAMERA_KEYS[event.key]) state.setCamera(CAMERA_KEYS[event.key])
      else if (event.key === ']') state.select((state.selected + 1) % n)
      else if (event.key === '[') state.select((state.selected - 1 + n) % n)
      else if (event.key === 'Escape') state.setCamera('orbit')
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
}
