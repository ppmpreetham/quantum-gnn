import { Suspense, lazy, useEffect } from 'react'
import type { ComponentType } from 'react'
import { useConsole } from '../state/store'
import { ConsoleMode } from './console/ConsoleMode'
import { LiveMission } from './LiveMission'
import { hashForMode, modeFromHash, modeSpec } from './modes'
import type { ModeId } from './modes'

/**
 * Each mode's root view. The default mode ships in the main bundle; the
 * others (and their heavy assets) load only when first entered.
 */
const VIEWS: Record<ModeId, ComponentType> = {
  console: ConsoleMode,
  photoreal: lazy(() => import('./photoreal/PhotorealMode').then(m => ({ default: m.PhotorealMode }))),
  showcase: lazy(() => import('./showcase/ShowcaseMode').then(m => ({ default: m.ShowcaseMode }))),
}

/** Keeps #mode=<id> in the address bar in step with the store, both ways. */
function useModeHash(mode: ModeId) {
  useEffect(() => {
    const next = hashForMode(mode)
    if (location.hash !== next) history.replaceState(null, '', next || `${location.pathname}${location.search}`)
  }, [mode])
  useEffect(() => {
    const onHash = () => useConsole.getState().setMode(modeFromHash(location.hash))
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
}

function ModeLoading({ mode }: { mode: ModeId }) {
  return (
    <div className="mode-loading" role="status">
      <span className="spinner" aria-hidden="true" />
      Loading {modeSpec(mode).label.toLowerCase()} mode
    </div>
  )
}

/** Renders the active mode; live modes share one mounted mission so switching between them keeps it running. */
export function ModeHost() {
  const mode = useConsole(s => s.mode)
  useModeHash(mode)
  const View = VIEWS[mode]
  const view = (
    <Suspense fallback={<ModeLoading mode={mode} />}>
      <View />
    </Suspense>
  )
  return modeSpec(mode).mission === 'live' ? <LiveMission>{view}</LiveMission> : view
}
