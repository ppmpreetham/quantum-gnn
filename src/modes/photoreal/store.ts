import { create } from 'zustand'
import { DEFAULT_ENVIRONMENT } from './environment'
import type { EnvironmentSettings } from './environment'

interface PhotorealState {
  settings: EnvironmentSettings
  /** Hides every panel for an unobstructed view (H). */
  hudHidden: boolean
  set: <K extends keyof EnvironmentSettings>(key: K, value: EnvironmentSettings[K]) => void
  reset: () => void
  toggleHud: () => void
}

export const usePhotoreal = create<PhotorealState>()(set => ({
  settings: { ...DEFAULT_ENVIRONMENT },
  hudHidden: false,
  set: (key, value) => set(state => ({ settings: { ...state.settings, [key]: value } })),
  reset: () => set({ settings: { ...DEFAULT_ENVIRONMENT } }),
  toggleHud: () => set(state => ({ hudHidden: !state.hudHidden })),
}))
