/**
 * The one place that defines the console's modes. Pure data (no React), so
 * it is unit-tested in Node; ModeHost maps each id to its lazily loaded view.
 *
 * mission: 'live' modes share one running mission (engine, planner loop and
 * history), so switching between them keeps the rovers and results;
 * 'scripted' modes bring their own deterministic mission.
 */
export const MODES = [
  {
    id: 'console', label: 'Console', mission: 'live',
    summary: 'The live planner console: QUBO link pruning and trust-aware task allocation.',
  },
  {
    id: 'photoreal', label: 'Photoreal', mission: 'live',
    summary: 'The same live mission on a photoreal desert test site, with full control of light and atmosphere.',
  },
  {
    id: 'showcase', label: 'Where we win', mission: 'scripted',
    summary: 'Seeded scenarios replayed side by side against a baseline, highlighting the cycles our pipeline wins.',
  },
] as const

export type ModeSpec = typeof MODES[number]
export type ModeId = ModeSpec['id']
export const DEFAULT_MODE: ModeId = 'console'

export const isModeId = (value: unknown): value is ModeId => MODES.some(m => m.id === value)
export const modeSpec = (id: ModeId): ModeSpec => MODES.find(m => m.id === id)!

/** Reads `#mode=<id>`; anything else (or nothing) is the default mode. */
export function modeFromHash(hash: string): ModeId {
  const match = /(?:^#|&)mode=([a-z]+)/.exec(hash)
  return match && isModeId(match[1]) ? match[1] : DEFAULT_MODE
}

/** The default mode keeps a clean URL. */
export const hashForMode = (mode: ModeId): string => (mode === DEFAULT_MODE ? '' : `#mode=${mode}`)

export function initialMode(): ModeId {
  return typeof location === 'undefined' ? DEFAULT_MODE : modeFromHash(location.hash)
}
