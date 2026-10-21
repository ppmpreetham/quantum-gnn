/**
 * Shape of public/photoreal/manifest.json, written by
 * scripts/photoreal-assets/build.mjs. Pure, so the shipped manifest is
 * checked in Node against the files that are really there.
 */
export interface Credit { title: string; author: string; license: string; url: string }
export interface PropAsset { id: string; file: string; credit: string; variants: number; alpha: boolean }
export interface GroundAsset { id: string; credit: string; tileMeters: number; color: string; normal: string; arh: string }
export interface SkyAsset { id: string; label: string; file: string; credit: string }
export interface Manifest {
  version: 1
  worldScale: number
  rover: { file: string; credit: string; lods: number }
  props: PropAsset[]
  grounds: GroundAsset[]
  skies: SkyAsset[]
  credits: Record<string, Credit>
}

/** Rejects a manifest that would make the scene reference missing pieces. */
export function validateManifest(value: unknown): Manifest {
  const m = value as Manifest
  const ok = !!m && m.version === 1 && typeof m.worldScale === 'number' && typeof m.rover?.file === 'string'
    && Array.isArray(m.props) && Array.isArray(m.grounds) && m.grounds.length >= 2 && Array.isArray(m.skies) && m.skies.length > 0
    && typeof m.credits === 'object' && m.credits !== null
    && [m.rover.credit, ...m.props.map(p => p.credit), ...m.grounds.map(g => g.credit), ...m.skies.map(s => s.credit)].every(c => c in m.credits)
  if (!ok) throw new Error('The photoreal asset manifest is incomplete; rebuild it with scripts/photoreal-assets')
  return m
}
