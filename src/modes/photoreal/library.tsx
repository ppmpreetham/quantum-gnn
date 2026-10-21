import { Suspense, createContext, use, useContext, useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import { useThree } from '@react-three/fiber'
import { AssetLibrary } from './assets'
import type { Manifest } from './assets'

const LibraryContext = createContext<AssetLibrary | null>(null)

/**
 * Owns the photoreal asset library for this renderer. It is created in an
 * effect and committed before anything suspends on it, so a suspended
 * child can never orphan a half-loaded library, and StrictMode's extra
 * mount cycle gets a fresh one. Everything it loaded is freed on unmount.
 */
export function AssetLibraryProvider({ children }: { children: ReactNode }) {
  const gl = useThree(s => s.gl)
  const [library, setLibrary] = useState<AssetLibrary | null>(null)
  useEffect(() => {
    const next = new AssetLibrary(gl)
    setLibrary(next)
    return () => next.dispose()
  }, [gl])
  if (!library) return null
  return (
    <LibraryContext.Provider value={library}>
      <Suspense fallback={null}>{children}</Suspense>
    </LibraryContext.Provider>
  )
}

/** The library and its manifest; suspends until the manifest is in. */
export function useLibrary(): { library: AssetLibrary; manifest: Manifest } {
  const library = useContext(LibraryContext)
  if (!library) throw new Error('useLibrary must be used inside AssetLibraryProvider')
  return { library, manifest: use(library.manifest()) }
}
