import { useCallback, useEffect, useRef, useState } from 'react'
import { loadGlbRover } from '../scene/model/glbRover'
import { createSdfRover } from '../scene/model/sdfRover'
import type { ForwardAxis, RoverModel } from '../scene/model/types'
import { useConsole } from '../state/store'
import type { ModelControls } from '../ui/panels/ViewTab'

/**
 * Owns the active rover model. A replaced model is disposed after React has
 * committed its successor, so the GPU never draws from freed buffers. Only the
 * newest load may win: a slow parse that finishes late is discarded.
 */
export function useRoverModel(): { model: RoverModel; controls: ModelControls } {
  const [model, setModel] = useState<RoverModel>(createSdfRover)
  const [loading, setLoading] = useState(false)
  const [forward, setForward] = useState<ForwardAxis>('auto')
  const file = useRef<File | null>(null)
  const generation = useRef(0)
  const error = useConsole(s => s.modelError)
  const setStoreModel = useConsole(s => s.setModel)

  useEffect(() => () => model.dispose(), [model])

  const load = useCallback(async (source: File, axis: ForwardAxis) => {
    const ticket = ++generation.current
    setLoading(true)
    try {
      const next = await loadGlbRover(source, axis)
      if (ticket !== generation.current) { next.dispose(); return }
      file.current = source
      setModel(next)
      setStoreModel(next.name, null)
    } catch (reason) {
      if (ticket === generation.current) setStoreModel(useConsole.getState().modelName, reason instanceof Error ? reason.message : String(reason))
    } finally {
      if (ticket === generation.current) setLoading(false)
    }
  }, [setStoreModel])

  const controls: ModelControls = {
    name: model.name, error, loading, forward, isCustom: model.source === 'glb',
    onFile: source => { void load(source, forward) },
    onForward: axis => { setForward(axis); if (file.current) void load(file.current, axis) },
    onReset: () => {
      generation.current++
      file.current = null
      const sdf = createSdfRover()
      setModel(sdf)
      setStoreModel(sdf.name, null)
    },
  }
  return { model, controls }
}
