import { useState } from 'react'
import type { DragEvent, ReactNode } from 'react'
import { Boundary } from './Boundary'

interface StageFrameProps {
  /** Renders the canvas; receives the callback that reports a lost or restored GPU context. */
  children: (onContextLost: (lost: boolean) => void) => ReactNode
  /** DOM overlays positioned over the canvas (labels, hints). */
  overlay?: ReactNode
  /** Accepts a dropped file (the console uses it for .glb rover models). */
  onFile?: (file: File) => void
  dropHint?: string
  className?: string
}

/** The full-bleed stage every mode draws into: error boundary, context-loss notice and optional file drop. */
export function StageFrame({ children, overlay, onFile, dropHint, className = '' }: StageFrameProps) {
  const [contextLost, setContextLost] = useState(false)
  const [dragging, setDragging] = useState(false)
  const drop = onFile && {
    onDragOver: (event: DragEvent) => { event.preventDefault(); setDragging(true) },
    onDragLeave: () => setDragging(false),
    onDrop: (event: DragEvent) => {
      event.preventDefault()
      setDragging(false)
      const file = event.dataTransfer.files[0]
      if (file) onFile(file)
    },
  }
  return (
    <main className={`stage ${className}`} {...drop}>
      <Boundary title="The 3D view stopped" advice="Check that hardware acceleration is on, then reload. The planner and panels keep running.">
        {children(setContextLost)}
      </Boundary>
      {overlay}
      {contextLost && (
        <div className="stage-notice" role="alert">
          <b>Graphics context lost</b>
          <p>The browser reclaimed the GPU. The view comes back on its own when it is restored, or reload now.</p>
          <button type="button" className="primary" onClick={() => window.location.reload()}>Reload</button>
        </div>
      )}
      {dragging && dropHint && <div className="drop-hint">{dropHint}</div>}
    </main>
  )
}
