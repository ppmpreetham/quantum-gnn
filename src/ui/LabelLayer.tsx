import { useFleet } from '../state/fleetContext'
import { roverId, useConsole } from '../state/store'
import { trustCss } from '../scene/palette'
import { ACTIVITY_LABEL } from './format'

/** DOM name tags positioned by LabelProjector inside the canvas; React only re-renders them at 5 Hz. */
export function LabelLayer() {
  const { engine, bus } = useFleet()
  const rovers = useConsole(s => s.telemetry?.rovers)
  const selected = useConsole(s => s.selected)
  return (
    <div className="label-layer" aria-hidden="true">
      {Array.from({ length: engine.count }, (_, i) => {
        const rover = rovers?.[i]
        return (
          <div key={i} className="label-anchor" ref={el => { bus.labels[i] = el }}>
            <div className={`rover-label ${i === selected ? 'selected' : ''}`}>
              <i style={{ background: trustCss(rover?.trust ?? 0.5) }} />
              {roverId(i)}
              {i === 0 && <span>Root</span>}
              {i === selected && rover && <small>{ACTIVITY_LABEL[rover.activity]}</small>}
            </div>
          </div>
        )
      })}
    </div>
  )
}
