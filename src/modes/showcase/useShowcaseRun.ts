import { useEffect } from 'react'
import { PlannerError, requestStep } from '../../api/planner'
import type { FaultRequest } from '../../api/planner'
import { sleep } from '../../api/sleep'
import { COMPARISONS, plannerParams } from '../../config/fleet'
import { compareCycle } from './compare'
import type { CycleComparison } from './compare'
import { faultsAt, traceVerdict } from './scenarios'
import type { Scenario } from './scenarios'
import type { ScriptedMotion, ScriptedSide } from './ScriptedFleet'
import { useShowcase } from './store'

/** Seconds of animation per cycle at 1×; highlights hold the frame longer so the camera can frame them. */
export const GLIDE_SECONDS = 1.1
const DWELL = { win: 3.0, loss: 2.4, tie: 0.9 } as const

export const baselineLabel = (scenario: Scenario) => COMPARISONS.find(c => c.id === scenario.comparison)?.short ?? scenario.comparison

/** True when a live cycle produced exactly what the verification run recorded for it. */
export function reproduces(scenario: Scenario, cycle: number, result: CycleComparison): boolean {
  const recorded = scenario.trace[cycle]
  if (!recorded) return false
  if (result.score) {
    const [o, ot, b, bt] = recorded as number[]
    return o === result.score.ours[0] && ot === result.score.ours[1] && b === result.score.baseline[0] && bt === result.score.baseline[1]
  }
  return recorded[0] === (result.solver?.verdict ?? 'skipped') && traceVerdict(recorded) === result.verdict
}

async function untilRunning(signal: AbortSignal) {
  while (!signal.aborted && useShowcase.getState().status === 'paused') await sleep(120, signal)
}

/** Holds for a dwell that honours pause and the current speed. */
async function hold(seconds: number, signal: AbortSignal) {
  let left = seconds * 1000
  while (left > 0 && !signal.aborted) {
    await untilRunning(signal)
    const slice = Math.min(100, left)
    await sleep(slice, signal)
    left -= slice * useShowcase.getState().speed
  }
}

/**
 * Replays a scenario through the planner: one fresh session, the recorded
 * positions and faults of each cycle, the scenario's comparison. Both sides
 * are updated from the same response, then the run waits long enough for the
 * highlight to be read. Changing scenario or restarting aborts the loop.
 */
export function useShowcaseRun(scenario: Scenario, motion: ScriptedMotion, ours: ScriptedSide, baseline: ScriptedSide) {
  const runId = useShowcase(s => s.runId)
  useEffect(() => {
    const controller = new AbortController()
    const { signal } = controller
    const label = baselineLabel(scenario)
    let sessionId: string | undefined

    async function step(cycle: number) {
      const faults: FaultRequest[] = faultsAt(scenario, cycle).map(f => (f.rover === null ? { kind: f.kind } : { kind: f.kind, rover: f.rover }))
      for (;;) {
        try {
          return await requestStep({ sessionId, params: plannerParams(scenario.params), positions: scenario.positions[cycle], faults, comparison: scenario.comparison }, signal)
        } catch (error) {
          // 503: the planner is finishing another solve; anything else ends the run
          if (!(error instanceof PlannerError && error.busy) || signal.aborted) throw error
          await sleep(400, signal)
        }
      }
    }

    async function run() {
      useShowcase.getState().begin()
      for (let cycle = 0; cycle < scenario.cycles && !signal.aborted; cycle++) {
        await untilRunning(signal)
        motion.target = cycle
        const response = await step(cycle)
        if (signal.aborted) return
        sessionId = response.sessionId
        while (motion.t < cycle - 1e-6 && !signal.aborted) await sleep(50, signal)
        const result = compareCycle(response, scenario.comparison, label)
        const faulted = response.faults.map(f => f.rover)
        ours.apply(result.ours, faulted)
        baseline.apply(result.baseline, faulted)
        useShowcase.getState().record(result, reproduces(scenario, cycle, result))
        await hold(DWELL[result.verdict], signal)
      }
      if (!signal.aborted) useShowcase.getState().finish()
    }
    run().catch(error => {
      if (!signal.aborted) useShowcase.getState().fail(error instanceof Error ? error.message : 'The planner failed')
    })
    return () => controller.abort()
  }, [scenario, runId, motion, ours, baseline])
}
