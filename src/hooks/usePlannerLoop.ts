import { useEffect } from 'react'
import { PlannerError, fetchHealth, requestStep } from '../api/planner'
import type { StepRequest } from '../api/planner'
import { sleep } from '../api/sleep'
import { plannerParams } from '../config/fleet'
import type { FleetEngine } from '../core/fleet'
import { useConsole } from '../state/store'

const MAX_BACKOFF_MS = 8000
/**
 * Survives fleet changes on purpose: the first request of a new fleet hands
 * the old id back, so the server replaces that session instead of leaving it
 * orphaned until it expires.
 */
let lastSessionId: string | undefined

const unassignedNotice = (count: number) =>
  `${count === 1 ? 'One task' : `${count} tasks`} found no eligible rover: every candidate was below the trust gate, low on battery, short of compute or cut off from the root.`

/**
 * One request in flight at a time: the next cycle starts only after the
 * previous answer is applied, so slow solves stretch the cadence instead of
 * queueing. Leaving the effect aborts the request, so an answer for an old
 * fleet can never be applied to a new one.
 */
export function usePlannerLoop(engine: FleetEngine): void {
  useEffect(() => {
    const controller = new AbortController()
    const { signal } = controller
    const store = useConsole
    let failures = 0

    void fetchHealth(signal).then(health => { if (health) store.getState().setSolvers(health.solvers) })

    async function loop() {
      while (!signal.aborted) {
        const state = store.getState()
        if (state.paused || document.hidden) {
          if (state.planner.state !== 'paused') state.setPlanner('paused')
          await sleep(250, signal)
          continue
        }
        const started = performance.now()
        const request: StepRequest = {
          sessionId: lastSessionId, params: plannerParams(state.params), positions: engine.positions(),
          faults: state.takeFaults(), comparison: state.comparison,
          ...(state.taskPlacement === 'click' && { tasks: state.takeTasks(engine.count) }),
        }
        try {
          const step = await requestStep(request, signal)
          if (signal.aborted) return
          if (step.rovers.length !== engine.count) { lastSessionId = undefined; continue }
          // a different id for the same fleet means the server expired or evicted our
          // mission and started a fresh one; the history no longer describes it
          if (request.sessionId && step.sessionId !== request.sessionId && step.cycle === 1) store.getState().resetMission()
          lastSessionId = step.sessionId
          if (failures > 0) void fetchHealth(signal).then(health => { if (health) store.getState().setSolvers(health.solvers) })
          failures = 0
          engine.applyPlan(step)
          store.getState().recordStep(step)
          if (step.unassigned?.length) store.getState().setTaskNotice(unassignedNotice(step.unassigned.length))
          store.getState().setPlanner('online', null, performance.now() - started)
        } catch (error) {
          if (signal.aborted) return
          request.faults.forEach(fault => store.getState().queueFault(fault))
          if (request.tasks?.length) store.getState().returnTasks(request.tasks)
          if (error instanceof PlannerError && error.busy) {
            // the server is finishing another solve; this is load, not an outage
            await sleep(400, signal)
            continue
          }
          failures++
          if (error instanceof PlannerError && !error.retryable) lastSessionId = undefined
          store.getState().setPlanner('offline', error instanceof Error ? error.message : 'Planner unavailable')
          await sleep(Math.min(MAX_BACKOFF_MS, 1000 * 2 ** (failures - 1)), signal)
          continue
        }
        const interval = store.getState().params.plannerInterval * 1000
        await sleep(Math.max(50, interval - (performance.now() - started)), signal)
      }
    }
    void loop()
    return () => controller.abort()
  }, [engine])
}
