import { useEffect } from 'react'
import { useConsole } from '../state/store'

const NOTICE_MS = 7000

/** Over-the-scene hint while placing tasks by click, plus the latest placement notice. */
export function TaskHint() {
  const placement = useConsole(s => s.taskPlacement)
  const pending = useConsole(s => s.pendingTasks.length)
  const notice = useConsole(s => s.taskNotice)
  const setTaskNotice = useConsole(s => s.setTaskNotice)
  useEffect(() => {
    if (!notice) return
    const timer = setTimeout(() => setTaskNotice(null), NOTICE_MS)
    return () => clearTimeout(timer)
  }, [notice, setTaskNotice])
  if (placement !== 'click' && !notice) return null
  return (
    <div className="task-hint" role="status" aria-live="polite">
      {placement === 'click' && (
        <span><b>Click the ground</b> to place a task.{pending > 0 && ` ${pending} waiting for the next planning cycle.`}</span>
      )}
      {notice && <span className="notice">{notice}</span>}
    </div>
  )
}
