import { useMemo } from 'react'
import { COMPARISONS } from '../config/fleet'
import type { Comparison } from '../config/fleet'
import { bootstrapMeanCI, mean, rate } from '../core/stats'
import { useConsole } from '../state/store'
import type { CycleRecord } from '../state/store'
import { PRUNE_MODES, ms, pct, signedPts } from './format'

const WINDOW = 12
/** Below this many paired cycles a percentile bootstrap undercovers badly. */
const EARLY = 15

function rolling(rows: [number, number][]): number[] {
  return rows.map((_, k) => {
    let s = 0, t = 0
    for (let j = Math.max(0, k - WINDOW + 1); j <= k; j++) { s += rows[j][0]; t += rows[j][1] }
    return t > 0 ? s / t : Number.NaN
  })
}

function Sparkline({ series, label }: { series: { values: number[]; className: string }[]; label: string }) {
  const length = Math.max(2, ...series.map(s => s.values.length))
  const points = (values: number[]) => values
    .map((v, k) => (Number.isFinite(v) ? `${((k / (length - 1)) * 100).toFixed(2)},${(38 - v * 34).toFixed(2)}` : null))
    .filter(Boolean).join(' ')
  return (
    <svg className="sparkline" viewBox="0 0 100 40" preserveAspectRatio="none" role="img" aria-label={label}>
      <line x1="0" x2="100" y1="21" y2="21" className="mid" />
      {series.map(s => <polyline key={s.className} points={points(s.values)} className={s.className} vectorEffect="non-scaling-stroke" />)}
    </svg>
  )
}

function AssignmentResult({ history, comparison }: { history: CycleRecord[]; comparison: Comparison }) {
  const stats = useMemo(() => {
    const name = comparison.baseline!
    const rows = history.filter(h => h.ours[1] > 0 && (h.baselines[name]?.[1] ?? 0) > 0)
    const ours = rows.map(h => h.ours)
    const base = rows.map(h => h.baselines[name])
    const sum = (list: [number, number][]) => list.reduce((acc, [s, t]) => [acc[0] + s, acc[1] + t], [0, 0])
    const [os, ot] = sum(ours), [bs, bt] = sum(base)
    const diffs = rows.map((_, k) => ours[k][0] / ours[k][1] - base[k][0] / base[k][1])
    return { cycles: rows.length, ours: rate(os, ot), base: rate(bs, bt), delta: mean(diffs), ci: bootstrapMeanCI(diffs), oursLine: rolling(ours), baseLine: rolling(base) }
  }, [history, comparison])
  const excludesZero = stats.ci !== null && (stats.ci[0] > 0 || stats.ci[1] < 0)
  return (
    <div className="strip-compare">
      <div className="compare-figures">
        <span className="figure ours"><small>Ours</small>{pct(stats.ours)}</span>
        <span className="figure base"><small>{comparison.short}</small>{pct(stats.base)}</span>
        <span className="figure delta"><small>Difference</small>{signedPts(stats.delta)}</span>
      </div>
      <Sparkline label={`Rolling success over ${WINDOW} cycles, ours against ${comparison.short}`}
        series={[{ values: stats.baseLine, className: 'base' }, { values: stats.oursLine, className: 'ours' }]} />
      <p className="compare-note">
        {stats.cycles < 2 ? 'Collecting paired cycles…'
          : <>Paired over {stats.cycles} cycles, 95% CI [{signedPts(stats.ci![0])}, {signedPts(stats.ci![1])}].{' '}
            {stats.cycles < EARLY ? `Early estimate: intervals from under ${EARLY} cycles are unreliable.`
              : excludesZero ? 'The interval excludes zero.' : 'The interval includes zero, so this run alone does not separate the two.'}</>}
      </p>
    </div>
  )
}

function SolverResult({ history, comparison }: { history: CycleRecord[]; comparison: Comparison }) {
  const solver = useConsole(s => s.solver)
  const runs = history.filter(h => h.solver?.baseline === comparison.baseline).map(h => h.solver!)
  const latest = solver?.latest
  if (latest && 'error' in latest) return <div className="strip-compare"><p className="compare-note">{latest.error}.</p></div>
  const t = solver?.totals
  const feasible = runs.filter(r => r.feasible)
  return (
    <div className="strip-compare">
      <div className="compare-figures">
        <span className="figure base"><small>Feasible</small>{t ? pct(rate(t.feasible, t.runs)) : '–'}</span>
        <span className="figure base"><small>Matched ours</small>{t ? pct(rate(t.matched, t.runs)) : '–'}</span>
        <span className="figure delta"><small>Mean gap</small>{feasible.length ? pct(mean(feasible.map(r => r.gap)), 1) : '–'}</span>
      </div>
      <Sparkline label={`Per-cycle energy gap of ${comparison.short} against our annealer`}
        series={[{ values: runs.map(r => (r.feasible ? Math.min(1, Math.max(0, r.gap * 4)) : 1)), className: 'base' }]} />
      <p className="compare-note">
        {t && t.runs > 0
          ? <>{comparison.short} on the same Q matrix and warm start, {t.runs} cycle{t.runs === 1 ? '' : 's'}: {ms(t.msSum / t.runs)} per solve against our {ms(solver!.oursMs)}. The line is the energy gap; infeasible answers plot at the top.</>
          : 'Waiting for a cycle where the QUBO was solved…'}
      </p>
    </div>
  )
}

export function MissionStrip() {
  const history = useConsole(s => s.history)
  const metrics = useConsole(s => s.metrics)
  const live = useConsole(s => s.telemetry?.live)
  const comparisonId = useConsole(s => s.comparison)
  const comparison = COMPARISONS.find(c => c.id === comparisonId) ?? COMPARISONS[0]
  const totals = useMemo(() => history.reduce((acc, h) => [acc[0] + h.ours[0], acc[1] + h.ours[1]], [0, 0]), [history])
  const mode = metrics ? PRUNE_MODES[metrics.pruneMode] : null

  return (
    <section className="panel strip" aria-label="Mission results">
      <div className="strip-score">
        <small>Mission success</small>
        <strong>{pct(rate(totals[0], totals[1]))}</strong>
        <span>{totals[0]} of {totals[1]} tasks delivered</span>
      </div>
      {comparison.kind === 'assignment' && <AssignmentResult history={history} comparison={comparison} />}
      {comparison.kind === 'solver' && <SolverResult history={history} comparison={comparison} />}
      {comparison.kind === 'none' && <div className="strip-compare"><p className="compare-note">Pick a baseline in the Mission tab to run it alongside this mission on identical scenarios.</p></div>}
      <div className="strip-guarantee">
        <span className={`badge ${mode?.tone ?? 'idle'}`} title={mode?.detail}>{mode?.label ?? 'Waiting for the planner'}</span>
        <dl>
          <div><dt>Links kept</dt><dd>{live ? `${live.kept} of ${live.candidates}` : '–'}{live && live.breaking > 0 ? `, ${live.breaking} stretching` : ''}</dd></div>
          <div><dt>Reach root</dt><dd>{live && metrics ? `${live.reachable} of ${metrics.fleet}` : '–'}</dd></div>
          <div><dt>Solve</dt><dd>{metrics ? `${ms(metrics.solveMs)}, ${metrics.variables} vars` : '–'}</dd></div>
          <div><dt>Budget</dt><dd>{metrics ? `${metrics.budget}${metrics.minSpanningCost !== null ? `, tree needs ${metrics.minSpanningCost}` : ''}` : '–'}</dd></div>
        </dl>
      </div>
    </section>
  )
}
