import { useEffect, useMemo } from 'react'
import { COMPARISONS } from '../../config/fleet'
import { createSdfRover } from '../../scene/model/sdfRover'
import { Segmented, Toggle } from '../../ui/controls'
import { ReferenceCard } from '../../ui/panels/MissionTab'
import { StageFrame } from '../../ui/shell/StageFrame'
import { TopBar } from '../../ui/TopBar'
import type { CycleComparison } from './compare'
import { roverName } from './compare'
import { SCENARIOS, SCENARIO_NOTE, isAssignment } from './scenarios'
import type { AssignmentSummary, Scenario, SolverSummary } from './scenarios'
import { ScriptedMotion, ScriptedSide } from './ScriptedFleet'
import { SplitStage } from './SplitStage'
import type { SplitLabels } from './SplitStage'
import { useShowcase } from './store'
import type { Speed } from './store'
import { baselineLabel, useShowcaseRun } from './useShowcaseRun'

const SPEEDS: { value: string; label: string }[] = [{ value: '1', label: '1×' }, { value: '2', label: '2×' }, { value: '4', label: '4×' }]

function measuredLine(s: Scenario): string {
  if (isAssignment(s)) {
    const m = s.measured as AssignmentSummary
    return `Verified run: ours ${m.ours[0]}/${m.ours[1]} delivered, ${baselineLabel(s)} ${m.baseline[0]}/${m.baseline[1]}; ours won ${m.cyclesWon} cycles, lost ${m.cyclesLost}.`
  }
  const m = s.measured as SolverSummary
  return `Verified run over ${s.cycles} cycles: bit-flip SA infeasible ${m.infeasible}, feasible but worse ${m.worse}, matched ${m.matched}, better ${m.better}.`
}

function ScenarioList({ current }: { current: Scenario }) {
  const select = useShowcase(s => s.select)
  return (
    <ul className="scenario-list" role="listbox" aria-label="Scenarios">
      {SCENARIOS.map(s => (
        <li key={s.id}>
          <button type="button" role="option" aria-selected={s.id === current.id} className={`scenario ${s.id === current.id ? 'on' : ''}`} onClick={() => select(s.id)}>
            <b>{s.title}</b>
            <span>vs {COMPARISONS.find(c => c.id === s.comparison)?.label ?? s.comparison}</span>
          </button>
        </li>
      ))}
    </ul>
  )
}

function ScenarioPanel({ scenario }: { scenario: Scenario }) {
  const comparison = COMPARISONS.find(c => c.id === scenario.comparison)
  const directed = useShowcase(s => s.directed)
  const setDirected = useShowcase(s => s.setDirected)
  return (
    <aside className="panel mission-panel showcase-panel" aria-label="Showcase scenarios">
      <div className="tab-scroll">
        <div className="tab-body">
          <div className="field">
            <span className="field-label">Where our pipeline wins</span>
            <p className="field-hint">Each scenario replays fixed positions, faults and seed through the real planner. Left: our pipeline. Right: the baseline on the same positions, faults and random seed.</p>
          </div>
          <ScenarioList current={scenario} />
          <section className="scenario-detail">
            <h3>{scenario.title}</h3>
            <p><b>Watch:</b> {scenario.watch}</p>
            <p><b>Why ours wins:</b> {scenario.mechanism}</p>
            <p className="verified">{measuredLine(scenario)}</p>
          </section>
          <p className="footnote strong">{SCENARIO_NOTE}</p>
          {comparison && <ReferenceCard comparison={comparison} />}
          <Toggle label="Camera follows each highlight" checked={directed} onChange={() => setDirected(!directed)} />
        </div>
      </div>
    </aside>
  )
}

function Tally({ scenario, results }: { scenario: Scenario; results: CycleComparison[] }) {
  if (isAssignment(scenario)) {
    const sum = (side: 'ours' | 'baseline') => results.reduce((acc, r) => [acc[0] + (r.score?.[side][0] ?? 0), acc[1] + (r.score?.[side][1] ?? 0)], [0, 0])
    const [os, ot] = sum('ours'), [bs, bt] = sum('baseline')
    return (
      <div className="compare-figures">
        <span className="figure ours"><small>Ours delivered</small>{os} / {ot}</span>
        <span className="figure base"><small>{baselineLabel(scenario)}</small>{bs} / {bt}</span>
      </div>
    )
  }
  const count = (v: string) => results.filter(r => r.solver?.verdict === v).length
  return (
    <div className="compare-figures">
      <span className="figure base"><small>Infeasible</small>{count('infeasible')}</span>
      <span className="figure base"><small>Worse</small>{count('worse')}</span>
      <span className="figure"><small>Matched</small>{count('matched')}</span>
      <span className="figure"><small>Better than ours</small>{count('better')}</span>
    </div>
  )
}

function Timeline({ scenario, results, reproduced }: { scenario: Scenario; results: CycleComparison[]; reproduced: boolean[] }) {
  return (
    <ol className="timeline" aria-label="Cycle outcomes">
      {Array.from({ length: scenario.cycles }, (_, k) => {
        const r = results[k]
        return (
          <li key={k} className={r ? r.verdict : 'pending'} title={r ? `Cycle ${k + 1}: ${r.headline}${reproduced[k] ? '' : ' (differs from the verified run)'}` : `Cycle ${k + 1}`}>
            {r && !reproduced[k] && <i aria-label="differs from the verified run">!</i>}
          </li>
        )
      })}
    </ol>
  )
}

function ResultStrip({ scenario }: { scenario: Scenario }) {
  const results = useShowcase(s => s.results)
  const reproduced = useShowcase(s => s.reproduced)
  const status = useShowcase(s => s.status)
  const error = useShowcase(s => s.error)
  const latest = results.at(-1)
  const matches = reproduced.filter(Boolean).length
  return (
    <section className="panel strip showcase-strip" aria-label="Scenario results" aria-live="polite">
      <div className="strip-score">
        <small>Cycle</small>
        <strong>{results.length}<span>/{scenario.cycles}</span></strong>
        <span>{status === 'done' ? 'Run complete' : status === 'paused' ? 'Paused' : status === 'error' ? 'Stopped' : 'Replaying'}</span>
      </div>
      <div className="strip-compare">
        <Tally scenario={scenario} results={results} />
        <Timeline scenario={scenario} results={results} reproduced={reproduced} />
        <p className={`compare-note headline ${latest?.verdict ?? ''}`}>
          {error ? `The planner stopped: ${error}` : latest ? `Cycle ${latest.cycle}: ${latest.headline}` : 'Starting a fresh planner session…'}
        </p>
      </div>
      <div className="strip-guarantee">
        <span className={`badge ${results.length && matches === results.length ? 'good' : results.length ? 'warn' : 'idle'}`}>
          {results.length === 0 ? 'Waiting for cycle 1' : matches === results.length ? 'Reproduces the verified run' : `${results.length - matches} cycle(s) differ from the verified run`}
        </span>
        <p className="field-hint">Green cells are cycles ours won, orange the ones the baseline won, grey ties. Every cell comes from this live replay.</p>
      </div>
    </section>
  )
}

function Controls() {
  const status = useShowcase(s => s.status)
  const togglePlay = useShowcase(s => s.togglePlay)
  const restart = useShowcase(s => s.restart)
  const speed = useShowcase(s => s.speed)
  const setSpeed = useShowcase(s => s.setSpeed)
  return (
    <>
      <Segmented label="Replay speed" value={String(speed)} options={SPEEDS} onChange={v => setSpeed(Number(v) as Speed)} />
      <button type="button" className="ghost" onClick={restart}>Replay</button>
      <button type="button" className="primary" onClick={togglePlay} disabled={status === 'done' || status === 'error'} aria-pressed={status === 'paused'}>
        {status === 'paused' ? 'Resume' : 'Pause'}
      </button>
    </>
  )
}

function SideCaptions({ scenario }: { scenario: Scenario }) {
  return (
    <div className="split-captions" aria-hidden="true">
      <span className="ours">Our pipeline: QUBO pruning + Trust-GNN</span>
      <span className="base">{COMPARISONS.find(c => c.id === scenario.comparison)?.label ?? scenario.comparison}</span>
      <i className="split-divider" />
    </div>
  )
}

function Labels({ count, labels, latest }: { count: number; labels: SplitLabels; latest: CycleComparison | undefined }) {
  const tone = (marks: number[] | undefined, i: number) => (marks?.[i] === 1 ? 'good' : marks?.[i] === 2 ? 'bad' : '')
  return (
    <div className="label-layer" aria-hidden="true">
      {(['left', 'right'] as const).map(side => Array.from({ length: count }, (_, i) => (
        <div key={`${side}-${i}`} className="label-anchor" ref={el => { labels[side][i] = el }}>
          <div className={`rover-label ${tone(side === 'left' ? latest?.ours.roverMarks : latest?.baseline.roverMarks, i)}`}>{roverName(i)}{i === 0 && <span>Root</span>}</div>
        </div>
      )))}
    </div>
  )
}

/**
 * "Where we win": scripted, seeded scenarios replayed through the real
 * planner, our pipeline and a baseline side by side, with the camera and
 * colour marks pointing at the cycles that decide the comparison.
 */
export function ShowcaseMode() {
  const scenarioId = useShowcase(s => s.scenarioId)
  const runId = useShowcase(s => s.runId)
  const latest = useShowcase(s => s.results.at(-1))
  const scenario = SCENARIOS.find(s => s.id === scenarioId) ?? SCENARIOS[0]
  const model = useMemo(createSdfRover, [])
  useEffect(() => () => model.dispose(), [model])
  // fresh motion and sides for every run, so a replay starts from the scenario's first frame
  const { motion, ours, baseline } = useMemo(() => {
    const motion = new ScriptedMotion(scenario)
    return { motion, ours: new ScriptedSide(motion), baseline: new ScriptedSide(motion) }
  }, [scenario, runId])
  const labels = useMemo<SplitLabels>(() => ({ left: [], right: [] }), [scenario])
  useShowcaseRun(scenario, motion, ours, baseline)

  return (
    <div className="console showcase">
      <StageFrame overlay={<><Labels count={motion.count} labels={labels} latest={latest} /><SideCaptions scenario={scenario} /></>}>
        {onContextLost => <SplitStage model={model} motion={motion} ours={ours} baseline={baseline} labels={labels} onContextLost={onContextLost} />}
      </StageFrame>
      <TopBar center={<span className="showcase-title">{scenario.title}</span>}><Controls /></TopBar>
      <ScenarioPanel scenario={scenario} />
      <ResultStrip scenario={scenario} />
    </div>
  )
}
