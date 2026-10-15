import { createRng } from './math.ts'

export const mean = (values: readonly number[]) =>
  values.length ? values.reduce((sum, v) => sum + v, 0) / values.length : Number.NaN

/**
 * Percentile bootstrap CI of the mean of paired differences, the same
 * statistic the paper reports. Deterministic for a given seed so the
 * interval does not jitter between renders. Undefined below 2 samples.
 */
export function bootstrapMeanCI(diffs: readonly number[], level = 0.95, resamples = 2000, seed = 7): [number, number] | null {
  const n = diffs.length
  if (n < 2) return null
  const rng = createRng(seed)
  const means = new Float64Array(resamples)
  for (let b = 0; b < resamples; b++) {
    let sum = 0
    for (let k = 0; k < n; k++) sum += diffs[(rng() * n) | 0]
    means[b] = sum / n
  }
  means.sort()
  const tail = (1 - level) / 2
  const at = (q: number) => means[Math.min(resamples - 1, Math.max(0, Math.floor(q * resamples)))]
  return [at(tail), at(1 - tail)]
}

/** Success rate with zero tasks reported as NaN, never as 0%. */
export const rate = (successes: number, tasks: number) => (tasks > 0 ? successes / tasks : Number.NaN)
