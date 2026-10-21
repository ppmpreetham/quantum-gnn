/** Resolves after ms, or as soon as the signal aborts (never rejects). */
export function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise(resolve => {
    const timer = setTimeout(done, ms)
    signal.addEventListener('abort', done, { once: true })
    function done() { clearTimeout(timer); signal.removeEventListener('abort', done); resolve() }
  })
}
