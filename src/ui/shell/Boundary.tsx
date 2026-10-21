import { Component } from 'react'
import type { ReactNode } from 'react'

/** Error boundary with a reload action; a failing subtree reports itself instead of blanking the page. */
export class Boundary extends Component<{ children: ReactNode; title: string; advice: string }, { error: string | null }> {
  state = { error: null as string | null }
  static getDerivedStateFromError(error: unknown) { return { error: error instanceof Error ? error.message : 'Unknown error' } }
  render() {
    if (!this.state.error) return this.props.children
    return (
      <div className="stage-notice" role="alert">
        <b>{this.props.title}</b>
        <p>{this.state.error}. {this.props.advice}</p>
        <button type="button" className="primary" onClick={() => window.location.reload()}>Reload</button>
      </div>
    )
  }
}
