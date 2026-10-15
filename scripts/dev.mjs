import { existsSync } from 'node:fs'
import { spawn } from 'node:child_process'

// prefer the project venv (it can hold the optional dwave-neal / tabu / openjij solvers)
const python = process.env.FLEET_PYTHON || (existsSync('.venv/bin/python') ? '.venv/bin/python' : 'python3')
const children = [
  spawn(python, ['-u', 'web/server.py'], { stdio: 'inherit' }),
  spawn(process.execPath, ['node_modules/vite/bin/vite.js', '--host', '127.0.0.1'], { stdio: 'inherit' }),
]
let stopping = false
function stop(code = 0) {
  if (stopping) return
  stopping = true
  children.forEach(child => child.kill('SIGTERM'))
  process.exitCode = code
}
children.forEach(child => {
  child.on('error', error => { console.error(error.message); stop(1) })
  child.on('exit', code => stop(code || 0))
})
process.on('SIGINT', () => stop())
process.on('SIGTERM', () => stop())
