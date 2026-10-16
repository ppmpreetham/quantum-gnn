import { spawn } from 'node:child_process'

// Use 'uv' to execute the script in a managed environment
const pythonCmd = 'uv'
const pythonArgs = ['run', 'python', '-u', 'web/server.py']

const children = [
  spawn(pythonCmd, pythonArgs, { stdio: 'inherit', shell: true }),
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
