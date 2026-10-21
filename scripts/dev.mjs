import { spawn } from 'node:child_process'

// FLEET_API_PORT moves the planner off 8765 when that port is taken; vite.config.ts proxies to the same port
const apiPort = process.env.FLEET_API_PORT ?? '8765'

const children = [
  spawn('uv', ['run', 'python', '-u', 'web/server.py', '--port', apiPort], { stdio: 'inherit' }),
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
