import { clamp, sinc, wrapAngle } from './math.ts'

/**
 * Ground frame: x and z span the plane, y is up. Heading yaw = 0 faces +x and
 * positive yaw turns toward +z. A rover facing +x therefore has its right
 * side at +z and its left side at -z (right = forward × up). In three.js the
 * body rotation is rotation.y = -yaw.
 */
export interface Pose { x: number; z: number; yaw: number }
export interface Twist { v: number; omega: number }
export interface DriveLimits {
  wheelRadius: number; track: number; maxWheelSpeed: number
  maxAccel: number; maxBrake: number; maxYawRate: number; maxYawAccel: number
}

/**
 * Exact pose update for a constant twist over dt (unicycle model). The chord
 * of an arc of angle θ has length 2 sin(θ/2)/ω · v and points along the mean
 * heading yaw + θ/2, which is v·dt·sinc(θ/2) without a singularity at ω = 0.
 */
export function integratePose(pose: Pose, v: number, omega: number, dt: number, out: Pose = { x: 0, z: 0, yaw: 0 }): Pose {
  const half = (omega * dt) / 2
  const chord = v * dt * sinc(half)
  const heading = pose.yaw + half
  out.x = pose.x + chord * Math.cos(heading)
  out.z = pose.z + chord * Math.sin(heading)
  out.yaw = wrapAngle(pose.yaw + omega * dt)
  return out
}

/**
 * Wheel rim speeds of a differential drive. Turning toward +z (omega > 0) is
 * a right turn, so the left wheel (at -z) runs on the outside of the arc.
 */
export function wheelSpeeds(v: number, omega: number, track: number) {
  const offset = (omega * track) / 2
  return { left: v + offset, right: v - offset }
}

/** Twist from wheel rim speeds: the inverse of wheelSpeeds. */
export function twistFromWheels(left: number, right: number, track: number): Twist {
  return { v: (left + right) / 2, omega: (left - right) / track }
}

/**
 * Scales v and omega by the same factor when either wheel would exceed its
 * limit. Uniform scaling keeps the curvature omega / v, so the rover follows
 * the same arc, only slower, instead of drifting off the commanded path.
 */
export function saturateTwist(v: number, omega: number, limits: DriveLimits, out: Twist = { v: 0, omega: 0 }): Twist {
  const yawLimited = clamp(omega, -limits.maxYawRate, limits.maxYawRate)
  const scaleYaw = omega === 0 ? 1 : yawLimited / omega
  const { left, right } = wheelSpeeds(v * scaleYaw, yawLimited, limits.track)
  const peak = Math.max(Math.abs(left), Math.abs(right))
  const scale = peak > limits.maxWheelSpeed ? limits.maxWheelSpeed / peak : 1
  out.v = v * scaleYaw * scale
  out.omega = yawLimited * scale
  return out
}

/** Rate limiter: approaches target no faster than rise (speeding up) or fall (slowing down). */
export function approach(current: number, target: number, rise: number, fall: number, dt: number): number {
  const delta = target - current
  const speedingUp = Math.abs(target) > Math.abs(current) && Math.sign(target) !== -Math.sign(current)
  const limit = (speedingUp ? rise : fall) * dt
  return current + clamp(delta, -limit, limit)
}
