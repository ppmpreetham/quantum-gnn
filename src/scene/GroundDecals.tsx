import { useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { ARENA } from '../config/fleet'
import { useFleet } from '../state/fleetContext'
import { useConsole } from '../state/store'
import { InstanceWriter } from './instancing'
import { SCENE } from './palette'

const SHADOW = 0, RING = 1, DASHED = 2, GLOW = 3, SELECT = 4
const CAPACITY = 160

const vertexShader = /* glsl */ `
attribute vec4 iPlace;   // x, z, radius x, radius z
attribute float iAngle;
attribute vec4 iColor;
attribute float iKind;
varying vec2 vLocal;
varying vec4 vColor;
varying float vKind;
varying float vScale;
void main() {
  vLocal = position.xy;
  vColor = iColor;
  vKind = iKind;
  vScale = iPlace.z;
  float c = cos(iAngle), s = sin(iAngle);
  vec2 p = vec2(position.x * iPlace.z, position.y * iPlace.w);
  // heading convention: yaw turns +x toward +z
  vec2 xz = vec2(p.x * c - p.y * s, p.x * s + p.y * c) + iPlace.xy;
  gl_Position = projectionMatrix * viewMatrix * vec4(xz.x, 0.012 + iKind * 0.002, xz.y, 1.0);
}`

const fragmentShader = /* glsl */ `
uniform float uTime;
varying vec2 vLocal;
varying vec4 vColor;
varying float vKind;
varying float vScale;
float ring(float r, float at, float width) {
  float aa = fwidth(r) * 1.25;
  return 1.0 - smoothstep(width, width + aa, abs(r - at));
}
void main() {
  float r = length(vLocal);
  if (r > 1.0) discard;
  float a = 0.0;
  if (vKind < 0.5) {
    a = 1.0 - smoothstep(0.15, 1.0, r);
    a *= a;
  } else if (vKind < 1.5) {
    a = ring(r, 0.93, 0.02);
  } else if (vKind < 2.5) {
    float angle = atan(vLocal.y, vLocal.x);
    float dashes = floor(vScale * 9.0);
    float dash = step(0.42, fract(angle / 6.2831853 * dashes + uTime * 0.04));
    a = ring(r, 0.985, 0.004) * dash + (1.0 - smoothstep(0.0, 1.0, r)) * 0.05;
  } else if (vKind < 3.5) {
    a = pow(1.0 - r, 2.2) * 0.9 + ring(r, 0.9, 0.025);
  } else {
    float angle = atan(vLocal.y, vLocal.x);
    float notches = step(0.86, abs(cos(angle * 2.0)));
    a = ring(r, 0.9, 0.03) + ring(r, 0.78, 0.012) * 0.6 + notches * ring(r, 0.97, 0.05);
  }
  gl_FragColor = vec4(vColor.rgb, clamp(a, 0.0, 1.0) * vColor.a);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`

const C = {
  shadow: new THREE.Color(0, 0, 0),
  root: new THREE.Color(SCENE.foil),
  select: new THREE.Color(SCENE.signal).multiplyScalar(1.8),
  range: new THREE.Color(SCENE.baseline),
  selectedRange: new THREE.Color(SCENE.foil),
  enroute: new THREE.Color(SCENE.ceramic),
  success: new THREE.Color(SCENE.mint).multiplyScalar(1.4),
  failure: new THREE.Color(SCENE.signal).multiplyScalar(1.4),
}

/** Every flat marker on the ground (shadows, rings, ranges, beacon pads) in one instanced draw. */
export function GroundDecals() {
  const { engine, poses } = useFleet()
  const writer = useMemo(() => new InstanceWriter(new THREE.PlaneGeometry(2, 2), CAPACITY,
    { iPlace: 4, iAngle: 1, iColor: 4, iKind: 1 }), [])
  const material = useMemo(() => new THREE.ShaderMaterial({
    vertexShader, fragmentShader, uniforms: { uTime: { value: 0 } },
    transparent: true, depthWrite: false, side: THREE.DoubleSide,
  }), [])
  useEffect(() => () => { writer.dispose(); material.dispose() }, [writer, material])

  useFrame(({ clock }) => {
    const { selected, overlays, faults } = useConsole.getState()
    material.uniforms.uTime.value = clock.elapsedTime
    writer.begin()
    const add = (x: number, z: number, rx: number, rz: number, angle: number, color: THREE.Color, alpha: number, kind: number) => {
      const slot = writer.next()
      if (slot < 0) return
      writer.write('iPlace', slot, x, z, rx, rz)
      writer.write('iAngle', slot, angle)
      writer.write('iColor', slot, color.r, color.g, color.b, alpha)
      writer.write('iKind', slot, kind)
    }
    const range = engine.radioRange
    for (let i = 0; i < engine.count; i++) {
      const x = poses.x[i], z = poses.z[i]
      add(x, z, 0.66, 0.54, poses.yaw[i], C.shadow, 0.6, SHADOW)
      if (overlays.ranges || i === selected) {
        add(x, z, range, range, 0, i === selected ? C.selectedRange : C.range, i === selected ? 0.32 : 0.16, DASHED)
      }
    }
    if (engine.count > 0) add(poses.x[0], poses.z[0], ARENA.bodyRadius + 0.2, ARENA.bodyRadius + 0.2, 0, C.root, 0.55, RING)
    if (selected < engine.count) {
      const pulse = 0.85 + 0.15 * Math.sin(clock.elapsedTime * 3)
      add(poses.x[selected], poses.z[selected], ARENA.bodyRadius + 0.32, ARENA.bodyRadius + 0.32, poses.yaw[selected], C.select, pulse, SELECT)
    }
    for (const fault of faults) {
      if (fault.rover === null || fault.rover >= engine.count) continue
      add(poses.x[fault.rover], poses.z[fault.rover], 1.0, 1.0, 0, C.failure, 0.35 + 0.25 * Math.sin(clock.elapsedTime * 6), GLOW)
    }
    if (overlays.beacons) {
      for (const beacon of engine.beacons) {
        const resolved = beacon.state === 'resolved'
        const fade = resolved ? Math.max(0, 1 - (engine.elapsed - beacon.resolvedAt) / 4) : 1
        const color = !resolved ? C.enroute : beacon.success ? C.success : C.failure
        add(beacon.x, beacon.z, 0.5, 0.5, 0, color, (resolved ? 0.9 : 0.45) * fade, GLOW)
      }
    }
    writer.commit()
  })

  return <mesh geometry={writer.geometry} material={material} frustumCulled={false} renderOrder={1} />
}
