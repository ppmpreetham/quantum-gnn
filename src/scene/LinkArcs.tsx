import { useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { LINK_CANDIDATE, LINK_KEPT } from '../core/fleet'
import { MARK_BAD, MARK_GOOD } from '../core/fleetView'
import { pathToRoot } from '../core/graph'
import { useScene } from '../state/fleetContext'
import { useConsole } from '../state/store'
import { InstanceWriter } from './instancing'
import { FOG_GLSL, SCENE, fogUniforms, syncFog } from './palette'

const SEGMENTS = 24
const MOUNT_HEIGHT = 0.52
const MAX_ROVERS = 16
const CAPACITY = (MAX_ROVERS * (MAX_ROVERS - 1)) / 2

/** Link styles, read by the shader from aStyle.z. GOOD and BAD are showcase highlights. */
const CANDIDATE = 0, KEPT = 1, BREAKING = 2, PATH = 3, GOOD = 4, BAD = 5

/** A ribbon strip along t in [0, 1]; the vertex shader bends it into an arc and faces it to the camera. */
function ribbon(): THREE.BufferGeometry {
  const t = new Float32Array((SEGMENTS + 1) * 2)
  const side = new Float32Array((SEGMENTS + 1) * 2)
  const index: number[] = []
  for (let k = 0; k <= SEGMENTS; k++) {
    t[k * 2] = t[k * 2 + 1] = k / SEGMENTS
    side[k * 2] = -1
    side[k * 2 + 1] = 1
    if (k < SEGMENTS) {
      const a = k * 2
      index.push(a, a + 1, a + 2, a + 1, a + 3, a + 2)
    }
  }
  const geometry = new THREE.BufferGeometry()
  // position is unused by the shader but three.js expects it for draw ranges
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array((SEGMENTS + 1) * 6), 3))
  geometry.setAttribute('aT', new THREE.BufferAttribute(t, 1))
  geometry.setAttribute('aSide', new THREE.BufferAttribute(side, 1))
  geometry.setIndex(index)
  return geometry
}

const vertexShader = /* glsl */ `
attribute float aT;
attribute float aSide;
attribute vec3 aStart;
attribute vec3 aEnd;
attribute vec4 aColor;
attribute vec4 aStyle;   // width, flow direction (-1, 0, 1), style, phase
varying vec4 vColor;
varying vec4 vStyle;
varying float vT;
varying float vSide;
varying float vLength;
varying float vDepth;
void main() {
  vec3 chord = aEnd - aStart;
  float len = length(chord);
  float height = clamp(0.08 + 0.06 * len, 0.1, 0.5);
  vec3 up = vec3(0.0, 1.0, 0.0);
  // quadratic bump 4t(1-t): zero at the antennas, peak height mid-span
  vec3 p = aStart + chord * aT + up * (height * 4.0 * aT * (1.0 - aT));
  vec3 tangent = chord + up * (height * 4.0 * (1.0 - 2.0 * aT));
  vec3 toCamera = cameraPosition - p;
  vec3 across = cross(tangent, toCamera);
  float acrossLength = length(across);
  // looking straight down the link the cross product vanishes; fall back to a horizontal side
  across = acrossLength > 1e-5 ? across / acrossLength : normalize(cross(tangent + vec3(1e-4), up));
  p += across * aSide * aStyle.x * 0.5;
  vec4 view = viewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * view;
  vColor = aColor;
  vStyle = aStyle;
  vT = aT;
  vSide = aSide;
  vLength = len;
  vDepth = -view.z;
}`

const fragmentShader = /* glsl */ `
${FOG_GLSL}
uniform float uTime;
uniform float uFlow;
uniform float uStrength;
varying vec4 vColor;
varying vec4 vStyle;
varying float vT;
varying float vSide;
varying float vLength;
varying float vDepth;
void main() {
  float core = 1.0 - smoothstep(0.35, 1.0, abs(vSide));
  float ends = smoothstep(0.0, 0.06, vT) * smoothstep(0.0, 0.06, 1.0 - vT);
  float style = vStyle.z;
  float glow = 1.0;
  if (abs(vStyle.y) > 0.5 && uFlow > 0.5) {
    // packets travel from child to parent, toward the root
    float u = vStyle.y > 0.0 ? vT : 1.0 - vT;
    float s = fract(u * vLength / 0.85 - uTime * 0.9 + vStyle.w);
    glow += 2.4 * smoothstep(0.78, 0.97, s) * (1.0 - smoothstep(0.97, 1.0, s));
  }
  if ((style > 1.5 && style < 2.5) || style > 4.5) glow *= 0.55 + 0.45 * sin(uTime * 14.0 + vStyle.w * 6.0);
  float alpha = vColor.a * core * ends * (1.0 - fogFactor(vDepth)) * uStrength;
  gl_FragColor = vec4(vColor.rgb * glow, alpha);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`

const COLORS = {
  [CANDIDATE]: new THREE.Color(SCENE.baseline),
  [KEPT]: new THREE.Color(SCENE.foil).multiplyScalar(1.6),
  [BREAKING]: new THREE.Color(SCENE.signal).multiplyScalar(1.6),
  [PATH]: new THREE.Color('#fff1cf').multiplyScalar(2.4),
  [GOOD]: new THREE.Color(SCENE.mint).multiplyScalar(2.2),
  [BAD]: new THREE.Color(SCENE.signal).multiplyScalar(2.2),
}
const WIDTH = { [CANDIDATE]: 0.022, [KEPT]: 0.05, [BREAKING]: 0.04, [PATH]: 0.075, [GOOD]: 0.08, [BAD]: 0.08 }
const ALPHA = { [CANDIDATE]: 0.4, [KEPT]: 0.95, [BREAKING]: 0.9, [PATH]: 1, [GOOD]: 1, [BAD]: 1 }

/**
 * Every radio link in one instanced draw call. Geometry follows the rovers
 * each frame; which links are kept comes from the latest plan. A kept link
 * whose rovers drifted out of range since that plan is drawn as breaking.
 */
export function LinkArcs({ selection = true, strength = 1 }: { selection?: boolean; strength?: number }) {
  const { view, poses } = useScene()
  const writer = useMemo(() => new InstanceWriter(ribbon(), CAPACITY,
    { aStart: 3, aEnd: 3, aColor: 4, aStyle: 4 }), [])
  const fog = useMemo(fogUniforms, [])
  const material = useMemo(() => new THREE.ShaderMaterial({
    vertexShader, fragmentShader,
    uniforms: { ...fog, uTime: { value: 0 }, uFlow: { value: 1 }, uStrength: { value: 1 } },
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
  }), [fog])
  useEffect(() => () => { writer.dispose(); material.dispose() }, [writer, material])
  const onPath = useMemo(() => new Uint8Array(MAX_ROVERS * MAX_ROVERS), [])
  const reduceMotion = useMemo(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches, [])

  useFrame(({ clock, scene }) => {
    const { selected, overlays, paused } = useConsole.getState()
    if (!paused && !reduceMotion) material.uniforms.uTime.value = clock.elapsedTime
    material.uniforms.uFlow.value = overlays.flow && !reduceMotion ? 1 : 0
    syncFog(fog, scene.fog)
    material.uniforms.uStrength.value = strength
    const n = view.count
    onPath.fill(0)
    if (selection && overlays.path && selected < n) {
      const path = pathToRoot(view.parent, selected, 0)
      for (let k = 1; k < path.length; k++) {
        const a = Math.min(path[k - 1], path[k]), b = Math.max(path[k - 1], path[k])
        onPath[a * n + b] = 1
      }
    }
    const range = view.radioRange
    const marks = view.linkMarks
    writer.begin()
    for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) {
      const dx = poses.x[a] - poses.x[b], dz = poses.z[a] - poses.z[b]
      const inRange = dx * dx + dz * dz <= range * range
      const plan = view.planState[a * n + b]
      const mark = marks?.[a * n + b]
      let style: number
      if (mark === MARK_GOOD) style = GOOD
      else if (mark === MARK_BAD) style = BAD
      else if (plan === LINK_KEPT) style = !inRange ? BREAKING : onPath[a * n + b] ? PATH : KEPT
      else if (inRange && overlays.candidates && (plan === LINK_CANDIDATE || plan === 0)) style = CANDIDATE
      else continue
      const slot = writer.next()
      if (slot < 0) break
      // flow direction: +1 when data goes a -> b (a's parent is b), -1 for b -> a
      const flow = style === KEPT || style === PATH || style === GOOD ? (view.parent[a] === b ? 1 : view.parent[b] === a ? -1 : 0) : 0
      const quality = view.planQuality[a * n + b]
      const color = COLORS[style as keyof typeof COLORS]
      const alpha = ALPHA[style as keyof typeof ALPHA] * (style === KEPT ? 0.55 + 0.45 * quality : 1)
      writer.write('aStart', slot, poses.x[a], MOUNT_HEIGHT, poses.z[a])
      writer.write('aEnd', slot, poses.x[b], MOUNT_HEIGHT, poses.z[b])
      writer.write('aColor', slot, color.r, color.g, color.b, alpha)
      writer.write('aStyle', slot, WIDTH[style as keyof typeof WIDTH], flow, style, ((a * 7 + b * 13) % 17) / 17)
    }
    writer.commit()
  })

  return <mesh geometry={writer.geometry} material={material} frustumCulled={false} renderOrder={2} />
}
