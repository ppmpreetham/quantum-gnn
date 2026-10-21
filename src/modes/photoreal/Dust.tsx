import { useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { createRng } from '../../core/math'
import type { Vec3 } from './environment'

const COUNT = 900
const BOX = new THREE.Vector3(26, 5, 26)

const vertexShader = /* glsl */ `
uniform float uTime;
uniform vec3 uBox;
uniform vec3 uWind;
uniform float uPixelScale;
attribute float aSeed;
varying float vFade;
void main() {
  vec3 p = position + uWind * uTime + vec3(sin(uTime * 0.3 + aSeed * 40.0), sin(uTime * 0.21 + aSeed * 17.0) * 0.4, cos(uTime * 0.27 + aSeed * 23.0)) * 0.25;
  p = mod(p + uBox * 0.5, uBox) - uBox * 0.5;
  vec4 view = modelViewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * view;
  gl_PointSize = uPixelScale * (0.6 + aSeed) / -view.z;
  vFade = smoothstep(0.0, 0.6, p.y + uBox.y * 0.5) * (1.0 - smoothstep(10.0, 24.0, -view.z));
}`

const fragmentShader = /* glsl */ `
uniform vec3 uColor;
uniform float uOpacity;
varying float vFade;
void main() {
  float d = length(gl_PointCoord - 0.5);
  float a = smoothstep(0.5, 0.0, d);
  gl_FragColor = vec4(uColor, a * a * uOpacity * vFade);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`

/** Fine dust drifting through the light in one additive point draw; brighter when lit from behind. */
export function Dust({ amount, sun, sunColor }: { amount: number; sun: Vec3; sunColor: THREE.Color }) {
  const geometry = useMemo(() => {
    const rng = createRng(4242)
    const position = new Float32Array(COUNT * 3), seed = new Float32Array(COUNT)
    for (let i = 0; i < COUNT; i++) {
      position.set([(rng() - 0.5) * BOX.x, (rng() - 0.5) * BOX.y, (rng() - 0.5) * BOX.z], i * 3)
      seed[i] = rng()
    }
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(position, 3))
    g.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1))
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), BOX.length())
    return g
  }, [])
  const material = useMemo(() => new THREE.ShaderMaterial({
    vertexShader, fragmentShader, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: {
      uTime: { value: 0 }, uBox: { value: BOX }, uWind: { value: new THREE.Vector3(0.35, 0.02, 0.12) },
      uPixelScale: { value: 1 }, uColor: { value: new THREE.Color() }, uOpacity: { value: 0 },
    },
  }), [])
  useEffect(() => () => { geometry.dispose(); material.dispose() }, [geometry, material])
  useFrame(({ clock, size, gl }) => {
    const u = material.uniforms
    u.uTime.value = clock.elapsedTime
    u.uPixelScale.value = size.height * gl.getPixelRatio() * 0.012
    u.uColor.value.copy(sunColor)
    u.uOpacity.value = amount * (0.25 + 0.5 * Math.max(0, 1 - sun[1]))
  })
  return <points geometry={geometry} material={material} position-y={BOX.y / 2 - 0.2} frustumCulled={false} renderOrder={4} />
}
