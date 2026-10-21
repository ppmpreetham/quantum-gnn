import { useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { useScene } from '../state/fleetContext'
import { useConsole } from '../state/store'
import { InstanceWriter } from './instancing'
import { FOG_GLSL, SCENE, fogUniforms, syncFog } from './palette'

const CAPACITY = 32
const HEIGHT = 1.6

const vertexShader = /* glsl */ `
attribute vec4 iBeacon;   // x, z, height scale, pulse phase
attribute vec4 iColor;
varying float vHeight;
varying vec4 vColor;
varying float vDepth;
varying float vPhase;
void main() {
  vec3 p = position;
  p.y = (p.y + 0.5) * ${HEIGHT.toFixed(2)} * iBeacon.z;
  p.xz += iBeacon.xy;
  vec4 view = viewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * view;
  vHeight = position.y + 0.5;
  vColor = iColor;
  vDepth = -view.z;
  vPhase = iBeacon.w;
}`

const fragmentShader = /* glsl */ `
${FOG_GLSL}
uniform float uTime;
uniform float uStrength;
varying float vHeight;
varying vec4 vColor;
varying float vDepth;
varying float vPhase;
void main() {
  float fade = pow(1.0 - vHeight, 1.6);
  float band = 0.6 + 0.4 * smoothstep(0.7, 1.0, fract(vHeight * 3.0 - uTime * 0.8 + vPhase));
  float alpha = vColor.a * fade * band * (1.0 - fogFactor(vDepth)) * uStrength;
  gl_FragColor = vec4(vColor.rgb, alpha);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`

const ENROUTE = new THREE.Color(SCENE.ceramic).multiplyScalar(1.2)
const SUCCESS = new THREE.Color(SCENE.mint).multiplyScalar(2)
const FAILURE = new THREE.Color(SCENE.signal).multiplyScalar(2)

/** Task sites as light columns: white while a rover is en route, mint or orange once resolved. */
export function Beacons({ strength = 1 }: { strength?: number }) {
  const { view } = useScene()
  const writer = useMemo(() => new InstanceWriter(new THREE.CylinderGeometry(0.16, 0.24, 1, 20, 1, true), CAPACITY,
    { iBeacon: 4, iColor: 4 }), [])
  const fog = useMemo(fogUniforms, [])
  const material = useMemo(() => new THREE.ShaderMaterial({
    vertexShader, fragmentShader, uniforms: { ...fog, uTime: { value: 0 }, uStrength: { value: 1 } },
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
  }), [fog])
  useEffect(() => () => { writer.dispose(); material.dispose() }, [writer, material])

  useFrame(({ clock, scene }) => {
    material.uniforms.uTime.value = clock.elapsedTime
    syncFog(fog, scene.fog)
    material.uniforms.uStrength.value = strength
    writer.begin()
    if (useConsole.getState().overlays.beacons) {
      for (const beacon of view.beacons) {
        const slot = writer.next()
        if (slot < 0) break
        const resolved = beacon.state === 'resolved'
        const age = resolved ? view.elapsed - beacon.resolvedAt : 0
        const color = !resolved ? ENROUTE : beacon.success ? SUCCESS : FAILURE
        const alpha = resolved ? Math.max(0, 1 - age / 4) : 0.5
        const rise = resolved ? 1 + Math.min(0.6, age * 0.5) : 0.7
        writer.write('iBeacon', slot, beacon.x, beacon.z, rise, (beacon.x * 3.7 + beacon.z) % 1)
        writer.write('iColor', slot, color.r, color.g, color.b, alpha)
      }
    }
    writer.commit()
  })

  return <mesh geometry={writer.geometry} material={material} frustumCulled={false} renderOrder={3} />
}
