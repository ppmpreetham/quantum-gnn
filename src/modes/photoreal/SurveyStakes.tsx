import { useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import { ARENA } from '../../config/fleet'

const COUNT = 24
const RADIUS = ARENA.radius + 0.35

/** Timber survey stakes with orange flagging tape marking the arena's edge, as on a real test site; one instanced draw. */
export function SurveyStakes() {
  const geometry = useMemo(() => {
    const paint = (g: THREE.BufferGeometry, color: string) => {
      const c = new THREE.Color(color)
      const colors = new Float32Array(g.attributes.position.count * 3)
      for (let i = 0; i < colors.length; i += 3) colors.set([c.r, c.g, c.b], i)
      g.setAttribute('color', new THREE.BufferAttribute(colors, 3))
      return g.index ? g.toNonIndexed() : g
    }
    const post = paint(new THREE.BoxGeometry(0.035, 0.42, 0.035).translate(0, 0.19, 0), '#8a6a45')
    const tape = paint(new THREE.PlaneGeometry(0.16, 0.035).translate(0.08, 0.36, 0), '#ff5a1f')
    const merged = mergeGeometries([post, tape])!
    post.dispose(); tape.dispose()
    return merged
  }, [])
  const material = useMemo(() => new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85, side: THREE.DoubleSide }), [])
  useEffect(() => () => { geometry.dispose(); material.dispose() }, [geometry, material])
  const ref = useRef<THREE.InstancedMesh>(null)
  useLayoutEffect(() => {
    const mesh = ref.current!
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), up = new THREE.Vector3(0, 1, 0)
    for (let i = 0; i < COUNT; i++) {
      const a = (i / COUNT) * Math.PI * 2
      q.setFromAxisAngle(up, -a + 0.4 * Math.sin(i * 2.7)).multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), 0.06 * Math.sin(i * 1.3)))
      mesh.setMatrixAt(i, m.compose(new THREE.Vector3(RADIUS * Math.cos(a), -0.02, RADIUS * Math.sin(a)), q, new THREE.Vector3(1, 0.85 + 0.3 * ((i * 7) % 5) / 5, 1)))
    }
    mesh.instanceMatrix.needsUpdate = true
    mesh.computeBoundingSphere()
  }, [])
  return <instancedMesh ref={ref} args={[geometry, material, COUNT]} castShadow receiveShadow />
}
