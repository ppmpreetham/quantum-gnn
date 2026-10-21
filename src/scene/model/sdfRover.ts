import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import { ARENA } from '../../config/fleet'
import { singleLod } from './types'
import type { RoverModel, WheelPart } from './types'
import { createVertexPbrMaterial, mergePainted, paint } from './vertexPbr'

const CERAMIC = '#E7EAED'
const FOIL = '#C9973A'
const GRAPHITE = '#2B3038'
const GLASS = '#0F141B'
const RUBBER = '#1B1F25'
const TREAD = '#272C33'
const ALLOY = '#A7B0BA'
const ORANGE = '#FF5A1F'

/*
 * Dimensions from gazebo_sim/models/rover/model.sdf. Gazebo is z-up with y
 * to the left; here y is up and the left track sits at -z. The SDF wheel
 * centre is 0.12 m above the model origin with a 0.16 m radius, so the whole
 * model is lifted 0.04 m to put the tyres on the ground.
 */
const LIFT = 0.04
const CHASSIS = { length: 0.9, height: 0.25, width: 0.65, y: 0.2 + LIFT }
const WHEEL = { radius: ARENA.wheelRadius, width: 0.08, y: 0.12 + LIFT, z: ARENA.track / 2 }
const TOP = CHASSIS.y + CHASSIS.height / 2

function at(geometry: THREE.BufferGeometry, x: number, y: number, z: number) {
  return geometry.translate(x, y, z)
}
/** Small bevelled box; one bevel segment keeps a part near 100 triangles. */
const box = (w: number, h: number, d: number, r: number) => new RoundedBoxGeometry(w, h, d, 1, r)
/** Sharp box for tiny repeated parts (tread, spokes) where bevels would cost thousands of triangles per rover. */
const block = (w: number, h: number, d: number) => new THREE.BoxGeometry(w, h, d)
const rod = (radius: number, length: number, segments = 12) => new THREE.CylinderGeometry(radius, radius, length, segments)
/** Cylinder with its axis along z (the wheel axle direction). */
const axle = (radius: number, width: number, segments: number) => rod(radius, width, segments).rotateX(Math.PI / 2)

function chassis(): THREE.BufferGeometry {
  const lens = (z: number) => paint(at(rod(0.022, 0.02, 16).rotateZ(Math.PI / 2), 0.365, TOP + 0.235, z), GLASS, 0.1, 0.7)
  return mergePainted([
    paint(at(new RoundedBoxGeometry(CHASSIS.length, CHASSIS.height, CHASSIS.width, 3, 0.045), 0, CHASSIS.y, 0), CERAMIC, 0.5, 0.05),
    paint(at(box(CHASSIS.length + 0.02, 0.06, CHASSIS.width + 0.02, 0.02), 0, CHASSIS.y - 0.1, 0), GRAPHITE, 0.6, 0.3),
    paint(at(box(0.58, 0.03, 0.52, 0.01), -0.08, TOP + 0.015, 0), FOIL, 0.3, 0.9),
    paint(at(box(0.04, 0.08, 0.5, 0.015), CHASSIS.length / 2, CHASSIS.y + 0.02, 0), GLASS, 0.12, 0.6),
    paint(at(box(0.02, 0.1, 0.44, 0.008), -CHASSIS.length / 2 - 0.005, CHASSIS.y + 0.01, 0), GRAPHITE, 0.55, 0.4),
    paint(at(rod(0.018, 0.22), 0.3, TOP + 0.11, 0), GRAPHITE, 0.5, 0.5),
    paint(at(box(0.1, 0.07, 0.2, 0.02), 0.31, TOP + 0.235, 0), CERAMIC, 0.45, 0.05),
    lens(-0.05), lens(0.05),
    paint(at(rod(0.03, 0.03), -0.32, TOP + 0.015, 0.22), GRAPHITE, 0.5, 0.5),
    paint(at(rod(0.01, 0.5, 8), -0.32, TOP + 0.25, 0.22), GRAPHITE, 0.45, 0.6),
    paint(at(new THREE.SphereGeometry(0.05, 10, 6), 0.33, 0.05, 0), GRAPHITE, 0.6, 0.3),
    paint(at(new THREE.SphereGeometry(0.05, 10, 6), -0.33, 0.05, 0), GRAPHITE, 0.6, 0.3),
  ])
}

function wheel(): THREE.BufferGeometry {
  const parts = [
    paint(axle(WHEEL.radius - 0.008, WHEEL.width, 36), RUBBER, 0.9, 0),
    paint(axle(0.095, WHEEL.width + 0.006, 28), ALLOY, 0.35, 0.85),
    paint(axle(0.038, WHEEL.width + 0.014, 20), ORANGE, 0.5, 0.15),
  ]
  for (let k = 0; k < 3; k++) {
    parts.push(paint(block(0.17, 0.022, WHEEL.width + 0.01).rotateZ((k * Math.PI) / 3), GRAPHITE, 0.5, 0.4))
  }
  // tread blocks make rotation readable at a glance
  for (let k = 0; k < 18; k++) {
    const a = (k / 18) * Math.PI * 2
    parts.push(paint(block(0.034, 0.016, WHEEL.width + 0.004)
      .translate(0, WHEEL.radius - 0.006, 0).rotateZ(a), TREAD, 0.85, 0))
  }
  return mergePainted(parts)
}

function lights(): THREE.BufferGeometry {
  const merged = mergePainted([
    paint(at(new THREE.SphereGeometry(0.028, 10, 8), -0.32, TOP + 0.51, 0.22), '#ffffff', 1, 0),
    paint(at(box(0.012, 0.02, 0.42, 0.004), CHASSIS.length / 2 + 0.022, CHASSIS.y + 0.04, 0), '#ffffff', 1, 0),
    paint(at(box(0.03, 0.014, 0.3, 0.005), -0.39, TOP + 0.02, 0), '#ffffff', 1, 0),
  ])
  merged.deleteAttribute('color')
  merged.deleteAttribute('pbr')
  return merged
}

export function createSdfRover(): RoverModel {
  const material = createVertexPbrMaterial(1.1)
  const body = chassis()
  const tyre = wheel()
  const glow = lights()
  const wheels: WheelPart[] = ([-1, 1] as const).map(side => ({
    geometry: tyre, material, side, radius: WHEEL.radius,
    pivot: new THREE.Vector3(0, WHEEL.y, side * WHEEL.z),
  }))
  const triangles = [body, tyre, tyre, glow].reduce((sum, g) => sum + (g.index?.count ?? g.attributes.position.count) / 3, 0)
  return {
    name: 'Gazebo rover (SDF)', source: 'sdf',
    lods: singleLod([{ geometry: body, material }], wheels), lights: glow, triangles,
    dispose: () => { body.dispose(); tyre.dispose(); glow.dispose(); material.dispose() },
  }
}
