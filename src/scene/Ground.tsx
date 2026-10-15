import { useMemo } from 'react'
import * as THREE from 'three'
import { ARENA } from '../config/fleet'
import { FOG_GLSL, SCENE, fogUniforms } from './palette'

const vertexShader = /* glsl */ `
varying vec2 vXZ;
varying float vDepth;
void main() {
  vec4 world = modelMatrix * vec4(position, 1.0);
  vXZ = world.xz;
  vec4 view = viewMatrix * world;
  vDepth = -view.z;
  gl_Position = projectionMatrix * view;
}`

const fragmentShader = /* glsl */ `
${FOG_GLSL}
uniform vec3 uBase;
uniform vec3 uGrid;
uniform vec3 uRim;
uniform float uArena;
varying vec2 vXZ;
varying float vDepth;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}
// anti-aliased grid line: distance to the nearest line in pixels, via fwidth
float grid(vec2 p, float spacing, float width) {
  vec2 coord = p / spacing;
  vec2 d = abs(fract(coord - 0.5) - 0.5) / fwidth(coord);
  return 1.0 - clamp(min(d.x, d.y) - width, 0.0, 1.0);
}

void main() {
  float r = length(vXZ);
  vec3 color = uBase * (0.95 + 0.05 * noise(vXZ * 0.3) + 0.025 * noise(vXZ * 3.1));
  color *= mix(1.14, 0.8, smoothstep(0.0, uArena * 2.4, r));
  float inside = 1.0 - smoothstep(uArena - 0.4, uArena + 3.0, r);
  float lines = grid(vXZ, 1.0, 0.15) * 0.22 + grid(vXZ, 5.0, 0.55) * 0.45;
  color = mix(color, uGrid, lines * inside);
  float rimWidth = fwidth(r) * 1.2;
  float rim = 1.0 - smoothstep(rimWidth, rimWidth * 2.5, abs(r - uArena));
  float angle = atan(vXZ.y, vXZ.x);
  float ticks = step(0.92, fract(angle / 6.2831853 * 72.0)) * (1.0 - smoothstep(0.0, 0.25, abs(r - uArena - 0.18)));
  color = mix(color, uRim, clamp(rim * 0.8 + ticks * 0.5, 0.0, 1.0));
  color = mix(color, uFogColor, fogFactor(vDepth));
  gl_FragColor = vec4(color, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`

/** A single plain, matte ground disc: grid, arena rim and fog in one draw call. */
export function Ground() {
  const material = useMemo(() => new THREE.ShaderMaterial({
    vertexShader, fragmentShader,
    uniforms: {
      ...fogUniforms(),
      uBase: { value: new THREE.Color(SCENE.basalt) },
      uGrid: { value: new THREE.Color(SCENE.grid) },
      uRim: { value: new THREE.Color(SCENE.foil).multiplyScalar(0.55) },
      uArena: { value: ARENA.radius },
    },
  }), [])
  return (
    <mesh rotation-x={-Math.PI / 2} material={material} renderOrder={-1}>
      <circleGeometry args={[70, 128]} />
    </mesh>
  )
}
