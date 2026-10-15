import * as THREE from 'three'

/**
 * Fixed-capacity per-instance attribute writer. Buffers are allocated once;
 * each frame writes the live instances, sets instanceCount and uploads only
 * the touched range, so the CPU-to-GPU traffic tracks what is on screen.
 */
export class InstanceWriter {
  readonly geometry: THREE.InstancedBufferGeometry
  readonly attributes: Record<string, THREE.InstancedBufferAttribute>
  count = 0

  constructor(base: THREE.BufferGeometry, readonly capacity: number, layout: Record<string, number>) {
    this.geometry = new THREE.InstancedBufferGeometry()
    this.geometry.index = base.index
    for (const [name, attribute] of Object.entries(base.attributes)) this.geometry.setAttribute(name, attribute)
    this.attributes = {}
    for (const [name, size] of Object.entries(layout)) {
      const attribute = new THREE.InstancedBufferAttribute(new Float32Array(capacity * size), size)
      attribute.setUsage(THREE.DynamicDrawUsage)
      this.geometry.setAttribute(name, attribute)
      this.attributes[name] = attribute
    }
    this.geometry.instanceCount = 0
    // instances move every frame; a sphere around the whole arena keeps culling honest
    this.geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 80)
  }

  begin(): void { this.count = 0 }

  /** Reserves the next slot and returns its index, or -1 once full. */
  next(): number { return this.count < this.capacity ? this.count++ : -1 }

  /** Writes up to four components without allocating (this runs per instance per frame). */
  write(name: string, slot: number, a: number, b = 0, c = 0, d = 0): void {
    const attribute = this.attributes[name]
    const array = attribute.array as Float32Array
    const offset = slot * attribute.itemSize
    array[offset] = a
    if (attribute.itemSize > 1) array[offset + 1] = b
    if (attribute.itemSize > 2) array[offset + 2] = c
    if (attribute.itemSize > 3) array[offset + 3] = d
  }

  commit(): void {
    for (const attribute of Object.values(this.attributes)) {
      attribute.clearUpdateRanges()
      if (this.count > 0) attribute.addUpdateRange(0, this.count * attribute.itemSize)
      attribute.needsUpdate = this.count > 0
    }
    this.geometry.instanceCount = this.count
  }

  dispose(): void { this.geometry.dispose() }
}
