/** Undirected link between rover indices, stored with a < b. */
export interface Link { a: number; b: number; quality: number }

export const pairIndex = (a: number, b: number, n: number) => (a < b ? a * n + b : b * n + a)

/**
 * Widest-path (maximum bottleneck) tree from the root over the given links.
 * Task data crosses its path's weakest link, so this matches the planner's
 * delivery model. A label-correcting search is exact here because the
 * bottleneck min(·) is monotone along a path. parent[root] = -1 and
 * unreachable rovers keep parent -1 and bottleneck 0.
 */
export function widestPathTree(n: number, root: number, links: readonly Link[]) {
  const parent = new Int32Array(n).fill(-1)
  const bottleneck = new Float32Array(n)
  if (root < 0 || root >= n) return { parent, bottleneck }
  const adjacency: number[][] = Array.from({ length: n }, () => [])
  links.forEach((link, i) => {
    if (link.a >= n || link.b >= n) return
    adjacency[link.a].push(i)
    adjacency[link.b].push(i)
  })
  bottleneck[root] = 1
  const stack = [root]
  while (stack.length) {
    const u = stack.pop()!
    for (const i of adjacency[u]) {
      const link = links[i]
      const v = link.a === u ? link.b : link.a
      const candidate = Math.min(bottleneck[u], link.quality)
      if (candidate > bottleneck[v] && v !== root) {
        bottleneck[v] = candidate
        parent[v] = u
        stack.push(v)
      }
    }
  }
  return { parent, bottleneck }
}

/** Rovers reachable from root, using only links accepted by the predicate. */
export function reachableCount(n: number, root: number, links: readonly Link[], usable: (link: Link) => boolean): number {
  if (root < 0 || root >= n) return 0
  const seen = new Uint8Array(n)
  seen[root] = 1
  let count = 1
  let changed = true
  while (changed) {
    changed = false
    for (const link of links) {
      if (!usable(link) || seen[link.a] === seen[link.b]) continue
      seen[link.a] = seen[link.b] = 1
      count++
      changed = true
    }
  }
  return count
}

/** Path from a rover to the root following parent pointers; empty when unreachable. */
export function pathToRoot(parent: Int32Array, from: number, root: number): number[] {
  const path = [from]
  let node = from
  for (let guard = 0; node !== root && guard < parent.length; guard++) {
    node = parent[node]
    if (node < 0) return []
    path.push(node)
  }
  return node === root ? path : []
}
