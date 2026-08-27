"""
mask_to_graph.py  –  CV ➜ Graph Theory Bridge  (Phase II: Topological Healing)
══════════════════════════════════════════════════════════════════════════
Converts a binary road-segmentation mask (numpy uint8 array) into a
fully weighted, undirected networkx.Graph suitable for shortest-path
routing (Dijkstra, A*, etc.).

Pipeline
────────
1. Morphological pre-cleaning     (close micro-fractures before thinning)
2. Morphological skeletonization  (skimage.morphology.skeletonize)
3. Node extraction               (junction & endpoint detection via 8-neighbor kernel)
4. Edge tracing                   (BFS walk along skeleton arcs between nodes)
5. Graph construction             (networkx.Graph with pixel-length edge weights)
6. **Topological Healing**        (MST + Union-Find to bridge occlusion gaps)

The healing layer identifies disconnected components, scores candidate
inter-component bridges using a combined Euclidean-distance + angular-
alignment cost function, computes a minimum spanning tree over the
inter-component graph, and inserts synthetic edges to produce a single
connected routable network.

Author : NetSight Project  (ISRO NNRMS / Disaster Management Framework)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import networkx as nx
from scipy import ndimage
from skimage.morphology import skeletonize, disk, closing

# ── 8-connectivity offsets (row, col) ─────────────────────────────────────
_NEIGHBORS_8 = np.array([
    [-1, -1], [-1,  0], [-1,  1],
    [ 0, -1],           [ 0,  1],
    [ 1, -1], [ 1,  0], [ 1,  1],
], dtype=np.int32)

# ── Pixel-step distances: 1.0 for cardinal, √2 for diagonal ──────────────
_STEP_COSTS = np.array([
    np.sqrt(2), 1.0, np.sqrt(2),
    1.0,             1.0,
    np.sqrt(2), 1.0, np.sqrt(2),
], dtype=np.float64)

# ── Healing configuration defaults ────────────────────────────────────────
_DEFAULT_MAX_GAP_PX = 50.0       # Max Euclidean gap (px) we will attempt to bridge
_DEFAULT_ANGULAR_PENALTY = 2.0   # Multiplier for angular misalignment cost
_DEFAULT_DIRECTION_DEPTH = 10    # Pixels to look back for tangent estimation


# ══════════════════════════════════════════════════════════════════════════
# 0.  MORPHOLOGICAL PRE-CLEANING
# ══════════════════════════════════════════════════════════════════════════
def _morphological_preclean(mask: np.ndarray, closing_radius: int = 3) -> np.ndarray:
    """Close micro-fractures (1–3 px gaps) in the raw binary mask.

    Applies morphological closing (dilation → erosion) with a disk
    structuring element.  This bridges tiny hairline cracks that are
    artefacts of the segmentation model rather than real discontinuities,
    reducing the workload on the graph-level healer.

    Parameters
    ----------
    mask : np.ndarray  (H, W) uint8, non-zero = road
    closing_radius : int
        Radius of the disk structuring element.  Larger values bridge
        wider gaps but risk merging genuinely separate roads.

    Returns
    -------
    cleaned : np.ndarray  (H, W) bool
    """
    binary = mask.astype(bool)
    selem = disk(closing_radius)
    return closing(binary, selem)


# ══════════════════════════════════════════════════════════════════════════
# 1.  SKELETONIZE
# ══════════════════════════════════════════════════════════════════════════
def _skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Thin a binary road mask to a 1-pixel-wide centerline.

    Parameters
    ----------
    mask : np.ndarray
        2-D binary array (H, W) with dtype uint8 or bool.  Non-zero = road.

    Returns
    -------
    skeleton : np.ndarray  (bool, same shape)
    """
    binary = mask.astype(bool) if mask.dtype != bool else mask
    return skeletonize(binary)


# ══════════════════════════════════════════════════════════════════════════
# 2.  NEIGHBOR COUNTING  &  NODE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════
def _neighbor_count_map(skeleton: np.ndarray) -> np.ndarray:
    """Return an integer array where each skeleton pixel holds its 8-neighbor count."""
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    counts = ndimage.convolve(skeleton.astype(np.uint8), kernel,
                              mode='constant', cval=0)
    # Zero out non-skeleton pixels so only skeleton pixels carry a count
    counts[~skeleton] = 0
    return counts


def _extract_nodes(skeleton: np.ndarray,
                   neighbor_counts: np.ndarray) -> dict[tuple[int, int], int]:
    """Classify skeleton pixels into graph-theoretic nodes.

    A pixel qualifies as a node if it is:
        • a **junction**   – neighbor_count > 2   (branching point)
        • an **endpoint**  – neighbor_count == 1   (dead-end / terminal)

    Returns
    -------
    nodes : dict[tuple[int, int], int]
        Mapping  (row, col) → node_id  (0-indexed).
    """
    junction_mask = skeleton & (neighbor_counts > 2)
    endpoint_mask = skeleton & (neighbor_counts == 1)

    # Cluster adjacent junction pixels into single logical nodes.
    # Raw skeletons often produce small 2×2 or 3×3 blobs at crossings;
    # connected-component labeling merges them into one node per junction.
    junction_labels, n_junctions = ndimage.label(junction_mask,
                                                  structure=np.ones((3, 3)))
    nodes: dict[tuple[int, int], int] = {}
    node_id = 0

    # Representative pixel = centroid (rounded) of each junction cluster
    if n_junctions > 0:
        centroids = ndimage.center_of_mass(junction_mask, junction_labels,
                                           range(1, n_junctions + 1))
        for cy, cx in centroids:
            r, c = int(round(cy)), int(round(cx))
            # Snap centroid to the nearest actual skeleton pixel
            r, c = _snap_to_skeleton(skeleton, r, c)
            nodes[(r, c)] = node_id
            node_id += 1

    # Every endpoint is its own node
    ep_rows, ep_cols = np.nonzero(endpoint_mask)
    for r, c in zip(ep_rows, ep_cols):
        if (r, c) not in nodes:
            nodes[(r, c)] = node_id
            node_id += 1

    return nodes


def _snap_to_skeleton(skeleton: np.ndarray, r: int, c: int,
                      radius: int = 3) -> tuple[int, int]:
    """Snap (r, c) to the nearest True pixel in *skeleton* within a search window."""
    H, W = skeleton.shape
    r = np.clip(r, 0, H - 1)
    c = np.clip(c, 0, W - 1)
    if skeleton[r, c]:
        return (r, c)

    best, best_d2 = (r, c), float('inf')
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and skeleton[nr, nc]:
                d2 = dr * dr + dc * dc
                if d2 < best_d2:
                    best, best_d2 = (nr, nc), d2
    return best


# ══════════════════════════════════════════════════════════════════════════
# 3.  EDGE TRACING  (BFS walk along skeleton arcs)
# ══════════════════════════════════════════════════════════════════════════
def _is_node_pixel(pos: tuple[int, int],
                   nodes: dict[tuple[int, int], int],
                   neighbor_counts: np.ndarray) -> bool:
    """Check whether *pos* belongs to any node cluster (junction or endpoint)."""
    if pos in nodes:
        return True
    r, c = pos
    return neighbor_counts[r, c] > 2


def _trace_edges(skeleton: np.ndarray,
                 neighbor_counts: np.ndarray,
                 nodes: dict[tuple[int, int], int]
                 ) -> list[tuple[int, int, float, list[tuple[int, int]]]]:
    """Walk the skeleton from every node, tracing each arc until another node is reached.

    Returns
    -------
    edges : list of (node_id_a, node_id_b, physical_length, path_pixels)
    """
    H, W = skeleton.shape
    edges: list[tuple[int, int, float, list[tuple[int, int]]]] = []
    visited_edges: set[tuple[int, int]] = set()   # (min_id, max_id) dedup

    # Build a lookup so junction-cluster pixels resolve to their node id
    junction_lookup = dict(nodes)  # copy; we augment below
    # Expand: assign every junction-cluster pixel to the nearest node id
    junction_mask = skeleton & (neighbor_counts > 2)
    junction_labels, n_junctions = ndimage.label(junction_mask,
                                                  structure=np.ones((3, 3)))
    if n_junctions > 0:
        for label_id in range(1, n_junctions + 1):
            cluster = np.argwhere(junction_labels == label_id)
            # Find which node_id this cluster corresponds to
            matched_nid = None
            for (cr, cc) in cluster:
                if (cr, cc) in nodes:
                    matched_nid = nodes[(cr, cc)]
                    break
            if matched_nid is None:
                # Fallback: assign to closest existing node
                for (cr, cc) in cluster:
                    best_nid, best_d = None, float('inf')
                    for (nr, nc), nid in nodes.items():
                        d = abs(cr - nr) + abs(cc - nc)
                        if d < best_d:
                            best_nid, best_d = nid, d
                    matched_nid = best_nid
                    break
            if matched_nid is not None:
                for (cr, cc) in cluster:
                    junction_lookup[(cr, cc)] = matched_nid

    def _resolve_node(pos: tuple[int, int]) -> int | None:
        return junction_lookup.get(pos)

    # BFS-walk from each node along each outgoing branch
    for start_pos, start_id in nodes.items():
        sr, sc = start_pos
        for i, (dr, dc) in enumerate(_NEIGHBORS_8):
            nr, nc = sr + dr, sc + dc
            if not (0 <= nr < H and 0 <= nc < W) or not skeleton[nr, nc]:
                continue
            # Don't re-enter the same junction cluster
            nid = _resolve_node((nr, nc))
            if nid is not None and nid == start_id:
                continue

            # ── Walk the arc ──────────────────────────────────────────
            path: list[tuple[int, int]] = [start_pos, (nr, nc)]
            length = _STEP_COSTS[i]
            prev = start_pos
            cur = (nr, nc)

            while True:
                # Did we arrive at a node?
                end_nid = _resolve_node(cur)
                if end_nid is not None and cur != start_pos:
                    edge_key = (min(start_id, end_nid), max(start_id, end_nid))
                    if edge_key not in visited_edges:
                        visited_edges.add(edge_key)
                        edges.append((start_id, end_nid, length, path))
                    break

                # Otherwise, continue walking to the single next neighbor
                moved = False
                cr, cc = cur
                for j, (ddr, ddc) in enumerate(_NEIGHBORS_8):
                    nnr, nnc = cr + ddr, cc + ddc
                    if (nnr, nnc) == prev:
                        continue
                    if not (0 <= nnr < H and 0 <= nnc < W):
                        continue
                    if not skeleton[nnr, nnc]:
                        continue
                    # Skip back into the starting junction cluster
                    check_nid = _resolve_node((nnr, nnc))
                    if check_nid is not None and check_nid == start_id and len(path) < 4:
                        continue

                    prev = cur
                    cur = (nnr, nnc)
                    length += _STEP_COSTS[j]
                    path.append(cur)
                    moved = True
                    break

                if not moved:
                    # Dead-end reached that isn't a classified node
                    # (can happen with tiny spurs < 2 px); discard arc
                    break

    return edges


# ══════════════════════════════════════════════════════════════════════════
# 5.  UNION-FIND  (Disjoint Set with Path Compression + Union by Rank)
# ══════════════════════════════════════════════════════════════════════════
class UnionFind:
    """Production-grade Disjoint Set Forest.

    Supports path compression and union-by-rank for near-O(α(n))
    amortized operations, where α is the inverse Ackermann function.

    Used by the topological healer to efficiently track which graph
    nodes belong to which connected component during iterative merging.
    """

    __slots__ = ('_parent', '_rank', '_n_components')

    def __init__(self, n: int):
        """Initialize n singleton sets {0}, {1}, …, {n-1}."""
        self._parent: List[int] = list(range(n))
        self._rank: List[int] = [0] * n
        self._n_components: int = n

    # ── Core operations ───────────────────────────────────────────────

    def find(self, x: int) -> int:
        """Return the canonical representative of the set containing *x*.

        Uses **path compression**: every node visited during the
        traversal is pointed directly at the root, flattening the tree
        for subsequent queries.
        """
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression pass
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, x: int, y: int) -> bool:
        """Merge the sets containing *x* and *y*.

        Uses **union by rank**: the shorter tree is always attached
        under the taller tree, keeping the structure balanced.

        Returns True if a merge actually occurred (they were in
        different sets), False if they were already in the same set.
        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        # Attach smaller-rank tree under larger-rank tree
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1
        self._n_components -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        """Check whether *x* and *y* are in the same set."""
        return self.find(x) == self.find(y)

    @property
    def n_components(self) -> int:
        """Current number of disjoint components."""
        return self._n_components

    def component_map(self) -> dict[int, list[int]]:
        """Return {root_id: [member_ids]} for every component."""
        comp: dict[int, list[int]] = {}
        for i in range(len(self._parent)):
            root = self.find(i)
            comp.setdefault(root, []).append(i)
        return comp


# ══════════════════════════════════════════════════════════════════════════
# 6.  ANGULAR DIRECTION ESTIMATION
# ══════════════════════════════════════════════════════════════════════════
def _estimate_direction(G: nx.Graph, node_id: int,
                        depth: int = _DEFAULT_DIRECTION_DEPTH) -> Optional[np.ndarray]:
    """Estimate the local road direction (unit tangent vector) at a node.

    Walks outward from *node_id* along attached edges, collecting the
    first *depth* skeleton pixels, and fits a direction vector via the
    displacement from the node to the furthest collected pixel.

    Parameters
    ----------
    G : nx.Graph
        The road graph (nodes must have 'pos' attribute).
    node_id : int
        The node whose tangent direction we want.
    depth : int
        How many pixels along the skeleton to consider.

    Returns
    -------
    direction : np.ndarray of shape (2,) — unit vector (row, col), or
                None if no edges are attached.
    """
    pos = np.array(G.nodes[node_id]['pos'], dtype=np.float64)

    # Collect pixel paths from all edges incident to this node
    best_path: Optional[list] = None
    best_length = 0.0
    for _, neighbor, data in G.edges(node_id, data=True):
        path = data.get('path', [])
        length = data.get('w', 0.0)
        if length > best_length and len(path) >= 2:
            best_path = path
            best_length = length

    if best_path is None or len(best_path) < 2:
        return None

    # Determine path orientation: the path may start or end at our node
    first = np.array(best_path[0], dtype=np.float64)
    last = np.array(best_path[-1], dtype=np.float64)
    if np.linalg.norm(first - pos) < np.linalg.norm(last - pos):
        # Path starts at our node; walk forward
        walk = best_path[:depth + 1]
    else:
        # Path ends at our node; walk backward
        walk = best_path[-(depth + 1):][::-1]

    if len(walk) < 2:
        return None

    # Direction = displacement from node toward the skeleton interior
    origin = np.array(walk[0], dtype=np.float64)
    target = np.array(walk[-1], dtype=np.float64)
    vec = target - origin
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return None
    return vec / norm


def _angular_cost(dir_a: Optional[np.ndarray],
                  dir_b: Optional[np.ndarray],
                  bridge_vec: np.ndarray) -> float:
    """Compute the angular alignment penalty for a proposed bridge.

    The bridge is "good" if:
      1. It aligns with the direction of road A (at the A end).
      2. It aligns with the direction of road B (at the B end).

    We measure the angle between each road's tangent and the bridge
    direction.  Perfectly aligned = 0 cost.  Perpendicular = π/2 cost.

    Parameters
    ----------
    dir_a, dir_b : optional unit vectors  (road tangent at each end)
    bridge_vec   : unit vector from node A → node B

    Returns
    -------
    penalty : float in [0.0, π].  Lower is better.
    """
    penalty = 0.0

    if dir_a is not None:
        # The bridge should continue in roughly the same direction as road A.
        # We compare dir_a (pointing INTO the road from A) with bridge_vec
        # (pointing from A toward B).  A good bridge should be roughly
        # anti-parallel to dir_a (since dir_a points inward, and bridge
        # goes outward), OR parallel if the road curves back.
        # Use the minimum angle between bridge_vec and ±dir_a.
        cos_a = np.clip(np.dot(dir_a, bridge_vec), -1.0, 1.0)
        angle_a = math.acos(abs(cos_a))  # [0, π/2]
        penalty += angle_a

    if dir_b is not None:
        cos_b = np.clip(np.dot(dir_b, -bridge_vec), -1.0, 1.0)
        angle_b = math.acos(abs(cos_b))
        penalty += angle_b

    return penalty


# ══════════════════════════════════════════════════════════════════════════
# 7.  MST-BASED TOPOLOGICAL HEALING
# ══════════════════════════════════════════════════════════════════════════
def _heal_graph(G: nx.Graph,
                max_gap_px: float = _DEFAULT_MAX_GAP_PX,
                angular_penalty_weight: float = _DEFAULT_ANGULAR_PENALTY,
                direction_depth: int = _DEFAULT_DIRECTION_DEPTH,
                ) -> Tuple[nx.Graph, int]:
    """Bridge disconnected components using MST-guided topological healing.

    Algorithm
    ---------
    1. Initialise a Union-Find over all graph nodes.
    2. Register existing edges (they already connect their endpoints).
    3. For every pair of **endpoint** nodes in *different* components,
       compute a healing cost:
           cost  =  euclidean_distance  +  angular_penalty_weight × angular_cost
       Only candidates with euclidean_distance ≤ max_gap_px are kept.
    4. Sort candidates by cost (ascending).
    5. Greedily pick the cheapest candidate that merges two distinct
       components (Kruskal-style MST construction).
    6. Insert the accepted edges into G as synthetic edges.

    Parameters
    ----------
    G : nx.Graph
        The road graph (may be disconnected).
    max_gap_px : float
        Maximum Euclidean distance (pixels) to consider healing.
    angular_penalty_weight : float
        Weight multiplier for angular misalignment in the cost function.
    direction_depth : int
        Pixel look-back depth for tangent estimation.

    Returns
    -------
    G : nx.Graph
        The same graph object, now with synthetic edges inserted.
    n_healed : int
        Number of synthetic edges that were added.
    """
    n_nodes = G.number_of_nodes()
    if n_nodes < 2:
        return G, 0

    # Map node IDs to contiguous 0..n-1 for Union-Find
    node_list = sorted(G.nodes())
    node_to_idx = {nid: i for i, nid in enumerate(node_list)}

    uf = UnionFind(n_nodes)

    # Register existing edges
    for u, v in G.edges():
        uf.union(node_to_idx[u], node_to_idx[v])

    if uf.n_components <= 1:
        return G, 0  # Already fully connected — nothing to heal

    # Pre-compute positions and directions for all nodes
    positions: dict[int, np.ndarray] = {}
    directions: dict[int, Optional[np.ndarray]] = {}
    for nid in node_list:
        positions[nid] = np.array(G.nodes[nid]['pos'], dtype=np.float64)
        directions[nid] = _estimate_direction(G, nid, depth=direction_depth)

    # Collect candidate healing edges (only between different components)
    # Prioritize endpoint nodes — they are the natural bridge points
    candidates: List[Tuple[float, int, int]] = []  # (cost, node_a, node_b)

    for i, nid_a in enumerate(node_list):
        idx_a = node_to_idx[nid_a]
        pos_a = positions[nid_a]
        dir_a = directions[nid_a]

        for j in range(i + 1, len(node_list)):
            nid_b = node_list[j]
            idx_b = node_to_idx[nid_b]

            # Skip pairs already in the same component
            if uf.find(idx_a) == uf.find(idx_b):
                continue

            pos_b = positions[nid_b]
            dist = float(np.linalg.norm(pos_a - pos_b))

            # Gate: skip if too far
            if dist > max_gap_px:
                continue

            # Bridge direction vector
            bridge_vec = (pos_b - pos_a)
            bridge_norm = np.linalg.norm(bridge_vec)
            if bridge_norm < 1e-8:
                continue
            bridge_unit = bridge_vec / bridge_norm

            dir_b = directions[nid_b]
            ang_cost = _angular_cost(dir_a, dir_b, bridge_unit)

            total_cost = dist + angular_penalty_weight * ang_cost
            candidates.append((total_cost, nid_a, nid_b))

    # Sort by cost (Kruskal's algorithm)
    candidates.sort(key=lambda x: x[0])

    # Greedily accept cheapest inter-component edges (MST of inter-component graph)
    n_healed = 0
    for cost, nid_a, nid_b in candidates:
        idx_a = node_to_idx[nid_a]
        idx_b = node_to_idx[nid_b]
        if uf.union(idx_a, idx_b):
            # Build a straight-line synthetic path
            pos_a = positions[nid_a]
            pos_b = positions[nid_b]
            dist = float(np.linalg.norm(pos_a - pos_b))
            n_steps = max(2, int(np.ceil(dist)))
            synthetic_path = [
                (int(round(pos_a[0] + t * (pos_b[0] - pos_a[0]) / (n_steps - 1))),
                 int(round(pos_a[1] + t * (pos_b[1] - pos_a[1]) / (n_steps - 1))))
                for t in range(n_steps)
            ]
            G.add_edge(nid_a, nid_b,
                       w=dist,
                       path=synthetic_path,
                       synthetic=True,
                       healing_cost=cost)
            n_healed += 1

            # Early exit: if the graph is now fully connected, stop
            if uf.n_components <= 1:
                break

    return G, n_healed


# ══════════════════════════════════════════════════════════════════════════
# 8.  PUBLIC API:  GRAPH CONSTRUCTION + HEALING
# ══════════════════════════════════════════════════════════════════════════
def mask_to_graph(
    mask: np.ndarray,
    *,
    prune_min_length: float = 0.0,
    morphological_closing: int = 3,
    heal: bool = True,
    max_gap_px: float = _DEFAULT_MAX_GAP_PX,
    angular_penalty_weight: float = _DEFAULT_ANGULAR_PENALTY,
) -> tuple[nx.Graph, np.ndarray, dict[tuple[int, int], int]]:
    """Convert a binary road mask to a weighted, undirected, healed graph.

    Parameters
    ----------
    mask : np.ndarray
        2-D binary array (H, W), dtype uint8.  Non-zero pixels = road.
    prune_min_length : float, optional
        Drop edges shorter than this many pixels (removes tiny spurs).
        Default 0.0 (keep everything).
    morphological_closing : int, optional
        Radius for morphological closing pre-step.  Set to 0 to disable.
    heal : bool, optional
        If True (default), run the MST-based topological healer to
        bridge disconnected components.
    max_gap_px : float, optional
        Maximum gap distance (pixels) the healer will attempt to bridge.
    angular_penalty_weight : float, optional
        Weight for angular alignment in the healing cost function.

    Returns
    -------
    G : networkx.Graph
        Undirected graph where:
        • Each node has attributes ``pos = (row, col)`` and ``kind``
          (``"junction"`` or ``"endpoint"``).
        • Each edge has attributes ``w`` (physical pixel length),
          ``path`` (ordered list of (row, col) skeleton pixels), and
          optionally ``synthetic = True`` for healed edges.
    skeleton : np.ndarray (bool)
        The 1-pixel-wide skeleton that was extracted.
    nodes : dict[tuple[int, int], int]
        Mapping  (row, col) → node_id.

    Example
    -------
    >>> import numpy as np
    >>> from mask_to_graph import mask_to_graph
    >>> mask = my_model_output   # (H, W) uint8
    >>> G, skeleton, nodes = mask_to_graph(mask, heal=True)
    >>> nx.is_connected(G)
    True
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {mask.shape}")

    # Step 0 – Morphological pre-cleaning
    if morphological_closing > 0:
        cleaned = _morphological_preclean(mask, closing_radius=morphological_closing)
    else:
        cleaned = mask.astype(bool)

    # Step 1 – Skeletonize
    skeleton = _skeletonize_mask(cleaned)

    # Guard: empty skeleton → empty graph
    if not skeleton.any():
        G = nx.Graph()
        return G, skeleton, {}

    # Step 2 – Classify pixels
    ncounts = _neighbor_count_map(skeleton)
    nodes = _extract_nodes(skeleton, ncounts)

    # Guard: if no junctions and no endpoints, the skeleton is a single
    # closed loop.  Insert an arbitrary node so the graph isn't empty.
    if not nodes:
        skel_pixels = np.argwhere(skeleton)
        anchor = tuple(skel_pixels[0])
        nodes[anchor] = 0

    # Step 3 – Trace edges
    raw_edges = _trace_edges(skeleton, ncounts, nodes)

    # Step 4 – Build the graph
    G = nx.Graph()

    for pos, nid in nodes.items():
        r, c = pos
        kind = "junction" if ncounts[r, c] > 2 else "endpoint"
        G.add_node(nid, pos=pos, kind=kind)

    for u, v, length, path in raw_edges:
        if length < prune_min_length:
            continue
        G.add_edge(u, v, w=length, path=path, synthetic=False)

    # Step 5 – Topological Healing
    n_healed = 0
    if heal and G.number_of_nodes() >= 2:
        G, n_healed = _heal_graph(
            G,
            max_gap_px=max_gap_px,
            angular_penalty_weight=angular_penalty_weight,
        )

    # Store healing metadata on the graph object
    G.graph['n_healed_edges'] = n_healed
    G.graph['n_components_post_heal'] = nx.number_connected_components(G) if G.number_of_nodes() > 0 else 0

    return G, skeleton, nodes


# ══════════════════════════════════════════════════════════════════════════
# CLI  –  Sanity check with a FRAGMENTED synthetic mask
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── Build a deliberately FRAGMENTED mask to test healing ──────────
    H, W = 256, 256
    mask = np.zeros((H, W), dtype=np.uint8)

    # Horizontal road with a GAP in the middle (simulating tree canopy)
    mask[125:131, 20:110] = 1    # Left segment
    # GAP: columns 110–140 are missing (30px occlusion)
    mask[125:131, 140:236] = 1   # Right segment

    # Vertical road with a GAP (simulating shadow)
    mask[20:100, 125:131] = 1    # Top segment
    # GAP: rows 100–130 are missing
    mask[130:236, 125:131] = 1   # Bottom segment

    # Isolated diagonal road (should be connected to the network)
    for offset in range(-2, 3):
        rr = np.arange(30, 90)
        cc = np.clip(200 + offset, 0, W - 1)
        mask[rr, np.full_like(rr, cc)] = 1

    print("=" * 70)
    print("  mask_to_graph.py  –  Phase II Topological Healing Sanity Check")
    print("=" * 70)
    print(f"  Mask shape       : {mask.shape}")
    print(f"  Road pixels      : {int(mask.sum()):,}")

    # ── Without healing ──────────────────────────────────────────────
    G_raw, skel, _ = mask_to_graph(mask, prune_min_length=5.0, heal=False)
    n_comp_raw = nx.number_connected_components(G_raw) if G_raw.number_of_nodes() > 0 else 0

    print()
    print(f"  -- Before Healing -----------------------------------------")
    print(f"  Nodes |V|        : {G_raw.number_of_nodes()}")
    print(f"  Edges |E|        : {G_raw.number_of_edges()}")
    print(f"  Components       : {n_comp_raw}")

    # ── With healing ─────────────────────────────────────────────────
    G_healed, skel, nodes = mask_to_graph(mask, prune_min_length=5.0, heal=True, max_gap_px=60.0)
    n_comp_healed = G_healed.graph.get('n_components_post_heal', '?')
    n_healed = G_healed.graph.get('n_healed_edges', 0)

    print()
    print(f"  -- After Healing ------------------------------------------")
    print(f"  Nodes |V|        : {G_healed.number_of_nodes()}")
    print(f"  Edges |E|        : {G_healed.number_of_edges()}")
    print(f"  Synthetic edges  : {n_healed}")
    print(f"  Components       : {n_comp_healed}")
    print()

    for nid in sorted(G_healed.nodes):
        data = G_healed.nodes[nid]
        print(f"  Node {nid:>3d}  |  pos={data['pos']}  |  kind={data['kind']}")

    print()
    for u, v, attr in G_healed.edges(data=True):
        syn_tag = " [SYNTHETIC]" if attr.get('synthetic', False) else ""
        cost_tag = f"  heal_cost={attr['healing_cost']:.2f}" if 'healing_cost' in attr else ""
        print(f"  Edge ({u:>2d} <-> {v:>2d})  |  w = {attr['w']:>8.2f} px{syn_tag}{cost_tag}")

    print()
    is_connected = nx.is_connected(G_healed) if G_healed.number_of_nodes() > 0 else False
    status = "PASS" if is_connected else "FAIL"
    print(f"  Connectivity test : {status}  (nx.is_connected = {is_connected})")
    print(f"  [OK] Healed graph is ready for Dijkstra / A* pathfinding.")
    print("=" * 70)
