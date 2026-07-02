"""
mask_to_graph.py  –  CV ➜ Graph Theory Bridge
══════════════════════════════════════════════════════════════════════════
Converts a binary road-segmentation mask (numpy uint8 array) into a
fully weighted, undirected networkx.Graph suitable for shortest-path
routing (Dijkstra, A*, etc.).

Pipeline
────────
1. Morphological skeletonization  (skimage.morphology.skeletonize)
2. Node extraction               (junction & endpoint detection via 8-neighbor kernel)
3. Edge tracing                   (BFS walk along skeleton arcs between nodes)
4. Graph construction             (networkx.Graph with pixel-length edge weights)

Author : NetSight Project
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from scipy import ndimage
from skimage.morphology import skeletonize

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


# ══════════════════════════════════════════════════════════════════════════
# 1.  SKELETONIZE
# ══════════════════════════════════════════════════════════════════════════
def _skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Thin a binary road mask to a 1-pixel-wide centerline.

    Parameters
    ----------
    mask : np.ndarray
        2-D binary array (H, W) with dtype uint8.  Non-zero = road.

    Returns
    -------
    skeleton : np.ndarray  (bool, same shape)
    """
    binary = mask.astype(bool)
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
# 4.  GRAPH CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════
def mask_to_graph(
    mask: np.ndarray,
    *,
    prune_min_length: float = 0.0,
) -> tuple[nx.Graph, np.ndarray, dict[tuple[int, int], int]]:
    """Convert a binary road mask to a weighted, undirected graph.

    Parameters
    ----------
    mask : np.ndarray
        2-D binary array (H, W), dtype uint8.  Non-zero pixels = road.
    prune_min_length : float, optional
        Drop edges shorter than this many pixels (removes tiny spurs).
        Default 0.0 (keep everything).

    Returns
    -------
    G : networkx.Graph
        Undirected graph where:
        • Each node has attributes ``pos = (row, col)`` and ``kind``
          (``"junction"`` or ``"endpoint"``).
        • Each edge has attributes ``w`` (physical pixel length) and
          ``path`` (ordered list of (row, col) skeleton pixels).
    skeleton : np.ndarray (bool)
        The 1-pixel-wide skeleton that was extracted.
    nodes : dict[tuple[int, int], int]
        Mapping  (row, col) → node_id.

    Example
    -------
    >>> import numpy as np
    >>> from mask_to_graph import mask_to_graph
    >>> mask = my_model_output   # (H, W) uint8
    >>> G, skeleton, nodes = mask_to_graph(mask)
    >>> nx.shortest_path(G, source=0, target=5, weight='w')
    [0, 3, 5]
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {mask.shape}")

    # Step 1 – Skeletonize
    skeleton = _skeletonize_mask(mask)

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
        G.add_edge(u, v, w=length, path=path)

    return G, skeleton, nodes


# ══════════════════════════════════════════════════════════════════════════
# CLI  –  Quick sanity-check with a synthetic mask
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── Build a synthetic crossroads mask for demonstration ───────────
    H, W = 256, 256
    mask = np.zeros((H, W), dtype=np.uint8)

    # Horizontal road  (row 128, width 6 px)
    mask[125:131, 20:236] = 1

    # Vertical road    (col 128, width 6 px)
    mask[20:236, 125:131] = 1

    # Diagonal road    (bottom-left → top-right, width ~4 px)
    for offset in range(-2, 3):
        rr = np.arange(30, 226)
        cc = np.clip(30 + (rr - 30) + offset, 0, W - 1)
        valid = cc < W
        mask[rr[valid], cc[valid]] = 1

    print("=" * 60)
    print("  mask_to_graph.py  -  Synthetic Sanity Check")
    print("=" * 60)
    print(f"  Mask shape      : {mask.shape}")
    print(f"  Road pixels     : {int(mask.sum()):,}")

    G, skeleton, nodes = mask_to_graph(mask, prune_min_length=5.0)

    print(f"  Skeleton pixels : {int(skeleton.sum()):,}")
    print(f"  Nodes |V|       : {G.number_of_nodes()}")
    print(f"  Edges |E|       : {G.number_of_edges()}")
    print()

    for nid in sorted(G.nodes):
        data = G.nodes[nid]
        print(f"  Node {nid:>3d}  |  pos={data['pos']}  |  kind={data['kind']}")

    print()
    for u, v, attr in G.edges(data=True):
        print(f"  Edge ({u:>2d} <-> {v:>2d})  |  w = {attr['w']:>8.2f} px")

    print()
    print("  [OK] Graph is ready for Dijkstra / A* pathfinding.")
    print("=" * 60)
