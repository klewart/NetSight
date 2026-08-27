"""
network_analysis.py  --  Structural Intelligence & Stress Testing Engine
========================================================================
Production-grade network analysis for the NetSight disaster management
framework.  Provides node-centric centrality analysis, systematic
ablation simulation, and formal Resilience Index computation.

Designed for ISRO NNRMS urban infrastructure resilience assessment.

Key Capabilities
----------------
1.  **Centrality Calculation**
    Node betweenness centrality (weighted) to identify Gatekeeper Nodes --
    intersections that act as single points of failure across the city.

2.  **Node Ablation Simulation**
    Systematically removes critical nodes in descending centrality order,
    recording global efficiency after each step to produce a degradation
    curve that quantifies network fragility.

3.  **Resilience Index**
    R = E_post / E_pre  (ratio of global efficiencies).
    R ~ 1.0 = robust network;  R ~ 0.0 = catastrophically fragile.

4.  **Spatial Disaster Simulation**
    Removes all nodes (and incident edges) within a physical blast radius
    of a specified epicenter, simulating floods, earthquakes, or
    explosions.

Author : NetSight Project  (ISRO NNRMS / Disaster Management Framework)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx
import numpy as np


# =========================================================================
# DATA STRUCTURES
# =========================================================================
@dataclass
class GatekeeperNode:
    """A node identified as a critical bottleneck in the road network."""
    node_id: int
    pos: Tuple[int, int]
    kind: str                    # "junction" or "endpoint"
    betweenness: float           # normalised betweenness centrality [0, 1]
    degree: int                  # number of incident edges
    rank: int                    # 1 = most critical


@dataclass
class AblationStep:
    """Metrics snapshot after removing one node during ablation."""
    step: int                    # 0 = baseline (no removal)
    removed_node_id: Optional[int]
    removed_node_pos: Optional[Tuple[int, int]]
    global_efficiency: float
    resilience_index: float      # E_current / E_baseline
    n_components: int
    largest_component_size: int
    isolated_nodes: int
    avg_shortest_path: Optional[float]  # within largest CC; None if trivial


@dataclass
class AblationResult:
    """Complete result of a systematic node ablation study."""
    steps: List[AblationStep]
    gatekeepers: List[GatekeeperNode]
    baseline_efficiency: float
    final_efficiency: float
    final_resilience_index: float
    total_nodes_removed: int


@dataclass
class DisasterResult:
    """Result of a spatial disaster (blast radius) simulation."""
    G_pre: nx.Graph
    G_post: nx.Graph
    removed_nodes: Set[int]
    removed_node_positions: List[Tuple[int, int]]
    removed_edges: Set[FrozenSet]
    bottleneck_edges: Set[FrozenSet]
    eff_pre: float
    eff_post: float
    resilience_index: float
    pct_drop: float
    n_components: int
    isolated_nodes: int
    time_penalty_pct: float
    edges_destroyed: int
    nodes_destroyed: int


# =========================================================================
# 1.  CENTRALITY CALCULATION
# =========================================================================
def compute_node_centrality(G: nx.Graph,
                            weight: str = "w",
                            top_k: Optional[int] = None,
                            ) -> List[GatekeeperNode]:
    """Compute node betweenness centrality and return ranked Gatekeeper Nodes.

    Parameters
    ----------
    G : nx.Graph
        The road network graph.  Nodes must have ``pos`` and ``kind``
        attributes.  Edges should have weight attribute *weight*.
    weight : str
        Edge attribute to use as weight for shortest-path computation.
    top_k : int or None
        If set, return only the top-k most critical nodes.
        If None, return all nodes ranked by centrality.

    Returns
    -------
    gatekeepers : list of GatekeeperNode, sorted by betweenness descending.
    """
    if G.number_of_nodes() == 0:
        return []

    # Weighted betweenness centrality (normalised)
    bc = nx.betweenness_centrality(G, weight=weight, normalized=True)

    # Build ranked list
    ranked: List[GatekeeperNode] = []
    for nid, centrality in sorted(bc.items(), key=lambda x: x[1], reverse=True):
        data = G.nodes[nid]
        ranked.append(GatekeeperNode(
            node_id=nid,
            pos=tuple(int(x) for x in data.get("pos", (0, 0))),
            kind=data.get("kind", "unknown"),
            betweenness=centrality,
            degree=G.degree(nid),
            rank=len(ranked) + 1,
        ))

    if top_k is not None:
        ranked = ranked[:top_k]

    return ranked


def compute_edge_criticality(G: nx.Graph, weight: str = "w") -> Dict[FrozenSet, float]:
    """Compute edge betweenness centrality for Heatmap Overlay.
    
    Returns
    -------
    dict: Mapping from frozenset(u, v) to normalized centrality score [0, 1].
    """
    if G.number_of_edges() == 0:
        return {}
        
    edge_bc = nx.edge_betweenness_centrality(G, weight=weight, normalized=True)
    
    # Normalize to [0, 1] across all edges to map directly to a color scale
    max_bc = max(edge_bc.values()) if edge_bc else 1.0
    if max_bc == 0:
        max_bc = 1.0
        
    return {frozenset(e): bc / max_bc for e, bc in edge_bc.items()}


# =========================================================================
# 2.  NODE ABLATION SIMULATION
# =========================================================================
def _compute_avg_shortest_path(G: nx.Graph, weight: str = "w") -> Optional[float]:
    """Average shortest path length in the largest connected component.

    Returns None if the largest component has fewer than 2 nodes.
    """
    if G.number_of_nodes() < 2:
        return None

    components = sorted(nx.connected_components(G), key=len, reverse=True)
    largest_cc = G.subgraph(components[0]).copy()

    if largest_cc.number_of_nodes() < 2:
        return None

    try:
        return nx.average_shortest_path_length(largest_cc, weight=weight)
    except (nx.NetworkXError, ZeroDivisionError):
        return None


def _snapshot_metrics(G: nx.Graph,
                      step: int,
                      baseline_eff: float,
                      removed_id: Optional[int] = None,
                      removed_pos: Optional[Tuple[int, int]] = None,
                      weight: str = "w",
                      ) -> AblationStep:
    """Capture a complete metrics snapshot of the graph state."""
    eff = nx.global_efficiency(G) if G.number_of_nodes() > 1 else 0.0
    ri = eff / baseline_eff if baseline_eff > 0 else 0.0

    components = sorted(nx.connected_components(G), key=len, reverse=True) \
        if G.number_of_nodes() > 0 else []
    n_comp = len(components)
    largest_size = len(components[0]) if components else 0
    isolated = sum(len(c) for c in components[1:]) if n_comp > 1 else 0

    avg_sp = _compute_avg_shortest_path(G, weight=weight)

    return AblationStep(
        step=step,
        removed_node_id=removed_id,
        removed_node_pos=removed_pos,
        global_efficiency=eff,
        resilience_index=ri,
        n_components=n_comp,
        largest_component_size=largest_size,
        isolated_nodes=isolated,
        avg_shortest_path=avg_sp,
    )


def run_node_ablation(G: nx.Graph,
                      n_steps: Optional[int] = None,
                      fraction: float = 0.20,
                      weight: str = "w",
                      ) -> AblationResult:
    """Systematically remove critical nodes and record degradation.

    The algorithm:
    1. Compute baseline global efficiency.
    2. At each step, recompute node betweenness centrality on the
       *current* (damaged) graph and remove the node with the highest
       centrality.  This adaptive approach is more realistic than using
       a static ranking, because the most critical node changes as the
       network degrades.
    3. Record global efficiency, resilience index, component count, and
       average shortest path length after each removal.

    Parameters
    ----------
    G : nx.Graph
        The road network (will NOT be modified; a copy is used).
    n_steps : int or None
        Number of nodes to remove.  If None, uses ``fraction``.
    fraction : float
        Fraction of nodes to remove (default 20%).  Ignored if
        ``n_steps`` is set.
    weight : str
        Edge weight attribute for shortest-path computation.

    Returns
    -------
    AblationResult with full degradation trajectory.
    """
    G_work = G.copy()
    n_total = G_work.number_of_nodes()

    if n_steps is None:
        n_steps = max(1, int(math.ceil(n_total * fraction)))
    n_steps = min(n_steps, n_total - 1)  # can't remove all nodes

    # Baseline snapshot
    baseline_eff = nx.global_efficiency(G_work) if n_total > 1 else 0.0

    # Compute initial gatekeepers on the full graph
    initial_gatekeepers = compute_node_centrality(G, weight=weight)

    steps: List[AblationStep] = []
    steps.append(_snapshot_metrics(G_work, step=0, baseline_eff=baseline_eff,
                                   weight=weight))

    for i in range(1, n_steps + 1):
        if G_work.number_of_nodes() < 2:
            break

        # Adaptive re-ranking: recompute betweenness on current graph
        bc = nx.betweenness_centrality(G_work, weight=weight, normalized=True)
        if not bc:
            break

        # Find the most critical node in the current state
        target_nid = max(bc, key=bc.get)
        target_pos = tuple(int(x) for x in G_work.nodes[target_nid].get("pos", (0, 0)))

        # Remove the node (and all incident edges)
        G_work.remove_node(target_nid)

        steps.append(_snapshot_metrics(
            G_work, step=i, baseline_eff=baseline_eff,
            removed_id=target_nid, removed_pos=target_pos,
            weight=weight,
        ))

    final_eff = steps[-1].global_efficiency if steps else 0.0
    final_ri = steps[-1].resilience_index if steps else 0.0

    return AblationResult(
        steps=steps,
        gatekeepers=initial_gatekeepers,
        baseline_efficiency=baseline_eff,
        final_efficiency=final_eff,
        final_resilience_index=final_ri,
        total_nodes_removed=len(steps) - 1,  # exclude baseline
    )


# =========================================================================
# 3.  SPATIAL DISASTER SIMULATION  (Node-Centric)
# =========================================================================
_BOTTLENECK_FRACTION = 0.10  # top 10% of edges become bottlenecks


def simulate_disaster(G: nx.Graph,
                      epicenter: Optional[Tuple[float, float]] = None,
                      blast_radius_px: float = 80.0,
                      fallback_fraction: float = 0.05,
                      weight: str = "w",
                      ) -> DisasterResult:
    """Simulate a localised disaster by removing all nodes within a radius.

    Parameters
    ----------
    G : nx.Graph
        The road network.
    epicenter : (row, col) or None
        Geographic center of the disaster.  If None, uses the image
        center (256, 256) as default.
    blast_radius_px : float
        Radius (pixels) of the disaster zone.
    fallback_fraction : float
        If blast_radius_px <= 0, fall back to removing the top
        ``fallback_fraction`` of nodes by betweenness centrality
        (analytical stress test).
    weight : str
        Edge weight attribute.

    Returns
    -------
    DisasterResult with pre/post graphs and all impact metrics.
    """
    if G.number_of_nodes() < 2:
        return DisasterResult(
            G_pre=G, G_post=G,
            removed_nodes=set(), removed_node_positions=[],
            removed_edges=set(), bottleneck_edges=set(),
            eff_pre=0.0, eff_post=0.0, resilience_index=1.0,
            pct_drop=0.0, n_components=1 if G.number_of_nodes() > 0 else 0,
            isolated_nodes=0, time_penalty_pct=0.0,
            edges_destroyed=0, nodes_destroyed=0,
        )

    eff_pre = nx.global_efficiency(G)
    G_post = G.copy()
    removed_nodes: Set[int] = set()
    removed_positions: List[Tuple[int, int]] = []

    if blast_radius_px > 0:
        # Spatial disaster: remove all nodes within radius
        if epicenter is None:
            # Default: center of a 512x512 tile
            epicenter = (256.0, 256.0)
        center = np.array(epicenter, dtype=np.float64)

        for nid, data in G.nodes(data=True):
            pos = np.array(data.get("pos", (0, 0)), dtype=np.float64)
            if np.linalg.norm(pos - center) <= blast_radius_px:
                removed_nodes.add(nid)
                removed_positions.append(tuple(int(x) for x in pos))
    else:
        # Analytical: remove top-N by betweenness
        bc = nx.betweenness_centrality(G, weight=weight, normalized=True)
        n_remove = max(1, int(math.ceil(len(bc) * fallback_fraction)))
        sorted_nodes = sorted(bc.items(), key=lambda x: x[1], reverse=True)
        for nid, _ in sorted_nodes[:n_remove]:
            pos = tuple(int(x) for x in G.nodes[nid].get("pos", (0, 0)))
            removed_nodes.add(nid)
            removed_positions.append(pos)

    # Track which edges are destroyed (for visualization)
    removed_edges: Set[FrozenSet] = set()
    for nid in removed_nodes:
        if G_post.has_node(nid):
            for neighbor in list(G_post.neighbors(nid)):
                removed_edges.add(frozenset((nid, neighbor)))
            G_post.remove_node(nid)

    # Post-disaster metrics
    eff_post = nx.global_efficiency(G_post) if G_post.number_of_nodes() > 1 else 0.0
    ri = eff_post / eff_pre if eff_pre > 0 else 0.0
    pct_drop = ((eff_pre - eff_post) / eff_pre * 100) if eff_pre > 0 else 0.0
    time_penalty = ((eff_pre / eff_post) - 1.0) * 100 if eff_post > 0 else 999.0

    components = sorted(nx.connected_components(G_post), key=len, reverse=True) \
        if G_post.number_of_nodes() > 0 else []
    n_comp = len(components)
    isolated = sum(len(c) for c in components[1:]) if n_comp > 1 else 0

    # Identify new bottleneck edges in the damaged network
    bottlenecks: Set[FrozenSet] = set()
    if G_post.number_of_edges() > 0:
        new_bc = nx.edge_betweenness_centrality(G_post, weight=weight)
        n_bn = max(1, int(math.ceil(len(new_bc) * _BOTTLENECK_FRACTION)))
        bottlenecks = {frozenset(e) for e, _ in sorted(
            new_bc.items(), key=lambda x: x[1], reverse=True
        )[:n_bn]}

    return DisasterResult(
        G_pre=G, G_post=G_post,
        removed_nodes=removed_nodes,
        removed_node_positions=removed_positions,
        removed_edges=removed_edges,
        bottleneck_edges=bottlenecks,
        eff_pre=eff_pre, eff_post=eff_post,
        resilience_index=ri,
        pct_drop=pct_drop,
        n_components=n_comp,
        isolated_nodes=isolated,
        time_penalty_pct=time_penalty,
        edges_destroyed=len(removed_edges),
        nodes_destroyed=len(removed_nodes),
    )


# =========================================================================
# 4.  CONVENIENCE:  FULL ANALYSIS PIPELINE
# =========================================================================
def full_analysis(G: nx.Graph,
                  ablation_fraction: float = 0.20,
                  weight: str = "w",
                  ) -> Tuple[List[GatekeeperNode], AblationResult]:
    """Run the complete structural intelligence pipeline.

    Returns
    -------
    gatekeepers : list of GatekeeperNode
    ablation    : AblationResult with degradation trajectory
    """
    gatekeepers = compute_node_centrality(G, weight=weight)
    ablation = run_node_ablation(G, fraction=ablation_fraction, weight=weight)
    return gatekeepers, ablation


# =========================================================================
# CLI SANITY CHECK
# =========================================================================
if __name__ == "__main__":
    # Build a small test graph (diamond + tail)
    #
    #     0 --- 1 --- 2
    #      \   / \   /
    #       \ /   \ /
    #        3 --- 4 --- 5
    #
    G = nx.Graph()
    positions = {
        0: (0, 0), 1: (0, 100), 2: (0, 200),
        3: (100, 50), 4: (100, 150), 5: (100, 250),
    }
    for nid, pos in positions.items():
        G.add_node(nid, pos=pos, kind="junction" if nid in (1, 4) else "endpoint")

    edges = [
        (0, 1, 100), (1, 2, 100), (0, 3, 112),
        (1, 3, 112), (1, 4, 112), (2, 4, 112),
        (3, 4, 100), (4, 5, 100),
    ]
    for u, v, w in edges:
        G.add_edge(u, v, w=float(w), path=[positions[u], positions[v]], synthetic=False)

    print("=" * 70)
    print("  network_analysis.py  --  Phase III Sanity Check")
    print("=" * 70)
    print(f"  Graph: |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")
    print()

    # --- 1. Centrality ---
    gatekeepers = compute_node_centrality(G)
    print("  -- Gatekeeper Nodes (by Betweenness Centrality) --")
    for gk in gatekeepers:
        print(f"    Rank {gk.rank:>2d}  |  Node {gk.node_id:>2d}  |  "
              f"BC={gk.betweenness:.4f}  |  degree={gk.degree}  |  "
              f"pos={gk.pos}  |  {gk.kind}")
    print()

    # --- 2. Ablation ---
    ablation = run_node_ablation(G, n_steps=3)
    print("  -- Node Ablation Degradation Curve --")
    print(f"    {'Step':>4s}  {'Removed':>8s}  {'Efficiency':>10s}  "
          f"{'R-Index':>8s}  {'Components':>10s}  {'Isolated':>8s}")
    for s in ablation.steps:
        removed_str = str(s.removed_node_id) if s.removed_node_id is not None else "---"
        print(f"    {s.step:>4d}  {removed_str:>8s}  {s.global_efficiency:>10.4f}  "
              f"{s.resilience_index:>8.4f}  {s.n_components:>10d}  {s.isolated_nodes:>8d}")
    print(f"\n    Final Resilience Index: {ablation.final_resilience_index:.4f}")
    print()

    # --- 3. Spatial Disaster ---
    disaster = simulate_disaster(G, epicenter=(50, 100), blast_radius_px=60)
    print("  -- Spatial Disaster Simulation --")
    print(f"    Epicenter       : (50, 100)")
    print(f"    Blast Radius    : 60 px")
    print(f"    Nodes Destroyed : {disaster.nodes_destroyed}")
    print(f"    Edges Destroyed : {disaster.edges_destroyed}")
    print(f"    Efficiency      : {disaster.eff_pre:.4f} -> {disaster.eff_post:.4f}")
    print(f"    Resilience Index: {disaster.resilience_index:.4f}")
    print(f"    Efficiency Drop : {disaster.pct_drop:.1f}%")
    print(f"    Components      : {disaster.n_components}")
    print()
    print("  [OK] Network analysis engine is operational.")
    print("=" * 70)
