# mst_guidance.py
# -*- coding: utf-8 -*-
"""
MST-guided sampling utilities for DiT / diffusion latent sampling.

What you get:
- MSTGuidance: edge_cost(z_t -> z_{t-1}) = λq * Lq + λd * Ldist + λdir * Ldir
- SamplingGraph: layered DAG storage + pruning
- topk_shortest_paths: DP for Top-K shortest paths on layered DAG (t -> t-1)

Assumptions:
- You build a SamplingGraph where each node belongs to an integer time t in [0..T]
- Directed edges always go from t -> t-1 (i.e., decreasing time)
- MST nodes are latent vectors (flattened) representing the class manifold skeleton
- MST edges are pairs of node indices into mst_nodes

This file is standalone (PyTorch only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# ----------------------------
# MST Guidance
# ----------------------------


def _flatten_latent(z: torch.Tensor) -> torch.Tensor:
    """
    z: [B, C, H, W] or [B, D]
    returns [B, D]
    """
    if z.dim() == 4:
        return z.flatten(1)
    if z.dim() == 2:
        return z
    raise ValueError(f"Unsupported latent shape: {tuple(z.shape)}")


def _chunked_cdist_min(
    x: torch.Tensor,
    y: torch.Tensor,
    chunk: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (min_dist, argmin) over y for each x row, using chunked cdist to control memory.

    x: [B, D]
    y: [N, D]
    min_dist: [B]
    argmin:   [B] (long)
    """
    device = x.device
    y = y.to(device=device, dtype=x.dtype)

    B = x.size(0)
    best_d = torch.full((B,), float("inf"), device=device, dtype=x.dtype)
    best_i = torch.zeros((B,), device=device, dtype=torch.long)

    # Process y in chunks
    for start in range(0, y.size(0), chunk):
        ys = y[start : start + chunk]  # [M, D]
        d = torch.cdist(x, ys)  # [B, M]
        dmin, imin = d.min(dim=1)  # [B], [B]
        better = dmin < best_d
        best_d[better] = dmin[better]
        best_i[better] = imin[better] + start

    return best_d, best_i


class MSTGuidance:
    """
    Computes MST-guided edge costs for sampling transitions z_t -> z_{t-1}.

    Components:
    - quality:      Lq(z) = quality_fn(z) (you decide the sign convention)
    - distance:     Ldist(z) = min_{v in MST_nodes} ||z - v||_2
    - direction:    Ldir(z_t, z_{t-1}) = 1 - cos( d_sample, d_data )
        where d_sample = z_{t-1} - z_t
              d_data   = direction of the nearest MST edge (approx via nearest edge midpoint)
    """

    def __init__(
        self,
        mst_nodes: torch.Tensor,
        mst_edges: torch.Tensor,
        lambda_q: float = 0.0,
        lambda_d: float = 1.0,
        lambda_dir: float = 0.0,
        quality_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        pca_proj = None,
        pca_scale = None,
        cdist_chunk: int = 4096,
        eps: float = 1e-8,
    ):
        """
        mst_nodes: [N, D] float tensor (latent vectors, flattened)
        mst_edges: [E, 2] long tensor (indices into mst_nodes)
        quality_fn: callable(z_[B,...]) -> [B] float tensor (lower is better if you treat as cost)
        """
        self.P = pca_proj
        self.S = pca_scale
        self.mst_nodes = mst_nodes
        if mst_nodes.dim() != 2:
            raise ValueError(f"mst_nodes must be [N, D], got {tuple(mst_nodes.shape)}")
        if mst_edges.dim() != 2 or mst_edges.size(1) != 2:
            raise ValueError(f"mst_edges must be [E,2], got {tuple(mst_edges.shape)}")

        self.mst_nodes = mst_nodes.contiguous()
        self.mst_edges = mst_edges.to(dtype=torch.long).contiguous()

        self.lambda_q = float(lambda_q)
        self.lambda_d = float(lambda_d)
        self.lambda_dir = float(lambda_dir)

        self.quality_fn = quality_fn if quality_fn is not None else (lambda z: torch.zeros(z.size(0), device=z.device, dtype=z.dtype))
        self.cdist_chunk = int(cdist_chunk)
        self.eps = float(eps)

        # Precompute edge midpoints & directions (in node latent space)
        vi = self.mst_nodes[self.mst_edges[:, 0]]  # [E, D]
        vj = self.mst_nodes[self.mst_edges[:, 1]]  # [E, D]
        self.edge_mid = (vi + vj) * 0.5  # [E, D]
        d = vj - vi  # [E, D]
        self.edge_dir = d / (d.norm(dim=1, keepdim=True) + self.eps)  # [E, D]
        
    def _proj(self, z):
        zf = z.flatten(1)   # [B, D]
        if self.P is not None:
            P = self.P.to(device=zf.device, dtype=zf.dtype)
            zf = zf @ P    # [B, q]
            if self.S is not None:
                zf = zf / self.S.to(zf.device, zf.dtype)
        return zf

    @torch.no_grad()
    def mst_distance(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, C, H, W] or [B, D]
        returns: [B] Euclidean distance to nearest MST node
        """
        zf = self._proj(z)
        dmin, _ = _chunked_cdist_min(zf, self.mst_nodes, chunk=self.cdist_chunk)
        return dmin

    @torch.no_grad()
    def nearest_edge_dir(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, ...]
        returns: [B, D] unit direction vector of nearest MST edge (approx via nearest edge midpoint)
        """
        zf = self._proj(z)
        _, eidx = _chunked_cdist_min(zf, self.edge_mid, chunk=self.cdist_chunk)  # [B]
        # Gather edge_dir
        return self.edge_dir.to(device=zf.device, dtype=zf.dtype)[eidx]

    @torch.no_grad()
    def direction_loss(self, z_t: torch.Tensor, z_prev: torch.Tensor) -> torch.Tensor:
        """
        1 - cosine_similarity between sample step direction and nearest MST edge direction.
        returns [B]
        """
        zt = self._proj(z_t)
        zp = self._proj(z_prev)
        d_sample = zp - zt
        d_sample = d_sample / (d_sample.norm(dim=1, keepdim=True) + self.eps)

        d_data = self.nearest_edge_dir(z_t)  # [B, D], already normalized
        d_data = d_data.to(device=d_sample.device, dtype=d_sample.dtype)

        cos = (d_sample * d_data).sum(dim=1).clamp(-1.0, 1.0)
        return 1.0 - cos

    @torch.no_grad()
    def edge_cost(
        self,
        z_t: torch.Tensor,
        z_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns per-sample edge cost [B].
        This is your w_sample(t) for each edge (z_t -> z_{t-1}).
        """
        B = z_prev.size(0)
        cost = torch.zeros((B,), device=z_prev.device, dtype=z_prev.dtype)

        if self.lambda_q != 0.0:
            q = self.quality_fn(z_prev)  # [B]
            if q.dim() != 1 or q.size(0) != B:
                raise ValueError(f"quality_fn must return [B], got {tuple(q.shape)}")
            cost = cost + self.lambda_q * q

        if self.lambda_d != 0.0:
            d = self.mst_distance(z_prev)  # [B]
            cost = cost + self.lambda_d * d

        if self.lambda_dir != 0.0:
            dl = self.direction_loss(z_t, z_prev)  # [B]
            cost = cost + self.lambda_dir * dl

        return cost


# ----------------------------
# Sampling Graph (layered DAG)
# ----------------------------

@dataclass
class Node:
    node_id: int
    t: int
    z: torch.Tensor            # latent tensor (any shape)
    cost: float                # cumulative cost from start (larger t) down to this node
    # DP results (for top-k paths from this node down to time 0):
    kbest_costs: Optional[torch.Tensor] = None  # [K]
    kbest_choice: Optional[List[Tuple[int, int]]] = None  # [(child_id, child_rank)] length K


class SamplingGraph:
    """
    Layered DAG with nodes grouped by time t, edges from t -> t-1.
    Stores adjacency lists and supports cost-based pruning per time layer.
    """

    def __init__(self):
        self._next_id = 0
        self.nodes: Dict[int, Node] = {}                   # node_id -> Node
        self.layer: Dict[int, List[int]] = {}              # t -> [node_id...]
        self.children: Dict[int, List[Tuple[int, float]]] = {}  # parent_id -> [(child_id, edge_cost)]
        self.parents: Dict[int, int] = {}                  # child_id -> parent_id (optional, if you want)
        self.edge_cost_to_parent: Dict[int, float] = {}    # child_id -> edge_cost
        


    def add_node(
        self,
        t: int,
        z: torch.Tensor,
        cost: float,
        parent_id: Optional[int] = None,
        edge_cost: float = 0.0,
        detach_z: bool = True,
    ) -> int:
        """
        Add a node at time t. Optionally connect parent (time t+1) -> this node (time t).
        """
        if detach_z:
            z = z.detach()
        nid = self._next_id
        self._next_id += 1

        self.nodes[nid] = Node(node_id=nid, t=int(t), z=z, cost=float(cost))
        self.layer.setdefault(int(t), []).append(nid)

        if parent_id is not None:
            self.children.setdefault(int(parent_id), []).append((nid, float(edge_cost)))
            self.parents[nid] = int(parent_id)
            self.edge_cost_to_parent[nid] = float(edge_cost)

        return nid

    def nodes_at_time(self, t: int) -> List[int]:
        return self.layer.get(int(t), [])

    def max_time(self) -> int:
        return max(self.layer.keys()) if self.layer else 0

    def min_time(self) -> int:
        return min(self.layer.keys()) if self.layer else 0

    def add_nodes_pruned(
        self,
        t: int,
        candidates: Sequence[Tuple[torch.Tensor, int, float]],
        # each candidate: (z, parent_id, edge_cost) and node cost must be computed as parent.cost + edge_cost externally OR we compute here
        max_nodes: int,
        detach_z: bool = True,
    ) -> List[int]:
        """
        Add many candidates at time t, then prune the layer to keep only max_nodes by cumulative cost.

        candidates: list of (z, parent_id, edge_cost_total_increment)
                    cumulative_cost = nodes[parent_id].cost + edge_cost_total_increment

        Returns the kept node_ids (after pruning).
        """
        new_ids: List[int] = []
        for z, parent_id, edge_cost in candidates:
            parent_cost = self.nodes[int(parent_id)].cost
            nid = self.add_node(
                t=int(t),
                z=z,
                cost=parent_cost + float(edge_cost),
                parent_id=int(parent_id),
                edge_cost=float(edge_cost),
                detach_z=detach_z,
            )
            new_ids.append(nid)

        # prune this layer
        kept = self.prune_time(int(t), max_nodes=max_nodes)
        return kept

    def prune_time(self, t: int, max_nodes: int) -> List[int]:
        """
        Keep the max_nodes nodes with smallest cumulative cost in layer t.
        Removes pruned nodes and any outgoing edges from them.
        """
        t = int(t)
        ids = self.layer.get(t, [])
        if len(ids) <= max_nodes:
            return ids

        # sort by cumulative cost
        ids_sorted = sorted(ids, key=lambda nid: self.nodes[nid].cost)
        kept = set(ids_sorted[:max_nodes])
        removed = [nid for nid in ids if nid not in kept]

        # update layer
        self.layer[t] = list(kept)

        # remove nodes + their edges
        for nid in removed:
            # remove outgoing edges
            if nid in self.children:
                del self.children[nid]
            # remove node itself
            del self.nodes[nid]

        # also clean children lists of remaining nodes to drop edges pointing to removed nodes
        removed_set = set(removed)
        for pid, chlist in list(self.children.items()):
            new_ch = [(cid, w) for (cid, w) in chlist if cid not in removed_set]
            self.children[pid] = new_ch

        # NOTE: we don't fully clean parents/edge_cost_to_parent maps for removed nodes
        # because we don't need them for DP; but we can clean to be safe:
        for nid in removed:
            self.parents.pop(nid, None)
            self.edge_cost_to_parent.pop(nid, None)

        return self.layer[t]


# ----------------------------
# Top-K shortest paths on DAG
# ----------------------------

def _min_k_from_candidates(
    candidates: List[Tuple[float, int, int]],
    K: int,
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """
    candidates: list of (cost, child_id, child_rank)
    Returns:
      kbest_costs: [K] (float tensor)
      kbest_choice: [(child_id, child_rank)] length K
    """
    if not candidates:
        # unreachable
        inf = float("inf")
        return torch.full((K,), inf), [(-1, -1)] * K

    candidates.sort(key=lambda x: x[0])
    picked = candidates[:K]
    costs = torch.tensor([c for (c, _, _) in picked], dtype=torch.float32)
    choice = [(cid, crank) for (_, cid, crank) in picked]

    # pad if fewer than K
    if len(picked) < K:
        pad_n = K - len(picked)
        costs = torch.cat([costs, torch.full((pad_n,), float("inf"))], dim=0)
        choice.extend([(-1, -1)] * pad_n)

    return costs, choice


@torch.no_grad()
def topk_shortest_paths(
    graph: SamplingGraph,
    start_time: int,
    end_time: int,
    K: int,
) -> List[List[Node]]:
    """
    Compute Top-K shortest paths from ANY node in layer start_time down to layer end_time (usually 0),
    using dynamic programming on layered DAG.

    Returns a list of K paths, each path is a list of Node objects in time-descending order:
      [node_at_start_time, ..., node_at_end_time]

    Notes:
    - This is the DP version matching your described recurrence:
        Kbest(u) = minK_{v in Child(u)} { w(u,v) + c  for c in Kbest(v) }
    - Base case: nodes at end_time have Kbest_costs = [0, inf, inf, ...]
    """
    start_time = int(start_time)
    end_time = int(end_time)
    assert start_time >= end_time, "Expected start_time >= end_time"
    assert K >= 1

    # initialize base layer
    end_ids = graph.nodes_at_time(end_time)
    if not end_ids:
        return []

    for nid in end_ids:
        node = graph.nodes[nid]
        node.kbest_costs = torch.full((K,), float("inf"))
        node.kbest_costs[0] = 0.0
        node.kbest_choice = [(-1, -1)] * K  # no child

    # DP upward in time (end_time+1 -> start_time)
    for t in range(end_time + 1, start_time + 1):
        ids = graph.nodes_at_time(t)
        if not ids:
            continue

        for nid in ids:
            # gather candidates from children
            ch = graph.children.get(nid, [])
            cand: List[Tuple[float, int, int]] = []
            for (cid, w) in ch:
                if cid not in graph.nodes:
                    continue
                child = graph.nodes[cid]
                if child.kbest_costs is None:
                    continue
                # combine child k costs
                for r in range(K):
                    c = float(child.kbest_costs[r].item())
                    if c == float("inf"):
                        continue
                    cand.append((w + c, cid, r))

            costs, choice = _min_k_from_candidates(cand, K)
            node = graph.nodes[nid]
            node.kbest_costs = costs
            node.kbest_choice = choice

    # Choose K best paths among all start nodes
    start_ids = graph.nodes_at_time(start_time)
    if not start_ids:
        return []

    start_cand: List[Tuple[float, int, int]] = []
    for sid in start_ids:
        node = graph.nodes[sid]
        if node.kbest_costs is None:
            continue
        for r in range(K):
            c = float(node.kbest_costs[r].item())
            if c == float("inf"):
                continue
            start_cand.append((c, sid, r))

    if not start_cand:
        return []

    start_cand.sort(key=lambda x: x[0])
    picked = start_cand[:K]

    # Reconstruct each path by following (child_id, child_rank) pointers
    paths: List[List[Node]] = []
    for _, sid, srank in picked:
        path_nodes: List[Node] = []
        cur_id = sid
        cur_rank = srank

        while True:
            cur_node = graph.nodes[cur_id]
            path_nodes.append(cur_node)

            if cur_node.t == end_time:
                break

            if cur_node.kbest_choice is None:
                break

            child_id, child_rank = cur_node.kbest_choice[cur_rank]
            if child_id == -1:
                break

            cur_id = child_id
            cur_rank = child_rank

        paths.append(path_nodes)

    return paths
