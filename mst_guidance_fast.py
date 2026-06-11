# mst_guidance_fast.py
# -*- coding: utf-8 -*-
"""
优化版 MST-guided sampling utilities.

主要优化:
1. PCA 投影矩阵预缓存到 GPU
2. 批量化 cdist 计算
3. 减少 CPU-GPU 数据传输
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# ----------------------------
# MST Guidance (优化版)
# ----------------------------

class MSTGuidanceFast:
    """
    优化版 MST 引导，支持批量计算 edge_cost.
    """

    def __init__(
        self,
        mst_nodes: torch.Tensor,
        mst_edges: torch.Tensor,
        lambda_q: float = 0.0,
        lambda_d: float = 1.0,
        lambda_dir: float = 0.0,
        quality_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        pca_proj: Optional[torch.Tensor] = None,
        pca_scale: Optional[torch.Tensor] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-8,
    ):
        """
        mst_nodes: [N, D] float tensor (latent vectors, flattened)
        mst_edges: [E, 2] long tensor (indices into mst_nodes)
        """
        self.device = device
        self.dtype = dtype
        self.eps = eps

        self.lambda_q = float(lambda_q)
        self.lambda_d = float(lambda_d)
        self.lambda_dir = float(lambda_dir)

        self.quality_fn = quality_fn if quality_fn is not None else (
            lambda z: torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        )

        # 预处理并缓存到 GPU
        self.mst_nodes = mst_nodes.to(device=device, dtype=dtype).contiguous()
        self.mst_edges = mst_edges.to(device=device, dtype=torch.long).contiguous()

        # PCA 投影矩阵预缓存
        self.P = None
        self.S = None
        if pca_proj is not None:
            self.P = pca_proj.to(device=device, dtype=dtype).contiguous()
        if pca_scale is not None:
            self.S = pca_scale.to(device=device, dtype=dtype).contiguous()

        # 预计算 edge midpoints & directions
        vi = self.mst_nodes[self.mst_edges[:, 0]]  # [E, D]
        vj = self.mst_nodes[self.mst_edges[:, 1]]  # [E, D]
        self.edge_mid = ((vi + vj) * 0.5).contiguous()  # [E, D]
        d = vj - vi
        self.edge_dir = (d / (d.norm(dim=1, keepdim=True) + self.eps)).contiguous()  # [E, D]

    def _proj(self, z: torch.Tensor) -> torch.Tensor:
        """投影到 PCA 空间 (已优化，无重复数据传输)"""
        zf = z.flatten(1).to(dtype=self.dtype)  # [B, D]
        if self.P is not None:
            zf = zf @ self.P  # [B, q]
            if self.S is not None:
                zf = zf / self.S
        return zf

    @torch.no_grad()
    def mst_distance_batch(self, z: torch.Tensor) -> torch.Tensor:
        """
        批量计算到最近 MST 节点的距离.
        z: [B, C, H, W] or [B, D]
        returns: [B]
        """
        zf = self._proj(z)  # [B, D]
        # 使用 cdist，对于大 batch 更高效
        dist = torch.cdist(zf, self.mst_nodes)  # [B, N]
        return dist.min(dim=1).values  # [B]

    @torch.no_grad()
    def nearest_edge_dir_batch(self, z: torch.Tensor) -> torch.Tensor:
        """
        批量获取最近 MST edge 的方向向量.
        z: [B, ...]
        returns: [B, D]
        """
        zf = self._proj(z)  # [B, D]
        dist = torch.cdist(zf, self.edge_mid)  # [B, E]
        eidx = dist.argmin(dim=1)  # [B]
        return self.edge_dir[eidx]  # [B, D]

    @torch.no_grad()
    def direction_loss_batch(self, z_t: torch.Tensor, z_prev: torch.Tensor) -> torch.Tensor:
        """
        批量计算方向损失: 1 - cos(sample_dir, mst_edge_dir)
        returns [B]
        """
        zt = self._proj(z_t)   # [B, D]
        zp = self._proj(z_prev)  # [B, D]

        d_sample = zp - zt
        d_sample = d_sample / (d_sample.norm(dim=1, keepdim=True) + self.eps)

        d_data = self.nearest_edge_dir_batch(z_t)  # [B, D]

        cos = (d_sample * d_data).sum(dim=1).clamp(-1.0, 1.0)
        return 1.0 - cos

    @torch.no_grad()
    def edge_cost_batch(
        self,
        z_t: torch.Tensor,
        z_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        批量计算 edge cost.
        z_t, z_prev: [B, C, H, W] or [B, D]
        returns: [B]
        """
        B = z_prev.size(0)
        cost = torch.zeros(B, device=self.device, dtype=self.dtype)

        if self.lambda_q != 0.0:
            q = self.quality_fn(z_prev)
            cost = cost + self.lambda_q * q

        if self.lambda_d != 0.0:
            d = self.mst_distance_batch(z_prev)
            cost = cost + self.lambda_d * d

        if self.lambda_dir != 0.0:
            dl = self.direction_loss_batch(z_t, z_prev)
            cost = cost + self.lambda_dir * dl

        return cost


# ----------------------------
# Sampling Graph (优化版)
# ----------------------------

@dataclass
class NodeFast:
    node_id: int
    t: int
    z: torch.Tensor
    cost: float
    parent_id: Optional[int] = None


class SamplingGraphFast:
    """
    优化版 Sampling Graph，减少 Python 开销.
    """

    def __init__(self):
        self._next_id = 0
        self.nodes: Dict[int, NodeFast] = {}
        self.layer: Dict[int, List[int]] = {}

    def add_node(
        self,
        t: int,
        z: torch.Tensor,
        cost: float,
        parent_id: Optional[int] = None,
    ) -> int:
        nid = self._next_id
        self._next_id += 1

        self.nodes[nid] = NodeFast(
            node_id=nid,
            t=int(t),
            z=z.detach(),
            cost=float(cost),
            parent_id=parent_id
        )
        self.layer.setdefault(int(t), []).append(nid)
        return nid

    def nodes_at_time(self, t: int) -> List[int]:
        return self.layer.get(int(t), [])

    def add_nodes_and_prune(
        self,
        t: int,
        z_batch: torch.Tensor,  # [N, C, H, W]
        costs: torch.Tensor,    # [N]
        parent_ids: List[int],  # [N]
        max_nodes: int,
    ) -> List[int]:
        """
        批量添加节点并剪枝，保留 cost 最小的 max_nodes 个.
        """
        N = z_batch.size(0)

        # 获取 parent costs
        parent_costs = torch.tensor(
            [self.nodes[pid].cost for pid in parent_ids],
            device=costs.device, dtype=costs.dtype
        )
        total_costs = parent_costs + costs  # [N]

        # 选择 top-k (最小 cost)
        if N > max_nodes:
            _, keep_idx = torch.topk(total_costs, max_nodes, largest=False)
            keep_idx = keep_idx.cpu().tolist()
        else:
            keep_idx = list(range(N))

        # 添加保留的节点
        new_ids = []
        for idx in keep_idx:
            nid = self.add_node(
                t=t,
                z=z_batch[idx],
                cost=total_costs[idx].item(),
                parent_id=parent_ids[idx]
            )
            new_ids.append(nid)

        return new_ids

    def get_best_path(self, end_time: int = 0) -> List[NodeFast]:
        """
        获取到达 end_time 层 cost 最小的路径.
        """
        end_nodes = self.nodes_at_time(end_time)
        if not end_nodes:
            return []

        # 找 cost 最小的终点
        best_nid = min(end_nodes, key=lambda nid: self.nodes[nid].cost)

        # 回溯路径
        path = []
        cur_nid = best_nid
        while cur_nid is not None:
            node = self.nodes[cur_nid]
            path.append(node)
            cur_nid = node.parent_id

        return path[::-1]  # 从起点到终点
