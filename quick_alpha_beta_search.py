#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick approximate alpha/beta search with normalized objective.

- Uses close_avg and sep_avg (normalized)
- Uses fast close-only selection
- Runs in seconds for ImageNet-scale (C ~ 1000)
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple


# -------------------------------------------------
# Basic utils
# -------------------------------------------------

def cosine_dist(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Cosine distance: 1 - cosine similarity
    u, v: (..., D)
    """
    return 1.0 - torch.sum(u * v, dim=-1).clamp(-1, 1)


# -------------------------------------------------
# Normalized objective
# -------------------------------------------------

@torch.no_grad()
def eval_objective_avg(
    selected_idx: List[int],
    cand_centroids: List[torch.Tensor],
    real_centroids: List[torch.Tensor],
    eps: float = 1e-6,
) -> Tuple[float, float]:
    """
    Evaluate normalized objective terms.

    Returns:
        close_avg: average over classes
        sep_avg  : average over class pairs
    """
    C = len(cand_centroids)

    picked = []
    close_terms = []

    for i in range(C):
        c_i = cand_centroids[i][selected_idx[i]]
        r_i = real_centroids[i]
        d_real = cosine_dist(c_i, r_i)
        close_terms.append(-torch.log(d_real + eps))
        picked.append(c_i)

    picked = torch.stack(picked, dim=0)  # (C, D)

    # close: average over classes
    close_avg = torch.stack(close_terms).mean()

    # sep: average over pairs
    sims = (picked @ picked.t()).clamp(-1, 1)
    dmat = 1.0 - sims
    i_idx, j_idx = torch.triu_indices(C, C, offset=1)
    sep_avg = torch.log(dmat[i_idx, j_idx] + eps).mean()

    return float(close_avg.item()), float(sep_avg.item())


# -------------------------------------------------
# Fast initialization (close-only)
# -------------------------------------------------

@torch.no_grad()
def fast_init_selection(
    cand_centroids: List[torch.Tensor],
    real_centroids: List[torch.Tensor],
) -> List[int]:
    """
    For each class, pick the group closest to real centroid.

    Complexity: O(C * G)
    """
    selected = []
    for i in range(len(cand_centroids)):
        r = real_centroids[i].unsqueeze(0)  # (1, D)
        d = cosine_dist(cand_centroids[i], r)  # (G,)
        selected.append(int(torch.argmin(d)))
    return selected


# -------------------------------------------------
# Quick alpha / beta search
# -------------------------------------------------

def quick_search_alpha_beta(
    selected: List[int],
    cand_centroids: List[torch.Tensor],
    real_centroids: List[torch.Tensor],
):
    close_avg, sep_avg = eval_objective_avg(
        selected, cand_centroids, real_centroids
    )

    print(f"[Fixed selection]")
    print(f"  close_avg = {close_avg:.6f}")
    print(f"  sep_avg   = {sep_avg:.6f}")
    print()

    # Coarse but sufficient grid
    alphas = [0.2, 0.4, 0.6, 0.8, 1.0]
    betas  = [0.05, 0.1, 0.2, 0.4, 0.8]

    best = None
    best_score = -1e18

    print("[Searching alpha / beta]")
    for a in alphas:
        for b in betas:
            score = a * close_avg + b * sep_avg
            print(f"  alpha={a:.2f}, beta={b:.2f} -> score={score:.6f}")
            if score > best_score:
                best_score = score
                best = (a, b, score)

    return best, close_avg, sep_avg


# -------------------------------------------------
# Main entry
# -------------------------------------------------

def quick_param_estimation(
    cand_centroids: List[torch.Tensor],
    real_centroids: List[torch.Tensor],
):
    """
    Main function to estimate good (alpha, beta).
    """

    print("[Quick] initializing selection (close-only)...")
    selected = fast_init_selection(cand_centroids, real_centroids)

    print("[Quick] evaluating normalized objective...")
    best, close_avg, sep_avg = quick_search_alpha_beta(
        selected, cand_centroids, real_centroids
    )

    alpha, beta, score = best

    print("\n========== Approximate optimal parameters ==========")
    print(f"alpha     = {alpha}")
    print(f"beta      = {beta}")
    print(f"score     = {score:.6f}")
    print(f"close_avg = {close_avg:.6f}")
    print(f"sep_avg   = {sep_avg:.6f}")
    print("===================================================")

    return {
        "alpha": alpha,
        "beta": beta,
        "score": score,
        "close_avg": close_avg,
        "sep_avg": sep_avg,
    }


# -------------------------------------------------
# Example usage (pseudo)
# -------------------------------------------------

if __name__ == "__main__":
    """
    IMPORTANT:
    You must provide:
        cand_centroids: List[Tensor(G, D)]
        real_centroids: List[Tensor(D)]

    Typically, you would:
    - import this file
    - call quick_param_estimation(...) right after
      your original script finishes computing centroids
    """

    raise RuntimeError(
        "This script is meant to be imported and called after "
        "cand_centroids and real_centroids are computed."
    )
