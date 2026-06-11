"""
build_mst_fast.py
=================
Optimized version of the per-class MST builder.

Key improvements over original:
  1. Pre-build class→indices mapping  O(N_total) instead of O(C × N_total)
  2. Sparse adjacency matrix          O(N·K) instead of O(N²) for MST
  3. float16 VAE inference            ~1.5-2× faster encoding, half VRAM
  4. torch.compile (PyTorch ≥ 2.0)   ~1.3× faster VAE forward
  5. Multi-process parallelism        one worker per GPU / CPU core
  6. channels_last memory format      faster conv kernels on NVIDIA
  7. tqdm progress bars with timing   wall-clock visibility at every level
"""

import os
import time
import argparse
import warnings
from collections import defaultdict
from typing import Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from diffusers.models import AutoencoderKL
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


# ──────────────────────────────────────────────
# ARGPARSE
# ──────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description="Build per-class MST in VAE latent space (fast version)"
    )
    parser.add_argument("--data-root",   type=str,   default="/root/autodl-tmp/imagenet/train")
    parser.add_argument("--save-root",   type=str,   default="./mst")
    parser.add_argument("--spec",        type=str,   default="imagenet1k")
    parser.add_argument("--max-samples", type=int,   default=200)
    parser.add_argument("--batch-size",  type=int,   default=64,
                        help="VAE encode batch size (larger = faster, needs more VRAM)")
    parser.add_argument("--pca-dim",     type=int,   default=128)
    parser.add_argument("--knn-k",       type=int,   default=10)
    parser.add_argument("--num-workers", type=int,   default=1,
                        help="Number of parallel class-worker processes. "
                             "Set to number of available GPUs, or 1 for single-GPU.")
    parser.add_argument("--device",      type=str,   default="cuda")
    parser.add_argument("--eps",         type=float, default=1e-6)
    parser.add_argument("--fp16",        action="store_true", default=True,
                        help="Use float16 for VAE inference (default: True)")
    parser.add_argument("--vae-path",    type=str,   default="stabilityai/sd-vae-ft-mse",
                        help="Local path or HF model ID for the VAE. "
                             "Use a local snapshot dir to avoid network access. "
                             "Find it with: find ~/.cache/huggingface -name config.json | grep sd-vae")
    parser.add_argument("--compile",     action="store_true", default=False,
                        help="torch.compile the VAE (PyTorch >= 2.0, adds ~30s warm-up)")
    parser.add_argument("--pca-niter",   type=int,   default=2,
                        help="Randomized SVD iterations for pca_lowrank (2=default, 1=faster)")
    return parser.parse_args()


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def fmt_time(seconds: float) -> str:
    """Human-readable elapsed time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def load_vae(device: str, fp16: bool, compile_model: bool, vae_path: str = "stabilityai/sd-vae-ft-mse") -> AutoencoderKL:
    dtype = torch.float16 if fp16 else torch.float32
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        torch_dtype=dtype,
    ).to(device)
    vae = vae.to(memory_format=torch.channels_last)
    vae.eval()
    if compile_model:
        try:
            vae = torch.compile(vae, mode="reduce-overhead")
            print("  [info] torch.compile applied to VAE")
        except Exception as e:
            print(f"  [warn] torch.compile failed: {e}")
    return vae


def build_class_index(dataset) -> dict:
    """
    Build class_idx → [sample_indices] mapping.
    Uses dataset.targets directly (no image decoding) — O(N_total), near-instant.
    Falls back to enumerate loop only if .targets is unavailable.
    """
    mapping = defaultdict(list)
    if hasattr(dataset, "targets"):
        # ImageFolder always exposes .targets: List[int], zero decoding cost
        for i, y in enumerate(dataset.targets):
            mapping[y].append(i)
    else:
        for i, (_, y) in enumerate(
            tqdm(dataset, desc="  Building class index", leave=False, unit="img")
        ):
            mapping[y].append(i)
    return mapping


def encode_latents(
    vae: AutoencoderKL,
    subset: Subset,
    batch_size: int,
    device: str,
    fp16: bool,
    desc: str = "  Encoding",
) -> torch.Tensor:
    """Encode a Subset to VAE latents, returns [N, D] float32 tensor."""
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    latents = []
    ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if fp16 else torch.no_grad()
    with torch.no_grad(), ctx:
        for x, _ in tqdm(loader, desc=desc, leave=False, unit="batch"):
            x = x.to(device, memory_format=torch.channels_last)
            z = vae.encode(x).latent_dist.mean
            z = z / 0.18215
            latents.append(z.flatten(1).float())   # store in fp32
    return torch.cat(latents, dim=0)


def pca_whiten(
    Z: torch.Tensor,
    pca_dim: int,
    eps: float,
    niter: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PCA + whitening. Returns (Z_pca, V, S, Z_mean)."""
    Z_mean = Z.mean(dim=0, keepdim=True)
    Zc = Z - Z_mean
    q = min(pca_dim, Zc.shape[0] - 1)
    _, S, V = torch.pca_lowrank(Zc, q=q, niter=niter)
    S = torch.clamp(S, min=eps)
    Z_pca = (Zc @ V) / S
    if torch.isnan(Z_pca).any():
        raise RuntimeError("NaN detected in PCA output — check your data or increase --eps")
    return Z_pca, V, S, Z_mean


def build_sparse_knn(Z_np: np.ndarray, k: int) -> csr_matrix:
    """
    Build a symmetric sparse adjacency matrix from kNN distances.
    O(N·K) memory instead of O(N²).
    """
    N = Z_np.shape[0]
    k_eff = min(k, N - 1)
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", n_jobs=-1).fit(Z_np)
    dist, idx = nbrs.kneighbors(Z_np)

    rows, cols, vals = [], [], []
    for i in range(N):
        for j, d in zip(idx[i], dist[i]):
            rows.append(i);  cols.append(j);  vals.append(d)
            rows.append(j);  cols.append(i);  vals.append(d)

    return csr_matrix((vals, (rows, cols)), shape=(N, N))


# ──────────────────────────────────────────────
# PER-CLASS WORKER
# ──────────────────────────────────────────────

def process_class(
    class_idx: int,
    class_name: str,
    indices: list,
    dataset,
    save_root: str,
    args,
    vae: AutoencoderKL,
    pbar_position: int = 0,
) -> dict:
    """
    Full pipeline for one class. Returns a timing dict for reporting.
    """
    timings = {}
    out_dir = os.path.join(save_root, class_name)

    # Skip if already done
    if all(
        os.path.exists(os.path.join(out_dir, f))
        for f in ("mst_nodes.pt", "mst_edges.pt", "pca_proj.pt")
    ):
        return {"class": class_name, "skipped": True}

    os.makedirs(out_dir, exist_ok=True)
    indices = indices[: args.max_samples]
    N = len(indices)
    if N == 0:
        return {"class": class_name, "skipped": True, "reason": "no samples"}

    subset = Subset(dataset, indices)

    # ── 1. VAE encode ──────────────────────────
    t0 = time.perf_counter()
    Z = encode_latents(
        vae, subset, args.batch_size, args.device, args.fp16,
        desc=f"  [{class_name}] encode"
    )
    timings["encode"] = time.perf_counter() - t0

    # ── 2. PCA + whiten ────────────────────────
    t0 = time.perf_counter()
    Z_pca, V, S, Z_mean = pca_whiten(Z, args.pca_dim, args.eps, args.pca_niter)
    timings["pca"] = time.perf_counter() - t0

    # ── 3. kNN sparse graph ────────────────────
    t0 = time.perf_counter()
    Z_np = Z_pca.cpu().numpy()
    adj_sparse = build_sparse_knn(Z_np, args.knn_k)
    timings["knn"] = time.perf_counter() - t0

    # ── 4. MST ─────────────────────────────────
    t0 = time.perf_counter()
    mst = minimum_spanning_tree(adj_sparse).tocoo()
    edges = np.stack([mst.row, mst.col], axis=1)
    timings["mst"] = time.perf_counter() - t0

    if edges.shape[0] != N - 1:
        warnings.warn(
            f"[{class_name}] MST edges={edges.shape[0]}, expected {N-1}. "
            "kNN graph may be disconnected — consider increasing --knn-k."
        )

    # ── 5. Save ────────────────────────────────
    t0 = time.perf_counter()
    torch.save(Z_pca.cpu(),                                    os.path.join(out_dir, "mst_nodes.pt"))
    torch.save(V.cpu(),                                        os.path.join(out_dir, "pca_proj.pt"))
    torch.save(S.cpu(),                                        os.path.join(out_dir, "pca_scale.pt"))
    torch.save(Z_mean.cpu(),                                   os.path.join(out_dir, "pca_mean.pt"))
    torch.save(torch.tensor(edges, dtype=torch.long),          os.path.join(out_dir, "mst_edges.pt"))
    timings["save"] = time.perf_counter() - t0

    timings["class"] = class_name
    timings["N"] = N
    timings["skipped"] = False
    return timings


# ──────────────────────────────────────────────
# MULTI-PROCESS ENTRY (one process per GPU)
# ──────────────────────────────────────────────

def worker_main(
    rank: int,
    class_chunks: list,       # list of (class_idx, class_name, indices)
    dataset,
    save_root: str,
    args,
):
    """
    Each worker process handles its assigned chunk of classes.
    Loads its own VAE on its own GPU.
    """
    device = f"cuda:{rank}" if torch.cuda.device_count() > 1 else args.device
    args_copy = argparse.Namespace(**vars(args))
    args_copy.device = device

    print(f"\n[Worker {rank}] Starting on {device}, "
          f"handling {len(class_chunks)} classes")

    vae = load_vae(device, args.fp16, args.compile, args.vae_path)

    pbar = tqdm(
        class_chunks,
        desc=f"Worker {rank}",
        position=rank,
        leave=True,
        unit="class",
        dynamic_ncols=True,
    )

    total_times = defaultdict(float)
    n_done = 0

    for class_idx, class_name, indices in pbar:
        t_class = time.perf_counter()
        result = process_class(
            class_idx, class_name, indices, dataset,
            save_root, args_copy, vae, pbar_position=rank + 1
        )
        elapsed = time.perf_counter() - t_class

        if result.get("skipped"):
            pbar.set_postfix_str(f"{class_name} SKIPPED")
            continue

        n_done += 1
        for k in ("encode", "pca", "knn", "mst", "save"):
            total_times[k] += result.get(k, 0)

        pbar.set_postfix({
            "cls":    class_name,
            "N":      result["N"],
            "enc":    fmt_time(result["encode"]),
            "pca":    fmt_time(result["pca"]),
            "knn":    fmt_time(result["knn"]),
            "mst":    fmt_time(result["mst"]),
            "total":  fmt_time(elapsed),
        })

    # Per-worker summary
    print(f"\n[Worker {rank}] Done {n_done} classes. "
          f"Avg time breakdown →  "
          f"encode:{fmt_time(total_times['encode']/max(n_done,1))}  "
          f"pca:{fmt_time(total_times['pca']/max(n_done,1))}  "
          f"knn:{fmt_time(total_times['knn']/max(n_done,1))}  "
          f"mst:{fmt_time(total_times['mst']/max(n_done,1))}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    args = get_args()
    os.makedirs(args.save_root, exist_ok=True)

    print("=" * 60)
    print("  MST Builder  (fast edition)")
    print("=" * 60)
    print(f"  data-root   : {args.data_root}")
    print(f"  save-root   : {args.save_root}")
    print(f"  max-samples : {args.max_samples}")
    print(f"  batch-size  : {args.batch_size}")
    print(f"  pca-dim     : {args.pca_dim}")
    print(f"  knn-k       : {args.knn_k}")
    print(f"  num-workers : {args.num_workers}")
    print(f"  vae-path    : {args.vae_path}")
    print(f"  compile     : {args.compile}")
    print(f"  pca-niter   : {args.pca_niter}")
    print("=" * 60)

    # ── Load dataset (no decoding yet, just index) ──
    t0 = time.perf_counter()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    dataset = datasets.ImageFolder(args.data_root, transform=transform)
    class_names = dataset.classes
    print(f"\n[Setup] Dataset loaded: {len(dataset)} images, "
          f"{len(class_names)} classes  ({fmt_time(time.perf_counter()-t0)})")

    # ── Build index O(N_total) ──────────────────
    t0 = time.perf_counter()
    print("[Setup] Building class index...")
    class_to_indices = build_class_index(dataset)
    print(f"[Setup] Index ready  ({fmt_time(time.perf_counter()-t0)})")

    # ── Prepare work list ───────────────────────
    all_classes = [
        (idx, name, class_to_indices[idx])
        for idx, name in enumerate(class_names)
        if len(class_to_indices[idx]) > 0
    ]
    print(f"[Setup] {len(all_classes)} non-empty classes to process\n")

    # ── Dispatch ────────────────────────────────
    t_global = time.perf_counter()

    num_workers = min(args.num_workers, torch.cuda.device_count() or 1)

    if num_workers <= 1:
        # ── Single process path ──────────────────
        vae = load_vae(args.device, args.fp16, args.compile, args.vae_path)
        outer_bar = tqdm(
            all_classes,
            desc="Classes",
            unit="class",
            dynamic_ncols=True,
            colour="green",
        )
        timing_log = []
        for class_idx, class_name, indices in outer_bar:
            t_cls = time.perf_counter()
            result = process_class(
                class_idx, class_name, indices, dataset,
                args.save_root, args, vae
            )
            elapsed = time.perf_counter() - t_cls

            if result.get("skipped"):
                outer_bar.set_postfix_str(f"{class_name} SKIPPED")
                continue

            timing_log.append(result)
            outer_bar.set_postfix({
                "cls":   class_name,
                "N":     result["N"],
                "enc":   fmt_time(result["encode"]),
                "pca":   fmt_time(result["pca"]),
                "knn":   fmt_time(result["knn"]),
                "mst":   fmt_time(result["mst"]),
                "cls_t": fmt_time(elapsed),
            })

    else:
        # ── Multi-process path (one process per GPU) ──
        # Split classes evenly across workers
        chunks = [[] for _ in range(num_workers)]
        for i, item in enumerate(all_classes):
            chunks[i % num_workers].append(item)

        mp.set_start_method("spawn", force=True)
        processes = []
        for rank in range(num_workers):
            p = mp.Process(
                target=worker_main,
                args=(rank, chunks[rank], dataset, args.save_root, args),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        timing_log = []   # detailed logs not aggregated in multi-process mode

    # ── Final summary ────────────────────────────
    total_elapsed = time.perf_counter() - t_global
    n_done = len(timing_log)

    print("\n" + "=" * 60)
    print(f"  Finished {n_done} classes in {fmt_time(total_elapsed)}")
    if n_done > 0:
        avg = lambda k: sum(r[k] for r in timing_log) / n_done
        print(f"  Avg per class:")
        print(f"    encode : {fmt_time(avg('encode'))}")
        print(f"    pca    : {fmt_time(avg('pca'))}")
        print(f"    knn    : {fmt_time(avg('knn'))}")
        print(f"    mst    : {fmt_time(avg('mst'))}")
        print(f"    save   : {fmt_time(avg('save'))}")
        print(f"  Throughput : {n_done/total_elapsed:.2f} classes/s  |  "
              f"~{total_elapsed/max(n_done,1):.2f}s per class")
    print("=" * 60)


if __name__ == "__main__":
    main()