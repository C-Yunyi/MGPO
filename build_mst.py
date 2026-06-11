import os
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np

# ======================
# ARGPARSE
# ======================
def get_args():
    parser = argparse.ArgumentParser("Build per-class MST in VAE latent space")

    parser.add_argument("--data-root", type=str, default="/root/autodl-tmp/imagenette2/train",
                        help="Root directory of dataset (ImageFolder style)")
    parser.add_argument("--save-root", type=str, default="./mst",
                        help="Root directory to save MST results")
    parser.add_argument("--spec", type=str, default="default",
                        help="Dataset spec name (e.g. nette / woof / imagenet)")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Max samples per class")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eps", type=float, default=1e-6)

    return parser.parse_args()

# ======================
# MAIN
# ======================
def main():
    args = get_args()

    DATA_ROOT = args.data_root
    SAVE_ROOT = os.path.join(args.save_root)
    MAX_SAMPLES = args.max_samples
    BATCH_SIZE = args.batch_size
    DEVICE = args.device
    PCA_DIM = args.pca_dim
    KNN_K = args.knn_k
    EPS = args.eps

    os.makedirs(SAVE_ROOT, exist_ok=True)

    # ======================
    # LOAD DATA
    # ======================
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    dataset = datasets.ImageFolder(DATA_ROOT, transform=transform)
    class_names = dataset.classes

    print(f"Dataset loaded from {DATA_ROOT}")
    print(f"Classes: {len(class_names)}")

    # ======================
    # LOAD VAE
    # ======================
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse"
    ).to(DEVICE)
    vae.eval()

    # ======================
    # PER-CLASS BUILD
    # ======================
    for class_idx, class_name in enumerate(class_names):
        print(f"\n=== Building MST for class [{class_idx}] {class_name} ===")

        indices = [i for i, (_, y) in enumerate(dataset) if y == class_idx]
        if len(indices) == 0:
            print("  [skip] no samples")
            continue

        indices = indices[:MAX_SAMPLES]
        loader = DataLoader(
            torch.utils.data.Subset(dataset, indices),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # ======================
        # ENCODE TO LATENT
        # ======================
        latents = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(DEVICE)
                z = vae.encode(x).latent_dist.mean
                z = z / 0.18215
                latents.append(z.flatten(1))

        Z = torch.cat(latents, dim=0)  # [N, D]
        N, D = Z.shape
        print(f"  Encoded latents: {Z.shape}")

        # ======================
        # PCA + WHITEN
        # ======================
        Z_mean = Z.mean(dim=0, keepdim=True)
        Zc = Z - Z_mean

        U, S, V = torch.pca_lowrank(Zc, q=min(PCA_DIM, Zc.shape[0]-1))
        S = torch.clamp(S, min=EPS)

        Z_pca = (Zc @ V) / S

        if torch.isnan(Z_pca).any():
            raise RuntimeError("NaN detected in PCA output")

        # ======================
        # kNN GRAPH
        # ======================
        Z_np = Z_pca.cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=min(KNN_K, N-1)).fit(Z_np)
        dist, idx = nbrs.kneighbors(Z_np)

        adj = np.full((N, N), np.inf)
        for i in range(N):
            for j, d in zip(idx[i], dist[i]):
                adj[i, j] = d
                adj[j, i] = d

        # ======================
        # MST
        # ======================
        mst = minimum_spanning_tree(adj).tocoo()
        edges = np.stack([mst.row, mst.col], axis=1)

        if edges.shape[0] != N - 1:
            print(f"  [warn] MST edges = {edges.shape[0]}, expected {N-1}")

        # ======================
        # SAVE
        # ======================
        out_dir = os.path.join(SAVE_ROOT, class_name)
        os.makedirs(out_dir, exist_ok=True)

        torch.save(Z_pca.cpu(), os.path.join(out_dir, "mst_nodes.pt"))
        torch.save(V.cpu(),     os.path.join(out_dir, "pca_proj.pt"))
        torch.save(S.cpu(),     os.path.join(out_dir, "pca_scale.pt"))
        torch.save(Z_mean.cpu(),os.path.join(out_dir, "pca_mean.pt"))
        torch.save(torch.tensor(edges, dtype=torch.long),
                   os.path.join(out_dir, "mst_edges.pt"))

        print(f"  Saved to {out_dir}")
        print(f"  Nodes: {Z_pca.shape}, Edges: {edges.shape}")

if __name__ == "__main__":
    main()

