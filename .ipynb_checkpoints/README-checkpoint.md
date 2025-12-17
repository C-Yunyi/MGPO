# Geometric-Aware Sampling for Diffusion Models

This repository implements **Geometric-Aware Sampling**, a diffusion sampling framework that explicitly aligns reverse diffusion trajectories with the **intrinsic geometry of real data manifolds**.

The method is built on top of **DiT / Minimax Diffusion**, and replaces standard greedy reverse diffusion with an **MST-guided DAG (beam search) sampling process** in latent space.

This is a **research prototype** intended for analysis, ablation, and method development.

---

## Motivation

In high-dimensional spaces, real data typically lie on a **low-dimensional manifold**.
However, standard diffusion sampling proceeds via **local denoising steps**, which often causes sampling trajectories to drift away from the true data manifold, harming dataset distillation quality.

To address this, we:
- Approximate the data manifold using a **Minimum Spanning Tree (MST)** built in VAE latent space.
- Treat each reverse diffusion step as an edge in a **time-layered DAG**.
- Select diffusion trajectories that stay close to the manifold by **path-level optimization**, rather than greedy step-wise sampling.

---

## Pipeline Overview

Real Images
↓
VAE Latent Encoding
↓
(Optional) PCA Projection
↓
kNN Graph Construction
↓
Per-Class MST (Manifold Skeleton)
↓
MST-Guided DAG / Beam Search Sampling
↓
Synthetic Images
↓
Downstream Training / Evaluation

## Getting Started

Download the repo:
```bash
git clone https://github.com/vimar-gu/MinimaxDiffusion.git
cd MinimaxDiffusion
```

Set up the environment:
```bash
conda create -n diff python=3.8
conda activate diff
pip install -r requirements.txt
```

Prepare the pre-trained DiT model:
```bash
python download.py
```


### Building MST-tree(Per-Class Manifold Skeleton)

For each class, we:

Encode real images into VAE latent space.

Optionally apply PCA for dimensionality reduction.

Build a kNN graph in latent space.

Extract a Minimum Spanning Tree (MST) to approximate the data manifold skeleton.

Command
```bash
python build_mst.py \
    --data-root /root/autodl-tmp/imagewoof2 \
    --save-root ./mst \
    --spec woof \
    --max-samples 200 \
    --pca-dim 128

```


### Geometric-Aware Sampling (MST-Guided DAG Sampling)
Instead of standard greedy reverse diffusion, we perform DAG-based beam search:

Each diffusion timestep corresponds to one DAG layer.

Nodes represent latent states.

Edges represent diffusion transitions.

Edge costs penalize deviation from the MST manifold.

The final sample is obtained by selecting the minimum-cost path from timestep T to 0.
```bash
python sample_mst.py \
    --model DiT-XL/2 \
    --image-size 256 \
    --ckpt pretrained_models/DiT-XL-2-256x256.pt \
    --save-dir ./results/mst_dag \
    --spec woof \
    --use-dag \
    --mst-root ./mst \
    --beam-width 16 \
    --dag-k 3 \
    --guide-start-t 25 \
    --lambda-d 1.0 \
    --lambda-dir 0.0

```

### Validation

```bash
python train.py -d imagenet --imagenet_dir ./results/mst_dag /root/autodl-tmp/imagewoof2/ \
    -n resnet_ap --nclass 10 --norm_type instance --ipc 100 --tag test --slct_type random --spec woof
```
