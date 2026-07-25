# MGPO: Manifold-Guided Policy Optimization for Dataset Distillation

MGPO fine-tunes a pretrained diffusion model (DiT-XL/2) with reinforcement learning so that
its *reverse diffusion policy* internalizes two complementary objectives, instead of relying
on expensive inference-time guidance:

- **Discriminative reward** (pixel space): a proxy classifier score, pushing generated samples
  toward separable, class-discriminative regions.
- **Geometric reward** (latent space): distance/direction to a per-class **Minimum Spanning Tree
  (MST)** skeleton built from real data in VAE latent space, anchoring generation to the
  manifold's intrinsic structure and preventing mode collapse.

Both rewards are combined via a GRPO-style (group relative policy optimization) objective.
Because the alignment is baked into the model weights during training, **sampling afterwards is
a single standard DDIM pass — no beam search or other inference-time guidance is required.**

This repo also ships two *training-free* MST-guided DAG/beam-search sampling scripts
(`sample_mst.py`, `sample_mst_fast.py`) that apply the same MST skeleton at inference time
instead of / on top of a trained checkpoint. These are useful as ablations to check whether
additional inference-time guidance helps beyond what training-time alignment already provides,
but are **not** required for the main pipeline and are noticeably slower.

---

## 1. Environment

```bash
conda create -n mgpo python=3.10 -y
conda activate mgpo

# pick the torch/torchvision cu1xx wheels matching your driver, e.g.:
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

`requirements.txt` covers `diffusers`, `transformers`, `timm`, `scipy`, `scikit-learn`,
`efficientnet_pytorch`, `tqdm`, `matplotlib`, `tensorboard`, `ftfy`, `Jinja2` — the actual
imports used by the training/sampling/MST-building scripts.

> **Container/shared-cluster note:** if you hit
> `RuntimeError: DataLoader worker ... killed by signal: Bus error ... shared memory`,
> your container's `/dev/shm` is too small for PyTorch's multi-worker `DataLoader`. On
> Kubernetes-based clusters (e.g. RunAI) this is usually a one-flag fix at submission time
> (e.g. `--large-shm`); reducing `--workers` may not help if a codepath silently overrides it.

## 2. Data

Any ImageFolder-style dataset works (`<root>/train/<class>/*.jpg`, `<root>/val/<class>/*.jpg`).
For the ImageNet subsets used in the paper (ImageWoof / ImageNette / ImageIDC), the class lists
are in `misc/class_woof.txt`, `misc/class_nette.txt`, `misc/class_idc.txt` — if you already have
full ImageNet-1K on disk, you can build the subsets purely via symlinks instead of a separate
download:

```bash
SRC=/path/to/imagenet-1k          # expects SRC/train/<synset>, SRC/val/<synset>
DST=./data/imagewoof2

mkdir -p "$DST/train" "$DST/val"
while IFS= read -r c || [ -n "$c" ]; do
    ln -sfn "$SRC/train/$c" "$DST/train/$c"
    ln -sfn "$SRC/val/$c"   "$DST/val/$c"
done < misc/class_woof.txt
```

## 3. Pretrained assets

```bash
python download.py   # downloads pretrained_models/DiT-XL-2-{256x256,512x512}.pt
```

The VAE (`stabilityai/sd-vae-ft-mse`) is pulled automatically from Hugging Face the first time
any script constructs `AutoencoderKL.from_pretrained(...)`.

## 4. Build the per-class MST (manifold skeleton)

```bash
python build_mst.py \
    --data-root ./data/imagewoof2/train \
    --save-root ./mst_woof \
    --spec woof \
    --max-samples 200 \
    --pca-dim 128
```

Encodes real images into VAE latent space, optionally PCA-projects them, builds a sparse
k-NN graph, and extracts the per-class MST. Output: `mst_woof/<synset>/{mst_nodes,mst_edges,
pca_mean,pca_proj,pca_scale}.pt`.

## 5. Train MGPO (GRPO fine-tuning)

```bash
python train_multi_reward.py \
    --lr 1e-5 --kl-beta 0.1 \
    --lambda-mst-dist 1.0 --lambda-mst-dir 0.0 \
    --batch-size 4 --group-k 8 --num-sampling-steps 50 \
    --t-total 1000 --total-iters 600 --checkpoint-interval 550 \
    --spec woof --mst-root ./mst_woof \
    --reward-model resnet50 \
    --save-dir ./grpo_multi_samples_woof \
    --save-model-dir ./fine_tuned_model_woof
```

- `--reward-model` is an **off-the-shelf** ImageNet-1K-pretrained torchvision classifier
  (`resnet18/50/101/152`, or `convnext_base` / `efficientnet_b4` / `vit_b_16` / `swin_s`) — no
  separate classifier training is needed, since ImageWoof/Nette/IDC classes are literal
  subsets of the standard 1000 ImageNet classes.
- Checkpoints are saved as `{"model": state_dict}` under `--save-model-dir` every
  `--checkpoint-interval` steps (only once `step > 0`, so pick an interval that actually divides
  into `--total-iters`).

## 6. Sample from the fine-tuned model

**Standard sampling (recommended — matches the paper's reported pipeline):**

```bash
python sample.py \
    --model DiT-XL/2 --image-size 256 \
    --ckpt ./fine_tuned_model_woof/ckpt_step_550.pt \
    --save-dir ./results/imagewoof_ipc10 \
    --spec woof --nclass 10 --num-samples 10
```

**Optional: MST-guided DAG/beam-search sampling** (training-free, applies the same MST skeleton
again at inference time; useful as an ablation, not required):

```bash
python sample_mst_fast.py \
    --ckpt ./fine_tuned_model_woof/ckpt_step_550.pt \
    --spec woof --nclass 10 --num-samples 10 \
    --mst-root ./mst_woof \
    --save-dir ./results/imagewoof_ipc10_mst_fast \
    --beam-width 16 --dag-k 3 --guide-start-t 25 \
    --lambda-d 1.0 --lambda-dir 0.0
```

(`sample_mst.py` is the original, un-batched beam search — same guidance, ~4-5x slower than
`sample_mst_fast.py`, which batches the `dag-k` child-node forward passes.)

All three scripts write `<save-dir>/<synset>/<idx>.png`, ready to be consumed as an
ImageFolder-style dataset by the evaluation step below.

## 7. Downstream evaluation

Two supervision styles are commonly used to score a distilled dataset; which one a given paper
reports can vary by IPC/setting, so it's worth checking both.

**Hard-label** (train a student network from scratch on the synthetic set + real val split):

```bash
python train2.py -d imagenet \
    --imagenet_dir ./results/imagewoof_ipc10 ./data/imagewoof2/ \
    -n resnet_ap --nclass 10 --norm_type instance --ipc 10 \
    --tag hardlabel_test --slct_type random --spec woof --repeat 3
```

(`train.py` and `train2.py` are functionally identical; `train2.py` just adds tqdm progress
bars.)

**Soft-label** (knowledge distillation from a real-data teacher), via the
[CaO2](https://github.com/hatchetProject/CaO2) evaluation harness:

```bash
git clone https://github.com/hatchetProject/CaO2.git
cd CaO2
conda create -n cao2 python=3.10 -y && conda activate cao2
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
pip install numpy tqdm

# teacher/observer checkpoints (RDED-provided), place under CaO2/data/pretrain_models/
# https://drive.google.com/drive/folders/1HmrheO6MgX453a5UPJdxPHK4UTv-4aVt

python main_validate.py \
    --subset imagenet-woof --arch-name resnet18 --stud-name resnet18 \
    --factor 2 --num-crop 5 --mipc 300 --ipc 10 --re-epochs 300 --repeat 3 \
    --train-dir ../MGPO/data/imagewoof2/train --val-dir ../MGPO/data/imagewoof2/val \
    --syn-data-path ../MGPO/results/imagewoof_ipc10 -lr 0.1
```

> `--factor` controls CaO2/RDED's own patch-shuffle augmentation (`ShufflePatches`), which
> assumes each stored synthetic image packs `factor × factor` real sub-crops into one canvas.
> Single, independently generated diffusion samples (as produced by this repo) are **not**
> packed that way — but empirically `--factor 2` with `--num-crop 5` still trains a sensible
> student on them, matching the paper's reported numbers reasonably closely. If you evaluate a
> genuinely different distillation method, sanity-check this assumption first (visually inspect
> a few post-augmentation batches) rather than assuming it always transfers.

**RandAugment variant.** Both evaluation paths support real `torchvision.transforms.RandAugment`
(inserted before `ToTensor()`, since it operates on PIL/uint8 images):

```bash
# hard-label
python train2.py ... --randaug True --randaug_n 2 --randaug_m 8

# soft-label (CaO2)
python main_validate.py ... --randaug --randaug-n 2 --randaug-m 8
```

Note for CaO2: `argument_rded.py` silently overrides `--workers` based on `--ipc` (e.g. always
4 when `--ipc` is 10 or 50), so pass `--large-shm` (or otherwise enlarge `/dev/shm`) at job
submission time rather than trying to force `--workers 0`.

## Repository layout

- `argument.py`, `data.py`, `models.py`, `train_models/` — student-network training
  infrastructure (`train.py` / `train2.py`), shared across baselines and MGPO evaluation.
- `build_mst.py`, `fast_build_mst.py` — per-class MST construction (Algorithm 2 in the paper).
- `train_multi_reward*.py` — GRPO fine-tuning entry points (`_100`/`_1k` variants for
  ImageNet-100 / ImageNet-1K scale).
- `sample.py` — standard DDIM sampling from a fine-tuned checkpoint (recommended).
- `sample_mst*.py`, `mst_guidance*.py` — training-free MST-DAG beam-search sampling (ablation
  baseline, not required for the main pipeline).
- `misc/` — class-subset lists (`class_woof.txt`, `class_nette.txt`, `class_idc.txt`,
  `class_indices.txt`, ...) and small utilities.
