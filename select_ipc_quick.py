import os
import random
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms, models
from PIL import Image
import csv
import heapq


# ----------------------------
# Utils
# ----------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def list_images(root: Path, exts=(".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")) -> List[Path]:
    imgs = []
    for e in exts:
        imgs += list(root.rglob(f"*{e}"))
    return sorted(imgs)

def read_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        return [x.strip() for x in f.readlines() if x.strip()]

def _normalize_exts(exts: List[str]) -> Tuple[str, ...]:
    tokens: List[str] = []
    for e in exts:
        e = e.strip()
        if not e:
            continue
        if "," in e:
            tokens.extend([x.strip() for x in e.split(",") if x.strip()])
        else:
            tokens.append(e)
    uniq = set()
    for t in tokens:
        if not t.startswith("."):
            t = "." + t
        uniq.add(t)
        uniq.add(t.upper())
    return tuple(sorted(uniq))


# ----------------------------
# Feature extractor
# ----------------------------

class ResNet18Feats(torch.nn.Module):
    def __init__(self, device="cuda", img_size=224):
        super().__init__()
        self.device = device
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.backbone = models.resnet18(weights=weights)
        except Exception:
            self.backbone = models.resnet18(pretrained=True)

        self.backbone.fc = torch.nn.Identity()
        self.backbone = self.backbone.to(device).eval()

        self.tfm = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def encode(self, img_tensor_bchw: torch.Tensor) -> torch.Tensor:
        # input: [-1,1]
        x = (img_tensor_bchw + 1.0) * 0.5
        x = self.tfm(x)
        feats = self.backbone(x)
        return F.normalize(feats, dim=1)


def build_feature_extractor(name: str, device="cuda", img_size=224):
    # 你目前就用 resnet18 最稳
    return ResNet18Feats(device=device, img_size=img_size)


# ----------------------------
# Centroid & objective (normalized)
# ----------------------------

def compute_centroid(feats: torch.Tensor) -> torch.Tensor:
    c = feats.mean(dim=0)
    c = F.normalize(c, dim=0)
    return c

def cosine_dist(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.sum(u * v, dim=-1).clamp(-1, 1)

def eval_objective_terms_avg(
    selected_idx: List[int],
    cand_centroids: List[torch.Tensor],
    real_centroids: List[torch.Tensor],
    eps: float
) -> Tuple[float, float]:
    """
    返回 (close_avg, sep_avg)：
    close_avg = mean_i [-log(d_real+eps)]
    sep_avg   = mean_{i<j} [log(d_between+eps)]
    """
    C = len(cand_centroids)

    close_terms = []
    picked = []
    for i in range(C):
        c_i = cand_centroids[i][selected_idx[i]]
        r_i = real_centroids[i]
        d_real = cosine_dist(c_i, r_i)
        close_terms.append(-torch.log(d_real + eps))
        picked.append(c_i.unsqueeze(0))
    picked = torch.cat(picked, dim=0)  # (C,D)

    close_avg = torch.stack(close_terms).mean()

    sims = (picked @ picked.t()).clamp(-1, 1)
    dmat = 1.0 - sims
    i_idx, j_idx = torch.triu_indices(C, C, offset=1)
    sep_avg = torch.log(dmat[i_idx, j_idx] + eps).mean()

    return float(close_avg.item()), float(sep_avg.item())


@torch.no_grad()
def fast_init_selection_close_only(
    cand_centroids: List[torch.Tensor],
    real_centroids: List[torch.Tensor],
) -> List[int]:
    """
    每类选一个最接近真实中心的 group（O(C*G)，非常快）
    """
    C = len(cand_centroids)
    selected = []
    for i in range(C):
        r_i = real_centroids[i].unsqueeze(0)   # (1,D)
        d_all = cosine_dist(cand_centroids[i], r_i)  # (G,)
        gi0 = int(torch.argmin(d_all).item())
        selected.append(gi0)
    return selected


def quick_alpha_beta_search(
    cand_centroids: List[torch.Tensor],
    real_centroids: List[torch.Tensor],
    eps: float,
    alphas: List[float],
    betas: List[float],
) -> Tuple[float, float, float, float, float, List[int]]:
    """
    近似搜索最优 (alpha,beta)：
    - 固定 selection = close-only 的最优
    - 在这个 selection 上扫一个小网格得到最佳 alpha/beta

    返回: best_alpha, best_beta, best_score, close_avg, sep_avg, selected
    """
    selected = fast_init_selection_close_only(cand_centroids, real_centroids)
    close_avg, sep_avg = eval_objective_terms_avg(selected, cand_centroids, real_centroids, eps)

    best_alpha, best_beta, best_score = None, None, -1e18
    for a in alphas:
        for b in betas:
            score = a * close_avg + b * sep_avg
            if score > best_score:
                best_score = score
                best_alpha, best_beta = a, b

    return best_alpha, best_beta, best_score, close_avg, sep_avg, selected


# ----------------------------
# Load images -> features
# ----------------------------

@torch.no_grad()
def load_images_as_tensor(paths: List[Path], unify_hw=(256, 256)) -> torch.Tensor:
    to_tensor = transforms.ToTensor()
    Ht, Wt = unify_hw
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        t = to_tensor(img).unsqueeze(0)  # (1,3,H,W)
        t = F.interpolate(t, size=(Ht, Wt), mode="bilinear", align_corners=False).squeeze(0)
        t = t * 2.0 - 1.0
        imgs.append(t)
    return torch.stack(imgs, dim=0)

@torch.no_grad()
def compute_real_class_centroid(
    feat_net,
    class_dir: Path,
    device: str,
    max_imgs: int,
    exts: Tuple[str, ...],
    seed: int,
    unify_hw=(256, 256),
    encode_bs: int = 128,
) -> torch.Tensor:
    rng = random.Random(seed)
    paths = list_images(class_dir, exts)
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found in {class_dir}")
    if len(paths) > max_imgs:
        rng.shuffle(paths)
        paths = paths[:max_imgs]

    x = load_images_as_tensor(paths, unify_hw=unify_hw)
    feats_list = []
    for s in range(0, x.size(0), encode_bs):
        feats_list.append(feat_net.encode(x[s:s+encode_bs].to(device)))
    feats = torch.cat(feats_list, dim=0)
    return compute_centroid(feats)

@torch.no_grad()
def compute_candidate_group_centroids_for_class(
    feat_net,
    class_tmp_dir: Path,
    device: str,
    groups: int,
    ipc: int,
    unify_hw=(256, 256),
    encode_bs: int = 128,
) -> Tuple[torch.Tensor, List[List[Path]]]:
    centroids = []
    group_paths = []

    for g in range(groups):
        gdir = class_tmp_dir / f"group_{g:02d}"
        imgs = list_images(gdir, exts=(".png", ".jpg", ".jpeg", ".webp"))
        if len(imgs) < ipc:
            raise RuntimeError(f"{gdir} has only {len(imgs)} images, need {ipc}")
        imgs = imgs[:ipc]

        x = load_images_as_tensor(imgs, unify_hw=unify_hw)
        feats_list = []
        for s in range(0, x.size(0), encode_bs):
            feats_list.append(feat_net.encode(x[s:s+encode_bs].to(device)))
        feats = torch.cat(feats_list, dim=0)
        c = compute_centroid(feats)

        centroids.append(c.unsqueeze(0))
        group_paths.append(imgs)

    return torch.cat(centroids, dim=0), group_paths


# ----------------------------
# Main
# ----------------------------

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    tmp_root = Path(args.tmp_candidates)
    real_root = Path(args.real_train_dir)
    out_root = ensure_dir(Path(args.out_root))

    # class list (Imagenet 1k)
    all_classes = read_lines(args.class_index_file)
    sel_classes = read_lines(args.class_list_file)

    phase = max(0, args.phase)
    cls_from = args.nclass * phase
    cls_to = args.nclass * (phase + 1)
    sel_classes = sel_classes[cls_from:cls_to]
    class_labels = [all_classes.index(x) for x in sel_classes]
    assert len(sel_classes) == args.nclass

    feat_net = build_feature_extractor(args.feature_backbone, device=device, img_size=args.feat_img_size)
    real_exts = _normalize_exts(args.real_exts)

    # 1) real centroids
    print("[1] Compute real centroids ...")
    real_centroids: Dict[int, torch.Tensor] = {}
    for cname, cid in tqdm(list(zip(sel_classes, class_labels)), desc="RealCentroids"):
        cdir = real_root / cname
        real_centroids[cid] = compute_real_class_centroid(
            feat_net, cdir,
            device=device,
            max_imgs=args.real_max_per_class,
            exts=real_exts,
            seed=args.seed,
            unify_hw=(args.unify_h, args.unify_w),
            encode_bs=args.encode_bs,
        )

    # 2) candidate centroids
    print("[2] Compute candidate group centroids from tmp_candidates ...")
    cand_centroids: List[torch.Tensor] = []
    real_centroids_list: List[torch.Tensor] = []
    per_class_group_paths: List[List[List[Path]]] = []

    for cname, cid in tqdm(list(zip(sel_classes, class_labels)), desc="CandCentroids"):
        class_tmp_dir = tmp_root / cname
        if not class_tmp_dir.exists():
            raise FileNotFoundError(f"Missing tmp dir: {class_tmp_dir}")

        c_centroids, gpaths = compute_candidate_group_centroids_for_class(
            feat_net,
            class_tmp_dir,
            device=device,
            groups=args.groups,
            ipc=args.ipc,
            unify_hw=(args.unify_h, args.unify_w),
            encode_bs=args.encode_bs,
        )
        cand_centroids.append(c_centroids)              # (G,D)
        real_centroids_list.append(real_centroids[cid]) # (D,)
        per_class_group_paths.append(gpaths)            # (G,K)

    # 3) QUICK approximate alpha/beta (fast)
    print("[3] QUICK searching (alpha,beta) with normalized objective ...")

    # 小网格（默认 25 次），很快
    alphas = [0.2, 0.4, 0.6, 0.8, 1.0]
    betas  = [0.05, 0.1, 0.2, 0.4, 0.8]

    best_alpha, best_beta, best_score, close_avg, sep_avg, selected = quick_alpha_beta_search(
        cand_centroids=cand_centroids,
        real_centroids=real_centroids_list,
        eps=args.eps,
        alphas=alphas,
        betas=betas,
    )

    print("\n========== Approximate best params ==========")
    print(f"close_avg = {close_avg:.6f}")
    print(f"sep_avg   = {sep_avg:.6f}")
    print(f"best alpha={best_alpha:.2f}, best beta={best_beta:.2f} | score={best_score:.6f}")
    print("============================================\n")

    # 写一个 csv 给你记录
    csv_path = out_root / "approx_best_params.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["best_alpha", "best_beta", "best_score", "close_avg", "sep_avg"])
        writer.writerow([best_alpha, best_beta, best_score, close_avg, sep_avg])
    print(f"[Saved] {csv_path}")

    # 4) 可选：导出用这个 selection 的数据集（注意：这里 selection 是 close-only 的）
    if args.export:
        print("\n[4] Exporting dataset for the approximate selection ...")
        tag = f"approx_alpha{best_alpha:.2f}_beta{best_beta:.2f}"
        out_dir = ensure_dir(out_root / tag / "train")

        for i, cname in enumerate(sel_classes):
            gi = selected[i]
            dst_class_dir = ensure_dir(out_dir / cname)

            if args.clean:
                for ff in dst_class_dir.glob("*"):
                    if ff.is_file():
                        ff.unlink()

            src_paths = per_class_group_paths[i][gi]
            for p in src_paths:
                shutil.copy2(p, dst_class_dir / p.name)

        print(f"[Saved] {tag} -> {out_dir}")

    print("\nAll finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tmp-candidates", type=str, required=True)
    parser.add_argument("--real-train-dir", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)

    parser.add_argument("--class-index-file", type=str, default="./misc/class_indices.txt")
    parser.add_argument("--class-list-file", type=str, default="./misc/class_indices.txt")

    parser.add_argument("--nclass", type=int, default=1000)
    parser.add_argument("--phase", type=int, default=0)

    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--ipc", type=int, default=10)

    parser.add_argument("--feature-backbone", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--feat-img-size", type=int, default=224)

    parser.add_argument("--unify-h", type=int, default=256)
    parser.add_argument("--unify-w", type=int, default=256)
    parser.add_argument("--encode-bs", type=int, default=128)

    parser.add_argument("--real-max-per-class", type=int, default=200)
    parser.add_argument("--real-exts", nargs="+",
                        default=[".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff",".JPEG",".JPG",".PNG"])

    parser.add_argument("--eps", type=float, default=1e-6)

    parser.add_argument("--seed", type=int, default=0)

    # 这里保留接口（但 quick 版本不再用 topk 和 max-iters）
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-iters", type=int, default=5)

    parser.add_argument("--export", action="store_true", help="export dataset for the approximate selection")
    parser.add_argument("--clean", action="store_true")

    args = parser.parse_args()
    main(args)

