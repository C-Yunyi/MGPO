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
        x = (img_tensor_bchw + 1.0) * 0.5
        x = self.tfm(x)
        feats = self.backbone(x)
        return F.normalize(feats, dim=1)


def build_feature_extractor(device="cuda", img_size=224):
    return ResNet18Feats(device=device, img_size=img_size)


# ----------------------------
# Objective & selection
# ----------------------------

def compute_centroid(feats: torch.Tensor) -> torch.Tensor:
    c = feats.mean(dim=0)
    return F.normalize(c, dim=0)

def cosine_dist(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.sum(u * v, dim=-1).clamp(-1, 1)

def eval_objective_terms(
    selected_idx: List[int],
    cand_centroids: List[torch.Tensor],
    real_centroids: List[torch.Tensor],
    eps: float,
):
    C = len(cand_centroids)

    close_terms = []
    picked = []
    for i in range(C):
        c_i = cand_centroids[i][selected_idx[i]]
        r_i = real_centroids[i]
        d_real = cosine_dist(c_i, r_i)
        close_terms.append(-torch.log(d_real + eps))
        picked.append(c_i.unsqueeze(0))

    picked = torch.cat(picked, dim=0)
    close_sum = torch.stack(close_terms).sum()

    sims = (picked @ picked.t()).clamp(-1, 1)
    dmat = 1.0 - sims
    i_idx, j_idx = torch.triu_indices(C, C, offset=1)
    sep_sum = torch.log(dmat[i_idx, j_idx] + eps).sum()

    return float(close_sum.item()), float(sep_sum.item())

def combined_objective(
    selected_idx,
    cand_centroids,
    real_centroids,
    alpha,
    beta,
    eps,
):
    close_sum, sep_sum = eval_objective_terms(
        selected_idx, cand_centroids, real_centroids, eps
    )
    return alpha * close_sum + beta * sep_sum

def optimize_selection(
    cand_centroids: List[torch.Tensor],
    real_centroids: List[torch.Tensor],
    alpha: float,
    beta: float,
    eps: float,
    max_iters: int = 5,
) -> List[int]:
    C = len(cand_centroids)
    G = cand_centroids[0].size(0)

    # init: closest to real centroid
    selected = []
    for i in range(C):
        r = real_centroids[i].unsqueeze(0)
        d = cosine_dist(cand_centroids[i], r)
        selected.append(int(torch.argmin(d)))

    best_val = combined_objective(
        selected, cand_centroids, real_centroids, alpha, beta, eps
    )

    for _ in range(max_iters):
        improved = False
        for i in range(C):
            cur = selected[i]
            best_g = cur
            best_local = best_val
            for g in range(G):
                if g == cur:
                    continue
                trial = list(selected)
                trial[i] = g
                v = combined_objective(
                    trial, cand_centroids, real_centroids, alpha, beta, eps
                )
                if v > best_local:
                    best_local = v
                    best_g = g
            if best_g != cur:
                selected[i] = best_g
                best_val = best_local
                improved = True
        if not improved:
            break

    return selected


# ----------------------------
# Image loading
# ----------------------------

@torch.no_grad()
def load_images_as_tensor(paths: List[Path], unify_hw=(256, 256)) -> torch.Tensor:
    to_tensor = transforms.ToTensor()
    Ht, Wt = unify_hw
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        t = to_tensor(img).unsqueeze(0)
        t = F.interpolate(t, size=(Ht, Wt), mode="bilinear", align_corners=False)
        t = t.squeeze(0) * 2.0 - 1.0
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
    unify_hw,
    encode_bs,
):
    rng = random.Random(seed)
    paths = list_images(class_dir, exts)
    if len(paths) > max_imgs:
        rng.shuffle(paths)
        paths = paths[:max_imgs]

    x = load_images_as_tensor(paths, unify_hw)
    feats = []
    for i in range(0, x.size(0), encode_bs):
        feats.append(feat_net.encode(x[i:i+encode_bs].to(device)))
    return compute_centroid(torch.cat(feats, dim=0))

@torch.no_grad()
def compute_candidate_group_centroids_for_class(
    feat_net,
    class_tmp_dir: Path,
    device,
    groups,
    ipc,
    unify_hw,
    encode_bs,
):
    centroids = []
    paths_per_group = []

    for g in range(groups):
        gdir = class_tmp_dir / f"group_{g:02d}"
        imgs = list_images(gdir)[:ipc]

        x = load_images_as_tensor(imgs, unify_hw)
        feats = []
        for i in range(0, x.size(0), encode_bs):
            feats.append(feat_net.encode(x[i:i+encode_bs].to(device)))
        c = compute_centroid(torch.cat(feats, dim=0))

        centroids.append(c.unsqueeze(0))
        paths_per_group.append(imgs)

    return torch.cat(centroids, dim=0), paths_per_group


# ----------------------------
# Main
# ----------------------------

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    FINAL_ALPHA = 1.0
    FINAL_BETA  = 0.05

    tmp_root = Path(args.tmp_candidates)
    real_root = Path(args.real_train_dir)
    out_root = ensure_dir(Path(args.out_root))

    all_classes = read_lines(args.class_index_file)
    sel_classes = read_lines(args.class_list_file)[:args.nclass]
    class_labels = [all_classes.index(x) for x in sel_classes]

    feat_net = build_feature_extractor(device=device, img_size=args.feat_img_size)
    real_exts = _normalize_exts(args.real_exts)

    print("[1] Compute real centroids")
    real_centroids = {}
    for cname, cid in tqdm(zip(sel_classes, class_labels), total=len(sel_classes)):
        real_centroids[cid] = compute_real_class_centroid(
            feat_net,
            real_root / cname,
            device,
            args.real_max_per_class,
            real_exts,
            args.seed,
            (args.unify_h, args.unify_w),
            args.encode_bs,
        )

    print("[2] Compute candidate centroids")
    cand_centroids = []
    real_centroids_list = []
    group_paths = []

    for cname, cid in tqdm(zip(sel_classes, class_labels), total=len(sel_classes)):
        c_centroids, gpaths = compute_candidate_group_centroids_for_class(
            feat_net,
            tmp_root / cname,
            device,
            args.groups,
            args.ipc,
            (args.unify_h, args.unify_w),
            args.encode_bs,
        )
        cand_centroids.append(c_centroids)
        real_centroids_list.append(real_centroids[cid])
        group_paths.append(gpaths)

    print("[3] Final selection (alpha=1.0, beta=0.05)")
    selected = optimize_selection(
        cand_centroids,
        real_centroids_list,
        FINAL_ALPHA,
        FINAL_BETA,
        args.eps,
        args.max_iters,
    )

    close_sum, sep_sum = eval_objective_terms(
        selected, cand_centroids, real_centroids_list, args.eps
    )
    print(f"close_sum={close_sum:.2f}, sep_sum={sep_sum:.2f}")

    if args.export:
        print("[4] Exporting final distilled dataset")
        out_dir = ensure_dir(out_root / "final_alpha1.00_beta0.05" / "train")
        for i, cname in enumerate(sel_classes):
            dst = ensure_dir(out_dir / cname)
            for p in group_paths[i][selected[i]]:
                shutil.copy2(p, dst / p.name)

    print("All finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tmp-candidates", type=str, required=True)
    parser.add_argument("--real-train-dir", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)

    parser.add_argument("--class-index-file", type=str, default="./misc/class_indices.txt")
    parser.add_argument("--class-list-file", type=str, default="./misc/class_indices.txt")

    parser.add_argument("--nclass", type=int, default=1000)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--ipc", type=int, default=10)

    parser.add_argument("--feat-img-size", type=int, default=224)
    parser.add_argument("--unify-h", type=int, default=256)
    parser.add_argument("--unify-w", type=int, default=256)
    parser.add_argument("--encode-bs", type=int, default=128)

    parser.add_argument("--real-max-per-class", type=int, default=200)
    parser.add_argument("--real-exts", nargs="+",
                        default=[".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff"])

    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--export", action="store_true")

    args = parser.parse_args()
    main(args)
