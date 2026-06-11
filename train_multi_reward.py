"""
Train DiT with GRPO + Multi-Reward System
基于 train_grpo_mst.py，将 MST 惩罚项改为独立的 reward

Multi-Reward System:
- Reward 1: Classifier reward (log P(y|image))
- Reward 2: MST reward (negative distance + direction penalties)

每个 reward 独立计算 advantage，然后加权组合
"""

import os
import math
import random
import argparse
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models import DiT_models
from download import find_model
from data import ImageFolder

# =======================================================
# 1. Helpers
# =======================================================
def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def mark_difffit_trainable(model):
    keys = ["bias", "norm", "gamma", "y_embed"]
    for name, p in model.named_parameters():
        p.requires_grad = any(k in name for k in keys)


def get_class_ids(spec="none"):
    with open("./misc/class_indices.txt", "r") as f:
        all_synsets = [line.strip() for line in f]
    synset_to_id = {s: i for i, s in enumerate(all_synsets)}

    if spec == "woof":
        subset_file = "./misc/class_woof.txt"
    elif spec == "nette":
        subset_file = "./misc/class_nette.txt"
    elif spec == "idc":
        subset_file = "./misc/idc_list.txt"
    elif spec == "100":
        subset_file = "./misc/class100.txt"
    elif spec == "none":
        return list(range(1000))
    else:
        raise ValueError(f"Unknown spec: {spec}")

    with open(subset_file, "r") as f:
        subset_synsets = [line.strip() for line in f]

    class_ids = []
    for syn in subset_synsets:
        if syn in synset_to_id:
            class_ids.append(synset_to_id[syn])
    return class_ids


def get_class_synsets(spec="none"):
    """获取类别的 synset 列表"""
    if spec == "woof":
        subset_file = "./misc/class_woof.txt"
    elif spec == "nette":
        subset_file = "./misc/class_nette.txt"
    elif spec == "idc":
        subset_file = "./misc/idc_list.txt"
    elif spec == "100":
        subset_file = "./misc/class100.txt"
    else:
        subset_file = "./misc/class100.txt"

    with open(subset_file, "r") as f:
        return [line.strip() for line in f]


def load_reward_model(device, model_name="resnet50", custom_ckpt=None, num_classes=1000):
    """加载 reward model，支持多种预训练模型和自定义 checkpoint"""
    model_dict = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
        "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
        "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1),
        "resnet152": (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V1),
        "convnext_base": (models.convnext_base, models.ConvNeXt_Base_Weights.IMAGENET1K_V1),
        "efficientnet_b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.IMAGENET1K_V1),
        "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1),
        "swin_s": (models.swin_s, models.Swin_S_Weights.IMAGENET1K_V1),
    }

    if model_name not in model_dict:
        raise ValueError(f"Unknown reward model: {model_name}. Available: {list(model_dict.keys())}")

    model_fn, weights = model_dict[model_name]
    print(f"Loading reward model: {model_name}")

    if custom_ckpt is not None:
        print(f"Loading custom checkpoint: {custom_ckpt}")
        model = model_fn(weights=None)
        # 修改最后一层以匹配 num_classes
        if hasattr(model, 'fc'):
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        elif hasattr(model, 'classifier'):
            if isinstance(model.classifier, torch.nn.Linear):
                model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
            else:
                model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
        elif hasattr(model, 'head'):
            model.head = torch.nn.Linear(model.head.in_features, num_classes)

        ckpt = torch.load(custom_ckpt, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        elif 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)
    else:
        model = model_fn(weights=weights)

    model.eval().to(device)
    model.requires_grad_(False)
    return model


@torch.no_grad()
def compute_reward(resnet, images, class_ids):
    """
    Reward 1: Classifier reward
    images: [-1,1] range
    reward: log P(y|image)
    """
    logits = resnet(images)
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs[torch.arange(len(class_ids), device=class_ids.device), class_ids]


@torch.no_grad()
def compute_reward2(mst_reward, x0_latent, x_mid, class_ids, id_to_synset, group_k,
                    lambda_dist=1.0, lambda_dir=1.0):
    """
    Reward 2: MST-based reward
    将 MST distance 和 direction penalty 转换为 reward（取负值）

    Args:
        mst_reward: MSTReward instance
        x0_latent: [B, C, H, W] final latent
        x_mid: [B, C, H, W] intermediate latent
        class_ids: [batch_size] class IDs (not expanded by group_k)
        id_to_synset: dict mapping class_id -> synset
        group_k: group size
        lambda_dist: weight for distance reward
        lambda_dir: weight for direction reward

    Returns:
        reward2: [B] MST-based reward (higher is better)
        dist_penalty: [B] raw distance penalty
        dir_penalty: [B] raw direction penalty
    """
    B = x0_latent.size(0)
    device = x0_latent.device

    dist_penalty = torch.zeros(B, device=device)
    dir_penalty = torch.zeros(B, device=device)

    if mst_reward is None:
        return torch.zeros(B, device=device), dist_penalty, dir_penalty

    batch_size = len(class_ids)

    # 按类别分组计算
    for i, cid in enumerate(class_ids):
        start_idx = i * group_k
        end_idx = (i + 1) * group_k
        synset = id_to_synset[cid]

        if lambda_dist > 0:
            dist_penalty[start_idx:end_idx] = mst_reward.compute_batch_distance_penalty(
                x0_latent[start_idx:end_idx], synset
            )

        if lambda_dir > 0:
            dir_penalty[start_idx:end_idx] = mst_reward.compute_batch_direction_penalty(
                x_mid[start_idx:end_idx], x0_latent[start_idx:end_idx], synset
            )

    # 转换为 reward（penalty 越小越好，所以取负值）
    reward2 = -(lambda_dist * dist_penalty + lambda_dir * dir_penalty)

    return reward2, dist_penalty, dir_penalty


def _extract(diffusion, arr_name, t, shape, device):
    arr = getattr(diffusion, arr_name)
    out = torch.from_numpy(arr).to(device=device)[t].float()
    while len(out.shape) < len(shape):
        out = out[..., None]
    return out + torch.zeros(shape, device=device)


def dit_eps(model, x, t, y):
    out = model(x, t, y)
    eps = out[:, :x.shape[1]]
    return eps


# =======================================================
# 2. MST Penalty Functions
# =======================================================
class MSTReward:
    """
    计算 MST 相关的惩罚/奖励
    - distance penalty: 到最近 MST 节点的距离
    - direction penalty: 与 MST 边方向的不一致性

    注意: MST 节点存储在 PCA 降维空间，需要先投影
    """
    def __init__(
        self,
        mst_root: str,
        class_synsets: list,
        device: str = "cuda",
        eps: float = 1e-8,
    ):
        self.device = device
        self.eps = eps
        self.class_mst = {}  # synset -> {nodes, edges, edge_mid, edge_dir, P, S}

        # 加载所有类的 MST
        print("Loading MST nodes for each class...")
        for synset in class_synsets:
            nodes_path = os.path.join(mst_root, synset, "mst_nodes.pt")
            edges_path = os.path.join(mst_root, synset, "mst_edges.pt")
            pca_proj_path = os.path.join(mst_root, synset, "pca_proj.pt")
            pca_scale_path = os.path.join(mst_root, synset, "pca_scale.pt")

            if os.path.exists(nodes_path) and os.path.exists(edges_path):
                nodes = torch.load(nodes_path, map_location=device).float()  # [N, D_pca]
                edges = torch.load(edges_path, map_location=device).long()   # [E, 2]

                # 加载 PCA 投影矩阵
                P = None
                S = None
                if os.path.exists(pca_proj_path):
                    P = torch.load(pca_proj_path, map_location=device).float()  # [D_latent, D_pca]
                if os.path.exists(pca_scale_path):
                    S = torch.load(pca_scale_path, map_location=device).float()  # [D_pca]

                # 预计算 edge midpoints & directions
                vi = nodes[edges[:, 0]]  # [E, D_pca]
                vj = nodes[edges[:, 1]]  # [E, D_pca]
                edge_mid = (vi + vj) * 0.5  # [E, D_pca]
                d = vj - vi
                edge_dir = d / (d.norm(dim=1, keepdim=True) + self.eps)  # [E, D_pca]

                self.class_mst[synset] = {
                    "nodes": nodes,
                    "edges": edges,
                    "edge_mid": edge_mid,
                    "edge_dir": edge_dir,
                    "P": P,
                    "S": S,
                }

        print(f"Loaded MST for {len(self.class_mst)} classes")

    def _proj(self, z_flat: torch.Tensor, synset: str) -> torch.Tensor:
        """投影到 PCA 空间"""
        if synset not in self.class_mst:
            return z_flat

        P = self.class_mst[synset]["P"]
        S = self.class_mst[synset]["S"]

        if P is not None:
            z_flat = z_flat @ P  # [B, D_pca]
            if S is not None:
                z_flat = z_flat / S
        return z_flat

    @torch.no_grad()
    def compute_batch_distance_penalty(self, z_latent: torch.Tensor, synset: str) -> torch.Tensor:
        """
        批量计算同一类别的距离惩罚
        z_latent: [B, C, H, W]
        synset: 单一类别
        returns: [B]
        """
        if synset not in self.class_mst:
            return torch.zeros(z_latent.size(0), device=self.device)

        z_flat = z_latent.flatten(1).float()  # [B, D]
        z_proj = self._proj(z_flat, synset)   # [B, D_pca]
        nodes = self.class_mst[synset]["nodes"]  # [N, D_pca]

        dist = torch.cdist(z_proj, nodes)  # [B, N]
        return dist.min(dim=1).values  # [B]

    @torch.no_grad()
    def compute_batch_direction_penalty(
        self,
        z_t: torch.Tensor,
        z_prev: torch.Tensor,
        synset: str
    ) -> torch.Tensor:
        """
        批量计算同一类别的方向惩罚
        """
        if synset not in self.class_mst:
            return torch.zeros(z_t.size(0), device=self.device)

        zt_flat = z_t.flatten(1).float()  # [B, D]
        zp_flat = z_prev.flatten(1).float()  # [B, D]

        # 投影到 PCA 空间
        zt_proj = self._proj(zt_flat, synset)  # [B, D_pca]
        zp_proj = self._proj(zp_flat, synset)  # [B, D_pca]

        edge_mid = self.class_mst[synset]["edge_mid"]  # [E, D_pca]
        edge_dir = self.class_mst[synset]["edge_dir"]  # [E, D_pca]

        # sample directions in PCA space
        d_sample = zp_proj - zt_proj  # [B, D_pca]
        d_norm = d_sample.norm(dim=1, keepdim=True) + self.eps
        d_sample = d_sample / d_norm

        # 找最近的 edge
        dist = torch.cdist(zt_proj, edge_mid)  # [B, E]
        nearest_idx = dist.argmin(dim=1)  # [B]
        d_data = edge_dir[nearest_idx]  # [B, D_pca]

        # cosine similarity
        cos = (d_sample * d_data).sum(dim=1).clamp(-1.0, 1.0)
        return 1.0 - cos


# =======================================================
# 3. Two-step Sampling Functions
# =======================================================
@torch.no_grad()
def two_step_sample_and_oldlogp(
    diffusion,
    model,
    z_T,
    y,
    t_mid,
    cfg_scale,
    eta_sde,
    noise_sde,
    t_start,
):
    """
    Two-step:
      step1 (SDE):  t=t_start -> t=t_mid
      step2 (ODE):  t=t_mid -> 0
    Returns:
      x0 (latent), x_mid, old_logp, mean1, sigma1
    """
    device = z_T.device
    B = z_T.shape[0]
    t1 = torch.full((B,), t_start, device=device, dtype=torch.long)
    t2 = torch.full((B,), int(t_mid), device=device, dtype=torch.long)
    t0 = torch.zeros((B,), device=device, dtype=torch.long)

    # ---- Step 1: stochastic DDIM jump (SDE) ----
    eps1 = dit_eps(model, z_T, t1, y)
    pred_x0_1 = (
        _extract(diffusion, "sqrt_recip_alphas_cumprod", t1, z_T.shape, device) * z_T
        - _extract(diffusion, "sqrt_recipm1_alphas_cumprod", t1, z_T.shape, device) * eps1
    ).clamp(-1, 1)

    a1 = _extract(diffusion, "alphas_cumprod", t1, z_T.shape, device)
    a2 = _extract(diffusion, "alphas_cumprod", t2, z_T.shape, device)

    sigma1 = (
        eta_sde
        * torch.sqrt((1 - a2) / (1 - a1))
        * torch.sqrt(1 - a1 / a2)
    )
    mean1 = pred_x0_1 * torch.sqrt(a2) + torch.sqrt(torch.clamp(1 - a2 - sigma1**2, min=0.0)) * eps1
    x_mid = mean1 + sigma1 * noise_sde

    deltaT = float(t_start - t_mid)
    sigma1_scaled = (sigma1 * deltaT).clamp(min=1e-6)
    standardized = (x_mid - mean1) / sigma1_scaled
    quad = 0.5 * (standardized.flatten(1) ** 2).sum(dim=1)
    old_logp = -quad

    # ---- Step 2: deterministic DDIM jump (ODE) t2 -> 0 ----
    eps2 = dit_eps(model, x_mid, t2, y)
    pred_x0_2 = (
        _extract(diffusion, "sqrt_recip_alphas_cumprod", t2, x_mid.shape, device) * x_mid
        - _extract(diffusion, "sqrt_recipm1_alphas_cumprod", t2, x_mid.shape, device) * eps2
    ).clamp(-1, 1)
    a0 = _extract(diffusion, "alphas_cumprod", t0, x_mid.shape, device)

    x0 = pred_x0_2 * torch.sqrt(a0) + torch.sqrt(torch.clamp(1 - a0, min=0.0)) * eps2

    return x0, x_mid, old_logp, mean1, sigma1


def newlogp_and_kl_two_steps(
    diffusion,
    model,
    ref_model,
    z_T,
    y,
    t_mid,
    cfg_scale,
    eta_sde,
    noise_sde,
    x_mid_target,
    x0_target,
    t_start,
    null_class,
):
    """
    Recompute both steps with gradient.
    """
    device = z_T.device
    B = z_T.shape[0]
    t1 = torch.full((B,), t_start, device=device, dtype=torch.long)
    t2 = torch.full((B,), int(t_mid), device=device, dtype=torch.long)
    t0 = torch.zeros((B,), device=device, dtype=torch.long)

    # ===== Step 1 (SDE): recompute logp with gradient =====
    eps1 = dit_eps(model, z_T, t1, y)
    pred_x0_1 = (
        _extract(diffusion, "sqrt_recip_alphas_cumprod", t1, z_T.shape, device) * z_T
        - _extract(diffusion, "sqrt_recipm1_alphas_cumprod", t1, z_T.shape, device) * eps1
    ).clamp(-1, 1)

    a1 = _extract(diffusion, "alphas_cumprod", t1, z_T.shape, device)
    a2 = _extract(diffusion, "alphas_cumprod", t2, z_T.shape, device)

    sigma1 = (
        eta_sde
        * torch.sqrt((1 - a2) / (1 - a1))
        * torch.sqrt(1 - a1 / a2)
    )
    mean1 = pred_x0_1 * torch.sqrt(a2) + torch.sqrt(torch.clamp(1 - a2 - sigma1**2, min=0.0)) * eps1

    deltaT = float(t_start - t_mid)
    sigma1_scaled = (sigma1 * deltaT).clamp(min=1e-6)
    standardized = (x_mid_target - mean1) / sigma1_scaled
    quad = 0.5 * (standardized.flatten(1) ** 2).sum(dim=1)
    new_logp = -quad

    # ===== Step 2 (ODE): compute pred_x0 =====
    eps2 = dit_eps(model, x_mid_target, t2, y)
    pred_x0_2 = (
        _extract(diffusion, "sqrt_recip_alphas_cumprod", t2, x_mid_target.shape, device) * x_mid_target
        - _extract(diffusion, "sqrt_recipm1_alphas_cumprod", t2, x_mid_target.shape, device) * eps2
    ).clamp(-1, 1)

    # ODE step loss (optional)
    loss_ode = F.mse_loss(pred_x0_2, x0_target.detach(), reduction='none').flatten(1).mean(dim=1)

    # ===== KL divergence =====
    with torch.no_grad():
        eps1_ref = dit_eps(ref_model, z_T, t1, y)
        pred_x0_ref = (
            _extract(diffusion, "sqrt_recip_alphas_cumprod", t1, z_T.shape, device) * z_T
            - _extract(diffusion, "sqrt_recipm1_alphas_cumprod", t1, z_T.shape, device) * eps1_ref
        ).clamp(-1, 1)
        mean_ref = pred_x0_ref * torch.sqrt(a2) + torch.sqrt(torch.clamp(1 - a2 - sigma1**2, min=0.0)) * eps1_ref
        eps2_ref = dit_eps(ref_model, x_mid_target, t2, y)

    kl_step1 = ((mean1 - mean_ref) ** 2 / (2 * torch.clamp(sigma1**2, min=1e-8))).flatten(1).mean(dim=1)
    kl_step2 = ((eps2 - eps2_ref) ** 2).flatten(1).mean(dim=1)
    kl = kl_step1 + kl_step2

    return new_logp, kl, loss_ode


# =======================================================
# 4. Multi-Reward Advantage Computation
# =======================================================
def compute_multi_reward_advantage(
    reward1: torch.Tensor,  # [B] classifier reward
    reward2: torch.Tensor,  # [B] MST reward
    batch_size: int,
    group_k: int,
    adv_eps: float = 1e-8,
    alpha1: float = 1.0,    # weight for reward1
    alpha2: float = 1.0,    # weight for reward2
    combine_mode: str = "weighted_sum",  # "weighted_sum" or "separate"
):
    """
    计算 multi-reward 的 advantage

    Args:
        reward1: classifier reward [B]
        reward2: MST reward [B]
        batch_size: number of unique samples
        group_k: group size
        adv_eps: epsilon for numerical stability
        alpha1: weight for reward1
        alpha2: weight for reward2
        combine_mode:
            - "weighted_sum": 先加权求和再计算 advantage
            - "separate": 分别计算 advantage 再加权求和

    Returns:
        adv: [B] combined advantage
        adv1: [B] advantage from reward1 (for logging)
        adv2: [B] advantage from reward2 (for logging)
    """
    rollout_B = batch_size * group_k

    # Reshape for group normalization
    r1_reshaped = reward1.view(batch_size, group_k)
    r2_reshaped = reward2.view(batch_size, group_k)

    if combine_mode == "weighted_sum":
        # 先加权求和
        total_reward = alpha1 * reward1 + alpha2 * reward2
        total_reshaped = total_reward.view(batch_size, group_k)

        # Group normalize
        mean = total_reshaped.mean(dim=1, keepdim=True)
        std = total_reshaped.std(dim=1, keepdim=True)
        adv_reshaped = (total_reshaped - mean) / (std + adv_eps)
        adv = adv_reshaped.view(rollout_B)

        # 为了 logging，也计算单独的 advantage
        r1_mean = r1_reshaped.mean(dim=1, keepdim=True)
        r1_std = r1_reshaped.std(dim=1, keepdim=True)
        adv1 = ((r1_reshaped - r1_mean) / (r1_std + adv_eps)).view(rollout_B)

        r2_mean = r2_reshaped.mean(dim=1, keepdim=True)
        r2_std = r2_reshaped.std(dim=1, keepdim=True)
        adv2 = ((r2_reshaped - r2_mean) / (r2_std + adv_eps)).view(rollout_B)

    elif combine_mode == "separate":
        # 分别计算 advantage
        r1_mean = r1_reshaped.mean(dim=1, keepdim=True)
        r1_std = r1_reshaped.std(dim=1, keepdim=True)
        adv1_reshaped = (r1_reshaped - r1_mean) / (r1_std + adv_eps)
        adv1 = adv1_reshaped.view(rollout_B)

        r2_mean = r2_reshaped.mean(dim=1, keepdim=True)
        r2_std = r2_reshaped.std(dim=1, keepdim=True)
        adv2_reshaped = (r2_reshaped - r2_mean) / (r2_std + adv_eps)
        adv2 = adv2_reshaped.view(rollout_B)

        # 加权求和 advantage
        adv_combined = alpha1 * adv1 + alpha2 * adv2

        # 对批次内所有rollout的总优势进行归一化
        adv_mean = adv_combined.mean()
        adv_std = adv_combined.std()
        adv = (adv_combined - adv_mean) / (adv_std + adv_eps)
    else:
        raise ValueError(f"Unknown combine_mode: {combine_mode}")

    return adv, adv1, adv2


# =======================================================
# 5. Main Training
# =======================================================
def main(args):
    device = args.device if torch.cuda.is_available() else "cpu"

    latent_size = args.image_size // 8
    null_class = args.num_classes
    t_start = args.t_total - 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ---------- DiT ----------
    model = DiT_models[args.model_name](input_size=latent_size, num_classes=args.num_classes).to(device)
    if args.resume_ckpt is not None:
        print(f"Resuming from checkpoint: {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model.load_state_dict(ckpt['model'], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
    else:
        ckpt = find_model(f"DiT-XL-2-{args.image_size}x{args.image_size}.pt")
        model.load_state_dict(ckpt, strict=False)

    mark_difffit_trainable(model)
    model.train()

    # ---------- Reference ----------
    ref_model = deepcopy(model).eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # ---------- Diffusion ----------
    diffusion = create_diffusion("")
    assert diffusion.num_timesteps == args.t_total
    diffusion_sample = create_diffusion(str(args.num_sampling_steps))

    # ---------- VAE ----------
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # ---------- Reward Model ----------
    reward_model = load_reward_model(
        device, args.reward_model,
        custom_ckpt=args.reward_ckpt,
        num_classes=args.reward_num_classes
    )
    class_ids_pool = get_class_ids(args.spec)
    class_synsets = get_class_synsets(args.spec)

    # 如果使用自定义 reward model，需要映射 class IDs
    use_local_class_ids = args.reward_ckpt is not None and args.reward_num_classes < 1000
    imagenet_to_local = {cid: i for i, cid in enumerate(class_ids_pool)}
    if use_local_class_ids:
        print(f"Using local class IDs mapping: {imagenet_to_local}")

    # 建立 class_id -> synset 的映射
    with open("./misc/class_indices.txt", "r") as f:
        all_synsets = [line.strip() for line in f]
    id_to_synset = {i: s for i, s in enumerate(all_synsets)}

    # ---------- MST Reward ----------
    mst_reward = None
    if args.alpha_mst > 0:
        mst_reward = MSTReward(
            mst_root=args.mst_root,
            class_synsets=class_synsets,
            device=device,
        )

    # Create save directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_model_dir, exist_ok=True)

    # ---------- Training ----------
    pbar = tqdm(range(args.start_step, args.total_iters), desc="Training", unit="step")
    for step in pbar:
        # Sample class labels
        class_ids = np.random.choice(class_ids_pool, size=args.batch_size, replace=True)
        y = torch.tensor(np.repeat(class_ids, args.group_k), device=device, dtype=torch.long)
        rollout_B = args.batch_size * args.group_k

        #t_mid = int(np.random.randint(1, t_start))
        #comment out, test interval
        low = int(0.2 * t_start)
        high = int(0.8 * t_start)
        t_mid = int(np.random.randint(low, high + 1))
        ###########################################
        z_T = torch.randn(rollout_B, args.latent_channels, latent_size, latent_size, device=device)
        noise_sde = torch.randn_like(z_T)

        # ---- Rollout (no grad) ----
        with torch.no_grad():
            x0_latent, x_mid, old_logp, _, _ = two_step_sample_and_oldlogp(
                diffusion=diffusion,
                model=model,
                z_T=z_T,
                y=y,
                t_mid=t_mid,
                cfg_scale=args.cfg_scale,
                eta_sde=args.eta_sde,
                noise_sde=noise_sde,
                t_start=t_start,
            )

            # ===== Reward 1: Classifier =====
            images = vae.decode(x0_latent / 0.18215).sample
            if use_local_class_ids:
                y_reward = torch.tensor([imagenet_to_local[cid.item()] for cid in y], device=device)
            else:
                y_reward = y
            reward1 = compute_reward(reward_model, images, y_reward)

            # ===== Reward 2: MST =====
            reward2, dist_penalty, dir_penalty = compute_reward2(
                mst_reward=mst_reward,
                x0_latent=x0_latent,
                x_mid=x_mid,
                class_ids=class_ids,
                id_to_synset=id_to_synset,
                group_k=args.group_k,
                lambda_dist=args.lambda_mst_dist,
                lambda_dir=args.lambda_mst_dir,
            )

            # ===== Multi-Reward Advantage =====
            adv, adv1, adv2 = compute_multi_reward_advantage(
                reward1=reward1,
                reward2=reward2,
                batch_size=args.batch_size,
                group_k=args.group_k,
                adv_eps=args.adv_eps,
                alpha1=args.alpha_cls,
                alpha2=args.alpha_mst,
                combine_mode=args.combine_mode,
            )

        # ---- Recompute with grad ----
        new_logp, kl_vec, loss_ode_vec = newlogp_and_kl_two_steps(
            diffusion=diffusion,
            model=model,
            ref_model=ref_model,
            z_T=z_T,
            y=y,
            t_mid=t_mid,
            cfg_scale=args.cfg_scale,
            eta_sde=args.eta_sde,
            noise_sde=noise_sde,
            x_mid_target=x_mid.detach(),
            x0_target=x0_latent.detach(),
            t_start=t_start,
            null_class=null_class,
        )

        # PPO clip
        logp_diff = new_logp - old_logp
        ratio = torch.exp(logp_diff)
        ratio_clipped = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)
        policy_loss = -(torch.minimum(ratio * adv, ratio_clipped * adv))

        kl_loss = kl_vec
        loss_ode = loss_ode_vec.mean()

        grpo = policy_loss + args.kl_beta * kl_loss
        loss_grpo = grpo.mean()

        # Total loss
        total_loss = loss_grpo + args.lambda_ode * loss_ode

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Logging
        pbar.set_postfix({
            'Loss': f'{total_loss.item():.4f}',
            'R1': f'{reward1.mean().item():.3f}',
            'R2': f'{reward2.mean().item():.3f}',
            'KL': f'{kl_loss.mean().item():.4f}'
        })

        if step % args.log_interval == 0:
            print(
                f"\n[{step}] t_mid={t_mid} "
                f"Total={total_loss.item():.4f} GRPO={loss_grpo.item():.4f} "
                f"KL={kl_loss.mean().item():.4f} "
                f"R1(cls)={reward1.mean().item():.3f} R2(mst)={reward2.mean().item():.3f} "
                f"Adv1={adv1.mean().item():.3f} Adv2={adv2.mean().item():.3f} "
                f"MST_dist={dist_penalty.mean().item():.3f} MST_dir={dir_penalty.mean().item():.3f}"
            )

            # Sample images for visualization
            with torch.no_grad():
                num_samples = min(16, args.batch_size)
                sample_class_ids = class_ids[:num_samples] if len(class_ids) >= num_samples else class_ids
                num_samples = len(sample_class_ids)

                z_sample = torch.randn(num_samples, args.latent_channels, latent_size, latent_size, device=device)
                y_sample = torch.tensor(sample_class_ids, device=device, dtype=torch.long)

                z_in = torch.cat([z_sample, z_sample], dim=0)
                y_null = torch.full_like(y_sample, null_class)
                y_in = torch.cat([y_sample, y_null], dim=0)
                model_kwargs = dict(y=y_in, cfg_scale=args.cfg_scale)

                sample_latent = diffusion_sample.p_sample_loop(
                    model.forward_with_cfg,
                    z_in.shape,
                    noise=z_in,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    device=device,
                    progress=False
                )
                x0_sample = sample_latent.chunk(2, dim=0)[0]
                images_sample = vae.decode(x0_sample / 0.18215).sample

            save_path = os.path.join(args.save_dir, f"step_{step}.png")
            save_image(images_sample, save_path, normalize=True, value_range=(-1, 1))

        # Save checkpoint
        if step % args.checkpoint_interval == 0 and step > 0:
            checkpoint = {
                "model": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
            }
            checkpoint_path = os.path.join(args.save_model_dir, f"ckpt_step_{step}.pt")
            torch.save(checkpoint, checkpoint_path)
            print(f"\n[✓ Checkpoint Saved] Step: {step}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DiT with GRPO + Multi-Reward")

    # Device and seed
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    # Model config
    parser.add_argument("--model-name", type=str, default="DiT-XL/2", choices=list(DiT_models.keys()))
    parser.add_argument("--image-size", type=int, default=256, choices=[256, 512])
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--resume-ckpt", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--start-step", type=int, default=0, help="Starting step number")

    # Diffusion config
    parser.add_argument("--t-total", type=int, default=1000)
    parser.add_argument("--eta-sde", type=float, default=1.0)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)

    # Training config
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-k", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--total-iters", type=int, default=600)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--adv-eps", type=float, default=1e-8)
    parser.add_argument("--kl-beta", type=float, default=0.1)
    parser.add_argument("--lambda-ode", type=float, default=0.0, help="ODE loss weight (default 0)")

    # Multi-Reward config
    parser.add_argument("--alpha-cls", type=float, default=1.0, help="Weight for classifier reward")
    parser.add_argument("--alpha-mst", type=float, default=1.0, help="Weight for MST reward")
    parser.add_argument("--combine-mode", type=str, default="weighted_sum",
                        choices=["weighted_sum", "separate"],
                        help="How to combine multi-reward advantages")

    # MST reward config
    parser.add_argument("--mst-root", type=str, default="./mst_fast", help="MST data directory")
    parser.add_argument("--lambda-mst-dist", type=float, default=1.0, help="Weight for MST distance in reward2")
    parser.add_argument("--lambda-mst-dir", type=float, default=0.0, help="Weight for MST direction in reward2")

    # Data config
    parser.add_argument("--spec", type=str, default="none", choices=["woof", "nette", "idc", "100", "none"])

    # Reward model config
    parser.add_argument("--reward-model", type=str, default="resnet50",
                        choices=["resnet18", "resnet50", "resnet101", "resnet152", "convnext_base", "efficientnet_b4", "vit_b_16", "swin_s"],
                        help="Reward model for GRPO training")
    parser.add_argument("--reward-ckpt", type=str, default=None, help="Custom reward model checkpoint")
    parser.add_argument("--reward-num-classes", type=int, default=1000, help="Number of classes for reward model")

    # VAE config
    parser.add_argument("--vae", type=str, default="mse", choices=["ema", "mse"])

    # Save config
    parser.add_argument("--save-dir", type=str, default="/root/autodl-tmp/grpo_multi_samples")
    parser.add_argument("--save-model-dir", type=str, default="/root/autodl-tmp/fine-tuned-model-multi")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=550)

    args = parser.parse_args()
    main(args)
