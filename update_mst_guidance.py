"""
Sample new images from a pre-trained DiT (with optional MST-DAG guided sampling).
Final Optimized Version v7 (Configurable Distance Metric):
1. Vectorized MST Cost calculation.
2. Correct Batch Slicing.
3. MST-Aware Diversity Loss.
4. Teacher Confidence Quality Cost (Linear/MSE/CE).
5. ✅ Configurable MST Distance Metric (L2/L1/Cosine).
6. Real-time Cost Magnitude Debugging.
"""
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse

# MST-DAG
from mst_guidance import MSTGuidance, SamplingGraph, topk_shortest_paths

# -----------------------------------------------------------------------------
# 通用的 MST 距离度量计算函数
# -----------------------------------------------------------------------------
def compute_mst_distance_metric(z_batch, mst_instance, metric="l2"):
    """
    计算 Latent 到 MST 骨架（最近节点）的距离，支持多种度量。
    返回: [B] Tensor
    """
    B = z_batch.shape[0]
    if B == 0: return torch.tensor([], device=z_batch.device)

    # 1. 投影到 PCA 空间 (保持流形一致性)
    z_proj = mst_instance._proj(z_batch) # [B, D]
    mst_nodes = mst_instance.mst_nodes   # [N, D]

    if metric == "l2":
        # 原版逻辑：欧氏距离
        dists = torch.cdist(z_proj, mst_nodes, p=2)
        min_dist, _ = dists.min(dim=1) # [B]
        return min_dist

    elif metric == "l1":
        # 曼哈顿距离：Sum(|x - y|)
        dists = torch.cdist(z_proj, mst_nodes, p=1)
        min_dist, _ = dists.min(dim=1)
        return min_dist

    elif metric == "cosine":
        # 余弦距离：1 - CosineSimilarity
        # 手动归一化计算相似度
        z_norm = F.normalize(z_proj, p=2, dim=1)
        nodes_norm = F.normalize(mst_nodes, p=2, dim=1)
        
        # [B, N] 相似度矩阵
        similarity = torch.mm(z_norm, nodes_norm.t())
        
        # 距离 = 1 - 最相似度 (Similarity越大，距离越小)
        max_sim, _ = similarity.max(dim=1)
        return 1.0 - max_sim

    else:
        raise ValueError(f"Unknown metric: {metric}")

# -----------------------------------------------------------------------------
# 2. Teacher Quality Cost 计算
# -----------------------------------------------------------------------------
def compute_teacher_quality_cost(z_batch, vae, teacher_model, resize_trans, target_label, loss_type="linear"):
    """
    计算基于教师模型置信度的 Cost。
    支持 linear, mse, ce 三种模式。
    """
    B = z_batch.shape[0]
    if B == 0: return torch.tensor([], device=z_batch.device)
    
    # VAE Decode (耗时操作)
    with torch.no_grad():
        images = vae.decode(z_batch / 0.18215).sample
    
    # Resize
    images = resize_trans(images)
    
    # Teacher Inference
    logits = teacher_model(images)
    probs = F.softmax(logits, dim=1) # [B, 1000]
    
    target_probs = probs[:, target_label]
    
    if loss_type == "linear":
        cost = 1.0 - target_probs
    elif loss_type == "mse":
        cost = (1.0 - target_probs) ** 2
    elif loss_type == "ce":
        cost = -torch.log(target_probs + 1e-6)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return cost

# -----------------------------------------------------------------------------
# 辅助函数：多样性 Cost
# -----------------------------------------------------------------------------
def compute_per_sample_diversity_cost(z_batch, mst_instance, temperature=0.1):
    """ Computes per-sample diversity cost. """
    B = z_batch.shape[0]
    if B <= 1: return torch.zeros(B, device=z_batch.device)
    
    z_proj = mst_instance._proj(z_batch) 
    dists_to_mst = torch.cdist(z_proj, mst_instance.mst_nodes, p=2)
    
    weights = F.softmax(-dists_to_mst / temperature, dim=1) 
    v_soft = torch.matmul(weights, mst_instance.mst_nodes) 
    
    anchor_dists = torch.cdist(v_soft, v_soft, p=2) 
    total_dist_per_sample = anchor_dists.sum(dim=1) 
    
    cost = - total_dist_per_sample / (B - 1 + 1e-6)
    return cost

# -----------------------------------------------------------------------------
# 辅助函数：Diffusion Step
# -----------------------------------------------------------------------------
@torch.no_grad()
def diffusion_step(diffusion, model_fn, z, t, model_kwargs, clip_denoised=False):
    B = z.shape[0]
    device = z.device
    if not torch.is_tensor(t): t = torch.tensor([t], device=device, dtype=torch.long)
    if t.dim() == 0: t = t.view(1)
    if t.numel() == 1: t = t.expand(B)
    else: assert t.shape == (B,), f"t.shape {tuple(t.shape)} != ({B},)"
    
    out = diffusion.p_sample(model_fn, z, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
    
    if isinstance(out, dict):
        if "sample" in out: return out["sample"]
        for k in ["x_prev", "prev_sample", "pred_prev_sample"]:
            if k in out: return out[k]
        raise KeyError(f"p_sample returned dict without known keys: {list(out.keys())}")
    return out

# -----------------------------------------------------------------------------
# 辅助函数：加载 MST 资源
# -----------------------------------------------------------------------------
def load_mst_assets(mst_root, sel_class, device):
    cls_dir = os.path.join(mst_root, sel_class)
    nodes_path = os.path.join(cls_dir, "mst_nodes.pt")
    edges_path = os.path.join(cls_dir, "mst_edges.pt")
    proj_path  = os.path.join(cls_dir, "pca_proj.pt")
    scale_path = os.path.join(cls_dir, "pca_scale.pt")
    
    mst_nodes = torch.load(nodes_path, map_location=device)
    mst_edges = torch.load(edges_path, map_location=device)
    
    pca_proj = None
    pca_scale = None
    if os.path.exists(proj_path):
        pca_proj = torch.load(proj_path, map_location="cpu")
    if os.path.exists(scale_path):
        pca_scale = torch.load(scale_path, map_location="cpu")
        
    return mst_nodes, mst_edges, pca_proj, pca_scale


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 准备类别标签
    try:
        with open('./misc/class_indices.txt', 'r') as fp:
            all_classes = [x.strip() for x in fp.readlines()]
        if args.spec == 'woof': file_list = './misc/class_woof.txt'
        elif args.spec == 'nette': file_list = './misc/class_nette.txt'
        else: file_list = './misc/class100.txt'
        with open(file_list, 'r') as fp: sel_classes = [x.strip() for x in fp.readlines()]
    except FileNotFoundError:
        print("Warning: Class list files not found. Using dummy classes.")
        all_classes = []; sel_classes = []

    if sel_classes:
        phase = max(0, args.phase)
        cls_from = args.nclass * phase
        cls_to = args.nclass * (phase + 1)
        sel_classes = sel_classes[cls_from:cls_to]
        class_labels = [all_classes.index(c) for c in sel_classes]
    else:
        class_labels = [0]; sel_classes = ["dummy"]

    if args.ckpt is None: assert args.model == "DiT-XL/2"

    # 2. 加载 DiT 模型
    latent_size = args.image_size // 8
    model = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes).to(device)
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps), learn_sigma=True)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    # 3. 初始化 Teacher Model
    teacher_model = None
    resize_trans = None

    if args.lambda_q > 0.0:
        print(f"Initializing teacher model: {args.arch_name}")
        teacher_model = torchvision.models.__dict__[args.arch_name](pretrained=False)
        if args.teacher_ckpt is not None:
            print(f"Loading teacher checkpoint from: {args.teacher_ckpt}")
            ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                teacher_model.load_state_dict(ckpt["state_dict"], strict=False)
            else:
                teacher_model.load_state_dict(ckpt, strict=False)
        else:
            print("Using torchvision pretrained weights (ImageNet).")
            teacher_model = torchvision.models.__dict__[args.arch_name](pretrained=True)

        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        for p in teacher_model.parameters(): p.requires_grad = False

        resize_trans = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        ])

    batch_size = 1

    # 4. 开始按类别循环
    for class_label, sel_class in zip(class_labels, sel_classes):
        os.makedirs(os.path.join(args.save_dir, sel_class), exist_ok=True)

        mst = None
        if args.use_dag:
            mst_nodes, mst_edges, pca_proj, pca_scale = load_mst_assets(args.mst_root, sel_class, device=device)
            # 关键修改：初始化 MST 时将 lambda_d 设为 0，因为我们在 main 中手动计算距离
            mst = MSTGuidance(
                mst_nodes=mst_nodes,
                mst_edges=mst_edges,
                lambda_d=0.0,             # <--- 禁用内部 L2 距离
                lambda_dir=args.lambda_dir, 
                pca_proj=pca_proj,
                pca_scale=pca_scale,
            )

        # 5. 采样循环
        for shift in tqdm(range(args.num_samples // batch_size)):
            z = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
            y = torch.tensor([class_label], device=device)

            z_full = torch.cat([z, z], 0) 
            y_null = torch.tensor([1000] * batch_size, device=device)
            y_full = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y_full, cfg_scale=args.cfg_scale)

            # --- Vanilla Sampling ---
            if not args.use_dag:
                samples = diffusion.p_sample_loop(
                    model.forward_with_cfg, z_full.shape, z_full,
                    clip_denoised=False, model_kwargs=model_kwargs,
                    progress=False, device=device
                )
                samples, _ = samples.chunk(2, dim=0)
                samples = vae.decode(samples / 0.18215).sample

            # --- MST-DAG Guided Sampling ---
            else:
                T = args.num_sampling_steps - 1 
                graph = SamplingGraph()
                root_id = graph.add_node(t=T, z=z_full, cost=0.0)

                for i in range(T, -1, -1):
                    parents = graph.nodes_at_time(i)
                    if len(parents) == 0:
                        if i > 0: raise RuntimeError(f"Empty layer at t={i}.")
                        else: break

                    # A. Prepare Batch
                    current_k = 1 if i > args.guide_start_t else args.dag_k
                    zc, zu, pid_mapping = [], [], []
                    parent_conds_list = []

                    for pid in parents:
                        parent_z = graph.nodes[pid].z
                        c, u = parent_z.chunk(2)
                        if i <= args.guide_start_t:
                            parent_conds_list.append(c.repeat(current_k, 1, 1, 1))
                        zc.extend([c] * current_k)
                        zu.extend([u] * current_k)
                        pid_mapping.extend([pid] * current_k)

                    # B. Inference (Diffusion Step)
                    z_t_batch = torch.cat(zc + zu, dim=0) 
                    t_batch = torch.full((z_t_batch.shape[0],), i, device=device, dtype=torch.long)
                    y_batch = torch.cat([y.repeat(len(zc)), y_null.repeat(len(zc))], dim=0)
                    model_kwargs_batch = dict(y=y_batch, cfg_scale=args.cfg_scale)
                    
                    z_prev_batch = diffusion_step(
                        diffusion, model.forward_with_cfg, z_t_batch, t_batch, model_kwargs_batch
                    )

                    # C. Cost Calculation & Pruning
                    all_conds, all_unconds = z_prev_batch.chunk(2, dim=0)
                    
                    if i <= args.guide_start_t:
                        parents_batch = torch.cat(parent_conds_list, dim=0)
                        children_batch = all_conds 

                        # 1. MST Direction Cost (Only Direction, since lambda_d=0)
                        dir_costs = mst.edge_cost(parents_batch, children_batch)
                        if dir_costs.dim() > 1:
                            dir_costs = dir_costs.view(dir_costs.shape[0], -1).mean(dim=1)
                        
                        total_costs = dir_costs.clone()
                        dir_val = dir_costs.mean().item()

                        # 2. MST Distance Cost (Custom Metric)
                        dist_val = 0.0
                        if args.lambda_d > 0.0:
                            dist_costs = compute_mst_distance_metric(
                                children_batch, 
                                mst, 
                                metric=args.dist_metric # <--- Configurable Metric
                            )
                            total_costs = total_costs + args.lambda_d * dist_costs
                            dist_val = dist_costs.mean().item()

                        # 3. Diversity Cost
                        div_val = 0.0
                        if args.lambda_div > 0.0:
                            div_costs = compute_per_sample_diversity_cost(children_batch, mst, temperature=args.div_temp)
                            total_costs = total_costs + args.lambda_div * div_costs
                            div_val = div_costs.mean().item()
                        
                        # 4. Teacher Quality Cost
                        qual_val = 0.0
                        if teacher_model is not None and args.lambda_q > 0.0:
                            qual_costs = compute_teacher_quality_cost(
                                children_batch, 
                                vae, 
                                teacher_model, 
                                resize_trans, 
                                target_label=class_label,
                                loss_type=args.quality_loss_type
                            )
                            total_costs = total_costs + args.lambda_q * qual_costs
                            qual_val = qual_costs.mean().item()

                        # 5. Debug Printing
                        #if shift == 0 and i % 5 == 0:
                            #final_val = total_costs.mean().item()
                            #print(f"\n[Step {i}] Cost Magnitudes (Mean):")
                            #print(f"  > MST Direction:       {dir_val:.4f} (W={args.lambda_dir})")
                            #print(f"  > MST Dist ({args.dist_metric}):     {dist_val:.4f} (W={args.lambda_d})")
                            #print(f"  > Diversity:           {div_val:.4f} (W={args.lambda_div})")
                            #print(f"  > Quality ({args.quality_loss_type}):     {qual_val:.4f} (W={args.lambda_q})")
                            #print(f"  > FINAL Weighted Sum:  {final_val:.4f}")
                            #print("-" * 40)

                        costs_list = total_costs.tolist()
                    else:
                        costs_list = [0.0] * len(all_conds)

                    candidates = []
                    for idx, pid in enumerate(pid_mapping):
                        child_z = torch.cat([all_conds[idx:idx+1], all_unconds[idx:idx+1]], dim=0)
                        cost = costs_list[idx]
                        candidates.append((child_z, pid, cost))

                    next_t = i - 1 if i > 0 else -1
                    graph.add_nodes_pruned(t=next_t, candidates=candidates, max_nodes=args.beam_width)

                # D. Pick Best Path
                paths = topk_shortest_paths(graph, start_time=T, end_time=-1, K=1)
                if not paths: raise RuntimeError("No path found.")
                best_node = paths[0][-1]
                z_final = best_node.z[:batch_size]
                if z_final.shape[0] > batch_size: z_final = z_final[:batch_size]
                samples = vae.decode(z_final / 0.18215).sample

            # 6. 保存图像
            for image_index, image in enumerate(samples):
                save_image(
                    image,
                    os.path.join(args.save_dir, sel_class, f"{image_index + shift * batch_size + args.total_shift}.png"),
                    normalize=True,
                    value_range=(-1, 1),
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--spec", type=str, default='woof')
    parser.add_argument("--save-dir", type=str, default='../logs/test')
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--total-shift", type=int, default=0)
    parser.add_argument("--nclass", type=int, default=10)
    parser.add_argument("--phase", type=int, default=0)

    # DAG flags
    parser.add_argument("--use-dag", action="store_true")
    parser.add_argument("--mst-root", type=str, default="./mst")
    parser.add_argument("--dag-k", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--guide-start-t", type=int, default=25)
    parser.add_argument("--lambda-d", type=float, default=1.0)
    parser.add_argument("--lambda-dir", type=float, default=0.0)
    parser.add_argument("--lambda-div", type=float, default=1.0)
    parser.add_argument("--div-temp", type=float, default=0.5)

    # MST Distance Metric
    parser.add_argument(
        "--dist-metric", 
        type=str, 
        default="l2", 
        choices=["l2", "l1", "cosine"],
        help="Distance metric for MST guidance: 'l2', 'l1', or 'cosine'"
    )

    # Teacher Quality Flags
    parser.add_argument("--lambda-q", type=float, default=0.0, help="Weight for teacher confidence quality loss")
    parser.add_argument("--arch-name", type=str, default="resnet50", help="Teacher model architecture")
    parser.add_argument("--teacher-ckpt", type=str, default=None, help="Path to teacher checkpoint (optional)")
    parser.add_argument(
        "--quality-loss-type", 
        type=str, 
        default="linear", 
        choices=["linear", "mse", "ce"],
        help="Type of teacher quality loss: 'linear' (1-p), 'mse' ((1-p)^2), or 'ce' (-log p)"
    )

    args = parser.parse_args()
    main(args)