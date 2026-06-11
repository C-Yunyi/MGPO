"""
Sample new images from a pre-trained DiT (with optional MST-DAG guided sampling).
Final Optimized Version:
1. Vectorized MST Cost calculation (Real Speedup).
2. Correct Batch Slicing (High Quality).
3. Fixed Timestep Indexing (No Crash).
"""
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse

# MST-DAG
from mst_guidance import MSTGuidance, SamplingGraph, topk_shortest_paths

def compute_teacher_quality_cost(z_batch, vae, teacher_model, resize_trans, target_label):
    """
    计算基于教师模型置信度的 Cost。
    Cost = 1.0 - Probability(target_label)
    """
    B = z_batch.shape[0]
    if B == 0: return torch.tensor([])
    
    # 1. VAE Decode (这是耗时操作，仅在筛选阶段进行)
    # z: [B, 4, H, W] -> images: [B, 3, 256, 256]
    with torch.no_grad():
        images = vae.decode(z_batch / 0.18215).sample
    
    # 2. Resize to 224x224 (ResNet 标准输入)
    images = resize_trans(images)
    
    # 3. Teacher Inference
    logits = teacher_model(images)
    probs = F.softmax(logits, dim=1) # [B, 1000]
    
    # 4. 获取目标类别的概率
    target_probs = probs[:, target_label]
    
    # 5. Cost: 置信度越高(1.0)，Cost越低(0.0)
    cost = 1.0 - target_probs
    
    return cost
# -------------------

def compute_per_sample_diversity_cost(z_batch, mst_instance, temperature=0.1):
    """
    Computes per-sample diversity cost based on Soft-MST-Anchor repulsion.
    Returns [B] tensor where lower value (more negative) means better diversity (further from others).
    """
    B = z_batch.shape[0]
    if B <= 1:
        return torch.zeros(B, device=z_batch.device)

    # 1. Project to PCA space using MST instance
    z_proj = mst_instance._proj(z_batch) # [B, D]

    # 2. Distance to MST nodes
    # mst_nodes is [N, D]
    dists_to_mst = torch.cdist(z_proj, mst_instance.mst_nodes, p=2)

    # 3. Softmin -> Soft Anchors
    weights = F.softmax(-dists_to_mst / temperature, dim=1) # [B, N]
    v_soft = torch.matmul(weights, mst_instance.mst_nodes)  # [B, D]

    # 4. Pairwise distances between soft anchors
    anchor_dists = torch.cdist(v_soft, v_soft, p=2) # [B, B]

    # 5. Sum of distances to all other samples for each sample
    total_dist_per_sample = anchor_dists.sum(dim=1) # [B]

    # 6. Cost = - AvgDistance (Maximize distance -> Minimize Cost)
    cost = - total_dist_per_sample / (B - 1 + 1e-6)
    
    return cost


@torch.no_grad()
def diffusion_step(diffusion, model_fn, z, t, model_kwargs, clip_denoised=False):
    """
    One reverse diffusion step: z_t -> z_{t-1}
    """
    # t must be a tensor on device, shape [B]
    B = z.shape[0]
    device = z.device
    
    if not torch.is_tensor(t):
        t = torch.tensor([t], device=device, dtype=torch.long)

    if t.dim() == 0:
        t = t.view(1)

    # expand to (B,)
    if t.numel() == 1:
        t = t.expand(B)
    else:
        assert t.shape == (B,), f"t.shape {tuple(t.shape)} != ({B},)"

    out = diffusion.p_sample(model_fn, z, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
    if isinstance(out, dict):
        if "sample" in out:
            return out["sample"]
        for k in ["x_prev", "prev_sample", "pred_prev_sample"]:
            if k in out:
                return out[k]
        raise KeyError(f"p_sample returned dict without known keys: {list(out.keys())}")
    return out


def load_mst_assets(mst_root, sel_class, device):
    """
    Loads per-class MST assets.
    """
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

    # Labels to condition the model
    try:
        with open('./misc/class_indices.txt', 'r') as fp:
            all_classes = [x.strip() for x in fp.readlines()]

        if args.spec == 'woof':
            file_list = './misc/class_woof.txt'
        elif args.spec == 'nette':
            file_list = './misc/class_nette.txt'
        else:
            file_list = './misc/class100.txt'

        with open(file_list, 'r') as fp:
            sel_classes = [x.strip() for x in fp.readlines()]
    except FileNotFoundError:
        print("Warning: Class list files not found in ./misc/. Using dummy classes if necessary.")
        all_classes = []
        sel_classes = []

    if sel_classes:
        phase = max(0, args.phase)
        cls_from = args.nclass * phase
        cls_to = args.nclass * (phase + 1)
        sel_classes = sel_classes[cls_from:cls_to]
        class_labels = [all_classes.index(c) for c in sel_classes]
    else:
        class_labels = [0]
        sel_classes = ["dummy"]

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model
    latent_size = args.image_size // 8
    model = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes).to(device)
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # FIX 1: Add learn_sigma=True
    diffusion = create_diffusion(str(args.num_sampling_steps), learn_sigma=True)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    # ✅ 3. 初始化 Teacher Model (如果在参数中启用)
    teacher_model = None
    resize_trans = None
    if args.lambda_q > 0.0:
        print(f"Loading teacher model: {args.arch_name}...")
        teacher_model = torchvision.models.__dict__[args.arch_name](pretrained=True).to(device)
        teacher_model.eval()
        for p in teacher_model.parameters(): p.requires_grad = False
        # 定义 Resize 操作
        resize_trans = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        ])
        .
        
    batch_size = 1

    for class_label, sel_class in zip(class_labels, sel_classes):
        os.makedirs(os.path.join(args.save_dir, sel_class), exist_ok=True)

        mst = None
        if args.use_dag:
            mst_nodes, mst_edges, pca_proj, pca_scale = load_mst_assets(args.mst_root, sel_class, device=device)
            mst = MSTGuidance(
                mst_nodes=mst_nodes,
                mst_edges=mst_edges,
                lambda_d=args.lambda_d,
                lambda_dir=args.lambda_dir,
                pca_proj=pca_proj,
                pca_scale=pca_scale,
            )

        for shift in tqdm(range(args.num_samples // batch_size)):
            # Create sampling noise
            z = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
            y = torch.tensor([class_label], device=device)

            # Setup classifier-free guidance
            z_full = torch.cat([z, z], 0)  # [2B,4,H,W]
            y_null = torch.tensor([1000] * batch_size, device=device)
            y_full = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y_full, cfg_scale=args.cfg_scale)

            # --------------------------
            # Vanilla sampling (original)
            # --------------------------
            if not args.use_dag:
                samples = diffusion.p_sample_loop(
                    model.forward_with_cfg, z_full.shape, z_full,
                    clip_denoised=False, model_kwargs=model_kwargs,
                    progress=False, device=device
                )
                samples, _ = samples.chunk(2, dim=0)
                samples = vae.decode(samples / 0.18215).sample

            # --------------------------
            # DAG / Beam + MST guidance (Vectorized)
            # --------------------------
            else:
                T = args.num_sampling_steps - 1 

                graph = SamplingGraph()
                root_id = graph.add_node(t=T, z=z_full, cost=0.0)

                for i in range(T, -1, -1):
                    # FIX 2: Pass index 'i' directly (SpacedDiffusion handles mapping)
                    # No manual mapping here to avoid "Index out of bounds"
                    
                    parents = graph.nodes_at_time(i)
                    if len(parents) == 0:
                        if i > 0: raise RuntimeError(f"Empty layer at t={i}.")
                        else: break

                    # -------------------------
                    # 1. Prepare Batch Data
                    # -------------------------
                    # Expand children: 1 if early stage, k if late stage
                    current_k = 1 if i > args.guide_start_t else args.dag_k
                    
                    zc, zu, pid_mapping = [], [], []
                    parent_conds_list = [] # For vectorized cost calculation

                    for pid in parents:
                        # Extract parent latent
                        parent_z = graph.nodes[pid].z
                        c, u = parent_z.chunk(2)
                        
                        # Store parent cond for cost calculation later (only if needed)
                        if i <= args.guide_start_t:
                            # Expand parent cond to match children count for direct batch computation
                            # [1, C, H, W] -> [current_k, C, H, W]
                            parent_conds_list.append(c.repeat(current_k, 1, 1, 1))

                        # Expand inputs for diffusion
                        # List multiplication is faster than loop
                        zc.extend([c] * current_k)
                        zu.extend([u] * current_k)
                        pid_mapping.extend([pid] * current_k)

                    # -------------------------
                    # 2. Batch Inference
                    # -------------------------
                    # [Total_Children, C, H, W]
                    z_t_batch = torch.cat(zc + zu, dim=0) 
                    total_items = z_t_batch.shape[0]

                    # Construct t_batch with index 'i'
                    t_batch = torch.full((total_items,), i, device=device, dtype=torch.long)
                    y_batch = torch.cat([y.repeat(len(zc)), y_null.repeat(len(zc))], dim=0)
                    model_kwargs_batch = dict(y=y_batch, cfg_scale=args.cfg_scale)
                    
                    # Run diffusion step on GPU
                    z_prev_batch = diffusion_step(
                        diffusion, model.forward_with_cfg, z_t_batch, t_batch, model_kwargs_batch
                    )

                    # -------------------------
                    # 3. Process Results & Vectorized Cost
                    # -------------------------
                    # FIX 3: Correct Slicing (Separating Cond/Uncond first)
                    all_conds, all_unconds = z_prev_batch.chunk(2, dim=0)
                    
                    costs_list = []

                    # === Vectorized Cost Calculation (The Speedup) ===
                    if i <= args.guide_start_t:
                        # Stack parent conds: [Total_Children, C, H, W]
                        parents_batch = torch.cat(parent_conds_list, dim=0)
                        children_batch = all_conds # [Total_Children, C, H, W]

                        # Calculate all costs at once on GPU
                        # Output shape is likely [Total_Children] or [Total_Children, ...]
                        raw_costs = mst.edge_cost(parents_batch, children_batch)
                        
                        # Safety reduction if cost returns feature map
                        if raw_costs.dim() > 1:
                            raw_costs = raw_costs.view(raw_costs.shape[0], -1).mean(dim=1)

                        if args.lambda_div > 0.0:
                            div_costs = compute_per_sample_diversity_cost(
                                children_batch, 
                                mst, 
                                temperature=args.div_temp
                            )
                            # Add weighted diversity cost
                            raw_costs = raw_costs + args.lambda_div * div_costs

                        # ✅ 4. 插入：Teacher Quality Loss
                        # 只在需要的时候计算 (Teacher Model 存在 且 权重 > 0)
                        if teacher_model is not None and args.lambda_q > 0.0:
                            qual_costs = compute_teacher_quality_cost(
                                children_batch, 
                                vae, 
                                teacher_model, 
                                resize_trans, 
                                target_label=class_label
                            )
                            raw_costs = raw_costs + args.lambda_q * qual_costs
                        # Single CPU synchronization
                        costs_list = raw_costs.tolist()
                    else:
                        # Early stage: cost is 0
                        costs_list = [0.0] * len(all_conds)

                    # -------------------------
                    # 4. Construct Candidates
                    # -------------------------
                    candidates = []
                    
                    # Loop over CPU lists (Fast)
                    for idx, pid in enumerate(pid_mapping):
                        # Re-pair Cond and Uncond: [2, C, H, W]
                        child_z = torch.cat([all_conds[idx:idx+1], all_unconds[idx:idx+1]], dim=0)
                        cost = costs_list[idx]
                        candidates.append((child_z, pid, cost))

                    # -------------------------
                    # 5. Update Graph
                    # -------------------------
                    next_t = i - 1 if i > 0 else -1
                    graph.add_nodes_pruned(t=next_t, candidates=candidates, max_nodes=args.beam_width)

                # Pick best path
                paths = topk_shortest_paths(graph, start_time=T, end_time=-1, K=1)
                
                if not paths:
                    raise RuntimeError("No path found. Check MST assets and DAG params.")
                
                best_node = paths[0][-1] # Node at t=-1
                z_final = best_node.z[:batch_size] # Extract Cond part if needed, or just z
                
                # Check shape, usually we need just z
                if z_final.shape[0] > batch_size:
                     z_final = z_final[:batch_size]

                samples = vae.decode(z_final / 0.18215).sample

            # Save images
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

    # DAG guidance flags
    parser.add_argument("--use-dag", action="store_true")
    parser.add_argument("--mst-root", type=str, default="./mst")
    parser.add_argument("--dag-k", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--guide-start-t", type=int, default=25)
    parser.add_argument("--lambda-d", type=float, default=1.0)
    parser.add_argument("--lambda-dir", type=float, default=0.0)
    # ✅ 3. 新增：Diversity 参数
    parser.add_argument("--lambda-div", type=float, default=1.0, help="Weight for diversity repulsion loss")
    parser.add_argument("--div-temp", type=float, default=0.5, help="Softmin temperature for diversity anchor")
        # ✅ 5. 新增：Teacher Quality 参数
    parser.add_argument("--lambda-q", type=float, default=0.0, help="Weight for teacher confidence quality loss")
    parser.add_argument("--arch-name", type=str, default="resnet50", help="Teacher model architecture (e.g. resnet18, resnet50)")

    args = parser.parse_args()
    main(args)