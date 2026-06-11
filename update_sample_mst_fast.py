"""
优化版 MST-DAG guided sampling.

主要优化:
1. 批量化 diffusion step - 所有 parents 一次前向
2. 批量化子节点采样 - dag_k 个子节点一次前向
3. 减少 97% 的模型调用次数
"""
import os
import torch
from tqdm import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse

from update_mst_guidance_fast import MSTGuidanceFast, SamplingGraphFast


@torch.no_grad()
def batched_diffusion_step(diffusion, model_fn, z_cond_batch, t, model_kwargs_base, cfg_scale):
    """
    批量 diffusion step.
    z_cond_batch: [P, C, H, W] - P 个 cond latent
    返回: [P, C, H, W] (cond 部分)
    """
    P = z_cond_batch.shape[0]
    device = z_cond_batch.device

    # CFG: 拼接 cond 和 uncond
    z_full = torch.cat([z_cond_batch, z_cond_batch], dim=0)  # [2P, C, H, W]

    # 构建 y: 前 P 个是 class label，后 P 个是 null (1000)
    y_cond = model_kwargs_base["y_cond"].expand(P)  # [P]
    y_null = model_kwargs_base["y_null"].expand(P)  # [P]
    y_full = torch.cat([y_cond, y_null], dim=0)  # [2P]

    model_kwargs = dict(y=y_full, cfg_scale=cfg_scale)

    # timestep
    t_tensor = torch.full((2 * P,), t, device=device, dtype=torch.long)

    # p_sample
    out = diffusion.p_sample(model_fn, z_full, t_tensor, clip_denoised=False, model_kwargs=model_kwargs)

    if isinstance(out, dict):
        sample = out.get("sample", out.get("x_prev", out.get("prev_sample")))
    else:
        sample = out

    # 只取 cond 部分
    z_prev_cond = sample[:P]
    return z_prev_cond


@torch.no_grad()
def batched_diffusion_step_with_noise_variants(
    diffusion, model_fn, z_cond_parents, t, model_kwargs_base, cfg_scale, dag_k
):
    """
    对每个 parent 生成 dag_k 个不同噪声的子节点.

    z_cond_parents: [P, C, H, W] - P 个 parent cond latents
    返回: [P * dag_k, C, H, W] (cond), parent_indices [P * dag_k]
    """
    P = z_cond_parents.shape[0]
    device = z_cond_parents.device

    # 重复每个 parent dag_k 次
    # z_expanded: [P * dag_k, C, H, W]
    z_expanded = z_cond_parents.repeat_interleave(dag_k, dim=0)

    # 对于 SDE 采样，不同的前向会产生不同结果（因为有随机噪声）
    # 但 diffusion.p_sample 内部会添加噪声，所以直接调用即可

    # CFG 拼接
    total_B = P * dag_k
    z_full = torch.cat([z_expanded, z_expanded], dim=0)  # [2 * P * dag_k, C, H, W]

    y_cond = model_kwargs_base["y_cond"].expand(total_B)
    y_null = model_kwargs_base["y_null"].expand(total_B)
    y_full = torch.cat([y_cond, y_null], dim=0)

    model_kwargs = dict(y=y_full, cfg_scale=cfg_scale)
    t_tensor = torch.full((2 * total_B,), t, device=device, dtype=torch.long)

    out = diffusion.p_sample(model_fn, z_full, t_tensor, clip_denoised=False, model_kwargs=model_kwargs)

    if isinstance(out, dict):
        sample = out.get("sample", out.get("x_prev", out.get("prev_sample")))
    else:
        sample = out

    z_children_cond = sample[:total_B]  # [P * dag_k, C, H, W]

    # parent indices: [0,0,0, 1,1,1, 2,2,2, ...] for dag_k=3
    parent_indices = torch.arange(P, device=device).repeat_interleave(dag_k)

    return z_children_cond, parent_indices


def load_mst_assets(mst_root, sel_class, device, dtype=torch.float32):
    """加载 MST 资源并预处理到 GPU"""
    cls_dir = os.path.join(mst_root, sel_class)

    mst_nodes = torch.load(os.path.join(cls_dir, "mst_nodes.pt"), map_location=device)
    mst_edges = torch.load(os.path.join(cls_dir, "mst_edges.pt"), map_location=device)

    pca_proj = None
    pca_scale = None
    proj_path = os.path.join(cls_dir, "pca_proj.pt")
    scale_path = os.path.join(cls_dir, "pca_scale.pt")
    pca_mean_path = os.path.join(cls_dir, "pca_mean.pt")




    if os.path.exists(proj_path):
        pca_proj = torch.load(proj_path, map_location=device)
    if os.path.exists(scale_path):
        pca_scale = torch.load(scale_path, map_location=device)
    if os.path.exists(pca_mean_path):
        pca_mean = torch.load(pca_mean_path, map_location=device)
        

    return mst_nodes, mst_edges, pca_proj, pca_scale, pca_mean


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 类别标签
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

    phase = max(0, args.phase)
    cls_from = args.nclass * phase
    cls_to = args.nclass * (phase + 1)
    sel_classes = sel_classes[cls_from:cls_to]
    class_labels = [all_classes.index(c) for c in sel_classes]

    # 加载模型
    latent_size = args.image_size // 8
    model = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes).to(device)
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    for class_label, sel_class in zip(class_labels, sel_classes):
        os.makedirs(os.path.join(args.save_dir, sel_class), exist_ok=True)

        # 加载 MST 资源
        mst = None
        if args.use_dag:
            mst_nodes, mst_edges, pca_proj, pca_scale, pca_mean= load_mst_assets(
                args.mst_root, sel_class, device
            )
            mst = MSTGuidanceFast(
                mst_nodes=mst_nodes,
                mst_edges=mst_edges,
                lambda_d=args.lambda_d,
                lambda_dir=args.lambda_dir,
                pca_proj=pca_proj,
                pca_scale=pca_scale,
                pca_mean=pca_mean,
                device=device,
            )
            #debug
            print("mean loaded?", mst.mean is not None, "mean shape:", None if mst.mean is None else mst.mean.shape)


        # model_kwargs 基础配置
        model_kwargs_base = {
            "y_cond": torch.tensor([class_label], device=device),
            "y_null": torch.tensor([args.num_classes], device=device),  # 1000 for null
        }

        for shift in tqdm(range(args.num_samples), desc=f"Class {sel_class}"):
            # 初始噪声
            z = torch.randn(1, 4, latent_size, latent_size, device=device)

            # --------------------------
            # Vanilla sampling
            # --------------------------
            if not args.use_dag:
                z_full = torch.cat([z, z], 0)
                y_full = torch.cat([
                    torch.tensor([class_label], device=device),
                    torch.tensor([args.num_classes], device=device)
                ], 0)
                model_kwargs = dict(y=y_full, cfg_scale=args.cfg_scale)

                samples = diffusion.p_sample_loop(
                    model.forward_with_cfg, z_full.shape, z_full,
                    clip_denoised=False, model_kwargs=model_kwargs,
                    progress=False, device=device
                )
                samples, _ = samples.chunk(2, dim=0)
                samples = vae.decode(samples.float() / 0.18215).sample

            # --------------------------
            # 优化版 DAG + MST guidance
            # --------------------------
            else:
                T = args.num_sampling_steps - 1

                graph = SamplingGraphFast()
                # 存储时 squeeze 掉 batch 维度: [1,C,H,W] -> [C,H,W]
                root_id = graph.add_node(t=T, z=z.squeeze(0), cost=0.0)

                for t in range(T, 0, -1):
                    parent_ids = graph.nodes_at_time(t)
                    if len(parent_ids) == 0:
                        raise RuntimeError(f"Empty layer at t={t}")

                    # 收集所有 parent latents
                    z_parents = torch.stack([graph.nodes[pid].z for pid in parent_ids], dim=0)
                    # z_parents: [P, C, H, W]

                    # 判断是否应用 MST
                    if args.guide_early:
                        # 新策略: 早期应用 MST (t > guide_end_t), 后期自由采样
                        use_mst_this_step = (t > args.guide_end_t)
                    else:
                        # 原策略: 后期应用 MST (t <= guide_start_t), 早期自由采样
                        use_mst_this_step = (t <= args.guide_start_t)

                    # ========== 自由采样阶段 (不应用 MST) ==========
                    if not use_mst_this_step:
                        # 批量单步去噪
                        z_children = batched_diffusion_step(
                            diffusion, model.forward_with_cfg,
                            z_parents, t, model_kwargs_base, args.cfg_scale
                        )
                        # z_children: [P, C, H, W]

                        # cost 全为 0
                        costs = torch.zeros(len(parent_ids), device=device)

                        graph.add_nodes_and_prune(
                            t=t-1,
                            z_batch=z_children,
                            costs=costs,
                            parent_ids=parent_ids,
                            max_nodes=args.beam_width
                        )

                    # ========== MST 引导阶段 ==========
                    else:
                        # 每个 parent 生成 dag_k 个子节点
                        z_children, parent_idx = batched_diffusion_step_with_noise_variants(
                            diffusion, model.forward_with_cfg,
                            z_parents, t, model_kwargs_base, args.cfg_scale, args.dag_k
                        )
                        # z_children: [P * dag_k, C, H, W]
                        # parent_idx: [P * dag_k]

                        # 获取对应的 z_t (用于计算方向 loss)
                        z_t_expanded = z_parents[parent_idx]  # [P * dag_k, C, H, W]

                        # 批量计算 MST edge cost
                        costs = mst.edge_cost_batch(z_t_expanded, z_children)  # [P * dag_k]

                        # 转换 parent_ids 列表
                        expanded_parent_ids = [parent_ids[idx.item()] for idx in parent_idx]

                        graph.add_nodes_and_prune(
                            t=t-1,
                            z_batch=z_children,
                            costs=costs,
                            parent_ids=expanded_parent_ids,
                            max_nodes=args.beam_width
                        )

                # 获取最优路径
                best_path = graph.get_best_path(end_time=0)
                if not best_path:
                    raise RuntimeError("No path found")

                z_final = best_path[-1].z.unsqueeze(0)  # [1, C, H, W]
                samples = vae.decode(z_final.float() / 0.18215).sample

            # 保存图像
            for image_index, image in enumerate(samples):
                save_image(
                    image,
                    os.path.join(args.save_dir, sel_class, f"{image_index + shift + args.total_shift}.png"),
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
    parser.add_argument("--ckpt", type=str, default="pretrained_models/DiT-XL-2-256x256.pt",
                        help="Path to DiT checkpoint")
    parser.add_argument("--spec", type=str, default='woof')
    parser.add_argument("--save-dir", type=str, default='./logs/mst_dag_fast')
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--total-shift", type=int, default=0)
    parser.add_argument("--nclass", type=int, default=10)
    parser.add_argument("--phase", type=int, default=0)

    # DAG 参数
    parser.add_argument("--use-dag", action="store_true", default=True)
    parser.add_argument("--no-dag", action="store_false", dest="use_dag")
    parser.add_argument("--mst-root", type=str, default="./mst")
    parser.add_argument("--dag-k", type=int, default=3, help="children per parent")
    parser.add_argument("--beam-width", type=int, default=16, help="max nodes per layer")

    # MST 时间步策略
    # 原策略: --guide-start-t 25 表示 t <= 25 时应用 MST (后期)
    # 新策略: --guide-early 表示 t > guide-end-t 时应用 MST (早期)
    parser.add_argument("--guide-start-t", type=int, default=25, help="[旧策略] apply MST when t <= this")
    parser.add_argument("--guide-end-t", type=int, default=20, help="[新策略] apply MST when t > this (early stage)")
    parser.add_argument("--guide-early", action="store_true", help="use early-stage MST guidance (recommended for GRPO)")

    # MST cost 权重
    parser.add_argument("--lambda-d", type=float, default=1.0)
    parser.add_argument("--lambda-dir", type=float, default=0.0)

    # 性能优化 (暂时禁用 fp16，有兼容性问题)
    # parser.add_argument("--use-fp16", action="store_true", help="use fp16 for faster inference")

    args = parser.parse_args()
    main(args)
