"""
Sample new images from a pre-trained DiT (with optional MST-DAG guided sampling).
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

# MST-DAG
from mst_guidance import MSTGuidance, SamplingGraph, topk_shortest_paths


@torch.no_grad()
def diffusion_step(diffusion, model_fn, z, t, model_kwargs, clip_denoised=False):
    """
    One reverse diffusion step: z_t -> z_{t-1}

    Handles common return formats:
      - dict with key "sample"
      - raw tensor
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
        # if already a vector, ensure correct size
        assert t.shape == (B,), f"t.shape {tuple(t.shape)} != ({B},)"

    out = diffusion.p_sample(model_fn, z, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
    if isinstance(out, dict):
        if "sample" in out:
            return out["sample"]
        # some impls use "x_prev" or similar
        for k in ["x_prev", "prev_sample", "pred_prev_sample"]:
            if k in out:
                return out[k]
        raise KeyError(f"p_sample returned dict without known keys: {list(out.keys())}")
    return out


def load_mst_assets(mst_root, sel_class, device):
    """
    Loads per-class MST assets.
    mst_nodes is moved to device (for cdist).
    pca_proj/scale left on CPU; MSTGuidance._proj will move to z's device/dtype.
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

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model
    latent_size = args.image_size // 8
    model = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes).to(device)
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    ckpt = find_model(ckpt_path)
    if isinstance(ckpt, dict):
        if "ema" in ckpt:
            state_dict = ckpt["ema"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    batch_size = 1

    for class_label, sel_class in zip(class_labels, sel_classes):
        os.makedirs(os.path.join(args.save_dir, sel_class), exist_ok=True)

        # Load MST assets for this class (only if DAG enabled)
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
            # DAG / Beam + MST guidance
            # --------------------------
            else:
                T = args.num_sampling_steps - 1  # timesteps index [T..0]

                graph = SamplingGraph()
                root_id = graph.add_node(t=T, z=z_full, cost=0.0)

                for t in range(T, 0, -1):
                    parents = graph.nodes_at_time(t)
                    if len(parents) == 0:
                        raise RuntimeError(f"Empty layer at t={t}. Increase --beam-width or reduce --dag-k.")

                    # early: keep beam, but do NOT apply mst cost (cheap & avoids noisy guidance)
                    if t > args.guide_start_t:
                        cand = []
                        for pid in parents:
                            z_t_full = graph.nodes[pid].z
                            z_prev_full = diffusion_step(diffusion, model.forward_with_cfg, z_t_full, t, model_kwargs)
                            cand.append((z_prev_full, pid, 0.0))
                        graph.add_nodes_pruned(t=t-1, candidates=cand, max_nodes=args.beam_width)
                        continue

                    # late: expand K children per parent and score edges
                    candidates = []
                    for pid in parents:
                        z_t_full = graph.nodes[pid].z
                        z_t_real = z_t_full[:batch_size]  # cond part for cost

                        for _ in range(args.dag_k):
                            z_prev_full = diffusion_step(diffusion, model.forward_with_cfg, z_t_full, t, model_kwargs)
                            z_prev_real = z_prev_full[:batch_size]

                            # edge cost from MSTGuidance (mean over batch)
                            edge_cost = mst.edge_cost(z_t_real, z_prev_real).mean().item()
                            candidates.append((z_prev_full, pid, edge_cost))

                    graph.add_nodes_pruned(t=t-1, candidates=candidates, max_nodes=args.beam_width)

                # pick best path
                paths = topk_shortest_paths(graph, start_time=T, end_time=0, K=1)
                if not paths:
                    raise RuntimeError("No path found. Check MST assets and DAG params.")
                best_path = paths[0]
                z_final_full = best_path[-1].z
                z_final = z_final_full[:batch_size]

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
    parser.add_argument("--ckpt", type=str, default="/root/MinimaxDiffusion/pretrained_models/DiT-XL-2-256x256.pt",
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--spec", type=str, default='woof', help='specific subset for generation')
    parser.add_argument("--save-dir", type=str, default='../logs/test', help='the directory to put the generated images')
    parser.add_argument("--num-samples", type=int, default=10, help='the desired IPC for generation')
    parser.add_argument("--total-shift", type=int, default=0, help='index offset for the file name')
    parser.add_argument("--nclass", type=int, default=10, help='the class number for generation')
    parser.add_argument("--phase", type=int, default=0, help='the phase number for generating large datasets')

    # DAG guidance flags
    parser.add_argument("--use-dag", action="store_true", help="enable MST-DAG guided sampling (beam search)")
    parser.add_argument("--mst-root", type=str, default="./mst", help="root dir containing per-class mst assets")
    parser.add_argument("--dag-k", type=int, default=3, help="children samples per parent per step")
    parser.add_argument("--beam-width", type=int, default=16, help="max nodes per timestep layer")
    parser.add_argument("--guide-start-t", type=int, default=25, help="apply MST costs when t <= guide_start_t")

    # MST cost weights
    parser.add_argument("--lambda-d", type=float, default=1.0, help="weight for MST node distance cost")
    parser.add_argument("--lambda-dir", type=float, default=0.0, help="weight for MST direction alignment cost")

    args = parser.parse_args()
    main(args)
