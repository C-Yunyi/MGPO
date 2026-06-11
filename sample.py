"""
Sample new images from a pre-trained DiT.
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


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Labels to condition the model
    with open('./misc/class_indices.txt', 'r') as fp:
        all_classes = fp.readlines()
    all_classes = [class_index.strip() for class_index in all_classes]
    if args.spec == 'woof':
        file_list = './misc/class_woof.txt'
    elif args.spec == 'nette':
        file_list = './misc/class_nette.txt'
    elif args.spec == 'idc':
        file_list = './misc/idc_list.txt'
    else:
        file_list = './misc/class_indices.txt'
    with open(file_list, 'r') as fp:
        sel_classes = fp.readlines()

    phase = max(0, args.phase)
    cls_from = args.nclass * phase
    cls_to = args.nclass * (phase + 1)
    sel_classes = sel_classes[cls_from:cls_to]
    sel_classes = [sel_class.strip() for sel_class in sel_classes]
    class_labels = []
    
    for sel_class in sel_classes:
        class_labels.append(all_classes.index(sel_class))

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    ckpt = find_model(ckpt_path)

    print("[ckpt] path:", ckpt_path)
    print("[ckpt] type:", type(ckpt))

    if isinstance(ckpt, dict):
        print("[ckpt] keys:", list(ckpt.keys()))

    # 选择正确的权重分支
    if isinstance(ckpt, dict):
        if "ema" in ckpt:
            print("[ckpt] using: ema")
            sd = ckpt["ema"]
        elif "model" in ckpt:
            print("[ckpt] using: model")
            sd = ckpt["model"]
        elif "state_dict" in ckpt:
            print("[ckpt] using: state_dict")
            sd = ckpt["state_dict"]
        else:
            print("[ckpt] using: raw dict as state_dict")
            sd = ckpt
    else:
        sd = ckpt

    # 去掉常见前缀
    new_sd = {}
    for k, v in sd.items():
        k2 = k
        if k2.startswith("module."):
            k2 = k2[len("module."):]
        if k2.startswith("_orig_mod."):
            k2 = k2[len("_orig_mod."):]
        new_sd[k2] = v
    sd = new_sd

    # 加载并打印匹配情况
    missing, unexpected = model.load_state_dict(sd, strict=False)

    # print("[ckpt] missing:", len(missing))
    # print("[ckpt] unexpected:", len(unexpected))
    # print("[ckpt] first 20 missing:", missing[:20])
    # print("[ckpt] first 20 unexpected:", unexpected[:20])

    if len(missing) > 50:
        raise RuntimeError("Too many missing keys. Model config likely mismatched with checkpoint.")

    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    batch_size = 1

    for class_label, sel_class in zip(class_labels, sel_classes):
        os.makedirs(os.path.join(args.save_dir, sel_class), exist_ok=True)
        for shift in tqdm(range(args.num_samples // batch_size)):
            # Create sampling noise:
            z = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
            y = torch.tensor([class_label], device=device)

            # Setup classifier-free guidance:
            # Setup classifier-free guidance:
            print("[DEBUG] z before cat:", z.shape)

            z_in = torch.cat([z, z], 0)
            print("[DEBUG] z_in:", z_in.shape)
            print("[DEBUG] shape passed to loop:", z_in.shape)

            y_null = torch.tensor([1000] * batch_size, device=device)
            y_in = torch.cat([y, y_null], 0)
            print("[DEBUG] y:", y.shape, "y_in:", y_in.shape)

            model_kwargs = dict(y=y_in, cfg_scale=args.cfg_scale)

            # Sample images:
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg,
                z_in.shape,
                z_in,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=device,
            )

            print("[DEBUG] samples out:", samples.shape)

            samples_only, _ = samples.chunk(2, dim=0)
            print("[latent] min/max/mean/std:",
                  samples_only.min().item(),
                  samples_only.max().item(),
                  samples_only.mean().item(),
                  samples_only.std().item())
            samples = samples_only

            samples = vae.decode(samples / 0.18215).sample

            # Save and display images:
            for image_index, image in enumerate(samples):
                save_image(image, os.path.join(args.save_dir, sel_class,
                                               f"{image_index + shift * batch_size + args.total_shift}.png"), normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default="/root/autodl-tmp/ckpt_step_500.pt",
                        help="None")
    parser.add_argument("--spec", type=str, default='woof', help='specific subset for generation')
    parser.add_argument("--save-dir", type=str, default='../autodl-tmp/test_500', help='the directory to put the generated images')
    parser.add_argument("--num-samples", type=int, default=10, help='the desired IPC for generation')
    parser.add_argument("--total-shift", type=int, default=0, help='index offset for the file name')
    parser.add_argument("--nclass", type=int, default=10, help='the class number for generation')
    parser.add_argument("--phase", type=int, default=0, help='the phase number for generating large datasets')
    args = parser.parse_args()
    main(args)
