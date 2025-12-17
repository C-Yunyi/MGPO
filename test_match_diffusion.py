import torch
from diffusion import create_diffusion
from models import DiT_models
from download import find_model

# -------------------------
# config（按你项目实际改）
# -------------------------
MODEL_NAME = "DiT-XL/2"
IMAGE_SIZE = 256
NUM_CLASSES = 1000
NUM_STEPS = 50
CFG_SCALE = 4.0
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# setup
# -------------------------
torch.manual_seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

latent_size = IMAGE_SIZE // 8

model = DiT_models[MODEL_NAME](
    input_size=latent_size,
    num_classes=NUM_CLASSES,
).to(DEVICE)

ckpt_path = f"DiT-XL-2-{IMAGE_SIZE}x{IMAGE_SIZE}.pt"
state_dict = find_model(ckpt_path)
model.load_state_dict(state_dict, strict=False)
model.eval()

diffusion = create_diffusion(str(NUM_STEPS))

# -------------------------
# conditioning (CFG)
# -------------------------
batch_size = 1
class_label = 0

y = torch.tensor([class_label], device=DEVICE)
y_null = torch.tensor([NUM_CLASSES], device=DEVICE)

y_cfg = torch.cat([y, y_null], dim=0)
model_kwargs = dict(y=y_cfg, cfg_scale=CFG_SCALE)

# -------------------------
# initial noise
# -------------------------
torch.manual_seed(SEED)
z0 = torch.randn(batch_size, 4, latent_size, latent_size, device=DEVICE)
z0 = torch.cat([z0, z0], dim=0)  # for CFG

# ============================================================
# 1️⃣ 原版 p_sample_loop
# ============================================================
torch.manual_seed(SEED)
with torch.no_grad():
    out_loop = diffusion.p_sample_loop(
        model.forward_with_cfg,
        z0.shape,
        z0.clone(),
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=False,
        device=DEVICE,
    )

out_loop, _ = out_loop.chunk(2, dim=0)

# ============================================================
# 2️⃣ 手写 for-loop + p_sample_step
# ============================================================
torch.manual_seed(SEED)
x = z0.clone()

with torch.no_grad():
    for t in reversed(range(diffusion.num_timesteps)):
        x = diffusion.p_sample_step(
            model.forward_with_cfg,
            x,
            t,
            clip_denoised=False,
            model_kwargs=model_kwargs,
        )

out_step, _ = x.chunk(2, dim=0)

# ============================================================
# compare
# ============================================================
diff = (out_loop - out_step).abs()

print("max abs diff:", diff.max().item())
print("mean abs diff:", diff.mean().item())
