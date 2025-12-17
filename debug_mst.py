import torch
from diffusion import create_diffusion
from models import DiT_models
from download import find_model
from mst_guidance import MSTGuidance

# =========================
# CONFIG（刻意很小）
# =========================
MODEL = "DiT-XL/2"
IMAGE_SIZE = 256
NUM_CLASSES = 1000
CLASS_LABEL = 0
T = 20                  # 少步数，快
M = 3                   # 每步 3 个分支
CFG_SCALE = 4.0
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# SETUP
# =========================
torch.manual_seed(SEED)
latent_size = IMAGE_SIZE // 8

model = DiT_models[MODEL](
    input_size=latent_size,
    num_classes=NUM_CLASSES,
).to(DEVICE)
state_dict = find_model(f"DiT-XL-2-{IMAGE_SIZE}x{IMAGE_SIZE}.pt")
model.load_state_dict(state_dict, strict=False)
model.eval()

diffusion = create_diffusion(str(T))

# CFG conditioning
y = torch.tensor([CLASS_LABEL], device=DEVICE)
y_null = torch.tensor([NUM_CLASSES], device=DEVICE)
y_cfg = torch.cat([y, y_null], dim=0)
model_kwargs = dict(y=y_cfg, cfg_scale=CFG_SCALE)

# =========================
# LOAD MST (你提前算好的)
# =========================
# 这里假设你已经有：
# mst_nodes: [N, D] tensor
# mst_edges: [E, 2] tensor
mst_nodes = torch.load("mst_nodes.pt").to(DEVICE)
mst_edges = torch.load("mst_edges.pt").to(DEVICE)
pca_proj = torch.load("pca_proj.pt")
pca_scale = torch.load("pca_scale.pt")


mst = MSTGuidance(
    mst_nodes=mst_nodes,
    mst_edges=mst_edges,
    lambda_q=0.0,        # STEP 1 先关掉
    lambda_d=1.0,
    lambda_dir=0.5,
    pca_proj=pca_proj,
    pca_scale=pca_scale, 
    
)

# =========================
# INITIAL NOISE
# =========================
z = torch.randn(1, 4, latent_size, latent_size, device=DEVICE)
z = torch.cat([z, z], dim=0)  # for CFG

# =========================
# VANILLA TRAJECTORY (对照)
# =========================
z_vanilla = z.clone()
vanilla_dist = []

with torch.no_grad():
    for t in reversed(range(T)):
        z_vanilla = diffusion.p_sample_step(
            model.forward_with_cfg,
            z_vanilla,
            t,
            clip_denoised=False,
            model_kwargs=model_kwargs,
        )
        z_real, _ = z_vanilla.chunk(2, dim=0)
        vanilla_dist.append(mst.mst_distance(z_real).item())

# =========================
# MST-GUIDED (GREEDY)
# =========================
z_guided = z.clone()
guided_dist = []

with torch.no_grad():
    for t in reversed(range(T)):
        best_cost = float("inf")
        best_z = None

        for m in range(M):
            eps = torch.randn_like(z_guided)
            z_prev = diffusion.p_sample_step(
                model.forward_with_cfg,
                z_guided,
                t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                noise=eps,
            )

            cost = mst.edge_cost(z_guided[:1], z_prev[:1]).item()

            if cost < best_cost:
                best_cost = cost
                best_z = z_prev

        z_guided = best_z
        z_real, _ = z_guided.chunk(2, dim=0)
        guided_dist.append(mst.mst_distance(z_real).item())

# =========================
# LOG RESULT
# =========================
print("t | vanilla_dist | guided_dist")
for i in range(T):
    print(f"{T-1-i:2d} | {vanilla_dist[i]:.4f} | {guided_dist[i]:.4f}")
