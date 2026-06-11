import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ====== [新增] 字体 & 等距 minor grid 支持 ======
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MultipleLocator



plt.style.use("seaborn-v0_8-whitegrid")

# ====== [新增] Times New Roman（更稳：serif fallback）======
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
mpl.rcParams["mathtext.fontset"] = "stix"

mpl.rcParams["font.size"] = 25        # 全局默认字体
mpl.rcParams["axes.labelsize"] = 27   # 坐标轴标签
mpl.rcParams["xtick.labelsize"] = 23  # x轴刻度
mpl.rcParams["ytick.labelsize"] = 23  # y轴刻度
mpl.rcParams["legend.fontsize"] = 27  # legend

# ====== [新增] 等距网格：major + minor（统一风格）======
# def apply_equal_grid(ax):
#     # 让 minor tick 均匀分割（每个 major 间隔分成 2 份 -> 产生等距小网格）
#     ax.xaxis.set_minor_locator(AutoMinorLocator(2))
#     ax.yaxis.set_minor_locator(AutoMinorLocator(2))

#     # major grid
#     ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.55)
#     # minor grid（更淡）
#     ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.35)

# from matplotlib.ticker import MultipleLocator

# def apply_equal_grid(ax):

#     # major grid
#     ax.grid(True, which="major",
#             linestyle="--",
#             linewidth=0.6,
#             alpha=0.55)

#     # minor grid 固定密度
#     ax.minorticks_on()

#     ax.xaxis.set_minor_locator(MultipleLocator(
#         (ax.get_xlim()[1] - ax.get_xlim()[0]) / 10
#     ))

#     ax.yaxis.set_minor_locator(MultipleLocator(
#         (ax.get_ylim()[1] - ax.get_ylim()[0]) / 10
#     ))

#     ax.grid(True,
#             which="minor",
#             linestyle=":",
#             linewidth=0.5,
#             alpha=0.35)

from matplotlib.ticker import AutoMinorLocator, MaxNLocator

def apply_equal_grid(ax):
    # 让四张子图的 major 网格密度更统一（可选但很稳）
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # 每个 major 间隔均分成 2 份 -> minor 网格等距、且会随lim自动更新
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.55)
    ax.grid(True, which="minor", linestyle=":",  linewidth=0.5, alpha=0.35)

# ========================
# 数据
# ========================

# Ex1(step)
x_step = np.array([150,350,550,750,950,1150,1350,1550,1750,1950,2150])
y_step_10 = np.array([35.7,37.0,40.9,36.2,37.7,35.4,37.6,37.3,37.3,37.1,37.4])
y_step_20 = np.array([39.9,40.3,44.5,39.8,42.0,40.9,42.2,42.7,40.8,41.8,42.9])
y_step_50 = np.array([51.5,50.8,57.9,50.4,51.9,50.5,52.8,53.6,51.0,49.8,52.7])

# Ex2(batch)
x_batch = np.array([1,2,3,4])
y_batch_10 = np.array([37.3,36.8,36.5,40.9])
y_batch_20 = np.array([40.5,43.2,40.6,44.5])
y_batch_50 = np.array([51.1,53.8,51.0,57.9])

# Ex3(group)
x_group = np.array([2,3,4,5,6,7,8])
y_group_10 = np.array([37.6,37.2,36.7,37.4,37.2,36.9,40.9])
y_group_20 = np.array([40.2,42.1,42.2,42.5,42.6,42.0,44.5])
y_group_50 = np.array([51.4,52.8,53.3,54.2,53.1,53.0,57.9])

# Ex4 reward sweep (Cls -> r_disc, Mst -> r_geo)
x_weight = np.array([0.2,0.4,0.6,0.8])

disc10 = np.array([36.7,37.8,37.1,37.6])
geo10  = np.array([37.8,37.6,37.4,36.9])

disc20 = np.array([40.8,40.6,41.1,41.3])
geo20  = np.array([42.4,42.8,42.5,42.5])

disc50 = np.array([51.4,50.6,51.1,51.1])
geo50  = np.array([52.5,52.3,52.8,53.4])

# ========================
# 配色（低饱和度）
# ========================
C10 = "#55a868"
C20 = "#f18d38"
C50 = "#4c72b0"
C_MARK = "#d95f5f"   # 竖线+星标

SHADE = 0.45
LEGEND_FONTSIZE = 20

# ========================
# 工具：在最优点画“接地到星星”的竖虚线 + 星标（不穿星、不出界）
# ========================
def mark_best(ax, x, y, color=C_MARK, prefer="max", zorder=10, star_offset=1.5):
    y = np.asarray(y, dtype=float)
    idx = int(np.argmax(y)) if prefer == "max" else int(np.argmin(y))

    x_best = x[idx]
    y_best = y[idx]
    y_star = y_best + star_offset

    ax.autoscale_view()
    y_bottom, y_top = ax.get_ybound()

    if y_star >= y_top:
        ax.set_ylim(y_bottom, y_star + 0.5)
        y_bottom, y_top = ax.get_ybound()

    ax.vlines(
        x_best, y_bottom, y_star,
        colors=color, linestyles="--", linewidth=1.2,
        alpha=0.9, zorder=zorder-1
    )

    ax.scatter(
        [x_best], [y_star],
        marker="*", s=320,   # 星星大小你可在这里调
        color=color, edgecolor="white", linewidth=0.8,
        zorder=zorder
    )

    return x_best, y_best


# ========================
# 子图函数：三条曲线 + 阴影 + 自动标最优
# ========================
def plot_three(ax, x, y10, y20, y50, xlabel, ylabel,
               ylim=None, xticks=None,
               best_on="ipc20", prefer="max"):

    x = np.asarray(x)
    y10 = np.asarray(y10, dtype=float)
    y20 = np.asarray(y20, dtype=float)
    y50 = np.asarray(y50, dtype=float)

    e10 = np.full_like(y10, SHADE)
    e20 = np.full_like(y20, SHADE)
    e50 = np.full_like(y50, SHADE)

    ax.plot(x, y10, marker="o", color=C10, linewidth=2)
    ax.fill_between(x, y10-e10, y10+e10, color=C10, alpha=0.20, linewidth=0)

    ax.plot(x, y20, marker="o", color=C20, linewidth=2)
    ax.fill_between(x, y20-e20, y20+e20, color=C20, alpha=0.20, linewidth=0)

    ax.plot(x, y50, marker="o", color=C50, linewidth=2)
    ax.fill_between(x, y50-e50, y50+e50, color=C50, alpha=0.20, linewidth=0)

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if ylim is not None:
        ax.set_ylim(*ylim)
    if xticks is not None:
        ax.set_xticks(xticks)

    # ====== [替换] 等距 grid ======
    apply_equal_grid(ax)

    # 选择用哪条曲线决定“最优点”
    if best_on == "ipc10":
        y_best_ref = y10
    elif best_on == "ipc50":
        y_best_ref = y50
    elif best_on == "mean":
        y_best_ref = (y10 + y20 + y50) / 3.0
    else:
        y_best_ref = y50  # 建议默认就 IPC=20（更符合你的调用 best_on="ipc20"）

    mark_best(ax, x, y_best_ref, color=C_MARK, prefer=prefer)


def plot_ex4(ax, x, d10, g10, d20, g20, d50, g50,
             ylim=(35, 60),
             best_on="geo_ipc50",
             prefer="max"):

    x = np.asarray(x, dtype=float)
    shade = 0.25
    e = np.full_like(x, shade)

    # r_disc 实线
    ax.plot(x, d10, marker="o", color=C10, linewidth=2)
    ax.fill_between(x, d10-e, d10+e, color=C10, alpha=0.15, linewidth=0)

    ax.plot(x, d20, marker="o", color=C20, linewidth=2)
    ax.fill_between(x, d20-e, d20+e, color=C20, alpha=0.15, linewidth=0)

    ax.plot(x, d50, marker="o", color=C50, linewidth=2)
    ax.fill_between(x, d50-e, d50+e, color=C50, alpha=0.15, linewidth=0)

    # r_geo 虚线
    ax.plot(x, g10, marker="o", linestyle="--", color=C10, linewidth=2)
    ax.fill_between(x, g10-e, g10+e, color=C10, alpha=0.10, linewidth=0)

    ax.plot(x, g20, marker="o", linestyle="--", color=C20, linewidth=2)
    ax.fill_between(x, g20-e, g20+e, color=C20, alpha=0.10, linewidth=0)

    ax.plot(x, g50, marker="o", linestyle="--", color=C50, linewidth=2)
    ax.fill_between(x, g50-e, g50+e, color=C50, alpha=0.10, linewidth=0)

    ax.set_xlabel("Reward Weight", fontsize=16)
    ax.set_ylabel("Accuracy (%)", fontsize=16)
    ax.set_xticks([0.2,0.4,0.6,0.8])

    ax.set_ylim(*ylim)
    ax.set_yticks([35, 40, 45, 50, 55, 60])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ====== [替换] 等距 grid ======
    apply_equal_grid(ax)

    # 选择 Ex4 的“最优参考曲线”
    if best_on == "disc_ipc10":
        y_ref = np.asarray(d10, dtype=float)
    elif best_on == "disc_ipc20":
        y_ref = np.asarray(d20, dtype=float)
    elif best_on == "disc_ipc50":
        y_ref = np.asarray(d50, dtype=float)
    elif best_on == "geo_ipc10":
        y_ref = np.asarray(g10, dtype=float)
    elif best_on == "geo_ipc20":
        y_ref = np.asarray(g20, dtype=float)
    else:
        y_ref = np.asarray(g50, dtype=float)  # 默认 geo_ipc50

    mark_best(ax, x, y_ref, color=C_MARK, prefer=prefer)


# ========================
# 2×2 图（更大尺寸）
# ========================
fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=180)


# fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)

plot_three(
    axes[0,0], x_step, y_step_10, y_step_20, y_step_50,
    # axes[0], x_step, y_step_10, y_step_20, y_step_50,
    xlabel="Step", ylabel="Accuracy (%)",
    ylim=(35, 60),
    best_on="ipc20",
    prefer="max"
)

plot_three(
    axes[0,1], x_batch, y_batch_10, y_batch_20, y_batch_50,
    # axes[1], x_batch, y_batch_10, y_batch_20, y_batch_50,
    xlabel="Batch Size", ylabel="Accuracy (%)",
    ylim=(35, 60),
    xticks=[1,2,3,4],
    best_on="ipc20",
    prefer="max"
)

plot_three(
    axes[1,0], x_group, y_group_10, y_group_20, y_group_50,
    #axes[2], x_group, y_group_10, y_group_20, y_group_50,
    xlabel="Group Size", ylabel="Accuracy (%)",
    ylim=(35, 60),
    xticks=[2,3,4,5,6,7,8],
    best_on="ipc20",
    prefer="max"
)

plot_ex4(
    axes[1,1], x_weight,
    #axes[3], x_weight,
    disc10, geo10,
    disc20, geo20,
    disc50, geo50,
    ylim=(35, 60),
    best_on="geo_ipc50",
    prefer="max"
)

# ========================
# 子图标签 (a)(b)(c)(d)
# ========================
labels = ["(a)", "(b)", "(c)", "(d)"]
for ax, lab in zip(axes.flatten(), labels):
    ax.text(0.5, -0.24, lab, transform=ax.transAxes,
            ha="center", va="center", fontsize=18)

for ax in axes.flatten():
    ax.margins(y=0.12)

# ✅ 最关键：所有ylim/xlim都稳定后，再统一刷新网格
for ax in axes.flatten():
    apply_equal_grid(ax)

# ========================
# 全局 legend（更大字体）
# ========================
ipc_handles = [
    Line2D([0],[0], color=C10, lw=2, marker="o", label="IPC=10"),
    Line2D([0],[0], color=C20, lw=2, marker="o", label="IPC=20"),
    Line2D([0],[0], color=C50, lw=2, marker="o", label="IPC=50"),
]

best_handles = [
    Line2D([0],[0], color=C_MARK, lw=1.2, linestyle="--", label="Best x"),
    Line2D([0],[0], color=C_MARK, marker="*", lw=0, markersize=12, label="Best point"),
]

fig.legend(
    handles=ipc_handles + best_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.97),
    ncol=5,
    frameon=False,
    columnspacing=1.0,
    prop={"size":LEGEND_FONTSIZE , "weight": "bold"},  # 替换原来的 fontsize=LEGEND_FONTSIZE
    handlelength=2.0, 
)

plt.tight_layout(rect=[0, 0, 1, 0.91])
plt.subplots_adjust(hspace=0.45)  # 在tight_layout之后加这行

# ========================
# 保存
# ========================
save_dir = "/root/autodl-tmp/figs"
os.makedirs(save_dir, exist_ok=True)

out_pdf = os.path.join(save_dir, "init_update.pdf")
out_png = os.path.join(save_dir, "init_update.png")

plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, dpi=300, bbox_inches="tight")

print("[Saved]", out_pdf)
print("[Saved]", out_png)

plt.show()