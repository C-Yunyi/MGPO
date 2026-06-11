import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

# ======================
# Style
# ======================
plt.style.use("seaborn-v0_8-whitegrid")

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
mpl.rcParams["mathtext.fontset"] = "stix"


mpl.rcParams["font.size"] = 20        # 全局默认字体
# mpl.rcParams["axes.labelsize"] = 27   # 坐标轴标签
# mpl.rcParams["legend.fontsize"] = 27  # legend

# ======================
# 原来的颜色配置
# ======================
C10 = "#55a868"
C20 = "#f18d38"
C50 = "#4c72b0"
C_MARK = "#d95f5f"

MAIN_COLOR = C50
FILL_COLOR = C50
BAR_COLOR = C50
LEGEND_FONTSIZE = 18

# ======================
# Data
# ======================

# (a) KL beta
x_kl = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20])
y_kl = np.array([37.2, 37.2, 37.6, 38.1, 36.7, 37.6, 37.6, 37.6, 37.4, 37.5])
std_kl = np.zeros_like(y_kl)

# (b) MST direction
x_mst_dir = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
y_mst_dir = np.array([37.2, 37.1, 37.4, 37.4, 36.6])
std_mst_dir = np.zeros_like(y_mst_dir)

# (c) MST distance
x_mst_dist = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
y_mst_dist = np.array([37.7, 37.9, 36.6, 37.0, 36.9])
std_mst_dist = np.zeros_like(y_mst_dist)

# (d) Reward model
x_reward_labels = ["ResNet18", "ResNet50", "ResNet101"]
y_reward = np.array([36.7, 37.2, 37.3])
std_reward = np.zeros_like(y_reward)

# ======================
# Helper
# ======================
def apply_equal_grid(ax):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.55)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.35)

def mark_best_line(ax, x, y, color=C_MARK, prefer="max", star_offset=0.18):
    y = np.asarray(y, dtype=float)
    idx = int(np.argmax(y)) if prefer == "max" else int(np.argmin(y))

    x_best = x[idx]
    y_best = y[idx]
    y_star = y_best + star_offset

    ax.autoscale_view()
    y_bottom, y_top = ax.get_ybound()

    if y_star >= y_top:
        ax.set_ylim(y_bottom, y_star + 0.12)
        y_bottom, y_top = ax.get_ybound()

    ax.vlines(
        x_best, y_bottom, y_star,
        colors=color, linestyles="--", linewidth=1.2,
        alpha=0.9, zorder=5
    )

    ax.scatter(
        [x_best], [y_star],
        marker="*", s=260,
        color=color, edgecolor="white", linewidth=0.8,
        zorder=6
    )
    return idx

def plot_sweep(ax, x, y, ystd, xlabel, ylabel="Accuracy (%)", ylim=(36.4, 38.4), xticks=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ystd = np.asarray(ystd, dtype=float)

    shade = np.where(ystd > 0, ystd, 0.05)

    ax.plot(x, y, marker="o", color=MAIN_COLOR, linewidth=2.2, markersize=6)
    ax.fill_between(x, y - shade, y + shade, color=FILL_COLOR, alpha=0.18, linewidth=0)

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_ylim(*ylim)

    if xticks is not None:
        ax.set_xticks(xticks)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    apply_equal_grid(ax)
    mark_best_line(ax, x, y)

def plot_reward_model(ax, labels, y, ystd, ylabel="Accuracy (%)", ylim=(36.4, 38.4)):
    y = np.asarray(y, dtype=float)
    ystd = np.asarray(ystd, dtype=float)
    x_pos = np.arange(len(labels))

    bars = ax.bar(
        x_pos, y,
        color=BAR_COLOR,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
        width=0.62,
        zorder=3
    )

    if np.any(ystd > 0):
        ax.errorbar(
            x_pos, y, yerr=ystd,
            fmt="none", ecolor="black", elinewidth=1, capsize=3, zorder=4
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_xlabel("Reward Model", fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_ylim(*ylim)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    apply_equal_grid(ax)

    # 不加竖虚线和星标
    #for rect, val in zip(bars, y):
     #   ax.text(
      #      rect.get_x() + rect.get_width() / 2,
       #     rect.get_height() + 0.03,
        #    f"{val:.1f}",
         #   ha="center", va="bottom", fontsize=11
        #)

# ======================
# Figure: 2x2
# ======================
fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=180)

plot_sweep(
    axes[0, 0],
    x_kl, y_kl, std_kl,
    xlabel=r"KL Weight $\beta$",
    ylim=(36.4, 38.4),
    xticks=x_kl
)

plot_sweep(
    axes[0, 1],
    x_mst_dir, y_mst_dir, std_mst_dir,
    xlabel=r"MST Direction Weight $\lambda_{\mathrm{dir}}$",
    ylim=(36.4, 38.4),
    xticks=x_mst_dir
)

plot_sweep(
    axes[1, 0],
    x_mst_dist, y_mst_dist, std_mst_dist,
    xlabel=r"MST Distance Weight $\lambda_{\mathrm{dist}}$",
    ylim=(36.4, 38.4),
    xticks=x_mst_dist
)

plot_reward_model(
    axes[1, 1],
    x_reward_labels, y_reward, std_reward,
    ylim=(36.4, 38.4)
)

# ======================
# Subfigure labels
# ======================
sub_labels = ["(a)", "(b)", "(c)", "(d)"]
for ax, lab in zip(axes.flatten(), sub_labels):
    ax.text(
        0.5, -0.24, lab,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=18
    )

# ======================
# Legend
# 只保留前三张 line plot 需要的 legend
# ======================
legend_handles = [
    Line2D([0], [0], color=MAIN_COLOR, lw=2.2, marker="o", label="Performance curve"),
    Line2D([0], [0], color=C_MARK, lw=1.2, linestyle="--", label="Best setting"),
    Line2D([0], [0], color=C_MARK, marker="*", lw=0, markersize=12, label="Best point"),
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.97),
    ncol=3,
    frameon=False,
    prop={"size":LEGEND_FONTSIZE , "weight": "bold"},
    columnspacing=1.4,
    handlelength=2.0
)

plt.tight_layout(rect=[0, 0, 1, 0.91])
plt.subplots_adjust(hspace=0.5, wspace=0.35)  # 新增这行

# ======================
# Save
# ======================
save_dir = "/root/autodl-tmp/figs"
os.makedirs(save_dir, exist_ok=True)

out_pdf = os.path.join(save_dir, "ablation_update.pdf")
out_png = os.path.join(save_dir, "ablation_update.png")

plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, dpi=300, bbox_inches="tight")

print("[Saved]", out_pdf)
print("[Saved]", out_png)

plt.show()