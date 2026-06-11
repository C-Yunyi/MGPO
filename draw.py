import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# ================= CONFIG =================
# 配色方案 (学术风格)
COLOR_MST = '#005EB8'      # 深蓝 (Ground Truth)
COLOR_GEN = '#D32F2F'      # 深红 (Generated)
COLOR_DIST = '#424242'     # 深灰 (Distance Penalty)
COLOR_ANGLE = '#FFC107'    # 琥珀色 (Angle Arc)

FIG_SIZE = (8, 6)
FONT_SIZE_LABEL = 14
FONT_SIZE_TITLE = 16

# ================= 1. 定义几何点和向量 =================
# A. MST (真实数据流形)
# 定义一条 MST 边，连接两个节点 N1 和 N2
N1 = np.array([2.0, 3.0])
N2 = np.array([10.0, 4.5])
v_mst = N2 - N1
v_mst_norm = v_mst / np.linalg.norm(v_mst) # 单位向量

# B. 生成轨迹 (模型生成的一步)
# 从 x_mid 生成到 x_0
x_mid = np.array([4.0, 7.5])
x_0 = np.array([9.0, 10.5])
v_sample = x_0 - x_mid
v_sample_len = np.linalg.norm(v_sample)

# C. 关键点计算
# 1. 最近节点 (用于距离惩罚)
# 在这个例子中，直观设定 N2 是离 x_0 最近的节点
nearest_node = N2

# 2. 用于方向比较的向量平移
# 为了形成“三角形”视觉效果，将 v_mst 平移到 x_mid 起点
# 为了美观，将平移后的向量长度缩放至与 v_sample 相同
v_mst_shifted_end = x_mid + v_mst_norm * v_sample_len

# ================= 2. 绘图 =================
fig, ax = plt.subplots(figsize=FIG_SIZE)
ax.set_aspect('equal')
ax.axis('off') # 去除坐标轴，保持极简

# --- A. 画 MST (底层流形) ---
# 画边
ax.add_patch(patches.FancyArrowPatch(N1, N2, arrowstyle='-', color=COLOR_MST, lw=4, alpha=0.4, zorder=1))
# 画节点
ax.scatter([N1[0], N2[0]], [N1[1], N2[1]], c=COLOR_MST, s=200, zorder=2)
# 标注 MST 向量方向 (在边上)
mid_edge = (N1 + N2) / 2
ax.arrow(mid_edge[0], mid_edge[1], v_mst[0]*0.2, v_mst[1]*0.2, head_width=0.3, head_length=0.4, fc=COLOR_MST, ec=COLOR_MST, lw=2, zorder=3)
ax.text(mid_edge[0], mid_edge[1]-0.8, r'$\mathbf{v}_{mst}$ (Data Direction)', color=COLOR_MST, fontsize=12, ha='center')
# 标注节点
ax.text(N1[0]-0.5, N1[1], '$N_1$', color=COLOR_MST, fontsize=FONT_SIZE_LABEL, va='center', ha='right')
ax.text(N2[0]+0.5, N2[1], '$N_2$', color=COLOR_MST, fontsize=FONT_SIZE_LABEL, va='center', ha='left')


# --- B. 画生成轨迹 (v_sample) ---
# 画红色粗箭头
ax.add_patch(patches.FancyArrowPatch(x_mid, x_0, arrowstyle='-|>', mutation_scale=25, color=COLOR_GEN, lw=4, zorder=4))
# 画起点和终点
ax.scatter(x_mid[0], x_mid[1], c='white', edgecolors=COLOR_GEN, s=150, lw=2, zorder=5) # 空心圆
ax.scatter(x_0[0], x_0[1], c=COLOR_GEN, marker='*', s=350, zorder=5) # 实心星
# 标注
ax.text(x_mid[0]-0.6, x_mid[1], '$x_{mid}$', color=COLOR_GEN, fontsize=FONT_SIZE_LABEL, va='center', ha='right')
ax.text(x_0[0]+0.6, x_0[1], '$x_0$', color=COLOR_GEN, fontsize=FONT_SIZE_LABEL, va='center', ha='left')
ax.text((x_mid[0]+x_0[0])/2-0.5, (x_mid[1]+x_0[1])/2+1.2, r'$\mathbf{v}_{sample}$', color=COLOR_GEN, fontsize=12, ha='center')


# --- C. 可视化惩罚项 (Penalties) ---

# 1. 方向惩罚 (Direction Penalty - 夹角)
# 画平移后的 v_mst (虚线，形成三角形的底边)
ax.add_patch(patches.FancyArrowPatch(x_mid, v_mst_shifted_end, arrowstyle='-|>', mutation_scale=20, color=COLOR_MST, lw=3, ls='--', alpha=0.6, zorder=3))
ax.text(v_mst_shifted_end[0]+0.5, v_mst_shifted_end[1], r'$\mathbf{v}_{mst}$ (Shifted)', color=COLOR_MST, fontsize=12, alpha=0.8, va='center')

# 画夹角弧线
# 计算角度
vec1 = v_sample
vec2 = v_mst
angle1 = np.degrees(np.arctan2(vec1[1], vec1[0]))
angle2 = np.degrees(np.arctan2(vec2[1], vec2[0]))
if angle1 < 0: angle1 += 360
if angle2 < 0: angle2 += 360
# 确保画的是两个向量之间的小角
if abs(angle1 - angle2) > 180:
    if angle1 > angle2: angle2 += 360
    else: angle1 += 360
theta1, theta2 = min(angle1, angle2), max(angle1, angle2)

# 绘制弧形
arc_radius = 2.5
arc = patches.Arc(x_mid, arc_radius*2, arc_radius*2, angle=0, theta1=theta1, theta2=theta2, color='k', lw=1.5, zorder=6)
ax.add_patch(arc)
# 标注角度
angle_mid = np.radians((theta1 + theta2) / 2)
arc_label_pos = x_mid + np.array([np.cos(angle_mid), np.sin(angle_mid)]) * (arc_radius + 0.6)
ax.text(arc_label_pos[0], arc_label_pos[1], r'$\theta_{dir}$', fontsize=16, fontweight='bold', ha='center', va='center')
ax.text(arc_label_pos[0], arc_label_pos[1]-0.8, 'Direction\nPenalty', fontsize=10, ha='center', va='top', color=COLOR_DIST)


# 2. 距离惩罚 (Distance Penalty - 连线)
# 画黑色虚线连接 x_0 和 nearest_node (N2)
ax.plot([x_0[0], nearest_node[0]], [x_0[1], nearest_node[1]], color=COLOR_DIST, ls=':', lw=2, zorder=3)
# 标注距离
mid_dist = (x_0 + nearest_node) / 2
ax.text(mid_dist[0]+0.3, mid_dist[1], r'$d_{dist}$', fontsize=16, fontweight='bold', color=COLOR_DIST, va='center')
ax.text(mid_dist[0]+0.3, mid_dist[1]-0.8, 'Distance\nPenalty', fontsize=10, color=COLOR_DIST, va='top', ha='center')

# --- D. 整体布局调整 ---
ax.set_xlim(0, 13)
ax.set_ylim(1, 13)

# 添加图例 (手动构建以保持整洁)
legend_elements = [
    Line2D([0], [0], color=COLOR_MST, lw=4, alpha=0.6, label='MST Manifold (Ground Truth)'),
    Line2D([0], [0], color=COLOR_GEN, lw=4, label='Generated Trajectory (Model)'),
    Line2D([0], [0], color=COLOR_DIST, ls=':', lw=2, label='Distance Penalty ($d_{dist}$)'),
    Line2D([0], [0], color='k', lw=1.5, label='Direction Penalty (Angle $\\theta_{dir}$)')
]
ax.legend(handles=legend_elements, loc='upper left', frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig('mst_matching_triangle_v2.pdf', transparent=True, bbox_inches='tight')
# plt.show()