# -*- coding: utf-8 -*-
"""
Test script for Gaussian3DBackbone

- 使用当前 backbone3d_semantic.py 的键名
- 在 voxel 坐标系下可视化 3D 高斯椭球
- 坐标范围严格遵循 GRID_SIZE
- 不同颜色区分 feature Gaussians 和 query Gaussians
- 输出图片: gaussian_3d_visualization.png
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# =============== 1. 路径设置 ===============
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gaussian_modules.backbone_3d.backbone3d_gaussian_query import Gaussian3DBackbone


# =============== 2. 配置 ===============
model_cfg = {
    "GRID_SIZE": [200, 704, 32],  # [H, W, Z] [y,x,z]
    "VOXEL_SIZE": [0.4, 0.4, 0.125],
    "POINT_CLOUD_RANGE": [-140.8, -40, -3, 140.8, 40, 1],
    "NUM_POINT_FEATURES": 4,
    "VFE": {
        "USE_NORM": True,
        "WITH_DISTANCE": False,
        "USE_ABSLOTE_XYZ": True,
        "NUM_FILTERS": [128, 128],
        "RETURN_ABS_COORDS": False
    },
    "BACKBONE_3D": {
        "NUM_FEATURES": 128,
        "HIDDEN_DIM": 128,
        "MAX_GAUSSIAN_RATIO": 0.1,
        "PROJECTION_METHOD": "scatter_mean",
        "USE_GUMBEL": False,
        "GUMBEL_TEMPERATURE": 0.1,
        "NUM_CLASSES": 4,
        "SCALE_RANGE": [0.01, 3.2],
        "QUERY_VOXEL_STRIDE": [2, 8, 12],
    }
}

GRID_SIZE = model_cfg["GRID_SIZE"]  # [H, W, Z]
VOXEL_SIZE = model_cfg["VOXEL_SIZE"]
PC_RANGE = model_cfg["POINT_CLOUD_RANGE"]


# =============== 3. 初始化模型 ===============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Gaussian3DBackbone(model_cfg).to(device)
model.eval()


# =============== 4. 构造虚拟点云 ===============
num_points = 80000
points = torch.rand((num_points, 4), device=device)

points[:, 0] = points[:, 0] * (PC_RANGE[3] - PC_RANGE[0]) + PC_RANGE[0]
points[:, 1] = points[:, 1] * (PC_RANGE[4] - PC_RANGE[1]) + PC_RANGE[1]
points[:, 2] = points[:, 2] * (PC_RANGE[5] - PC_RANGE[2]) + PC_RANGE[2]
points[:, 3] = torch.rand(num_points, device=device)

batch_dict = {"vehicle": {"origin_lidar": points}}


# =============== 5. 前向推理 ===============
with torch.no_grad():
    output = model(batch_dict, agent="vehicle")

print("\n========== [测试输出结果 Shape 汇总] ==========")
vfe_feats = output["vehicle"]["voxel_features"]
vfe_coords = output["vehicle"]["voxel_coords"]
print(f"VFE voxel_features: {tuple(vfe_feats.shape)}")
print(f"VFE voxel_coords:   {tuple(vfe_coords.shape)}")

gaussians_feat = output["vehicle"]["lidar_gaussians"]
queries_3d = output["vehicle"]["lidar_gaussian_queries"]

print("\n[Feature Gaussians]")
for k, v in gaussians_feat.items():
    print(f"  {k:10s}: {tuple(v.shape)}")

print("\n[Query Gaussians]")
for k, v in queries_3d.items():
    print(f"  {k:10s}: {tuple(v.shape)}")


# =============== 6. world -> voxel index ===============
def world_to_voxel(mu_world: torch.Tensor,
                   voxel_size,
                   pc_range,
                   grid_size):
    """
    world 坐标 -> voxel index (连续值)
    """
    vx, vy, vz = voxel_size
    x_min, y_min, z_min = pc_range[0], pc_range[1], pc_range[2]
    H, W, Z = grid_size  # [H, W, Z]

    x_idx = (mu_world[:, 0] - x_min) / vx
    y_idx = (mu_world[:, 1] - y_min) / vy
    z_idx = (mu_world[:, 2] - z_min) / vz

    x_idx = torch.clamp(x_idx, 0, W - 1)
    y_idx = torch.clamp(y_idx, 0, H - 1)
    z_idx = torch.clamp(z_idx, 0, Z - 1)

    return x_idx, y_idx, z_idx


def scale_world_to_voxel(scale_world: torch.Tensor, voxel_size):
    """
    world 尺度 -> voxel 半径
    """
    vx, vy, vz = voxel_size
    # scale_world: [K,3] (sx, sy, sz) in world units
    s_x = scale_world[:, 0] / vx
    s_y = scale_world[:, 1] / vy
    s_z = scale_world[:, 2] / vz
    return s_x, s_y, s_z


# =============== 7. 准备可视化椭球的中心和半径 ===============
# feature gaussians
mu_f_world = gaussians_feat["mu"]        # [K_f, 3]
scale_f_world = gaussians_feat["scale"]  # [K_f, 3]

x_f, y_f, z_f = world_to_voxel(mu_f_world, VOXEL_SIZE, PC_RANGE, GRID_SIZE)
sx_f, sy_f, sz_f = scale_world_to_voxel(scale_f_world, VOXEL_SIZE)

# query gaussians
mu_q_world = queries_3d["mu"]        # [K_q, 3]
scale_q_world = queries_3d["scale"]  # [K_q, 3]

x_q, y_q, z_q = world_to_voxel(mu_q_world, VOXEL_SIZE, PC_RANGE, GRID_SIZE)
sx_q, sy_q, sz_q = scale_world_to_voxel(scale_q_world, VOXEL_SIZE)

# 转为 numpy
x_f = x_f.cpu().numpy()
y_f = y_f.cpu().numpy()
z_f = z_f.cpu().numpy()
sx_f = sx_f.cpu().numpy()
sy_f = sy_f.cpu().numpy()
sz_f = sz_f.cpu().numpy()

x_q = x_q.cpu().numpy()
y_q = y_q.cpu().numpy()
z_q = z_q.cpu().numpy()
sx_q = sx_q.cpu().numpy()
sy_q = sy_q.cpu().numpy()
sz_q = sz_q.cpu().numpy()

# 限制椭球数量，避免太卡
max_ellipsoids_feat = 700
max_ellipsoids_query = 200

def random_subset(x, y, z, sx, sy, sz, max_n):
    n = x.shape[0]
    if n <= max_n:
        return x, y, z, sx, sy, sz
    idx = np.random.choice(n, max_n, replace=False)
    return x[idx], y[idx], z[idx], sx[idx], sy[idx], sz[idx]

x_f, y_f, z_f, sx_f, sy_f, sz_f = random_subset(
    x_f, y_f, z_f, sx_f, sy_f, sz_f, max_ellipsoids_feat
)
x_q, y_q, z_q, sx_q, sy_q, sz_q = random_subset(
    x_q, y_q, z_q, sx_q, sy_q, sz_q, max_ellipsoids_query
)


# =============== 8. 椭球绘制函数 ===============
def plot_ellipsoid(ax, center, radii, color, alpha=0.25, n_steps=16):
    """
    在 ax 上画 3D 椭球 (不考虑旋转，当前 r 是单位四元数)
    center: (cx, cy, cz) in voxel index
    radii:  (rx, ry, rz) in voxel units
    """
    cx, cy, cz = center
    rx, ry, rz = radii

    u = np.linspace(0.0, 2.0 * np.pi, n_steps)
    v = np.linspace(0.0, np.pi, n_steps)
    uu, vv = np.meshgrid(u, v)

    x = rx * np.cos(uu) * np.sin(vv) + cx
    y = ry * np.sin(uu) * np.sin(vv) + cy
    z = rz * np.cos(vv) + cz

    ax.plot_surface(
        x, y, z,
        rstride=1, cstride=1,
        linewidth=0, antialiased=False,
        alpha=alpha, color=color
    )


# =============== 9. 3D 可视化 ===============
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 先画 query 椭球（更“重要”）
for cx, cy, cz, rx, ry, rz in zip(x_q, y_q, z_q, sx_q, sy_q, sz_q):
    plot_ellipsoid(ax, (cx, cy, cz), (rx, ry, rz), color="#FF7F0E", alpha=0.5)

# 再画 feature 椭球（淡一点）
for cx, cy, cz, rx, ry, rz in zip(x_f, y_f, z_f, sx_f, sy_f, sz_f):
    plot_ellipsoid(ax, (cx, cy, cz), (rx, ry, rz), color="#1F77B4", alpha=0.2)

# 再把中心点画出来，方便看位置
ax.scatter(x_f, y_f, z_f, s=5, c="#1F77B4", marker="o", label="Feature Gaussians")
ax.scatter(x_q, y_q, z_q, s=20, c="#FF7F0E", marker="^", label="Query Gaussians")

H, W, Z = GRID_SIZE
ax.set_xlim(0, W - 1)
ax.set_ylim(0, H - 1)
ax.set_zlim(0, Z - 1)

ax.set_xlabel("X (voxel index, W)", fontsize=12)
ax.set_ylabel("Y (voxel index, H)", fontsize=12)
ax.set_zlabel("Z (voxel index, Z)", fontsize=12)
ax.set_title("3D Voxel-space Ellipsoids of Feature & Query Gaussians", fontsize=14)
ax.legend(loc="upper right")

plt.tight_layout()
save_path = os.path.join(current_dir, "gaussian_3d_visualization.png")
plt.savefig(save_path, dpi=200)
plt.close(fig)

print("\n=============================================")
print(f"3D 椭球可视化已保存到: {save_path}")
print("=============================================\n")
