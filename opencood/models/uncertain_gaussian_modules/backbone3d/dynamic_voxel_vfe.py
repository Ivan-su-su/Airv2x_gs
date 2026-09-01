import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe.vfe_template import VFETemplate
from .vfe.dynamic_pillar_vfe import PFNLayerV2


class DynamicVoxelVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSOLUTE_XYZ
        self.return_abs_coords = self.model_cfg.get('RETURN_ABS_COORDS', False)
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

        # 存储原始值，在forward时转换为tensor并移动到正确设备
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def _visualize_points_and_voxels(self, points_xyz, voxel_coords, point_cloud_range, voxel_size, agent, batch_dict):
        """
        将原始点云与「有点的体素」画成一张 3D 图并保存。
        点云与体素均按统一 point_cloud_range 绘制，每个 agent 存一张图。
        """
        # 转为 numpy，统一范围
        pcr = point_cloud_range.cpu().numpy() if torch.is_tensor(point_cloud_range) else np.array(point_cloud_range)
        vs = voxel_size.cpu().numpy() if torch.is_tensor(voxel_size) else np.array(voxel_size)
        pts = points_xyz.detach().cpu().numpy()
        vc = voxel_coords.detach().cpu().numpy()
        # voxel_coords: (M, 4) -> [batch, z, y, x]，体素中心世界坐标
        x_idx, y_idx, z_idx = vc[:, 3], vc[:, 2], vc[:, 1]
        vox_x = pcr[0] + (x_idx + 0.5) * vs[0]
        vox_y = pcr[1] + (y_idx + 0.5) * vs[1]
        vox_z = pcr[2] + (z_idx + 0.5) * vs[2]
        vox_xyz = np.stack([vox_x, vox_y, vox_z], axis=1)

        save_dir = batch_dict.get('vfe_vis_dir', './vfe_visualization')
        os.makedirs(save_dir, exist_ok=True)
        frame_id = batch_dict.get('frame_id', batch_dict.get('batch_idx', 0))
        if torch.is_tensor(frame_id):
            frame_id = frame_id.item()
        agent_name = agent if agent is not None else 'default'
        fname = os.path.join(save_dir, f'vfe_points_voxels_{agent_name}_{frame_id}.png')

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='#1f77b4', s=0.3, alpha=0.4, label='points')
        ax.scatter(vox_xyz[:, 0], vox_xyz[:, 1], vox_xyz[:, 2], c='#d62728', s=2, alpha=0.7, label='voxels')
        ax.set_xlim(pcr[0], pcr[3])
        ax.set_ylim(pcr[1], pcr[4])
        ax.set_zlim(pcr[2], pcr[5])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title(f'Points & Voxels ({agent_name})')
        plt.tight_layout()
        plt.savefig(fname, dpi=120, bbox_inches='tight')
        plt.close()

    def forward(self, batch_dict, agent = None, **kwargs):
        points = batch_dict[f'origin_lidar_{agent}'] if agent != 'vehicle' else batch_dict['origin_lidar'] # (batch_idx, x, y, z, i, e)
        
        # 检查点云是否为空或数据太少，如果为空则跳过该agent的处理
        if points.shape[0] == 0 or points.shape[1] == 0:
            print(f"[DynamicVoxelVFE] 跳过空的{agent} agent LiDAR数据")
            return batch_dict
        
        # 处理points形状
        if len(points.shape) == 3 and points.shape[0] == 1:
            # 去掉batch维度：[1, N, 4] -> [N, 4]
            points = points.squeeze(0)
        if points.shape[1] == 3:
            intensity = torch.zeros(points.shape[0], 1, device=points.device)
            points = torch.cat([points, intensity], dim=1)
        
        # 在运行时创建正确设备的张量
        point_cloud_range = torch.tensor(self.point_cloud_range, device=points.device, dtype=points.dtype)
        voxel_size = torch.tensor(self.voxel_size, device=points.device, dtype=points.dtype)
        grid_size = torch.tensor(self.grid_size, device=points.device, dtype=torch.int32)
        
        # 在体素化之前计算ori_coords_height（与原始MambaFusion一致）
        if self.return_abs_coords:
            ori_coords_height = (points[:, 2] - point_cloud_range[2]) / voxel_size[2]
        
        points_coords = torch.floor((points[:, [0,1,2]] - point_cloud_range[[0,1,2]]) / voxel_size[[0,1,2]]).int()
        mask = ((points_coords >= 0) & (points_coords < grid_size[[0,1,2]])).all(dim=1)
        
        # 应用mask到ori_coords_height
        if self.return_abs_coords:
            ori_coords_height = ori_coords_height[mask]
        
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [0, 1, 2]].contiguous()

        # 在运行时计算scale值
        scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        scale_yz = grid_size[1] * grid_size[2]
        scale_z = grid_size[2]
        
        merge_coords = points_coords[:, 0] * scale_yz + \
                       points_coords[:, 1] * scale_z + \
                       points_coords[:, 2]
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        if self.return_abs_coords:
            # 在体素化后通过scatter_mean聚合ori_coords_height（与原始MambaFusion一致）
            ori_coords_height = torch_scatter.scatter_mean(ori_coords_height, unq_inv, dim=0)
            # 确保agent键存在
            if agent is not None and agent not in batch_dict:
                batch_dict[agent] = {}
            elif agent is None and 'ori_coords_height' not in batch_dict:
                batch_dict['ori_coords_height'] = ori_coords_height
            else:
                if agent is not None:
                    batch_dict[agent]['ori_coords_height'] = ori_coords_height
                else:
                    batch_dict['ori_coords_height'] = ori_coords_height
        f_cluster = points_xyz - points_mean[unq_inv, :]

        # 在运行时计算offset值
        x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        z_offset = voxel_size[2] / 2 + point_cloud_range[2]
        
        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * voxel_size[0] + x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * voxel_size[1] + y_offset)
        f_center[:, 2] = points_xyz[:, 2] - (points_coords[:, 2].to(points_xyz.dtype) * voxel_size[2] + z_offset)

        if self.use_absolute_xyz:
            features = [points[:, [0,1,2,3]], f_cluster, f_center]
        else:
            features = [points[:, 3], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(points[:, 0:3], 2, dim=1, keepdim=True)
            features.append(points_dist)
        
        features = torch.cat(features, dim=-1)

        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // scale_yz,
                                    (unq_coords % scale_yz) // scale_z,
                                    unq_coords % scale_z), dim=1)
        voxel_coords = voxel_coords[:, [2, 1, 0]]
        voxel_coords = torch.cat([torch.zeros(voxel_coords.shape[0], 1, device=voxel_coords.device).int(), voxel_coords], dim=1)
        
        # 可视化：原始点云 + 有点的体素，每个 agent 存一张 3D 图（需设置 batch_dict['vfe_vis_enable']=True）
        # self._visualize_points_and_voxels(
        #     points_xyz, voxel_coords, point_cloud_range, voxel_size, agent, batch_dict
        # )

        batch_dict[agent]['pillar_features'] = batch_dict[agent]['voxel_features'] = features
        batch_dict[agent]['voxel_coords'] = voxel_coords
        batch_dict[agent]['voxel_num_points'] = unq_cnt.int()

        return batch_dict