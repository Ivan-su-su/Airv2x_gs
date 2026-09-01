# -*- coding: utf-8 -*-
"""
Optimized Gaussian Backbone 3D
基于稀疏卷积的可学习高斯生成骨干网络
优化版本：提高效率、稳定性和GPU友好性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import spconv.pytorch as spconv
import torch_scatter

from .vfe.vfe_template import VFETemplate
from .vfe.dynamic_voxel_vfe import PFNLayerV2
from opencood.utils.seg_label_utils import (
    SegLabelMapper, 
    create_seg_label_mapper_from_config
)
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class DynamicVoxelVFE(VFETemplate):
    """
    优化版动态体素特征提取器
    - 使用register_buffer避免重复创建tensor
    - 简化坐标合并逻辑
    
    注意：grid_size 语义为 [X, Y, Z] = [W, H, Z]（体素化按 x, y, z 顺序）
    """
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.use_norm = self.model_cfg.get('USE_NORM', True)
        self.with_distance = self.model_cfg.get('WITH_DISTANCE', False)
        self.use_absolute_xyz = self.model_cfg.get('USE_ABSLOTE_XYZ', True)
        self.return_abs_coords = self.model_cfg.get('RETURN_ABS_COORDS', False)
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.get('NUM_FILTERS', [128, 128])
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

        # 注册静态张量为buffer，避免每次forward重新创建
        self.register_buffer("point_cloud_range_tensor", torch.tensor(point_cloud_range))
        self.register_buffer("voxel_size_tensor", torch.tensor(voxel_size))
        self.register_buffer("grid_size_tensor", torch.tensor(grid_size, dtype=torch.int32))
        
        # 存储原始值用于计算
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def forward(self, batch_dict, agent=None, **kwargs):
        # 处理agent参数
        if agent is None or agent == 'vehicle':
            points = batch_dict.get('origin_lidar', None)
        else:
            points = batch_dict.get(f'origin_lidar_{agent}', None)
        
        if points is None:
            raise KeyError(f"Could not find 'origin_lidar' or 'origin_lidar_{agent}' in batch_dict")
        
       # 检查点云是否为空
        if len(points.shape) == 3 and points.shape[0] == 1:
            # 去掉batch维度：[1, N, 4] -> [N, 4]
            points = points.squeeze(0)
        
        # 确保points有正确的列数
        if points.shape[1] == 4:  # [N, 4]
            # 不执行任何操作
            pass
        elif points.shape[1] == 3:  # [N, 3] -> [N, 4]
            # 添加intensity (0)
            intensity = torch.zeros(points.shape[0], 1, device=points.device)
            points = torch.cat([points, intensity], dim=1)  # [N, 4]
        
        # 在运行时创建正确设备的张量
        point_cloud_range = torch.tensor(self.point_cloud_range, device=points.device, dtype=points.dtype)
        voxel_size = torch.tensor(self.voxel_size, device=points.device, dtype=points.dtype)
        grid_size = torch.tensor(self.grid_size, device=points.device, dtype=torch.int32)
        
        # 在体素化之前计算ori_coords_height（与原始MambaFusion一致）
        if self.return_abs_coords:  #False
            ori_coords_height = (points[:, 2] - point_cloud_range[2]) / voxel_size[2]
            # 点高度的原始相对高度（未取整）
        
        points_coords = torch.floor((points[:, [0,1,2]] - point_cloud_range[[0,1,2]]) / voxel_size[[0,1,2]]).int()
        mask = ((points_coords >= 0) & (points_coords < grid_size[[1,0,2]])).all(dim=1)
        
        # 应用mask到ori_coords_height
        if self.return_abs_coords:
            ori_coords_height = ori_coords_height[mask]
        
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [0,1,2]].contiguous()

        # 在运行时计算scale值 dwb 为什么不写在init里？
        scale_yz = grid_size[0] * grid_size[2]
        scale_z = grid_size[2]
        
        merge_coords = points_coords[:, 0] * scale_yz + \
                       points_coords[:, 1] * scale_z + \
                       points_coords[:, 2]
        # 把3维索引压成1维唯一值

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)
        # unq_coords: 所有体素的唯一索引值，长度为V
        # unq_inv: 每个点对应的体素在unq_coords中的索引
        # unq_cnt: 每个体素包含的点数

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        # 每个体素内点的均值坐标，形状为[V, 3]
        if self.return_abs_coords:
            # 在体素化后通过scatter_mean聚合ori_coords_height（与原始MambaFusion一致）
            ori_coords_height = torch_scatter.scatter_mean(ori_coords_height, unq_inv, dim=0)
            batch_dict[agent]['ori_coords_height'] = ori_coords_height
        
        f_cluster = points_xyz - points_mean[unq_inv, :]
        # 每个点相对于体素内均值坐标的偏移，(N,3)

        # 在运行时计算offset值 dwb 为什么不写在init里？
        x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        z_offset = voxel_size[2] / 2 + point_cloud_range[2]
        
        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * voxel_size[0] + x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * voxel_size[1] + y_offset)
        f_center[:, 2] = points_xyz[:, 2] - (points_coords[:, 2].to(points_xyz.dtype) * voxel_size[2] + z_offset)
        # 每个点相对于体素中心的偏移，(N,3)

        if self.use_absolute_xyz:   #True
            features = [points[:, [0,1,2,3]], f_cluster, f_center]
            # [N, 4+3+3] x,y,z,intensity
        else:
            features = [points[:, 3], f_cluster, f_center]
            # [N, 1+3+3] intensity

        if self.with_distance:   #False
            points_dist = torch.norm(points[:, 0:3], 2, dim=1, keepdim=True)
            features.append(points_dist)
        
        features = torch.cat(features, dim=-1)

        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)
            # 最终输出体素的特征，(num_voxel,128)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // scale_yz,
                                    (unq_coords % scale_yz) // scale_z,
                                    unq_coords % scale_z), dim=1)
        # 将之前的合并索引解码回三元组索引并重排为[z, y, x]的格式，(num_voxel,3)
        voxel_coords = voxel_coords[:, [2, 1, 0]]
        # 增加一个batch_idx维度，因为SparseConvTensor需要batch_idx维度 (num_voxel,4)
        voxel_coords = torch.cat([torch.zeros(voxel_coords.shape[0], 1, device=voxel_coords.device).int(), voxel_coords], dim=1)
        
        batch_dict[agent]['pillar_features'] = batch_dict[agent]['voxel_features'] = features
        batch_dict[agent]['voxel_coords'] = voxel_coords
        batch_dict[agent]['voxel_num_points'] = unq_cnt.int()
        
        return batch_dict



class GaussianBackbone3DForPretraining(nn.Module):
    """
    预训练版本的 Gaussian Backbone 3D
    --------------------------------------------------------
    仅包含语义分类所需的模块：
    - VFE (DynamicVoxelVFE): 点云 → 体素特征
    - Encoder: 稀疏卷积编码器
    - Semantic Head: 语义分类头
    --------------------------------------------------------
    注意：grid_size 语义为 [H, W, Z] = [y, x, z]
    --------------------------------------------------------
    输出：
      batch_dict[agent]['semantic_logits']: [N_voxel, num_classes]
      batch_dict[agent]['semantic_targets']: [N_voxel]
    """

    def __init__(self, model_cfg, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        # 配置参数
        self.num_features = model_cfg.get('NUM_FEATURES')
        self.hidden_dim = model_cfg.get('HIDDEN_DIM')
        
        # 语义分类配置
        self.num_classes = model_cfg.get('NUM_CLASSES')  # 0类为背景，1..(m-1)为前景
        assert self.num_classes > 1, "NUM_CLASSES must be > 1 (0 for background, >=1 for foreground)."
        self.use_static_supervision = model_cfg.get('USE_STATIC_LABEL', False)
        
        # 初始化语义标签映射器（用于从真实世界坐标查询标签）
        seg_hw = model_cfg.get('seg_hw', 512)
        seg_res = model_cfg.get('seg_res', 0.25)
        self.seg_label_mapper = SegLabelMapper(
            seg_hw=seg_hw,
            seg_res=seg_res,
            lidar_range=point_cloud_range,
            ego_center=True  # 假设标签图以ego为中心
        )

        # 稀疏卷积编码器 - 使用 SparseBatchNorm
        self.encoder = spconv.SparseSequential(
            spconv.SubMConv3d(self.num_features, self.hidden_dim, 3, padding=1, bias=False),
            spconv.SparseBatchNorm(self.hidden_dim),
            nn.ReLU(True),
            spconv.SubMConv3d(self.hidden_dim, self.hidden_dim, 3, padding=1, bias=False),
            spconv.SparseBatchNorm(self.hidden_dim),
            nn.ReLU(True)
        )

        # Semantic Head: 输出 m 类 (包含背景类0)
        self.semantic_head = spconv.SubMConv3d(self.hidden_dim, self.num_classes, kernel_size=1)
        
        # 参数初始化
        self._init_weights()

        print(f"[GaussianBackbone3DForPretraining] 初始化完成:")
        print(f"  - Grid Size: {self.grid_size}")
        print(f"  - Feature Dim: {self.num_features}")
        print(f"  - Hidden Dim: {self.hidden_dim}")
        print(f"  - Num Classes: {self.num_classes}")
        print(f"  - Mode: Pretraining")
    
    def _init_weights(self):
        """初始化网络权重，提高训练稳定性"""
        for m in self.modules():
            if isinstance(m, spconv.SubMConv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, spconv.SparseBatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # dwb 添加语义监督
    def _build_semantic_supervision(self, batch_dict, agent, voxel_coords):
        """
        根据BEV分割标签为体素生成监督信号，并选出重要体素。
        Returns:
            semantic_targets: [N_voxel] long tensor, 0=背景,1=动态,2=静态(可选)
            important_mask:  [N_voxel] bool tensor
        """
        label_dict = batch_dict.get('label_dict', None)

        # 使用原始标签图（需要坐标映射）
        dyn_orig = label_dict.get('dynamic_seg_label', None)
        stat_orig = label_dict.get('static_seg_label', None) if self.use_static_supervision else None
        
        device = voxel_coords.device
        semantic_targets = torch.zeros(voxel_coords.shape[0], dtype=torch.long, device=device)

        if dyn_orig is not None:
            # 获取标签和像素坐标（用于可视化）
            dyn_labels, valid_mask, pixel_coords = self._gather_bev_labels_from_world_coords(
                dyn_orig, voxel_coords, return_pixel_coords=True
            )
            if dyn_labels is not None:
                semantic_targets = torch.where(valid_mask & (dyn_labels > 0), dyn_labels.long(), semantic_targets)
            
            # 可视化：创建动态分割地图并用红点标记查询到的像素坐标位置
            # if True:
            #     try:
            #         self._visualize_seg_label_with_pixel_coords(
            #             seg_label=dyn_orig,
            #             pixel_coords=pixel_coords,
            #             valid_mask=valid_mask,
            #             batch_dict=batch_dict,
            #             agent=agent
            #         )
            #     except Exception as e:
            #         print(f"[backbone3d_semantic] 可视化失败: {e}")
            #         import traceback
            #         traceback.print_exc()

        # 处理静态标签
        # if self.use_static_supervision:
        #     if stat_orig is not None:
        #         static_labels, valid_mask = self._gather_bev_labels_from_world_coords(stat_orig, voxel_coords)
        #         if static_labels is not None:
        #             static_mask = valid_mask & (static_labels > 0) & (semantic_targets == 0)
        #             semantic_targets = torch.where(static_mask, torch.full_like(semantic_targets, 2), semantic_targets)

        return semantic_targets

    def _gather_bev_labels(self, bev_label, voxel_coords):
        """
        从BEV标签图中查询体素对应的语义标签
        使用真实世界坐标映射，支持原始标签图和调整后的BEV标签图
        
        Args:
            bev_label: [B, H, W] 或 [H, W] BEV分割标签图（可能是调整后的BEV标签图）
            voxel_coords: [N, 4] 体素坐标 [batch, z, y, x]
        
        Returns:
            labels: [N] 语义标签
        """
        if bev_label is None:
            return None
        
        # 如果提供了调整后的BEV标签图（dynamic_seg_label_bev），直接使用体素坐标索引
        # 因为BEV标签图已经与体素网格对齐
        if bev_label.dim() == 2:
            bev_label = bev_label.unsqueeze(0)
        bev_label = bev_label.to(voxel_coords.device)
        B, H, W = bev_label.shape

        coords = voxel_coords.long()
        batch_idx = coords[:, 0].clamp(0, B - 1)
        y_idx = coords[:, 2].clamp(0, H - 1)
        x_idx = coords[:, 3].clamp(0, W - 1)

        return bev_label[batch_idx, y_idx, x_idx]
    
    def _gather_bev_labels_from_world_coords(self, seg_label, voxel_coords, return_pixel_coords=False):
        """
        从原始分割标签图（dynamic_seg_label/static_seg_label）查询标签
        使用真实世界坐标映射，考虑标签图的坐标变换
        
        Args:
            seg_label: [B, H, W] 或 [H, W] 原始分割标签图（512x512）
            voxel_coords: [N, 4] 体素坐标 [batch, z, y, x]
            return_pixel_coords: 是否返回像素坐标（用于可视化）
        
        Returns:
            labels: [N] 语义标签
            valid_mask: [N] bool tensor，表示体素是否在标签图范围内
            pixel_coords: [N, 2] 像素坐标 (u, v)，仅在 return_pixel_coords=True 时返回
        """
        
        # 使用SegLabelMapper查询标签
        result = self.seg_label_mapper.voxel_coords_to_labels(
            seg_label=seg_label,
            voxel_coords=voxel_coords,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range,
            default_label=0,
            return_pixel_coords=return_pixel_coords
        )
        
        if return_pixel_coords:
            labels, valid_mask, pixel_coords = result
            return labels, valid_mask, pixel_coords
        else:
            labels, valid_mask = result
            return labels, valid_mask

    def _select_important_voxels(self, semantic_targets, num_points):
        
        importance_mask = semantic_targets > 0
        if num_points is None or num_points.numel() == 0 or not importance_mask.any():
            return importance_mask

        num_points = num_points.to(importance_mask.device).float()
        selected_counts = num_points[importance_mask]
        if selected_counts.numel() == 0:
            return importance_mask

        
        topk = max(1, int(self.important_voxel_ratio * selected_counts.numel()))
        threshold = torch.topk(selected_counts, topk).values.min()
        importance_mask = importance_mask & (num_points >= threshold)
        return importance_mask
    
    def _visualize_seg_label_with_pixel_coords(self, seg_label, pixel_coords, valid_mask, batch_dict, agent=None):
        """
        可视化动态分割标签图，并用红点标记查询到的像素坐标位置（u, v）
        
        Args:
            seg_label: [H, W] 或 [B, H, W] BEV分割标签图
            pixel_coords: [N, 2] 像素坐标 (u, v)，来自 seg_label_utils.py 第170-171行
            valid_mask: [N] bool tensor，表示哪些像素坐标有效
            batch_dict: batch字典，用于获取scenario信息
            agent: agent类型 ('vehicle', 'rsu', 'drone' 等)
        """
        # 转换标签图为numpy数组
        if isinstance(seg_label, torch.Tensor):
            seg_label_np = seg_label.detach().cpu().numpy()
        else:
            seg_label_np = np.array(seg_label)
        
        # 确保是2D数组
        if seg_label_np.ndim == 3:
            seg_label_np = seg_label_np[0]  # 取第一个batch
        
        seg_hw = seg_label_np.shape[0]
        assert seg_label_np.shape == (seg_hw, seg_hw), \
            f"标签图形状应为 ({seg_hw}, {seg_hw})，得到 {seg_label_np.shape}"
        
        # 创建颜色映射
        num_classes = int(seg_label_np.max()) + 1
        colors = plt.cm.tab10(np.linspace(0, 1, max(10, num_classes)))
        colors[0] = [0, 0, 0, 1]  # 背景黑色
        
        # 创建RGB图像
        seg_rgb = np.zeros((seg_hw, seg_hw, 3), dtype=np.uint8)
        for label_id in range(num_classes):
            mask = seg_label_np == label_id
            seg_rgb[mask] = (colors[label_id][:3] * 255).astype(np.uint8)
        
        # 转换像素坐标为numpy数组
        if isinstance(pixel_coords, torch.Tensor):
            pixel_coords_np = pixel_coords.detach().cpu().numpy()
        else:
            pixel_coords_np = np.array(pixel_coords)
        
        if isinstance(valid_mask, torch.Tensor):
            valid_mask_np = valid_mask.detach().cpu().numpy()
        else:
            valid_mask_np = np.array(valid_mask)
        
        # 只保留有效的像素坐标
        valid_pixel_coords = pixel_coords_np[valid_mask_np]
        
        # 创建可视化图像
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(seg_rgb, origin='upper', interpolation='nearest')
        
        # 绘制像素坐标位置（红点）
        # pixel_coords是[u, v]格式，其中u是行索引（y），v是列索引（x）
        # imshow需要[y, x]格式，所以u对应y，v对应x
        if len(valid_pixel_coords) > 0:
            u_coords = valid_pixel_coords[:, 0].astype(int)  # u -> y (行索引)
            v_coords = valid_pixel_coords[:, 1].astype(int)  # v -> x (列索引)
            
            # 确保坐标在有效范围内
            valid_x = (v_coords >= 0) & (v_coords < seg_hw)
            valid_y = (u_coords >= 0) & (u_coords < seg_hw)
            valid_both = valid_x & valid_y
            
            if valid_both.sum() > 0:
                ax.scatter(
                    v_coords[valid_both],  # x坐标（列）
                    u_coords[valid_both],  # y坐标（行）
                    c='white', 
                    s=5,  # 点的大小
                    alpha=0.9,
                    marker='o',
                    edgecolors='white',
                    linewidths=0.5,
                    label=f'Query Points (u, v) ({valid_both.sum()}/{len(valid_pixel_coords)})'
                )
        
        # 获取scenario信息
        scenario_info = ""
        if 'scenario_index_list' in batch_dict:
            scenario_idx = batch_dict['scenario_index_list'][0] if len(batch_dict['scenario_index_list']) > 0 else 0
            scenario_info = f"Scenario: {scenario_idx}"
        elif 'scenario_index' in batch_dict:
            scenario_info = f"Scenario: {batch_dict['scenario_index']}"
        
        ax.set_title(f"Dynamic Segmentation Map with Query Pixel Coordinates (u, v)\n{scenario_info}", 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('X (pixels, v)', fontsize=12)
        ax.set_ylabel('Y (pixels, u)', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 添加颜色图例
        legend_elements = []
        for label_id in range(min(num_classes, 8)):  # 最多显示8个类别
            color = colors[label_id]
            label_name = f'Class {label_id}' if label_id > 0 else 'Background'
            legend_elements.append(
                plt.Line2D([0], [0], marker='s', color='w', 
                           markerfacecolor=color[:3], markersize=10, 
                           label=label_name, markeredgecolor='black', markeredgewidth=0.5)
            )
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
        
        plt.tight_layout()
        
        # 保存图像
        save_dir = Path('visualizations')
        save_dir.mkdir(parents=True, exist_ok=True)
        agent_name = str(agent) if agent is not None else "default"
        save_path = save_dir / f'seg_label_with_pixel_coords_{scenario_info.replace(" ", "_").replace(":", "")}_{agent_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[backbone3d_semantic] 可视化图像已保存到: {save_path}")
        
        plt.close(fig)

    def forward(self, batch_dict, agent=None, **kwargs):
        """
        预训练模式的前向传播
        只计算到语义分类，不计算高斯参数
        
        Args:
            batch_dict: 包含体素特征的字典
            agent: agent类型 ('vehicle', 'rsu', 'drone' 等)
        
        Returns:
            batch_dict: 更新后的字典，包含：
                - semantic_logits: [N_voxel, num_classes]
                - semantic_targets: [N_voxel]
        """
        if agent is not None:
            voxel_features = batch_dict[agent]['voxel_features']
            voxel_coords = batch_dict[agent]['voxel_coords']
        else:
            voxel_features = batch_dict['voxel_features']
            voxel_coords = batch_dict['voxel_coords']

        device = voxel_features.device
        
        # self.grid_size: [H, W, Z] = [y, x, z]
        batch_size = 1
        spatial_shape = (
            self.grid_size[2],  # Z -> z_size
            self.grid_size[0],  # H -> y_size
            self.grid_size[1]   # W -> x_size
        )

        # dwb 添加语义监督
        semantic_targets = self._build_semantic_supervision(batch_dict, agent, voxel_coords)
        if semantic_targets is not None:
            if agent is not None:
                batch_dict[agent]['semantic_targets'] = semantic_targets
            else:
                batch_dict['semantic_targets'] = semantic_targets

        # step 1: SparseConv 编码
        x = spconv.SparseConvTensor(voxel_features, voxel_coords, spatial_shape, batch_size)
        encoded = self.encoder(x)  # encoded.features: [N_voxel, 128]

        # step 2: 语义分类预测
        semantic_logits = self.semantic_head(encoded)  # SparseConvTensor
        semantic_logits_dense = semantic_logits.features  # [N_voxel, num_classes]
        # print(f"[backbone3d_semantic] semantic_logits_dense: {semantic_logits_dense.shape}")
        # print(f"[backbone3d_semantic] semantic_logits_dense max: {semantic_logits_dense.max()}, min: {semantic_logits_dense.min()}")
        
        # 存储到batch_dict中，供损失函数使用
        if agent is not None:
            batch_dict[agent]['semantic_logits'] = semantic_logits_dense
        else:
            batch_dict['semantic_logits'] = semantic_logits_dense

        return batch_dict

    def clip_gradients(self, max_norm=5.0):
        """梯度裁剪，防止梯度爆炸"""
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)


class Gaussian3DBackboneForPretraining(nn.Module):
    """
    Gaussian 3D Backbone - 预训练版本
    整合 DynamicVoxelVFE 和 GaussianBackbone3DForPretraining 的预训练流程
    只训练语义分类部分，不计算高斯参数
    """
    def __init__(self, model_cfg, **kwargs):
        super(Gaussian3DBackboneForPretraining, self).__init__()
        
        self.model_cfg = model_cfg
        
        # 从model_cfg中获取grid_size, voxel_size, point_cloud_range
        # 如果未提供则使用默认值
        # grid_size 语义为 [H, W, Z] = [y, x, z]
        H, W, Z = self.model_cfg.get('GRID_SIZE')
        self.grid_size_hwz = [H, W, Z]  # 保存为 [H, W, Z] 语义
        self.voxel_size =  self.model_cfg.get('VOXEL_SIZE')
        self.point_cloud_range =  self.model_cfg.get('POINT_CLOUD_RANGE')
        
        # 1. VFE配置
        vfe_cfg = self.model_cfg.get('VFE', {})
        vfe_cfg.setdefault('USE_NORM', True)
        vfe_cfg.setdefault('WITH_DISTANCE', False)
        vfe_cfg.setdefault('USE_ABSLOTE_XYZ', True)
        vfe_cfg.setdefault('NUM_FILTERS', [128, 128])
        vfe_cfg.setdefault('RETURN_ABS_COORDS', False)
        
        # 2. Backbone配置（预训练版本）
        backbone_cfg = self.model_cfg.get('BACKBONE_3D', {})
        backbone_cfg.setdefault('NUM_FEATURES', 128)
        backbone_cfg.setdefault('HIDDEN_DIM', 128)
        backbone_cfg.setdefault('NUM_CLASSES', 7)
        
        num_point_features = self.model_cfg.get('NUM_POINT_FEATURES', 4)  # x,y,z,intensity
        self.vfe = DynamicVoxelVFE(
            model_cfg=vfe_cfg,
            num_point_features=num_point_features,
            voxel_size=self.voxel_size,
            grid_size=[H, W, Z],
            point_cloud_range=self.point_cloud_range
        )
        
        # Backbone初始化（预训练版本）
        # Backbone 使用 [H, W, Z] 语义
        self.backbone = GaussianBackbone3DForPretraining(
            model_cfg=backbone_cfg, 
            grid_size=[H, W, Z],  # [H, W, Z] 语义
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        
        print(f"[Gaussian3DBackboneForPretraining] 初始化完成:")
        print(f"  - VFE Filters: {vfe_cfg['NUM_FILTERS']}")
        print(f"  - Backbone Features: {backbone_cfg.get('NUM_FEATURES')}")
        print(f"  - Grid Size: {self.grid_size_hwz} [H, W, Z]")
        print(f"  - Mode: Pretraining (semantic classification only)")
    
    def forward(self, batch_dict, available_agent=None, **kwargs):
        """
        预训练模式的前向传播流程
        
        Args:
            batch_dict: 包含点云数据的字典
                - 输入: batch_dict[agent]['origin_lidar']: [N, 4] 或 [N, 3]
                - 输出: batch_dict[agent]['semantic_logits']: [N_voxel, num_classes]
                        batch_dict[agent]['semantic_targets']: [N_voxel]
            available_agent: agent列表 ('vehicle', 'rsu', 'drone' 等)
        
        Returns:
            batch_dict: 更新后的字典，包含语义分类结果
        """
        for agent in available_agent:
            # Step 1: VFE处理 - 点云 → 体素特征
            batch_dict = self.vfe(batch_dict, agent=agent)
            
            # Step 2: Backbone处理 - 体素特征 → 语义分类
            batch_dict = self.backbone(batch_dict, agent=agent)
        
        return batch_dict
    
    def load_pretrained_weights(self, checkpoint_path, strict=True):
        """
        加载预训练权重到整个流程：VFE → Encoder → Semantic Head
        
        预训练包含的模块（包含可训练参数）：
        1. VFE (DynamicVoxelVFE): 点云 → 体素特征
           - 包含 pfn_layers 等可训练参数
        2. Encoder: 稀疏卷积编码器
           - 包含 SubMConv3d 和 SparseBatchNorm 等可训练参数
        3. Semantic Head: 语义分类头
           - 包含 SubMConv3d 等可训练参数
        
        注意：SparseConvTensor 只是数据格式转换（不包含参数），无需加载权重
        
        完整流程：
        原始点云 → VFE → SparseConvTensor(格式转换) → Encoder → Semantic Head → 语义logits
        
        Args:
            checkpoint_path (str): checkpoint文件路径
            strict (bool): 是否严格匹配所有键
        
        Returns:
            missing_keys (list): 缺失的键
            unexpected_keys (list): 意外的键
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 如果checkpoint包含model_state_dict，则提取它
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 获取当前模型的状态字典
        vfe_state_dict = self.vfe.state_dict()
        backbone_state_dict = self.backbone.state_dict()
        
        # 收集需要加载的权重
        vfe_pretrained = {}
        backbone_pretrained = {}
        
        for k, v in state_dict.items():
            # 匹配VFE的权重
            if k.startswith('vfe.'):
                new_key = k.replace('vfe.', '')
                if new_key in vfe_state_dict:
                    vfe_pretrained[new_key] = v
            elif not k.startswith('backbone.') and not k.startswith('encoder.') and not k.startswith('semantic_head.'):
                # 尝试直接匹配VFE的键（如果checkpoint中没有'vfe.'前缀）
                if k in vfe_state_dict:
                    vfe_pretrained[k] = v
            
            # 匹配backbone的encoder和semantic_head权重
            if k.startswith('backbone.encoder.') or k.startswith('backbone.semantic_head.'):
                # 去掉'backbone.'前缀
                new_key = k.replace('backbone.', '')
                if new_key in backbone_state_dict:
                    backbone_pretrained[new_key] = v
            elif k.startswith('encoder.') or k.startswith('semantic_head.'):
                # 直接匹配encoder或semantic_head的键
                if k in backbone_state_dict:
                    backbone_pretrained[k] = v
        
        # 加载VFE权重
        vfe_missing, vfe_unexpected = self.vfe.load_state_dict(vfe_pretrained, strict=strict)
        
        # 加载backbone权重
        backbone_missing, backbone_unexpected = self.backbone.load_state_dict(backbone_pretrained, strict=strict)
        
        # 合并缺失和意外的键
        missing_keys = list(set(vfe_missing + backbone_missing))
        unexpected_keys = list(set(vfe_unexpected + backbone_unexpected))
        
        print(f"[Gaussian3DBackboneForPretraining] 加载预训练权重:")
        print(f"  - Checkpoint: {checkpoint_path}")
        print(f"  - VFE加载的键数量: {len(vfe_pretrained)}")
        print(f"  - Backbone加载的键数量: {len(backbone_pretrained)}")
        print(f"  - 总加载键数量: {len(vfe_pretrained) + len(backbone_pretrained)}")
        if missing_keys:
            print(f"  - 缺失的键: {len(missing_keys)} 个")
            if len(missing_keys) <= 10:
                for key in missing_keys:
                    print(f"    - {key}")
        if unexpected_keys:
            print(f"  - 意外的键: {len(unexpected_keys)} 个")
            if len(unexpected_keys) <= 10:
                for key in unexpected_keys:
                    print(f"    - {key}")
        
        return missing_keys, unexpected_keys
