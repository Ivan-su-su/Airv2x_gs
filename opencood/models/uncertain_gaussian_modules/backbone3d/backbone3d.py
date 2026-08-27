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

from .dynamic_voxel_vfe import DynamicVoxelVFE
from .lion_backbone_one_stride import LION3DBackboneOneStride

class Gaussian3DBackbone(nn.Module):
    """
    Gaussian 3D Backbone - 主控制器类
    整合 DynamicVoxelVFE 和 GaussianBackbone3D 的完整点云高斯生成流程
    """
    def __init__(self, model_cfg, **kwargs):
        super(Gaussian3DBackbone, self).__init__()
        
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
        vfe_cfg.setdefault('USE_ABSOLUTE_XYZ', True)
        vfe_cfg.setdefault('NUM_FILTERS', [128, 128])
        vfe_cfg.setdefault('RETURN_ABS_COORDS', False)
        
        # 2. Backbone配置
        backbone_cfg = self.model_cfg.get('LION3DBackboneOneStride', {})
        backbone_cfg.setdefault('FEATURE_DIM', 128)
        backbone_cfg.setdefault('NUM_LAYERS', 12)
        backbone_cfg.setdefault('DEPTHS', [2, 2, 2, 2])
        backbone_cfg.setdefault('LAYER_DOWN_SCALES', [2, 2, 2, 2])
        backbone_cfg.setdefault('DIRECTION', 'forward')
        backbone_cfg.setdefault('DIFFUSION', False)
        backbone_cfg.setdefault('SHIFT', False)
        backbone_cfg.setdefault('DIFF_SCALE', 0.2)
        
        backbone_cfg.setdefault('WINDOW_SHAPE', [7, 7, 7])
        backbone_cfg.setdefault('GROUP_SIZE', 7)
        backbone_cfg.setdefault('LAYER_DIM', 128)
        backbone_cfg.setdefault('OPERATOR', 'multi_head_attn')
        backbone_cfg.setdefault('USE_PREBACKBONE', False)
        backbone_cfg.setdefault('RETURN_ABS_COORDS', False)
        
        backbone_cfg.setdefault('USE_HEIGHT_FIDELITY', False)
        backbone_cfg.setdefault('USE_INVERSE', False)
        backbone_cfg.setdefault('USE_CHECKPOINT', True)
        
        # VFE初始化
        # VFE 内部使用 [x, y, z] 语义，所以传 [W, H, Z]
        num_point_features = self.model_cfg.get('NUM_POINT_FEATURES', 4)  # x,y,z,intensity
        self.vfe = DynamicVoxelVFE(
            model_cfg=vfe_cfg,
            num_point_features=num_point_features,
            voxel_size=self.voxel_size,
            grid_size=[H, W, Z],
            point_cloud_range=self.point_cloud_range
        )
        
        # Backbone初始化
        # Backbone 使用 [H, W, Z] 语义
        self.backbone = LION3DBackboneOneStride(
            model_cfg=backbone_cfg,
            input_channels=num_point_features,
            grid_size=[H, W, Z]
        )
        
        print(f"[Gaussian3DBackbone] 初始化完成:")
        print(f"  - VFE Filters: {vfe_cfg['NUM_FILTERS']}")
        print(f"  - Backbone Features: {backbone_cfg.get('NUM_FEATURES')}")
        print(f"  - Grid Size: {self.grid_size_hwz} [H, W, Z]")
    
    
    def forward(self, batch_dict, available_agent=None, **kwargs):
        """
        完整的前向传播流程
        
        Args:
            batch_dict: 包含点云数据的字典
                - 输入: batch_dict[agent]['origin_lidar']: [N, 4] 或 [N, 3]
                - 输出: batch_dict[agent]['lidar_gaussians']: 高斯点字典
            agent: agent类型 ('vehicle', 'rsu', 'drone' 等)
        
        Returns:
            batch_dict: 更新后的字典，包含高斯点和TPV特征
        """
        for agent in available_agent:
            # Step 1: VFE处理 - 点云 → 体素特征
            batch_dict = self.vfe(batch_dict, agent=agent)
            
            # Step 2: Backbone处理 - 体素特征 → 高斯点
            batch_dict = self.backbone(batch_dict, agent=agent)
        
        # dict_keys(['batch_merged_lidar_features_torch', 'batch_merged_cam_inputs', 'record_len', 'batch_idxs',
        # 'instance_voxel_mask', 'instance_valid_mask', 'pillar_features', 'voxel_features', 'voxel_coords', 'voxel_num_points',
        # 'instance_voxel_mask_pre_backbone', 'encoded_spconv_tensor', 'encoded_spconv_tensor_stride',
        # 'multi_scale_3d_features', 'multi_scale_3d_strides'])
        
        return batch_dict