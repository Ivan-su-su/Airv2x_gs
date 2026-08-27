# -*- coding: utf-8 -*-
# Author: AI Assistant
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Gaussian Image Backbone for Multi-Agent Collaborative 3D Gaussian Perception System
实现图像特征提取、2D检测、深度预测和TPV投影的完整流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from efficientnet_pytorch import EfficientNet TODO
import torchvision.models as models
import numpy as np
import cv2

from opencood.utils.camera_utils import (
    QuickCumsum,
    bin_depths,
    cumsum_trick,
    depth_discretization,
    gen_dx_bx,
)
from opencood.utils.seg_label_utils import SegLabelMapper


# 默认配置模板
DEFAULT_MODEL_CFG = {
    'IMAGE_BACKBONE': 'SimpleCNN',
    'IMAGE_FEATURES': 128,
    'TPV_FEATURES': 64,
    'TPV_SIZE': [200, 704, 32],
    'PC_RANGE': [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
    'VOXEL_SIZE': [0.54, 0.54, 0.25],
    'DEPTH_BINS': 80,
    'DBOUND': [2.0, 50.0, 0.5],
    'TOP_K_DEPTHS': 20,
    'MASK_THRESHOLD': 0.5,
    'GAUSSIAN_THRESHOLD': 0,
    # === 新增语义检测相关默认配置 ===
    'NUM_CLASSES': 4,
    'EMPTY_CLASS_INDEX': 1,
    'TOPK_PIXELS': 1000,
    'GAUSSIAN_SCALE_RANGE': [0.1, 1.5],
    'USE_SPATIAL_ATTENTION': False,
    'USE_MORPHOLOGY': False,
    'AGENT_TYPES': ['vehicle', 'rsu', 'drone'],
    # === FPN 多尺度配置 ===
    'USE_FPN_MULTISCALE': True,
    'AGENT_FEATURE_SCALE': {
        'drone': 'P2',    # 64×176 高分辨率，用于小目标
        'vehicle': 'P3',  # 32×88
        'rsu': 'P3',      # 32×88
    },
    'IMAGE_SHAPE_P2': [64, 176],
    'IMAGE_SHAPE_P3': [32, 88],
}

class GaussianImageBackbone(nn.Module):
    """
    高斯感知系统的图像backbone
    实现完整的图像处理流程：特征提取 -> 2D检测 -> 深度预测 -> TPV投影
    """
    def __init__(self, model_cfg=None):
        super(GaussianImageBackbone, self).__init__()
        
        # 如果model_cfg为None，使用默认配置
        if model_cfg is None:
            import copy
            model_cfg = copy.deepcopy(DEFAULT_MODEL_CFG)
        
        self.model_cfg = model_cfg
        self.grid_size = model_cfg.get('GRID_SIZE')
        self.voxel_size = model_cfg.get('VOXEL_SIZE')
        self.point_cloud_range = model_cfg.get('POINT_CLOUD_RANGE')
        self.image_shape = model_cfg.get('IMAGE_SHAPE', [32, 88])   #TODO: 可删？
        
        # FPN 多尺度配置
        self.use_fpn_multiscale = model_cfg.get('USE_FPN_MULTISCALE')
        self.image_shape_p2 = model_cfg.get('IMAGE_SHAPE_P2', [64, 176])
        self.image_shape_p3 = model_cfg.get('IMAGE_SHAPE_P3', [32, 88])
        if self.use_fpn_multiscale:
            print(f"[Backbone2D] FPN multi-scale enabled: drone→P2{self.image_shape_p2}, vehicle/rsu→P3{self.image_shape_p3}")
        
        # 1. 图像特征提取backbone
        self.image_backbone = GaussianImageFeatureExtractor(model_cfg)
        
        # 2. 2D检测头（类似YOLO）
        self.detection_head = GaussianDetectionHead(model_cfg)
        
        # 初始化语义标签映射器（用于从真实世界坐标查询标签）
        seg_hw = model_cfg.get('seg_hw', 512)

        # 图片语义真值配置
        self.use_image_semantic_gt = model_cfg.get('USE_IMAGE_SEMANTIC_GT', True)
        print(f"[Backbone2D] Use image semantic GT: {self.use_image_semantic_gt}")
        
        # dwb
        # 可视化开关（用于调试投影是否正确）
        self.visualize_projection = model_cfg.get('visualize_projection', True)
        self.visualization_save_dir = model_cfg.get('visualization_save_dir', './visualization/lidar_projection')
        seg_res = model_cfg.get('seg_res', 0.25)
        pc_range = model_cfg.get('POINT_CLOUD_RANGE', self.point_cloud_range)
        self.seg_label_mapper = SegLabelMapper(
            seg_hw=seg_hw,
            seg_res=seg_res,
            lidar_range=pc_range,
            ego_center=True  # 假设标签图以ego为中心
        )

    def forward(self, batch_dict, available_agent):
        """
        完整的前向传播流程 - 双分辨率架构
        Args:
            batch_dict: 包含多Agent图像数据的字典
        Returns:
            dict: 包含TPV特征的输出字典
        """
        agent_idx = {}
        count = 0
        for agent in available_agent:
            agent_idx[agent] = count
            count += batch_dict[agent]['record_len'].item()
            
        # 处理每个Agent的图像数据，分别存储
        self.agent_types = available_agent
        for agent_type in self.agent_types:
            
            # 0127
            # if agent_type == 'rsu' and 'batch_merged_cam_inputs' in batch_dict[agent_type]:
            if agent_type in batch_dict and 'batch_merged_cam_inputs' in batch_dict[agent_type]:
                agent_data = batch_dict[agent_type]
                camera_num = batch_dict[agent_type]['batch_merged_cam_inputs']['imgs'].shape[1]
                # 1. 图像特征提取（FPN 多尺度或单尺度）
                
                image_feature = self.image_backbone(agent_data, agent_type=agent_type)
                
                B, N, C_feat, H, W = image_feature.shape
                image_features_5d = image_feature.view(1, B*N, -1, H, W)
                
                # 2. 获取语义logits（用于损失计算）
                # 从detection_head内部获取logits
                semantic_logits_dense = image_feature.view(B*N, -1, H, W)
                semantic_logits = self.detection_head.lightweight_cls_head(semantic_logits_dense)  # [B*N, M, H, W]
                # print(f"semantic_logits: {semantic_logits.shape}")
                # print(f"semantic_logits: {semantic_logits[0, :, :, :].max()}")
                # print(f"semantic_logits: {semantic_logits[0, :, :, :].min()}")
                
                
                # 3. 构建语义监督标签（feat_h=H, feat_w=W 与当前尺度一致）
                # 判断使用图片语义真值还是从LiDAR投影
                if self.use_image_semantic_gt:
                    # 方法1：从.bin文件读取的图片语义真值
                    semantic_targets = self._build_semantic_supervision_from_image_gt(
                        batch_dict, agent_type, B, N, feat_h=H, feat_w=W
                    )
                    
                    batch_dict[agent_type]['semantic_targets'] = semantic_targets
                    batch_dict[agent_type]['semantic_logits'] = semantic_logits
                
                else:
                    # 方法2：从LiDAR点投影的语义真值（原有方法）
                    # 获取相机参数（参考 airv2x_encoder.py 的投影方式）
                    cam_inputs = agent_data['batch_merged_cam_inputs']
                    intrinsics = cam_inputs['intrinsics'].view(B, N, 3, 3)  # [B, N, 3, 3]
                    extrinsics = cam_inputs['extrinsics'].view(B, N, 4, 4)  # [B, N, 4, 4]
                    rots = cam_inputs['rots'].view(B, N, 3, 3)  # [B, N, 3, 3] 相机到agent本地lidar坐标系的旋转
                    trans = cam_inputs['trans'].view(B, N, 3)  # [B, N, 3] 相机到agent本地lidar坐标系的平移
                    post_rots = cam_inputs['post_rots'].view(B, N, 3, 3)  # [B, N, 3, 3] 数据增强的旋转
                    post_trans = cam_inputs['post_trans'].view(B, N, 3)  # [B, N, 3] 数据增强的平移
                    
                    # 获取从agent本地坐标系到ego坐标系的变换矩阵
                    agent_to_ego_transform = batch_dict['img_pairwise_t_matrix_collab'][0,agent_idx[agent_type]:agent_idx[agent_type]+batch_dict[agent_type]['record_len'],0,:,:]
                    # [B, 4, 4]
                    # img_pairwise_t_matrix_collab: [B, L, L, 4, 4]
                    # img_pairwise_t_matrix_collab[batch_idx, i, j, :, :] 表示从agent i到agent j的变换矩阵
                    # img_pairwise_t_matrix_collab[0, agent_idx, 0, :, :] 表示从agent到ego(索引0)的变换
                    agent_to_ego_transform = agent_to_ego_transform.unsqueeze(0)
                    agent_to_ego_transform = agent_to_ego_transform.repeat_interleave(camera_num, dim=1)
                    # [1, B*N, 4, 4] 同一个重复N次

                    label_dict = batch_dict.get('label_dict', None)
                    if label_dict is not None:
                        # 获取LiDAR点
                        # 0127
                        # lidar_key = 'origin_lidar'
                        if agent_type == 'vehicle' or agent_type is None:
                            lidar_key = 'origin_lidar'
                        else:
                            lidar_key = f'origin_lidar_{agent_type}'
                        lidar_points = batch_dict.get(lidar_key, None)
                        
                        if lidar_points is not None:
                            # 处理lidar_points的形状
                            if lidar_points.dim() == 3 and lidar_points.shape[0] == 1:
                                lidar_points = lidar_points.squeeze(0)  # [N_lidar, 4]
                            if lidar_points.shape[1] == 4:
                                lidar_points = lidar_points[:, :3]
                            
                            ########## 查询LiDAR点的语义标签
                            lidar_labels, lidar_valid_mask = self.query_semantic_labels(
                                lidar_points, label_dict, label_type='dynamic'
                            )
                            print(f"lidar_points: {lidar_points.shape}")
                            
                            if lidar_labels is not None and lidar_valid_mask.any():
                                # 只使用有效的LiDAR点
                                valid_lidar_points = lidar_points[lidar_valid_mask]  # [N_valid, 3]
                                print(f"valid_lidar_points: {valid_lidar_points.shape}")
                                valid_lidar_labels = lidar_labels[lidar_valid_mask]  # [N_valid]
                                
                                ########## 生成语义监督标签
                                semantic_targets = self._build_semantic_supervision_from_lidar(
                                    batch_dict, agent_type,
                                    valid_lidar_points, valid_lidar_labels,
                                    intrinsics, extrinsics, 
                                    post_rots, post_trans,
                                    agent_to_ego_transform,
                                    H, W  # feat_h, feat_w
                                )  # [B, N, H, W]
                                semantic_targets = semantic_targets.view(1, B * N, H, W)
                                
                                # 存储到batch_dict
                                batch_dict[agent_type]['semantic_targets'] = semantic_targets
                                batch_dict[agent_type]['semantic_logits'] = semantic_logits
        
        return batch_dict

    
    def query_semantic_labels(self, world_coords, label_dict, label_type='dynamic'):
        """
        查询给定3D点对应的语义标签
        
        Args:
            world_coords: [N, 3] 或 [B, N, 3] 真实世界坐标 (x, y, z)
            label_dict: 包含 'dynamic_seg_label' 或 'static_seg_label' 的字典
            label_type: 'dynamic' 或 'static'，指定查询哪种标签
        
        Returns:
            labels: [N] 或 [B, N] 语义标签
            valid_mask: [N] 或 [B, N] bool tensor，表示点是否在标签图范围内
        """
        label_key = f'{label_type}_seg_label'
        seg_label = label_dict.get(label_key, None)
        
        if seg_label is None:
            return None, None
        
        labels, valid_mask = self.seg_label_mapper.query_labels(
            seg_label=seg_label,
            world_coords=world_coords,
            default_label=0
        )
        
        return labels, valid_mask
    
    def _project_lidar_to_image(self, lidar_points, intrinsics, extrinsics, 
                                 post_rots, post_trans, agent_to_ego_transform,
                                 img_h, img_w):
        """
        将LiDAR点云投影到图像平面
        
        Args:
            lidar_points: [N_lidar, 3] ego坐标系下的LiDAR点
            intrinsics: [B, N, 3, 3] 相机内参
            extrinsics: [B, N, 4, 4] 相机外参
            post_rots: [B, N, 3, 3] 图像增强旋转矩阵
            post_trans: [B, N, 3] 图像增强平移向量
            agent_to_ego_transform: [B, N, 4, 4] agent到ego的变换矩阵
            img_h, img_w: 图像的高度和宽度
            
        Returns:
            pixel_coords: [B, N, N_lidar, 2] 归一化图像坐标 (u, v)，范围 [0, 1]
            valid_mask: [B, N, N_lidar] bool tensor，表示点是否有效
            depth_values: [B, N, N_lidar] 深度值
        """
        B, N = intrinsics.shape[:2]
        BN = B * N
        device = lidar_points.device
        N_lidar = lidar_points.shape[0]

        print(f"  B={B}, N={N}, N_lidar={N_lidar}")
        # Debug: 打印输入形状
        print(f"  intrinsics: {intrinsics.shape}")   #[B, N, 3, 3]
        print(f"  extrinsics: {extrinsics.shape}")   #[B, N, 4, 4]
        print(f"  post_rots: {post_rots.shape}")   #[B, N, 3, 3]
        print(f"  post_trans: {post_trans.shape}")   #[B, N, 3]
        if agent_to_ego_transform is not None:
            print(f"  agent_to_ego_transform: {agent_to_ego_transform.shape}")   #[1, B*N, 4, 4]
            # print(f"  agent_to_ego_transform: {agent_to_ego_transform[0, :, :, :]}")
        else:
            print(f"  agent_to_ego_transform: None")

        if intrinsics.shape[-2:] != (3, 3):
            # 将intrinsics转换为3x3矩阵
            intrinsics_3 = intrinsics[:, :, :3, :3]
        else:
            intrinsics_3 = intrinsics

        if extrinsics.shape[-2:] != (4, 4):
            extrinsics_4 = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
            extrinsics_4[:, :, :3, :3] = extrinsics
            extrinsics_4[:, :, :3, 3] = extrinsics[:, :, :, 3] if extrinsics.dim() == 4 else extrinsics[:, :, 3]
        else:
            extrinsics_4 = extrinsics
        # 对每个[4,4]的外参都取逆
        extrinsics_4 = torch.inverse(extrinsics_4)
        print(f"  extrinsics_4: {extrinsics_4.shape}")
        # print(f"  extrinsics_4: {extrinsics_4[0, :, :, :]}")


        if post_rots.shape[-2:] == (3, 3):
            post_rots_4 = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
            post_rots_4[:, :, :3, :3] = post_rots
            post_rots_4[:, :, :3, 3] = 0
        else:
            post_rots_4 = post_rots
        if post_trans.shape[-1] == 3:
            post_trans_4 = torch.zeros(B, N, 4, device=device)
            post_trans_4[:, :, :3] = post_trans
            post_trans_4[:, :, 3] = 1
        else:
            post_trans_4 = post_trans

        img_aug_matrix = post_rots_4.view(1, -1, 4, 4)

        # 1.从ego到agent，从agent到camera
        extrinsics_4 = extrinsics_4.view(1, -1, 4, 4)


        if agent_to_ego_transform is not None:
            ego_to_agent = torch.inverse(agent_to_ego_transform)   #[1, B*N, 4, 4]
            # print("ego_to_agent: ", ego_to_agent.shape)   #[1, B*N, 4, 4]
            # print("ego_to_agent: ", ego_to_agent.shape)   #[1, B*N, 4, 4]
            lidar2cam = torch.matmul(extrinsics_4[0,:,:,:], ego_to_agent)
        # lidar2cam = extrinsics_4
        
        ones = torch.ones((N_lidar, 1), device=device)
        lidar_points_homo = torch.cat([lidar_points, ones], dim=1)  # [N_lidar, 4]

        # [1, BN, N_lidar, 4]
        lidar_points_homo = lidar_points_homo.view(1, 1, N_lidar, 4).repeat(1, BN, 1, 1)

        points_cam = torch.matmul(
            lidar2cam.unsqueeze(2),              # [1, BN, 1, 4, 4]
            lidar_points_homo.unsqueeze(-1)         # [1, BN, N_lidar, 4, 1]
        ).squeeze(-1)                               # [1, BN, N_lidar, 4]

        # 将points_cam转换为对应的OpenCV相机坐标系
        points_cam_xyz = points_cam[..., :3]  # [1, BN, N_lidar, 3]
        # 分别取y, -z, x
        # dwb 不需要重新排列
        points_cam_reordered = points_cam_xyz
        # points_cam_reordered = torch.stack([
        #     points_cam_xyz[..., 1],              # y
        #     -points_cam_xyz[..., 2],             # -z
        #     points_cam_xyz[..., 0]               # x
        # ], dim=-1)  # [1, BN, N_lidar, 3]

        # 2.从camera到image
        points_img = torch.matmul(
            intrinsics_3.view(1, -1, 3, 3).unsqueeze(2),            # [1, BN, 1, 3, 3]
            points_cam_reordered.unsqueeze(-1)         # [1, BN, N_lidar, 3, 1]
        ).squeeze(-1)                               # [1, BN, N_lidar, 3]

        depth = points_img[..., 2]
        eps = 1e-5
        valid_mask = depth > eps
        
        u_ori = points_img[..., 0] / torch.clamp(depth, min=eps)
        v_ori = points_img[..., 1] / torch.clamp(depth, min=eps)
        
        point_img = torch.stack(
            [u_ori, v_ori, torch.ones_like(u_ori), torch.ones_like(u_ori)],
            dim=-1
        )  # [1, BN, N_lidar, 4]

        # dwb 不需要应用增强矩阵
        point_img_aug = point_img
        # point_img_aug = torch.matmul(
        #     img_aug_matrix.unsqueeze(2),    # [1, BN, 1, 4, 4]
        #     point_img.unsqueeze(-1)
        # ).squeeze(-1)                       # [1, BN, N_lidar, 4]

        u = point_img_aug[..., 0] / 1280
        v = point_img_aug[..., 1] / 720

        pixel_coords = torch.stack([u, v], dim=-1)  # [1, BN, N_lidar, 2]
        print(f"  pixel_coords: {pixel_coords.shape}")   #[1, BN, N_lidar, 2]
        # 将无效点坐标设为一个特殊值（例如-1），保持形状不变
        pixel_coords = torch.where(
            valid_mask.unsqueeze(-1), 
            pixel_coords, 
            torch.full_like(pixel_coords, 0.0)
        )
        print(f"  pixel_coords: {pixel_coords.shape}")   #[B, N, N_lidar, 2]
        pixel_coords = pixel_coords.view(B, N, N_lidar, 2)
        
        print(f"  pixel_coords[0, 0, :, :].max(): {pixel_coords[0, 0, :, :].max()}, pixel_coords[0, 0, :, :].min():  {pixel_coords[0, 0, :, :].min()}")

        valid_mask = valid_mask & \
                    (u >= 0) & (u <= 1) & \
                    (v >= 0) & (v <= 1)
        valid_mask = valid_mask.view(B, N, N_lidar)
        print(f"  valid_mask: {valid_mask.shape}")   #[B, N, N_lidar]

        depth = depth.view(B, N, N_lidar)
        
        return pixel_coords, valid_mask, depth
    
    def _map_pixel_to_feature(self, pixel_coords, valid_mask, img_h, img_w, feat_h, feat_w):
        """
        将归一化图像坐标映射到低分辨率特征图坐标
        参考 unitr_utils.py 第356-359行的实现
        
        Args:
            pixel_coords: [B, N, N_lidar, 2] 归一化图像坐标 (u, v)，范围 [0, 1]
            valid_mask: [B, N, N_lidar] bool tensor，表示点是否有效
            img_h, img_w: 原始图像的高度和宽度（用于文档说明，实际不使用）
            feat_h, feat_w: 特征图的高度和宽度
        
        Returns:
            feat_coords: [B, N, N_lidar, 2] 特征图坐标 (u_feat, v_feat)
            valid_mask_feat: [B, N, N_lidar] bool tensor，映射后的有效mask
        """
        B, N, N_lidar = valid_mask.shape
        device = pixel_coords.device
        
        # 参考 unitr_utils.py 第356-359行：直接将归一化坐标乘以特征图尺寸
        # lidar2image_coords_xyz[:, 0] = lidar2image_coords_xyz[:, 0] * hw_shape[1]
        # lidar2image_coords_xyz[:, 1] = lidar2image_coords_xyz[:, 1] * hw_shape[0]
        u_feat = (pixel_coords[:, :, :, 0] * feat_w).long()  # [B, N, N_lidar]
        v_feat = (pixel_coords[:, :, :, 1] * feat_h).long()  # [B, N, N_lidar]
        
        # 确保坐标在特征图范围内
        u_feat = torch.clamp(u_feat, 0, feat_w - 1)
        v_feat = torch.clamp(v_feat, 0, feat_h - 1)
        
        feat_coords = torch.stack([u_feat, v_feat], dim=-1)  # [B, N, N_lidar, 2]
        
        # 更新有效mask（确保在特征图范围内）
        valid_mask_feat = valid_mask & (
            (u_feat >= 0) & (u_feat < feat_w) &
            (v_feat >= 0) & (v_feat < feat_h)
        )
        
        return feat_coords, valid_mask_feat
    
    def _visualize_lidar_projection(self, batch_dict, agent_type, pixel_coords, valid_mask, 
                                     save_dir=None, save_prefix="lidar_projection", max_vis=5):
        """
        可视化LiDAR点投影到图像上的结果
        
        注意：pixel_coords 是归一化坐标 [0, 1]，需要转换为像素坐标才能可视化。
        
        Args:
            batch_dict: batch字典
            agent_type: agent类型 ('vehicle', 'rsu', 'drone')
            pixel_coords: [B, N, N_lidar, 2] 归一化图像坐标 (u, v)，范围 [0, 1]
            valid_mask: [B, N, N_lidar] bool tensor，表示点是否在图像范围内
            save_dir: 保存目录，如果为None则只打印信息不保存
            save_prefix: 保存文件前缀
            max_vis: 最多可视化的图像数量（每个batch最多可视化max_vis张图像）
        """
        if 'batch_merged_cam_inputs' not in batch_dict.get(agent_type, {}):
            print(f"[Warning] Cannot visualize: missing batch_merged_cam_inputs for {agent_type}")
            return
        
        # 优先使用原始图像（未经过变换）
        use_original = True
        cam_inputs = batch_dict[agent_type]['batch_merged_cam_inputs']
        # TODO：dwb
        if use_original:
            original_imgs = cam_inputs['original_imgs']  # [B, N, 3, H_orig, W_orig]
            print(f"[Visualization] Using original images (no augmentation): {original_imgs.shape}")
        else:
            original_imgs = None
            imgs = cam_inputs['imgs']  # [B, N, C, H, W]
            print(f"[Visualization] Using augmented images: {imgs.shape}")
        
        B, N = pixel_coords.shape[:2]
        
        # 限制可视化数量
        B_vis = B  #min(B, max_vis)
        
        for b in range(B_vis):
            for n in range(N):
                if use_original:
                    # 使用原始图像（未经过变换，值范围[0, 1]）
                    img_tensor = original_imgs[b, n]  # [3, H_orig, W_orig]
                    img_rgb = img_tensor  # 已经是RGB格式
                    orig_h, orig_w = img_rgb.shape[1], img_rgb.shape[2]
                    
                    # 原始图像不需要反归一化，直接使用
                    # 转换为numpy并调整维度顺序 [H, W, 3]
                    img = img_rgb.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
                    
                    # 转换为[0, 255]范围的uint8
                    img = (img * 255).astype(np.uint8)
                    
                    # 将归一化坐标转换为原始图像的像素坐标
                    pixel_coords_normalized = pixel_coords[b, n, valid_mask[b, n]]  # [N_valid, 2]
                    pixel_coords_pixel = pixel_coords_normalized.clone()
                    pixel_coords_pixel[:, 0] *= orig_w  # u坐标：归一化 -> 像素
                    pixel_coords_pixel[:, 1] *= orig_h  # v坐标：归一化 -> 像素
                    
                    # 检查坐标是否在原始图像范围内
                    valid_mask_scaled = (
                        (pixel_coords_pixel[:, 0] >= 0) & 
                        (pixel_coords_pixel[:, 0] < orig_w) &
                        (pixel_coords_pixel[:, 1] >= 0) & 
                        (pixel_coords_pixel[:, 1] < orig_h)
                    )
                    valid_pixels_to_use = pixel_coords_pixel[valid_mask_scaled].cpu().numpy()
                    # dwb 这里valid_idx_to_use有误
                    valid_idx_to_use = valid_mask[b, n].cpu().numpy()
                    # valid_idx_to_use = valid_idx_to_use & valid_mask_scaled.cpu().numpy() 会报错，因为valid_mask_scaled已经被缩短过了
                else:
                    # 使用增强后的图像
                    img_tensor = imgs[b, n]  # [4, H, W] 或 [C, H, W]
                    
                    # 取前3个通道（RGB）
                    if img_tensor.shape[0] >= 3:
                        img_rgb = img_tensor[:3, :, :]  # [3, H, W]
                    else:
                        # 如果通道数不足3，复制通道
                        img_rgb = img_tensor.repeat(3, 1, 1)  # [3, H, W]
                    
                    # 反归一化：ImageNet归一化 (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_rgb.device)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_rgb.device)
                    img_denorm = img_rgb * std + mean  # 反归一化
                    
                    # 限制到[0, 1]范围
                    img_denorm = torch.clamp(img_denorm, 0, 1)
                    
                    # 转换为numpy并调整维度顺序 [H, W, 3]
                    img = img_denorm.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
                    
                    # 转换为[0, 255]范围的uint8
                    img = (img * 255).astype(np.uint8)
                    
                    # 获取图像尺寸
                    img_h_vis, img_w_vis = img.shape[:2]
                    
                    # 将归一化坐标转换为像素坐标
                    # pixel_coords 是归一化坐标 [0, 1]，需要乘以图像尺寸
                    pixel_coords_normalized = pixel_coords[b, n, valid_mask[b, n]]  # [N_valid, 2]
                    pixel_coords_pixel = pixel_coords_normalized.clone()
                    pixel_coords_pixel[:, 0] *= img_w_vis  # v坐标：归一化 -> 像素
                    pixel_coords_pixel[:, 1] *= img_h_vis  # u坐标：归一化 -> 像素
                    valid_pixels_to_use = pixel_coords_pixel.cpu().numpy()
                    valid_idx_to_use = valid_mask[b, n].cpu().numpy()
                
                # 转换为BGR格式（OpenCV使用BGR）
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_vis = img_bgr.copy()
                
                # 使用处理后的投影点和mask
                if valid_idx_to_use.any():
                    # 检查投影坐标范围
                    u_min, u_max = valid_pixels_to_use[:, 0].min(), valid_pixels_to_use[:, 0].max()
                    v_min, v_max = valid_pixels_to_use[:, 1].min(), valid_pixels_to_use[:, 1].max()
                    img_h_vis, img_w_vis = img_vis.shape[:2]
                    print(f"[Visualization] {agent_type} batch={b} camera={n}: "
                          f"image size={img_h_vis}x{img_w_vis}, "
                          f"pixel_coords (pixel) u range=[{u_min:.1f}, {u_max:.1f}], "
                          f"v range=[{v_min:.1f}, {v_max:.1f}], "
                          f"valid points={valid_idx_to_use.sum()}")
                    
                    # 绘制小红点（OpenCV使用BGR格式，红色是(0, 0, 255)）
                    # 注意：pixel_coords是(u, v)格式，其中u是列坐标（x），v是行坐标（y）
                    # OpenCV的circle函数使用(x, y)格式，所以(u, v)对应(x, y)
                    points_in_image = 0
                    points_out_of_range = 0
                    for pixel in valid_pixels_to_use:
                        u, v = float(pixel[0]), float(pixel[1])
                        u_int, v_int = int(round(u)), int(round(v))
                        # 确保坐标在图像范围内
                        # u对应列（宽度），v对应行（高度）
                        if 0 <= u_int < img_w_vis and 0 <= v_int < img_h_vis:
                            # 绘制红色圆点（半径3像素，更明显）
                            cv2.circle(img_vis, (u_int, v_int), 3, (0, 0, 255), -1)
                            points_in_image += 1
                        else:
                            points_out_of_range += 1
                    
                    if points_out_of_range > 0:
                        print(f"[Visualization] Warning: {points_out_of_range} points out of image range")
                    print(f"[Visualization] {agent_type} batch={b} camera={n}: "
                          f"{points_in_image}/{valid_idx_to_use.sum()} points drawn in image")
                
                # 保存图像（已经是BGR格式，直接保存）
                if save_dir is not None:
                    import os
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{save_prefix}_{agent_type}_b{b}_c{n}.png")
                    cv2.imwrite(save_path, img_vis)
                    print(f"[Visualization] Saved to {save_path}")
                else:
                    # 如果没有指定保存目录，只打印信息
                    print(f"[Visualization] {agent_type} batch={b} camera={n}: "
                          f"image shape={img_vis.shape}, valid points={valid_idx.sum()}")
    
    def _build_semantic_supervision_from_lidar(self, batch_dict, agent_type, 
                                                lidar_points, lidar_labels,
                                                intrinsics, extrinsics, 
                                                post_rots, post_trans,
                                                agent_to_ego_transform,
                                                feat_h, feat_w):
        """
        从LiDAR点和标签生成图像特征图的语义监督标签
        
        Args:
            batch_dict: batch字典
            agent_type: agent类型 ('vehicle', 'rsu', 'drone')
            lidar_points: [N_lidar, 3] ego坐标系下的LiDAR点
            lidar_labels: [N_lidar] LiDAR点的语义标签
            intrinsics: [B, N, 3, 3] 相机内参
            extrinsics: [B, N, 4, 4] 相机外参
            post_rots: [B, N, 3, 3] 数据增强旋转
            post_trans: [B, N, 3] 数据增强平移
            agent_to_ego_transform: [B, N, 4, 4] agent到ego的变换矩阵
            feat_h, feat_w: 特征图的高度和宽度
        
        Returns:
            semantic_targets: [B, N, feat_h, feat_w] 语义标签图
        """
        B, N = intrinsics.shape[:2]
        device = intrinsics.device
        
        # 获取原始图像分辨率
        # 从batch_dict中获取图像尺寸
        if 'batch_merged_cam_inputs' in batch_dict.get(agent_type, {}):
            # TODO: dwb
            imgs = batch_dict[agent_type]['batch_merged_cam_inputs']['imgs']
            img_h, img_w = imgs.shape[-2:]  # [B, N, C, H, W]
            # TODO: 这里imgs是256x704，与原始图像分辨率不一致，需要查看读取数据时是如何处理的
        else:
            # 默认分辨率（如果无法获取）
            img_h, img_w = 720, 1280
            print(f"[Warning] Cannot get image size from batch_dict, using default {img_h}x{img_w}")
        
        # Step 1: 投影LiDAR点到图像平面
        pixel_coords, valid_mask, depth_values = self._project_lidar_to_image(
            lidar_points, intrinsics, extrinsics, post_rots, post_trans,
            agent_to_ego_transform, img_h, img_w
        )  # pixel_coords: [B, N, N_lidar, 2], valid_mask: [B, N, N_lidar]
        
        # dwb
        # 可视化投影结果（可选，用于调试和验证投影是否正确）
        # 可视化会将valid_lidar_points投影到图像上，生成小红点
        # 可以通过设置model_cfg['visualize_projection']=True来启用
        if self.visualize_projection:
            # 获取实际图像尺寸用于可视化
            self._visualize_lidar_projection(
                batch_dict, agent_type, pixel_coords, valid_mask,
                save_dir=self.visualization_save_dir,
                save_prefix=f"lidar_proj_{agent_type}",
                max_vis=2  # 每个batch最多可视化2张图像
            )
        exit()
            
        
        # Step 2: 映射到特征图坐标
        feat_coords, valid_mask_feat = self._map_pixel_to_feature(
            pixel_coords, valid_mask, img_h, img_w, feat_h, feat_w
        )  # feat_coords: [B, N, N_lidar, 2], valid_mask_feat: [B, N, N_lidar]
        
        # Step 3: 生成语义标签图
        # 获取类别数量（用于统计每个标签的出现次数）
        num_classes = self.model_cfg.get('NUM_CLASSES', 7)
        semantic_targets = torch.zeros(B, N, feat_h, feat_w, dtype=torch.long, device=device)
        
        # 扩展lidar_labels到 [B, N, N_lidar]
        lidar_labels_expanded = lidar_labels.unsqueeze(0).unsqueeze(0).expand(B, N, -1)  # [B, N, N_lidar]
        
        # 为每个batch和相机填充标签
        for b in range(B):
            for n in range(N):
                # 获取当前相机有效的点和标签
                valid_idx = valid_mask_feat[b, n]  # [N_lidar]
                if not valid_idx.any():
                    continue
                
                valid_feat_coords = feat_coords[b, n, valid_idx]  # [N_valid, 2]
                valid_labels = lidar_labels_expanded[b, n, valid_idx]  # [N_valid]
                
                # 过滤掉标签为0的点（背景类）
                non_zero_mask = valid_labels != 0  # [N_valid]
                if not non_zero_mask.any():
                    continue  # 如果没有非0标签的点，跳过
                
                # 只保留非0标签的点
                valid_feat_coords_filtered = valid_feat_coords[non_zero_mask]  # [N_nonzero, 2]
                valid_labels_filtered = valid_labels[non_zero_mask]  # [N_nonzero]
                
                u_feat = valid_feat_coords_filtered[:, 0]  # [N_nonzero]
                v_feat = valid_feat_coords_filtered[:, 1]  # [N_nonzero]
                
                # 计算一维索引
                flat_idx = v_feat * feat_w + u_feat  # [N_nonzero] 一维索引
                
                # 创建标签计数tensor: [feat_h * feat_w, num_classes]
                # 用于统计每个像素位置每个标签的出现次数
                label_counts = torch.zeros(feat_h * feat_w, num_classes, dtype=torch.float32, device=device)
                
                # 使用scatter_add统计每个像素位置每个标签的出现次数
                # 为每个点创建一个one-hot向量，然后累加到对应的像素位置
                # 将标签转换为one-hot编码: [N_nonzero, num_classes]
                labels_onehot = torch.zeros(len(valid_labels_filtered), num_classes, dtype=torch.float32, device=device)
                labels_onehot.scatter_(1, valid_labels_filtered.unsqueeze(1).long(), 1.0)
                
                # 使用scatter_add将one-hot向量累加到对应的像素位置
                # 对于每个像素位置，累加所有映射到该位置的点的标签计数
                for i in range(len(flat_idx)):
                    pixel_idx = flat_idx[i]
                    label_counts[pixel_idx] += labels_onehot[i]
                
                # 对每个像素位置，选择出现次数最多的标签（argmax）
                # 如果某个像素位置没有非0标签的点，则保持为0（背景类）
                max_counts, argmax_labels = label_counts.max(dim=1)  # [feat_h * feat_w]
                
                # 只有当最大计数大于0时，才更新标签（即该像素位置至少有一个非0标签的点）
                update_mask = max_counts > 0  # [feat_h * feat_w]
                semantic_targets[b, n].view(-1)[update_mask] = argmax_labels[update_mask].long()
        
        return semantic_targets

    def _build_semantic_supervision_from_image_gt(self, batch_dict, agent_type, B, N, feat_h, feat_w):
        """
        从.bin文件读取的图片语义真值构建监督标签
        
        Args:
            batch_dict: batch字典
            agent_type: agent类型 ('vehicle', 'rsu', 'drone')
            B: batch size
            N: 相机数量
            feat_h, feat_w: 特征图的高度和宽度
        
        Returns:
            semantic_targets: [1, B*N, feat_h, feat_w] 语义标签图，如果没有真值则返回None
        """
        # 从batch_dict中获取相机输入
        if agent_type not in batch_dict or 'batch_merged_cam_inputs' not in batch_dict[agent_type]:
            print(f"[Warning] No camera inputs found for agent type: {agent_type}")
            return None
        
        cam_inputs = batch_dict[agent_type]['batch_merged_cam_inputs']
        
        # 检查是否有图片语义真值
        if 'image_semantic_gts' not in cam_inputs:
            print(f"[Warning] No image semantic GT found for agent type: {agent_type}")
            return None
        
        image_semantic_gts = cam_inputs['image_semantic_gts']  # [B, N, H_aug, W_aug]
        # 检查形状
        if image_semantic_gts.dim() == 3:
            # [B*N, H, W] -> [B, N, H, W]
            image_semantic_gts = image_semantic_gts.view(B, N, image_semantic_gts.shape[1], image_semantic_gts.shape[2])
        elif image_semantic_gts.dim() == 4:
            # 已经是 [B, N, H, W]
            pass
        else:
            print(f"[Error] Unexpected shape for image_semantic_gts: {image_semantic_gts.shape}")
            return None
        
        device = image_semantic_gts.device
        B_gt, N_gt, H_aug, W_aug = image_semantic_gts.shape
        
        # 检查尺寸是否匹配
        if B_gt != B or N_gt != N:
            print(f"[Warning] Batch/Camera number mismatch: expected B={B}, N={N}, got B={B_gt}, N={N_gt}")
            return None
        
        # 将语义真值从增强后的图像尺寸 resize 到特征图尺寸
        # image_semantic_gts: [B, N, H_aug, W_aug] -> [B*N, H_aug, W_aug]
        image_semantic_gts_flat = image_semantic_gts.view(B * N, H_aug, W_aug)
        
        # 使用最近邻插值（保持标签的离散性）
        # [B*N, H_aug, W_aug] -> [B*N, 1, H_aug, W_aug] -> [B*N, 1, feat_h, feat_w] -> [B*N, feat_h, feat_w]
        semantic_targets = F.interpolate(
            image_semantic_gts_flat.unsqueeze(1).float(),
            size=(feat_h, feat_w),
            mode='nearest'
        ).squeeze(1).long()
        
        # reshape 成 [1, B*N, feat_h, feat_w] 以匹配损失函数的期望格式
        semantic_targets = semantic_targets.view(1, B * N, feat_h, feat_w)
        
        # print(f"[Info] Built semantic supervision from image GT: {semantic_targets.shape}")
        # print(f"[Info] Semantic labels range: [{semantic_targets.min()}, {semantic_targets.max()}]")
        
        return semantic_targets
        
    # TODO: 加载预训练权重以及冻结参数


class GaussianImageFeatureExtractor(nn.Module):
    """
    图像特征提取backbone，支持 FPN 多尺度输出（P2: 64×176, P3: 32×88）
    兼容 SimpleCNN 和 ResNet101
    """
    def __init__(self, model_cfg):
        super(GaussianImageFeatureExtractor, self).__init__()
        self.model_cfg = model_cfg
        self.backbone_type = model_cfg.get('IMAGE_BACKBONE')
        self.out_channels = model_cfg.get('IMAGE_FEATURES', 128)
        self.image_feature_size_fix = model_cfg.get('IMAGE_FEATURE_SIZE_FIX', False)
        self.use_fpn_multiscale = model_cfg.get('USE_FPN_MULTISCALE')
        self.agent_feature_scale = model_cfg.get('AGENT_FEATURE_SCALE', {
            'drone': 'P2', 'vehicle': 'P3', 'rsu': 'P3'
        })
        
        if self.backbone_type == 'EfficientNet':
            # self.backbone = EfficientNet.from_pretrained("efficientnet-b0") #TODO 导入efficientnet模型
            # TODO: EfficientNet 暂不实现 FPN，保持原有逻辑
            self.feature_fusion = nn.Sequential(
                nn.Conv2d(320 + 112, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, self.out_channels, kernel_size=1),
            )
            self._has_fpn = False
        elif self.backbone_type == 'ResNet101':
            trunk = models.resnet101(pretrained=False, zero_init_residual=False)  # 使用预训练权重，不需 zero_init
            # 加载 ResNet101 ImageNet 预训练权重（若配置了路径）
            resnet101_ckpt = model_cfg.get('RESNET101_PRETRAINED_PATH', None)
            if resnet101_ckpt:
                self._load_resnet101_pretrained(trunk, resnet101_ckpt)
            self.conv1 = trunk.conv1
            self.bn1 = trunk.bn1
            self.relu = nn.ReLU()
            self.maxpool = trunk.maxpool
            self.layer1 = trunk.layer1
            self.layer2 = trunk.layer2
            self.layer3 = nn.Identity()
            # P2: layer1 输出 256ch @ 64×176, P3: layer2 输出 512ch @ 32×88
            self.fusion_P2 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, self.out_channels, kernel_size=1),
            )
            self.fusion_P3 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, self.out_channels, kernel_size=1),
            )
            self.feature_fusion = self.fusion_P3  # 兼容非 FPN 模式
            self._has_fpn = True
        elif self.backbone_type == 'SimpleCNN':
            # 拆分为 stage1 (到 pool2) 和 stage2 (pool3)，输出 P2 和 P3
            self.stage1 = nn.Sequential(
                nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                # 第二层：64 -> 128通道，保持尺寸
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                # 第三层：128 -> 256通道，保持尺寸
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),   # 256×704 -> 128×352
                nn.MaxPool2d(kernel_size=2, stride=2),  # 128×352 -> 64×176
            )
            self.stage2 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),  # 64×176 -> 32×88
            )
            self.fusion_P2 = nn.Conv2d(256, self.out_channels, kernel_size=1)
            self.fusion_P3 = nn.Conv2d(256, self.out_channels, kernel_size=1)
            self.feature_fusion = self.fusion_P3  # 兼容非 FPN 模式
            self._has_fpn = True
        else:
            raise ValueError(f"Unsupported backbone_type: {self.backbone_type}")

    def _load_resnet101_pretrained(self, trunk, ckpt_path):
        """加载 ResNet101 ImageNet 预训练权重到 trunk（conv1,bn1,layer1,layer2）"""
        import os
        if not os.path.exists(ckpt_path):
            print(f"[Warning] ResNet101 pretrained not found: {ckpt_path}, skip loading")
            return
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        # 兼容 DDP/其他包装：移除 'module.' 前缀
        new_state = {}
        for k, v in state.items():
            if k.startswith('module.'):
                new_state[k[7:]] = v
            else:
                new_state[k] = v
        # 只加载 trunk 需要的部分（layer3、fc 可选忽略）
        missing, unexpected = trunk.load_state_dict(new_state, strict=False)
        n_loaded = len(new_state) - len(missing)
        print(f"[ResNet101] Loaded ImageNet pretrained from {ckpt_path} ({n_loaded} params, missing: {len(missing)})")

    def forward(self, agent_data, agent_type=None):
        """
        提取图像特征，支持 FPN 多尺度输出
        
        Args:
            agent_data: 包含相机输入数据的字典
            agent_type: agent 类型，用于 FPN 时按 scale 选择（可省略）
        
        Returns:
            若 USE_FPN_MULTISCALE 且 backbone 支持 FPN:
                dict: {'P2': [B,N,C,64,176], 'P3': [B,N,C,32,88]}
            否则:
                tensor: [B, N, C, H, W] 单尺度特征
        """
        
        cam_inputs = agent_data['batch_merged_cam_inputs']
        imgs = cam_inputs['imgs']  # [B, N, C, H, W]，C=3(RGB) 或 4(RGB+Depth)
        
        B, N, C, H, W = imgs.shape
        imgs = imgs.view(B * N, C, H, W)
        
        # ResNet101/EfficientNet 预训练权重针对 3 通道 RGB，4 通道时取前 3 通道
        if C > 3 and self.backbone_type in ('ResNet101', 'EfficientNet'):
            imgs = imgs[:, :3, :, :]
        
        if self.use_fpn_multiscale and self._has_fpn:
            # FPN 多尺度输出
            if self.backbone_type == 'ResNet101':
                feat_P2, feat_P3 = self._extract_resnet_fpn_features(imgs)
            elif self.backbone_type == 'SimpleCNN':
                feat_P2, feat_P3 = self._extract_simple_cnn_fpn_features(imgs)
            else:
                raise ValueError(f"FPN not supported for backbone_type={self.backbone_type}")
            scale_key = self.agent_feature_scale.get(agent_type)
            if scale_key == 'P2':
                P2 = self.fusion_P2(feat_P2)  # [B*N, 128, 64, 176]
                return P2.view(B, N, self.out_channels, 64, 176)
            elif scale_key == 'P3':
                P3 = self.fusion_P3(feat_P3)  # [B*N, 128, 32, 88]
                return P3.view(B, N, self.out_channels, 32, 88)
            else:
                raise ValueError(f"Unsupported scale_key: {scale_key}")
        
        # 单尺度（兼容原有逻辑）
        if self.backbone_type == 'EfficientNet':
            features = self._extract_eff_features(imgs)
        elif self.backbone_type == 'ResNet101':
            features = self._extract_resnet_features(imgs)
        elif self.backbone_type == 'SimpleCNN':
            features = self._extract_simple_cnn_features(imgs)
        else:
            raise ValueError(f"Unsupported backbone_type: {self.backbone_type}")
        # 特征融合
        features = self.feature_fusion(features)
        _, C_out, H_out, W_out = features.shape
        features = features.view(B, N, C_out, H_out, W_out)
        if self.image_feature_size_fix and (H_out != 64 or W_out != 176):
            features = F.interpolate(
                features.view(B * N, C_out, H_out, W_out),
                size=(64, 176),
                mode='bilinear',
                align_corners=False
            ).view(B, N, C_out, 64, 176)
        
        return features

    def _extract_eff_features(self, x):
        """使用EfficientNet提取特征"""
        endpoints = dict()
        
        # Stem
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x
        
        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints["reduction_{}".format(len(endpoints) + 1)] = prev_x
            prev_x = x
        
        # Head
        endpoints["reduction_{}".format(len(endpoints) + 1)] = x
        
        # 特征融合
        x = torch.cat([endpoints["reduction_5"], endpoints["reduction_4"]], dim=1)
        
        return x

    def _extract_resnet_features(self, x):
        """使用ResNet101提取特征（单尺度，输出 layer2）"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def _extract_resnet_fpn_features(self, x):
        """ResNet101 FPN: P2=layer1(64×176), P3=layer2(32×88)"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        feat_P2 = self.layer1(x)   # 512ch @ 64×176
        feat_P3 = self.layer2(feat_P2)  # 512ch @ 32×88
        return feat_P2, feat_P3

    def _extract_simple_cnn_features(self, x):
        """SimpleCNN 单尺度：stage1 + stage2 -> 32×88"""
        x = self.stage1(x)
        x = self.stage2(x)
        return x

    def _extract_simple_cnn_fpn_features(self, x):
        """SimpleCNN FPN: P2=stage1(64×176), P3=stage1+stage2(32×88)"""
        feat_P2 = self.stage1(x)   # 256ch @ 64×176
        feat_P3 = self.stage2(feat_P2)  # 256ch @ 32×88
        return feat_P2, feat_P3


class GaussianDetectionHead(nn.Module):
    """
    2. 价值区域检测头（二值化Mask生成）
    基于图像特征生成价值区域的二值化mask
    """
    def __init__(self, model_cfg):
        super(GaussianDetectionHead, self).__init__()
        self.model_cfg = model_cfg
        self.in_channels = model_cfg.get('IMAGE_FEATURES', 128)
        self.mask_threshold = model_cfg.get('MASK_THRESHOLD', 0.2)
        self.use_morphology = model_cfg.get('USE_MORPHOLOGY', False)
        # 语义分类配置
        self.num_classes = model_cfg.get('NUM_CLASSES', 7)
        self.empty_idx = model_cfg.get('EMPTY_CLASS_INDEX', 0)
        self.topk_pixels = model_cfg.get('TOPK_PIXELS', 1000)
        self.image_shape = model_cfg.get('IMAGE_SHAPE', [32, 88])
        
        
        # 轻量级多类分类头（用于backbone特征）
        self.lightweight_cls_head = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.num_classes, kernel_size=1)  # logits: [B*N, M, Hm, Wm]
        )
        
        # 可选：添加空间注意力机制
        self.use_spatial_attention = model_cfg.get('USE_SPATIAL_ATTENTION', False)
        if self.use_spatial_attention:
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(self.in_channels, 1, kernel_size=1),
                nn.Sigmoid()
            )
        
        # 初始化权重
        # self._init_weights()

    def _init_weights(self):
        """合理的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, image_features):
        raise NotImplementedError(
            "Use `forward_from_features(image_features)` → returns "
            "{'class_probs':[B,N,M,32,88], 'topk_mask':[B,N,32,88]}."
        )

    def forward_from_features(self, image_features):
        """
        从 backbone 特征生成语义概率（带 Top-K 约束）
        支持 FPN 多尺度：可变 H×W（如 64×176 或 32×88）
        Args:
            image_features: [B, N, C_feat, H_feat, W_feat]
        Returns:
            dict{
              'class_probs': [B,N,M,H,W],
              'topk_mask':   [B,N,H,W]
            }
        """
        B, N, C_feat, H_feat, W_feat = image_features.shape

        x = image_features.view(B * N, C_feat, H_feat, W_feat)

        # 多类 logits 与 softmax 概率（直接基于 backbone 特征）
        logits = self.lightweight_cls_head(x)                 # [B*N, M, 32, 88]
        M = self.num_classes
        probs = F.softmax(logits, dim=1).view(B, N, M, H_feat, W_feat)

        # 非空类集合与最佳非空概率/类别
        device = probs.device
        nonempty = [i for i in range(M) if i != self.empty_idx]
        if len(nonempty) == 0:
            topk_mask = torch.zeros(B, N, H_feat, W_feat, device=device, dtype=torch.bool)
            return {'class_probs': probs, 'topk_mask': topk_mask}

        probs_nonempty = probs[:, :, nonempty, :, :]                     # [B,N,M-1,32,88]
        best_nonempty_prob, _ = probs_nonempty.max(dim=2)                # [B,N,32,88]

        # 全图 Top-K（按最佳非空概率）
        flat_scores = best_nonempty_prob.view(B * N, -1)                  # [B*N, 32*88]
        K_cfg = int(self.topk_pixels)
        total = flat_scores.shape[1]
        # 防止配置过大导致等于全图：若 K_cfg>=total，按比例（10%）取 Top-K
        if K_cfg >= total:
            K = max(1, int(total * 0.1))
        else:
            K = max(1, K_cfg)
        _, topk_idx = torch.topk(flat_scores, k=K, dim=1)                 # [B*N, K]
        mask_topk = torch.zeros_like(flat_scores, dtype=torch.bool)
        mask_topk.scatter_(1, topk_idx, True)
        mask_topk = mask_topk.view(B, N, H_feat, W_feat)                  # [B,N,32,88]
        mask_topk = mask_topk > self.mask_threshold
        # 计算 argmax 类别，排除空类
        cls_idx_map = probs.argmax(dim=2)                                 # [B,N,H,W] 每个像素的预测类别
        mask_nonempty = (cls_idx_map != self.empty_idx)                   # [B,N,H,W] 非空类掩码
        
        # 最终 mask：Top-K 且非空类（在 detection head 里直接计算好）
        final_mask = mask_topk & mask_nonempty                             # [B,N,H,W]
        if self.use_morphology:
            final_mask = self._morphology_postprocess(final_mask)
        return {'class_probs': probs, 'topk_mask': final_mask}

    def _morphology_postprocess(self, mask):
        """
        形态学后处理，去除噪声和填充空洞
        Args:
            mask: [B, N, 1, H, W] 二值化mask
        Returns:
            processed_mask: [B, N, 1, H, W] 处理后的mask
        """
        # 转换为numpy进行形态学操作
        mask_np = mask.detach().cpu().numpy()
        processed_mask = mask_np.copy()
        
        for b in range(mask_np.shape[0]):
            for n in range(mask_np.shape[1]):
                # 获取当前mask
                current_mask = mask_np[b, n, 0]
                
                # 形态学操作
                kernel = np.ones((3, 3), np.uint8)
                # 开运算：先腐蚀后膨胀，去除小噪声
                current_mask = cv2.morphologyEx(current_mask, cv2.MORPH_OPEN, kernel)
                # 闭运算：先膨胀后腐蚀，填充空洞
                current_mask = cv2.morphologyEx(current_mask, cv2.MORPH_CLOSE, kernel)
                
                processed_mask[b, n, 0] = current_mask
        
        # 转换回tensor
        processed_mask = torch.from_numpy(processed_mask).to(mask.device)
        return processed_mask
