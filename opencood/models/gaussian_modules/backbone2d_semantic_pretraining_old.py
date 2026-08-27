# -*- coding: utf-8 -*-
# Author: AI Assistant
# License: TDG-Attribution-NonCommercial-NoDistrib
#
# =============================================================================
# 旧版备份：无 FPN 多尺度的单尺度版本
# - 仅支持单尺度输出 [B, N, C, 32, 88]
# - SimpleCNN: conv_layers 单路径，4 通道输入 -> 32×88
# - ResNet101: layer2 输出 512ch -> feature_fusion -> 32×88
# - 无 agent_type 按尺度选择逻辑
# - 注意：ResNet101 未处理 4 通道输入，若数据为 RGB+Depth 会报错
# =============================================================================

"""
Gaussian Image Backbone for Multi-Agent Collaborative 3D Gaussian Perception System
实现图像特征提取、2D检测、深度预测和TPV投影的完整流程（单尺度版本）
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


# 默认配置模板（无 FPN）
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
    # === 语义检测相关默认配置 ===
    'NUM_CLASSES': 4,
    'EMPTY_CLASS_INDEX': 1,
    'TOPK_PIXELS': 1000,
    'GAUSSIAN_SCALE_RANGE': [0.1, 1.5],
    'USE_SPATIAL_ATTENTION': False,
    'USE_MORPHOLOGY': False,
    'AGENT_TYPES': ['vehicle', 'rsu', 'drone'],
    'IMAGE_SHAPE': [32, 88],
}

class GaussianImageBackbone(nn.Module):
    """
    高斯感知系统的图像backbone（单尺度版本）
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
        self.image_shape = model_cfg.get('IMAGE_SHAPE', [32, 88])
        
        # 1. 图像特征提取backbone（单尺度）
        self.image_backbone = GaussianImageFeatureExtractor(model_cfg)
        
        # 2. 2D检测头（类似YOLO）
        self.detection_head = GaussianDetectionHead(model_cfg)
        
        # 初始化语义标签映射器（用于从真实世界坐标查询标签）
        seg_hw = model_cfg.get('seg_hw', 512)

        # 图片语义真值配置
        self.use_image_semantic_gt = model_cfg.get('USE_IMAGE_SEMANTIC_GT', True)
        print(f"[Backbone2D] Use image semantic GT: {self.use_image_semantic_gt}")
        
        self.visualize_projection = model_cfg.get('visualize_projection', True)
        self.visualization_save_dir = model_cfg.get('visualization_save_dir', './visualization/lidar_projection')
        seg_res = model_cfg.get('seg_res', 0.25)
        pc_range = model_cfg.get('POINT_CLOUD_RANGE', self.point_cloud_range)
        self.seg_label_mapper = SegLabelMapper(
            seg_hw=seg_hw,
            seg_res=seg_res,
            lidar_range=pc_range,
            ego_center=True
        )

    def forward(self, batch_dict, available_agent):
        """
        完整的前向传播流程（单尺度）
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
            
        self.agent_types = available_agent
        for agent_type in self.agent_types:
            if agent_type in batch_dict and 'batch_merged_cam_inputs' in batch_dict[agent_type]:
                agent_data = batch_dict[agent_type]
                camera_num = batch_dict[agent_type]['batch_merged_cam_inputs']['imgs'].shape[1]
                # 1. 图像特征提取（单尺度，无 agent_type）
                image_features = self.image_backbone(agent_data)
                B, N, C_feat, H, W = image_features.shape
                image_features_5d = image_features.view(1, B*N, -1, H, W)
                
                # 2. 获取语义logits
                semantic_logits_dense = image_features.view(B*N, -1, H, W)
                semantic_logits = self.detection_head.lightweight_cls_head(semantic_logits_dense)
                
                cam_inputs = agent_data['batch_merged_cam_inputs']
                intrinsics = cam_inputs['intrinsics'].view(B, N, 3, 3)
                extrinsics = cam_inputs['extrinsics'].view(B, N, 4, 4)
                post_rots = cam_inputs['post_rots'].view(B, N, 3, 3)
                post_trans = cam_inputs['post_trans'].view(B, N, 3)
                
                agent_to_ego_transform = batch_dict['img_pairwise_t_matrix_collab'][0,agent_idx[agent_type]:agent_idx[agent_type]+batch_dict[agent_type]['record_len'],0,:,:]
                agent_to_ego_transform = agent_to_ego_transform.unsqueeze(0)
                agent_to_ego_transform = agent_to_ego_transform.repeat_interleave(camera_num, dim=1)
                
                if self.use_image_semantic_gt:
                    semantic_targets = self._build_semantic_supervision_from_image_gt(
                        batch_dict, agent_type, B, N, feat_h=H, feat_w=W
                    )
                    batch_dict[agent_type]['semantic_targets'] = semantic_targets
                    batch_dict[agent_type]['semantic_logits'] = semantic_logits
                else:
                    label_dict = batch_dict.get('label_dict', None)
                    if label_dict is not None:
                        if agent_type == 'vehicle' or agent_type is None:
                            lidar_key = 'origin_lidar'
                        else:
                            lidar_key = f'origin_lidar_{agent_type}'
                        lidar_points = batch_dict.get(lidar_key, None)
                        if lidar_points is not None:
                            if lidar_points.dim() == 3 and lidar_points.shape[0] == 1:
                                lidar_points = lidar_points.squeeze(0)
                            if lidar_points.shape[1] == 4:
                                lidar_points = lidar_points[:, :3]
                            lidar_labels, lidar_valid_mask = self.query_semantic_labels(
                                lidar_points, label_dict, label_type='dynamic'
                            )
                            if lidar_labels is not None and lidar_valid_mask is not None and lidar_valid_mask.any():
                                valid_lidar_points = lidar_points[lidar_valid_mask]
                                valid_lidar_labels = lidar_labels[lidar_valid_mask]
                                semantic_targets = self._build_semantic_supervision_from_lidar(
                                    batch_dict, agent_type,
                                    valid_lidar_points, valid_lidar_labels,
                                    intrinsics, extrinsics, 
                                    post_rots, post_trans,
                                    agent_to_ego_transform,
                                    H, W
                                )
                                semantic_targets = semantic_targets.view(1, B * N, H, W)
                                batch_dict[agent_type]['semantic_targets'] = semantic_targets
                                batch_dict[agent_type]['semantic_logits'] = semantic_logits
        
        return batch_dict

    
    def query_semantic_labels(self, world_coords, label_dict, label_type='dynamic'):
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
        B, N = intrinsics.shape[:2]
        BN = B * N
        device = lidar_points.device
        N_lidar = lidar_points.shape[0]
        intrinsics_3 = intrinsics[:, :, :3, :3] if intrinsics.shape[-2:] != (3, 3) else intrinsics
        extrinsics_4 = extrinsics if extrinsics.shape[-2:] == (4, 4) else torch.eye(4, device=device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        extrinsics_4 = torch.inverse(extrinsics_4)
        if post_rots.shape[-2:] == (3, 3):
            post_rots_4 = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
            post_rots_4[:, :, :3, :3] = post_rots
        else:
            post_rots_4 = post_rots
        if post_trans.shape[-1] == 3:
            post_trans_4 = torch.zeros(B, N, 4, device=device)
            post_trans_4[:, :, :3] = post_trans
            post_trans_4[:, :, 3] = 1
        else:
            post_trans_4 = post_trans
        extrinsics_4 = extrinsics_4.view(1, -1, 4, 4)
        if agent_to_ego_transform is not None:
            ego_to_agent = torch.inverse(agent_to_ego_transform)
            lidar2cam = torch.matmul(extrinsics_4[0,:,:,:], ego_to_agent)
        else:
            lidar2cam = extrinsics_4[0]
        ones = torch.ones((N_lidar, 1), device=device)
        lidar_points_homo = torch.cat([lidar_points, ones], dim=1)
        lidar_points_homo = lidar_points_homo.view(1, 1, N_lidar, 4).repeat(1, BN, 1, 1)
        points_cam = torch.matmul(
            lidar2cam.unsqueeze(2),
            lidar_points_homo.unsqueeze(-1)
        ).squeeze(-1)
        points_cam_xyz = points_cam[..., :3]
        points_cam_reordered = points_cam_xyz
        points_img = torch.matmul(
            intrinsics_3.view(1, -1, 3, 3).unsqueeze(2),
            points_cam_reordered.unsqueeze(-1)
        ).squeeze(-1)
        depth = points_img[..., 2]
        eps = 1e-5
        valid_mask = depth > eps
        u_ori = points_img[..., 0] / torch.clamp(depth, min=eps)
        v_ori = points_img[..., 1] / torch.clamp(depth, min=eps)
        point_img = torch.stack(
            [u_ori, v_ori, torch.ones_like(u_ori), torch.ones_like(u_ori)],
            dim=-1
        )
        point_img_aug = point_img
        u = point_img_aug[..., 0] / 1280
        v = point_img_aug[..., 1] / 720
        pixel_coords = torch.stack([u, v], dim=-1)
        pixel_coords = torch.where(
            valid_mask.unsqueeze(-1), 
            pixel_coords, 
            torch.full_like(pixel_coords, 0.0)
        )
        pixel_coords = pixel_coords.view(B, N, N_lidar, 2)
        valid_mask = valid_mask & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
        valid_mask = valid_mask.view(B, N, N_lidar)
        depth = depth.view(B, N, N_lidar)
        return pixel_coords, valid_mask, depth
    
    def _map_pixel_to_feature(self, pixel_coords, valid_mask, img_h, img_w, feat_h, feat_w):
        B, N, N_lidar = valid_mask.shape
        u_feat = (pixel_coords[:, :, :, 0] * feat_w).long()
        v_feat = (pixel_coords[:, :, :, 1] * feat_h).long()
        u_feat = torch.clamp(u_feat, 0, feat_w - 1)
        v_feat = torch.clamp(v_feat, 0, feat_h - 1)
        feat_coords = torch.stack([u_feat, v_feat], dim=-1)
        valid_mask_feat = valid_mask & (
            (u_feat >= 0) & (u_feat < feat_w) &
            (v_feat >= 0) & (v_feat < feat_h)
        )
        return feat_coords, valid_mask_feat
    
    def _visualize_lidar_projection(self, batch_dict, agent_type, pixel_coords, valid_mask, 
                                     save_dir=None, save_prefix="lidar_projection", max_vis=5):
        if 'batch_merged_cam_inputs' not in batch_dict.get(agent_type, {}):
            return
        cam_inputs = batch_dict[agent_type]['batch_merged_cam_inputs']
        original_imgs = cam_inputs.get('original_imgs')
        B, N = pixel_coords.shape[:2]
        for b in range(min(B, max_vis)):
            for n in range(N):
                if original_imgs is not None:
                    img_tensor = original_imgs[b, n]
                    img = (img_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                else:
                    imgs = cam_inputs['imgs']
                    img_tensor = imgs[b, n]
                    img_rgb = img_tensor[:3] if img_tensor.shape[0] >= 3 else img_tensor.repeat(3, 1, 1)
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_rgb.device)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_rgb.device)
                    img_denorm = torch.clamp(img_rgb * std + mean, 0, 1)
                    img = (img_denorm.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if save_dir:
                    import os
                    os.makedirs(save_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(save_dir, f"{save_prefix}_{agent_type}_b{b}_c{n}.png"), img_bgr)
    
    def _build_semantic_supervision_from_lidar(self, batch_dict, agent_type, 
                                                lidar_points, lidar_labels,
                                                intrinsics, extrinsics, 
                                                post_rots, post_trans,
                                                agent_to_ego_transform,
                                                feat_h, feat_w):
        B, N = intrinsics.shape[:2]
        device = intrinsics.device
        if 'batch_merged_cam_inputs' in batch_dict.get(agent_type, {}):
            imgs = batch_dict[agent_type]['batch_merged_cam_inputs']['imgs']
            img_h, img_w = imgs.shape[-2:]
        else:
            img_h, img_w = 720, 1280
        pixel_coords, valid_mask, _ = self._project_lidar_to_image(
            lidar_points, intrinsics, extrinsics, post_rots, post_trans,
            agent_to_ego_transform, img_h, img_w
        )
        if self.visualize_projection:
            self._visualize_lidar_projection(
                batch_dict, agent_type, pixel_coords, valid_mask,
                save_dir=self.visualization_save_dir,
                save_prefix=f"lidar_proj_{agent_type}", max_vis=2
            )
        feat_coords, valid_mask_feat = self._map_pixel_to_feature(
            pixel_coords, valid_mask, img_h, img_w, feat_h, feat_w
        )
        num_classes = self.model_cfg.get('NUM_CLASSES', 7)
        semantic_targets = torch.zeros(B, N, feat_h, feat_w, dtype=torch.long, device=device)
        lidar_labels_expanded = lidar_labels.unsqueeze(0).unsqueeze(0).expand(B, N, -1)
        for b in range(B):
            for n in range(N):
                valid_idx = valid_mask_feat[b, n]
                if not valid_idx.any():
                    continue
                valid_feat_coords = feat_coords[b, n, valid_idx]
                valid_labels = lidar_labels_expanded[b, n, valid_idx]
                non_zero_mask = valid_labels != 0
                if not non_zero_mask.any():
                    continue
                valid_feat_coords_filtered = valid_feat_coords[non_zero_mask]
                valid_labels_filtered = valid_labels[non_zero_mask]
                u_feat = valid_feat_coords_filtered[:, 0]
                v_feat = valid_feat_coords_filtered[:, 1]
                flat_idx = v_feat * feat_w + u_feat
                label_counts = torch.zeros(feat_h * feat_w, num_classes, dtype=torch.float32, device=device)
                labels_onehot = torch.zeros(len(valid_labels_filtered), num_classes, dtype=torch.float32, device=device)
                labels_onehot.scatter_(1, valid_labels_filtered.unsqueeze(1).long(), 1.0)
                for i in range(len(flat_idx)):
                    label_counts[flat_idx[i]] += labels_onehot[i]
                max_counts, argmax_labels = label_counts.max(dim=1)
                update_mask = max_counts > 0
                semantic_targets[b, n].view(-1)[update_mask] = argmax_labels[update_mask].long()
        return semantic_targets

    def _build_semantic_supervision_from_image_gt(self, batch_dict, agent_type, B, N, feat_h, feat_w):
        if agent_type not in batch_dict or 'batch_merged_cam_inputs' not in batch_dict[agent_type]:
            return None
        cam_inputs = batch_dict[agent_type]['batch_merged_cam_inputs']
        if 'image_semantic_gts' not in cam_inputs:
            return None
        image_semantic_gts = cam_inputs['image_semantic_gts']
        if image_semantic_gts.dim() == 3:
            image_semantic_gts = image_semantic_gts.view(B, N, image_semantic_gts.shape[1], image_semantic_gts.shape[2])
        device = image_semantic_gts.device
        B_gt, N_gt, H_aug, W_aug = image_semantic_gts.shape
        if B_gt != B or N_gt != N:
            return None
        image_semantic_gts_flat = image_semantic_gts.view(B * N, H_aug, W_aug)
        semantic_targets = F.interpolate(
            image_semantic_gts_flat.unsqueeze(1).float(),
            size=(feat_h, feat_w),
            mode='nearest'
        ).squeeze(1).long()
        semantic_targets = semantic_targets.view(1, B * N, feat_h, feat_w)
        return semantic_targets


class GaussianImageFeatureExtractor(nn.Module):
    """
    图像特征提取backbone（单尺度，无 FPN）
    输出固定 [B, N, C, 32, 88]
    兼容 SimpleCNN 和 ResNet101
    """
    def __init__(self, model_cfg):
        super(GaussianImageFeatureExtractor, self).__init__()
        self.model_cfg = model_cfg
        self.backbone_type = model_cfg.get('IMAGE_BACKBONE')
        self.out_channels = model_cfg.get('IMAGE_FEATURES', 128)
        self.image_feature_size_fix = model_cfg.get('IMAGE_FEATURE_SIZE_FIX', False)
        
        if self.backbone_type == 'EfficientNet':
            self.feature_fusion = nn.Sequential(
                nn.Conv2d(320 + 112, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, self.out_channels, kernel_size=1),
            )
        elif self.backbone_type == 'ResNet101':
            trunk = models.resnet101(pretrained=False, zero_init_residual=True)
            self.conv1 = trunk.conv1
            self.bn1 = trunk.bn1
            self.relu = nn.ReLU()
            self.maxpool = trunk.maxpool
            self.layer1 = trunk.layer1
            self.layer2 = trunk.layer2
            self.layer3 = nn.Identity()
            self.feature_fusion = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, self.out_channels, kernel_size=1),
            )
        elif self.backbone_type == 'SimpleCNN':
            # 单路径 CNN，4 通道输入 -> 32×88
            self.conv_layers = nn.Sequential(
                nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.feature_fusion = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, self.out_channels, kernel_size=1),
            )
        else:
            raise ValueError(f"Unsupported backbone_type: {self.backbone_type}")

    def forward(self, agent_data):
        """
        提取图像特征（单尺度）
        Returns:
            tensor: [B, N, C, 32, 88]
        """
        cam_inputs = agent_data['batch_merged_cam_inputs']
        imgs = cam_inputs['imgs']
        B, N, C, H, W = imgs.shape
        imgs = imgs.view(B * N, C, H, W)
        
        if self.backbone_type == 'EfficientNet':
            features = self._extract_eff_features(imgs)
        elif self.backbone_type == 'ResNet101':
            features = self._extract_resnet_features(imgs)
        elif self.backbone_type == 'SimpleCNN':
            features = self._extract_simple_cnn_features(imgs)
        else:
            raise ValueError(f"Unsupported backbone_type: {self.backbone_type}")
        
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
        endpoints = dict()
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints["reduction_{}".format(len(endpoints) + 1)] = prev_x
            prev_x = x
        endpoints["reduction_{}".format(len(endpoints) + 1)] = x
        x = torch.cat([endpoints["reduction_5"], endpoints["reduction_4"]], dim=1)
        return x

    def _extract_resnet_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def _extract_simple_cnn_features(self, x):
        x = self.conv_layers(x)
        return x


class GaussianDetectionHead(nn.Module):
    """2D检测头，基于图像特征生成语义概率"""
    def __init__(self, model_cfg):
        super(GaussianDetectionHead, self).__init__()
        self.model_cfg = model_cfg
        self.in_channels = model_cfg.get('IMAGE_FEATURES', 128)
        self.mask_threshold = model_cfg.get('MASK_THRESHOLD', 0.2)
        self.use_morphology = model_cfg.get('USE_MORPHOLOGY', False)
        self.num_classes = model_cfg.get('NUM_CLASSES', 4)
        self.empty_idx = model_cfg.get('EMPTY_CLASS_INDEX', 0)
        self.topk_pixels = model_cfg.get('TOPK_PIXELS', 1000)
        self.image_shape = model_cfg.get('IMAGE_SHAPE', [32, 88])
        
        self.lightweight_cls_head = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.num_classes, kernel_size=1)
        )
        self.use_spatial_attention = model_cfg.get('USE_SPATIAL_ATTENTION', False)
        if self.use_spatial_attention:
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(self.in_channels, 1, kernel_size=1),
                nn.Sigmoid()
            )

    def forward(self, image_features):
        raise NotImplementedError("Use forward_from_features(image_features)")

    def forward_from_features(self, image_features):
        B, N, C_feat, H_feat, W_feat = image_features.shape
        x = image_features.view(B * N, C_feat, H_feat, W_feat)
        logits = self.lightweight_cls_head(x)
        M = self.num_classes
        probs = F.softmax(logits, dim=1).view(B, N, M, H_feat, W_feat)
        device = probs.device
        nonempty = [i for i in range(M) if i != self.empty_idx]
        if len(nonempty) == 0:
            topk_mask = torch.zeros(B, N, H_feat, W_feat, device=device, dtype=torch.bool)
            return {'class_probs': probs, 'topk_mask': topk_mask}
        probs_nonempty = probs[:, :, nonempty, :, :]
        best_nonempty_prob, _ = probs_nonempty.max(dim=2)
        flat_scores = best_nonempty_prob.view(B * N, -1)
        K_cfg = int(self.topk_pixels)
        total = flat_scores.shape[1]
        K = max(1, int(total * 0.1)) if K_cfg >= total else max(1, K_cfg)
        _, topk_idx = torch.topk(flat_scores, k=K, dim=1)
        mask_topk = torch.zeros_like(flat_scores, dtype=torch.bool)
        mask_topk.scatter_(1, topk_idx, True)
        mask_topk = mask_topk.view(B, N, H_feat, W_feat) > self.mask_threshold
        cls_idx_map = probs.argmax(dim=2)
        mask_nonempty = (cls_idx_map != self.empty_idx)
        final_mask = mask_topk & mask_nonempty
        if self.use_morphology:
            final_mask = self._morphology_postprocess(final_mask)
        return {'class_probs': probs, 'topk_mask': final_mask}

    def _morphology_postprocess(self, mask):
        mask_np = mask.detach().cpu().numpy()
        processed_mask = mask_np.copy()
        for b in range(mask_np.shape[0]):
            for n in range(mask_np.shape[1]):
                current_mask = mask_np[b, n, 0]
                kernel = np.ones((3, 3), np.uint8)
                current_mask = cv2.morphologyEx(current_mask, cv2.MORPH_OPEN, kernel)
                current_mask = cv2.morphologyEx(current_mask, cv2.MORPH_CLOSE, kernel)
                processed_mask[b, n, 0] = current_mask
        return torch.from_numpy(processed_mask).to(mask.device)
