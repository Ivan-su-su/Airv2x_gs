# -*- coding: utf-8 -*-
# Author: AI Assistant
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Gaussian Image Backbone for Multi-Agent Collaborative 3D Gaussian Perception System
实现图像特征提取、2D检测、深度预测和TPV投影的完整流程
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# from efficientnet_pytorch import EfficientNet TODO
import torchvision.models as models
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from opencood.utils.camera_utils import (
    QuickCumsum,
    bin_depths,
    cumsum_trick,
    depth_discretization,
    gen_dx_bx,
)



# 默认配置模板（与预训练 backbone2d_semantic_pretraining 对齐）
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
    # === 语义检测相关配置 ===
    'NUM_CLASSES': 4,
    'EMPTY_CLASS_INDEX': 1,
    'TOPK_PIXELS': 1000,
    'GAUSSIAN_SCALE_RANGE': [0.1, 1.5],
    'USE_SPATIAL_ATTENTION': False,
    'USE_MORPHOLOGY': False,
    'AGENT_TYPES': ['vehicle', 'rsu', 'drone'],
    # === FPN 多尺度（与预训练一致） ===
    'USE_FPN_MULTISCALE': True,
    'AGENT_FEATURE_SCALE': {'drone': 'P2', 'vehicle': 'P3', 'rsu': 'P3'},
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
        self.image_shape = model_cfg.get('IMAGE_SHAPE', [32, 88])
        
        # FPN 多尺度配置（与预训练一致，确保权重正确加载）
        self.use_fpn_multiscale = model_cfg.get('USE_FPN_MULTISCALE', False)
        self.agent_feature_scale = model_cfg.get('AGENT_FEATURE_SCALE', {
            'drone': 'P2', 'vehicle': 'P3', 'rsu': 'P3'
        })
        self.image_shape_p2 = model_cfg.get('IMAGE_SHAPE_P2', [64, 176])
        self.image_shape_p3 = model_cfg.get('IMAGE_SHAPE_P3', [32, 88])
        if self.use_fpn_multiscale:
            print(f"[Backbone2D] FPN multi-scale: drone→P2{self.image_shape_p2}, vehicle/rsu→P3{self.image_shape_p3}")
        
        # 1. 图像特征提取backbone
        self.image_backbone = GaussianImageFeatureExtractor(model_cfg)
        
        # 2. 2D检测头（类似YOLO）
        self.detection_head = GaussianDetectionHead(model_cfg)
        
        # 4. TPV投影模块
        self.tpv_projector = OptimizedLSSBasedTPVGeneratorV2(model_cfg)
        
    def load_pretrained_weights(self, pretrained_path, strict=False, freeze_pretrained=True):
        """
        加载预训练权重到 image_backbone 和 detection_head
        
        Args:
            pretrained_path (str): 预训练权重文件路径
            strict (bool): 是否严格匹配权重（默认 True）
        
        Returns:
            None
        """
        import os
        if not os.path.exists(pretrained_path):
            print(f"[Warning] Pretrained weights not found at: {pretrained_path}")
            return
        
        print(f"[Info] Loading pretrained weights from: {pretrained_path}")
        
        # 加载权重文件
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # 提取 state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 过滤出 image_backbone 和 detection_head 的权重
        pretrained_dict = {}
        for key, value in state_dict.items():
            # 移除可能的 module. 前缀（DDP训练产生的）
            if key.startswith('module.'):
                key = key[7:]
            
            # 只加载 image_backbone 和 detection_head 的权重
            if key.startswith('image_backbone.') or key.startswith('detection_head.'):
                pretrained_dict[key] = value
        
        # 加载权重
        missing_keys, unexpected_keys = self.load_state_dict(pretrained_dict, strict=False)
        
        # 如果设置了freeze_pretrained，冻结这些模块
        if freeze_pretrained:
            self.freeze_pretrained_modules()

        # 打印加载信息
        print(f"[Info] Loaded {len(pretrained_dict)} pretrained parameters")
        if missing_keys:
            print(f"[Info] Missing keys: {len(missing_keys)} (这些参数将使用随机初始化)")
        if unexpected_keys:
            print(f"[Warning] Unexpected keys: {len(unexpected_keys)}")
        
        print("[Info] Pretrained weights loaded successfully!")
    
    def freeze_pretrained_modules(self):
        """
        冻结预训练的 image_backbone 和 detection_head 模块
        即：设置 requires_grad=False，这些参数不会在训练中更新
        """
        print("[Info] Freezing pretrained modules (image_backbone & detection_head)...")
        
        frozen_params = 0
        # 冻结 image_backbone
        for param in self.image_backbone.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
        
        # 冻结 detection_head
        for param in self.detection_head.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
        
        print(f"[Info] Frozen {frozen_params:,} parameters in pretrained modules")
        
        # 打印可训练参数统计
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Info] Total parameters: {total_params:,}")
        print(f"[Info] Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    def unfreeze_pretrained_modules(self):
        """
        解冻预训练的 image_backbone 和 detection_head 模块
        即：设置 requires_grad=True，这些参数可以在训练中更新
        """
        print("[Info] Unfreezing pretrained modules (image_backbone & detection_head)...")
        
        unfrozen_params = 0
        # 解冻 image_backbone
        for param in self.image_backbone.parameters():
            param.requires_grad = True
            unfrozen_params += param.numel()
        
        # 解冻 detection_head
        for param in self.detection_head.parameters():
            param.requires_grad = True
            unfrozen_params += param.numel()
        
        print(f"[Info] Unfrozen {unfrozen_params:,} parameters in pretrained modules")
        
        # 打印可训练参数统计
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Info] Total parameters: {total_params:,}")
        print(f"[Info] Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

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
            if agent_type in batch_dict and 'batch_merged_cam_inputs' in batch_dict[agent_type]:
                agent_data = batch_dict[agent_type]
                camera_num = batch_dict[agent_type]['batch_merged_cam_inputs']['imgs'].shape[1]
                # 1. 图像特征提取（FPN 多尺度或单尺度，与预训练一致）
                multi_scale_feats = self.image_backbone(agent_data, agent_type=agent_type)
                if self.use_fpn_multiscale and isinstance(multi_scale_feats, dict):
                    scale_key = self.agent_feature_scale.get(agent_type, 'P3')
                    image_features = multi_scale_feats[scale_key]
                else:
                    image_features = multi_scale_feats
                B, N, C_feat, H, W = image_features.shape
                image_features = image_features.view(1, B*N, -1, H, W)
                
                # 2. 多类语义检测（支持可变 H×W）
                det_out = self.detection_head.forward_from_features(image_features)
                class_probs = det_out['class_probs']         # [1,B*N,M,H,W]
                topk_mask = det_out['topk_mask']             # [1,B*N,H,W]
                
                # 3. 获取相机参数（参考 airv2x_encoder.py 的投影方式）
                cam_inputs = agent_data['batch_merged_cam_inputs']
                intrinsics = cam_inputs['intrinsics'].view(1,B*N,3,3)  # [1, B*N, 3, 3]
                rots = cam_inputs['rots'].view(1,B*N,3,3)  # [1, B*N, 3, 3] 相机到agent本地lidar坐标系的旋转
                trans = cam_inputs['trans'].view(1,B*N,3)  # [1, B*N, 3] 相机到agent本地lidar坐标系的平移
                post_rots = cam_inputs['post_rots'].view(1,B*N,3,3)  # [1, B*N, 3, 3] 数据增强的旋转
                post_trans = cam_inputs['post_trans'].view(1,B*N,3)  # [1, B*N, 3] 数据增强的平移
                
                agent_to_ego_transform = batch_dict['img_pairwise_t_matrix_collab'][0,agent_idx[agent_type]:agent_idx[agent_type]+batch_dict[agent_type]['record_len'],0,:,:]
                # 4. 获取从agent本地坐标系到ego坐标系的变换矩阵
                # img_pairwise_t_matrix_collab: [B, L, L, 4, 4]
                # pairwise_t_matrix[0, i, 0, :, :] 表示从agent i到ego(agent 0)的变换
                agent_to_ego_transform = agent_to_ego_transform.unsqueeze(0)
                agent_to_ego_transform = agent_to_ego_transform.repeat_interleave(camera_num, dim=1)
                
                # 6. TPV投影和高斯生成（仅使用低分辨率 conf_map
                tpv_results = self.tpv_projector(
                    agent_type,
                    image_features,
                    conf_map=class_probs,
                    intrinsics=intrinsics,
                    rots=rots,
                    trans=trans,
                    post_rots=post_rots,
                    post_trans=post_trans,
                    topk_mask=topk_mask,
                    agent_to_ego_transform=agent_to_ego_transform  # 传递变换矩阵
                )
                
                # 7. 将结果存储到对应agent的batch_dict中
                batch_dict[agent_type].update({
                    "image_tpv_features": tpv_results['tpv_features'],
                    "image_tpv_xy": tpv_results['tpv_features']['xy'],
                    "image_tpv_xz": tpv_results['tpv_features']['xz'], 
                    "image_tpv_yz": tpv_results['tpv_features']['yz'],
                    "image_gaussians": tpv_results['gaussians']
                })
        
        return batch_dict

    def get_image_features(self, batch_dict):
        """获取图像特征"""
        return batch_dict.get("image_tpv_features", None)

    def visualize_features(self, batch_dict, save_path=None):
        """可视化TPV特征"""
        feats = self.get_image_features(batch_dict)
        if feats is None:
            return None
        for plane_name in ["xy", "xz", "yz"]:
            if plane_name in feats:
                vis = feats[plane_name][0, 0].detach().cpu().numpy()
                if save_path:
                    img = ((vis - vis.min()) / (vis.max() - vis.min() + 1e-8) * 255).astype(np.uint8)
                    cv2.imwrite(f"{save_path}_{plane_name}.png", img)
        return feats


class GaussianImageFeatureExtractor(nn.Module):
    """
    图像特征提取backbone，支持 FPN 多尺度（P2: 64×176, P3: 32×88）
    与预训练 backbone2d_semantic_pretraining 结构一致，确保权重正确加载
    """
    def __init__(self, model_cfg):
        super(GaussianImageFeatureExtractor, self).__init__()
        self.model_cfg = model_cfg
        self.backbone_type = model_cfg.get('IMAGE_BACKBONE', 'SimpleCNN')
        self.out_channels = model_cfg.get('IMAGE_FEATURES', 128)
        self.image_feature_size_fix = model_cfg.get('IMAGE_FEATURE_SIZE_FIX', False)
        self.use_fpn_multiscale = model_cfg.get('USE_FPN_MULTISCALE', False)
        
        if self.backbone_type == 'EfficientNet':
            self.feature_fusion = nn.Sequential(
                nn.Conv2d(320 + 112, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, self.out_channels, kernel_size=1),
            )
            self._has_fpn = False
        elif self.backbone_type == 'ResNet101':
            trunk = models.resnet101(pretrained=False, zero_init_residual=True)
            self.conv1 = trunk.conv1
            self.bn1 = trunk.bn1
            self.relu = nn.ReLU()
            self.maxpool = trunk.maxpool
            self.layer1 = trunk.layer1
            self.layer2 = trunk.layer2
            self.layer3 = nn.Identity()
            self.fusion_P2 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
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
            self.feature_fusion = self.fusion_P3
            self._has_fpn = True
        elif self.backbone_type == 'SimpleCNN':
            self.stage1 = nn.Sequential(
                nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.stage2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
            self.fusion_P2 = nn.Conv2d(256, self.out_channels, kernel_size=1)
            self.fusion_P3 = nn.Conv2d(256, self.out_channels, kernel_size=1)
            self.feature_fusion = self.fusion_P3
            self._has_fpn = True
        else:
            raise ValueError(f"Unsupported backbone_type: {self.backbone_type}")

    def forward(self, agent_data, agent_type=None):
        """
        提取图像特征，支持 FPN 多尺度输出
        Returns:
            USE_FPN_MULTISCALE 且 backbone 支持: dict {'P2': [B,N,C,64,176], 'P3': [B,N,C,32,88]}
            否则: tensor [B, N, C, H, W]
        """
        imgs = agent_data['batch_merged_cam_inputs']['imgs']
        B, N, C, H, W = imgs.shape
        imgs = imgs.view(B * N, C, H, W)
        
        if self.use_fpn_multiscale and self._has_fpn:
            if self.backbone_type == 'ResNet101':
                feat_P2, feat_P3 = self._extract_resnet_fpn_features(imgs)
            elif self.backbone_type == 'SimpleCNN':
                feat_P2, feat_P3 = self._extract_simple_cnn_fpn_features(imgs)
            else:
                raise ValueError(f"FPN not supported for {self.backbone_type}")
            P2 = self.fusion_P2(feat_P2)
            P3 = self.fusion_P3(feat_P3)
            return {
                'P2': P2.view(B, N, self.out_channels, 64, 176),
                'P3': P3.view(B, N, self.out_channels, 32, 88),
            }
        
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

    def _extract_resnet_fpn_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        feat_P2 = self.layer1(x)
        feat_P3 = self.layer2(feat_P2)
        return feat_P2, feat_P3

    def _extract_simple_cnn_features(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        return x

    def _extract_simple_cnn_fpn_features(self, x):
        feat_P2 = self.stage1(x)
        feat_P3 = self.stage2(feat_P2)
        return feat_P2, feat_P3


class GaussianDetectionHead(nn.Module):
    """
    2. 价值区域检测头（二值化Mask生成）
    基于图像特征生成价值区域的二值化mask
    """
    def __init__(self, model_cfg):
        super(GaussianDetectionHead, self).__init__()
        self.model_cfg = model_cfg
        self.in_channels = model_cfg.get('IMAGE_FEATURES')
        self.mask_threshold = model_cfg.get('MASK_THRESHOLD')
        self.use_morphology = model_cfg.get('USE_MORPHOLOGY', False)
        # 语义分类配置
        self.num_classes = model_cfg.get('NUM_CLASSES')
        self.empty_idx = model_cfg.get('EMPTY_CLASS_INDEX')
        self.topk_pixels = model_cfg.get('TOPK_PIXELS')
        self.image_shape = model_cfg.get('IMAGE_SHAPE')
        
        
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
            "{'class_probs':[B,N,M,64,176], 'topk_mask':[B,N,64,176]}."
        )

    def forward_from_features(self, image_features):
        """
        从 backbone 特征生成语义概率（带 Top-K 约束）
        支持 FPN 多尺度：可变 H×W（如 64×176 或 32×88），与预训练一致
        Args:
            image_features: [B, N, C_feat, H_feat, W_feat]
        Returns:
            dict{'class_probs': [B,N,M,H,W], 'topk_mask': [B,N,H,W]}
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
            final_mask = self._morphology_postprocess(final_mask)   #TODO: 还未查看
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



class OptimizedLSSBasedTPVGeneratorV2(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        # TPV 体素配置
        self.tpv_features = model_cfg.get('TPV_FEATURES', 64)
        self.tpv_size = model_cfg.get('TPV_SIZE')  # [H, W, D]
        self.pc_range = model_cfg.get('POINT_CLOUD_RANGE')
        self.voxel_size = model_cfg.get('VOXEL_SIZE')

        # 语义设置（用于生成语义嵌入：MLP M→2M→4）
        self.num_classes = model_cfg.get('NUM_CLASSES')
        self.empty_idx = model_cfg.get('EMPTY_CLASS_INDEX')
        # 缓存非空类索引（优化1：避免每次计算）
        self._nonempty_indices = [i for i in range(self.num_classes) if i != self.empty_idx]
        self.semantic_mlp = nn.Sequential(
            nn.Linear(self.num_classes, 2 * self.num_classes),
            nn.ReLU(inplace=True),
            nn.Linear(2 * self.num_classes, 4)
        )

        # 深度估计配置
        self.depth_bins = model_cfg.get('DEPTH_BINS', 80)
        self.dbound = model_cfg.get('DBOUND', [2.0, 50.0, 0.5])  # [min, max, step]
        self.dbound_drone = model_cfg.get('DBOUND_DRONE', [52.0, 100.0, 0.5])  # [min, max, step]

        # 高斯生成配置
        self.top_k_depths = model_cfg.get('TOP_K_DEPTHS', 20)
        self.gaussian_threshold = model_cfg.get('GAUSSIAN_THRESHOLD', 0.1)  # 最小阈值（保底值）
        self.gaussian_scale_range = model_cfg.get('GAUSSIAN_SCALE_RANGE', [0.01, 3.2])
        # 注册为 buffer 以便在 forward 中使用
        self.register_buffer("scale_range", torch.tensor(self.gaussian_scale_range, dtype=torch.float32))
        self.register_buffer("pc_min", torch.tensor(self.pc_range[:3], dtype=torch.float32))
        self.register_buffer("voxel_size_tensor", torch.tensor(self.voxel_size, dtype=torch.float32))

        
        # 自适应阈值配置
        self.target_gaussians_ratio = model_cfg.get('TARGET_GAUSSIANS_RATIO', 0.3)  # 目标保留比例（30%）
        self.min_gaussians_per_camera = model_cfg.get('MIN_GAUSSIANS_PER_CAMERA', 100)  # 每个相机最少高斯数
        self.max_gaussians_per_camera = model_cfg.get('MAX_GAUSSIANS_PER_CAMERA', 5000)  # 每个相机最多高斯数
        self.use_adaptive_threshold = model_cfg.get('USE_ADAPTIVE_THRESHOLD', True)  # 是否使用自适应阈值

        # 特征网络配置
        self.image_channels = model_cfg.get('IMAGE_FEATURES', 128)
        self.depthnet = nn.Conv2d(self.image_channels, self.image_channels + self.depth_bins, kernel_size=1)

        # 高斯点 3D 可视化（范围内绿色、范围外红色）
        self.visualize_gaussians_3d = model_cfg.get('visualize_gaussians_3d', True)
        self.gaussians_visualization_dir = model_cfg.get('gaussians_visualization_dir', './gaussians_3d_vis')
        self._gaussians_vis_count = 0

        # 输入图像尺寸（即 batch_dict 中的图像尺寸，对应 post_rots/post_trans 的操作空间）
        input_image_shape = model_cfg.get('INPUT_IMAGE_SHAPE', [256, 704])
        self.input_image_H = input_image_shape[0]
        self.input_image_W = input_image_shape[1]

        # 初始化参数 - 不再预先创建frustum，按需生成
        self._cached_frustums = {}  # 缓存不同尺寸的frustum

    # ====================================================
    # 按需生成 frustum
    # ====================================================
    def _create_frustum(self, agent_type, H, W):
        """按需创建指定尺寸的frustum，避免显存浪费"""
        key = (H, W)
        if key not in self._cached_frustums or agent_type == 'drone':
            D = self.depth_bins
            if agent_type == 'drone':
                ds = torch.linspace(self.dbound_drone[0], self.dbound_drone[1], D, dtype=torch.float, device=self.depthnet.weight.device)
            else:
                ds = torch.linspace(self.dbound[0], self.dbound[1], D, dtype=torch.float, device=self.depthnet.weight.device)
            # 像素坐标覆盖输入图像（batch_dict）的完整范围，采样点数为特征图分辨率
            # 与 airv2x_encoder.py 一致：坐标在 post_rots/post_trans 的操作空间内
            xs = torch.linspace(0, self.input_image_W - 1, W, dtype=torch.float, device=self.depthnet.weight.device)
            ys = torch.linspace(0, self.input_image_H - 1, H, dtype=torch.float, device=self.depthnet.weight.device)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [H,W]
            frustum = torch.stack([grid_x[None].repeat(D,1,1), 
                               grid_y[None].repeat(D,1,1), 
                               ds[:,None,None].repeat(1,H,W)], dim=-1)
            self._cached_frustums[key] = frustum
        return self._cached_frustums[key]

    # ====================================================
    # 主前向：LSS → TPV → Gaussian
    # ====================================================
    def forward(self, agent_type, image_feat, conf_map, intrinsics, rots, trans, post_rots, post_trans, topk_mask=None, agent_to_ego_transform=None):
        """
        Args:
            image_feat:  [B, N, C, 64, 176]   低分辨率图像特征
            conf_map:    [B, N, M, 64, 176]   Detection head 输出的 softmax 概率
            topk_mask:   [B, N, 64, 176] or None  Top-K 像素 mask（Top-K 且非空类）
            intrinsics:  [B, N, 3, 3]         相机内参
            rots:        [B, N, 3, 3]         相机到agent本地lidar坐标系的旋转矩阵
            trans:       [B, N, 3]            相机到agent本地lidar坐标系的平移向量
            post_rots:   [B, N, 3, 3]         数据增强的旋转矩阵（需要undo）
            post_trans:  [B, N, 3]            数据增强的平移向量（需要undo）
            agent_to_ego_transform: [B, N, 4, 4] or None  从agent本地坐标系到ego坐标系的变换矩阵
        """
        B, N, C, H, W = image_feat.shape
        device = image_feat.device

        # Step 1: 深度估计（低分辨率）
        depth_prob, image_features = self._predict_depth(image_feat)

        # Step 2: LSS 投影 (几何变换) - 按需生成正确的frustum尺寸
        _, _, _, H, W = image_features.shape
        geom_coords = self._compute_world_coords(agent_type, intrinsics, rots, trans, post_rots, post_trans, H, W, agent_to_ego_transform)
        # Step 3: scatter_add → TPV（使用低分辨率特征）
        tpv = self._build_tpv_from_lss_v2(image_features, depth_prob, geom_coords) #实测优化版 快0.5s
        # Step 4: 高斯生成（全部使用低分辨率，避免上采样）
        gaussians = self._generate_gaussians(conf_map, topk_mask, image_features, depth_prob, geom_coords)

        return {"tpv_features": tpv, "gaussians": gaussians}

    # ====================================================
    # Step 1: 深度估计
    # ====================================================
    def _predict_depth(self, img_feat):
        B, N, C, H, W = img_feat.shape
        x = img_feat.view(B * N, C, H, W)
        out = self.depthnet(x)
        depth_prob = F.softmax(out[:, :self.depth_bins, :, :], dim=1)
        # TODO: depthnet 输出通道数为 C+DEPTH_BINS，可以直接取前80通道为深度概率吗
        feat = out[:, self.depth_bins:, :, :]
        depth_prob = depth_prob.view(B, N, self.depth_bins, H, W)
        feat = feat.view(B, N, C, H, W)
        return depth_prob, feat

    # ====================================================
    # Step 2: 几何坐标计算（参考 airv2x_encoder.py 的 get_geometry 方法）
    # ====================================================
    def _compute_world_coords(self, agent_type, intrinsics, rots, trans, post_rots, post_trans, H=None, W=None, agent_to_ego_transform=None):
        """
        计算图像特征投影到世界坐标系的坐标（ego坐标系）
        参考 airv2x_encoder.py 的 get_geometry 方法，考虑数据增强的逆变换
        并应用从agent本地坐标系到ego坐标系的变换
        
        Args:
            intrinsics:  [1, B*N, 3, 3] 相机内参
            rots:        [1, B*N, 3, 3] 相机到agent本地lidar坐标系的旋转矩阵
            trans:       [1, B*N, 3]    相机到agent本地lidar坐标系的平移向量
            post_rots:   [1, B*N, 3, 3] 数据增强的旋转矩阵（需要undo）
            post_trans:  [1, B*N, 3]    数据增强的平移向量（需要undo）
            H, W:        图像高度和宽度
            agent_to_ego_transform: [B, N, 4, 4] or None  从agent本地坐标系到ego坐标系的变换矩阵
            
        Returns:
            world_coords: [B, N, D, H, W, 3] ego坐标系下的点坐标
        """
        B, N = intrinsics.shape[:2]
        if H is None or W is None:
            raise ValueError("H and W must be specified")
        frustum = self._create_frustum(agent_type, H, W).to(intrinsics.device)  # [D, H, W, 3]
        D = self.depth_bins

        # Step 1: Undo post-transformation (数据增强的逆变换)
        # 参考 airv2x_encoder.py: self.frustum 是 [D, H, W, 3]，通过 broadcasting 扩展到 [B, N, D, H, W, 3]
        # 使用 view 和 expand 来显式扩展，避免 broadcasting 可能的问题
        points = frustum.view(1, 1, D, H, W, 3).expand(B, N, -1, -1, -1, -1)  # [B, N, D, H, W, 3]
        points = points - post_trans.view(B, N, 1, 1, 1, 3)  # 减去 post_trans
        
        # 应用 post_rots 的逆变换（参考 airv2x_encoder.py 的实现）
        # 注意：airv2x_encoder.py 中这里没有显式 squeeze(-1)，但后续 torch.cat 需要 [B, N, D, H, W, 3]
        # 实际上 matmul 的结果是 [B, N, D, H, W, 3, 1]，需要 squeeze(-1) 才能用于后续操作
        inv_post_rots = torch.inverse(post_rots)
        points = inv_post_rots.view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)  # [B, N, D, H, W, 3]

        # Step 2: Convert to camera coordinates
        # 将像素坐标转换为相机坐标（考虑深度）
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],  # x, y 乘以深度
                points[:, :, :, :, :, 2:3],  # z (深度)
            ),
            dim=5,
        )  # [B, N, D, H, W, 3]

        # Step 3: Transform to agent local lidar frame (agent本地lidar坐标系)
        # 计算 rots @ inv(intrinsics)
        inv_intrins = torch.inverse(intrinsics)
        combine = rots.matmul(inv_intrins)  # [B, N, 3, 3]
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)  # [B, N, D, H, W, 3]
        points = points + trans.view(B, N, 1, 1, 1, 3)  # 加上平移
        # 此时points在agent本地lidar坐标系中

        # Step 4: Transform to ego frame (如果提供了变换矩阵)
        # 将agent本地坐标系中的点变换到ego坐标系
        if agent_to_ego_transform is not None:
            # agent_to_ego_transform: [B, N, 4, 4]
            # 提取旋转和平移部分
            agent_rots = agent_to_ego_transform[:, :, :3, :3]  # [B, N, 3, 3]
            agent_trans = agent_to_ego_transform[:, :, :3, 3]  # [B, N, 3]
            # 应用旋转变换
            points = agent_rots.view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)  # [B, N, D, H, W, 3]
            # 应用平移变换
            points = points + agent_trans.view(B, N, 1, 1, 1, 3)  # [B, N, D, H, W, 3]
        # 如果未提供变换矩阵，points仍在agent本地坐标系中（需要后续在agent_fuser中对齐）

        return points  # [B, N, D, H, W, 3] (在ego坐标系中，如果提供了agent_to_ego_transform)
    
    # ====================================================
    # Step 3: scatter_add 生成 TPV (优化版：基于 GPU 的 batched scatter)
    # ====================================================
    def _build_tpv_from_lss(self, image_feat, depth_prob, world_coords):
        B, N, C, H, W = image_feat.shape
        D = self.depth_bins
        device = image_feat.device

        # 初始化三平面
        tpv_xy = torch.zeros(B, C, self.tpv_size[0], self.tpv_size[1], device=device)
        tpv_xz = torch.zeros(B, C, self.tpv_size[1], self.tpv_size[2], device=device)
        tpv_yz = torch.zeros(B, C, self.tpv_size[0], self.tpv_size[2], device=device)

        # 全展开但避免复制 tensor
        coords = world_coords.reshape(B, N, D * H * W, 3)
        probs = depth_prob.reshape(B, N, D * H * W)
        feats = (
            image_feat.permute(0, 1, 3, 4, 2)
            .reshape(B, N, H * W, C)
            .unsqueeze(2)
            .repeat(1, 1, D, 1, 1)
            .reshape(B, N, D * H * W, C)
        )

        # 将世界坐标 [x,y,z] 转换为体素索引 [x_idx,y_idx,z_idx]
        # 直接使用注册的 buffer，它们会自动在正确的 device 上
        vxyz = ((coords - self.pc_min) / self.voxel_size_tensor).long()  # [B,N,D*H*W,3]

        # ✅ 用 ... 在最后一维上 clamp，而不是 vxyz[:,:,0]
        vxyz[..., 0] = torch.clamp(vxyz[..., 0], 0, self.tpv_size[1] - 1)  # x
        vxyz[..., 1] = torch.clamp(vxyz[..., 1], 0, self.tpv_size[0] - 1)  # y
        vxyz[..., 2] = torch.clamp(vxyz[..., 2], 0, self.tpv_size[2] - 1)  # z

        for b in range(B):
            vi_batch = vxyz[b].reshape(-1, 3)   # [N*D*H*W, 3] = [x,y,z]
            vf_batch = feats[b].reshape(-1, C)  # [N*D*H*W, C]
            vp_batch = probs[b].reshape(-1)     # [N*D*H*W]

            # 过滤无效点
            valid   = vp_batch > 1e-4
            vi_valid = vi_batch[valid]
            vf_valid = vf_batch[valid]
            vp_valid = vp_batch[valid]
            if vi_valid.shape[0] == 0:
                continue

            # 计算平面展平索引（tpv_size=[H,W,D]）
            x_idx, y_idx, z_idx = vi_valid[:, 0], vi_valid[:, 1], vi_valid[:, 2]
            flat_xy = y_idx * self.tpv_size[1] + x_idx          # H×W
            flat_xz = x_idx * self.tpv_size[2] + z_idx          # W×D
            flat_yz = y_idx * self.tpv_size[2] + z_idx          # H×D

            weighted_feats = vf_valid * vp_valid.unsqueeze(1)

            tpv_xy[b].view(C, -1).index_add_(1, flat_xy, weighted_feats.T)
            tpv_xz[b].view(C, -1).index_add_(1, flat_xz, weighted_feats.T)
            tpv_yz[b].view(C, -1).index_add_(1, flat_yz, weighted_feats.T)

        return {"xy": tpv_xy, "xz": tpv_xz, "yz": tpv_yz}

    def _build_tpv_from_lss_v2(self, image_feat, depth_prob, world_coords):
        """
        优化版 TPV 投影 (PyTorch ≥2.0, 使用 scatter_reduce 实现 GPU 全并行)
        - 保留 xy/xz/yz 三平面逻辑
        - 与 BEVPoolv2 等价的 rank 聚合逻辑
        - 无 Python 循环，无排序
        """
        B, N, C, H, W = image_feat.shape
        D = self.depth_bins
        device = image_feat.device

        # Step 1: flatten 所有输入
        coords = world_coords.reshape(B, N, D * H * W, 3)
        probs = depth_prob.reshape(B, N, D * H * W)
        feats = image_feat.permute(0, 1, 3, 4, 2).reshape(B, N, 1, H * W, C)
        feats = feats.repeat(1, 1, D, 1, 1).reshape(B, N, D * H * W, C)

        # 世界坐标 -> 体素索引
        # 直接使用注册的 buffer，它们会自动在正确的 device 上
        voxel_indices = ((coords - self.pc_min) / self.voxel_size_tensor).long()

        # TODO: 或许可以加一个掩码，把那些超出范围的点去掉
        # Clamp 保证合法索引（与 v1 对齐：x->W, y->H, z->D）
        H_tpv, W_tpv, D_tpv = self.tpv_size   #[200, 704, 16]
        # coords[...,0] 是 x
        voxel_indices[..., 0] = torch.clamp(voxel_indices[..., 0], 0, W_tpv - 1)  # x -> [0, W-1]
        # coords[...,1] 是 y
        voxel_indices[..., 1] = torch.clamp(voxel_indices[..., 1], 0, H_tpv - 1)  # y -> [0, H-1]
        voxel_indices[..., 2] = torch.clamp(voxel_indices[..., 2], 0, D_tpv - 1)  # z -> [0, D-1]

        # 合并 batch + camera
        voxel_indices = voxel_indices.reshape(-1, 3)
        feats = feats.reshape(-1, C)
        probs = probs.reshape(-1)
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, D * H * W).reshape(-1)

        valid = probs > 1e-4
        voxel_indices, feats, probs, batch_idx = \
            voxel_indices[valid], feats[valid], probs[valid], batch_idx[valid]

        # Step 2: 计算各平面 rank（每个 batch 内唯一，与 v1 对齐）
        x = voxel_indices[:, 0]  # [0, W-1]
        y = voxel_indices[:, 1]  # [0, H-1]
        z = voxel_indices[:, 2]  # [0, D-1]
        Hy, Wx, Dz = H_tpv, W_tpv, D_tpv

        rank_xy = batch_idx * (Hy * Wx) + (y * Wx + x)   # (H, W)
        rank_xz = batch_idx * (Wx * Dz) + (x * Dz + z)   # (W, D)
        rank_yz = batch_idx * (Hy * Dz) + (y * Dz + z)   # (H, D)

        weighted_feats = feats * probs.unsqueeze(1)

        # Step 3: 定义 GPU 原语聚合函数 (无循环, 无排序)
        def scatter_plane(rank, feats, plane_shape):
            """
            plane_shape: (H, W) 或 (W, D) 或 (H, D)
            rank: 行优先索引，rank = batch_idx * (H * W) + (y * W + x)
            与 v1 版本对齐：输出 [B, C, H, W] 格式
            """
            plane_voxels = plane_shape[0] * plane_shape[1]
            num_total = B * plane_voxels
            
            # 检查 rank 是否越界
            if rank.numel() > 0:
                max_rank = rank.max().item()
                if max_rank >= num_total:
                    raise ValueError(f"rank out of bounds: max_rank={max_rank}, num_total={num_total}")
            
            pooled = torch.zeros(num_total, C, device=device, dtype=feats.dtype)
            if rank.numel() > 0:
                pooled.scatter_reduce_(
                    dim=0,
                    index=rank.unsqueeze(1).expand(-1, C),
                    src=feats,
                    reduce="sum",   #TODO: sum好一点还是mean好一点
                    include_self=False
                )
            # pooled: [B * H * W, C] -> [B, H * W, C] -> [B, C, H * W] -> [B, C, H, W]
            # 按照行优先顺序：flat_idx = y * W + x -> reshape 为 [y, x] = [H, W]
            pooled_2d = pooled.view(B, plane_voxels, C).permute(0, 2, 1)  # [B, C, H*W]
            # plane_shape=(H, W)，reshape 为 [B, C, H, W]
            return pooled_2d.reshape(B, C, plane_shape[0], plane_shape[1])

        # Step 4: 三平面聚合
        
        tpv_xy = scatter_plane(rank_xy, weighted_feats, (Hy, Wx))
        tpv_xz = scatter_plane(rank_xz, weighted_feats, (Wx, Dz))
        tpv_yz = scatter_plane(rank_yz, weighted_feats, (Hy, Dz))

        return {"xy": tpv_xy, "xz": tpv_xz, "yz": tpv_yz}

    def _visualize_gaussians_3d(self, mu, in_range, pc_range, save_path):
        """
        在三维空间中可视化所有高斯点：范围内为绿色，范围外为红色，并绘制 POINT_CLOUD_RANGE 方框。
        Args:
            mu: [K, 3] 高斯中心 (x, y, z)，numpy 或 tensor
            in_range: [K] bool，是否在范围内
            pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
            save_path: 保存路径（.png）
        """
        if isinstance(mu, torch.Tensor):
            mu = mu.detach().cpu().numpy()
        in_range = in_range.detach().cpu().numpy() if isinstance(in_range, torch.Tensor) else np.asarray(in_range)
        if isinstance(pc_range, torch.Tensor):
            pc_range = pc_range.detach().cpu().numpy()
        pc_range = np.asarray(pc_range, dtype=np.float64)
        x_min, y_min, z_min = pc_range[0], pc_range[1], pc_range[2]
        x_max, y_max, z_max = pc_range[3], pc_range[4], pc_range[5]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        # 范围内：绿色
        if np.any(in_range):
            p_in = mu[in_range]
            ax.scatter(p_in[:, 0], p_in[:, 1], p_in[:, 2], c='green', s=1, alpha=0.6, label='in range')
        # 范围外：红色
        if np.any(~in_range):
            p_out = mu[~in_range]
            ax.scatter(p_out[:, 0], p_out[:, 1], p_out[:, 2], c='red', s=1, alpha=0.6, label='out of range')
        # 绘制 POINT_CLOUD_RANGE 方框（立方体 12 条棱）
        verts = np.array([
            [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]
        ])
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        for i, j in edges:
            ax.plot(verts[[i, j], 0], verts[[i, j], 1], verts[[i, j], 2], 'k-', linewidth=0.8)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title('Gaussians: green=in POINT_CLOUD_RANGE, red=out')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)

    # ====================================================
    # Step 4: conf_map 控制高斯生成
    # ====================================================
    def _generate_gaussians(self, conf_map, topk_mask, image_feat, depth_prob, world_coords):
        """
        生成高斯点 - 全部使用低分辨率特征，避免上采样
        Args:
            det_out: dict {
                'probs': [B, N, M, H, W],        # 低分辨率 softmax 概率
                'topk_mask': [B, N, H, W],       # Top-K 像素 mask
                'argmax_cls': [B, N, H, W]      # argmax 类别索引
              }
            image_feat: [B, N, C, H, W]          # 低分辨率特征 (64x176)
            depth_prob: [B, N, D, H, W]          # 低分辨率深度概率 (64x176)
            world_coords: [B, N, D, H, W, 3]      # 低分辨率世界坐标 (64x176)
        """
        B, N, C, H, W = image_feat.shape
        probs = conf_map  # [B, N, M, H, W]
        M = probs.shape[2]
        device = image_feat.device
        D = self.depth_bins

        # 直接收集所有高斯点（全部基于低分辨率网格）
        all_mu = []
        all_scale = []
        all_rotation = []
        all_features = []
        all_sem_emb = []
        
        for b in range(B):
            for n in range(N):
                # 直接使用 detection head 计算好的最终 mask（Top-K 且非空类）
                if topk_mask is not None:
                    mask = topk_mask[b, n]  # [H, W]
                else:
                    # 如果没有 topk_mask，回退到只用 argmax 判断非空类
                    cls_idx_map = probs[b, n].argmax(dim=0)  # [H,W]
                    mask = (cls_idx_map != self.empty_idx)   # [H, W]
                
                coords_2d = mask.nonzero(as_tuple=False)  # [num_pixels, 2] (y, x)
                if coords_2d.shape[0] == 0:
                    continue

                # 直接从低分辨率 depth_prob 索引
                dprob = depth_prob[b, n, :, coords_2d[:, 0], coords_2d[:, 1]].T  # [num_pixels, D]

                # 为每个像素取 M 维 softmax 概率向量，经 MLP 得到语义嵌入 ℝ^4
                p_vec = probs[b, n, :, coords_2d[:, 0], coords_2d[:, 1]].T  # [num_pixels, M]
                sem_emb_per_pixel = self.semantic_mlp(p_vec)  # [num_pixels, 4]   #TODO: 是不是应该和3d共用
                
                # 动态 TopK: 根据置信度自适应调整
                # 优化2：合并操作，直接从probs中提取非空类的最大概率（避免中间变量）
                if len(self._nonempty_indices) > 0:
                    # 先获取所有类别在所有像素位置的概率，再选择非空类别
                    max_nonempty_prob = probs[b, n, :, coords_2d[:, 0], coords_2d[:, 1]][self._nonempty_indices, :].max(dim=0)[0]  # [num_pixels]
                    avg_conf = max_nonempty_prob.mean().item()
                    # 有效像素处，非空类别最大概率的平均值
                else:
                    # 如果没有非空类，使用所有类的最大概率
                    max_nonempty_prob = probs[b, n, :, coords_2d[:, 0], coords_2d[:, 1]].max(dim=0)[0]  # [num_pixels]
                    avg_conf = 0.5  # 默认值
                
                # 根据平均置信度自适应调整Top-K深度数量
                adaptive_k = max(5, int(self.top_k_depths * avg_conf * 2))
                adaptive_k = min(adaptive_k, D)  # 不超过总深度bin数

                # 仅从 dprob 中落在 POINT_CLOUD_RANGE 内的深度 bin 取 adaptive_k 个：先建 [P, D] 范围内 mask
                P = dprob.shape[0]
                d_idx = torch.arange(D, device=device).unsqueeze(0).expand(P, -1)  # [P, D]
                y_idx = coords_2d[:, 0:1].expand(-1, D)
                x_idx = coords_2d[:, 1:2].expand(-1, D)
                wcoord_all = world_coords[b, n][d_idx, y_idx, x_idx, :]  # [P, D, 3]
                if self.pc_range is not None:
                    pc = self.pc_range
                    if not isinstance(pc, torch.Tensor):
                        pc = torch.tensor(pc, device=device, dtype=wcoord_all.dtype)
                    pc_min = pc[:3].view(1, 1, 3)
                    pc_max = pc[3:6].view(1, 1, 3)
                    in_range_mask_d = (
                        (wcoord_all >= pc_min) & (wcoord_all <= pc_max)
                    ).all(dim=-1)  # [P, D]
                else:
                    in_range_mask_d = torch.ones((P, D), dtype=torch.bool, device=device)
                dprob_masked = dprob.clone()
                dprob_masked[~in_range_mask_d] = -float('inf')
                topk_prob, topk_idx = torch.topk(dprob_masked, adaptive_k, dim=1)

                # TopK 中可能包含 -inf 填充（当某像素范围内深度 bin 不足 adaptive_k 时），用 isfinite 标记有效候选
                in_range_candidates = torch.isfinite(topk_prob)
                K = topk_idx.shape[1]

                total_candidates = in_range_candidates.sum().item()
                if total_candidates == 0:
                    continue

                # 自适应阈值：仅基于范围内候选数量与范围内概率
                if self.use_adaptive_threshold:
                    target_count = int(total_candidates * self.target_gaussians_ratio)
                    target_count = max(self.min_gaussians_per_camera,
                                     min(target_count, self.max_gaussians_per_camera))
                    flat_probs_in = topk_prob[in_range_candidates]
                    if flat_probs_in.numel() <= target_count:
                        adaptive_threshold = self.gaussian_threshold
                    else:
                        topk_values, _ = torch.topk(flat_probs_in, target_count, largest=True)
                        adaptive_threshold = topk_values[-1].item()
                        adaptive_threshold = max(adaptive_threshold, self.gaussian_threshold)
                else:
                    adaptive_threshold = self.gaussian_threshold

                valid_mask = (topk_prob > adaptive_threshold) & in_range_candidates
                sel_idx = torch.nonzero(valid_mask, as_tuple=False)

                # 强制保证最小高斯数量：仅从范围内候选中补足
                if sel_idx.shape[0] < self.min_gaussians_per_camera and total_candidates > 0:
                    flat_probs_in = topk_prob[in_range_candidates]
                    min_count = min(self.min_gaussians_per_camera, flat_probs_in.numel())
                    if min_count > 0:
                        topk_values, _ = torch.topk(flat_probs_in, min_count, largest=True)
                        adaptive_threshold = topk_values[-1].item()
                        valid_mask = (topk_prob > adaptive_threshold) & in_range_candidates
                        sel_idx = torch.nonzero(valid_mask, as_tuple=False)

                if sel_idx.shape[0] == 0:
                    continue
                            
                # 获取所有有效像素与深度索引
                px = coords_2d[sel_idx[:, 0]]  # [K', 2] (y, x)
                dz = topk_idx[sel_idx[:, 0], sel_idx[:, 1]]  # [K']
                pprob = topk_prob[sel_idx[:, 0], sel_idx[:, 1]]  # [K']
                
                # 使用soft mask：将像素置信度纳入概率计算
                # 获取选中像素的置信度（非空类的最大概率）
                px_pixel_conf = max_nonempty_prob[sel_idx[:, 0]]  # [K']
                pprob = pprob * px_pixel_conf  # 加权深度概率：将像素置信度与深度概率相乘
                
                # 对应的世界坐标（直接使用低分辨率索引）
                wcoord = world_coords[b, n, dz, px[:, 0], px[:, 1]]  # [K', 3]
                # 图像特征（直接使用低分辨率索引）
                feat = image_feat[b, n, :, px[:, 0], px[:, 1]].T * pprob.unsqueeze(1)  # [K', C]

                # 高斯参数计算：使用基于概率的指数插值生成scale
                # 插值区间调整为 [(s_max+s_min)/4, (s_max+s_min)*3/4]
                # 当pprob越接近0 → scale越接近(s_max+s_min)/4
                # 当pprob越接近1 → scale越接近(s_max+s_min)*3/4
                s_min, s_max = self.scale_range[0].item(), self.scale_range[1].item()
                scale_min = (s_max + s_min) / 4.0
                scale_max = (s_max + s_min) * 3.0 / 4.0
                scale = (scale_min * (scale_max / scale_min) ** pprob.unsqueeze(1)).repeat(1, 3)  # [K', 3]
                # 构造旋转四元数 [w, x, y, z] = [1, 0, 0, 0]（单位四元数，无旋转）
                rotation = torch.ones((wcoord.size(0), 4), device=device)
                rotation[:, 1:] = 0.0  # [K', 4]

                # 语义嵌入：对每个选中的像素-深度对复用对应像素的嵌入
                sem_emb = sem_emb_per_pixel[sel_idx[:, 0]]  # [K', 4]

                # 直接保存高斯点参数
                all_mu.append(wcoord)
                all_scale.append(scale)
                all_rotation.append(rotation)
                all_features.append(feat)
                all_sem_emb.append(sem_emb)
               
        
        # 堆叠为统一格式 [K, D]
        if len(all_mu) == 0:
            gaussians_compressed = {
                'mu': torch.empty(0, 3, device=device),
                'scale': torch.empty(0, 3, device=device),
                'rotation': torch.empty(0, 4, device=device),
                'features': torch.empty(0, C, device=device),
                'semantic': torch.empty(0, 4, device=device)
            }
        else:
            gaussians_compressed = {
                'mu': torch.cat(all_mu, dim=0),  # [K, 3]
                'scale': torch.cat(all_scale, dim=0),  # [K, 3]
                'rotation': torch.cat(all_rotation, dim=0),  # [K, 4]
                'features': torch.cat(all_features, dim=0),  # [K, C]
                'semantic': torch.cat(all_sem_emb, dim=0)  # [K, 4]
            }

            # 三维可视化：范围内绿色、范围外红色（在过滤前对全部高斯点可视化）
            if self.visualize_gaussians_3d and self.pc_range is not None and gaussians_compressed['mu'].shape[0] > 0:
                pc = self.pc_range
                if not isinstance(pc, torch.Tensor):
                    pc = torch.tensor(pc, device=device, dtype=gaussians_compressed['mu'].dtype)
                x_min, y_min, z_min = pc[0].item(), pc[1].item(), pc[2].item()
                x_max, y_max, z_max = pc[3].item(), pc[4].item(), pc[5].item()
                mu_all = gaussians_compressed['mu']
                in_range_all = (
                    (mu_all[:, 0] >= x_min) & (mu_all[:, 0] <= x_max) &
                    (mu_all[:, 1] >= y_min) & (mu_all[:, 1] <= y_max) &
                    (mu_all[:, 2] >= z_min) & (mu_all[:, 2] <= z_max)
                )
                save_path = os.path.join(
                    self.gaussians_visualization_dir,
                    f'gaussians_3d_{self._gaussians_vis_count:06d}.png'
                )
                self._visualize_gaussians_3d(mu_all, in_range_all, self.pc_range, save_path)
                self._gaussians_vis_count += 1

            # 去掉落在 POINT_CLOUD_RANGE 之外的高斯点
            if self.pc_range is not None and gaussians_compressed['mu'].shape[0] > 0:
                pc = self.pc_range
                if not isinstance(pc, torch.Tensor):
                    pc = torch.tensor(pc, device=device, dtype=gaussians_compressed['mu'].dtype)
                x_min, y_min, z_min = pc[0].item(), pc[1].item(), pc[2].item()
                x_max, y_max, z_max = pc[3].item(), pc[4].item(), pc[5].item()
                mu = gaussians_compressed['mu']  # [K, 3] (x, y, z)
                in_range = (
                    (mu[:, 0] >= x_min) & (mu[:, 0] <= x_max) &
                    (mu[:, 1] >= y_min) & (mu[:, 1] <= y_max) &
                    (mu[:, 2] >= z_min) & (mu[:, 2] <= z_max)
                )
                if in_range.any():
                    gaussians_compressed = {k: v[in_range] for k, v in gaussians_compressed.items()}
                else:
                    gaussians_compressed = {
                        'mu': torch.empty(0, 3, device=device),
                        'scale': torch.empty(0, 3, device=device),
                        'rotation': torch.empty(0, 4, device=device),
                        'features': torch.empty(0, C, device=device),
                        'semantic': torch.empty(0, 4, device=device)
                    }

        # 添加高斯点数量日志
        num_gaussians = gaussians_compressed['mu'].shape[0]
        print(f"[GaussianTPV] Generated {num_gaussians} Gaussians at {H}×{W} (low-res, no upsampling)")

        return gaussians_compressed
