# -*- coding: utf-8 -*-
"""
语义标签工具函数
用于将BEV分割标签图映射到真实世界坐标系，并支持从3D点查询语义标签
"""

import torch
import torch.nn.functional as F
import numpy as np
###
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
from pathlib import Path


class SegLabelMapper:
    """
    将BEV分割标签图映射到真实世界坐标系
    支持从3D点坐标查询对应的语义标签
    """
    
    def __init__(self, seg_hw=512, seg_res=0.25, lidar_range=None, ego_center=True):
        """
        Args:
            seg_hw: BEV标签图的高度和宽度（像素），默认512
            seg_res: 每个像素对应的真实世界距离（米），默认0.25
            lidar_range: LiDAR范围 [x_min, y_min, z_min, x_max, y_max, z_max]
                        如果为None，则假设标签图以ego为中心
            ego_center: 如果为True，标签图以ego车辆为中心；否则使用lidar_range的起点
        """
        self.seg_hw = seg_hw
        self.seg_res = seg_res
        self.ego_center = ego_center
        
        if lidar_range is not None:
            self.lidar_range = torch.tensor(lidar_range, dtype=torch.float32)
        else:
            # 默认范围：以ego为中心，128米×128米
            half_range = seg_hw * seg_res / 2.0  # 64米
            self.lidar_range = torch.tensor([
                -half_range, -half_range, -3.0,  # x_min, y_min, z_min
                half_range, half_range, 1.0      # x_max, y_max, z_max
            ], dtype=torch.float32)
        
        # 计算标签图的真实世界范围
        if ego_center:
            # 标签图以ego为中心，范围是 [-half_range, half_range]
            half_range = seg_hw * seg_res / 2.0
            self.seg_x_min = -half_range
            self.seg_y_min = -half_range
            self.seg_x_max = half_range
            self.seg_y_max = half_range
        else:
            # 使用lidar_range的起点
            self.seg_x_min = self.lidar_range[0].item()
            self.seg_y_min = self.lidar_range[1].item()
            self.seg_x_max = self.seg_x_min + seg_hw * seg_res
            self.seg_y_max = self.seg_y_min + seg_hw * seg_res
    
    def world_to_pixel(self, world_coords):
        """
        将真实世界坐标转换为BEV标签图的像素坐标
        
        Args:
            world_coords: [N, 3] 或 [N, 2] 或 [B, N, 3] 真实世界坐标 (x, y, z) 或 (x, y)
        
        Returns:
            pixel_coords: [N, 2] 或 [B, N, 2] 像素坐标 (u, v)
            valid_mask: [N] 或 [B, N] bool tensor，表示点是否在标签图范围内
        """
        
        original_shape = world_coords.shape
        if len(original_shape) == 3:
            # [B, N, 3] -> [B*N, 3]
            B, N, _ = original_shape
            world_coords = world_coords.view(-1, 3)
            has_batch = True
        else:
            # [N, 3] or [N, 2]
            world_coords = world_coords.view(-1, world_coords.shape[-1])
            has_batch = False
        
        # 提取x, y坐标
        x = world_coords[:, 0]
        y = world_coords[:, 1]
        
        # 转换为像素坐标
        # 注意：标签图经过了转置和翻转处理（label_map.T 和 label_map[:, ::-1]）
        # 原始映射：x -> W维度, y -> H维度
        # 转置后：x -> H维度, y -> W维度
        # 翻转后：y轴反转
        
        # 计算相对于标签图起点的偏移
        u = torch.floor((x - self.seg_x_min) / self.seg_res)  # x方向 -> 列索引
        v = torch.floor((y - self.seg_y_min) / self.seg_res)  # y方向 -> 行索引
        
        # 转换为整数索引
        u_idx = u.long()
        v_idx = v.long()
        # print(f"[seg_label_utils] u_idx shape: {u_idx.shape}")
        # print(f"[seg_label_utils] u_idx max: {u_idx.max()}")
        # print(f"[seg_label_utils] u_idx min: {u_idx.min()}")
        # print(f"[seg_label_utils] v_idx shape: {v_idx.shape}")
        # print(f"[seg_label_utils] v_idx max: {v_idx.max()}")
        # print(f"[seg_label_utils] v_idx min: {v_idx.min()}")
        
        # 检查是否在有效范围内
        valid_mask = (
            (u_idx >= 0) & (u_idx < self.seg_hw) &
            (v_idx >= 0) & (v_idx < self.seg_hw)
        )
        
        # 应用标签图的坐标变换（转置和翻转）
        # 转置：u_idx (x) -> v, v_idx (y) -> u
        # 翻转：v轴反转 -> seg_hw - 1 - v_idx
        # 最终：u_final = seg_hw - 1 - v_idx, v_final = u_idx
        # TODO: 有待考证
        u_final = self.seg_hw - 1 - v_idx
        v_final = u_idx
        
        pixel_coords = torch.stack([u_final, v_final], dim=-1)  # [N, 2]
        
        if has_batch:
            pixel_coords = pixel_coords.view(B, N, 2)
            valid_mask = valid_mask.view(B, N)
        
        return pixel_coords, valid_mask
    
    def query_labels(self, seg_label, world_coords, default_label=0):
        """
        从BEV标签图中查询给定3D点对应的语义标签
        
        Args:
            seg_label: [1, 512, 512] BEV分割标签图
            world_coords: [N, 3] 或 [B, N, 3] 真实世界坐标 (x, y, z)
            default_label: 超出范围的点返回的默认标签，默认0
        
        Returns:
            labels: [N] 或 [B, N] 语义标签
            valid_mask: [N] 或 [B, N] bool tensor，表示点是否在标签图范围内
        """
        if isinstance(seg_label, np.ndarray):
            seg_label = torch.from_numpy(seg_label)
        
        # 确保seg_label是3D tensor [B, H, W]
        if seg_label.dim() == 2:
            seg_label = seg_label.unsqueeze(0)  # [H, W] -> [1, H, W]
        
        B_label, H_label, W_label = seg_label.shape
        
        # 转换为像素坐标
        pixel_coords, valid_mask = self.world_to_pixel(world_coords)
        
        # 处理batch维度
        if len(world_coords.shape) == 3:
            B_coords, N_coords, _ = world_coords.shape
            pixel_coords = pixel_coords.view(-1, 2)
            valid_mask = valid_mask.view(-1)
            has_batch = True
        else:
            N_coords = world_coords.shape[0]
            pixel_coords = pixel_coords.view(-1, 2)
            valid_mask = valid_mask.view(-1)
            has_batch = False
        
        # 提取像素坐标
        u = pixel_coords[:, 0].long()  # 行索引
        v = pixel_coords[:, 1].long()  # 列索引
        
        # 确保索引在有效范围内
        u = torch.clamp(u, 0, H_label - 1)
        v = torch.clamp(v, 0, W_label - 1)
        
        # 查询标签（假设batch=0，如果需要支持多batch需要扩展）
        if has_batch:
            # 对于多batch情况，需要根据world_coords的batch索引来查询
            # 这里简化处理，假设所有点都属于第一个batch
            labels = seg_label[0, u, v]  # [N]
        else:
            labels = seg_label[0, u, v]  # [N]
        
        # 将超出范围的点设置为默认标签
        labels = torch.where(valid_mask, labels, torch.tensor(default_label, dtype=labels.dtype, device=labels.device))
        
        if has_batch:
            labels = labels.view(B_coords, N_coords)
            valid_mask = valid_mask.view(B_coords, N_coords)
        
        return labels, valid_mask
    
    def voxel_coords_to_labels(self, seg_label, voxel_coords, voxel_size, point_cloud_range, default_label=0, return_pixel_coords=False):
        """
        从体素坐标查询语义标签
        
        Args:
            seg_label: [B, H, W] 或 [H, W] BEV分割标签图
            voxel_coords: [N, 4] 体素坐标 [batch, z, y, x]
            voxel_size: [3] 体素大小 [vx, vy, vz]
            point_cloud_range: [6] 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
            default_label: 默认标签
            return_pixel_coords: 是否返回像素坐标
        
        Returns:
            labels: [N] 语义标签
            valid_mask: [N] bool tensor
            pixel_coords: [N, 2] 像素坐标 (u, v)，仅在 return_pixel_coords=True 时返回
        """
        if isinstance(seg_label, np.ndarray):
            seg_label = torch.from_numpy(seg_label)
        if isinstance(voxel_coords, np.ndarray):
            voxel_coords = torch.from_numpy(voxel_coords)
        
        # 将体素坐标转换为真实世界坐标(体素中心在世界坐标系中的位置)
        # voxel_coords: [N, 4] = [batch, z, y, x]
        world_x = voxel_coords[:, 3].float() * voxel_size[0] + point_cloud_range[0] + voxel_size[0] / 2
        world_y = voxel_coords[:, 2].float() * voxel_size[1] + point_cloud_range[1] + voxel_size[1] / 2
        world_z = voxel_coords[:, 1].float() * voxel_size[2] + point_cloud_range[2] + voxel_size[2] / 2
        
        world_coords = torch.stack([world_x, world_y, world_z], dim=-1)  # [N, 3]
        
        # 转换为像素坐标
        pixel_coords, valid_mask_pixel = self.world_to_pixel(world_coords)
        
        # 查询标签
        labels, valid_mask = self.query_labels(seg_label, world_coords, default_label)
        
        if return_pixel_coords:
            return labels, valid_mask, pixel_coords
        else:
            return labels, valid_mask


def create_seg_label_mapper_from_config(config):
    """
    从配置文件中创建SegLabelMapper实例
    
    Args:
        config: 配置字典，应包含：
            - seg_hw: BEV标签图尺寸
            - seg_res: BEV分辨率
            - preprocess.cav_lidar_range: LiDAR范围（可选）
    
    Returns:
        SegLabelMapper实例
    """
    seg_hw = config.get('seg_hw', 512)
    seg_res = config.get('seg_res', 0.25)
    
    # 尝试从preprocess配置中获取lidar_range
    lidar_range = None
    if 'preprocess' in config:
        lidar_range = config['preprocess'].get('cav_lidar_range', None)
    if lidar_range is None and 'POINT_CLOUD_RANGE' in config:
        lidar_range = config['POINT_CLOUD_RANGE']
    
    # 默认假设标签图以ego为中心
    ego_center = True
    
    return SegLabelMapper(seg_hw=seg_hw, seg_res=seg_res, lidar_range=lidar_range, ego_center=ego_center)


# 便捷函数：直接查询点云标签
def query_point_labels(points, seg_label, seg_hw=512, seg_res=0.25, lidar_range=None, default_label=0):
    """
    便捷函数：直接查询点云的语义标签
    
    Args:
        points: [N, 3] 或 [N, 4] 点云坐标 (x, y, z) 或 (x, y, z, intensity)
        seg_label: [H, W] 或 [B, H, W] BEV分割标签图
        seg_hw: BEV标签图尺寸
        seg_res: BEV分辨率
        lidar_range: LiDAR范围（可选）
        default_label: 默认标签
    
    Returns:
        labels: [N] 语义标签
        valid_mask: [N] bool tensor
    """
    mapper = SegLabelMapper(seg_hw=seg_hw, seg_res=seg_res, lidar_range=lidar_range)
    
    # 提取x, y, z坐标
    if points.shape[1] >= 3:
        world_coords = points[:, :3]
    else:
        raise ValueError(f"点云坐标维度不足，期望至少3维，得到{points.shape[1]}维")
    
    labels, valid_mask = mapper.query_labels(seg_label, world_coords, default_label)
    return labels, valid_mask