# -*- coding: utf-8 -*-
"""
语义标签工具使用示例
演示如何在backbone3d_semantic.py和backbone2d_semantic.py中使用语义标签查询功能
"""

import torch
import numpy as np
from opencood.utils.seg_label_utils import SegLabelMapper, query_point_labels


def example_usage_backbone3d():
    """
    示例：在backbone3d_semantic.py中使用语义标签查询
    """
    print("=" * 60)
    print("示例1: backbone3d_semantic.py 中的使用")
    print("=" * 60)
    
    # 1. 创建标签映射器
    seg_hw = 512
    seg_res = 0.25
    lidar_range = [-140.8, -40.0, -3.0, 140.8, 40.0, 1.0]
    mapper = SegLabelMapper(seg_hw=seg_hw, seg_res=seg_res, lidar_range=lidar_range, ego_center=True)
    
    # 2. 模拟BEV分割标签图（512x512）
    seg_label = torch.randint(0, 8, (1, seg_hw, seg_hw), dtype=torch.long)  # [B, H, W]
    
    # 3. 模拟体素坐标 [N, 4] = [batch, z, y, x]
    voxel_coords = torch.tensor([
        [0, 10, 100, 200],  # batch=0, z=10, y=100, x=200
        [0, 15, 150, 300],
        [0, 20, 200, 400],
    ], dtype=torch.long)
    
    # 4. 体素参数
    voxel_size = [0.4, 0.4, 2.0]
    point_cloud_range = [-140.8, -40.0, -3.0, 140.8, 40.0, 1.0]
    
    # 5. 查询语义标签
    labels, valid_mask = mapper.voxel_coords_to_labels(
        seg_label=seg_label,
        voxel_coords=voxel_coords,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        default_label=0
    )
    
    print(f"体素坐标形状: {voxel_coords.shape}")
    print(f"查询到的标签: {labels}")
    print(f"有效掩码: {valid_mask}")
    print()


def example_usage_backbone2d():
    """
    示例：在backbone2d_semantic.py中使用语义标签查询
    """
    print("=" * 60)
    print("示例2: backbone2d_semantic.py 中的使用")
    print("=" * 60)
    
    # 1. 创建标签映射器
    seg_hw = 512
    seg_res = 0.25
    lidar_range = [-140.8, -40.0, -3.0, 140.8, 40.0, 1.0]
    mapper = SegLabelMapper(seg_hw=seg_hw, seg_res=seg_res, lidar_range=lidar_range, ego_center=True)
    
    # 2. 模拟BEV分割标签图
    seg_label = torch.randint(0, 8, (seg_hw, seg_hw), dtype=torch.long)  # [H, W]
    
    # 3. 模拟3D点坐标（高斯点的世界坐标）
    world_coords = torch.tensor([
        [10.5, 5.2, 0.5],   # (x, y, z)
        [-5.3, 8.1, 0.3],
        [15.0, -10.0, 0.8],
    ], dtype=torch.float32)
    
    # 4. 查询语义标签
    labels, valid_mask = mapper.query_labels(
        seg_label=seg_label,
        world_coords=world_coords,
        default_label=0
    )
    
    print(f"世界坐标形状: {world_coords.shape}")
    print(f"查询到的标签: {labels}")
    print(f"有效掩码: {valid_mask}")
    print()


def example_usage_point_cloud():
    """
    示例：直接从点云查询语义标签
    """
    print("=" * 60)
    print("示例3: 从点云查询语义标签")
    print("=" * 60)
    
    # 1. 模拟点云数据 [N, 4] = [x, y, z, intensity]
    points = torch.tensor([
        [10.5, 5.2, 0.5, 0.8],
        [-5.3, 8.1, 0.3, 0.6],
        [15.0, -10.0, 0.8, 0.9],
    ], dtype=torch.float32)
    
    # 2. 模拟BEV分割标签图
    seg_label = torch.randint(0, 8, (512, 512), dtype=torch.long)
    
    # 3. 使用便捷函数查询
    labels, valid_mask = query_point_labels(
        points=points,
        seg_label=seg_label,
        seg_hw=512,
        seg_res=0.25,
        lidar_range=[-140.8, -40.0, -3.0, 140.8, 40.0, 1.0],
        default_label=0
    )
    
    print(f"点云形状: {points.shape}")
    print(f"查询到的标签: {labels}")
    print(f"有效掩码: {valid_mask}")
    print()


def example_coordinate_mapping():
    """
    示例：坐标映射的详细说明
    """
    print("=" * 60)
    print("示例4: 坐标映射说明")
    print("=" * 60)
    
    mapper = SegLabelMapper(seg_hw=512, seg_res=0.25, ego_center=True)
    
    # 测试几个已知坐标
    test_coords = torch.tensor([
        [0.0, 0.0, 0.0],      # ego中心
        [32.0, 32.0, 0.0],   # 右上象限
        [-32.0, -32.0, 0.0], # 左下象限
    ], dtype=torch.float32)
    
    pixel_coords, valid_mask = mapper.world_to_pixel(test_coords)
    
    print("世界坐标 -> 像素坐标映射:")
    for i, (world, pixel) in enumerate(zip(test_coords, pixel_coords)):
        print(f"  点{i+1}: 世界坐标 ({world[0]:.1f}, {world[1]:.1f}) -> "
              f"像素坐标 ({pixel[0]:.0f}, {pixel[1]:.0f}), 有效: {valid_mask[i]}")
    print()


if __name__ == "__main__":
    # 运行所有示例
    example_usage_backbone3d()
    example_usage_backbone2d()
    example_usage_point_cloud()
    example_coordinate_mapping()
    
    print("=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


