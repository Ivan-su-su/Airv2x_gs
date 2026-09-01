#!/usr/bin/env python3
"""
简单BEV可视化使用示例

这个脚本展示了如何使用新的简单BEV可视化功能
"""

import torch
import numpy as np
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pointpillar_scatter import PointPillarScatter3d

def create_mock_model_config():
    """创建模拟的模型配置"""
    class MockConfig:
        INPUT_SHAPE = [504, 200, 1]  # nx, ny, nz - 根据where2comm配置调整
        NUM_BEV_FEATURES = 128
    
    return MockConfig()

def create_single_agent_data():
    """创建单agent的模拟数据"""
    batch_size = 1
    num_pillars = 2000
    num_features = 128
    
    # 创建模拟的pillar features
    pillar_features = torch.randn(num_pillars, num_features)
    
    # 创建voxel coords: [batch_idx, z, y, x]
    batch_indices = torch.zeros(num_pillars, 1)  # 所有pillar属于batch 0
    z_coords = torch.zeros(num_pillars, 1)  # BEV中z=0
    y_coords = torch.randint(0, 200, (num_pillars, 1))  # Y方向: 0-200
    x_coords = torch.randint(0, 504, (num_pillars, 1))  # X方向: 0-504
    
    voxel_coords = torch.cat([batch_indices, z_coords, y_coords, x_coords], dim=1)
    
    batch_dict = {
        'pillar_features': pillar_features,
        'voxel_coords': voxel_coords
    }
    
    return batch_dict

def create_multi_agent_data():
    """创建多agent的模拟数据"""
    batch_dict = {}
    
    # 创建3个agent的数据
    agent_names = ['vehicle_0', 'rsu_1', 'drone_2']
    
    for agent_name in agent_names:
        num_pillars = np.random.randint(800, 2000)
        num_features = 128
        
        # 每个agent的pillar features
        pillar_features = torch.randn(num_pillars, num_features)
        
        # 每个agent的voxel coords (batch_idx=0)
        batch_indices = torch.zeros(num_pillars, 1)
        z_coords = torch.zeros(num_pillars, 1)
        
        # 为不同agent创建不同的感知区域
        if 'vehicle' in agent_name:
            # 车辆：前方区域
            y_coords = torch.randint(100, 200, (num_pillars, 1))  # 前方
            x_coords = torch.randint(200, 350, (num_pillars, 1))  # 中间
        elif 'rsu' in agent_name:
            # RSU：全区域
            y_coords = torch.randint(0, 200, (num_pillars, 1))
            x_coords = torch.randint(0, 504, (num_pillars, 1))
        else:  # drone
            # 无人机：高空俯视，覆盖更大区域
            y_coords = torch.randint(0, 200, (num_pillars, 1))
            x_coords = torch.randint(0, 504, (num_pillars, 1))
        
        voxel_coords = torch.cat([batch_indices, z_coords, y_coords, x_coords], dim=1)
        
        batch_dict[agent_name] = {
            'pillar_features': pillar_features,
            'voxel_coords': voxel_coords
        }
    
    return batch_dict

def main():
    """主函数：演示简单BEV可视化功能"""
    print("=== 简单BEV可视化示例 ===")
    
    # 创建模型配置
    model_cfg = create_mock_model_config()
    
    # 创建PointPillarScatter3d实例
    scatter_module = PointPillarScatter3d(model_cfg, None)
    
    print("\n1. 单Agent BEV可视化")
    print("-" * 40)
    
    # 创建单agent数据
    batch_dict = create_single_agent_data()
    
    # 进行BEV投影并可视化
    result_dict = scatter_module(batch_dict, visualize=True)
    
    print(f"BEV特征形状: {result_dict['spatial_features'].shape}")
    
    print("\n2. 多Agent BEV可视化")
    print("-" * 40)
    
    # 创建多agent数据
    multi_agent_dict = create_multi_agent_data()
    
    # 进行BEV投影并可视化
    result_dict = scatter_module(multi_agent_dict, visualize=True)
    
    print(f"融合后BEV特征形状: {result_dict['spatial_features'].shape}")
    
    print("\n3. 手动调用简单可视化函数")
    print("-" * 40)
    
    # 手动调用简单可视化函数
    scatter_module.visualize_bev_simple(result_dict, save_dir="./manual_simple_bev")
    scatter_module.visualize_multi_agent_bev_simple(result_dict, save_dir="./manual_multi_agent_simple")
    
    print("\n=== 简单可视化完成 ===")
    print("请查看以下目录中的可视化结果:")
    print("- ./multi_agent_bev_independent/  (各agent独立的BEV特征)")
    print("- ./bev_simple/                   (单agent或融合后的BEV特征)")
    print("- ./manual_simple_bev/            (手动调用的单agent可视化)")
    print("- ./manual_multi_agent_simple/    (手动调用的多agent可视化)")
    print("\n可视化说明:")
    print("- 热力图显示BEV网格的占用强度")
    print("- 红色/黄色区域表示高强度占用")
    print("- 黑色区域表示无占用")
    print("- 统计信息显示占用比例和强度分布")
    print("- 多agent可视化显示各agent独立的感知范围")

if __name__ == "__main__":
    main()
