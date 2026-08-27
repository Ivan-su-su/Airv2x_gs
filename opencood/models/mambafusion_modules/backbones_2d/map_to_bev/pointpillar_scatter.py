import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
import matplotlib.pyplot as plt
import numpy as np
import os

class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        import pdb; pdb.set_trace()
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict

class PointPillarScatter3d(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        
        self.model_cfg = model_cfg
        self.importance_generator = ImportanceGenerator(num_channels=self.model_cfg.NUM_BEV_FEATURES, use_softmax=True)
        self.nx, self.ny, self.nz = self.model_cfg.INPUT_SHAPE
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_bev_features_before_compression = self.model_cfg.NUM_BEV_FEATURES // self.nz

        

    def _generate_bev_features(self, pillar_features, coords):
        """
        为单个agent生成BEV特征的辅助方法
        
        Args:
            pillar_features: [num_pillars, num_features]
            coords: [num_pillars, 4] (batch_idx, z, y, x)
            
        Returns:
            bev_features: [batch_size, num_features, ny, nx]
        """
        
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features_before_compression,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            
            indices = this_coords[:, 1] * self.ny * self.nx + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()


            dense_feats = scatter_mean(pillars, indices, dim=1)
            
            dense_len = dense_feats.shape[1]
            spatial_feature[:, :dense_len] = dense_feats

            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features_before_compression * self.nz, self.ny, self.nx)
        
        # 根据网格尺寸进行池化以减少内存占用
        # 修改池化策略：保持长宽比，但限制最大尺寸
            
        
        return batch_spatial_features

    def forward(self, batch_dict, agent_name=None, **kwargs):
        # 仅生成并写回单个 agent 的 BEV 特征；不做多 agent 融合与可视化
        if agent_name is not None and agent_name in batch_dict and isinstance(batch_dict[agent_name], dict):
            pillar_features = batch_dict[agent_name]['pillar_features']
            coords = batch_dict[agent_name]['voxel_coords']
            bev = self._generate_bev_features(pillar_features, coords)
            batch_dict[agent_name]['spatial_features'] = bev
            return batch_dict

        # 兼容原始单字典格式
        if 'pillar_features' in batch_dict and 'voxel_coords' in batch_dict:
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
            bev = self._generate_bev_features(pillar_features, coords)
            batch_dict['spatial_features'] = bev
            return batch_dict

        raise KeyError('Expected pillar_features/voxel_coords in batch_dict or under the specified agent_name')

    def visualize_bev_simple(self, batch_dict, save_dir="./bev_simple", agent_names=None):
        """
        简单直观的BEV可视化 - 显示网格占用和强度分布
        
        Args:
            batch_dict: 包含spatial_features的字典
            save_dir: 保存图片的目录
            agent_names: agent名称列表
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if 'spatial_features' not in batch_dict:
            print("Warning: No spatial_features found in batch_dict")
            return
            
        spatial_features = batch_dict['spatial_features']  # [B, C, H, W]
        batch_size, num_channels, height, width = spatial_features.shape
        
        print(f"[BEV Simple] Features shape: {spatial_features.shape}")
        
        # 为每个batch样本创建可视化
        for batch_idx in range(batch_size):
            batch_features = spatial_features[batch_idx]  # [C, H, W]
            
            # 计算所有通道的平均强度作为占用强度
            occupancy_map = torch.mean(batch_features, dim=0).detach().cpu().numpy()  # [H, W]
            
            # 创建可视化
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 左图：占用强度热力图
            im1 = ax1.imshow(occupancy_map, cmap='hot', aspect='equal')
            ax1.set_title('BEV Occupancy Intensity')
            ax1.set_xlabel('X (width)')
            ax1.set_ylabel('Y (height)')
            plt.colorbar(im1, ax=ax1, label='Intensity')
            
            # 右图：占用统计
            occupied_cells = np.count_nonzero(occupancy_map)
            total_cells = occupancy_map.size
            occupancy_ratio = occupied_cells / total_cells
            
            # 强度统计
            max_intensity = np.max(occupancy_map)
            mean_intensity = np.mean(occupancy_map[occupancy_map > 0]) if occupied_cells > 0 else 0
            
            # 显示统计信息
            stats_text = f"""BEV Grid Statistics:
            
Grid Size: {height} × {width}
Total Cells: {total_cells:,}
Occupied Cells: {occupied_cells:,}
Occupancy Ratio: {occupancy_ratio:.2%}

Intensity Statistics:
Max Intensity: {max_intensity:.4f}
Mean Intensity: {mean_intensity:.4f}
Non-zero Mean: {mean_intensity:.4f}"""
            
            ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')
            ax2.set_title('Statistics')
            
            # 添加总标题
            agent_name = agent_names[batch_idx] if agent_names and batch_idx < len(agent_names) else f"Agent_{batch_idx}"
            fig.suptitle(f'BEV Occupancy Map - {agent_name}', fontsize=14)
            
            # 调整布局以避免重叠 - 增加更多间距
            plt.subplots_adjust(top=0.8, bottom=0.2, hspace=0.6)
            
            # 保存图片
            save_path = os.path.join(save_dir, f'bev_simple_{agent_name}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[BEV Simple] Saved: {save_path}")
            print(f"[BEV Simple] {agent_name}: {occupied_cells:,}/{total_cells:,} cells occupied ({occupancy_ratio:.2%})")
    
    def visualize_multi_agent_bev_simple(self, batch_dict, save_dir="./multi_agent_bev_simple"):
        """
        多Agent简单BEV对比可视化
        
        Args:
            batch_dict: 包含多个agent数据的字典
            save_dir: 保存图片的目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 收集所有agent的BEV特征
        agent_bev_features = {}
        agent_names = []
        
        for k, v in batch_dict.items():
            if isinstance(v, dict) and 'spatial_features' in v:
                agent_bev_features[k] = v['spatial_features']
                agent_names.append(k)
        
        if len(agent_bev_features) == 0:
            print("Warning: No agent spatial_features found in batch_dict")
            return
            
        print(f"[Multi-Agent BEV Simple] Found {len(agent_names)} agents: {agent_names}")
        
        # 获取特征维度
        first_agent = list(agent_bev_features.values())[0]
        batch_size, num_channels, height, width = first_agent.shape
        
        # 为每个batch创建对比图
        for batch_idx in range(batch_size):
            fig, axes = plt.subplots(2, len(agent_names), figsize=(4*len(agent_names), 8))
            if len(agent_names) == 1:
                axes = axes.reshape(2, 1)
            
            # 收集所有agent的占用图用于统一颜色范围
            all_occupancy_maps = []
            for agent_name in agent_names:
                features = agent_bev_features[agent_name][batch_idx]  # [C, H, W]
                # 使用L2范数而不是平均值，避免零值问题
                occupancy_map = torch.norm(features, dim=0).detach().cpu().numpy()
                all_occupancy_maps.append(occupancy_map)
            
            # 计算统一的颜色范围
            all_values = np.concatenate([m.flatten() for m in all_occupancy_maps])
            vmin, vmax = np.min(all_values), np.max(all_values)
            
            for i, agent_name in enumerate(agent_names):
                occupancy_map = all_occupancy_maps[i]
                
                # 上排：占用强度热力图
                im1 = axes[0, i].imshow(occupancy_map, cmap='hot', aspect='equal', vmin=vmin, vmax=vmax)
                axes[0, i].set_title(f'{agent_name}\nOccupancy Intensity')
                axes[0, i].set_xlabel('X')
                axes[0, i].set_ylabel('Y')
                
                # 下排：占用统计
                occupied_cells = np.count_nonzero(occupancy_map)
                total_cells = occupancy_map.size
                occupancy_ratio = occupied_cells / total_cells
                max_intensity = np.max(occupancy_map)
                mean_intensity = np.mean(occupancy_map[occupancy_map > 0]) if occupied_cells > 0 else 0
                
                stats_text = f"""Occupied: {occupied_cells:,}/{total_cells:,}
Ratio: {occupancy_ratio:.2%}
Max: {max_intensity:.3f}
Mean: {mean_intensity:.3f}"""
                
                axes[1, i].text(0.1, 0.9, stats_text, transform=axes[1, i].transAxes, 
                              fontsize=9, verticalalignment='top', fontfamily='monospace')
                axes[1, i].set_xlim(0, 1)
                axes[1, i].set_ylim(0, 1)
                axes[1, i].axis('off')
                axes[1, i].set_title('Statistics')
            
            # 添加颜色条
            plt.colorbar(im1, ax=axes[0, :], orientation='horizontal', pad=0.1, label='Intensity')
            
            fig.suptitle(f'Multi-Agent BEV Comparison - Batch {batch_idx}', fontsize=14)
            
            # 调整布局以避免重叠 - 增加更多间距
            plt.subplots_adjust(top=0.8, bottom=0.2, hspace=0.6)
            
            # 保存对比图
            save_path = os.path.join(save_dir, f'multi_agent_bev_simple_batch_{batch_idx}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[Multi-Agent BEV Simple] Saved: {save_path}")
            
            # 打印统计信息
            print(f"[Multi-Agent BEV Simple] Batch {batch_idx} Statistics:")
            for agent_name in agent_names:
                occupancy_map = all_occupancy_maps[agent_names.index(agent_name)]
                occupied_cells = np.count_nonzero(occupancy_map)
                total_cells = occupancy_map.size
                occupancy_ratio = occupied_cells / total_cells
                print(f"  {agent_name}: {occupied_cells:,}/{total_cells:,} cells ({occupancy_ratio:.2%})")
    
    def visualize_multi_agent_bev_simple_independent(self, agent_bev_features, agent_names, save_dir="./multi_agent_bev_independent"):
        """
        可视化各agent独立的BEV特征（在合并之前）
        
        Args:
            agent_bev_features: 字典，包含每个agent的BEV特征
            agent_names: agent名称列表
            save_dir: 保存图片的目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"[Multi-Agent BEV Independent] Found {len(agent_names)} agents: {agent_names}")
        
        # 获取特征维度
        first_agent = list(agent_bev_features.values())[0]
        batch_size, num_channels, height, width = first_agent.shape
        
        # 为每个batch创建对比图
        for batch_idx in range(batch_size):
            fig, axes = plt.subplots(2, len(agent_names), figsize=(4*len(agent_names), 8))
            if len(agent_names) == 1:
                axes = axes.reshape(2, 1)
            
            # 收集所有agent的占用图用于统一颜色范围
            all_occupancy_maps = []
            for agent_name in agent_names:
                features = agent_bev_features[agent_name][batch_idx]  # [C, H, W]
                # 使用L2范数而不是平均值，避免零值问题
                occupancy_map = torch.norm(features, dim=0).detach().cpu().numpy()
                all_occupancy_maps.append(occupancy_map)
            
            # 计算统一的颜色范围
            all_values = np.concatenate([m.flatten() for m in all_occupancy_maps])
            vmin, vmax = np.min(all_values), np.max(all_values)
            
            for i, agent_name in enumerate(agent_names):
                occupancy_map = all_occupancy_maps[i]
                
                # 上排：占用强度热力图
                im1 = axes[0, i].imshow(occupancy_map, cmap='hot', aspect='equal', vmin=vmin, vmax=vmax)
                axes[0, i].set_title(f'{agent_name}\nIndependent BEV')
                axes[0, i].set_xlabel('X')
                axes[0, i].set_ylabel('Y')
                
                # 下排：占用统计（只显示占用率）
                occupied_cells = np.count_nonzero(occupancy_map)
                total_cells = occupancy_map.size
                occupancy_ratio = occupied_cells / total_cells
                
                stats_text = f"""Occupied: {occupied_cells:,}/{total_cells:,}
Ratio: {occupancy_ratio:.2%}"""
                
                axes[1, i].text(0.1, 0.9, stats_text, transform=axes[1, i].transAxes, 
                              fontsize=9, verticalalignment='top', fontfamily='monospace')
                axes[1, i].set_xlim(0, 1)
                axes[1, i].set_ylim(0, 1)
                axes[1, i].axis('off')
                axes[1, i].set_title('Statistics')
            
            fig.suptitle(f'Independent Agent BEV Comparison - Batch {batch_idx}', fontsize=14)
            
            # 调整布局以避免重叠 - 去掉颜色条后减少底部空间
            plt.subplots_adjust(top=0.8, bottom=0.1, hspace=0.6)
            
            # 保存对比图
            save_path = os.path.join(save_dir, f'independent_agent_bev_batch_{batch_idx}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[Multi-Agent BEV Independent] Saved: {save_path}")
            
            # 打印统计信息
            print(f"[Multi-Agent BEV Independent] Batch {batch_idx} Statistics:")
            for agent_name in agent_names:
                occupancy_map = all_occupancy_maps[agent_names.index(agent_name)]
                occupied_cells = np.count_nonzero(occupancy_map)
                total_cells = occupancy_map.size
                occupancy_ratio = occupied_cells / total_cells
                print(f"  {agent_name}: {occupied_cells:,}/{total_cells:,} cells ({occupancy_ratio:.2%})")
    
    def visualize_multi_agent_bev(self, batch_dict, save_dir="./multi_agent_bev", max_features=4):
        """
        可视化多Agent的BEV特征图对比
        
        Args:
            batch_dict: 包含多个agent数据的字典
            save_dir: 保存图片的目录
            max_features: 最多可视化的特征通道数
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 收集所有agent的BEV特征
        agent_bev_features = {}
        agent_names = []
        
        for k, v in batch_dict.items():
            if isinstance(v, dict) and 'spatial_features' in v:
                agent_bev_features[k] = v['spatial_features']
                agent_names.append(k)
        
        if len(agent_bev_features) == 0:
            print("Warning: No agent spatial_features found in batch_dict")
            return
            
        print(f"[Multi-Agent BEV] Found {len(agent_names)} agents: {agent_names}")
        
        # 获取特征维度
        first_agent = list(agent_bev_features.values())[0]
        batch_size, num_channels, height, width = first_agent.shape
        num_vis_channels = min(max_features, num_channels)
        
        # 为每个batch和每个特征通道创建对比图
        for batch_idx in range(batch_size):
            for channel_idx in range(num_vis_channels):
                fig, axes = plt.subplots(1, len(agent_names), figsize=(4*len(agent_names), 4))
                if len(agent_names) == 1:
                    axes = [axes]
                
                for i, agent_name in enumerate(agent_names):
                    features = agent_bev_features[agent_name][batch_idx, channel_idx].detach().cpu().numpy()
                    
                    im = axes[i].imshow(features, cmap='viridis', aspect='equal')
                    axes[i].set_title(f'{agent_name}\nChannel {channel_idx}')
                    axes[i].set_xlabel('X')
                    axes[i].set_ylabel('Y')
                    
                    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
                
                fig.suptitle(f'Multi-Agent BEV Comparison - Batch {batch_idx}, Channel {channel_idx}', fontsize=14)
                
                # 调整布局以避免重叠
                plt.subplots_adjust(top=0.85, bottom=0.15, hspace=0.3)
                
                # 保存对比图
                save_path = os.path.join(save_dir, f'multi_agent_bev_batch_{batch_idx}_channel_{channel_idx}.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"[Multi-Agent BEV] Saved: {save_path}")
    
    def visualize_bev_statistics(self, batch_dict, save_dir="./bev_statistics"):
        """
        可视化BEV特征的统计信息
        
        Args:
            batch_dict: 包含spatial_features的字典
            save_dir: 保存图片的目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if 'spatial_features' not in batch_dict:
            print("Warning: No spatial_features found in batch_dict")
            return
            
        spatial_features = batch_dict['spatial_features']  # [B, C, H, W]
        batch_size, num_channels, height, width = spatial_features.shape
        
        # 计算统计信息
        features_np = spatial_features.detach().cpu().numpy()
        
        # 创建统计图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 特征值分布直方图
        all_values = features_np.flatten()
        axes[0, 0].hist(all_values, bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('Feature Value Distribution')
        axes[0, 0].set_xlabel('Feature Value')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. 每个通道的平均激活强度
        channel_means = np.mean(features_np, axis=(0, 2, 3))
        axes[0, 1].plot(channel_means)
        axes[0, 1].set_title('Mean Activation per Channel')
        axes[0, 1].set_xlabel('Channel Index')
        axes[0, 1].set_ylabel('Mean Activation')
        
        # 3. 非零像素比例
        non_zero_ratios = []
        for batch_idx in range(batch_size):
            for channel_idx in range(num_channels):
                channel_data = features_np[batch_idx, channel_idx]
                non_zero_ratio = np.count_nonzero(channel_data) / channel_data.size
                non_zero_ratios.append(non_zero_ratio)
        
        axes[1, 0].hist(non_zero_ratios, bins=20, alpha=0.7, color='green')
        axes[1, 0].set_title('Non-zero Pixel Ratio Distribution')
        axes[1, 0].set_xlabel('Non-zero Ratio')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. 特征图尺寸信息
        axes[1, 1].text(0.1, 0.8, f'Batch Size: {batch_size}', fontsize=12)
        axes[1, 1].text(0.1, 0.7, f'Channels: {num_channels}', fontsize=12)
        axes[1, 1].text(0.1, 0.6, f'Height: {height}', fontsize=12)
        axes[1, 1].text(0.1, 0.5, f'Width: {width}', fontsize=12)
        axes[1, 1].text(0.1, 0.4, f'Total Elements: {features_np.size:,}', fontsize=12)
        axes[1, 1].text(0.1, 0.3, f'Memory Size: {features_np.nbytes / 1024**2:.2f} MB', fontsize=12)
        axes[1, 1].set_title('BEV Feature Statistics')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        fig.suptitle('BEV Features Statistical Analysis', fontsize=16)
        
        # 调整布局以避免重叠
        plt.subplots_adjust(top=0.85, bottom=0.15, hspace=0.3)
        
        # 保存统计图
        save_path = os.path.join(save_dir, 'bev_statistics.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[BEV Statistics] Saved: {save_path}")
        print(f"[BEV Statistics] Feature shape: {spatial_features.shape}")
        print(f"[BEV Statistics] Mean value: {np.mean(features_np):.4f}")
        print(f"[BEV Statistics] Std value: {np.std(features_np):.4f}")
        print(f"[BEV Statistics] Min value: {np.min(features_np):.4f}")
        print(f"[BEV Statistics] Max value: {np.max(features_np):.4f}")
    
    def visualize_fused_bev_features(self, batch_spatial_features, agent_names, save_dir="./fused_bev_visualization"):
        """
        可视化融合后的BEV特征 - 类似DynamicVoxelVFE的简洁风格
        
        Args:
            batch_spatial_features: 融合后的BEV特征 [batch_size, channels, height, width]
            agent_names: agent名称列表
            save_dir: 保存图片的目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"[PointPillarScatter] Visualizing fused BEV features")
        print(f"[PointPillarScatter] Fused BEV shape: {batch_spatial_features.shape}")
        
        batch_size, num_channels, height, width = batch_spatial_features.shape
        
        # 为每个batch创建可视化
        for batch_idx in range(batch_size):
            batch_features = batch_spatial_features[batch_idx]  # [C, H, W]
            
            # 使用L2范数计算占用强度（与DynamicVoxelVFE一致）
            occupancy_map = torch.norm(batch_features, dim=0).detach().cpu().numpy()
            
            # 创建简洁的可视化图 - 类似DynamicVoxelVFE的布局
            fig, axes = plt.subplots(2, 1, figsize=(4, 8))
            
            # 上排：BEV占用强度热力图
            im1 = axes[0].imshow(occupancy_map, cmap='hot', aspect='equal')
            axes[0].set_title('Fused BEV\nIndependent Scatter')
            axes[0].set_xlabel('X')
            axes[0].set_ylabel('Y')
            
            # 下排：占用统计（只显示占用率）
            occupied_cells = np.count_nonzero(occupancy_map)
            total_cells = occupancy_map.size
            occupancy_ratio = occupied_cells / total_cells
            
            stats_text = f"""Occupied: {occupied_cells:,}/{total_cells:,}
Ratio: {occupancy_ratio:.2%}"""
            
            axes[1].text(0.1, 0.9, stats_text, transform=axes[1].transAxes, 
                        fontsize=9, verticalalignment='top', fontfamily='monospace')
            axes[1].set_xlim(0, 1)
            axes[1].set_ylim(0, 1)
            axes[1].axis('off')
            axes[1].set_title('Statistics')
            
            fig.suptitle(f'Independent Agent Scatter - Batch {batch_idx}', fontsize=14)
            
            # 调整布局以避免重叠 - 去掉颜色条后减少底部空间
            plt.subplots_adjust(top=0.8, bottom=0.1, hspace=0.6)
            
            # 保存对比图
            save_path = os.path.join(save_dir, f'independent_agent_scatter_batch_{batch_idx}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[PointPillarScatter] Saved fused BEV: {save_path}")
            print(f"[PointPillarScatter] Batch {batch_idx}: {occupied_cells:,}/{total_cells:,} cells ({occupancy_ratio:.2%})")


class ImportanceGenerator(nn.Module):
    """
    鲁棒的BEV融合权重生成器
    支持动态数量的agent输入
    """
    def __init__(self, num_channels=128, max_agents=3, use_softmax=True):
        super().__init__()
        self.num_channels = num_channels
        self.max_agents = max_agents
        self.use_softmax = use_softmax
        
        # 定义agent的embedding（支持最大数量的agent）
        self.agent_emb = nn.Embedding(max_agents, num_channels)  # max_agents个agent，每个C维
        
        # 动态融合网络：根据实际输入数量调整
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(num_channels * max_agents, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, max_agents, 1)   # 输出 (B,max_agents,H,W)
        )
        print(self.fuse_conv)
        
    def forward(self, bev_features_dict, agent_names=None):
        """
        鲁棒的BEV融合权重生成
        
        Args:
            bev_features_dict: Dict of BEV features, e.g., {'vehicle': Fv, 'drone': Fd, 'rsu': Fr}
            agent_names: List of agent names (optional, inferred from dict keys)
            
        Returns:
            fused_bev: 融合后的BEV特征 (B, C, H, W)
            importance_weights: 权重图 (B, num_agents, H, W)
            active_agents: List of active agent names
        """
        if agent_names is None:
            agent_names = list(bev_features_dict.keys())
        
        num_agents = len(agent_names)
        if num_agents == 0:
            raise ValueError("No agents provided")
        if num_agents > self.max_agents:
            raise ValueError(f"Too many agents: {num_agents} > {self.max_agents}")
        
        # 创建完整的特征张量（包含所有可能的agent）
        full_features = []
        agent_indices = []
        
        # 定义agent到索引的映射
        agent_to_idx = {'vehicle': 0, 'rsu': 1, 'drone': 2}
        
        for i, agent_name in enumerate(agent_names):
            if agent_name in bev_features_dict:
                # 添加身份embedding
                agent_idx = agent_to_idx.get(agent_name, i)
                agent_emb = self.agent_emb(torch.tensor(agent_idx, device=bev_features_dict[agent_name].device))
                agent_emb = agent_emb.view(1, self.num_channels, 1, 1)
                
                feat_with_emb = bev_features_dict[agent_name] + agent_emb
                full_features.append(feat_with_emb)
                agent_indices.append(agent_idx)
            else:
                print(f"Warning: Agent {agent_name} not found in input")
        
        # 如果agent数量不足，用零填充到max_agents
        while len(full_features) < self.max_agents:
            zero_feat = torch.zeros_like(full_features[0])
            full_features.append(zero_feat)
            agent_indices.append(len(full_features) - 1)
        
        # 拼接所有特征
        F_cat = torch.cat(full_features, dim=1)  # (B, max_agents*C, H, W)
        
        # 生成权重
        W = self.fuse_conv(F_cat)  # (B, max_agents, H, W)
        
        # 创建mask，只对有效的agent计算权重
        valid_mask = torch.zeros(self.max_agents, device=W.device)
        for i in range(num_agents):
            valid_mask[i] = 1.0
        
        # 应用mask
        W_masked = W * valid_mask.view(1, -1, 1, 1)
        
        # 归一化权重
        if self.use_softmax:
            W_masked = torch.softmax(W_masked, dim=1)
        else:
            W_masked = torch.sigmoid(W_masked)
            W_masked = W_masked / (W_masked.sum(dim=1, keepdim=True) + 1e-6)
        
        # 只使用有效agent的权重进行融合
        valid_features = full_features[:num_agents]  # 只取有效的特征
        F_stack = torch.stack(valid_features, dim=1)  # (B, num_agents, C, H, W)
        W_valid = W_masked[:, :num_agents, :, :]  # (B, num_agents, H, W)
        F_fused = (W_valid.unsqueeze(2) * F_stack).sum(dim=1)  # (B, C, H, W)
        
        return F_fused, W_masked, agent_names
    
    def forward_with_list(self, bev_features_list, agent_names):
        """
        使用列表输入的forward方法（向后兼容）
        
        Args:
            bev_features_list: List of BEV features
            agent_names: List of agent names
            
        Returns:
            fused_bev: 融合后的BEV特征 (B, C, H, W)
            importance_weights: 权重图 (B, num_agents, H, W)
            active_agents: List of active agent names
        """
        # 转换为字典格式
        bev_features_dict = {}
        for i, agent_name in enumerate(agent_names):
            if i < len(bev_features_list):
                bev_features_dict[agent_name] = bev_features_list[i]
        
        return self.forward(bev_features_dict, agent_names)
    
    def visualize_importance_weights(self, importance_weights, agent_names, save_dir="./importance_visualization"):
        """
        可视化重要性权重图（鲁棒版本）
        
        Args:
            importance_weights: 权重图 (B, max_agents, H, W)
            agent_names: agent名称列表
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        weights_np = importance_weights[0].detach().cpu().numpy()  # (max_agents, H, W)
        max_agents = weights_np.shape[0]
        num_active_agents = len(agent_names)
        
        # 只显示有效的agent
        fig, axes = plt.subplots(2, num_active_agents, figsize=(4*num_active_agents, 8))
        if num_active_agents == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(num_active_agents):
            agent_name = agent_names[i] if i < len(agent_names) else f"Agent_{i}"
            
            # 上排：权重热力图
            im1 = axes[0, i].imshow(weights_np[i], cmap='viridis', aspect='equal')
            axes[0, i].set_title(f'{agent_name}\nImportance Weight')
            axes[0, i].set_xlabel('X')
            axes[0, i].set_ylabel('Y')
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
            
            # 下排：权重统计
            weight_stats = weights_np[i]
            mean_weight = np.mean(weight_stats)
            max_weight = np.max(weight_stats)
            min_weight = np.min(weight_stats)
            std_weight = np.std(weight_stats)
            
            stats_text = f"""Mean: {mean_weight:.4f}
Max: {max_weight:.4f}
Min: {min_weight:.4f}
Std: {std_weight:.4f}"""
            
            axes[1, i].text(0.1, 0.9, stats_text, transform=axes[1, i].transAxes, 
                          fontsize=9, verticalalignment='top', fontfamily='monospace')
            axes[1, i].set_xlim(0, 1)
            axes[1, i].set_ylim(0, 1)
            axes[1, i].axis('off')
            axes[1, i].set_title('Weight Statistics')
        
        fig.suptitle(f'Importance Weights for BEV Fusion ({num_active_agents} agents)', fontsize=14)
        plt.subplots_adjust(top=0.8, bottom=0.1, hspace=0.6)
        
        save_path = os.path.join(save_dir, f'importance_weights_{num_active_agents}agents.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[ImportanceGenerator] Saved importance weights: {save_path}")
        print(f"[ImportanceGenerator] Active agents: {agent_names}")
