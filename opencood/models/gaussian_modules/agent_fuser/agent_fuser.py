'''
Fuse TPV features from multiple agents (vehicle, RSU, drone) using fusion strategies.
'''

from torch import nn
import torch
import torch.nn.functional as F
from collections import OrderedDict

# Default configuration template
DEFAULT_MODEL_CFG = {
    'IN_CHANNEL': 128,  # Input channel for each TPV feature
    'OUT_CHANNEL': 128,  # Output channel after fusion
    'AGENT_TYPES': ['vehicle', 'rsu', 'drone'],  # List of agent types to fuse
    'FUSION_METHOD': 'default',  # Fusion method (to be implemented)
    'TPV_PLANES': ['xy', 'xz', 'yz'],  # TPV plane types
    'AGENT_EMBED_DIM': 32,  # Dimension of agent embeddings
    'USE_AGENT_EMBED': True,  # Whether to use agent embeddings
    'AGENT_EMBED_TYPE': 'camera',  # 'index' | 'camera' | 'both'
    'CAMERA_EMBED_HIDDEN': 64,  # hidden dim for camera MLP
    'CONFIDENCE_MAP': {  # Configuration for confidence map generation
        'HIDDEN_DIM': 64,  # Hidden dimension in the confidence CNN
    },
}


class AgentFuser(nn.Module):
    """
    Fuse TPV features from multiple agents (vehicle, RSU, drone).
    For example, xy features from agent 1 are fused with xy features from agent 2 and agent 3.
    """
    
    def __init__(self, model_cfg=None):
        super(AgentFuser, self).__init__()
        
        self.model_cfg = model_cfg
        
        # If model_cfg is None, use default configuration
        if model_cfg is None:
            import copy
            model_cfg = copy.deepcopy(DEFAULT_MODEL_CFG)
        
        # Read necessary configs
        self.in_channel = self.model_cfg.get('IN_CHANNEL', 128)
        self.out_channel = self.model_cfg.get('OUT_CHANNEL', 128)
        self.agent_types = self.model_cfg.get('AGENT_TYPES', ['vehicle', 'rsu', 'drone'])
        self.fusion_method = self.model_cfg.get('FUSION_METHOD', 'default')
        self.tpv_planes = self.model_cfg.get('TPV_PLANES', ['xy', 'xz', 'yz'])
        
        # Agent embedding configuration
        self.agent_embed_dim = self.model_cfg.get('AGENT_EMBED_DIM', 32)
        self.use_agent_embed = self.model_cfg.get('USE_AGENT_EMBED', True)
        self.agent_embed_type = self.model_cfg.get('AGENT_EMBED_TYPE', 'camera')
        
        # Step 2: Initialize agent embeddings
        # Create a mapping from agent types to indices
        self.agent_to_idx = {agent: idx for idx, agent in enumerate(self.agent_types)}
        
        # Create learnable embeddings for each agent type
        if self.use_agent_embed:
            # Index-based embedding
            self.agent_embeddings = nn.Embedding(
                num_embeddings=len(self.agent_types),
                embedding_dim=self.agent_embed_dim
            )
            
            # Camera-based embedding MLP (intrinsics/extrinsics)
            # Input dim: intrinsics (3x3=9) + extrinsics (4x4=16) = 25
            camera_in_dim = 25
            camera_hidden = self.model_cfg.get('CAMERA_EMBED_HIDDEN', 64)
            self.camera_embed_mlp = nn.Sequential(
                nn.Linear(camera_in_dim, camera_hidden),
                nn.ReLU(True),
                nn.Linear(camera_hidden, self.agent_embed_dim)
            )
        
        # Step 2 (continued): Initialize confidence map generation CNN
        # This CNN takes concatenated [features, agent_embed] as input and outputs confidence map
        confidence_cfg = self.model_cfg.get('CONFIDENCE_MAP', {})
        confidence_hidden_dim = confidence_cfg.get('HIDDEN_DIM', 64)
        
        self.confidence_nets = nn.ModuleDict()
        for plane in self.tpv_planes:
            # Input: concatenated features + agent embedding = in_channel + agent_embed_dim
            input_dim = self.in_channel + self.agent_embed_dim if self.use_agent_embed else self.in_channel
            
            self.confidence_nets[plane] = nn.Sequential(
                nn.Conv2d(input_dim, confidence_hidden_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(confidence_hidden_dim),
                nn.ReLU(True),
                nn.Conv2d(confidence_hidden_dim, confidence_hidden_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(confidence_hidden_dim),
                nn.ReLU(True),
                nn.Conv2d(confidence_hidden_dim, 1, kernel_size=1, bias=False),  # Output: 1 channel confidence map
                nn.Sigmoid()  # Normalize to [0, 1] for confidence scores
            )
        
        # Initialize fusion modules (to be completed later)
        # Placeholder for fusion layers
        self.fusion_modules = nn.ModuleDict()
        
        # Initialize per-plane fusion modules (to be completed later)
        # For example, separate fusion for xy, xz, yz planes
        # self.plane_fusion_modules = nn.ModuleDict()
        
        # Initialize per-agent fusion weights (to be completed later)
        # self.agent_fusion_weights = nn.ModuleDict()
        
    def _get_agent_embedding(self, agent_type, device, agent_dict=None):
        """
        Get agent embedding for a specific agent type.
        
        Args:
            agent_type: str - Agent type ('vehicle', 'rsu', 'drone')
            device: torch.device - Device to place the embedding on
            agent_dict: dict - batch_dict[agent_type], used to extract camera params when needed
        
        Returns:
            agent_embed: [B, agent_embed_dim] - Agent embedding matrix per batch element
        """
        if not self.use_agent_embed:
            return None

        # Helper: index-based embedding -> [1, E]
        def _index_embed():
            idx = self.agent_to_idx[agent_type]
            idx_t = torch.tensor([idx], device=device)
            return self.agent_embeddings(idx_t)

        # Helper: camera-based embedding -> [B, E]
        def _camera_embed():
            if agent_dict is None:
                return None
            cam = agent_dict.get('batch_merged_cam_inputs', None)
            if cam is None:
                return None
            intrinsics = cam.get('intrinsics', None)  # [B, N, 3, 3]
            extrinsics = cam.get('extrinsics', None)  # [B, N, 4, 4]
            intrinsics = intrinsics.view(1,-1,3,3)
            extrinsics = extrinsics.view(1,-1,4,4)
            if intrinsics is None or extrinsics is None:
                return None
            B = intrinsics.shape[0]
            # Average over cameras N, then flatten
            intri_avg = intrinsics.mean(dim=1).reshape(B, -1)  # [B, 9]
            extra_avg = extrinsics.mean(dim=1).reshape(B, -1)  # [B, 16]
            cam_vec = torch.cat([intri_avg, extra_avg], dim=-1)  # [B, 25]
            cam_vec = cam_vec.to(device)
            return self.camera_embed_mlp(cam_vec)  # [B, E]

        if self.agent_embed_type == 'index':
            emb = _index_embed()  # [1, E]
            return emb  # will be expanded by caller
        elif self.agent_embed_type == 'camera':
            emb = _camera_embed()  # [B, E] or None
            if emb is None:
                emb = _index_embed().repeat(agent_dict['fused_tpv_xy'].shape[0], 1) if agent_dict and 'fused_tpv_xy' in agent_dict else _index_embed()
            return emb
        elif self.agent_embed_type == 'both':
            cam_emb = _camera_embed()
            idx_emb = _index_embed()  # [1, E]
            if cam_emb is not None:
                # Sum or concat; here we sum to keep dim E
                # If B>1, expand idx_emb to [B,E]
                if cam_emb.dim() == 2:
                    idx_emb = idx_emb.expand(cam_emb.shape[0], -1)
                return cam_emb + idx_emb
            else:
                return idx_emb
        else:
            # Fallback to index
            return _index_embed()
    
    def _generate_confidence_map(self, features, agent_type, plane_name, agent_dict=None):
        """
        Generate confidence map for features based on agent embedding.
        
        Args:
            features: [B, C, H, W] - TPV features from a specific agent
            agent_type: str - Agent type ('vehicle', 'rsu', 'drone')
            plane_name: str - TPV plane name ('xy', 'xz', 'yz')
        
        Returns:
            confidence_map: [B, 1, H, W] - Confidence scores in [0, 1] range
        """
        B, C, H, W = features.shape
        # Get agent embedding
        agent_embed = self._get_agent_embedding(agent_type, features.device, agent_dict=agent_dict)  # [B, E] or [1, E] or None
        
        if agent_embed is not None:
            # Expand agent embedding to match spatial dimensions
            if agent_embed.dim() == 1:
                agent_embed = agent_embed.unsqueeze(0)
            if agent_embed.shape[0] == 1 and B > 1:
                agent_embed = agent_embed.expand(B, -1)
            agent_embed_expanded = agent_embed.unsqueeze(-1).unsqueeze(-1)  # [B, E, 1, 1]
            agent_embed_expanded = agent_embed_expanded.expand(B, -1, H, W)  # [B, E, H, W]
            
            # Concatenate features with agent embedding along channel dimension
            features_with_embed = torch.cat([features, agent_embed_expanded], dim=1)  # [B, C+E, H, W]
        else:
            features_with_embed = features
        
        # Generate confidence map using CNN
        confidence_map = self.confidence_nets[plane_name](features_with_embed)  # [B, 1, H, W]
        
        return confidence_map
    
    def forward(self, batch_dict, available_agents):
        """
        Forward pass to fuse TPV features from multiple agents.
        
        Args:
            batch_dict: Dictionary containing TPV features from different agents
                Expected structure:
                {
                    'vehicle': {
                        'tpv_xy': [B, C, H, W],
                        'tpv_xz': [B, C, H, W],
                        'tpv_yz': [B, C, H, W],
                    },
                    'rsu': {...},
                    'drone': {...}
                }
        
        Returns:
            batch_dict: Updated with fused TPV features
                Structure:
                {
                    'fused_agents_tpv_xy': [B, C, H, W],
                    'fused_agents_tpv_xz': [B, C, H, W],
                    'fused_agents_tpv_yz': [B, C, H, W],
                }
        """
        # For each TPV plane, compute confidence maps per agent, weight features, and sum across agents
        fused_results = {}
        
        for plane in self.tpv_planes:
            plane_key = f'fused_tpv_{plane}'
            fused_sum = None
            
            for agent_type in available_agents:
                if agent_type not in batch_dict:
                    continue
                agent_dict = batch_dict[agent_type]
                if plane_key not in agent_dict:
                    continue
                
                features = agent_dict[plane_key]  # [B, C, H, W]
                
                confidence_map = self._generate_confidence_map(features, agent_type, plane, agent_dict=agent_dict)  # [B, 1, H, W]
                weighted_features = features * confidence_map
                
                if fused_sum is None:
                    fused_sum = weighted_features
                else:
                    fused_sum = fused_sum + weighted_features
                # Pop各agent的融合TPV特征，保留全局融合的fused_agents_tpv_xy/xz/yz
                batch_dict[agent_type].pop(plane_key, None)
            if fused_sum is not None:
                fused_results[plane] = fused_sum
        
        # Write fused results back to batch_dict under shared keys
        if 'xy' in fused_results:
            batch_dict['fused_agents_tpv_xy'] = fused_results['xy']
        if 'xz' in fused_results:
            batch_dict['fused_agents_tpv_xz'] = fused_results['xz']
        if 'yz' in fused_results:
            batch_dict['fused_agents_tpv_yz'] = fused_results['yz']

        # 聚合所有的高斯
        all_mu = []
        all_scale = []
        all_rotation = []
        all_features = []
        all_semantic = []
        for agent_type in available_agents:
            agent_dict = batch_dict[agent_type]
            if 'merged_gaussians' not in agent_dict:
                continue
            merged_gaussians = agent_dict['merged_gaussians']
            if 'mu' in merged_gaussians:
                mu = merged_gaussians['mu']
                all_mu.append(mu)
                scale = merged_gaussians['scale']
                all_scale.append(scale)
                rotation = merged_gaussians['rotation']
                all_rotation.append(rotation)
                features = merged_gaussians['features']
                all_features.append(features)
                semantic = merged_gaussians['semantic']
                all_semantic.append(semantic)
            # Pop各agent的融合TPV特征，保留全局融合的fused_agents_tpv_xy/xz/yz
            batch_dict[agent_type].pop('merged_gaussians', None)
        if len(all_mu) > 0:
            all_mu = torch.cat(all_mu, dim=0)
            all_scale = torch.cat(all_scale, dim=0)
            all_rotation = torch.cat(all_rotation, dim=0)
            all_features = torch.cat(all_features, dim=0)
            all_semantic = torch.cat(all_semantic, dim=0)
       
        batch_dict['fused_agents_gaussians'] = {
            'mu': all_mu,
            'scale': all_scale,
            'rotation': all_rotation,
            'features': all_features,
            'semantic': all_semantic
        }

        # 此时batch_dict的结构是：
        # {
        #     'vehicle': {
        #         'origin_lidar': tensor, [num_points, 4]
        #         'pillar_features': tensor, [num_voxel, 128]
        #         'voxel_features': tensor, [num_voxel, 128]
        #         'voxel_coords': tensor, [num_voxel, 4]
        #         'lidar_gaussians': {
        #             'mu': tensor, [num_lidar_gaussians, 3]
        #             'scale': tensor, [num_lidar_gaussians, 3]
        #             'rotation': tensor, [num_lidar_gaussians, 4]
        #             'features': tensor, [num_lidar_gaussians, 128]
        #         }
        #         'tpv_xy': tensor, [1, 128, 200, 704]
        #         'tpv_xz': tensor, [1, 128, 704, 32]
        #         'tpv_yz': tensor, [1, 128, 200, 32]
        #         TODO：下面这个也可以不要
        #         'image_tpv_features': dict
        #         'image_tpv_xy': tensor, [1, 128, 200, 704]
        #         'image_tpv_xz': tensor, [1, 128, 704, 32]
        #         'image_tpv_yz': tensor, [1, 128, 200, 32]
        #         'image_gaussians': {
        #             'mu': tensor, [num_image_gaussians, 3]
        #             'scale': tensor, [num_image_gaussians, 3]
        #             'rotation': tensor, [num_image_gaussians, 4]
        #             'features': tensor, [num_image_gaussians, 128]
        #         }
        #         'fused_tpv_xy': tensor, [B, 128, 200, 704]
        #         'fused_tpv_xz': tensor, [B, 128, 704, 32]
        #         'fused_tpv_yz': tensor, [B, 128, 200, 32]
        #         'updated_gaussians': {
        #             'mu': tensor, [num_gaussians, 3]
        #             'scale': tensor, [num_gaussians, 3]
        #             'rotation': tensor, [num_gaussians, 4]
        #             'features': tensor, [num_gaussians, 256]
        #             TODO: 以下三个可以删除
        #             'ref_xy': tensor, [num_gaussians, 2]
        #             'ref_xz': tensor, [num_gaussians, 2]
        #             'ref_yz': tensor, [num_gaussians, 2]
        #         }
        #     },
        #     ..., (rsu and drone)
        #     'fused_agents_tpv_xy': tensor, [B, 128, 200, 704]
        #     'fused_agents_tpv_xz': tensor, [B, 128, 704, 32]
        #     'fused_agents_tpv_yz': tensor, [B, 128, 200, 32]
        #     'fused_agents_gaussians': {
        #         'mu': tensor, [num_gaussians, 3]
        #         'scale': tensor, [num_gaussians, 3]
        #         'rotation': tensor, [num_gaussians, 4]
        #         'features': tensor, [num_gaussians, 256]
        #     }
        # }
        
        return batch_dict


# ====================================================
# 文件整体架构说明（AgentFuser）
# ====================================================
"""
本文件定义了多 Agent 的 TPV 融合模块 AgentFuser，核心目标：
- 针对不同 Agent（vehicle / rsu / drone）已融合的 TPV 特征（来自图像与 LiDAR 的融合输出）
- 学习每个 Agent 对应的可学习嵌入（embedding），并基于 [特征, 嵌入] 联合输入的小型 CNN 生成置信度图（confidence map）
- 按平面（xy/xz/yz）对每个 Agent 的特征进行逐像素加权（特征 × 置信度），再在 Agent 维度上求和得到最终 fused TPV


【Config 参数说明】
DEFAULT_MODEL_CFG 关键项：
- IN_CHANNEL: 128                 # 每个平面 TPV 的输入通道数
- OUT_CHANNEL: 128                # 预留的输出通道配置（当前未强制使用）
- AGENT_TYPES: ['vehicle', 'rsu', 'drone']  # 参与融合的 Agent 列表
- TPV_PLANES: ['xy', 'xz', 'yz']  # 三个 TPV 平面
- AGENT_EMBED_DIM: 32             # Agent 嵌入维度
- USE_AGENT_EMBED: True           # 是否启用 Agent 嵌入
- CONFIDENCE_MAP: {
    'HIDDEN_DIM': 64              # 置信图 CNN 的隐藏通道
  }


【输入 / 输出接口】
输入 batch_dict 需要包含每个 Agent 的平面特征（由上游 ConvFuserTPV* 产生）：
batch_dict[agent]['fused_tpv_xy']  # [B, C, H_xy, W_xy]
batch_dict[agent]['fused_tpv_xz']  # [B, C, H_xz, W_xz]
batch_dict[agent]['fused_tpv_yz']  # [B, C, H_yz, W_yz]

AgentFuser.forward 输出：
- batch_dict['fused_agents_tpv_xy']  # [B, C, H_xy, W_xy]
- batch_dict['fused_agents_tpv_xz']  # [B, C, H_xz, W_xz]
- batch_dict['fused_agents_tpv_yz']  # [B, C, H_yz, W_yz]


【前向流程】
1) 对每个平面 plane ∈ {xy, xz, yz}：
   - 遍历所有可用 Agent：从 batch_dict[agent] 读取对应 plane 的特征 features。
   - 基于 Agent 嵌入（可选）与 features 拼接，通过 confidence_nets[plane] 生成置信图 confidence_map ∈ [0, 1]。
   - 计算 weighted_features = features × confidence_map。
   - 将所有 Agent 的 weighted_features 在通道一致情况下进行逐像素求和，得到 fused_sum。
2) 将各平面的 fused_sum 写回 batch_dict 的统一键：
   - 'fused_agents_tpv_xy'、'fused_agents_tpv_xz'、'fused_agents_tpv_yz'。


【形状说明】
- features:      [B, C, H, W]
- agent_embed:   [1, E] → 广播为 [B, E, H, W]
- concat 输入:   [B, C+E, H, W]
- confidence:    [B, 1, H, W]（Sigmoid 后范围 [0, 1]）
- 加权特征:     [B, C, H, W]
- fused 输出:    [B, C, H, W]


【核心模块】
- nn.Embedding（每 Agent 一个嵌入向量），用于编码 Agent 类型差异。
- confidence_nets（按平面各一套 CNN）
  输入: concat([features, agent_embed_broadcast])
  结构: 3x3 Conv → BN → ReLU → 3x3 Conv → BN → ReLU → 1x1 Conv → Sigmoid
  输出: confidence_map ∈ [B, 1, H, W]


【典型使用示例】
# 初始化
model_cfg = {
    'IN_CHANNEL': 128,
    'AGENT_TYPES': ['vehicle', 'rsu', 'drone'],
    'TPV_PLANES': ['xy', 'xz', 'yz'],
    'AGENT_EMBED_DIM': 32,
    'USE_AGENT_EMBED': True,
    'CONFIDENCE_MAP': {
        'HIDDEN_DIM': 64
    }
}
agent_fuser = AgentFuser(model_cfg)

# 前置：确保 batch_dict 中存在各 Agent 的 fused_tpv_* 特征
# 例如：batch_dict['vehicle']['fused_tpv_xy'] = ...  [B, 128, 200, 704]
#       batch_dict['rsu']['fused_tpv_xy'] = ...      [B, 128, 200, 704]
#       batch_dict['drone']['fused_tpv_xy'] = ...    [B, 128, 200, 704]

# 调用前向
batch_dict = agent_fuser(batch_dict)

# 读取多 Agent 融合后的结果
tpv_xy = batch_dict['fused_agents_tpv_xy']  # [B, 128, 200, 704]
tpv_xz = batch_dict['fused_agents_tpv_xz']  # [B, 128, 704, 32]
tpv_yz = batch_dict['fused_agents_tpv_yz']  # [B, 128, 200, 32]


【注意事项】
- 若某个 Agent 在 batch_dict 中缺失或对应 plane 特征不存在，将自动跳过该 Agent。
- 通道数 C 需与 IN_CHANNEL 对齐；所有 Agent 的同一平面尺寸需一致以便直接逐像素求和。
- 当前实现为“求和融合”；如需归一化（例如按各 Agent 置信度总和归一）或更复杂的融合方式，可在求和处扩展。
"""