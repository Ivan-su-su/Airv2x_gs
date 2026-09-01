
# from opencood.models.mambafusion_modules.spconv_utils import find_all_spconv_keys

import os
import torch
import torch.nn as nn
import torch.profiler
import torch.nn.functional as F
from easydict import EasyDict
import opencood.models.gaussian_modules.backbone_3d as backbone_3d
import opencood.models.gaussian_modules.backbone_2d as backbone_2d
import opencood.models.gaussian_modules.fuser as fuser
import opencood.models.gaussian_modules.refiner as refiner
import opencood.models.gaussian_modules.agent_fuser as agent_fuser
import opencood.models.gaussian_modules.map_to_bev as map_to_bev
from opencood.models.common_modules.airv2x_base_model import Airv2xBase


class Airv2xGaussian(Airv2xBase):
    def __init__(self, model_cfg):
        super().__init__(args=model_cfg)
        self.model_cfg = EasyDict(model_cfg)
        self.iter_num = self.model_cfg.get('ITER_NUM', 2)
        # Agent类型定义
        self.agent = ['vehicle', 'rsu', 'drone']
        
        # 设置模块拓扑结构
        self.module_topology = [
            'backbone_3d',       # 1. LiDAR Backbone (TPV特征和高斯生成)
            'backbone_2d',       # 2. image Backbone (TPV特征和高斯生成)
            'TPVfuser',          # 3. LiDAR和image的TPV特征融合 (agent内)
            'gaussian_refiner',  # 4. 优化高斯参数 (agent内)
            'agent_fuser',       # 5. 多agent的TPV特征融合
            'gaussian2bev',      # 6. 
        ]
        
        # 初始化检测头
        num_anchors = 2
        num_classes = 7
        # 从 GAUSSIAN2BEV 配置中读取输出特征维度
        # 如果 FUSE_MODE='concat'，实际输出是 FEATURE_DIM + HEIGHT_EMBED_DIM
        # 如果 FUSE_MODE='add'，实际输出是 FEATURE_DIM
        gaussian2bev_cfg = self.model_cfg.get('GAUSSIAN2BEV', {})
        feature_dim = gaussian2bev_cfg.get('FEATURE_DIM', 128)
        height_embed_dim = gaussian2bev_cfg.get('HEIGHT_EMBED_DIM', 128)
        fuse_mode = gaussian2bev_cfg.get('FUSE_MODE', 'concat')
        if fuse_mode == 'concat':
            C = feature_dim + height_embed_dim
        else:
            C = feature_dim
        # 如果配置中指定了 OUTPUT_FEATURE_DIM，使用它（但需要确保 refiner 输出匹配）
        output_feature_dim = gaussian2bev_cfg.get('OUTPUT_FEATURE_DIM', None)
        if output_feature_dim is not None:
            C = output_feature_dim
        C = C + 4 #feature_dim + semantic_dim
        self.cls_head = torch.nn.Conv2d(C, num_anchors * num_classes, kernel_size=1)
        self.reg_head = torch.nn.Conv2d(C, 7 * num_anchors, kernel_size=1)
        self.obj_head = torch.nn.Conv2d(C, num_anchors, kernel_size=1)
        
        # 构建网络模块，直接返回字典映射
        self.module_dict = self.build_networks()
        # 为了向后兼容，保留 module_list（从字典中提取，按 topology 顺序）
        self.module_list = [self.module_dict.get(name) for name in self.module_topology]#可去
        self.time_list = []

        # 加载预训练 backbone 权重（如配置中指定）
        self._load_pretrained_backbones(model_cfg)

        # 打印各模块参数量
        self.print_module_params()


    def print_module_params(self):
        """打印各模块的参数量"""
        print("=" * 80)
        print("AirV2X Gaussian 模型参数量分析")
        print("=" * 80)
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        print()
        
        # 按模块统计参数量
        module_params = {}
        
        for name, module in self.named_modules():
            if len(list(module.children())) == 0:  # 叶子节点
                param_count = sum(p.numel() for p in module.parameters())
                if param_count > 0:
                    # 提取模块名称（去掉具体层名）
                    module_name = name.split('.')[0] if '.' in name else name
                    if module_name not in module_params:
                        module_params[module_name] = 0
                    module_params[module_name] += param_count
        
        # 按参数量排序
        sorted_modules = sorted(module_params.items(), key=lambda x: x[1], reverse=True)
        
        print("各模块参数量 (按参数量排序):")
        print("-" * 50)
        for module_name, param_count in sorted_modules:
            percentage = (param_count / total_params) * 100
            print(f"{module_name:25s}: {param_count:10,} ({percentage:5.1f}%)")
        
        print()
        print("模块拓扑结构参数量:")
        print("-" * 50)
        for i, module_name in enumerate(self.module_topology):
            param_count = module_params.get(module_name, 0)
            percentage = (param_count / total_params) * 100
            print(f"{i+1:2d}. {module_name:20s}: {param_count:10,} ({percentage:5.1f}%)")
        
        print()
        print("检测头参数量:")
        print("-" * 50)
        cls_params = sum(p.numel() for p in self.cls_head.parameters())
        reg_params = sum(p.numel() for p in self.reg_head.parameters())
        obj_params = sum(p.numel() for p in self.obj_head.parameters())
        head_total = cls_params + reg_params + obj_params
        
        print(f"cls_head: {cls_params:10,}")
        print(f"reg_head: {reg_params:10,}")
        print(f"obj_head: {obj_params:10,}")
        print(f"检测头总计: {head_total:10,} ({(head_total/total_params)*100:5.1f}%)")
        
        print("=" * 80)


    def _load_pretrained_backbones(self, model_cfg):
        """从配置的路径加载 backbone_3d 和 backbone_2d 的预训练权重。

        在 config.yaml 的模型 args 中配置：
            PRETRAINED_CKPTS:
              BACKBONE_3D: <path_to_3d_ckpt>
              BACKBONE_2D: <path_to_2d_ckpt>
        路径可以是绝对路径，也可以是相对于项目根目录的相对路径。
        """
        ckpt_cfg = model_cfg.get('PRETRAINED_CKPTS', None)
        if ckpt_cfg is None:
            return

        # 解析路径：相对路径以项目根目录（本文件上两级）为基准
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))

        ckpt_3d = ckpt_cfg.get('BACKBONE_3D', None)
        ckpt_2d = ckpt_cfg.get('BACKBONE_2D', None)

        if ckpt_3d and self.backbone_3d is not None:
            if not os.path.isabs(ckpt_3d):
                ckpt_3d = os.path.join(project_root, ckpt_3d)
            if os.path.exists(ckpt_3d):
                print(f"[Airv2xGaussian] 加载 backbone_3d 预训练权重: {ckpt_3d}")
                self.backbone_3d.load_pretrained_weights(
                    ckpt_3d, strict=False, freeze_pretrained=False
                )
            else:
                print(f"[Airv2xGaussian] 警告: backbone_3d 预训练权重文件不存在: {ckpt_3d}")

        if ckpt_2d and self.backbone_2d is not None:
            if not os.path.isabs(ckpt_2d):
                ckpt_2d = os.path.join(project_root, ckpt_2d)
            if os.path.exists(ckpt_2d):
                print(f"[Airv2xGaussian] 加载 backbone_2d 预训练权重: {ckpt_2d}")
                self.backbone_2d.load_pretrained_weights(ckpt_2d, strict=False)
            else:
                print(f"[Airv2xGaussian] 警告: backbone_2d 预训练权重文件不存在: {ckpt_2d}")

    def build_networks(self):
        model_info_dict = {
            'module_list': [],  # 保留用于向后兼容，但不再返回
            'module_dict': {},  # 直接构建字典映射
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
            # 同时添加到字典中（如果模块不为None）
            if module is not None:
                model_info_dict['module_dict'][module_name] = module
        return model_info_dict['module_dict']
    

    def build_backbone_3d(self, model_info_dict):
        """构建3D Backbone特征提取模块"""
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict
        
        backbone_3d_module = backbone_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
        )
        model_info_dict['module_list'].append(backbone_3d_module)
        
        return backbone_3d_module, model_info_dict


    def build_backbone_2d(self, model_info_dict):
        """构建2D Backbone特征提取模块"""
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            return None, model_info_dict
        
        backbone_2d_module = backbone_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D,
        )
        model_info_dict['module_list'].append(backbone_2d_module)
        return backbone_2d_module, model_info_dict


    def build_TPVfuser(self, model_info_dict):
        """构建TPVfuser模块"""
        if self.model_cfg.get('TPVFUSER', None) is None:
            return None, model_info_dict
        
        TPVfuser_module = fuser.__all__[self.model_cfg.TPVFUSER.NAME](
            model_cfg=self.model_cfg.TPVFUSER
        )
        model_info_dict['module_list'].append(TPVfuser_module)
        return TPVfuser_module, model_info_dict


    def build_gaussian_refiner(self, model_info_dict):
        """构建gaussian refiner模块"""
        if self.model_cfg.get('REFINER', None) is None:
            return None, model_info_dict
        
        gaussian_refiner_module = refiner.__all__[self.model_cfg.REFINER.NAME](
            model_cfg=self.model_cfg.REFINER
        )
        model_info_dict['module_list'].append(gaussian_refiner_module)
        return gaussian_refiner_module, model_info_dict


    def build_agent_fuser(self, model_info_dict):
        """构建agent fuser模块"""
        if self.model_cfg.get('AGENT_FUSER', None) is None:
            return None, model_info_dict

        agent_fuser_module = agent_fuser.__all__[self.model_cfg.AGENT_FUSER.NAME](
            model_cfg=self.model_cfg.AGENT_FUSER
        )
        model_info_dict['module_list'].append(agent_fuser_module)
        return agent_fuser_module, model_info_dict


    def build_gaussian2bev(self, model_info_dict):
        """构建Gaussian To BEV模块"""
        if self.model_cfg.get('GAUSSIAN2BEV', None) is None:
            return None, model_info_dict

        gaussian2bev_module = map_to_bev.__all__[self.model_cfg.GAUSSIAN2BEV.NAME](
            model_cfg=self.model_cfg.GAUSSIAN2BEV,
        )
        model_info_dict['module_list'].append(gaussian2bev_module)
        return gaussian2bev_module, model_info_dict

    '''
    def _load_state_dict(self, model_state_disk, *, strict=True):
        """加载模型状态字典"""
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()
            # adapt pretrain image backbone to mm backbone
            if 'image_backbone' in key:
                key = key.replace("image","mm")
                if 'input_layer' in key:
                    key = key.replace("input_layer","image_input_layer")

            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
            else:
                print("not exist",key)

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state
    '''
    def forward(self, batch_dict):
        """前向传播"""
        # 检查可用的agents
        available_agents = []
        for agent in self.agent:
            # dwb
            # 最开始batch_dict的结构是：
            # {
            #     'vehicle': {
            #         'origin_lidar': tensor,
            #         'batch_merged_cam_inputs': {
            #             'imgs': tensor, [B, N, C, H, W]
            #             'intrinsics': tensor, [B, N, 3, 3]
            #             'extrinsics': tensor, [B, N, 4, 4]
            #         }
            #     },
            #     'rsu': {
            #         'origin_lidar': tensor,
            #     },
            #     'drone': {
            #         'origin_lidar': tensor,
            #     },
            # }
            if agent == 'vehicle' and 'origin_lidar' in batch_dict:
                # 检查vehicle的origin_lidar是否有效
                origin_lidar = batch_dict['origin_lidar']
                if origin_lidar is not None and origin_lidar.numel() > 0 and torch.count_nonzero(origin_lidar).item() > 0:
                    available_agents.append(agent)
            elif agent != 'vehicle' and f'origin_lidar_{agent}' in batch_dict:
                # 检查RSU/Drone的origin_lidar是否有效
                origin_lidar = batch_dict[f'origin_lidar_{agent}']
                if origin_lidar is not None and origin_lidar.numel() > 0 and torch.count_nonzero(origin_lidar).item() > 0:
                    available_agents.append(agent)
        
        print("available_agents:", available_agents)
        import time
        start_time = time.time()
        # 按照模块拓扑结构执行前向传播
        for model_name,module in self.module_dict.items():
            if model_name == 'gaussian_refiner':
                for i in range(self.iter_num):
                    batch_dict = module(batch_dict, available_agents)
            elif model_name == 'gaussian2bev':
                for i in range(self.iter_num):
                    batch_dict = self.module_dict['gaussian_refiner'](batch_dict, available_agents,fused_iter=True)
                batch_dict = module(batch_dict, available_agents)
            else:
                batch_dict = module(batch_dict,available_agents)
        end_time = time.time()
        print(f"forward Time taken: {end_time - start_time} seconds")
        # for cur_module, model_name in zip(self.module_list, self.module_topology):
        #     # import pdb; pdb.set_trace()
        #     if model_name in ['gaussian_refiner','gaussian2bev']:
        #         batch_dict = cur_module(batch_dict, available_agents)
        #     else:
        #         batch_dict = cur_module(batch_dict)

        # 输出BEV特征格式
        spatial_features = batch_dict['spatial_features_2d']  # [B, C, H, W]
        
        if spatial_features is not None: #TODO
            B, C, H, W = spatial_features.shape
            target_h, target_w = 100, 352
            if (H, W) != (target_h, target_w):
                spatial_features = F.interpolate(
                    spatial_features, size=(target_h, target_w), mode='bilinear', align_corners=False
                )
                H, W = target_h, target_w
            
            # 通过头部网络得到最终输出
            psm = self.cls_head(spatial_features)  # [B, A*C, H, W] = [B, 2*7, 180, 180]
            rm = self.reg_head(spatial_features)   # [B, A*7, H, W] = [B, 2*7, 180, 180]
            obj = self.obj_head(spatial_features)  # [B, A, H, W] = [B, 2, 180, 180]
            
            print(f"[Airv2xGaussian] 输出尺寸: psm={psm.shape}, rm={rm.shape}, obj={obj.shape}")
            
        else:
            # 如果没有特征，创建空的特征
            default_H, default_W = 100, 352
            psm = torch.zeros(1, 14, default_H, default_W, device=next(self.parameters()).device)  # 2*7=14
            rm = torch.zeros(1, 14, default_H, default_W, device=next(self.parameters()).device)   # 2*7=14
            obj = torch.zeros(1, 2, default_H, default_W, device=next(self.parameters()).device)   # 2
        
        # 创建AirV2X兼容的输出格式
        output_dict = {
            'psm': psm,                  # [1, A*C, H, W] - 分类特征
            'rm': rm,                    # [1, A*7, H, W] - 回归特征  
            'obj': obj,                  # [1, A, H, W] - 目标特征
            'mask': 0,                   # 占位符
            'com': None,                 # 通信相关
            'comm_rate': None            # 通信率
        }
            
        return output_dict




