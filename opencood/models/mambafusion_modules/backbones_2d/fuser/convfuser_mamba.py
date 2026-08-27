import torch
from torch import nn
from typing import Any, Dict, Optional, Tuple, Union

from ...vmamba.vmamba import SS2D, VSSBlock, Linear2d, LayerNorm2d
from mamba_ssm.models.mixer_seq_simple import create_block
from collections import OrderedDict
from ..base_bev_backbone import BasicBlock
from ...model_utils.voxel_mamba_utils import get_hilbert_index_2d_mamba_lite
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
class ConvFuser(nn.Module):
    """
    【AirV2X多agent ConvFuser模块】
    
    功能：融合多agent的图像BEV特征和激光雷达BEV特征
    
    架构对比：
    - MambaFusion: 单agent多视角 -> 统一BEV -> 双模态融合
    - AirV2X: 多agent独立 -> 多BEV -> 多agent融合 -> 双模态融合
    
    输入：
    - 多agent情况: batch_dict[agent]['spatial_features_img'] 每个 [B_i, 80, H, W]
    - 单agent情况: batch_dict['spatial_features_img'] [B, 80, H, W]
    - 激光雷达: batch_dict['spatial_features'] [B, 128, H, W]
    
    输出：
    - batch_dict['spatial_features'] [B, 128, H, W] 与MambaFusion对齐
    
    融合策略：
    - mean: 平均融合，保留所有agent信息
    - max: 最大融合，突出最强特征
    - concat: 通道拼接，保留所有原始信息
    """
    def __init__(self,model_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        in_channel = self.model_cfg.IN_CHANNEL
        out_channel = self.model_cfg.OUT_CHANNEL
        self.image_channel = self.model_cfg.IMAGE_CHANNEL
        self.lidar_channel = self.model_cfg.LIDAR_CHANNEL
        self.merge_type = self.model_cfg.get('MERGE_TYPE', 'default')
        self.importance_generator = ImportanceGenerator(num_channels=out_channel, max_agents=3, use_softmax=True)
        # 使用空间相关的 BatchCompressorV2，对每个像素位置在 agent 维上做可学习加权融合
        self.batch_compressor = BatchCompressorV2(in_channels=self.image_channel,
                                                  mid_channels=self.image_channel*2,
                                                  out_channels=self.image_channel)
        # 支持只用雷达数据的情况
        self.lidar_only = self.model_cfg.get('LIDAR_ONLY', False)

        # 根据是否只用雷达数据调整输入通道数
        if self.lidar_only:
            # 只用雷达数据时，输入通道数只有雷达特征维度
            lidar_channel = 64  # 雷达特征维度
            actual_in_channel = lidar_channel
        else:
            # 使用图像+雷达数据时，输入通道数是两者之和
            actual_in_channel = in_channel
            
        if self.merge_type == 'default':
            self.conv = nn.Sequential(
                nn.Conv2d(actual_in_channel, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True)
                )
        else:
            self.conv = nn.Sequential(
                # DepthwiseSeparableConv(actual_in_channel, actual_in_channel, 3, 1, 1),
                nn.Conv2d(actual_in_channel, out_channel * 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel * 2),
                nn.ReLU(),
                nn.Conv2d(out_channel * 2, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
                )
        self.use_vmamba = model_cfg.get('USE_VMAMBA', False)
        self.use_checkpoint = model_cfg.get('USE_CHECKPOINT', True)
        self.use_merge_after = model_cfg.get('USE_MERGE_AFTER', False)
        self.agent_fusion_strategy = model_cfg.get('AGENT_FUSION_STRATEGY', 'mean')  # 'mean', 'concat', 'max'
        self.use_offset_guided_hierarchical_fusion = model_cfg.get('USE_OFFSET_GUIDED_HIERARCHICAL_FUSION', False)
        self.offset_guided_return_debug = model_cfg.get('OFFSET_GUIDED_HIERARCHICAL_RETURN_DEBUG', False)
        if self.use_offset_guided_hierarchical_fusion:
            fusion_cfg = model_cfg.get('OFFSET_GUIDED_HIERARCHICAL_FUSION', {})
            self.offset_guided_hierarchical_fusion = OffsetGuidedSelectiveHierarchicalMambaFusionBlock(
                channels=out_channel,
                num_points=fusion_cfg.get('NUM_POINTS', 4),
                offset_range=fusion_cfg.get('OFFSET_RANGE', 2.0),
                transmission_alpha=fusion_cfg.get('TRANSMISSION_ALPHA', 1.0),
                window_size=fusion_cfg.get('WINDOW_SIZE', 4),
                align_corners=fusion_cfg.get('ALIGN_CORNERS', False),
                padding_mode=fusion_cfg.get('PADDING_MODE', 'zeros'),
                local_mamba_depth=fusion_cfg.get('LOCAL_MAMBA_DEPTH', 1),
                global_fusion_type=fusion_cfg.get('GLOBAL_FUSION_TYPE', 'mamba'),
                global_mamba_depth=fusion_cfg.get('GLOBAL_MAMBA_DEPTH', 1),
                global_conv_depth=fusion_cfg.get('GLOBAL_CONV_DEPTH', 2),
                ssm_d_state=fusion_cfg.get('SSM_D_STATE', 1),
                ssm_ratio=fusion_cfg.get('SSM_RATIO', 1.0),
                ssm_dt_rank=fusion_cfg.get('SSM_DT_RANK', 'auto'),
                ssm_conv=fusion_cfg.get('SSM_CONV', 3),
                ssm_conv_bias=fusion_cfg.get('SSM_CONV_BIAS', False),
                mlp_ratio=fusion_cfg.get('MLP_RATIO', 4.0),
                mlp_drop_rate=fusion_cfg.get('MLP_DROP_RATE', 0.0),
                forward_type=fusion_cfg.get('FORWARD_TYPE', 'v05_noz'),
                sample_attention_type=fusion_cfg.get('SAMPLE_ATTENTION_TYPE', 'aggregator'),
                sample_cross_attention_dim=fusion_cfg.get('SAMPLE_CROSS_ATTENTION_DIM'),
                sample_cross_attention_ffn_ratio=fusion_cfg.get(
                    'SAMPLE_CROSS_ATTENTION_FFN_RATIO',
                    2.0,
                ),
                sample_cross_attention_drop_rate=fusion_cfg.get(
                    'SAMPLE_CROSS_ATTENTION_DROP_RATE',
                    0.0,
                ),
                sample_cross_attention_layer_scale_init=fusion_cfg.get(
                    'SAMPLE_CROSS_ATTENTION_LAYER_SCALE_INIT',
                    0.0,
                ),
            )
        if self.use_merge_after:
            depths = [1]
            num_block = len(depths)
            merge_dim = 144  # 80 + 60 = 140
            self.merge_blocks = nn.ModuleList()
            # self.merge_norm = nn.ModuleList()
            dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]
            for i_layer in range(num_block):
                self.merge_blocks.append(self._make_vmamba_layer(
                    dim=merge_dim,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    use_checkpoint=False,
                    norm_layer=LayerNorm2d,
                    downsample=nn.Identity(),
                    channel_first=True,
                    # =================
                    ssm_d_state=1,
                    ssm_ratio=1.0,
                    ssm_dt_rank='auto',
                    ssm_act_layer=nn.SiLU,
                    ssm_conv=3,
                    ssm_conv_bias=False,
                    ssm_drop_rate=0.0,
                    ssm_init='v0',
                    forward_type='v05_noz',
                    # =================
                    mlp_ratio=4.0,
                    mlp_act_layer=nn.GELU,
                    mlp_drop_rate=0.0,
                    gmlp=False,
                ))
        if self.use_vmamba:
            # self.img_pos_embed_layer = PositionEmbeddingLearned(20, 128)
            # self.lidar_pos_embed_layer = PositionEmbeddingLearned(3, 128)
            self.use_dw_conv = True
            depths = [1, 1, 1] # [1, 2, 2]
            num_block = len(depths)
            image_dim = 80
            point_dim = 64 # 从128改为60
            cross_dim = 128
            ssm_conv = 3
            max_channel = 1
            use_4x = False
            self.use_cross = False
            self.use_res_merge = False
            d_state = 1
            self.image_down_blocks = nn.ModuleList()
            self.image_de_blocks = nn.ModuleList()
            self.lidar_de_blocks = nn.ModuleList()
            self.lidar_down_blocks = nn.ModuleList()

            
            if self.use_res_merge:
                self.image_norm = nn.ModuleList()
                self.point_norm = nn.ModuleList()


            self.image_vmamba_blocks = nn.ModuleList()
            self.point_vmamba_blocks = nn.ModuleList()
            num_block_cross = 0

            if self.use_cross:
                depths_cross = [1, 1, 1]
                self.use_res_merge = False
                if not self.use_res_merge:
                    self.image_cross_blocks = nn.ModuleList()
                    self.point_cross_blocks = nn.ModuleList()
                
                self.image_up_blocks = nn.ModuleList()
                
                num_block_cross = len(depths_cross)
                dpr_cross = []
                for x in torch.linspace(0, 0.1, sum(depths_cross)):
                    dpr_cross.extend([x.item(), x.item()])
                self.cross_vmamba_blocks = nn.ModuleList()
                for i_layer in range(num_block_cross):
                    self.image_up_blocks.append(
                        nn.Sequential(
                            nn.Conv2d(image_dim, cross_dim, kernel_size=1),
                            nn.BatchNorm2d(cross_dim),
                            nn.ReLU(),
                            DepthwiseSeparableConv(cross_dim, cross_dim, 3, 1, 1),
                        )
                    )
                    if not self.use_res_merge:
                        self.image_cross_blocks.append(
                            nn.Sequential(
                                nn.Conv2d(cross_dim * 2, image_dim,  3, padding=1, bias=False),
                                nn.BatchNorm2d(image_dim),
                                nn.ReLU(),
                                # DepthwiseSeparableConv(cross_dim * 2, cross_dim * 2, 3, 1, 1),
                                DepthwiseSeparableConv(image_dim, image_dim, 3, 1, 1),
                            )
                        )
                        self.point_cross_blocks.append(
                            nn.Sequential(
                                nn.Conv2d(cross_dim * 2, cross_dim, 3, padding=1, bias=False),
                                nn.BatchNorm2d(cross_dim),
                                nn.ReLU(),
                                # DepthwiseSeparableConv(cross_dim * 2, cross_dim * 2, 3, 1, 1),
                                DepthwiseSeparableConv(cross_dim, cross_dim, 3, 1, 1),
                            )
                        )
                    self.cross_vmamba_blocks.append(self._make_vmamba_layer(
                        dim=cross_dim,
                        cross_dim=cross_dim,
                        drop_path = dpr_cross[sum(depths_cross[:i_layer]):sum(depths_cross[:i_layer + 1])],
                        use_checkpoint=False,
                        norm_layer=LayerNorm2d,
                        downsample=nn.Identity(),
                        channel_first=True,
                        # =================
                        ssm_d_state=d_state,
                        ssm_ratio=1.0,
                        ssm_dt_rank='auto',
                        ssm_act_layer=nn.SiLU,
                        ssm_conv=ssm_conv,
                        ssm_conv_bias=False,
                        ssm_drop_rate=0.0,
                        ssm_init='v0',
                        forward_type='cross_noz',
                        # =================
                        mlp_ratio=4.0,
                        mlp_act_layer=nn.GELU,
                        mlp_drop_rate=0.0,
                        gmlp=False,
                        cross=True,
                    ))
            if not self.use_res_merge:
                self.image_conv = nn.Sequential(
                        nn.Conv2d(image_dim * (num_block + 1), image_dim * 2, 3, padding=1, bias=False),
                        nn.BatchNorm2d(image_dim * 2),
                        nn.ReLU(),
                        DepthwiseSeparableConv(image_dim * 2, image_dim, 3, 1, 1),
                    )
                self.lidar_conv = nn.Sequential(
                        nn.Conv2d(point_dim * (num_block + 1), point_dim *2, 3, padding=1, bias=False),
                        nn.BatchNorm2d(point_dim * 2),
                        nn.ReLU(),
                        DepthwiseSeparableConv(point_dim * 2, point_dim, 3, 1, 1),
                    )

            dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]

            for i_layer in range(num_block):
                if self.use_res_merge:
                    self.image_norm.append(nn.BatchNorm2d(image_dim))
                    self.point_norm.append(nn.BatchNorm2d(point_dim))

                # if i_layer == 0 and use_4x:
                #     point_cur_layers.append(BasicBlock(point_dim, point_dim, 2, 1, True))
                if self.use_dw_conv:
                    image_cur_layers = [
                        BasicBlock(image_dim*min(i_layer + 1, max_channel), image_dim*min(i_layer + 2, max_channel), 2, 1, True),
                        DepthwiseSeparableConv(image_dim*min(i_layer + 2, max_channel), image_dim*min(i_layer + 2, max_channel), 3, 1, 1),
                    ]
                    point_cur_layers = [
                        BasicBlock(point_dim*min(i_layer + 1, max_channel), point_dim*min(i_layer + 2, max_channel), 2, 1, True),
                        DepthwiseSeparableConv(point_dim*min(i_layer + 2, max_channel), point_dim*min(i_layer + 2, max_channel), 3, 1, 1),
                    ]
                else:
                    image_cur_layers = [
                        BasicBlock(image_dim*min(i_layer + 1, max_channel), image_dim*min(i_layer + 2, max_channel), 2, 1, True),
                    ]
                    # if i_layer == 0 and use_4x:
                    #     image_cur_layers.append(BasicBlock(image_dim, image_dim, 2, 1, True))
                    
                    point_cur_layers = [
                        BasicBlock(point_dim*min(i_layer + 1, max_channel), point_dim*min(i_layer + 2, max_channel), 2, 1, True),
                    ]
                self.image_down_blocks.append(nn.Sequential(*image_cur_layers))
                self.lidar_down_blocks.append(nn.Sequential(*point_cur_layers))
                
                    
                image_cur_de_layers = []
                point_cur_de_layers = []

                for j in range(i_layer + 1):
                    # if self.use_cross:
                    #     image_cur_de_layers.append(nn.ConvTranspose2d(point_dim, image_dim, kernel_size=2, stride=2, bias=False))
                    # else:
                    image_cur_de_layers.append(nn.ConvTranspose2d(image_dim*min(i_layer + 2 - j, max_channel), image_dim*min(i_layer + 1 - j, max_channel), kernel_size=2, stride=2, bias=False))
                    image_cur_de_layers.append(nn.BatchNorm2d(image_dim*min(i_layer + 1 - j, max_channel)))
                    image_cur_de_layers.append(nn.ReLU())
                    point_cur_de_layers.append(nn.ConvTranspose2d(point_dim*min(i_layer + 2 - j, max_channel), point_dim*min(i_layer + 1 - j, max_channel), kernel_size=2, stride=2, bias=False))
                    point_cur_de_layers.append(nn.BatchNorm2d(point_dim*min(i_layer + 1 - j, max_channel)))
                    point_cur_de_layers.append(nn.ReLU())
                    if self.use_dw_conv:
                        image_cur_de_layers.append(DepthwiseSeparableConv(image_dim*min(i_layer + 1 - j, max_channel), image_dim*min(i_layer + 1 - j, max_channel), 3, 1, 1))
                        point_cur_de_layers.append(DepthwiseSeparableConv(point_dim*min(i_layer + 1 - j, max_channel), point_dim*min(i_layer + 1 - j, max_channel), 3, 1, 1))
                self.image_de_blocks.append(nn.Sequential(*image_cur_de_layers))
                self.lidar_de_blocks.append(nn.Sequential(*point_cur_de_layers))
                self.image_vmamba_blocks.append(self._make_vmamba_layer(
                    dim = image_dim*min(i_layer + 2, max_channel),
                    drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    use_checkpoint=False,
                    norm_layer=LayerNorm2d,
                    downsample=nn.Identity(),
                    channel_first=True,
                    # =================
                    ssm_d_state=d_state,
                    ssm_ratio=1.0,
                    ssm_dt_rank='auto',
                    ssm_act_layer=nn.SiLU,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=False,
                    ssm_drop_rate=0.0,
                    ssm_init='v0',
                    forward_type='v05_noz',
                    # =================
                    mlp_ratio=4.0,
                    mlp_act_layer=nn.GELU,
                    mlp_drop_rate=0.0,
                    gmlp=False,
                ))

                self.point_vmamba_blocks.append(self._make_vmamba_layer(
                    dim = point_dim*min(i_layer + 2, max_channel),
                    drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    use_checkpoint=False,
                    norm_layer=LayerNorm2d,
                    downsample=nn.Identity(),
                    channel_first=True,
                    # =================
                    ssm_d_state=d_state,
                    ssm_ratio=1.0,
                    ssm_dt_rank='auto',
                    ssm_act_layer=nn.SiLU,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=False,
                    ssm_drop_rate=0.0,
                    ssm_init='v0',
                    forward_type='v05_noz',
                    # =================
                    mlp_ratio=4.0,
                    mlp_act_layer=nn.GELU,
                    mlp_drop_rate=0.0,
                    gmlp=False,
                ))
    @staticmethod
    def _make_vmamba_layer(
        dim=96,
        cross_dim=0,
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        channel_first=False,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        cross=False,
        **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        
        if cross_dim != 0:
            blocks1 = []
            blocks2 = []
        else:
            blocks = []
        for d in range(depth):
            if cross_dim != 0:
                blocks1.append(VSSBlock(
                    hidden_dim=dim, 
                    cross_dim=cross_dim,
                    drop_path=drop_path[d],
                    norm_layer=norm_layer,
                    channel_first=channel_first,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=mlp_drop_rate,
                    gmlp=gmlp,
                    use_checkpoint=use_checkpoint,
                ))
                blocks2.append(VSSBlock(
                    hidden_dim=cross_dim, 
                    cross_dim=dim,
                    drop_path=drop_path[d],
                    norm_layer=norm_layer,
                    channel_first=channel_first,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=mlp_drop_rate,
                    
                    gmlp=gmlp,
                    use_checkpoint=use_checkpoint,
                ))
            else:
                blocks.append(VSSBlock(
                    hidden_dim=dim, 
                    cross_dim=0,
                    drop_path=drop_path[d],
                    norm_layer=norm_layer,
                    channel_first=channel_first,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=mlp_drop_rate,
                    gmlp=gmlp,
                    use_checkpoint=use_checkpoint,
                ))
        if not cross:
            return nn.Sequential(OrderedDict(
                blocks=nn.Sequential(*blocks,),
                downsample=downsample,
            ))
        else:
            return nn.Sequential(OrderedDict(
                blocks1=nn.Sequential(*blocks1),
                blocks2=nn.Sequential(*blocks2),
            ))
    def forward(self,batch_dict,available_agents = None,lidar_only = False):

        """
        Args:
            batch_dict:
                spatial_features_img (tensor): Bev features from image modality
                spatial_features (tensor): Bev features from lidar modality

        Returns:
            batch_dict:
                spatial_features (tensor): Bev features after muli-modal fusion
        """
        # 【MambaFusion融合策略分析】
        # 1. 直接从batch_dict获取两种模态的BEV特征
        # 2. 这里假设spatial_features_img已经是整合后的图像BEV特征
        # 3. 没有多agent处理，直接进行双模态融合
        agent_spatial_features = {}
        img_bev_dict = {}
        lidar_bev_dict = {}
        cat_bev_dict = {}
        
        for agent in available_agents:
            img_bev = self.batch_compressor(batch_dict[agent]['spatial_features_img'])  # [B, 80, H, W] - 图像BEV特征
            lidar_bev = batch_dict[agent]['spatial_features']   # [B, 128, H, W] - 激光雷达BEV特征
            
            # 存储用于可视化
            img_bev_dict[agent] = img_bev
            lidar_bev_dict[agent] = lidar_bev
            
            if self.use_vmamba:
                # 【VMamba融合】使用复杂的多尺度VMamba块进行融合
                if self.use_checkpoint:
                    cat_bev = checkpoint.checkpoint(self.mamba_forward, img_bev, lidar_bev)
                else:
                    cat_bev = self.mamba_forward(img_bev, lidar_bev)
            elif lidar_only:
                cat_bev = lidar_bev
            else:
                # 【简单拼接融合】直接在通道维度拼接两种模态
                cat_bev = torch.cat([img_bev, lidar_bev], dim=1)  # [B, 144, H, W]
            
            # 【后处理融合】可选的额外融合层
            if self.use_merge_after:
                for block in self.merge_blocks:
                    cat_bev = block(cat_bev)
            
            # 【最终卷积】将融合后的特征映射到目标通道数
            mm_bev = self.conv(cat_bev) # [B, 128, H, W]
            agent_spatial_features[agent] = mm_bev
            
            # 存储融合后的特征用于可视化
            cat_bev_dict[agent] = cat_bev
            
        # 多agent融合
        if self.use_offset_guided_hierarchical_fusion and 'vehicle' in agent_spatial_features:
            fusion_output = self.offset_guided_hierarchical_fusion(
                agent_spatial_features['vehicle'],
                agent_spatial_features.get('rsu'),
                agent_spatial_features.get('drone'),
                return_debug=self.offset_guided_return_debug,
            )
            if isinstance(fusion_output, tuple):
                mm_bev, fusion_debug = fusion_output
                batch_dict['offset_guided_hierarchical_debug'] = fusion_debug
            else:
                mm_bev = fusion_output
        else:
            # 首版安全策略：至少需要 vehicle 分支；若 vehicle 缺失则回退到原有融合逻辑。
            mm_bev = self.importance_generator(agent_spatial_features, available_agents)[0]
        
        # 可视化ConvFuser的特征处理过程
        # self.visualize_agent_features(
        #     img_bev_dict, lidar_bev_dict, cat_bev_dict, mm_bev, 
        #     available_agents, save_dir="./convfuser_visualization"
        # )
       
        batch_dict['spatial_features'] = mm_bev
        return batch_dict

    def mamba_forward(self, img_bev, lidar_bev):
        ups_img = []
        ups_img.append(img_bev)
        ups_lidar = []
        ups_lidar.append(lidar_bev)
        for i, (block_img, block_lidar) in enumerate(zip(self.image_vmamba_blocks, self.point_vmamba_blocks)):
            img_bev = self.image_down_blocks[i](img_bev) # [2, 80, 90, 90]
            img_bev = block_img(img_bev)
            lidar_bev = self.lidar_down_blocks[i](lidar_bev)
            lidar_bev = block_lidar(lidar_bev)
            if self.use_cross:
                img_bev = self.image_up_blocks[i](img_bev) # [batch_size, 128, 180, 180]
                img_bev_cross = self.cross_vmamba_blocks[i].blocks1((img_bev, lidar_bev)) # [batch_size, 128, 180, 180]
                lidar_bev_cross = self.cross_vmamba_blocks[i].blocks2((lidar_bev, img_bev))
                if not self.use_res_merge:
                    img_bev = self.image_cross_blocks[i](torch.cat([img_bev, lidar_bev_cross], dim=1)) # [batch_size, 128, 180, 180]
                    lidar_bev = self.point_cross_blocks[i](torch.cat([lidar_bev, img_bev_cross], dim=1)) # [batch_size, 128, 180, 180]
                else:
                    img_bev = img_bev_cross
                    lidar_bev = lidar_bev_cross
            if self.use_res_merge:
                img_bev = self.image_norm[i](img_bev + self.image_de_blocks[i](img_bev))
                lidar_bev = self.point_norm[i](lidar_bev + self.lidar_de_blocks[i](lidar_bev))
            else:
                ups_img.append(self.image_de_blocks[i](img_bev))
                ups_lidar.append(self.lidar_de_blocks[i](lidar_bev))
        if self.use_res_merge:
            merge_img = img_bev
            merge_lidar = lidar_bev
        else:
            merge_img = self.image_conv(torch.cat(ups_img, dim=1)) # [1, 80, 360, 360]
            merge_lidar = self.lidar_conv(torch.cat(ups_lidar, dim=1)) # [1, 64, 360, 360]
        cat_bev = torch.cat([merge_img,merge_lidar],dim=1)

        return cat_bev

    def visualize_agent_features(self, img_bev_dict, lidar_bev_dict, cat_bev_dict, mm_bev, agent_names, save_dir="./convfuser_visualization"):
        """
        可视化ConvFuser中不同agent的特征
        
        Args:
            img_bev_dict: 字典，包含每个agent的图像BEV特征
            lidar_bev_dict: 字典，包含每个agent的激光雷达BEV特征  
            cat_bev_dict: 字典，包含每个agent的融合后特征
            mm_bev: 最终的多模态融合特征
            agent_names: agent名称列表
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"[ConvFuser Visualization] Found {len(agent_names)} agents: {agent_names}")
        
        # 获取特征维度
        first_img = list(img_bev_dict.values())[0]
        first_lidar = list(lidar_bev_dict.values())[0]
        first_cat = list(cat_bev_dict.values())[0]
        
        batch_size = first_img.shape[0]
        img_channels = first_img.shape[1]
        lidar_channels = first_lidar.shape[1]
        cat_channels = first_cat.shape[1]
        mm_channels = mm_bev.shape[1]
        height, width = first_img.shape[2], first_img.shape[3]
        
        print(f"[ConvFuser Visualization] Feature shapes:")
        print(f"  Image BEV: {first_img.shape}")
        print(f"  Lidar BEV: {first_lidar.shape}")
        print(f"  Cat BEV: {first_cat.shape}")
        print(f"  MM BEV: {mm_bev.shape}")
        
        # 为每个batch创建可视化
        for batch_idx in range(batch_size):
            # 1. 可视化融合前的图像BEV特征
            self._visualize_pre_fusion_features(
                img_bev_dict, lidar_bev_dict, agent_names, batch_idx, 
                save_dir, "pre_fusion"
            )
            
            # 2. 可视化融合后的特征
            self._visualize_post_fusion_features(
                cat_bev_dict, mm_bev, agent_names, batch_idx,
                save_dir, "post_fusion"
            )
            
            # 3. 创建综合对比图
            self._visualize_comprehensive_comparison(
                img_bev_dict, lidar_bev_dict, cat_bev_dict, mm_bev, 
                agent_names, batch_idx, save_dir
            )

    def _visualize_pre_fusion_features(self, img_bev_dict, lidar_bev_dict, agent_names, batch_idx, save_dir, prefix):
        """可视化融合前的特征"""
        fig, axes = plt.subplots(2, len(agent_names), figsize=(4*len(agent_names), 8))
        if len(agent_names) == 1:
            axes = axes.reshape(2, 1)
        
        # 收集所有占用图用于统一颜色范围
        all_img_maps = []
        all_lidar_maps = []
        
        for agent_name in agent_names:
            img_features = img_bev_dict[agent_name][batch_idx]  # [C, H, W]
            lidar_features = lidar_bev_dict[agent_name][batch_idx]  # [C, H, W]
            
            # 使用L2范数计算占用强度
            img_map = torch.norm(img_features, dim=0).detach().cpu().numpy()
            lidar_map = torch.norm(lidar_features, dim=0).detach().cpu().numpy()
            
            all_img_maps.append(img_map)
            all_lidar_maps.append(lidar_map)
        
        # 计算统一的颜色范围
        all_img_values = np.concatenate([m.flatten() for m in all_img_maps])
        all_lidar_values = np.concatenate([m.flatten() for m in all_lidar_maps])
        img_vmin, img_vmax = np.min(all_img_values), np.max(all_img_values)
        lidar_vmin, lidar_vmax = np.min(all_lidar_values), np.max(all_lidar_values)
        
        for i, agent_name in enumerate(agent_names):
            img_map = all_img_maps[i]
            lidar_map = all_lidar_maps[i]
            
            # 上排：图像BEV特征
            im1 = axes[0, i].imshow(img_map, cmap='hot', aspect='equal', vmin=img_vmin, vmax=img_vmax)
            axes[0, i].set_title(f'{agent_name}\nImage BEV')
            axes[0, i].set_ylabel('Y')
            
            # 下排：激光雷达BEV特征
            im2 = axes[1, i].imshow(lidar_map, cmap='hot', aspect='equal', vmin=lidar_vmin, vmax=lidar_vmax)
            axes[1, i].set_title(f'{agent_name}\nLidar BEV')
            axes[1, i].set_ylabel('Y')
        
        # 移除颜色条，保持简洁
        
        fig.suptitle(f'Pre-Fusion Features - Batch {batch_idx}', fontsize=14)
        plt.subplots_adjust(top=0.8, bottom=0.2, hspace=0.6)
        
        save_path = os.path.join(save_dir, f'{prefix}_batch_{batch_idx}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[ConvFuser Visualization] Saved pre-fusion: {save_path}")

    def _visualize_post_fusion_features(self, cat_bev_dict, mm_bev, agent_names, batch_idx, save_dir, prefix):
        """可视化融合后的特征"""
        fig, axes = plt.subplots(2, len(agent_names) + 1, figsize=(4*(len(agent_names) + 1), 8))
        if len(agent_names) == 1:
            axes = axes.reshape(2, 2)
        
        # 收集所有占用图用于统一颜色范围
        all_cat_maps = []
        
        for agent_name in agent_names:
            cat_features = cat_bev_dict[agent_name][batch_idx]  # [C, H, W]
            cat_map = torch.norm(cat_features, dim=0).detach().cpu().numpy()
            all_cat_maps.append(cat_map)
        
        # 添加最终融合特征
        mm_map = torch.norm(mm_bev[batch_idx], dim=0).detach().cpu().numpy()
        all_cat_maps.append(mm_map)
        
        # 计算统一的颜色范围
        all_values = np.concatenate([m.flatten() for m in all_cat_maps])
        vmin, vmax = np.min(all_values), np.max(all_values)
        
        for i, agent_name in enumerate(agent_names):
            cat_map = all_cat_maps[i]
            
            # 上排：融合后特征
            im1 = axes[0, i].imshow(cat_map, cmap='hot', aspect='equal', vmin=vmin, vmax=vmax)
            axes[0, i].set_title(f'{agent_name}\nFused BEV')
            axes[0, i].set_ylabel('Y')
            
            # 下排：统计信息
            occupied_cells = np.count_nonzero(cat_map)
            total_cells = cat_map.size
            occupancy_ratio = occupied_cells / total_cells
            
            stats_text = f"""Occupied: {occupied_cells:,}/{total_cells:,}
            Ratio: {occupancy_ratio:.2%}"""
            
            axes[1, i].text(0.1, 0.9, stats_text, transform=axes[1, i].transAxes, 
                          fontsize=9, verticalalignment='top', fontfamily='monospace')
            axes[1, i].set_xlim(0, 1)
            axes[1, i].set_ylim(0, 1)
            axes[1, i].axis('off')
            axes[1, i].set_title('Statistics')
        
        # 最后一列：最终多模态融合结果
        im_final = axes[0, -1].imshow(mm_map, cmap='hot', aspect='equal', vmin=vmin, vmax=vmax)
        axes[0, -1].set_title('Final\nMulti-Modal BEV')
        axes[0, -1].set_ylabel('Y')
        
        # 最终结果统计
        occupied_cells = np.count_nonzero(mm_map)
        total_cells = mm_map.size
        occupancy_ratio = occupied_cells / total_cells
        
        stats_text = f"""Occupied: {occupied_cells:,}/{total_cells:,}
        Ratio: {occupancy_ratio:.2%}"""
        
        axes[1, -1].text(0.1, 0.9, stats_text, transform=axes[1, -1].transAxes, 
                        fontsize=9, verticalalignment='top', fontfamily='monospace')
        axes[1, -1].set_xlim(0, 1)
        axes[1, -1].set_ylim(0, 1)
        axes[1, -1].axis('off')
        axes[1, -1].set_title('Statistics')
        
        # 移除颜色条，保持简洁
        
        fig.suptitle(f'Post-Fusion Features - Batch {batch_idx}', fontsize=14)
        plt.subplots_adjust(top=0.8, bottom=0.2, hspace=0.6)
        
        save_path = os.path.join(save_dir, f'{prefix}_batch_{batch_idx}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[ConvFuser Visualization] Saved post-fusion: {save_path}")

    def _visualize_comprehensive_comparison(self, img_bev_dict, lidar_bev_dict, cat_bev_dict, mm_bev, agent_names, batch_idx, save_dir):
        """创建综合对比图"""
        fig, axes = plt.subplots(3, len(agent_names) + 1, figsize=(4*(len(agent_names) + 1), 12))
        if len(agent_names) == 1:
            axes = axes.reshape(3, 2)
        
        # 收集所有特征图
        all_maps = []
        for agent_name in agent_names:
            img_map = torch.norm(img_bev_dict[agent_name][batch_idx], dim=0).detach().cpu().numpy()
            lidar_map = torch.norm(lidar_bev_dict[agent_name][batch_idx], dim=0).detach().cpu().numpy()
            cat_map = torch.norm(cat_bev_dict[agent_name][batch_idx], dim=0).detach().cpu().numpy()
            all_maps.extend([img_map, lidar_map, cat_map])
        
        # 添加最终融合结果
        mm_map = torch.norm(mm_bev[batch_idx], dim=0).detach().cpu().numpy()
        all_maps.append(mm_map)
        
        # 计算统一的颜色范围
        all_values = np.concatenate([m.flatten() for m in all_maps])
        vmin, vmax = np.min(all_values), np.max(all_values)
        
        for i, agent_name in enumerate(agent_names):
            img_map = torch.norm(img_bev_dict[agent_name][batch_idx], dim=0).detach().cpu().numpy()
            lidar_map = torch.norm(lidar_bev_dict[agent_name][batch_idx], dim=0).detach().cpu().numpy()
            cat_map = torch.norm(cat_bev_dict[agent_name][batch_idx], dim=0).detach().cpu().numpy()
            
            # 第一行：图像BEV
            im1 = axes[0, i].imshow(img_map, cmap='hot', aspect='equal', vmin=vmin, vmax=vmax)
            axes[0, i].set_title(f'{agent_name}\nImage BEV')
            axes[0, i].set_ylabel('Y')
            
            # 第二行：激光雷达BEV
            im2 = axes[1, i].imshow(lidar_map, cmap='hot', aspect='equal', vmin=vmin, vmax=vmax)
            axes[1, i].set_title(f'{agent_name}\nLidar BEV')
            axes[1, i].set_ylabel('Y')
            
            # 第三行：融合后特征
            im3 = axes[2, i].imshow(cat_map, cmap='hot', aspect='equal', vmin=vmin, vmax=vmax)
            axes[2, i].set_title(f'{agent_name}\nFused BEV')
            axes[2, i].set_ylabel('Y')
        
        # 最后一列：最终多模态融合结果
        im_final = axes[0, -1].imshow(mm_map, cmap='hot', aspect='equal', vmin=vmin, vmax=vmax)
        axes[0, -1].set_title('Final\nMulti-Modal BEV')
        axes[0, -1].set_ylabel('Y')
        
        # 中间和底部行显示统计信息
        occupied_cells = np.count_nonzero(mm_map)
        total_cells = mm_map.size
        occupancy_ratio = occupied_cells / total_cells
        
        stats_text = f"""Final Fusion Result:
            Occupied: {occupied_cells:,}/{total_cells:,}
            Ratio: {occupancy_ratio:.2%}"""
        
        axes[1, -1].text(0.1, 0.9, stats_text, transform=axes[1, -1].transAxes, 
                        fontsize=9, verticalalignment='top', fontfamily='monospace')
        axes[1, -1].set_xlim(0, 1)
        axes[1, -1].set_ylim(0, 1)
        axes[1, -1].axis('off')
        axes[1, -1].set_title('Final Statistics')
        
        # 第三行显示融合过程
        axes[2, -1].text(0.1, 0.9, 'ConvFuser Pipeline:\n1. Image BEV\n2. Lidar BEV\n3. Agent Fusion\n4. Multi-Modal Fusion', 
                        transform=axes[2, -1].transAxes, fontsize=9, verticalalignment='top', fontfamily='monospace')
        axes[2, -1].set_xlim(0, 1)
        axes[2, -1].set_ylim(0, 1)
        axes[2, -1].axis('off')
        axes[2, -1].set_title('Pipeline')
        
        # 移除颜色条，保持简洁
        
        fig.suptitle(f'ConvFuser Comprehensive Comparison - Batch {batch_idx}', fontsize=16)
        plt.subplots_adjust(top=0.85, bottom=0.2, hspace=0.4)
        
        save_path = os.path.join(save_dir, f'comprehensive_comparison_batch_{batch_idx}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[ConvFuser Visualization] Saved comprehensive comparison: {save_path}")


def _to_2tuple(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Convert an int, list, or tuple into a validated 2-tuple."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"Expected a 2-element window size, got {value}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


class OffsetPredictor(nn.Module):
    """Predict multi-point local offsets for one BEV source."""

    def __init__(
        self,
        channels: int,
        num_points: int,
        offset_range: float,
        hidden_channels: Optional[int] = None,
    ) -> None:
        """Initialize the offset predictor.

        Args:
            channels: Input feature channel count.
            num_points: Number of offsets predicted at each BEV location.
            offset_range: Local offset range `r`; outputs are constrained to `[-r, r]`.
            hidden_channels: Optional hidden channel count for the predictor head.
        """
        super().__init__()
        hidden_dim = hidden_channels if hidden_channels is not None else channels
        self.num_points = num_points
        self.offset_range = float(offset_range)
        self.offset_head = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_points * 2, kernel_size=1, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_raw: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Predict constrained offsets.

        Args:
            x: Input tensor of shape `[B, C, H, W]`.
            return_raw: Whether to also return unconstrained raw offsets.

        Returns:
            Constrained offsets of shape `[B, K, 2, H, W]`.
            If `return_raw=True`, also returns raw offsets with the same shape.
        """
        batch_size, _, height, width = x.shape
        raw_offsets = self.offset_head(x).reshape(batch_size, self.num_points, 2, height, width)
        offsets = torch.tanh(raw_offsets) * self.offset_range
        if return_raw:
            return offsets, raw_offsets
        return offsets


class MultiPointSampler(nn.Module):
    """Sample K points from one BEV source using deformable bilinear sampling."""

    def __init__(
        self,
        align_corners: bool = False,
        padding_mode: str = 'zeros',
    ) -> None:
        """Initialize the sampler.

        Args:
            align_corners: Explicit `grid_sample` alignment choice.
            padding_mode: Explicit `grid_sample` padding mode shared by all sources.
        """
        super().__init__()
        self.align_corners = align_corners
        self.padding_mode = padding_mode

    def _normalize_coordinate(self, coord: torch.Tensor, size: int) -> torch.Tensor:
        """Normalize pixel coordinates into `[-1, 1]` for `grid_sample`."""
        if self.align_corners:
            if size <= 1:
                return torch.zeros_like(coord)
            return (2.0 * coord / float(size - 1)) - 1.0
        return ((2.0 * coord + 1.0) / float(size)) - 1.0

    def forward(self, feature: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Sample K offset-guided points from the same BEV source.

        Args:
            feature: Source feature map of shape `[B, C, H, W]`.
            offsets: Constrained offsets of shape `[B, K, 2, H, W]`.

        Returns:
            Sampled features of shape `[B, K, C, H, W]`.
        """
        if feature.dim() != 4:
            raise ValueError(f"feature must be [B, C, H, W], got {feature.shape}")
        if offsets.dim() != 5:
            raise ValueError(f"offsets must be [B, K, 2, H, W], got {offsets.shape}")

        batch_size, channels, height, width = feature.shape
        _, num_points, offset_dim, offset_height, offset_width = offsets.shape
        if offset_dim != 2:
            raise ValueError(f"offset dimension must be 2, got {offset_dim}")
        if offset_height != height or offset_width != width:
            raise ValueError("offset spatial shape must match feature spatial shape")

        yy, xx = torch.meshgrid(
            torch.arange(height, device=feature.device, dtype=feature.dtype),
            torch.arange(width, device=feature.device, dtype=feature.dtype),
            indexing='ij',
        )
        base_x = xx.unsqueeze(0).expand(batch_size, -1, -1)
        base_y = yy.unsqueeze(0).expand(batch_size, -1, -1)

        sampled_features = []
        for point_idx in range(num_points):
            offset_x = offsets[:, point_idx, 0, :, :]
            offset_y = offsets[:, point_idx, 1, :, :]
            sample_x = base_x + offset_x
            sample_y = base_y + offset_y

            grid_x = self._normalize_coordinate(sample_x, width)
            grid_y = self._normalize_coordinate(sample_y, height)
            grid = torch.stack([grid_x, grid_y], dim=-1)

            sampled = F.grid_sample(
                feature,
                grid,
                mode='bilinear',
                padding_mode=self.padding_mode,
                align_corners=self.align_corners,
            )
            sampled_features.append(sampled)

        return torch.stack(sampled_features, dim=1)


class SampleAttentionAggregator(nn.Module): #TODO
    """Aggregate K sampled features into one refined feature map."""

    def __init__(self, channels: int) -> None:
        """Initialize the sample attention head.

        Args:
            channels: Sample feature channel count.
        """
        super().__init__()
        self.attention_head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
        )

    def forward(self, sampled_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict K attention weights and aggregate sampled features.

        Args:
            sampled_features: Tensor of shape `[B, K, C, H, W]`.

        Returns:
            A tuple containing:
                - attention weights of shape `[B, K, H, W]`
                - refined feature of shape `[B, C, H, W]`
        """
        if sampled_features.dim() != 5:
            raise ValueError(
                f"sampled_features must be [B, K, C, H, W], got {sampled_features.shape}"
            )

        batch_size, num_points, channels, height, width = sampled_features.shape
        flat_samples = sampled_features.reshape(batch_size * num_points, channels, height, width)
        attention_logits = self.attention_head(flat_samples).reshape(batch_size, num_points, height, width)
        attention_weights = torch.softmax(attention_logits, dim=1)
        refined_feature = (attention_weights.unsqueeze(2) * sampled_features).sum(dim=1)
        return attention_weights, refined_feature


class SampleCrossAttention(nn.Module):
    """Refine one BEV feature with ego-conditioned sampled cross attention."""

    def __init__(
        self,
        channels: int,
        attention_dim: Optional[int] = None,
        ffn_ratio: float = 2.0,
        drop_rate: float = 0.0,
        layer_scale_init: float = 0.0,
    ) -> None:
        """Initialize the sample cross-attention block.

        Args:
            channels: Input feature channel count.
            attention_dim: Query/key projection dimension.
            ffn_ratio: Expansion ratio used by the FFN sub-layer.
            drop_rate: Dropout rate used by the FFN.
            layer_scale_init: Initial value of the residual layer-scale parameters.
        """
        super().__init__()
        inner_dim = int(attention_dim) if attention_dim is not None else channels
        hidden_dim = max(int(channels * float(ffn_ratio)), channels)
        self.scale = float(inner_dim) ** -0.5
        self.ego_norm = LayerNorm2d(channels)
        self.token_norm = LayerNorm2d(channels)
        self.q_proj = nn.Conv2d(channels, inner_dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(channels, inner_dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.attn_gamma = nn.Parameter(
            torch.full((1, channels, 1, 1), float(layer_scale_init))
        )
        self.ffn_norm = LayerNorm2d(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False),
            nn.Dropout(drop_rate),
        )
        self.ffn_gamma = nn.Parameter(
            torch.full((1, channels, 1, 1), float(layer_scale_init))
        )

    def _apply_token_norm(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply 2D normalization to a `[B, T, C, H, W]` token tensor."""
        batch_size, token_count, channels, height, width = tokens.shape
        flat_tokens = tokens.reshape(batch_size * token_count, channels, height, width)
        norm_tokens = self.token_norm(flat_tokens)
        return norm_tokens.reshape(batch_size, token_count, channels, height, width)

    def forward(
        self,
        ego_feature: torch.Tensor,
        sampled_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run ego-conditioned attention over self and sampled tokens.

        Args:
            ego_feature: Ego/source feature of shape `[B, C, H, W]`.
            sampled_features: Offset-sampled features of shape `[B, K, C, H, W]`.

        Returns:
            A tuple containing:
                - sampled-token attention weights of shape `[B, K, H, W]`
                - refined feature of shape `[B, C, H, W]`
        """
        if ego_feature.dim() != 4:
            raise ValueError(f"ego_feature must be [B, C, H, W], got {ego_feature.shape}")
        if sampled_features.dim() != 5:
            raise ValueError(
                f"sampled_features must be [B, K, C, H, W], got {sampled_features.shape}"
            )

        batch_size, num_points, channels, height, width = sampled_features.shape
        if ego_feature.shape != (batch_size, channels, height, width):
            raise ValueError(
                "ego_feature shape must match sampled feature layout, got "
                f"ego={ego_feature.shape}, sampled={sampled_features.shape}"
            )

        tokens = torch.cat([ego_feature.unsqueeze(1), sampled_features], dim=1)
        norm_ego = self.ego_norm(ego_feature)
        norm_tokens = self._apply_token_norm(tokens)

        query = self.q_proj(norm_ego).unsqueeze(1)
        flat_tokens = norm_tokens.reshape((batch_size * (num_points + 1)), channels, height, width)
        keys = self.k_proj(flat_tokens).reshape(batch_size, num_points + 1, -1, height, width)
        values = self.v_proj(flat_tokens).reshape(batch_size, num_points + 1, channels, height, width)

        attention_logits = (query * keys).sum(dim=2) * self.scale
        full_attention_weights = torch.softmax(attention_logits, dim=1)
        context = (full_attention_weights.unsqueeze(2) * values).sum(dim=1)

        attended_feature = ego_feature + (self.attn_gamma * self.out_proj(context))
        refined_feature = attended_feature + (
            self.ffn_gamma * self.ffn(self.ffn_norm(attended_feature))
        )

        sampled_attention_weights = full_attention_weights[:, 1:, :, :]
        sampled_attention_weights = sampled_attention_weights / sampled_attention_weights.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)
        return sampled_attention_weights, refined_feature


class GeometricTransmissionGate(nn.Module):
    """Convert offset magnitude into geometric transmission weights."""

    def __init__(self, alpha: float) -> None:
        """Initialize the geometric gate.

        Args:
            alpha: Exponential decay factor in `exp(-alpha * d)`.
        """
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, offsets: torch.Tensor, attention_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute offset-aware transmission gates.

        Args:
            offsets: Offset tensor of shape `[B, K, 2, H, W]`.
            attention_weights: Attention weights of shape `[B, K, H, W]`.

        Returns:
            A tuple containing:
                - distance map `[B, 1, H, W]`
                - gate map `[B, 1, H, W]`
        """
        if offsets.dim() != 5:
            raise ValueError(f"offsets must be [B, K, 2, H, W], got {offsets.shape}")
        if attention_weights.dim() != 4:
            raise ValueError(
                f"attention_weights must be [B, K, H, W], got {attention_weights.shape}"
            )

        offset_distance = torch.norm(offsets, p=2, dim=2)
        distance_map = (attention_weights * offset_distance).sum(dim=1, keepdim=True)
        gate_map = torch.exp(-self.alpha * distance_map)
        return distance_map, gate_map


class EgoFusionLayer(nn.Module):
    """Fuse ego, RSU, and drone refined features into one ego-centric BEV."""

    def __init__(self, channels: int) -> None:
        """Initialize the ego-centric source fusion layer.

        Args:
            channels: Channel count of each source BEV feature.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        refined_veh: torch.Tensor,
        refined_rsu: torch.Tensor,
        refined_drone: torch.Tensor,
        gate_rsu: torch.Tensor,
        gate_drone: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse three refined BEV maps.

        Args:
            refined_veh: Vehicle feature `[B, C, H, W]`.
            refined_rsu: RSU feature `[B, C, H, W]`.
            refined_drone: Drone feature `[B, C, H, W]`.
            gate_rsu: RSU gate `[B, 1, H, W]`.
            gate_drone: Drone gate `[B, 1, H, W]`.

        Returns:
            Ego-centric fused feature of shape `[B, C, H, W]`.
        """
        fused_input = torch.cat(
            [
                refined_veh,
                gate_rsu * refined_rsu,
                gate_drone * refined_drone,
            ],
            dim=1,
        )
        return self.proj(fused_input)


class WindowPositionalEncoding(nn.Module):
    """Add learnable 2D positional embedding inside each local window."""

    def __init__(self, channels: int, window_size: Union[int, Tuple[int, int]]) -> None:
        """Initialize the local window positional encoding.

        Args:
            channels: Channel count of the local window features.
            window_size: Window size `(wh, ww)` or scalar.
        """
        super().__init__()
        window_height, window_width = _to_2tuple(window_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, window_height, window_width))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embedding to window features.

        Args:
            x: Window features of shape `[B_windows, C, wh, ww]`.

        Returns:
            Position-enhanced window features with the same shape.
        """
        return x + self.pos_embed


class LocalReorderHead(nn.Module):
    """Predict one local scan-priority score per token within each window."""

    def __init__(self, channels: int) -> None:
        """Initialize the reorder score head.

        Args:
            channels: Input channel count.
        """
        super().__init__()
        self.score_head = nn.Conv2d(channels, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict local reorder scores.

        Args:
            x: Window tensor of shape `[B_windows, C, wh, ww]`.

        Returns:
            Flattened local scores of shape `[B_windows, wh * ww]`.
        """
        return self.score_head(x).flatten(start_dim=1)


class LocalMambaBlock(nn.Module):
    """Apply local window-wise reordering and local Mamba modeling."""

    def __init__(
        self,
        channels: int,
        window_size: Union[int, Tuple[int, int]],
        depth: int = 1,
        ssm_d_state: int = 1,
        ssm_ratio: float = 1.0,
        ssm_dt_rank: Union[int, str] = 'auto',
        ssm_conv: int = 3,
        ssm_conv_bias: bool = False,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        forward_type: str = 'v05_noz',
    ) -> None:
        """Initialize the local Mamba block.

        Args:
            channels: Input channel count.
            window_size: Local window size `(wh, ww)` or scalar.
            depth: Number of local VSS blocks.
            ssm_d_state: VSS state size.
            ssm_ratio: VSS inner ratio.
            ssm_dt_rank: VSS dt rank.
            ssm_conv: VSS depthwise conv kernel size.
            ssm_conv_bias: Whether VSS depthwise conv uses bias.
            mlp_ratio: VSS MLP ratio.
            mlp_drop_rate: VSS MLP dropout rate.
            forward_type: Existing VSS forward type to reuse.
        """
        super().__init__()
        self.window_size = _to_2tuple(window_size)
        self.pos_encoding = WindowPositionalEncoding(channels=channels, window_size=self.window_size)
        self.reorder_head = LocalReorderHead(channels=channels)
        self.local_blocks = nn.Sequential(
            *[
                VSSBlock(
                    hidden_dim=channels,
                    cross_dim=0,
                    drop_path=0.0,
                    norm_layer=LayerNorm2d,
                    channel_first=True,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=nn.SiLU,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=0.0,
                    ssm_init='v0',
                    forward_type=forward_type,
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=nn.GELU,
                    mlp_drop_rate=mlp_drop_rate,
                    gmlp=False,
                    use_checkpoint=False,
                )
                for _ in range(depth)
            ]
        )

    def _pad_to_window(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int]]:
        """Zero-pad BEV features so that they can be partitioned into windows."""
        batch_size, channels, height, width = x.shape
        window_height, window_width = self.window_size
        pad_height = (window_height - (height % window_height)) % window_height
        pad_width = (window_width - (width % window_width)) % window_width
        if pad_height > 0 or pad_width > 0:
            x = F.pad(x, (0, pad_width, 0, pad_height), mode='constant', value=0.0)
        padded_height = height + pad_height
        padded_width = width + pad_width
        meta = {
            'batch_size': batch_size,
            'channels': channels,
            'height': height,
            'width': width,
            'padded_height': padded_height,
            'padded_width': padded_width,
            'pad_height': pad_height,
            'pad_width': pad_width,
        }
        return x, meta

    def _partition_windows(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int]]:
        """Partition BEV features into local windows."""
        padded_x, meta = self._pad_to_window(x)
        batch_size, channels, padded_height, padded_width = padded_x.shape
        window_height, window_width = self.window_size
        num_windows_h = padded_height // window_height
        num_windows_w = padded_width // window_width

        windows = padded_x.reshape(
            batch_size,
            channels,
            num_windows_h,
            window_height,
            num_windows_w,
            window_width,
        )
        windows = windows.permute(0, 2, 4, 1, 3, 5).contiguous()
        windows = windows.reshape(
            batch_size * num_windows_h * num_windows_w,
            channels,
            window_height,
            window_width,
        )

        meta['num_windows_h'] = num_windows_h
        meta['num_windows_w'] = num_windows_w
        return windows, meta

    def _merge_windows(self, windows: torch.Tensor, meta: Dict[str, int]) -> torch.Tensor:
        """Merge local windows back to the BEV grid and crop zero padding."""
        batch_size = meta['batch_size']
        channels = meta['channels']
        num_windows_h = meta['num_windows_h']
        num_windows_w = meta['num_windows_w']
        window_height, window_width = self.window_size

        merged = windows.reshape(batch_size, num_windows_h, num_windows_w, channels, window_height, window_width)
        merged = merged.permute(0, 3, 1, 4, 2, 5).contiguous()
        merged = merged.reshape(batch_size, channels, meta['padded_height'], meta['padded_width'])
        return merged[:, :, :meta['height'], :meta['width']]

    def forward(
        self,
        x: torch.Tensor,
        return_debug: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Run local reordering and local Mamba.

        Args:
            x: Input feature map of shape `[B, C, H, W]`.
            return_debug: Whether to return reorder-related debug tensors.

        Returns:
            Local fused feature of shape `[B, C, H, W]`.
            If `return_debug=True`, also returns a debug dictionary.
        """
        windows, meta = self._partition_windows(x)
        windows = self.pos_encoding(windows)
        reorder_scores = self.reorder_head(windows)

        batch_windows, channels, window_height, window_width = windows.shape
        token_count = window_height * window_width
        tokens = windows.reshape(batch_windows, channels, token_count)

        sort_idx = torch.argsort(reorder_scores, dim=-1, descending=True)
        gather_idx = sort_idx.unsqueeze(1).expand(-1, channels, -1)
        sorted_tokens = torch.gather(tokens, dim=2, index=gather_idx)

        # 首版安全实现：按行优先将排序后的 token 序列回填成同尺寸伪 2D 网格，
        # 以便直接复用现有 2D `VSSBlock`，避免额外引入新的 1D Mamba 路径。
        sorted_windows = sorted_tokens.reshape(batch_windows, channels, window_height, window_width)
        local_windows = self.local_blocks(sorted_windows)

        local_tokens = local_windows.reshape(batch_windows, channels, token_count)
        inverse_idx = torch.argsort(sort_idx, dim=-1)
        inverse_gather_idx = inverse_idx.unsqueeze(1).expand(-1, channels, -1)
        restored_tokens = torch.gather(local_tokens, dim=2, index=inverse_gather_idx)
        restored_windows = restored_tokens.reshape(batch_windows, channels, window_height, window_width)
        restored_feature = self._merge_windows(restored_windows, meta)

        if not return_debug:
            return restored_feature

        reorder_score_map = reorder_scores.reshape(
            meta['batch_size'],
            meta['num_windows_h'],
            meta['num_windows_w'],
            window_height,
            window_width,
        )
        debug_info = {
            'reorder_scores': reorder_score_map,
            'sort_idx': sort_idx.reshape(
                meta['batch_size'],
                meta['num_windows_h'],
                meta['num_windows_w'],
                token_count,
            ),
        }
        return restored_feature, debug_info


class HilbertGlobalMambaBlock(nn.Module):
    """Apply global 1D Mamba after Hilbert-order serialization."""

    def __init__(
        self,
        channels: int,
        depth: int = 1,
        ssm_d_state: int = 16,
        ssm_ratio: float = 2.0,
        ssm_dt_rank: Union[int, str] = 'auto',
        ssm_conv: int = 3,
        ssm_conv_bias: bool = False,
        use_reverse_scan: bool = True,
        norm_epsilon: float = 1e-5,
    ) -> None:
        """Initialize the Hilbert global fusion block.

        Notes:
            This first dense-BEV version only borrows the Hilbert-order global
            serialization idea from the old sparse `GlobalMamba`. It intentionally
            does not reproduce the old sparse downsample / upsample path.

        Args:
            channels: Input channel count.
            depth: Number of 1D Mamba blocks.
            ssm_d_state: 1D Mamba state size.
            ssm_ratio: 1D Mamba expansion ratio.
            ssm_dt_rank: 1D Mamba dt rank.
            ssm_conv: 1D Mamba local conv kernel size.
            ssm_conv_bias: Whether 1D Mamba conv uses bias.
            use_reverse_scan: Whether to alternate reverse scanning across layers.
            norm_epsilon: Final sequence norm epsilon.
        """
        super().__init__()
        self.channels = channels
        self.use_reverse_scan = use_reverse_scan
        self.curve_template: Dict[str, torch.Tensor] = {}
        self.hilbert_spatial_size: Dict[str, Tuple[int, int, int]] = {}
        self._hilbert_cache: Dict[Tuple[int, int, int, str], Dict[str, torch.Tensor]] = {}

        self.pos_embed = nn.Sequential(
            nn.Linear(3, channels),
            nn.LayerNorm(channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        ssm_cfg = {
            'd_state': ssm_d_state,
            'd_conv': ssm_conv,
            'expand': ssm_ratio,
            'dt_rank': ssm_dt_rank,
            'conv_bias': ssm_conv_bias,
        }
        self.mamba_layers = nn.ModuleList(
            [
                create_block(
                    d_model=channels,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=False,
                    residual_in_fp32=True,
                    fused_add_norm=False,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(depth)
            ]
        )
        self.norm_f = nn.LayerNorm(channels, eps=norm_epsilon)

        template_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'ckpts', 'hilbert_template')
        )
        for rank in (7, 8, 9, 10):
            self._load_template(
                os.path.join(template_root, f'curve_template_3d_rank_{rank}.pth'),
                rank,
            )

    def _load_template(self, path: str, rank: int) -> None:
        """Load one Hilbert curve template."""
        template = torch.load(path, map_location='cpu')
        if isinstance(template, dict):
            curve = template['data'].reshape(-1).long()
            spatial_size = tuple(template['size'])
        else:
            curve = template.reshape(-1).long()
            side = 2 ** rank
            spatial_size = (1, side, side)
        self.curve_template[f'curve_template_rank{rank}'] = curve
        self.hilbert_spatial_size[f'curve_template_rank{rank}'] = spatial_size

    def _select_template_key(self, height: int, width: int) -> str:
        """Select a Hilbert template rank that covers the current BEV size."""
        max_dim = max(height, width)
        if max_dim > 512:
            template_key = 'curve_template_rank10'
        elif max_dim > 256:
            template_key = 'curve_template_rank9'
        elif max_dim > 128:
            template_key = 'curve_template_rank8'
        else:
            template_key = 'curve_template_rank7'

        _, hilbert_height, hilbert_width = self.hilbert_spatial_size[template_key]
        if hilbert_height < height or hilbert_width < width:
            raise ValueError(
                f"Hilbert template {template_key} cannot cover dense BEV size {(height, width)}. "
                f"Template spatial size is {(hilbert_height, hilbert_width)}."
            )
        return template_key

    def _build_hilbert_cache(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Build and cache Hilbert ordering indices for a dense BEV grid."""
        cache_key = (batch_size, height, width, str(device))
        if cache_key in self._hilbert_cache:
            return self._hilbert_cache[cache_key]

        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.long),
            torch.arange(width, device=device, dtype=torch.long),
            indexing='ij',
        )
        yy_flat = yy.reshape(-1)
        xx_flat = xx.reshape(-1)
        token_count = yy_flat.numel()

        coords_all = []
        pos_coords = []
        denom_h = max(height - 1, 1)
        denom_w = max(width - 1, 1)
        base_pos = torch.stack(
            [
                torch.zeros_like(yy_flat, dtype=torch.float32),
                yy_flat.to(torch.float32) / float(denom_h),
                xx_flat.to(torch.float32) / float(denom_w),
            ],
            dim=1,
        )
        for batch_idx in range(batch_size):
            batch_col = torch.full((token_count, 1), batch_idx, device=device, dtype=torch.long)
            z_col = torch.zeros((token_count, 1), device=device, dtype=torch.long)
            coords_all.append(torch.cat([batch_col, z_col, yy_flat[:, None], xx_flat[:, None]], dim=1))
            pos_coords.append(base_pos)

        coords = torch.cat(coords_all, dim=0)
        pos_coords_tensor = torch.stack(pos_coords, dim=0)

        template_key = self._select_template_key(height, width)
        template = self.curve_template[template_key].to(device)
        hilbert_size = self.hilbert_spatial_size[template_key]
        index_info = get_hilbert_index_2d_mamba_lite(
            template=template,
            coors=coords,
            batch_size=batch_size,
            hilbert_spatial_size=hilbert_size,
            shift=(0, 0),
            debug=False,
        )
        sort_idx = torch.stack(
            [index_info['inds_curt_to_next'][batch_idx] for batch_idx in range(batch_size)],
            dim=0,
        )
        inverse_idx = torch.stack(
            [index_info['inds_next_to_curt'][batch_idx] for batch_idx in range(batch_size)],
            dim=0,
        )
        cache = {
            'sort_idx': sort_idx,
            'inverse_idx': inverse_idx,
            'pos_coords': pos_coords_tensor,
        }
        self._hilbert_cache[cache_key] = cache
        return cache

    def _run_sequence_mamba(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run stacked 1D Mamba blocks on serialized global tokens."""
        hidden = tokens
        residual = None
        for layer_idx, layer in enumerate(self.mamba_layers):
            if self.use_reverse_scan and (layer_idx % 2 == 1):
                hidden_rev = hidden.flip(1)
                residual_rev = residual.flip(1) if residual is not None else None
                hidden_rev, residual_rev = layer(hidden_rev, residual_rev)
                hidden = hidden_rev.flip(1)
                residual = residual_rev.flip(1) if residual_rev is not None else None
            else:
                hidden, residual = layer(hidden, residual)
        residual = (hidden + residual) if residual is not None else hidden
        return self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Serialize dense BEV tokens by Hilbert order, run 1D Mamba, and restore."""
        batch_size, channels, height, width = x.shape
        cache = self._build_hilbert_cache(batch_size, height, width, x.device)
        tokens = x.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        pos_embed = self.pos_embed(cache['pos_coords'].to(dtype=x.dtype))
        tokens = tokens + pos_embed

        sort_idx = cache['sort_idx'].unsqueeze(-1).expand(-1, -1, channels)
        inverse_idx = cache['inverse_idx'].unsqueeze(-1).expand(-1, -1, channels)

        ordered_tokens = torch.gather(tokens, dim=1, index=sort_idx)
        global_tokens = self._run_sequence_mamba(ordered_tokens)
        restored_tokens = torch.gather(global_tokens, dim=1, index=inverse_idx)
        return restored_tokens.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()


class GlobalFusionBlock(nn.Module):
    """Apply configurable global fusion after local Mamba."""

    def __init__(
        self,
        channels: int,
        global_fusion_type: str = 'mamba',
        mamba_depth: int = 1,
        conv_depth: int = 2,
        ssm_d_state: int = 1,
        ssm_ratio: float = 1.0,
        ssm_dt_rank: Union[int, str] = 'auto',
        ssm_conv: int = 3,
        ssm_conv_bias: bool = False,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        forward_type: str = 'v05_noz',
    ) -> None:
        """Initialize the global fusion backend.

        Args:
            channels: Input channel count.
            global_fusion_type: Backend type, one of `'mamba'`, `'conv'`, or `'mamba_hilbert'`.
            mamba_depth: Number of global VSS blocks.
            conv_depth: Number of conv blocks when using conv backend.
            ssm_d_state: VSS state size.
            ssm_ratio: VSS inner ratio.
            ssm_dt_rank: VSS dt rank.
            ssm_conv: VSS depthwise conv kernel size.
            ssm_conv_bias: Whether VSS depthwise conv uses bias.
            mlp_ratio: VSS MLP ratio.
            mlp_drop_rate: VSS MLP dropout rate.
            forward_type: Existing VSS forward type to reuse.
        """
        super().__init__()
        self.global_fusion_type = global_fusion_type
        if global_fusion_type == 'mamba':
            self.backend = nn.Sequential(
                *[
                    VSSBlock(
                        hidden_dim=channels,
                        cross_dim=0,
                        drop_path=0.0,
                        norm_layer=LayerNorm2d,
                        channel_first=True,
                        ssm_d_state=ssm_d_state,
                        ssm_ratio=ssm_ratio,
                        ssm_dt_rank=ssm_dt_rank,
                        ssm_act_layer=nn.SiLU,
                        ssm_conv=ssm_conv,
                        ssm_conv_bias=ssm_conv_bias,
                        ssm_drop_rate=0.0,
                        ssm_init='v0',
                        forward_type=forward_type,
                        mlp_ratio=mlp_ratio,
                        mlp_act_layer=nn.GELU,
                        mlp_drop_rate=mlp_drop_rate,
                        gmlp=False,
                        use_checkpoint=False,
                    )
                    for _ in range(mamba_depth)
                ]
            )
        elif global_fusion_type == 'mamba_hilbert':
            self.backend = HilbertGlobalMambaBlock(
                channels=channels,
                depth=mamba_depth,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
            )
        elif global_fusion_type == 'conv':
            conv_blocks = []
            for _ in range(conv_depth):
                conv_blocks.append(DepthwiseSeparableConv(channels, channels, kernel_size=3, stride=1, padding=1))
            self.backend = nn.Sequential(*conv_blocks)
        else:
            raise ValueError(
                f"Unsupported global_fusion_type: {global_fusion_type}. "
                "Expected 'mamba', 'conv', or 'mamba_hilbert'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the configured global fusion backend.

        Args:
            x: Input feature map of shape `[B, C, H, W]`.

        Returns:
            Output feature map of shape `[B, C, H, W]`.
        """
        return self.backend(x)


class OffsetGuidedSelectiveHierarchicalMambaFusionBlock(nn.Module):
    """Fuse vehicle, RSU, and drone BEV features with selective hierarchical Mamba."""

    def __init__(
        self,
        channels: int,
        num_points: int = 4,
        offset_range: float = 2.0,
        transmission_alpha: float = 1.0,
        window_size: Union[int, Tuple[int, int]] = 4,
        align_corners: bool = False,
        padding_mode: str = 'zeros',
        local_mamba_depth: int = 1,
        global_fusion_type: str = 'mamba',
        global_mamba_depth: int = 1,
        global_conv_depth: int = 2,
        ssm_d_state: int = 1,
        ssm_ratio: float = 1.0,
        ssm_dt_rank: Union[int, str] = 'auto',
        ssm_conv: int = 3,
        ssm_conv_bias: bool = False,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        forward_type: str = 'v05_noz',
        sample_attention_type: str = 'aggregator',
        sample_cross_attention_dim: Optional[int] = None,
        sample_cross_attention_ffn_ratio: float = 2.0,
        sample_cross_attention_drop_rate: float = 0.0,
        sample_cross_attention_layer_scale_init: float = 0.0,
        return_debug: bool = False,
    ) -> None:
        """Initialize the selective hierarchical fusion block.

        Args:
            channels: Channel count of each aligned BEV feature map.
            num_points: Number of deformable sampling points per location.
            offset_range: Local offset range `r`.
            transmission_alpha: Exponential decay factor for external source gates.
            window_size: Local window size for local reorder and local Mamba.
            align_corners: Explicit `grid_sample` alignment choice.
            padding_mode: Explicit `grid_sample` padding mode shared by all sources.
            local_mamba_depth: Number of local VSS blocks.
            global_fusion_type: Global backend type, `'mamba'` or `'conv'`.
            global_mamba_depth: Number of global VSS blocks.
            global_conv_depth: Number of conv blocks in conv backend.
            ssm_d_state: Shared VSS state size.
            ssm_ratio: Shared VSS inner ratio.
            ssm_dt_rank: Shared VSS dt rank.
            ssm_conv: Shared VSS depthwise conv kernel size.
            ssm_conv_bias: Shared VSS depthwise conv bias flag.
            mlp_ratio: Shared VSS MLP ratio.
            mlp_drop_rate: Shared VSS MLP dropout rate.
            forward_type: Existing VSS forward type to reuse.
            sample_attention_type: Sample aggregation backend, `'aggregator'` or `'cross_attention'`.
            sample_cross_attention_dim: Query/key dimension used by `SampleCrossAttention`.
            sample_cross_attention_ffn_ratio: FFN expansion ratio used by `SampleCrossAttention`.
            sample_cross_attention_drop_rate: Dropout rate used by `SampleCrossAttention`.
            sample_cross_attention_layer_scale_init: Layer-scale init used by `SampleCrossAttention`.
            return_debug: Default debug return behavior.
        """
        super().__init__()
        self.return_debug = return_debug
        self.num_points = num_points
        self.sample_attention_type = str(sample_attention_type).lower()
        self.offset_predictors = nn.ModuleDict(
            {
                'vehicle': OffsetPredictor(channels, num_points, offset_range),
                'rsu': OffsetPredictor(channels, num_points, offset_range),
                'drone': OffsetPredictor(channels, num_points, offset_range),
            }
        )
        if self.sample_attention_type == 'aggregator':
            self.sample_attention = nn.ModuleDict(
                {
                    'vehicle': SampleAttentionAggregator(channels),
                    'rsu': SampleAttentionAggregator(channels),
                    'drone': SampleAttentionAggregator(channels),
                }
            )
        elif self.sample_attention_type == 'cross_attention':
            self.sample_attention = nn.ModuleDict(
                {
                    'vehicle': SampleCrossAttention(
                        channels=channels,
                        attention_dim=sample_cross_attention_dim,
                        ffn_ratio=sample_cross_attention_ffn_ratio,
                        drop_rate=sample_cross_attention_drop_rate,
                        layer_scale_init=sample_cross_attention_layer_scale_init,
                    ),
                    'rsu': SampleCrossAttention(
                        channels=channels,
                        attention_dim=sample_cross_attention_dim,
                        ffn_ratio=sample_cross_attention_ffn_ratio,
                        drop_rate=sample_cross_attention_drop_rate,
                        layer_scale_init=sample_cross_attention_layer_scale_init,
                    ),
                    'drone': SampleCrossAttention(
                        channels=channels,
                        attention_dim=sample_cross_attention_dim,
                        ffn_ratio=sample_cross_attention_ffn_ratio,
                        drop_rate=sample_cross_attention_drop_rate,
                        layer_scale_init=sample_cross_attention_layer_scale_init,
                    ),
                }
            )
        else:
            raise ValueError(
                f"Unsupported sample_attention_type: {sample_attention_type}. "
                "Expected 'aggregator' or 'cross_attention'."
            )
        self.sampler = MultiPointSampler(
            align_corners=align_corners,
            padding_mode=padding_mode,
        )
        self.transmission_gate = GeometricTransmissionGate(alpha=transmission_alpha)
        self.ego_fusion = EgoFusionLayer(channels=channels)
        self.local_mamba = LocalMambaBlock(
            channels=channels,
            window_size=window_size,
            depth=local_mamba_depth,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_conv=ssm_conv,
            ssm_conv_bias=ssm_conv_bias,
            mlp_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            forward_type=forward_type,
        )
        self.global_fusion = GlobalFusionBlock(
            channels=channels,
            global_fusion_type=global_fusion_type,
            mamba_depth=global_mamba_depth,
            conv_depth=global_conv_depth,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_conv=ssm_conv,
            ssm_conv_bias=ssm_conv_bias,
            mlp_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            forward_type=forward_type,
        )

    def _refine_source(
        self,
        source_name: str,
        feature: torch.Tensor,
        return_debug: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Run offset prediction, deformable sampling, and sample aggregation for one source."""
        offset_output = self.offset_predictors[source_name](feature, return_raw=return_debug)
        if return_debug:
            offsets, raw_offsets = offset_output
        else:
            offsets = offset_output
            raw_offsets = None

        sampled_features = self.sampler(feature, offsets)
        attention_module = self.sample_attention[source_name]
        if isinstance(attention_module, SampleCrossAttention):
            attention_weights, refined_feature = attention_module(feature, sampled_features)
        else:
            attention_weights, refined_feature = attention_module(sampled_features)
        debug_info: Dict[str, torch.Tensor] = {
            'offsets': offsets,
            'attention_maps': attention_weights,
        }
        if raw_offsets is not None:
            debug_info['raw_offsets'] = raw_offsets
        return refined_feature, offsets, attention_weights, debug_info

    def _build_missing_source_placeholders(
        self,
        feature_vehicle: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create final-fusion placeholders for a missing external source.

        Notes:
            Missing-source branch intermediates are represented as `None` in debug
            because the branch is not executed at all. In contrast, final ego fusion
            still requires fixed three-branch inputs, so we materialize zero tensors
            here right before `EgoFusionLayer`.

        Args:
            feature_vehicle: Vehicle feature used only for shape/device/dtype reference.

        Returns:
            A tuple containing:
                - zero refined feature `[B, C, H, W]`
                - zero gate `[B, 1, H, W]`
                - zero transmission distance `[B, 1, H, W]`
        """
        batch_size, _, height, width = feature_vehicle.shape
        zero_feature = torch.zeros_like(feature_vehicle)
        zero_gate = torch.zeros(
            (batch_size, 1, height, width),
            device=feature_vehicle.device,
            dtype=feature_vehicle.dtype,
        )
        zero_distance = torch.zeros_like(zero_gate)
        return zero_feature, zero_gate, zero_distance

    def _process_optional_external_source(
        self,
        source_name: str,
        feature: Optional[torch.Tensor],
        feature_vehicle: torch.Tensor,
        return_debug: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        """Process one optional external source or create final-fusion placeholders.

        Args:
            source_name: External source name, e.g. `'rsu'` or `'drone'`.
            feature: Optional external source feature `[B, C, H, W]`.
            feature_vehicle: Mandatory vehicle feature for shape reference.
            return_debug: Whether raw offsets should be collected.

        Returns:
            A tuple containing:
                - refined feature used by final fusion `[B, C, H, W]`
                - gate used by final fusion `[B, 1, H, W]`
                - transmission distance `[B, 1, H, W]`
                - debug dictionary
        """
        if feature is None:
            zero_feature, zero_gate, zero_distance = self._build_missing_source_placeholders(
                feature_vehicle
            )
            debug_info: Dict[str, Optional[torch.Tensor]] = {
                'offsets': None,
                'attention_maps': None,
                # These zero tensors are only final-fusion placeholders, not branch intermediates.
                'fusion_placeholder_feature': zero_feature,
                'fusion_placeholder_gate': zero_gate,
                'fusion_placeholder_distance': zero_distance,
            }
            if return_debug:
                debug_info['raw_offsets'] = None
            return zero_feature, zero_gate, zero_distance, debug_info

        refined_feature, offsets, attention_weights, source_debug = self._refine_source(
            source_name,
            feature,
            return_debug,
        )
        distance_map, gate_map = self.transmission_gate(offsets, attention_weights)
        source_debug['fusion_placeholder_feature'] = refined_feature
        source_debug['fusion_placeholder_gate'] = gate_map
        source_debug['fusion_placeholder_distance'] = distance_map
        return refined_feature, gate_map, distance_map, source_debug

    def forward(
        self,
        feature_vehicle: torch.Tensor,
        feature_rsu: Optional[torch.Tensor] = None,
        feature_drone: Optional[torch.Tensor] = None,
        return_debug: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Fuse three aligned BEV features.

        Args:
            feature_vehicle: Vehicle BEV feature of shape `[B, C, H, W]`.
            feature_rsu: Optional RSU BEV feature of shape `[B, C, H, W]`.
            feature_drone: Optional drone BEV feature of shape `[B, C, H, W]`.
            return_debug: Override the default debug return behavior.

        Returns:
            Final fused BEV feature of shape `[B, C, H, W]`.
            If debug is enabled, also returns a debug dictionary.
        """
        debug_flag = self.return_debug if return_debug is None else return_debug
        if feature_vehicle.dim() != 4:
            raise ValueError(
                "OffsetGuidedSelectiveHierarchicalMambaFusionBlock expects "
                f"`feature_vehicle` to be [B, C, H, W], got {feature_vehicle.shape}"
            )
        if feature_rsu is not None and feature_rsu.shape != feature_vehicle.shape:
            raise ValueError(
                "Non-None external inputs must match vehicle shape, got "
                f"vehicle={feature_vehicle.shape}, rsu={feature_rsu.shape}"
            )
        if feature_drone is not None and feature_drone.shape != feature_vehicle.shape:
            raise ValueError(
                "Non-None external inputs must match vehicle shape, got "
                f"vehicle={feature_vehicle.shape}, drone={feature_drone.shape}"
            )

        refined_vehicle, vehicle_offsets, attention_vehicle, debug_vehicle = self._refine_source(
            'vehicle', feature_vehicle, debug_flag
        )
        refined_rsu, gate_rsu, distance_rsu, debug_rsu = self._process_optional_external_source(
            'rsu',
            feature_rsu,
            feature_vehicle,
            debug_flag,
        )
        refined_drone, gate_drone, distance_drone, debug_drone = self._process_optional_external_source(
            'drone',
            feature_drone,
            feature_vehicle,
            debug_flag,
        )

        fused_feature = self.ego_fusion(
            refined_veh=refined_vehicle,
            refined_rsu=refined_rsu,
            refined_drone=refined_drone,
            gate_rsu=gate_rsu,
            gate_drone=gate_drone,
        )

        local_output = self.local_mamba(fused_feature, return_debug=debug_flag)
        if debug_flag:
            local_feature, local_debug = local_output
        else:
            local_feature = local_output
            local_debug = {}

        global_feature = self.global_fusion(local_feature)

        if not debug_flag:
            return global_feature

        debug_info = {
            'offsets': {
                'vehicle': vehicle_offsets,
                'rsu': debug_rsu['offsets'],
                'drone': debug_drone['offsets'],
            },
            'attention_maps': {
                'vehicle': debug_vehicle['attention_maps'],
                'rsu': debug_rsu['attention_maps'],
                'drone': debug_drone['attention_maps'],
            },
            'transmission_gates': {
                'rsu': gate_rsu,
                'drone': gate_drone,
            },
            'transmission_distances': {
                'rsu': distance_rsu,
                'drone': distance_drone,
            },
            'final_fusion_features': {
                'vehicle': refined_vehicle,
                'rsu': debug_rsu['fusion_placeholder_feature'],
                'drone': debug_drone['fusion_placeholder_feature'],
            },
            'final_fusion_gates': {
                'rsu': debug_rsu['fusion_placeholder_gate'],
                'drone': debug_drone['fusion_placeholder_gate'],
            },
            'reorder_scores': local_debug.get('reorder_scores'),
        }
        debug_info['raw_offsets'] = {
            'vehicle': debug_vehicle.get('raw_offsets'),
            'rsu': debug_rsu.get('raw_offsets'),
            'drone': debug_drone.get('raw_offsets'),
        }
        return global_feature, debug_info

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
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
class BatchCompressor(nn.Module):
    def __init__(self, in_channels = 80, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),  # 1x1卷积调整通道
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # 对每个batch分别处理
        features = []
        for i in range(B):
            feat = self.conv(x[i:i+1])  # [1, C', H, W]
            features.append(feat)
        
        # 堆叠并最大池化
        stacked = torch.cat(features, dim=0)  # [B, C', H, W]
        compressed = torch.max(stacked, dim=0, keepdim=True)[0]  # [1, C', H, W]
        
        return compressed


class BatchCompressorV2(nn.Module):
    """
    空间相关的 BEV 融合模块（支持动态 B）

    输入:
        x: [B, C, H, W]，这里的 B 是要融合的 BEV 数量（agent / 时刻），不固定
    输出:
        fused: [1, out_channels, H, W]

    设计:
        1. 共享特征编码: 对每个 BEV 通过同一个卷积块提取中间特征 feat
        2. 空间相关权重: 用 feat 生成每个 (b, h, w) 的标量 logits，并在 B 维做 softmax 得到 alpha
        3. 加权融合: 在 B 维按 alpha 对 feat 做加权和，得到 fused，再用 1x1 conv 调整通道
    """

    def __init__(self, in_channels: int = 80, mid_channels: int = None, out_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = mid_channels

        # Step 1: 共享特征编码 φ
        self.feat_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # Step 2: 生成空间相关 logits（每个 BEV、每个像素一个 score）
        self.score_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True),  # 输出 [B,1,H,W]
        )

        # Step 3: 融合后通道调整
        self.out_proj = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]，B 不固定
        if x.dim() != 4:
            raise ValueError(f"BatchCompressorV2 expects 4D input [B,C,H,W], got shape {x.shape}")

        B, C, H, W = x.shape
        if B == 1:
            # 只有一个 BEV 时，直接做特征编码和投影，避免数值不稳定的 softmax
            feat_single = self.feat_conv(x)
            return self.out_proj(feat_single)

        # 1) 特征编码
        feat = self.feat_conv(x)  # [B, C_mid, H, W]

        # 2) 空间相关权重: logits -> softmax over B 维
        logits = self.score_conv(feat)              # [B, 1, H, W]
        alpha = torch.softmax(logits, dim=0)        # [B, 1, H, W]，对每个 (h,w) 在 B 上归一化

        # 3) 按权重融合
        fused = (alpha * feat).sum(dim=0, keepdim=True)  # [1, C_mid, H, W]

        # 4) 通道调整
        fused = self.out_proj(fused)  # [1, out_channels, H, W]
        return fused