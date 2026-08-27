import torch
from torch import nn

from ...vmamba.vmamba import SS2D, VSSBlock, Linear2d, LayerNorm2d
from collections import OrderedDict
from ..base_bev_backbone import BasicBlock
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
        import pdb; pdb.set_trace()
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