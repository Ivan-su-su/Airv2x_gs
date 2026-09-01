'''
Fuse TPV features from image and lidar modalities using convolutional layers.
'''

from torch import nn
import torch
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from collections import OrderedDict
from opencood.models.gaussian_modules.fuser.vmamba.vmamba import SS2D, VSSBlock, Linear2d, LayerNorm2d

# 默认配置模板
DEFAULT_MODEL_CFG = {
    'IN_CHANNEL': 256,
    'OUT_CHANNEL': 128,
    'AGENT_TYPES': ['vehicle', 'rsu', 'drone'],
    'MERGE_TYPE': 'default',
    'USE_VMAMBA': True,
    'USE_CHECKPOINT': True,
    'USE_MERGE_AFTER': False
}


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        padding: int = 1,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu2 = nn.ReLU()
        self.downsample = downsample
        if self.downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
            )
        self.stride = stride

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample_layer(x)

        out += identity
        out = self.relu2(out)

        return out


class ConvFuserTPV1(nn.Module):
    def __init__(self, model_cfg) -> None:
        super().__init__()
        
        self.model_cfg = model_cfg
        # 如果model_cfg为None，使用默认配置
        if model_cfg is None:
            import copy
            model_cfg = copy.deepcopy(DEFAULT_MODEL_CFG)
        
        in_channel = self.model_cfg.IN_CHANNEL  #256
        out_channel = self.model_cfg.OUT_CHANNEL   #128
        self.agent_types = model_cfg.get('AGENT_TYPES', ['vehicle', 'rsu', 'drone'])
        self.merge_type = self.model_cfg.get('MERGE_TYPE', 'default')

        if self.merge_type == 'default':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True)
                )
        else:
            self.conv = nn.Sequential(
                # DepthwiseSeparableConv(in_channel, in_channel, 3, 1, 1),
                nn.Conv2d(in_channel, out_channel * 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel * 2),
                nn.ReLU(),
                nn.Conv2d(out_channel * 2, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
                )
        
        # USE_VMAMBA: 是否使用基于 VSSBlock 的复杂融合器（更强但更慢）
        # USE_CHECKPOINT: 是否对 mamba_forward 使用 checkpoint 以节省显存（会影响反向传播效率）
        # USE_MERGE_AFTER: 是否在拼接后额外再使用一组 merge_blocks 进行处理
        self.use_vmamba = model_cfg.get('USE_VMAMBA', True)   #True
        self.use_checkpoint = model_cfg.get('USE_CHECKPOINT', True)   #True
        self.use_merge_after = model_cfg.get('USE_MERGE_AFTER', False)
        self.downsample = model_cfg.get('DOWNSAMPLE', 2)
        self.use_shrink = self.downsample is not None and self.downsample > 1
        if self.use_shrink:
            self.shrink_xy = self._build_shrink_block(out_channel)
            self.shrink_xz = self._build_shrink_block(out_channel)
            self.shrink_yz = self._build_shrink_block(out_channel)
        else:
            self.shrink_xy = nn.Identity()
            self.shrink_xz = nn.Identity()
            self.shrink_yz = nn.Identity()
        if self.use_merge_after:
            depths = self.model_cfg.get('DEPTHS', [1])
            if isinstance(depths, list) and len(depths) > 0:
                depths = depths[:1]  # 只取第一个用于 merge_after
            else:
                depths = [1]
            num_block = len(depths)
            merge_dim = self.model_cfg.get('MERGE_DIM', 256)
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
            # 当启用 vmamba 分支时，会构建图像（image_）和点云（lidar_/point_）两条支路的下采样/上采样模块
            # 以及在每个尺度上使用 VSSBlock（通过 _make_vmamba_layer 创建）来进行时序 / 空间混合。
            # 变量说明：
            # image_dim: 图像分支内通道数（特征维度）
            # point_dim: 点云分支内通道数
            # cross_dim: 跨模态交互时的中间维度
            # self.img_pos_embed_layer = PositionEmbeddingLearned(20, 128)
            # self.lidar_pos_embed_layer = PositionEmbeddingLearned(3, 128)
            self.use_dw_conv = True
            depths = self.model_cfg.get('DEPTHS', [1, 1, 1])
            num_block = len(depths)
            image_dim = self.model_cfg.get('IMAGE_DIM', 128)
            point_dim = self.model_cfg.get('POINT_DIM', 128)
            cross_dim = self.model_cfg.get('CROSS_DIM', 128)
            ssm_conv = self.model_cfg.get('SSM_CONV', 3)
            max_channel = self.model_cfg.get('MAX_CHANNEL', 1)
            use_4x = False
            self.use_cross = False
            self.use_res_merge = False
            d_state = self.model_cfg.get('D_STATE', 1)
            self.image_down_blocks = nn.ModuleList()
            self.image_de_blocks = nn.ModuleList()
            self.lidar_de_blocks = nn.ModuleList()
            self.lidar_down_blocks = nn.ModuleList()

            if self.use_res_merge:   #False
                self.image_norm = nn.ModuleList()
                self.point_norm = nn.ModuleList()

            self.image_vmamba_blocks = nn.ModuleList()
            self.point_vmamba_blocks = nn.ModuleList()
            num_block_cross = 0

            if self.use_cross:   #False
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
            if not self.use_res_merge:   #True
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
                if self.use_res_merge:   #False
                    self.image_norm.append(nn.BatchNorm2d(image_dim))
                    self.point_norm.append(nn.BatchNorm2d(point_dim))

                # if i_layer == 0 and use_4x:
                #     point_cur_layers.append(BasicBlock(point_dim, point_dim, 2, 1, True))
                if self.use_dw_conv:   #True
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
                    if self.use_dw_conv:   #True
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


    # _make_vmamba_layer: 工厂函数，用于构建一组 VSSBlock（或双支路 blocks1/blocks2），
    # 返回一个 nn.Sequential 包含 blocks（序列）及可选的 downsample/其它子模块。
    @staticmethod
    def _make_vmamba_layer(
        dim=96,
        cross_dim=0,
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        channel_first=False,
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        cross=False,
        **kwargs,
    ):
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

    def _build_shrink_block(self, channels: int) -> nn.Module:
        """构建用于TPV平面压缩的卷积块"""
        if not self.use_shrink:
            return nn.Identity()
        return nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=self.downsample,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )


    def forward(self, batch_dict, available_agents):
        """
        Args:
            batch_dict:
                image_tpv_xy, image_tpv_xz, image_tpv_yz: TPV features from image modality
                tpv_xy, tpv_xz, tpv_yz: TPV features from lidar modality

        Returns:
            batch_dict: Updated with fused TPV features
        """
        #TODO: not robust, 如果TPV形状非偶数会出问题
        import pdb; pdb.set_trace()
        # Get image TPV features
        for agent_type in available_agents:
            img_xy = batch_dict[agent_type]['image_tpv_xy']   #[B, 128, 200, 704]
            img_xz = batch_dict[agent_type]['image_tpv_xz']   #[B, 128, 704, 32]
            img_yz = batch_dict[agent_type]['image_tpv_yz']   #[B, 128, 200, 32]

            # Get lidar TPV features
            lidar_xy = batch_dict[agent_type]['tpv_xy']   #[B, 128, 200, 704]
            lidar_xz = batch_dict[agent_type]['tpv_xz']   #[B, 128, 704, 32]
            lidar_yz = batch_dict[agent_type]['tpv_yz']   #[B, 128, 200, 32]

            if self.use_vmamba:
                if self.use_checkpoint:   #False
                    cat_xy = checkpoint.checkpoint(self.mamba_forward, img_xy, lidar_xy)
                    cat_xz = checkpoint.checkpoint(self.mamba_forward, img_xz, lidar_xz)
                    cat_yz = checkpoint.checkpoint(self.mamba_forward, img_yz, lidar_yz)
                else:
                    cat_xy = self.mamba_forward(img_xy, lidar_xy)
                    cat_xz = self.mamba_forward(img_xz, lidar_xz)
                    cat_yz = self.mamba_forward(img_yz, lidar_yz)
            else:
                cat_xy = torch.cat([img_xy, lidar_xy], dim=1)
                cat_xz = torch.cat([img_xz, lidar_xz], dim=1)
                cat_yz = torch.cat([img_yz, lidar_yz], dim=1)
            
            if self.use_merge_after:   #False
                for block in self.merge_blocks:
                    cat_xy = block(cat_xy)
                    cat_xz = block(cat_xz)
                    cat_yz = block(cat_yz)
            
            final_xy = self.conv(cat_xy)
            final_xz = self.conv(cat_xz)
            final_yz = self.conv(cat_yz)

            final_xy = self.shrink_xy(final_xy)
            final_xz = self.shrink_xz(final_xz)
            final_yz = self.shrink_yz(final_yz)

            batch_dict[agent_type]['fused_tpv_xy'] = final_xy
            batch_dict[agent_type]['fused_tpv_xz'] = final_xz
            batch_dict[agent_type]['fused_tpv_yz'] = final_yz
            # Pop原始TPV特征，保留融合后的fused_tpv_xy/xz/yz
            batch_dict[agent_type].pop('image_tpv_xy', None)
            batch_dict[agent_type].pop('image_tpv_xz', None)
            batch_dict[agent_type].pop('image_tpv_yz', None)
            batch_dict[agent_type].pop('tpv_xy', None)
            batch_dict[agent_type].pop('tpv_xz', None)
            batch_dict[agent_type].pop('tpv_yz', None)
            print("TPV Fusion Done for ", agent_type)
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
            #     },
            #     ...,
            # }

        return batch_dict
    

    def mamba_forward(self, img_tpv, lidar_tpv):
        # make sure triton kernels see GPU tensors
        target_device = next(self.parameters()).device
        if img_tpv.device != target_device:
            img_tpv = img_tpv.to(target_device, non_blocking=True)
        if lidar_tpv.device != target_device:
            lidar_tpv = lidar_tpv.to(target_device, non_blocking=True)
        # Triton kernels rely on current CUDA device; align it with tensors.
        if target_device.type == 'cuda':
            torch.cuda.set_device(target_device.index if target_device.index is not None else 0)

        ups_img = []
        ups_img.append(img_tpv)
        ups_lidar = []
        ups_lidar.append(lidar_tpv)
        # mamba_forward: 对图像和点云分支做多尺度下采样/处理，保存上采样分支以便多尺度融合
        # 每一步循环：先下采样（image_down_blocks / lidar_down_blocks），再用 VSSBlock 序列处理
        for i, (block_img, block_lidar) in enumerate(zip(self.image_vmamba_blocks, self.point_vmamba_blocks)):
            img_tpv = self.image_down_blocks[i](img_tpv) # [2, 80, 90, 90]
            # 下采样：例如 H,W -> H/2, W/2
            img_tpv = block_img(img_tpv)
            lidar_tpv = self.lidar_down_blocks[i](lidar_tpv)
            lidar_tpv = block_lidar(lidar_tpv)
            if self.use_cross:   #if use vmamba, False
                img_tpv = self.image_up_blocks[i](img_tpv) # [batch_size, 128, 180, 180]
                img_tpv_cross = self.cross_vmamba_blocks[i].blocks1((img_tpv, lidar_tpv)) # [batch_size, 128, 180, 180]
                lidar_tpv_cross = self.cross_vmamba_blocks[i].blocks2((lidar_tpv, img_tpv))
                if not self.use_res_merge:
                    img_tpv = self.image_cross_blocks[i](torch.cat([img_tpv, lidar_tpv_cross], dim=1)) # [batch_size, 128, 180, 180]
                    lidar_tpv = self.point_cross_blocks[i](torch.cat([lidar_tpv, img_tpv_cross], dim=1)) # [batch_size, 128, 180, 180]
                else:
                    img_tpv = img_tpv_cross
                    lidar_tpv = lidar_tpv_cross
            if self.use_res_merge:   #if use vmamba, False
                # 如果使用残差式融合，直接在当前尺度做残差连接（原特征 + 上采样(降采样再上采样)）
                img_tpv = self.image_norm[i](img_tpv + self.image_de_blocks[i](img_tpv))
                lidar_tpv = self.point_norm[i](lidar_tpv + self.lidar_de_blocks[i](lidar_tpv))
            else:
                # 否则将当前尺度上采样后的特征保存到列表，后续用 concat 融合多尺度信息
                ups_img.append(self.image_de_blocks[i](img_tpv))
                ups_lidar.append(self.lidar_de_blocks[i](lidar_tpv))
        if self.use_res_merge:   #if use vmamba, False
            merge_img = img_tpv
            merge_lidar = lidar_tpv
        else:
            merge_img = self.image_conv(torch.cat(ups_img, dim=1)) # [2, 80, 360, 360]
            merge_lidar = self.lidar_conv(torch.cat(ups_lidar, dim=1)) # [2, 128, 360, 360]
            # 将多尺度的上采样特征在通道维度拼接，再用 conv 进一步融合并恢复到目标尺寸
        cat_bev = torch.cat([merge_img, merge_lidar], dim=1)

        return cat_bev


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                  stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                  stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Depthwise separable conv: 先逐通道的 depthwise，再用 1x1 的 pointwise 做通道混合
        # 可以大幅减少参数与计算量，常用于轻量化卷积块
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x