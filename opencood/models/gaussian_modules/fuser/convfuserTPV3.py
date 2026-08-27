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
    'AGENT_TYPES': ['vehicle', 'rsu', 'drone']
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

class ConvFuserTPV3(nn.Module):
    def __init__(self, model_cfg) -> None:
        super().__init__()
        
        self.model_cfg = model_cfg
        # 如果model_cfg为None，使用默认配置
        if model_cfg is None:
            import copy
            model_cfg = copy.deepcopy(DEFAULT_MODEL_CFG)
        
        in_channel = self.model_cfg.IN_CHANNEL  #256
        out_channel = self.model_cfg.OUT_CHANNEL   #128
        self.merge_type = self.model_cfg.get('MERGE_TYPE', 'default')

        # Create separate parameter sets for xy, xz, and yz planes
        self.planes = ['xy', 'xz', 'yz']

        # Initialize plane-specific final conv layers
        self.conv = nn.ModuleDict()
        for plane in self.planes:
            if self.merge_type == 'default':
                self.conv[plane] = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(True)
                )
            else:
                self.conv[plane] = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel * 2, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channel * 2),
                    nn.ReLU(),
                    nn.Conv2d(out_channel * 2, out_channel, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(),
                )
        
        self.use_vmamba = model_cfg.get('USE_VMAMBA', False)
        self.use_checkpoint = model_cfg.get('USE_CHECKPOINT', True)
        self.use_merge_after = model_cfg.get('USE_MERGE_AFTER', False)
        
        if self.use_merge_after:
            self.merge_blocks = nn.ModuleDict()
            depths = [1]
            merge_dim = 256
            dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]
            
            for plane in self.planes:
                merge_blocks_list = []
                for i_layer in range(len(depths)):
                    merge_blocks_list.append(self._make_vmamba_layer(
                        dim=merge_dim,
                        drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                        use_checkpoint=False,
                        norm_layer=LayerNorm2d,
                        downsample=nn.Identity(),
                        channel_first=True,
                        ssm_d_state=1,
                        ssm_ratio=1.0,
                        ssm_dt_rank='auto',
                        ssm_act_layer=nn.SiLU,
                        ssm_conv=3,
                        ssm_conv_bias=False,
                        ssm_drop_rate=0.0,
                        ssm_init='v0',
                        forward_type='v05_noz',
                        mlp_ratio=4.0,
                        mlp_act_layer=nn.GELU,
                        mlp_drop_rate=0.0,
                        gmlp=False,
                    ))
                self.merge_blocks[plane] = nn.ModuleList(merge_blocks_list)

        if self.use_vmamba:
            self.use_dw_conv = True
            depths = [1, 1, 1]
            image_dim = 128
            point_dim = 128
            cross_dim = 128
            ssm_conv = 3
            max_channel = 1
            d_state = 1
            
            # Create separate ModuleDicts for each type of block
            self.image_down_blocks = nn.ModuleDict()
            self.image_de_blocks = nn.ModuleDict()
            self.lidar_down_blocks = nn.ModuleDict()
            self.lidar_de_blocks = nn.ModuleDict()
            self.image_vmamba_blocks = nn.ModuleDict()
            self.point_vmamba_blocks = nn.ModuleDict()
            
            # Initialize conv layers for each plane
            self.image_conv = nn.ModuleDict()
            self.lidar_conv = nn.ModuleDict()
            
            for plane in self.planes:
                # Initialize ModuleLists for each plane
                self.image_down_blocks[plane] = nn.ModuleList()
                self.image_de_blocks[plane] = nn.ModuleList()
                self.lidar_down_blocks[plane] = nn.ModuleList()
                self.lidar_de_blocks[plane] = nn.ModuleList()
                self.image_vmamba_blocks[plane] = nn.ModuleList()
                self.point_vmamba_blocks[plane] = nn.ModuleList()
                
                # Initialize conv layers for each plane
                self.image_conv[plane] = nn.Sequential(
                    nn.Conv2d(image_dim * (len(depths) + 1), image_dim * 2, 3, padding=1, bias=False),
                    nn.BatchNorm2d(image_dim * 2),
                    nn.ReLU(),
                    DepthwiseSeparableConv(image_dim * 2, image_dim, 3, 1, 1),
                )
                self.lidar_conv[plane] = nn.Sequential(
                    nn.Conv2d(point_dim * (len(depths) + 1), point_dim * 2, 3, padding=1, bias=False),
                    nn.BatchNorm2d(point_dim * 2),
                    nn.ReLU(),
                    DepthwiseSeparableConv(point_dim * 2, point_dim, 3, 1, 1),
                )

            dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]

            # Initialize blocks for each plane
            for plane in self.planes:
                for i_layer in range(len(depths)):
                    # Down blocks
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
                        point_cur_layers = [
                            BasicBlock(point_dim*min(i_layer + 1, max_channel), point_dim*min(i_layer + 2, max_channel), 2, 1, True),
                        ]

                    self.image_down_blocks[plane].append(nn.Sequential(*image_cur_layers))
                    self.lidar_down_blocks[plane].append(nn.Sequential(*point_cur_layers))

                    # Deconvolution blocks
                    image_cur_de_layers = []
                    point_cur_de_layers = []
                    for j in range(i_layer + 1):
                        image_cur_de_layers.append(nn.ConvTranspose2d(image_dim*min(i_layer + 2 - j, max_channel), image_dim*min(i_layer + 1 - j, max_channel), kernel_size=2, stride=2, bias=False))
                        image_cur_de_layers.append(nn.BatchNorm2d(image_dim*min(i_layer + 1 - j, max_channel)))
                        image_cur_de_layers.append(nn.ReLU())
                        point_cur_de_layers.append(nn.ConvTranspose2d(point_dim*min(i_layer + 2 - j, max_channel), point_dim*min(i_layer + 1 - j, max_channel), kernel_size=2, stride=2, bias=False))
                        point_cur_de_layers.append(nn.BatchNorm2d(point_dim*min(i_layer + 1 - j, max_channel)))
                        point_cur_de_layers.append(nn.ReLU())
                        if self.use_dw_conv:
                            image_cur_de_layers.append(DepthwiseSeparableConv(image_dim*min(i_layer + 1 - j, max_channel), image_dim*min(i_layer + 1 - j, max_channel), 3, 1, 1))
                            point_cur_de_layers.append(DepthwiseSeparableConv(point_dim*min(i_layer + 1 - j, max_channel), point_dim*min(i_layer + 1 - j, max_channel), 3, 1, 1))

                    self.image_de_blocks[plane].append(nn.Sequential(*image_cur_de_layers))
                    self.lidar_de_blocks[plane].append(nn.Sequential(*point_cur_de_layers))

                    # Vmamba blocks
                    self.image_vmamba_blocks[plane].append(self._make_vmamba_layer(
                        dim=image_dim*min(i_layer + 2, max_channel),
                        drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                        use_checkpoint=False,
                        norm_layer=LayerNorm2d,
                        downsample=nn.Identity(),
                        channel_first=True,
                        ssm_d_state=d_state,
                        ssm_ratio=1.0,
                        ssm_dt_rank='auto',
                        ssm_act_layer=nn.SiLU,
                        ssm_conv=ssm_conv,
                        ssm_conv_bias=False,
                        ssm_drop_rate=0.0,
                        ssm_init='v0',
                        forward_type='v05_noz',
                        mlp_ratio=4.0,
                        mlp_act_layer=nn.GELU,
                        mlp_drop_rate=0.0,
                        gmlp=False,
                    ))

                    self.point_vmamba_blocks[plane].append(self._make_vmamba_layer(
                        dim=point_dim*min(i_layer + 2, max_channel),
                        drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                        use_checkpoint=False,
                        norm_layer=LayerNorm2d,
                        downsample=nn.Identity(),
                        channel_first=True,
                        ssm_d_state=d_state,
                        ssm_ratio=1.0,
                        ssm_dt_rank='auto',
                        ssm_act_layer=nn.SiLU,
                        ssm_conv=ssm_conv,
                        ssm_conv_bias=False,
                        ssm_drop_rate=0.0,
                        ssm_init='v0',
                        forward_type='v05_noz',
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

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                image_tpv_xy, image_tpv_xz, image_tpv_yz: TPV features from image modality
                tpv_xy, tpv_xz, tpv_yz: TPV features from lidar modality

        Returns:
            batch_dict: Updated with fused TPV features
        """
        plane_inputs = {
            'xy': (batch_dict['vehicle']['image_tpv_xy'], batch_dict['vehicle']['tpv_xy']),
            'xz': (batch_dict['vehicle']['image_tpv_xz'], batch_dict['vehicle']['tpv_xz']),
            'yz': (batch_dict['vehicle']['image_tpv_yz'], batch_dict['vehicle']['tpv_yz'])
        }
        
        results = {}
        for plane, (img_tpv, lidar_tpv) in plane_inputs.items():
            if self.use_vmamba:
                if self.use_checkpoint:
                    cat_features = checkpoint.checkpoint(self.mamba_forward, img_tpv, lidar_tpv, plane)
                else:
                    cat_features = self.mamba_forward(img_tpv, lidar_tpv, plane)
            else:
                cat_features = torch.cat([img_tpv, lidar_tpv], dim=1)
            
            if self.use_merge_after:
                for block in self.merge_blocks[plane]:
                    cat_features = block(cat_features)
            
            results[plane] = self.conv[plane](cat_features)

        batch_dict['vehicle']['fused_tpv_xy'] = results['xy']
        batch_dict['vehicle']['fused_tpv_xz'] = results['xz']
        batch_dict['vehicle']['fused_tpv_yz'] = results['yz']

        return batch_dict
    
    def mamba_forward(self, img_tpv, lidar_tpv, plane):
        ups_img = []
        ups_img.append(img_tpv)
        ups_lidar = []
        ups_lidar.append(lidar_tpv)

        for i, (block_img, block_lidar) in enumerate(zip(self.image_vmamba_blocks[plane], self.point_vmamba_blocks[plane])):
            img_tpv = self.image_down_blocks[plane][i](img_tpv)
            img_tpv = block_img(img_tpv)
            lidar_tpv = self.lidar_down_blocks[plane][i](lidar_tpv)
            lidar_tpv = block_lidar(lidar_tpv)

            ups_img.append(self.image_de_blocks[plane][i](img_tpv))
            ups_lidar.append(self.lidar_de_blocks[plane][i](lidar_tpv))

        merge_img = self.image_conv[plane](torch.cat(ups_img, dim=1))
        merge_lidar = self.lidar_conv[plane](torch.cat(ups_lidar, dim=1))
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