import copy
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import numpy as np
import os
import cv2

from ..model_utils.swin_utils import PatchEmbed
from ..model_utils.unitr_utils_old import MapImage2Lidar, MapLidar2Image
from ..model_utils.dsvt_utils import PositionEmbeddingLearned
from ..backbones_3d.dsvt import _get_activation_fn, DSVTInputLayer
from ..ops.ingroup_inds.ingroup_inds_op import ingroup_inds
from ..vmamba.vmamba import SS2D, VSSBlock
from collections import OrderedDict
from torchvision.ops import DeformConv2d
get_inner_win_inds_cuda = ingroup_inds
import torch.nn.functional as F
from ..vmamba.vmamba import SS2D, VSSBlock, Linear2d, LayerNorm2d
from ..backbones_image.swin import SwinTransformer
from ..vmamba.vmamba import Backbone_VSSM
from mamba_ssm.models.mixer_seq_simple import create_block
from ..backbones_3d.local_mamba import GlobalMamba
from functools import partial
from ..spconv_utils import replace_feature, spconv


from ..backbones_3d.lion_backbone_one_stride import LocalMamba
from mamba_ssm import Block2 as MambaBlock
from easydict import EasyDict

from ..backbones_image.img_neck.generalized_lss import MY_FPN

class MambaFusion(nn.Module):
    def __init__(self, model_cfg, use_map=False, **kwargs):
        super().__init__()
        self.use_cross_img = model_cfg.get('USE_CROSS_IMG', False)
        self.use_winmamba = model_cfg.get('USE_WINMAMBA', False)
        # self.use_mamba2 = model_cfg.get('USE_MAMBA2', False)
        self.use_vmamba_pretrain = model_cfg.get('USE_VMAMBA_PRETRAIN', False)
        self.use_prebackbone = model_cfg.get('USE_PREBACKBONE', False)
        self.use_mamba_inter = model_cfg.get('USE_MAMBA_INTER', False)
        self.use_mamba_inter2 = model_cfg.get('USE_MAMBA_INTER2', False)
        self.use_checkpoint_inter = model_cfg.get('USE_CHECKPOINT_INTER', False)
        self.use_checkpoint_inter2 = model_cfg.get('USE_CHECKPOINT_INTER2', False)
        self.mixed_version = model_cfg.get('MIXED_VERSION', 0)
        self.image_shape = model_cfg.get('IMAGE_SHAPE', [256, 704])
        self.use_more_res = model_cfg.get('USE_MORE_RES', False)
        if self.use_more_res:
            res_len = len(model_cfg.out_indices) - 1
            self.res_blocks_lidar = nn.ModuleList()
            self.res_blocks_camera = nn.ModuleList()
            for i in range(res_len):
                self.res_blocks_lidar.append(nn.ModuleList([nn.LayerNorm(128),nn.LayerNorm(128)]))
                self.res_blocks_camera.append(nn.ModuleList([nn.LayerNorm(128),nn.LayerNorm(128)]))

        self.inter2_use_expand = model_cfg.get('INTER2_USE_EXPAND', False)
        self.use_fixed_mapping = model_cfg.get('USE_FIXED_MAPPING', False)
        self.use_inverse = model_cfg.get('USE_INVERSE', False)
        self.use_checkpoint_global = model_cfg.get('USE_CHECKPOINT_GLOBAL', True)
        self.use_checkpoint_local = model_cfg.get('USE_CHECKPOINT_LOCAL', True)
        # self.inter2_down_scales = model_cfg.get('INTER2_DOWN_SCALES', [[2, 2, 1], [2, 2, 1]])
        self.use_down_scale_inter2 = model_cfg.get('USE_DOWN_SCALE_INTER2', True)
        self.use_all_mamba = model_cfg.get('USE_ALL_MAMBA', False)
        self.mask_out_img = model_cfg.get('MASK_OUT_IMG', False)
        self.use_shift = model_cfg.get('USE_SHIFT', True)


        # if self.use_mamba2:
        #     self.winmamba_info = {'NAME': 'Mamba2', 'CFG': {'d_state': 128, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}
        #     self.ssm_cfg = {'layer': 'Mamba2', 'd_state': 128, 'd_conv': 4, 'expand': 2, 'headdim': 32}
        # else:
        self.localmamba_info = {'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}
        self.ssm_cfg = None
        if self.use_down_scale_inter2:
            self.inter2_down_scales = [[2, 2, 1], [2, 2, 1]]
            self.inter2_down_scales_global = [1, 2]
        else:
            self.inter2_down_scales = model_cfg.get('INTER2_DOWN_SCALES', [[1, 1, 1], [1, 1, 1]])
            self.inter2_down_scales_global = [1, 1]
        if self.use_fixed_mapping:
            assert max(max(self.inter2_down_scales)) == 1, 'Fixed mapping only support downscale 1'
        self.inter1_win_shape = model_cfg.get('INTER1_WIN_SHAPE', [13, 13, 1])
        self.inter1_win_size = model_cfg.get('INTER1_WIN_SIZE', 256)
        self.inter2_win_shape = model_cfg.get('INTER2_WIN_SHAPE', [30, 30, 1])
        self.inter2_win_size = model_cfg.get('INTER2_WIN_SIZE', 90)

        self.win_version = model_cfg.get('WIN_VERSION', 'v2')
        if self.image_shape != [256, 704]:
            model_cfg.PATCH_EMBED.image_size = self.image_shape
            model_cfg.IMAGE_INPUT_LAYER.sparse_shape = [int(self.image_shape[0] / 8), int(self.image_shape[1] / 8), 1]
            model_cfg.FUSE_BACKBONE.LIDAR2IMAGE.lidar2image_layer.sparse_shape = [int(self.image_shape[0] / 8 * 3), int(self.image_shape[1] / 8 * 3), 6]
        
        self.use_mixed_scale = model_cfg.get('USE_MIXED_SCALE', False)
        self.use_multi_scale = model_cfg.get('USE_MULTI_SCALE', False)
        self.use_multi_scalev = model_cfg.get('USE_MULTI_SCALEV', False)
        self.use_multi_scalev_down = model_cfg.get('USE_MULTI_SCALEV_DOWN', False)
        self.use_denoise = model_cfg.get('USE_DENOISE', False)
        self.use_profiler = model_cfg.get('USE_PROFILER', False)
        self.return_abs_coords = model_cfg.get('RETURN_ABS_COORDS', False)
        self.use_more_vbackbone = model_cfg.get('USE_MORE_VBACKBONE', False)
        
        # 可视化保存目录（写死）
        self.vis_save_dir = './vis_lidar2image'
        self.vis_iter_count = 0

        self.use_mixed = model_cfg.get('USE_MIXED', False)
        if self.use_vmamba_pretrain and not self.use_prebackbone:
            if self.use_more_vbackbone:
                depths = (2, 2, 15)
                out_indices = [1, 2]
                fpn_model_cfg = EasyDict({
                    'IN_CHANNELS': [256, 512],  # 输入的特征图通道数
                    'OUT_CHANNELS': 256,  # 期望的输出通道数
                    'NUM_OUTS': 2,  # 生成的FPN层数
                    'START_LEVEL': 0,
                    'END_LEVEL': -1,  # 使用所有的特征层
                    'USE_BIAS': False,
                    'ALIGN_CORNERS': False
                })
                self.vssm_fpn = MY_FPN(fpn_model_cfg)
            else:
                depths = (2, 2)
                out_indices = [1] if not self.use_multi_scalev else [0, 1]
            if self.use_multi_scalev_down:
                depths = (2, 2, 2)
                out_indices += [2]
                self.vssm_multi_scalev_down_block = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0)
            args = {
                'norm_layer': 'ln2d',
                'patch_size': 4,
                'in_chans': 3,
                'depths': depths, # , 15, 2
                'out_indices': out_indices,
                'dims': 128,
                'ssm_d_state': 1,
                'ssm_conv_bias':False,
                'forward_type': 'v05_noz',
                'drop_path_rate': 0.2,
                'downsample_version': 'v3',
                'patchembed_version': 'v2',
            }
            # import pickle
            # with open('swin_model_cfg.pkl', 'rb') as file:
            #     swin_model_cfg = pickle.load(file)
            self.backbone_vssm = Backbone_VSSM(**args)
            # self.backbone_vssm = SwinTransformer(swin_model_cfg)
            # self.vssm_down_block = nn.Conv2d(in_channels=1344, out_channels=128, kernel_size=1, stride=1, padding=0)
            # self.backbone_vssm.register_forward_pre_hook(forward_hook_start)
            # self.backbone_vssm.register_forward_hook(forward_hook_end)
            # self.backbone_vssm.register_backward_hook(backward_hook_start)
            # self.backbone_vssm.register_backward_hook(backward_hook_end)
            
            self.vssm_down_block = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
            # self.remove_layers(self.backbone_vssm, ['outnorm2', 'outnorm3'])
            # self.remove_layers(self.backbone_vssm.layers, ['2', '3'])
        

        self.use_vmamba = model_cfg.get('USE_VMAMBA', False)
        if self.use_vmamba:
            self.img_pos_embed_layer = PositionEmbeddingLearned(20, 128)
            self.lidar_pos_embed_layer = PositionEmbeddingLearned(3, 128)

            self.vmamba_blocks = nn.ModuleList()
            depths = [2, 2]
            num_block = len(depths)
            twin_flag = [True, True]
            self.twin_flag = twin_flag
            assert len(twin_flag) == num_block, 'The length of twin_flag should be equal to num_block'
            dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))] 
            self.use_in_mid = False
            self.use_after = True
            self.use_conv = True

            self.lidar_fc = nn.Sequential(
                nn.Linear(128*(1+num_block), 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
            )

            if self.use_conv:
                for i in range(len(depths)):
                    layer = nn.LayerNorm(128)
                    layer_name = f'out_norm{i + 4}'
                    self.add_module(layer_name, layer)

                self.img_fc = nn.Sequential(
                    nn.Conv2d(in_channels=128*(1+num_block), out_channels=512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    )
            else:
                self.img_fc = nn.Sequential(
                    nn.Linear(128*3, 128),
                    nn.LayerNorm(128),
                    nn.ReLU(),
                )

            for i_layer in range(num_block):
                self.vmamba_blocks.append(self._make_vmamba_layer(
                    dim = 128,
                    drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    use_checkpoint=False,
                    norm_layer=nn.LayerNorm,
                    downsample=nn.Identity(),
                    channel_first=False,
                    # =================
                    ssm_d_state=1,
                    ssm_ratio=1.0,
                    ssm_dt_rank='auto',
                    ssm_act_layer=nn.SiLU,
                    ssm_conv=0,
                    ssm_conv_bias=False,
                    ssm_drop_rate=0.0,
                    ssm_init='v0',
                    forward_type='v1dcross_noz',
                    # =================
                    mlp_ratio=4.0,
                    mlp_act_layer=nn.GELU,
                    mlp_drop_rate=0.0,
                    gmlp=False,
                    twin=twin_flag[i_layer],
                ))
            
        self.model_cfg = model_cfg
        self.set_info = set_info = self.model_cfg.set_info
        self.d_model = d_model = self.model_cfg.d_model
        self.nhead = nhead = self.model_cfg.nhead
        self.stage_num = stage_num = 1  # only support plain bakbone
        self.num_shifts = [2] * self.stage_num
        self.checkpoint_blocks = self.model_cfg.checkpoint_blocks
        self.image_pos_num, self.lidar_pos_num = set_info[0][-1], set_info[0][-1]
        self.accelerate = self.model_cfg.get('ACCELERATE', False)
        self.use_map = use_map

        # 将来自 3D Backbone / VFE 的 64 维体素特征统一映射到 128 维，
        # 以匹配 MM_BACKBONE 配置中的 d_model: [128]
        self.voxel_proj = nn.Linear(64, 128)

        self.image_input_layer = UniTRInputLayer(
            self.model_cfg.IMAGE_INPUT_LAYER, self.accelerate)
        self.lidar_input_layer = UniTRInputLayer(
            self.model_cfg.LIDAR_INPUT_LAYER)

        # image patch embedding
        patch_embed_cfg = self.model_cfg.PATCH_EMBED
        if (not self.use_vmamba_pretrain) and (not self.use_prebackbone):
            self.patch_embed = PatchEmbed(
                in_channels=patch_embed_cfg.in_channels,
                embed_dims=patch_embed_cfg.embed_dims,
                conv_type='Conv2d',
                kernel_size=patch_embed_cfg.patch_size,
                stride=patch_embed_cfg.patch_size,
                norm_cfg=patch_embed_cfg.norm_cfg if patch_embed_cfg.patch_norm else None
            )   
        patch_size = [patch_embed_cfg.image_size[0] // patch_embed_cfg.patch_size,
                      patch_embed_cfg.image_size[1] // patch_embed_cfg.patch_size]
        self.patch_size = patch_size
        patch_x, patch_y = torch.meshgrid(torch.arange(
            patch_size[0]), torch.arange(patch_size[1]))
        patch_z = torch.zeros((patch_size[0] * patch_size[1], 1))
        self.patch_zyx = torch.cat(
            [patch_z, patch_y.reshape(-1, 1), patch_x.reshape(-1, 1)], dim=-1)
        # patch coords with batch id
        self.patch_coords = None

        # image branch output norm
        self.out_indices = self.model_cfg.out_indices
        for i in self.out_indices:
            layer = nn.LayerNorm(d_model[-1])
            layer_name = f'out_norm{i}'
            self.add_module(layer_name, layer)

        # Sparse Regional Attention Blocks
        dim_feedforward = self.model_cfg.dim_feedforward
        dropout = self.model_cfg.dropout
        activation = self.model_cfg.activation
        layer_cfg = self.model_cfg.layer_cfg
        

        # Fuse Backbone
        fuse_cfg = self.model_cfg.get('FUSE_BACKBONE', None)
        self.fuse_on = fuse_cfg is not None
        if self.fuse_on:
            # image2lidar
            image2lidar_cfg = fuse_cfg.get('IMAGE2LIDAR', None)
            self.image2lidar_on = image2lidar_cfg is not None
            if self.image2lidar_on:
                # block range of image2lidar
                self.image2lidar_start = image2lidar_cfg.block_start
                self.image2lidar_end = image2lidar_cfg.block_end
                self.map_image2lidar_layer = MapImage2Lidar(
                    image2lidar_cfg, self.accelerate, self.use_map)
                if not (self.use_mamba_inter and not self.use_mixed):
                    self.image2lidar_input_layer = UniTRInputLayer(
                        image2lidar_cfg.image2lidar_layer)
                    self.image2lidar_pos_num = image2lidar_cfg.image2lidar_layer.set_info[0][1]
                    # encode the position of each patch from the closest point in image space
                    self.neighbor_pos_embed = PositionEmbeddingLearned(
                        2, self.d_model[-1])

            # lidar2image
            lidar2image_cfg = fuse_cfg.get('LIDAR2IMAGE', None)
            self.lidar2image_on = lidar2image_cfg is not None
            if self.lidar2image_on:
                # block range of lidar2image
                self.lidar2image_start = lidar2image_cfg.block_start
                self.lidar2image_end = lidar2image_cfg.block_end
                self.map_lidar2image_layer = MapLidar2Image(
                    lidar2image_cfg, self.accelerate, self.use_map, self.use_denoise)
                if not (self.use_mamba_inter2 and not self.use_mixed):
                    self.lidar2image_input_layer = UniTRInputLayer(
                        lidar2image_cfg.lidar2image_layer)
                    self.lidar2image_pos_num = lidar2image_cfg.lidar2image_layer.set_info[0][1]
        # new
        block_id = 0
        self.pos_embed_inter2 = nn.ModuleList()
        if self.image2lidar_on:
            self.inter_block = [i for i in range(self.image2lidar_start, self.image2lidar_end)]
        if self.lidar2image_on:
            self.inter2_block = [i for i in range(self.lidar2image_start, self.lidar2image_end)]
        if self.use_mamba_inter2:
            for block_id in self.inter2_block:
                self.image_input_layer.posembed_layers[0][block_id] = nn.Identity()
                self.lidar_input_layer.posembed_layers[0][block_id] = nn.Identity()
        if self.use_mamba_inter:
            for block_id in self.inter_block:
                self.image_input_layer.posembed_layers[0][block_id] = nn.Identity()
                self.lidar_input_layer.posembed_layers[0][block_id] = nn.Identity()
        if self.use_winmamba:
            self.image_input_layer.posembed_layers[0][0] = nn.Identity()
            self.lidar_input_layer.posembed_layers[0][0] = nn.Identity()        
        self.dsb_len = self.model_cfg.get('DSB_LEN', 1)
        self.use_offset = self.model_cfg.get('USE_OFFSET', 0)
        if self.use_offset:
            self.generate_offset = nn.ModuleList() # N*128 -> N*2 生成的offset在[-1, 1]之间
        if self.use_mamba_inter or self.use_mamba_inter2 or self.use_winmamba:
            self.hilbert_config = {'curve_template_path_rank10': '../ckpts/hilbert_template/curve_template_3d_rank_10.pth', 
                                        'curve_template_path_rank9': '../ckpts/hilbert_template/curve_template_3d_rank_9.pth', 
                                        'curve_template_path_rank8': '../ckpts/hilbert_template/curve_template_3d_rank_8.pth', 
                                        'curve_template_path_rank7': '../ckpts/hilbert_template/curve_template_3d_rank_7.pth'
                                        }
            # self.hilbert_spatial_sis
            self.curve_template = {}
            self.template_on_device = False
            self.hilbert_spatial_size = {}
            self.load_template('/home/suyi/AirV2X-Perception_copy/opencood/models/mambafusion_modules/ckpts/hilbert_template/curve_template_3d_rank_10.pth', 10)
            self.load_template('/home/suyi/AirV2X-Perception_copy/opencood/models/mambafusion_modules/ckpts/hilbert_template/curve_template_3d_rank_9.pth', 9)
            self.load_template('/home/suyi/AirV2X-Perception_copy/opencood/models/mambafusion_modules/ckpts/hilbert_template/curve_template_3d_rank_8.pth', 8)
            self.load_template('/home/suyi/AirV2X-Perception_copy/opencood/models/mambafusion_modules/ckpts/hilbert_template/curve_template_3d_rank_7.pth', 7)
        # 支持矩形BEV：优先从 BEV_SIZE_H/W 读取，否则从 LIDAR_INPUT_LAYER.sparse_shape 推断，最后使用 BEV_SIZE 或默认值
        if 'BEV_SIZE_H' in model_cfg and 'BEV_SIZE_W' in model_cfg:
            self.bev_size_H = model_cfg.get('BEV_SIZE_H', 200)
            self.bev_size_W = model_cfg.get('BEV_SIZE_W', 704)
        elif 'LIDAR_INPUT_LAYER' in model_cfg and 'sparse_shape' in model_cfg.LIDAR_INPUT_LAYER:
            # 从 LIDAR_INPUT_LAYER.sparse_shape 推断：[X, Y, Z] -> [704, 200, 1]
            sparse_shape = model_cfg.LIDAR_INPUT_LAYER.sparse_shape
            self.bev_size_H = sparse_shape[1]  # Y维度 = 200
            self.bev_size_W = sparse_shape[0]  # X维度 = 704
        else:
            # 向后兼容：使用 BEV_SIZE 或默认值（假设正方形）
            self.bev_size_H = model_cfg.get('BEV_SIZE', 360)
            self.bev_size_W = model_cfg.get('BEV_SIZE', 360)
        self.bev_size = self.bev_size_H  # 保留用于向后兼容
        self.shape_inter = [self.inter1_win_shape[-1], self.bev_size_H, self.bev_size_W]
        self.shape_inter2 = [self.inter2_win_shape[-1], int(int(self.image_shape[1] / 8 * 3)), int(self.image_shape[0] / 8 * 3)]
        if self.mask_out_img:
            self.shape_inter2 = [self.shape_inter2[0], self.shape_inter2[1]/3,self.shape_inter2[2]/3,]
        
        # 动态选择Hilbert模板，根据BEV尺寸选择合适的rank
        # shape_inter用于LiDAR BEV特征，shape_inter2用于LiDAR->Image交互
        max_bev_dim = max(self.bev_size_H, self.bev_size_W)
        max_inter2_dim = max(self.shape_inter2[1], self.shape_inter2[2])
        
        # 为shape_inter选择模板 (原始BEV尺寸)
        if max_bev_dim > 512:
            self.inter_downsample_ori = 'curve_template_rank10'
            self.inter_downsample_lvl = 'curve_template_rank9'
        elif max_bev_dim > 256:
            self.inter_downsample_ori = 'curve_template_rank9'
            self.inter_downsample_lvl = 'curve_template_rank8'
        else:
            self.inter_downsample_ori = 'curve_template_rank8'
            self.inter_downsample_lvl = 'curve_template_rank7'
        
        # 为shape_inter2选择模板 (LiDAR->Image交互尺寸)
        if max_inter2_dim > 512:
            self.inter2_downsample_ori = 'curve_template_rank10'
            self.inter2_downsample_lvl = 'curve_template_rank9'
        elif max_inter2_dim > 256:
            self.inter2_downsample_ori = 'curve_template_rank9'
            self.inter2_downsample_lvl = 'curve_template_rank8'
        else:
            self.inter2_downsample_ori = 'curve_template_rank8'
            self.inter2_downsample_lvl = 'curve_template_rank7'
        
        print(f"[MambaFusion] BEV size: ({self.bev_size_H}, {self.bev_size_W}), "
              f"shape_inter={self.shape_inter}, shape_inter2={self.shape_inter2}")
        print(f"[MambaFusion] Using templates: inter=({self.inter_downsample_ori}, {self.inter_downsample_lvl}), "
              f"inter2=({self.inter2_downsample_ori}, {self.inter2_downsample_lvl})")
        
        self.multi_scale_norm_list = nn.ModuleList()
        for stage_id in range(stage_num):
            num_blocks_this_stage = set_info[stage_id][-1]
            dmodel_this_stage = d_model[stage_id]
            dfeed_this_stage = dim_feedforward[stage_id]
            num_head_this_stage = nhead[stage_id]
            block_list, norm_list = [], []
            
            for i in range(num_blocks_this_stage):
                if (self.use_mamba_inter and i in self.inter_block)\
                    or (self.use_mamba_inter2 and  i in self.inter2_block):
                    if self.use_mamba_inter and i in self.inter_block:
                        self.pos_embed_inter = nn.Sequential(
                            nn.Linear(9, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 128),
                            )
                    elif self.use_mamba_inter2 and  i in self.inter2_block:
                        if self.mixed_version == 0 or self.mixed_version == 2 or self.mixed_version == 3:
                            self.pos_embed_inter2.append(nn.Sequential(
                                nn.Linear(9, 128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 128),
                                ))
                    else:
                        raise ValueError('Invalid use_mamba_inter or use_mamba_inter2')
                    if self.use_offset:
                        offset_list = nn.ModuleList()
                        for i in range(self.dsb_len):
                            offset_list.append(
                                spconv.SparseSequential(
                                    spconv.SubMConv3d(128, 2, (5, 1, 1), stride=(1, 1, 1), padding=1, bias=False),
                                    nn.BatchNorm1d(2, eps=1e-3, momentum=0.01),
                                    nn.Tanh(),)
                                )
                        self.generate_offset.append(offset_list)
                    if (self.use_multi_scalev or self.use_multi_scalev_down or self.use_multi_scale) and i != self.lidar2image_end - 1:
                        self.multi_scale_norm_list.append(nn.LayerNorm(128))

                    dsb = nn.ModuleList()
                    
                    for j in range(self.dsb_len):
                        
                        if self.use_multi_scalev:
                            # dsb.append(WinMamba_Block(dim=64, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=[13, 13, 1], group_size=256, direction=['x', 'y'], shift=True,
                            #     operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34))
                            dsb.append(LocalMamba(dim=128, depth=2, down_scales=self.inter2_down_scales, window_shape=self.inter2_win_shape, group_size=self.inter2_win_size, direction=['x', 'y'], shift=True,
                                    operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34, use_expand=self.inter2_use_expand))
                            dsb.append(GlobalMamba(128, ssm_cfg=None, norm_epsilon=1e-05, rms_norm=True, 
                                down_kernel_size=[3, 3], down_stride=[1, 2], num_down=[0, 1], 
                                norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=self.shape_inter2, hilbert_config=self.hilbert_config,
                                downsample_ori=self.inter2_downsample_ori,  # 动态选择
                                downsample_lvl=self.inter2_downsample_lvl,  # 动态选择
                                down_resolution=True, residual_in_fp32=True, fused_add_norm=True, 
                                device='cuda', dtype=torch.float32))
                        elif self.use_multi_scalev_down:
                            # dsb.append(WinMamba_Block(dim=64, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=[13, 13, 1], group_size=256, direction=['x', 'y'], shift=True,
                            #                    operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34))
                            dsb.append(LocalMamba(dim=128, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=self.inter2_win_shape, group_size=self.inter2_win_size, direction=['x', 'y'], shift=True,
                                    operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34))
                            dsb.append(GlobalMamba(128, ssm_cfg=None, norm_epsilon=1e-05, rms_norm=True, 
                                down_kernel_size=[3, 3], down_stride=[1, 2], num_down=[0, 1], 
                                norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=self.shape_inter2, hilbert_config=self.hilbert_config,
                                downsample_ori=self.inter2_downsample_ori,  # 动态选择
                                downsample_lvl=self.inter2_downsample_lvl,  # 动态选择
                                down_resolution=True, residual_in_fp32=True, fused_add_norm=True, 
                                device='cuda', dtype=torch.float32))
                            
                        elif self.use_multi_scale:
                            # dsb.append(WinMamba_Block(dim=64, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=[13, 13, 1], group_size=256, direction=['x', 'y'], shift=True,
                            #                    operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34))
                            dsb.append(LocalMamba(dim=128, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=self.inter2_win_shape, group_size=self.inter2_win_size, direction=['x', 'y'], shift=True,
                                    operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34))
                            dsb.append(GlobalMamba(128, ssm_cfg=None, norm_epsilon=1e-05, rms_norm=True, 
                                down_kernel_size=[3, 3], down_stride=[1, 2], num_down=[0, 1], 
                                norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=self.shape_inter2, hilbert_config=self.hilbert_config,
                                downsample_ori=self.inter2_downsample_ori,  # 动态选择
                                downsample_lvl=self.inter2_downsample_lvl,  # 动态选择
                                down_resolution=True, residual_in_fp32=True, fused_add_norm=True, 
                                device='cuda', dtype=torch.float32))
                        else:
                            if self.mixed_version == 0:
                                # dsb.append(WinMamba_Block(dim=64, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=[13, 13, 1], group_size=256, direction=['x', 'y'], shift=True,
                                #     operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34))
                                dsb.append(LocalMamba(dim=128, depth=2, down_scales=self.inter2_down_scales, window_shape=self.inter2_win_shape, group_size=self.inter2_win_size, direction=['x', 'y'], shift=self.use_shift,
                                    operator=EasyDict(self.localmamba_info),layer_id=0, n_layer=34, win_version=self.win_version, use_expand=self.inter2_use_expand, use_fixed_mapping=self.use_fixed_mapping, use_inverse=self.use_inverse, use_checkpoint=self.use_checkpoint_local))
                                # dsb.append(WinMamba_Block(dim=64, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=[30, 30, 1], group_size=90, direction=['x', 'y'], shift=True,
                                #     operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34))
                                
                                dsb.append(GlobalMamba(128, ssm_cfg=self.ssm_cfg, norm_epsilon=1e-05, rms_norm=True, 
                                    down_kernel_size=[3, 3], down_stride=self.inter2_down_scales_global, num_down=[0, 1], 
                                    norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=self.shape_inter2, hilbert_config=self.hilbert_config,
                                    downsample_ori=self.inter2_downsample_ori,  # 动态选择
                                    downsample_lvl=self.inter2_downsample_lvl if self.use_down_scale_inter2 else self.inter2_downsample_ori,  # 动态选择
                                    down_resolution=self.use_down_scale_inter2, residual_in_fp32=True, fused_add_norm=True, 
                                    device='cuda', dtype=torch.float32, use_checkpoint=self.use_checkpoint_global))
                            elif self.mixed_version == 1:
                                # dsb.append(WinMamba_Block(dim=64, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=[13, 13, 1], group_size=256, direction=['x', 'y'], shift=True,
                                #                 operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34))
                                # dsb.append(WinMamba_Block(dim=64, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=[13, 13, 1], group_size=256, direction=['x', 'y'], shift=True,
                                #                 operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34))
                                dsb.append(LocalMamba(dim=128, depth=2, down_scales=self.inter2_down_scales, window_shape=self.inter2_win_shape, group_size=self.inter2_win_size, direction=['x', 'y'], shift=True,
                                    operator=EasyDict(self.localmamba_info),layer_id=0, n_layer=34, win_version=self.win_version))
                                dsb.append(LocalMamba(dim=128, depth=2, down_scales=self.inter2_down_scales, window_shape=self.inter2_win_shape, group_size=self.inter2_win_size, direction=['x', 'y'], shift=True,
                                    operator=EasyDict(self.localmamba_info),layer_id=0, n_layer=34, win_version=self.win_version))
                                # dsb.append(WinMamba_Block(dim=64, depth=2, down_scales=self.inter2_down_scales, window_shape=self.inter2_win_shape, group_size=self.inter2_win_size, direction=['x', 'y'], shift=True,
                                #     operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34, win_version=self.win_version))
                            elif self.mixed_version == 2:
                                dsb.append(GlobalMamba(128, ssm_cfg=None, norm_epsilon=1e-05, rms_norm=True, 
                                    down_kernel_size=[3, 3], down_stride=[1, 2], num_down=[0, 1], 
                                    norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=self.shape_inter2, hilbert_config=self.hilbert_config,
                                    downsample_ori='curve_template_rank9',
                                    downsample_lvl='curve_template_rank8',
                                    down_resolution=True, residual_in_fp32=True, fused_add_norm=True, 
                                    device='cuda', dtype=torch.float32))
                                dsb.append(GlobalMamba(128, ssm_cfg=None, norm_epsilon=1e-05, rms_norm=True, 
                                    down_kernel_size=[3, 3], down_stride=[1, 2], num_down=[0, 1], 
                                    norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=self.shape_inter2, hilbert_config=self.hilbert_config,
                                    downsample_ori='curve_template_rank9',
                                    downsample_lvl='curve_template_rank8',
                                    down_resolution=True, residual_in_fp32=True, fused_add_norm=True, 
                                    device='cuda', dtype=torch.float32))         
                            elif self.mixed_version == 3:                    
                                dsb.append(GlobalMamba(128, ssm_cfg=None, norm_epsilon=1e-05, rms_norm=True, 
                                    down_kernel_size=[3, 3], down_stride=[1, 2], num_down=[0, 1], 
                                    norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=self.shape_inter2, hilbert_config=self.hilbert_config,
                                    downsample_ori='curve_template_rank9',
                                    downsample_lvl='curve_template_rank8',
                                    down_resolution=True, residual_in_fp32=True, fused_add_norm=True, 
                                    device='cuda', dtype=torch.float32))
                                dsb.append(GlobalMamba(128, ssm_cfg=None, norm_epsilon=1e-05, rms_norm=True, 
                                    down_kernel_size=[3, 3], down_stride=[1, 2], num_down=[0, 1], 
                                    norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=self.shape_inter2, hilbert_config=self.hilbert_config,
                                    downsample_ori='curve_template_rank9',
                                    downsample_lvl='curve_template_rank8',
                                    down_resolution=True, residual_in_fp32=True, fused_add_norm=True, 
                                    device='cuda', dtype=torch.float32))     
                    
                    # if self.use_mixed:                
                    #     block_list.append(
                    #         nn.ModuleList([
                    #             dsb,
                    #             UniTRBlock(dmodel_this_stage, num_head_this_stage, dfeed_this_stage,
                    #                 dropout, activation, batch_first=True, block_id=block_id,
                    #                 dout=dmodel_this_stage, layer_cfg=layer_cfg)
                    #         ]
                    #         )
                            
                    #     )
                    # else:
                    block_list.append(dsb)
                else:

                    if self.use_winmamba and i == 0:
                        if self.inter1_win_shape[-1] != 1:
                            from opencood.models.mambafusion_modules.backbones_3d.lion_backbone_one_stride import PatchMerging3D
                            self.dow5 = PatchMerging3D(128, 128, down_scale=[1, 1, 2],
                                        norm_layer=partial(nn.LayerNorm), diffusion=True, diff_scale=0.2)
                        if self.mixed_version == 0 or self.mixed_version == 2:
                            self.pos_embed_intra = nn.Sequential(
                                nn.Linear(9, 128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 128),
                                )
                        if self.mixed_version == 0:
                            block_list.append(
                                nn.ModuleList([
                                    LocalMamba(dim=128, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=self.inter1_win_shape, group_size=self.inter1_win_size, direction=['x', 'y'], shift=self.use_shift,
                                                operator=EasyDict(self.localmamba_info),layer_id=0, n_layer=34, use_checkpoint=self.use_checkpoint_local),
                                    GlobalMamba(128, ssm_cfg=self.ssm_cfg, norm_epsilon=1e-05, rms_norm=True, 
                                        down_kernel_size=[3, 3], down_stride=[1, 2], num_down=[0, 1], 
                                        norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=self.shape_inter, hilbert_config=self.hilbert_config,
                                        downsample_ori=self.inter_downsample_ori,  # 动态选择，根据BEV尺寸
                                        downsample_lvl=self.inter_downsample_lvl,  # 动态选择
                                        down_resolution=True, residual_in_fp32=True, fused_add_norm=True, 
                                        device='cuda', dtype=torch.float32, use_checkpoint=self.use_checkpoint_global)
                                ])
                            )
                        elif self.mixed_version == 1:      
                            block_list.append(
                                nn.ModuleList([
                                    LocalMamba(dim=128, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=self.inter1_win_shape, group_size=self.inter1_win_size, direction=['x', 'y'], shift=True,
                                                operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34),
                                    LocalMamba(dim=128, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=self.inter1_win_shape, group_size=self.inter1_win_size, direction=['x', 'y'], shift=True,
                                                operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34),
                                ])
                            )
                        elif self.mixed_version == 2:
                            block_list.append(
                                nn.ModuleList([
                                    GlobalMamba(128, ssm_cfg=None, norm_epsilon=1e-05, rms_norm=True,
                                        down_kernel_size=[3, 3], down_stride=[1, 2], num_down=[0, 1],
                                        norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=self.shape_inter, hilbert_config=self.hilbert_config,
                                        downsample_ori=self.inter_downsample_ori,  # 动态选择
                                        downsample_lvl=self.inter_downsample_lvl,  # 动态选择
                                        down_resolution=True, residual_in_fp32=True, fused_add_norm=True,
                                        device='cuda', dtype=torch.float32),
                                    GlobalMamba(128, ssm_cfg=None, norm_epsilon=1e-05, rms_norm=True,
                                        down_kernel_size=[3, 3], down_stride=[1, 2], num_down=[0, 1],
                                        norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), indice_key='stem0_layer0', sparse_shape=self.shape_inter, hilbert_config=self.hilbert_config,
                                        downsample_ori=self.inter_downsample_ori,  # 动态选择
                                        downsample_lvl=self.inter_downsample_lvl,  # 动态选择
                                        down_resolution=True, residual_in_fp32=True, fused_add_norm=True,
                                        device='cuda', dtype=torch.float32)
                                ])
                            )
                        elif self.mixed_version == 3:
                            block_list.append(
                                nn.ModuleList([
                                    LocalMamba(dim=128, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=self.inter1_win_shape, group_size=self.inter1_win_size, direction=['x', 'y'], shift=True,
                                                operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34),
                                    LocalMamba(dim=128, depth=2, down_scales=[[2, 2, 1], [2, 2, 1]], window_shape=self.inter1_win_shape, group_size=self.inter1_win_size, direction=['x', 'y'], shift=True,
                                                operator=EasyDict({'NAME': 'Mamba', 'CFG': {'d_state': 16, 'd_conv': 4, 'expand': 2, 'drop_path': 0.2}}),layer_id=0, n_layer=34),

                                ])
                            )
                    else: 
                        raise ValueError('Unexpected Parameters')                  
                        # block_list.append(
                        #     UniTRBlock(dmodel_this_stage, num_head_this_stage, dfeed_this_stage,
                        #             dropout, activation, batch_first=True, block_id=block_id,
                        #             dout=dmodel_this_stage, layer_cfg=layer_cfg)
                        # )
                norm_list.append(nn.LayerNorm(dmodel_this_stage))
                block_id += 1
            self.__setattr__(f'stage_{stage_id}', nn.ModuleList(block_list))
            self.__setattr__(
                f'residual_norm_stage_{stage_id}', nn.ModuleList(norm_list))
            if layer_cfg.get('split_residual', False):
                # use different norm for lidar and image
                lidar_norm_list = [nn.LayerNorm(
                    dmodel_this_stage) for _ in range(num_blocks_this_stage)]
                self.__setattr__(
                    f'lidar_residual_norm_stage_{stage_id}', nn.ModuleList(lidar_norm_list))



        self._reset_parameters()
        # self.register_hooks(self) # 注册钩子
    def remove_layers(self, att_name,  layers_to_remove):
        for layer_name in layers_to_remove:
            if hasattr(att_name, layer_name):
                delattr(att_name, layer_name)
            else:
                print(f"Layer {layer_name} not found in Backbone_VSSM")
    def load_template(self, path, rank):
        template = torch.load(path)
        if isinstance(template, dict):
            self.curve_template[f'curve_template_rank{rank}'] = template['data'].reshape(-1)
            self.hilbert_spatial_size[f'curve_template_rank{rank}'] = template['size'] 
        else:
            self.curve_template[f'curve_template_rank{rank}'] = template.reshape(-1)
            spatial_size = 2 ** rank
            self.hilbert_spatial_size[f'curve_template_rank{rank}'] = (1, spatial_size, spatial_size) #[z, y, x]
    def register_hooks(self, module_name):
        def create_hook(module_name):
            def check_param(module, input, output):
                if module_name in self.used_params:
                    raise ValueError(f"Parameter {module_name} is used more than once")
                else:
                    self.used_params.add(module_name)
            return check_param
        def is_leaf_module(module):
            return len(list(module.children())) == 0
        # 注册钩子
        self.all_params = set()
        self.used_params = set()
        for name, module in module_name.named_modules():           
            if len(list(module.parameters())) > 0 and is_leaf_module(module):  # 只对有参数的模块注册钩子
                self.all_params.add(name)
                module.register_forward_hook(create_hook(name))
    @staticmethod
    def _make_vmamba_layer(
        dim=96, 
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
        twin=False,
        cross=False,
        **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, 
                cross_dim=dim,
                drop_path= drop_path[d % int(depth/2)] if twin else drop_path[d],
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
        if twin:
            return nn.Sequential(OrderedDict(
                blocks1=nn.Sequential(*blocks[:int(len(blocks)/2)]),
                blocks2=nn.Sequential(*blocks[int(len(blocks)/2):]),
            ))
        elif cross:
            return nn.Sequential(OrderedDict(
                blocks=nn.Sequential(*blocks,),
            ))
        else:
            return nn.Sequential(OrderedDict(
                blocks=nn.Sequential(*blocks,),
                downsample=downsample,
            ))
    def forward_stage2(self, block, output_sparse, lidar2image_coor, batch_dict, i, fixed_mapping_list=None, voxel_num=None):

        for j, b in enumerate(block):
            if self.use_more_res:
                res_output = output_sparse.features.clone()
            if isinstance(b, LocalMamba):
                if self.use_fixed_mapping:
                    output_sparse, fixed_mapping_list = b(output_sparse, fixed_mapping_list)
                else:
                    output_sparse = b(output_sparse)

                if self.use_more_res:
                    output = output_sparse.features
                    # output = self.res_blocks[i - 1][j](output + res_output)
                    output = torch.cat([self.res_blocks_lidar[i - 1][j](output[:voxel_num] + res_output[:voxel_num]), 
                                        self.res_blocks_camera[i - 1][j](output[voxel_num:] + res_output[voxel_num:])])
                    output_sparse = replace_feature(output_sparse, output)
                else:
                    if j == len(block) - 1:
                        output = output_sparse.features
            elif isinstance(b, GlobalMamba):
                output, _ = b(output_sparse.features, lidar2image_coor, batch_dict['batch_size'],
                                        self.shape_inter2, self.curve_template, self.hilbert_spatial_size,
                                        self.pos_embed_inter2[i - self.lidar2image_start], 0, False)
                # if not (j == len(block) - 1 and i == self.lidar2image_end - 1):
                if self.use_more_res:
                    # output = self.res_blocks[i - 1][j](output + res_output)
                    output = torch.cat([self.res_blocks_lidar[i - 1][j](output[:voxel_num] + res_output[:voxel_num]), 
                                        self.res_blocks_camera[i - 1][j](output[voxel_num:] + res_output[voxel_num:])])
                    output_sparse = replace_feature(output_sparse, output)
                else:
                    if not (j == len(block) - 1 and i == self.lidar2image_end - 1):
                        output_sparse = replace_feature(output_sparse, output)

                
            else:
                raise ValueError("Block type not supported")

        # if i == self.lidar2image_end - 1:
        #     return output_extended[:ori_len], output_sparse
        if self.use_fixed_mapping:
            return output, output_sparse, fixed_mapping_list
        else:
            return output, output_sparse
    
    
    
    def forward(self, batch_dict, agent=None):
        '''
        Args:
            bacth_dict (dict): 
                The dict contains the following keys
                - voxel_features (Tensor[float]): Voxel features after VFE. Shape of (N, d_model[0]), 
                    where N is the number of input voxels.
                - voxel_coords (Tensor[int]): Shape of (N, 4), corresponding voxel coordinates of each voxels.
                    Each row is (batch_id, z, y, x). 
                - camera_imgs (Tensor[float]): multi view images, shape of (B, N, C, H, W),
                    where N is the number of image views.
                - ...
        
        Returns:
            bacth_dict (dict):
                The dict contains the following keys
                - pillar_features (Tensor[float]):
                - voxel_coords (Tensor[int]):
                - image_features (Tensor[float]):
        '''
        import time
        start_time = time.time()
        # Select per-agent dict if provided
        self.agent = agent
        outer_dict = None

        if agent is not None:
            outer_dict = batch_dict
            batch_dict = batch_dict[agent]
        if 'batch_size' not in batch_dict:
            batch_dict['batch_size'] = batch_dict.get('batch_size', 1)
        # Normalize camera images early so downstream uses (e.g., lidar2image_view_num)
        # always see batch_dict['camera_imgs'] even if the dataloader provides
        try:
            self._extract_camera_imgs(batch_dict)
        except Exception as e:
            print(f"[MambaFusion] 图像初始化失败: {e}")
            # 如果没有有效的图像数据，跳过该agent的处理
            if agent is not None:
                print(f"[MambaFusion] 跳过agent {agent} 的图像处理")
                return batch_dict
            else:
                raise e
        # lidar(3d) and image(2d) preprocess
        self.used_params = set()
        batch_dict['use_all_mamba'] = self.use_all_mamba
        if self.use_all_mamba:
            multi_feat, voxel_info, patch_info = self._input_preprocess2(batch_dict)
            multi_pos_embed_list = None
        else:
            if self.use_cross_img:
                multi_feat, voxel_info, patch_info, multi_set_voxel_inds_list, multi_set_voxel_masks_list, multi_pos_embed_list, sweeps_imgs, num_set_ponit = self._input_preprocess(
                    batch_dict)
            else:
                multi_feat, voxel_info, patch_info, multi_set_voxel_inds_list, multi_set_voxel_masks_list, multi_pos_embed_list = self._input_preprocess(
                    batch_dict)
        # lidar(3d) and image(3d) preprocess multi_set_voxel_inds_list[0][1][:, num_set_ponit:,: ] - batch_dict['voxel_num']
        if self.image2lidar_on: # 将image的feature映射到lidar的feature 将图像数据预处理为激光雷达特征，以增强点云的特征表示
            if self.use_mamba_inter:
                image2lidar_inds_list, image2lidar_masks_list, multi_pos_embed_list, image2lidar_coor = self._image2lidar_preprocess(
                    batch_dict, multi_feat, multi_pos_embed_list)
                if self.shape_inter[0] == 2:
                    image2lidar_coor[:batch_dict['voxel_num'], 1] = 1
            else:
                image2lidar_inds_list, image2lidar_masks_list, multi_pos_embed_list = self._image2lidar_preprocess(
                    batch_dict, multi_feat, multi_pos_embed_list)
        # lidar(2d) and image(2d) preprocess
        if self.lidar2image_on: # 将lidar的feature映射到image的feature 将激光雷达数据预处理为图像特征，以增强图像的特征表示
            if self.use_mamba_inter2: #True
                # per-agent number of camera views
                lidar2image_view_num = int(batch_dict['camera_imgs'].shape[1])
                if self.use_multi_scale: #False
                    lidar2image_inds_list, lidar2image_masks_list, multi_pos_embed_list, lidar2image_coor, lidar2image_coords_bzyx_list = self._lidar2image_preprocess(
                        batch_dict, multi_feat, multi_pos_embed_list)
                    # 与原始 MambaFusion 对齐：不做 x/y 交换，只合并 batch_id 与 view_idx
                    for i in range(len(lidar2image_coords_bzyx_list)):
                        lidar2image_coords_bzyx_list[i][:, 0] = lidar2image_coords_bzyx_list[i][:, 0] * lidar2image_view_num + lidar2image_coords_bzyx_list[i][:, 1]
                        lidar2image_coords_bzyx_list[i][:, 1] = 0

                else:
                    lidar2image_inds_list, lidar2image_masks_list, multi_pos_embed_list, lidar2image_coor = self._lidar2image_preprocess(
                        batch_dict, multi_feat, multi_pos_embed_list,outer_dict)
                if self.win_version == 'v4':#False
                    pass
                else:
                    lidar2image_coor[:, 0] = lidar2image_coor[:, 0] * lidar2image_view_num + lidar2image_coor[:, 1]
                    lidar2image_coor[:, 1] = 0
                    if self.shape_inter2[0] == 2:
                        lidar2image_coor[:batch_dict['voxel_num'], 1] = 1
            else:
                lidar2image_inds_list, lidar2image_masks_list, multi_pos_embed_list = self._lidar2image_preprocess(
                    batch_dict, multi_feat, multi_pos_embed_list)
        output = multi_feat # torch.Size([num_of_voxel + num_of_patch, 128])
        block_id = 0
        voxel_num = batch_dict['voxel_num'] # num_of_voxels
        
        batch_dict['image_features'] = []
        if self.use_vmamba:
            voxel_pos_embeding = self.lidar_pos_embed_layer(batch_dict['voxel_coords'][:, 1:].float())
            lidar2image = batch_dict['lidar2image'].clone()
            lidar2image_view_num = int(batch_dict['camera_imgs'].shape[1])
            lidar2image = lidar2image.view(batch_dict['batch_size'] * lidar2image_view_num, 16)
            lidar2image = lidar2image.repeat_interleave(2816, dim=0) # [33792, 16]
            img_patch_coords = self.patch_coords.clone()
            img_patch_coords[:, 0] = img_patch_coords[:, 0] % lidar2image_view_num # [33792, 4]
            img_pos_embeding = self.img_pos_embed_layer(torch.cat([img_patch_coords, lidar2image], dim=1).float())
            multi_pos_embeding = torch.cat([voxel_pos_embeding, img_pos_embeding], dim=0) # 
            # img_pos_embeding = 
        if self.use_mamba_inter or self.use_mamba_inter2 or self.use_winmamba and not self.template_on_device:
            with torch.no_grad():
                for name, _ in self.curve_template.items():
                    self.curve_template[name] = self.curve_template[name].to(output.device)
            self.template_on_device = True
        # block forward
        for stage_id in range(self.stage_num): # 1
            block_layers = self.__getattr__(f'stage_{stage_id}')
            residual_norm_layers = self.__getattr__(
                f'residual_norm_stage_{stage_id}')
            for i in range(len(block_layers)): # 4 (1 2 1) use hilbert curve
                block = block_layers[i]
                residual = output.clone() # torch.Size([num_of_voxel + num_of_patch, 128])                 

                if self.image2lidar_on and i >= self.image2lidar_start and i < self.image2lidar_end: # i == 3 模态间注意力机制
                    if self.use_mamba_inter:
                        
                        if self.use_mixed: 
                            output, _ = block[0](output, image2lidar_coor, batch_dict['batch_size'], self.shape_inter,
                                    self.curve_template, self.hilbert_spatial_size, self.pos_embed_inter, 0, False)
                            output = block[1](output, image2lidar_inds_list[stage_id], image2lidar_masks_list[stage_id], multi_pos_embed_list[stage_id][i],
                                    block_id=block_id, voxel_num=voxel_num, using_checkpoint=block_id in self.checkpoint_blocks)
                            
                        else:
                            if self.use_checkpoint_inter:
                                output, _ = checkpoint(block, output, image2lidar_coor, batch_dict['batch_size'], self.shape_inter, 
                                    self.curve_template, self.hilbert_spatial_size, self.pos_embed_inter, 0, False,use_reentrant=False)
                            else:
                                output, _ = block(output, image2lidar_coor, batch_dict['batch_size'], self.shape_inter,
                                        self.curve_template, self.hilbert_spatial_size, self.pos_embed_inter, 0, False)
                    else:
                        output = block(output, image2lidar_inds_list[stage_id], image2lidar_masks_list[stage_id], multi_pos_embed_list[stage_id][i],
                                    block_id=block_id, voxel_num=voxel_num, using_checkpoint=block_id in self.checkpoint_blocks)
                    if self.use_vmamba and self.use_in_mid:
                        output = self.do_vmamba_inter(output, voxel_num, batch_dict, block_id)
                elif self.lidar2image_on and i >= self.lidar2image_start and i < self.lidar2image_end: #? i == 1 or 2  可能有两个不同的阶段需要进行激光雷达到图像的特征映射操作
                    if self.use_mamba_inter2:
                        if self.use_mixed:
                            output, _ = block[0](output, lidar2image_coor, batch_dict['batch_size'], self.shape_inter2,
                                    self.curve_template, self.hilbert_spatial_size, self.pos_embed_inter2[i - self.lidar2image_start], 0, False)
                            output = block[1](output, image2lidar_inds_list[stage_id], image2lidar_masks_list[stage_id], multi_pos_embed_list[stage_id][i],
                                    block_id=block_id, voxel_num=voxel_num, using_checkpoint=block_id in self.checkpoint_blocks)
                        elif self.use_mixed_scale:
                            if i == self.lidar2image_start:
                                output_extended = output
                                ori_len = output.shape[0]
                                if self.use_multi_scalev_down:
                                    lidar2image_coor = torch.cat([lidar2image_coor, batch_dict['multi_scalev_down_features_coords']], dim=0)   
                                    output_extended = torch.cat([output_extended, batch_dict['multi_scalev_down_features']], dim=0)
                                if self.use_multi_scale:
                                    use_conv_list = ['x_conv3']
                                    multi_scale_3d_features = batch_dict['multi_scale_3d_features']
                                    extends = torch.cat([multi_scale_3d_features[conv_name].features for conv_name in use_conv_list], dim=1)
                                    lidar2image_coor = torch.cat([lidar2image_coor, torch.cat(lidar2image_coords_bzyx_list)], dim=0)
                                    output_extended = torch.cat([output_extended, extends], dim=0)
                                
                                # DEBUG: 打印创建SparseConvTensor前的坐标范围（use_multi_scale路径）
                                if lidar2image_coor.shape[0] > 0:
                                    sparse_y_min, sparse_y_max = lidar2image_coor[:, 2].min().item(), lidar2image_coor[:, 2].max().item()
                                    sparse_x_min, sparse_x_max = lidar2image_coor[:, 3].min().item(), lidar2image_coor[:, 3].max().item()
                                    # print(f"[mambafusion] use_multi_scale创建SparseConvTensor前坐标范围: y=[{sparse_y_min}, {sparse_y_max}], x=[{sparse_x_min}, {sparse_x_max}], shape_inter2={self.shape_inter2}")
                                    # print(f"[mambafusion] use_multi_scale创建SparseConvTensor前超出范围: y>{self.shape_inter2[1]-1}有{(lidar2image_coor[:, 2] > self.shape_inter2[1]-1).sum().item()}个, x>{self.shape_inter2[2]-1}有{(lidar2image_coor[:, 3] > self.shape_inter2[2]-1).sum().item()}个")
                                
                                # 注意：暂时不进行clamp，先观察对齐后的坐标分布
                                
                                output_sparse = spconv.SparseConvTensor(
                                    features=output_extended,
                                    indices=lidar2image_coor.int(),
                                    spatial_shape=self.shape_inter2,
                                    batch_size=batch_dict['batch_size'] * lidar2image_view_num
                                )
                            else:
                                output_sparse = replace_feature(output_sparse, output_extended)
                            if i != self.lidar2image_end - 1:
                                residual_extended = output_extended[ori_len:].clone()
                            output_extended, output_sparse = self.forward_stage2(block, output_sparse, lidar2image_coor, batch_dict, i)

                        elif self.use_multi_scalev_down:
                            if i == self.lidar2image_start:
                                lidar2image_coor = torch.cat([lidar2image_coor, batch_dict['multi_scalev_down_features_coords']], dim=0)
                                ori_len = output.shape[0]
                                output_extended = torch.cat([output, batch_dict['multi_scalev_down_features']], dim=0)
                                output_sparse = spconv.SparseConvTensor(
                                    features=output_extended,
                                    indices=lidar2image_coor.int(),
                                    spatial_shape=self.shape_inter2,
                                    batch_size=batch_dict['batch_size'] * lidar2image_view_num
                                )
                            else:
                                output_sparse = replace_feature(output_sparse, output_extended)
                            if i != self.lidar2image_end - 1:
                                residual_extended = output_extended[ori_len:].clone()
                            output_extended, output_sparse = self.forward_stage2(block, output_sparse, lidar2image_coor, batch_dict, i)

                        elif self.use_multi_scalev:
                            if i == self.lidar2image_start:
                                lidar2image_coor[:, -1] *= 2
                                lidar2image_coor[:, -2] *= 2
                                lidar2image_coor = torch.cat([lidar2image_coor, batch_dict['multi_scalev_features_coords']], dim=0)
                                ori_len = output.shape[0]
                                output_extended = torch.cat([output, batch_dict['multi_scalev_features']], dim=0)
                                output_sparse = spconv.SparseConvTensor(
                                    features=output_extended,
                                    indices=lidar2image_coor.int(),
                                    spatial_shape=[1, self.shape_inter2[1] * 2, self.shape_inter2[2] * 2],
                                    batch_size=batch_dict['batch_size'] * lidar2image_view_num
                                )      
                            else:
                                output_sparse = replace_feature(output_sparse, output_extended)
                            if i != self.lidar2image_end - 1:
                                residual_extended = output_extended[ori_len:].clone()
                            if self.use_profiler:
                                with torch.profiler.record_function("block_{}".format(i)):
                                    output_extended, output_sparse = self.forward_stage2(block, output_sparse, lidar2image_coor, batch_dict, i)
                            else:
                                output_extended, output_sparse = self.forward_stage2(block, output_sparse, lidar2image_coor, batch_dict, i)
                                    
                        elif self.use_multi_scale:
                            if i == self.lidar2image_start:
                                use_conv_list = ['x_conv3']
                                multi_scale_3d_features = batch_dict['multi_scale_3d_features']
                                extends = torch.cat([multi_scale_3d_features[conv_name].features for conv_name in use_conv_list], dim=1)
                                lidar2image_coor = torch.cat([lidar2image_coor, torch.cat(lidar2image_coords_bzyx_list)], dim=0)
                                output_extended = torch.cat([output, extends], dim=0)
                                ori_len = output.shape[0]
                                output_sparse = spconv.SparseConvTensor(
                                    features=output_extended,
                                    indices=lidar2image_coor.int(),
                                    spatial_shape=self.shape_inter2,
                                    batch_size=batch_dict['batch_size'] * lidar2image_view_num
                                )
                            else:
                                output_sparse = replace_feature(output_sparse, output_extended)
                            
                            if i != self.lidar2image_end - 1:
                                residual_extended = output_extended[ori_len:].clone()

                            output_extended, output_sparse = self.forward_stage2(block, output_sparse, lidar2image_coor, batch_dict, i)


                        else:
                            if i == self.lidar2image_start:
                                if self.mask_out_img:
                                    out_put_ori = output.clone()
                                    selected_indices = (lidar2image_coor[:, 2] >= 88) & (lidar2image_coor[:, 2] < 176) & (lidar2image_coor[:, 3] >= 32) & (lidar2image_coor[:, 3] < 64)
                                    lidar2image_coor = lidar2image_coor[selected_indices]
                                    output = output[selected_indices]
                                    lidar2image_coor[:, 2] -= 88
                                    lidar2image_coor[:, 3] -= 32
                                    output_sparse = spconv.SparseConvTensor(
                                        features=output,
                                        indices=lidar2image_coor.int(),
                                        spatial_shape=self.shape_inter2,
                                        batch_size=batch_dict['batch_size'] * lidar2image_view_num
                                    )
                                else:
                                    output_sparse = spconv.SparseConvTensor(
                                        features=output,
                                        indices=lidar2image_coor.int(),
                                        spatial_shape=self.shape_inter2,
                                        batch_size=batch_dict['batch_size'] * lidar2image_view_num
                                    )
                            else: # 由于进行了norm以及融合了residual，所以这里需要将output_sparse的features替换为output
                                if self.mask_out_img:
                                    output_sparse = replace_feature(output_sparse, output[selected_indices])
                                else:
                                    output_sparse = replace_feature(output_sparse, output)
                            
                            if self.use_fixed_mapping:
                                if i == self.lidar2image_start:
                                    fixed_mapping_list = None
                                output, output_sparse, fixed_mapping_list = self.forward_stage2(block, output_sparse, lidar2image_coor, batch_dict, i, fixed_mapping_list)
                            else:
                                output, output_sparse = self.forward_stage2(block, output_sparse, lidar2image_coor.int(), batch_dict, i, voxel_num=voxel_num)
                            if self.mask_out_img:
                                out_put_ori[selected_indices] = output
                                output = out_put_ori



                    else:
                        output = block(output, lidar2image_inds_list[stage_id], lidar2image_masks_list[stage_id], multi_pos_embed_list[stage_id][i],
                                    block_id=block_id, voxel_num=voxel_num, using_checkpoint=block_id in self.checkpoint_blocks)
                    if self.use_vmamba and self.use_in_mid:
                        output = self.do_vmamba_inter(output, voxel_num, batch_dict, block_id)
                else: # i == 0 模态内注意力机制 反正也是分开进行的，所以这里可以分开？
                    if self.use_vmamba and self.use_in_mid:
                        output = output + multi_pos_embeding

                    if self.use_winmamba:
                        # 原版MambaFusion逻辑：patch_coords和voxel_coords的batch索引已经混在一起
                        # 直接合并即可，不需要额外的偏移处理
                        image_coords = batch_dict['patch_coords'].clone()
                        image_coords[:, 0] = image_coords[:, 0] + batch_dict['batch_size']
                        indices = torch.cat([batch_dict['voxel_coords'], image_coords], dim=0)
                        h_size = self.inter1_win_shape[-1]
                        # 计算实际的batch数量：包括lidar和所有camera views
                        actual_batch_size = batch_dict['batch_size'] * (1 + int(batch_dict['camera_imgs'].shape[1]))
                        output_sparse = spconv.SparseConvTensor(
                            features=output,
                            indices=indices.int(),
                            spatial_shape=[h_size, self.bev_size_H, self.bev_size_W],
                            batch_size=actual_batch_size
                        )
                        for j, b in enumerate(block):
                            if isinstance(b, LocalMamba):
                                output_sparse = b(output_sparse)
                                if j == len(block) - 1:
                                    output = output_sparse.features
                            elif isinstance(b, GlobalMamba):
                                output, _ = b(output_sparse.features, indices, batch_dict['batch_size'], [h_size, self.bev_size_H, self.bev_size_W],
                                    self.curve_template, self.hilbert_spatial_size, self.pos_embed_intra, 0, False)
                                if not (j == len(block) - 1):
                                    output_sparse = replace_feature(output_sparse, output)
                            else:
                                raise ValueError("Block type not supported")
                            # if j % 2 == 0:
                            #     output_sparse = b(output_sparse)
                            # else:
                            #     output, _ = b(output_sparse.features, indices, batch_dict['batch_size'], [1, 360, 360],
                            #         self.curve_template, self.hilbert_spatial_size, self.pos_embed_intra, 0, False)


                    else:
                        output = block(output, multi_set_voxel_inds_list[stage_id], multi_set_voxel_masks_list[stage_id], multi_pos_embed_list[stage_id][i],
                                    block_id=block_id, voxel_num=voxel_num, using_checkpoint=block_id in self.checkpoint_blocks)

                        
                    # if self.use_vmamba and self.use_in_mid:
                    #     output = self.do_vmamba_intra(output, voxel_num, batch_dict, block_id)
                # use different norm for lidar and image
                if self.model_cfg.layer_cfg.get('split_residual', False): # True 分别进行两个模态的norm
                    if 'output_extended' in locals():
                        output = output_extended[:ori_len]
                        if i != self.lidar2image_end - 1:
                            extended = self.multi_scale_norm_list[i-self.lidar2image_start](output_extended[ori_len:] + residual_extended)
                    output = torch.cat([self.__getattr__(f'lidar_residual_norm_stage_{stage_id}')[i](output[:voxel_num] + residual[:voxel_num]),
                                       residual_norm_layers[i](output[voxel_num:] + residual[voxel_num:])], dim=0)
                    if 'output_extended' in locals() and i != self.lidar2image_end - 1:
                        output_extended = torch.cat([output, extended], dim=0)
                    
                else:
                    output = residual_norm_layers[i](output + residual)
                block_id += 1
                # recover image feature shape
                if i in self.out_indices: # []
                    if i == self.out_indices[-1] and self.use_vmamba and self.use_after:

                        # output_new = output + multi_pos_embeding
                        output_new_self_list = []
                        output_new_cross_list = []
                        for block_id in range(len(self.vmamba_blocks)):
                            if self.twin_flag[block_id]:
                                output_new_self_list.append(self.do_vmamba_intra(output, voxel_num, batch_dict, block_id, multi_pos_embeding))
                            else:
                                output_new_cross_list.append(self.do_vmamba_inter(output, voxel_num, batch_dict, block_id, multi_pos_embeding))
                        # output_new_self = self.do_vmamba_intra(output, voxel_num, batch_dict, 0, multi_pos_embeding)
                        # output_new_cross = self.do_vmamba_inter(output, voxel_num, batch_dict, 1, multi_pos_embeding)
                        # output_new_self2 = self.do_vmamba_intra(output_new_self, voxel_num, batch_dict, 2, multi_pos_embeding)
                        # output_new_cross2 = self.do_vmamba_inter(output_new_self, voxel_num, batch_dict, 3, multi_pos_embeding)
                        # output_new = self.do_vmamba_intra(output_new, voxel_num, batch_dict, 2)
                        # output_new = self.do_vmamba_inter(output_new, voxel_num, batch_dict, 3)

                        # lidar_part = torch.cat([output[:voxel_num], output_new_self[:voxel_num], output_new_cross[:voxel_num], output_new_self2[:voxel_num], output_new_cross2[:voxel_num]], dim=-1).contiguous()
                        lidar_part = torch.cat([output[:voxel_num]] + [output_new_self[:voxel_num] for output_new_self in output_new_self_list] + [output_new_cross[:voxel_num] for output_new_cross in output_new_cross_list], dim=-1).contiguous()
                        lidar_part = self.lidar_fc(lidar_part)

                        if self.use_conv:
                            batch_spatial_features_new_self_list = [self._recover_image(pillar_features=output_new_self_list[block_id][voxel_num:],
                                                                            coords=patch_info[f'voxel_coors_stage{self.stage_num - 1}'], indices=i+1+block_id)
                                                                            for block_id in range(len(output_new_self_list))]
                            batch_spatial_features_new_cross_list = [self._recover_image(pillar_features=output_new_cross_list[block_id][voxel_num:],
                                                                            coords=patch_info[f'voxel_coors_stage{self.stage_num - 1}'], indices=i+1+block_id + len(output_new_self_list))
                                                                            for block_id in range(len(output_new_cross_list))]
                            # batch_spatial_features_new_self = self._recover_image(pillar_features=output_new_self[voxel_num:],
                            #                                             coords=patch_info[f'voxel_coors_stage{self.stage_num - 1}'], indices=i + 1)
                            # batch_spatial_features_new_cross = self._recover_image(pillar_features=output_new_cross[voxel_num:],
                            #                                             coords=patch_info[f'voxel_coors_stage{self.stage_num - 1}'], indices=i + 2)
                            # batch_spatial_features_new_self2 = self._recover_image(pillar_features=output_new_self2[voxel_num:],
                            #                                             coords=patch_info[f'voxel_coors_stage{self.stage_num - 1}'], indices=i + 3)
                            # batch_spatial_features_new_cross2 = self._recover_image(pillar_features=output_new_cross2[voxel_num:],
                            #                                             coords=patch_info[f'voxel_coors_stage{self.stage_num - 1}'], indices=i + 4)

                            batch_spatial_features = self._recover_image(pillar_features=output[voxel_num:],
                                                                        coords=patch_info[f'voxel_coors_stage{self.stage_num - 1}'], indices=i)

                            # batch_spatial_features = torch.cat([batch_spatial_features, batch_spatial_features_new_self, batch_spatial_features_new_cross], dim=1) # [24, 128, 32, 88]
                            # batch_spatial_features = torch.cat([batch_spatial_features, batch_spatial_features_new_self, batch_spatial_features_new_cross, batch_spatial_features_new_self2, batch_spatial_features_new_cross2], dim=1) # [24, 128, 32, 88]
                            batch_spatial_features = torch.cat([batch_spatial_features] + batch_spatial_features_new_self_list + batch_spatial_features_new_cross_list, dim=1) # [24, 128, 32, 88]
                            batch_spatial_features = self.img_fc(batch_spatial_features)
                            image_part = batch_spatial_features.permute(0, 2, 3, 1).contiguous().view(-1, 128)
                            output = torch.cat([lidar_part, image_part], dim=0)
                        else:
                            image_part = torch.cat([output[voxel_num:], output_new_self[voxel_num:], output_new_cross[voxel_num:]], dim=-1).contiguous()
                            image_part = self.img_fc(image_part)
                            output = torch.cat([lidar_part, image_part], dim=0)
                            batch_spatial_features = self._recover_image(pillar_features=output[voxel_num:],
                                                                    coords=patch_info[f'voxel_coors_stage{self.stage_num - 1}'], indices=i)
                        
                        
                    else:
                        batch_spatial_features = self._recover_image(pillar_features=output[voxel_num:], # 前面分开norm，这里+1？
                                                                    coords=patch_info[f'voxel_coors_stage{self.stage_num - 1}'], indices=i)
                    batch_dict['image_features'].append(batch_spatial_features) # [24, 128, 32, 88]
       
        if self.use_winmamba and h_size != 1:
            lidar_sparse = spconv.SparseConvTensor(
                            features=output[:voxel_num],
                            indices=batch_dict['voxel_coords'],
                            spatial_shape=[h_size, self.bev_size_H, self.bev_size_W],  # [Z, Y, X] 格式
                            batch_size=batch_dict['batch_size']
                        )
            lidar_sparse, _ = self.dow5(lidar_sparse)
            batch_dict['pillar_features'] = lidar_sparse.features
            batch_dict['voxel_coords'] = lidar_sparse.indices
        else:
            batch_dict['pillar_features'] = batch_dict['voxel_features'] = output[:voxel_num]
            batch_dict['voxel_coords'] = voxel_info[f'voxel_coors_stage{self.stage_num - 1}']
        end_time = time.time()
        print(f"Voxel number: {batch_dict['voxel_num']}")
        print(f"Image number: {lidar2image_view_num}")
        print(f"Time taken: {end_time - start_time} seconds")
        if outer_dict is not None:
            outer_dict[agent] = batch_dict
            return outer_dict
        return batch_dict
    def do_vmamba_inter(self, output, voxel_num, batch_dict, block_id, embeding=None):
        img_num_per_frame = 16896
        img_part = output[voxel_num:, :]
        img_part = img_part.reshape(batch_dict['batch_size'], img_num_per_frame, 128)
        point_part = output[:voxel_num, :]
        new_img_part_frame_list = []
        new_point_part_frame_list = []
        if embeding is not None:
            point_embeding = embeding[:voxel_num]
            img_embeding = embeding[voxel_num:]
        for batch_id in range(batch_dict['batch_size']):
            current_index = batch_dict['voxel_coords'][:, 0] == batch_id
            point_part_frame = point_part[current_index]
            multi_part_frame = torch.cat([point_part_frame, img_part[batch_id]], dim=0).contiguous()
            if embeding is not None:
                point_part_frame_with_embed = point_part_frame + point_embeding[current_index]
                img_part_frame_with_embed = img_part[batch_id] + img_embeding[batch_id]
                multi_part_frame_with_embed = torch.cat([point_part_frame_with_embed, img_part_frame_with_embed], dim=0).contiguous()
                multi_part_frame_new = self.vmamba_blocks[block_id]((multi_part_frame, multi_part_frame_with_embed)).squeeze()
            else:
                multi_part_frame_new = self.vmamba_blocks[block_id](multi_part_frame).squeeze()
            new_point_part_frame = multi_part_frame_new[:point_part_frame.shape[0]]
            new_img_part_frame = multi_part_frame_new[point_part_frame.shape[0]:]
            new_img_part_frame_list.append(new_img_part_frame)
            new_point_part_frame_list.append(new_point_part_frame)
            
        output_new = torch.cat(new_point_part_frame_list + new_img_part_frame_list, dim=0)
        return output_new
    def _extract_camera_imgs(self, batch_dict):
        """Normalize image source into batch_dict['camera_imgs'] with shape [B, N, C, H, W].
        Supports per-agent batch format with nested 'batch_merged_cam_inputs'.
        """
        if 'camera_imgs' in batch_dict:
            imgs = batch_dict['camera_imgs']
        elif 'batch_merged_cam_inputs' in batch_dict and isinstance(batch_dict['batch_merged_cam_inputs'], dict):
            cam_inputs = batch_dict['batch_merged_cam_inputs']
            if 'imgs' in cam_inputs and cam_inputs['imgs'] is not None:
                imgs = cam_inputs['imgs']  # expected [B, N, C, H, W]
                # 检查图像数据是否有效
                if imgs.numel() == 0 or torch.count_nonzero(imgs).item() == 0:
                    print(f"[MambaFusion] 检测到无效的图像数据，跳过该agent的图像处理")
                    raise ValueError("Invalid image data: empty or all zeros")
            else:
                print(f"[MambaFusion] 没有找到有效的图像数据，跳过该agent的图像处理")
                raise ValueError("No valid image data found")
        else:
            print(f"[MambaFusion] 没有找到相机数据，跳过该agent的图像处理")
            raise KeyError("No camera images found in batch_dict: expected 'camera_imgs' or 'batch_merged_cam_input")

        # Ensure 5D and channel-first within per-view
        if imgs.dim() != 5:
            raise ValueError(f"camera imgs must be 5D [B,N,C,H,W], got shape {tuple(imgs.shape)}")
        B, N, C, H, W = imgs.shape
        # Some loaders may provide 4 channels; keep first 3 (RGB)
        if C > 3:
            imgs = imgs[:, :, :3, ...]
            C = 3
        # Optional: interpret leading B as number of agents and collapse to one batch with B*N views
        agents_as_views = batch_dict.get('agents_as_views', None)
        if agents_as_views is None:
            agents_as_views = getattr(self.model_cfg, 'AGENTS_AS_VIEWS', False)
        if agents_as_views:
            imgs = imgs.view(1, B * N, C, H, W)
            batch_dict['batch_size'] = 1
            B_eff, N_eff = 1, B * N
            batch_dict['camera_imgs'] = imgs
            return imgs, (B_eff, N_eff, C, H, W)
        else:
            batch_dict['camera_imgs'] = imgs
            return imgs, (B, N, C, H, W)
    def do_vmamba_intra(self, output, voxel_num, batch_dict, block_id, embeding=None):
        img_num_per_frame = 16896
        img_part = output[voxel_num:, :]
        img_part = img_part.reshape(batch_dict['batch_size'], img_num_per_frame, 128)
        point_part = output[:voxel_num, :]
        new_img_part_frame_list = []
        new_point_part_frame_list = []
        len_vmamba = len(self.vmamba_blocks[block_id])
        if embeding is not None:
            point_embeding = embeding[:voxel_num]
            img_embeding = embeding[voxel_num:].reshape(batch_dict['batch_size'], img_num_per_frame, 128)
        for batch_id in range(batch_dict['batch_size']):
            current_index = batch_dict['voxel_coords'][:, 0] == batch_id
            point_part_frame = point_part[current_index] # [num_of_voxel, 128]
            if embeding is not None:
                point_part_frame_with_embed = point_part_frame + point_embeding[current_index]
                new_point_part_frame = self.vmamba_blocks[block_id].blocks1((point_part_frame, point_part_frame_with_embed)).squeeze()
                img_part_frame_with_embed = img_part[batch_id] + img_embeding[batch_id]
                new_img_part_frame =  self.vmamba_blocks[block_id].blocks2((img_part[batch_id], img_part_frame_with_embed)).squeeze()
            else:
                new_point_part_frame = self.vmamba_blocks[block_id].blocks1(point_part_frame).squeeze()
                new_img_part_frame =  self.vmamba_blocks[block_id].blocks2(img_part[batch_id]).squeeze()
            new_img_part_frame_list.append(new_img_part_frame)
            new_point_part_frame_list.append(new_point_part_frame)
            
        output_new = torch.cat(new_point_part_frame_list + new_img_part_frame_list, dim=0) # [num_of_voxel+num_of_patch, 128]
        return output_new
    def _input_preprocess(self, batch_dict):
        # image branch
        imgs, (B, N, C, H, W) = self._extract_camera_imgs(batch_dict) # normalize image source
        imgs = imgs.view(B * N, C, H, W) # 12, 3, 256, 704
        if self.use_prebackbone:
            imgs = batch_dict['backbone_features']
            imgs = imgs.permute(0, 2, 3, 1).contiguous().view(B * N, -1, 128) # [6, 2816, 128]
            hw_shape = [self.image_shape[0] // 8, self.image_shape[1] // 8] # [32, 88]
        elif self.use_vmamba_pretrain:

            imgs_list = checkpoint(self.backbone_vssm, imgs, use_reentrant=False)
            if self.use_multi_scalev:
                batch_dict['multi_scalev_features'] = imgs_list[0]
                batch_dict['multi_scalev_features'] = batch_dict['multi_scalev_features'].permute(0, 2, 3, 1).contiguous().view(-1, 128)
                # batch_dict['multi_scalev_features_coords'] = None
            if self.use_multi_scalev_down:
                down_imgs = imgs_list[-1]
                down_imgs = self.vssm_multi_scalev_down_block(down_imgs)
                batch_dict['multi_scalev_down_features'] = down_imgs.permute(0, 2, 3, 1).contiguous().view(-1, 128)
                # batch_dict['multi_scalev_down_features_coords'] = None
            imgs = imgs_list[1] if self.use_multi_scalev else imgs_list[0]
            imgs = self.vssm_down_block(imgs) # [6, 128, 32, 88]

            imgs = imgs.permute(0, 2, 3, 1).contiguous().view(B * N, -1, 128) # [6, 2816, 128]
            hw_shape = [self.image_shape[0] // 8, self.image_shape[1] // 8] # [32, 88]
        else:
            imgs, hw_shape = self.patch_embed(imgs)  # [12, 2816, 128] (32, 88) 将图像转换为patch 2816个patch，每个patch 128维
        if self.use_cross_img:
            sweeps_imgs = batch_dict['img_sweeps']
            T = batch_dict['img_sweeps'].shape[2]
            sweeps_imgs = sweeps_imgs.view(B * N * T, C, H, W)
            with torch.no_grad():
                if self.use_vmamba_pretrain:
                    sweeps_imgs = self.backbone_vssm(sweeps_imgs)[0]
                    # sweeps_imgs = checkpoint(self.backbone_vssm, sweeps_imgs)[0]
                    sweeps_imgs = self.vssm_down_block(sweeps_imgs) 
                    sweeps_imgs = sweeps_imgs.permute(0, 2, 3, 1).contiguous().view(B * N, T, -1, 128) # [6, 2, 2816, 128]
                else:
                    sweeps_imgs, _ = self.patch_embed(sweeps_imgs) # [12, 2816, 128])
                    sweeps_imgs = sweeps_imgs.view(B * N, T, sweeps_imgs.shape[1], 128) # [6, 2, 2816, 128]


        batch_dict['hw_shape'] = hw_shape

        # 36*2816, C
        batch_dict['patch_features'] = imgs.view(-1, imgs.shape[-1]) # [num_of_patch, 128]
        if self.patch_coords is not None and ((self.patch_coords[:, 0].max().int().item() + 1) == B*N):
            batch_dict['patch_coords'] = self.patch_coords.clone() # [num_of_patch, 4] image_id z y x
            if self.use_multi_scalev:
                batch_dict['multi_scalev_features_coords'] = self.new_patch_coords.clone()
            if self.use_multi_scalev_down:
                batch_dict['multi_scalev_down_features_coords'] = self.new_patch_coords_down.clone()
        else:
            batch_idx = torch.arange(
                B*N, device=imgs.device).unsqueeze(1).repeat(1, hw_shape[0] * hw_shape[1]).view(-1, 1)
            batch_dict['patch_coords'] = torch.cat([batch_idx, self.patch_zyx.clone().to(imgs.device)[
                                                   None, ::].repeat(B*N, 1, 1).view(-1, 3)], dim=-1).long()
            self.patch_coords = batch_dict['patch_coords'].clone()
            if self.use_multi_scalev:
                new_hw_shape = [size * 2 for size in hw_shape]
                new_batch_idx = torch.arange(
                    B*N, device=imgs.device).unsqueeze(1).repeat(1, new_hw_shape[0] * new_hw_shape[1]).view(-1, 1)
                new_patch_x, new_patch_y = torch.meshgrid(
                    torch.arange(new_hw_shape[0], device=imgs.device), torch.arange(new_hw_shape[1], device=imgs.device))
                new_patch_z = torch.zeros((new_hw_shape[0] * new_hw_shape[1], 1), device=imgs.device)
                new_patch_zyx = torch.cat([new_patch_z, new_patch_y.reshape(-1, 1), new_patch_x.reshape(-1, 1)], dim=-1)
                self.new_patch_coords = torch.cat([new_batch_idx, new_patch_zyx[
                                                    None, ::].repeat(B*N, 1, 1).view(-1, 3)], dim=-1).long()
                self.new_patch_coords[:, -1] += new_hw_shape[0]
                self.new_patch_coords[:, -2] += new_hw_shape[1]
                batch_dict['multi_scalev_features_coords'] = self.new_patch_coords.clone()
            if self.use_multi_scalev_down:
                new_hw_shape = [size // 2 for size in hw_shape]
                new_batch_idx = torch.arange(
                    B*N, device=imgs.device).unsqueeze(1).repeat(1, new_hw_shape[0] * new_hw_shape[1]).view(-1, 1)
                new_patch_x, new_patch_y = torch.meshgrid(
                    torch.arange(new_hw_shape[0], device=imgs.device), torch.arange(new_hw_shape[1], device=imgs.device))
                new_patch_z = torch.zeros((new_hw_shape[0] * new_hw_shape[1], 1), device=imgs.device)
                new_patch_zyx = torch.cat([new_patch_z, new_patch_y.reshape(-1, 1), new_patch_x.reshape(-1, 1)], dim=-1)
                self.new_patch_coords_down = torch.cat([new_batch_idx, new_patch_zyx[
                                                    None, ::].repeat(B*N, 1, 1).view(-1, 3)], dim=-1).long()
                self.new_patch_coords_down[:, -2:] *= 2
                self.new_patch_coords_down[:, -1] += new_hw_shape[0] * 2
                self.new_patch_coords_down[:, -2] += new_hw_shape[1] * 2
                batch_dict['multi_scalev_down_features_coords'] = self.new_patch_coords_down.clone()
        patch_info = self.image_input_layer(batch_dict)
        patch_feat = batch_dict['patch_features'] # [num_of_patch, 128]
        patch_set_voxel_inds_list = [[patch_info[f'set_voxel_inds_stage{s}_shift{i}'] # 每个set里voxel按照xy排序后的index 每个元素 [2, num_of_set, 90]
                                      for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
        patch_set_voxel_masks_list = [[patch_info[f'set_voxel_mask_stage{s}_shift{i}'] # 重复voxel的mask 每个元素 [2, num_of_set, 90]
                                       for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
        patch_pos_embed_list = [[[patch_info[f'pos_embed_stage{s}_block{b}_shift{i}'] # 每个set里每个block的pos_embed 每个元素 [num_of_voxel, 128]
                                  for i in range(self.num_shifts[s])] for b in range(self.image_pos_num)] for s in range(len(self.set_info))]

        # lidar branch
        voxel_info = self.lidar_input_layer(batch_dict)
        voxel_feat = batch_dict['voxel_features']
        # 将来自 BACKBONE_3D 的 64 维体素特征映射到 128 维，以与 d_model[0]=128 对齐
        if voxel_feat.shape[1] == 64:
            voxel_feat = self.voxel_proj(voxel_feat)
        set_voxel_inds_list = [[voxel_info[f'set_voxel_inds_stage{s}_shift{i}']
                                for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
        set_voxel_masks_list = [[voxel_info[f'set_voxel_mask_stage{s}_shift{i}']
                                 for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
        pos_embed_list = [[[voxel_info[f'pos_embed_stage{s}_block{b}_shift{i}']
                            for i in range(self.num_shifts[s])] for b in range(self.lidar_pos_num)] for s in range(len(self.set_info))]

        # multi-modality parallel
        voxel_num = voxel_feat.shape[0] # num_of_voxels
        batch_dict['voxel_num'] = voxel_num
        multi_feat = torch.cat([voxel_feat, patch_feat], dim=0) # [num_of_voxels+num_of_patch, 128]
        
        multi_set_voxel_inds_list = [[torch.cat([set_voxel_inds_list[s][i], patch_set_voxel_inds_list[s][i]+voxel_num], dim=1)
                                        for i in range(self.num_shifts[s])] for s in range(len(self.set_info))] # 每个set里voxel按照xy排序后的index 每个元素 [2, num_of_set(voxel)+num_of_set(patch), 90]
        multi_set_voxel_masks_list = [[torch.cat([set_voxel_masks_list[s][i], patch_set_voxel_masks_list[s][i]], dim=1)
                                       for i in range(self.num_shifts[s])] for s in range(len(self.set_info))] # 重复voxel的mask 每个元素 [2, num_of_set(voxel)+num_of_set(patch), 90]
        multi_pos_embed_list = []
        for s in range(len(self.set_info)): # 1
            block_pos_embed_list = []
            for b in range(self.set_info[s][1]): # 4
                shift_pos_embed_list = []
                for i in range(self.num_shifts[s]): # 2
                    if b < self.lidar_pos_num and b < self.image_pos_num: # ? 好像必然是这个branch
                        if (self.use_mamba_inter2 and b in self.inter2_block ) or (self.use_mamba_inter and b in self.inter_block) or (self.use_winmamba and b == 0):
                            shift_pos_embed_list.append([])
                        else:
                            shift_pos_embed_list.append(
                                torch.cat([pos_embed_list[s][b][i], patch_pos_embed_list[s][b][i]], dim=0)) # [num_of_voxel+num_of_patch, 128]
                    elif b < self.lidar_pos_num and b >= self.image_pos_num:
                        shift_pos_embed_list.append(pos_embed_list[s][b][i])
                    elif b >= self.lidar_pos_num and b < self.image_pos_num:
                        shift_pos_embed_list.append(
                            patch_pos_embed_list[s][b][i])
                    else:
                        raise NotImplementedError
                block_pos_embed_list.append(shift_pos_embed_list)
            multi_pos_embed_list.append(block_pos_embed_list)
        if self.use_cross_img: # multi_set_voxel_inds_list[0][1][:, set_voxel_inds_list[0][0].shape[1]:,: ] - batch_dict['voxel_num']
            return multi_feat, voxel_info, patch_info, multi_set_voxel_inds_list, multi_set_voxel_masks_list, multi_pos_embed_list, sweeps_imgs, [set_voxel_inds_list[0][i].shape[1] for i in range(self.num_shifts[0])]
        else:
            return multi_feat, voxel_info, patch_info, multi_set_voxel_inds_list, multi_set_voxel_masks_list, multi_pos_embed_list

    def _input_preprocess2(self, batch_dict):
        # image branch
        imgs, (B, N, C, H, W) = self._extract_camera_imgs(batch_dict)
        imgs = imgs.view(B * N, C, H, W) # 12, 3, 256, 704
        if self.use_prebackbone:
            imgs = batch_dict['backbone_features']
            imgs = imgs.permute(0, 2, 3, 1).contiguous().view(B * N, -1, 128) # [6, 2816, 128]
            hw_shape = [self.image_shape[0] // 8, self.image_shape[1] // 8] # [32, 88]
        elif self.use_vmamba_pretrain:

            if self.training:
                imgs_list = checkpoint(self.backbone_vssm, imgs, use_reentrant=False)
            else:
                imgs_list = self.backbone_vssm(imgs)
            if self.use_more_vbackbone:
                imgs_list = self.vssm_fpn(imgs_list)
            if self.use_multi_scalev:
                batch_dict['multi_scalev_features'] = imgs_list[0]
                batch_dict['multi_scalev_features'] = batch_dict['multi_scalev_features'].permute(0, 2, 3, 1).contiguous().view(-1, 128)
                # batch_dict['multi_scalev_features_coords'] = None
            if self.use_multi_scalev_down:
                down_imgs = imgs_list[-1]
                down_imgs = self.vssm_multi_scalev_down_block(down_imgs)
                batch_dict['multi_scalev_down_features'] = down_imgs.permute(0, 2, 3, 1).contiguous().view(-1, 128)
                # batch_dict['multi_scalev_down_features_coords'] = None
            imgs = imgs_list[1] if self.use_multi_scalev else imgs_list[0]
            imgs = self.vssm_down_block(imgs) # [6, 128, 32, 88]

            imgs = imgs.permute(0, 2, 3, 1).contiguous().view(B * N, -1, 128) # [6, 2816, 128]
            hw_shape = [self.image_shape[0] // 8, self.image_shape[1] // 8] # [32, 88]
        else:
            imgs, hw_shape = self.patch_embed(imgs)  # [12, 2816, 128] (32, 88) 将图像转换为patch 2816个patch，每个patch 128维

        batch_dict['hw_shape'] = hw_shape

        # 36*2816, C
        batch_dict['patch_features'] = imgs.view(-1, imgs.shape[-1]) # [num_of_patch, 128]
        if self.patch_coords is not None and ((self.patch_coords[:, 0].max().int().item() + 1) == B*N):
            batch_dict['patch_coords'] = self.patch_coords.clone() # [num_of_patch, 4] image_id z y x
            if self.use_multi_scalev:
                batch_dict['multi_scalev_features_coords'] = self.new_patch_coords.clone()
            if self.use_multi_scalev_down:
                batch_dict['multi_scalev_down_features_coords'] = self.new_patch_coords_down.clone()
        else:
            batch_idx = torch.arange(
                B*N, device=imgs.device).unsqueeze(1).repeat(1, hw_shape[0] * hw_shape[1]).view(-1, 1)
            batch_dict['patch_coords'] = torch.cat([batch_idx, self.patch_zyx.clone().to(imgs.device)[
                                                   None, ::].repeat(B*N, 1, 1).view(-1, 3)], dim=-1).long()
            self.patch_coords = batch_dict['patch_coords'].clone()
            if self.use_multi_scalev:
                new_hw_shape = [size * 2 for size in hw_shape]
                new_batch_idx = torch.arange(
                    B*N, device=imgs.device).unsqueeze(1).repeat(1, new_hw_shape[0] * new_hw_shape[1]).view(-1, 1)
                new_patch_x, new_patch_y = torch.meshgrid(
                    torch.arange(new_hw_shape[0], device=imgs.device), torch.arange(new_hw_shape[1], device=imgs.device))
                new_patch_z = torch.zeros((new_hw_shape[0] * new_hw_shape[1], 1), device=imgs.device)
                new_patch_zyx = torch.cat([new_patch_z, new_patch_y.reshape(-1, 1), new_patch_x.reshape(-1, 1)], dim=-1)
                self.new_patch_coords = torch.cat([new_batch_idx, new_patch_zyx[
                                                    None, ::].repeat(B*N, 1, 1).view(-1, 3)], dim=-1).long()
                self.new_patch_coords[:, -1] += new_hw_shape[0]
                self.new_patch_coords[:, -2] += new_hw_shape[1]
                batch_dict['multi_scalev_features_coords'] = self.new_patch_coords.clone()
            if self.use_multi_scalev_down:
                new_hw_shape = [size // 2 for size in hw_shape]
                new_batch_idx = torch.arange(
                    B*N, device=imgs.device).unsqueeze(1).repeat(1, new_hw_shape[0] * new_hw_shape[1]).view(-1, 1)
                new_patch_x, new_patch_y = torch.meshgrid(
                    torch.arange(new_hw_shape[0], device=imgs.device), torch.arange(new_hw_shape[1], device=imgs.device))
                new_patch_z = torch.zeros((new_hw_shape[0] * new_hw_shape[1], 1), device=imgs.device)
                new_patch_zyx = torch.cat([new_patch_z, new_patch_y.reshape(-1, 1), new_patch_x.reshape(-1, 1)], dim=-1)
                self.new_patch_coords_down = torch.cat([new_batch_idx, new_patch_zyx[
                                                    None, ::].repeat(B*N, 1, 1).view(-1, 3)], dim=-1).long()
                self.new_patch_coords_down[:, -2:] *= 2
                self.new_patch_coords_down[:, -1] += new_hw_shape[0] * 2
                self.new_patch_coords_down[:, -2] += new_hw_shape[1] * 2
                batch_dict['multi_scalev_down_features_coords'] = self.new_patch_coords_down.clone()
        patch_info = {f'voxel_coors_stage0': batch_dict['patch_coords']}
        patch_feat = batch_dict['patch_features'] # [num_of_patch, 128]
        

        # lidar branch
        voxel_info = {f'voxel_coors_stage0': batch_dict['voxel_coords']}
        voxel_feat = batch_dict['voxel_features']
        
        # 将来自 BACKBONE_3D 的体素特征统一到 128 维：
        # - 如果是 64 维（与 VFE/BACKBONE_3D 配置一致），使用线性层升到 128
        # - 如果是 256 维，沿用原逻辑截取前 128 维
        if voxel_feat.shape[1] == 64:
            voxel_feat = self.voxel_proj(voxel_feat)
        elif voxel_feat.shape[1] == 256:
            voxel_feat = voxel_feat[:, :128]  # Take first 128 dimensions

        # multi-modality parallel
        voxel_num = voxel_feat.shape[0] # num_of_voxels
        batch_dict['voxel_num'] = voxel_num
        multi_feat = torch.cat([voxel_feat, patch_feat], dim=0) # [num_of_voxels+num_of_patch, 128]

        return multi_feat, voxel_info, patch_info


    def _image2lidar_preprocess(self, batch_dict, multi_feat, multi_pos_embed_list):
        N = batch_dict['camera_imgs'].shape[1] # 6
        voxel_num = batch_dict['voxel_num'] # num_of_voxels
        image2lidar_coords_zyx, nearest_dist = self.map_image2lidar_layer(
            batch_dict) # [num_of_patch, 3] [num_of_patch, 1] 最近的3D点的坐标和距离
        image2lidar_coords_bzyx = torch.cat(
            [batch_dict['patch_coords'][:, :1].clone(), image2lidar_coords_zyx], dim=1) # [num_of_patch, 4] image_id z y x
        image2lidar_coords_bzyx[:, 0] = image2lidar_coords_bzyx[:, 0] // N # torch.Size([33792, 4]) batch_id z y x
        if not (self.use_mamba_inter and not self.use_mixed):
            image2lidar_batch_dict = {}
            image2lidar_batch_dict['voxel_features'] = multi_feat.clone() # torch.Size([num_of_voxel + num_of_patch, 128])
            image2lidar_batch_dict['voxel_coords'] = torch.cat( # torch.Size([num_of_voxel + num_of_patch, 4])
                [batch_dict['voxel_coords'], image2lidar_coords_bzyx], dim=0)

            image2lidar_info = self.image2lidar_input_layer(image2lidar_batch_dict) # 得到增强点云的信息 
            image2lidar_inds_list = [[image2lidar_info[f'set_voxel_inds_stage{s}_shift{i}']
                                    for i in range(self.num_shifts[s])] for s in range(len(self.set_info))] # 每个set里voxel按照xy排序后的index 每个元素 [2, num_of_set, 90]
            image2lidar_masks_list = [[image2lidar_info[f'set_voxel_mask_stage{s}_shift{i}']
                                    for i in range(self.num_shifts[s])] for s in range(len(self.set_info))] # 重复voxel的mask 每个元素 [2, num_of_set, 90]
            image2lidar_pos_embed_list = [[[image2lidar_info[f'pos_embed_stage{s}_block{b}_shift{i}']   # 每个set里每个block的pos_embed 每个元素 [num_of_voxel, 128]
                                            for i in range(self.num_shifts[s])] for b in range(self.image2lidar_pos_num)] for s in range(len(self.set_info))]
            image2lidar_neighbor_pos_embed = self.neighbor_pos_embed(nearest_dist) # [num_of_patch, 128]

            for b in range(self.image2lidar_start, self.image2lidar_end): # 3, 4
                for i in range(self.num_shifts[0]): # 2
                    image2lidar_pos_embed_list[0][b - 
                                                self.image2lidar_start][i][voxel_num:] += image2lidar_neighbor_pos_embed # [num_of_patch, 128]
                    multi_pos_embed_list[0][b][i] += image2lidar_pos_embed_list[0][b -
                                                                                self.image2lidar_start][i] # [num_of_voxel + num_of_patch, 128]
        if self.use_mamba_inter:
            if self.use_mixed:
                return image2lidar_inds_list, image2lidar_masks_list, multi_pos_embed_list, image2lidar_batch_dict['voxel_coords']
            else:
                return None, None, multi_pos_embed_list, torch.cat([batch_dict['voxel_coords'], image2lidar_coords_bzyx], dim=0)
        else:

            return image2lidar_inds_list, image2lidar_masks_list, multi_pos_embed_list

    def _lidar2image_preprocess(self, batch_dict, multi_feat, multi_pos_embed_list,outer_dict = None):
        N = batch_dict['camera_imgs'].shape[1] # 6
        hw_shape = batch_dict['hw_shape'] # [32, 88]
        lidar2image_coords_zyx, lidar2image_coords_bzyx_list = self.map_lidar2image_layer(batch_dict, self.use_multi_scale) # [num_of_voxel, 3] view_idx, x, y 点云投影到图像上的坐标 TODO
        lidar2image_coords_bzyx = torch.cat(
            [batch_dict['voxel_coords'][:, :1].clone(), lidar2image_coords_zyx], dim=1) # torch.Size([num_of_voxel, 4])
        # 可视化投影结果（启用可视化以验证投影是否正确）
       
       # self._visualize_lidar2image_projection(batch_dict, lidar2image_coords_zyx, lidar2image_coords_bzyx,outer_dict)
        
        multiview_coords = batch_dict['patch_coords'].clone() # torch.Size([num_of_patch, 4])
        multiview_coords[:, 0] = batch_dict['patch_coords'][:, 0] // N # 得到batch_id
        multiview_coords[:, 1] = batch_dict['patch_coords'][:, 0] % N # 得到view_id

        # DEBUG: 打印multiview_coords在加上hw_shape之前的范围（与原始 MambaFusion 对齐）
        if multiview_coords.shape[0] > 0:
            y_before_min, y_before_max = multiview_coords[:, 2].min().item(), multiview_coords[:, 2].max().item()
            x_before_min, x_before_max = multiview_coords[:, 3].min().item(), multiview_coords[:, 3].max().item()
            # print(f"[mambafusion] multiview_coords加hw_shape前: y=[{y_before_min}, {y_before_max}], x=[{x_before_min}, {x_before_max}], hw_shape={hw_shape}")

        # 与原始 MambaFusion 一致：第 2 列加 hw_shape[1] (width 方向)，第 3 列加 hw_shape[0] (height 方向)
        multiview_coords[:, 2] += hw_shape[1] # width
        multiview_coords[:, 3] += hw_shape[0] # height

        # DEBUG: 打印multiview_coords在加上hw_shape之后的范围
        if multiview_coords.shape[0] > 0:
            y_after_min, y_after_max = multiview_coords[:, 2].min().item(), multiview_coords[:, 2].max().item()
            x_after_min, x_after_max = multiview_coords[:, 3].min().item(), multiview_coords[:, 3].max().item()
            # print(f"[mambafusion] multiview_coords加hw_shape后: y=[{y_after_min}, {y_after_max}], x=[{x_after_min}, {x_after_max}], shape_inter2={self.shape_inter2}")
            # print(f"[mambafusion] multiview_coords超出shape_inter2范围: y>{self.shape_inter2[1]-1}有{(multiview_coords[:, 2] > self.shape_inter2[1]-1).sum().item()}个, x>{self.shape_inter2[2]-1}有{(multiview_coords[:, 3] > self.shape_inter2[2]-1).sum().item()}个")

        # ===== 与原始 MambaFusion 对齐：不做 x/y 交换，直接拼接 =====
        if not (self.use_mamba_inter2 and not self.use_mixed):

            lidar2image_batch_dict = {}
            lidar2image_batch_dict['voxel_features'] = multi_feat.clone() # torch.Size([num_of_voxel + num_of_patch, 128])
            lidar2image_batch_dict['voxel_coords'] = torch.cat(
                [lidar2image_coords_bzyx, multiview_coords], dim=0) # [num_of_voxel + num_of_patch, 4] batch_id view_id x(width) y(height)

            lidar2image_info = self.lidar2image_input_layer(lidar2image_batch_dict)
            lidar2image_inds_list = [[lidar2image_info[f'set_voxel_inds_stage{s}_shift{i}']
                                    for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
            lidar2image_masks_list = [[lidar2image_info[f'set_voxel_mask_stage{s}_shift{i}']
                                    for i in range(self.num_shifts[s])] for s in range(len(self.set_info))]
            lidar2image_pos_embed_list = [[[lidar2image_info[f'pos_embed_stage{s}_block{b}_shift{i}']
                                            for i in range(self.num_shifts[s])] for b in range(self.lidar2image_pos_num)] for s in range(len(self.set_info))]

            for b in range(self.lidar2image_start, self.lidar2image_end):
                for i in range(self.num_shifts[0]):
                    multi_pos_embed_list[0][b][i] += lidar2image_pos_embed_list[0][b -
                                                                               self.lidar2image_start][i]

        if self.use_mamba_inter2:
            if self.use_mixed:
                final_coords = lidar2image_batch_dict['voxel_coords']
            elif self.use_multi_scale:
                final_coords = torch.cat([lidar2image_coords_bzyx, multiview_coords], dim=0)
            else:
                final_coords = torch.cat([lidar2image_coords_bzyx, multiview_coords], dim=0)
            
            # DEBUG: 打印最终返回的坐标范围（已恢复原始格式：[batch_id, view_id, x(width), y(height)]）
            
            # if final_coords.shape[0] > 0:
            #     print(f"[mambafusion] _lidar2image_preprocess 最终返回坐标信息:")
            #     print(f"  shape: {final_coords.shape}")
            #     print(f"  格式: [batch_id, view_id, x(width), y(height)]")
            #     print(f"  x(width)范围: [{final_coords[:, 2].min().item():.1f}, {final_coords[:, 2].max().item():.1f}], shape_inter2[1]={self.shape_inter2[1]}")
            #     print(f"  y(height)范围: [{final_coords[:, 3].min().item():.1f}, {final_coords[:, 3].max().item():.1f}], shape_inter2[2]={self.shape_inter2[2]}")
            #     print(f"  超出shape_inter2范围: x>{self.shape_inter2[1]-1}有{(final_coords[:, 2] > self.shape_inter2[1]-1).sum().item()}个, y>{self.shape_inter2[2]-1}有{(final_coords[:, 3] > self.shape_inter2[2]-1).sum().item()}个")
            #     print(f"  shape_inter2={self.shape_inter2} (格式: [Z, width, height])")

            if self.use_mixed:
                return lidar2image_inds_list, lidar2image_masks_list, multi_pos_embed_list, final_coords
            elif self.use_multi_scale:
                return None, None, multi_pos_embed_list, final_coords, lidar2image_coords_bzyx_list
            else:
                return None, None, multi_pos_embed_list, final_coords
        else:
            return lidar2image_inds_list, lidar2image_masks_list, multi_pos_embed_list

    def _visualize_lidar2image_projection(self, batch_dict, lidar2image_coords_zyx, lidar2image_coords_bzyx,outer_dict = None):
        """
        可视化雷达投影到图像的结果，用于判断投影矩阵是否对齐
        同时把 batch_dict 里的“原始 lidar 点”投影到同一张图上（红色点）。
        """
        import os
        import cv2
        import numpy as np
        import torch

        # ====== 可调开关 ======
        USE_IMAGENET_DENORM = True   # camera_imgs 是否是 ImageNet normalize 后的输入
        USE_AUG = True               # 如果 batch_dict 有 img_aug_matrix，是否把投影后的像素再做一次 aug（对齐增强图）
        RAW_LIDAR_RADIUS = 1         # 红点半径
        VOX_RADIUS = 2               # 体素投影点半径
        MAX_RAW_LIDAR_POINTS = 50000 # 防止太慢，最多画这么多红点（可调）

        self.vis_iter_count += 1
        os.makedirs(self.vis_save_dir, exist_ok=True)

        if 'camera_imgs' not in batch_dict:
            return

        camera_imgs = batch_dict['camera_imgs']  # [B_img, N_img, 3, H, W]
        B_img, N_img, C, H, W = camera_imgs.shape
        assert C == 3, f"camera_imgs channel != 3, got {C}"

        # 真实 view 数量（更可信：lidar2image 的 N）
        if 'lidar2image' in batch_dict:
            B_lidar, N_lidar = batch_dict['lidar2image'].shape[:2]
            B = B_lidar
            N = N_lidar
        else:
            B = B_img
            N = N_img

        # patch / hw_shape
        hw_shape = batch_dict.get('hw_shape', [H // 8, W // 8])
        if isinstance(hw_shape, torch.Tensor):
            hw_shape = hw_shape.detach().cpu().tolist()
        hw_h, hw_w = int(hw_shape[0]), int(hw_shape[1])

        patch_size = 8
        if hasattr(self, "model_cfg") and hasattr(self.model_cfg, "PATCH_EMBED") and hasattr(self.model_cfg.PATCH_EMBED, "patch_size"):
            patch_size = int(self.model_cfg.PATCH_EMBED.patch_size)

        # ====== 坐标转 numpy ======
        bzyx = lidar2image_coords_bzyx.detach().cpu().numpy().astype(np.float32)  # [M,4] b, view, x, y
        zyx = lidar2image_coords_zyx.detach().cpu().numpy().astype(np.float32)    # [M,3] view, x, y（仅 debug）

        # ====== 自适应缩放：把 shape_inter2 空间缩回 hw_shape 空间 ======
        scale_x = 1.0
        scale_y = 1.0
        if hasattr(self, "shape_inter2") and self.shape_inter2 is not None:
            try:
                si = self.shape_inter2
                if isinstance(si, torch.Tensor):
                    si = si.detach().cpu().tolist()
                if len(si) >= 3 and hw_w > 0 and hw_h > 0:
                    si_x = float(si[1])
                    si_y = float(si[2])
                    scale_x = max(1.0, si_x / float(hw_w))
                    scale_y = max(1.0, si_y / float(hw_h))
            except Exception:
                scale_x, scale_y = 1.0, 1.0

        if scale_x > 1.5 or scale_y > 1.5:
            bzyx[:, 2] /= scale_x
            bzyx[:, 3] /= scale_y
            zyx[:, 1] /= scale_x
            zyx[:, 2] /= scale_y

        # ====== 体素深度着色需要 voxel_coords（可选） ======
        voxel_coords = None
        if 'voxel_coords' in batch_dict:
            vc = batch_dict['voxel_coords']
            if isinstance(vc, torch.Tensor):
                vc = vc.detach().cpu().numpy()
            voxel_coords = vc
            if voxel_coords.shape[0] != bzyx.shape[0]:
                # 行数不对齐就不用深度着色，避免错配
                voxel_coords = None

        # ====== 找原始 lidar 点（尽量自动） ======
        def _find_raw_lidar_points(outer_dict):
            keys = ['origin_lidar', 'origin_lidar_drone', 'origin_lidar_rsu']
            for key in keys:
                if key in outer_dict:
                    return key, outer_dict[key]
            return None, None

        raw_key, raw_pts = _find_raw_lidar_points(outer_dict)
        # raw_pts: torch tensor
        if raw_pts is not None:
            # 允许 [M,3/4] 或 [B,M,3/4]
            if raw_pts.ndim == 2:
                pass
            elif raw_pts.ndim == 3:
                pass
            else:
                # 形状太怪，放弃
                raw_pts = None

        # ====== 选择用于投影的 lidar2image 矩阵 ======
        # 优先用 lidar2image_cam（通常是 ego->cam->img 的那条），没有就用 lidar2image
        proj_mat = batch_dict.get('lidar2image', None)
        if proj_mat is None:
            proj_mat = batch_dict.get('lidar2image_cam', None)

        # 允许 proj_mat 可能是 [1, B*N, 4,4] 之类，这里统一到 tensor
        if proj_mat is not None and not isinstance(proj_mat, torch.Tensor):
            proj_mat = torch.as_tensor(proj_mat)

        # img_aug_matrix（可选）
        img_aug = batch_dict.get('img_aug_matrix', None)
        if img_aug is not None and not isinstance(img_aug, torch.Tensor):
            img_aug = torch.as_tensor(img_aug)

        # ====== ImageNet 反归一化（只对单张做，避免 5D transpose 错） ======
        def _get_img_bgr(b_img, view_idx_img):
            img_chw = camera_imgs[b_img, view_idx_img]  # [3,H,W]
            if USE_IMAGENET_DENORM:
                mean = torch.tensor([0.485, 0.456, 0.406], device=img_chw.device).view(3,1,1)
                std  = torch.tensor([0.229, 0.224, 0.225], device=img_chw.device).view(3,1,1)
                img_den = torch.clamp(img_chw * std + mean, 0, 1)
                img_rgb = (img_den.permute(1,2,0).detach().cpu().numpy() * 255.0).astype(np.uint8)
            else:
                # 假设已经是 0~1 或 0~255
                arr = img_chw.detach().cpu().numpy()
                if arr.max() <= 1.5:
                    arr = np.clip(arr, 0, 1) * 255.0
                img_rgb = arr.transpose(1,2,0).astype(np.uint8)
            return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # ====== 投影 raw lidar 点到像素 ======
        def _project_points_to_pixel(points_xyz, P_4x4, A_4x4=None):
            """
            points_xyz: [M,3] float32 in ego/lidar
            P_4x4: [4,4] lidar/ego -> image homogeneous
            A_4x4: [4,4] image augmentation matrix (optional)
            return: u,v,depth (all float)
            """
            if points_xyz.shape[1] >= 4:
                xyz = points_xyz[:, :3]
            else:
                xyz = points_xyz

            ones = torch.ones((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)
            X = torch.cat([xyz, ones], dim=1)  # [M,4]

            # [M,4] = [M,4] @ [4,4]^T  （row-vector 习惯） or do (P @ X^T)^T
            # 为避免搞混，这里用列向量写法：
            x_img = (P_4x4 @ X.t()).t()  # [M,4]
            # 深度通常在 x_img[:,2]
            z = x_img[:, 2].clone()

            # 透视除法
            eps = 1e-6
            u = x_img[:, 0] / (z + eps)
            v = x_img[:, 1] / (z + eps)

            # 可选：应用增强矩阵到 (u,v)（对齐增强后的 camera_imgs）
            if A_4x4 is not None and USE_AUG:
                uv1 = torch.stack([u, v, torch.ones_like(u), torch.ones_like(u)], dim=1)  # [M,4]
                uv1_aug = (A_4x4 @ uv1.t()).t()
                u = uv1_aug[:, 0]
                v = uv1_aug[:, 1]

            return u, v, z

        # ====== 遍历 batch/view，画图 ======
        unique_batch_ids = np.unique(bzyx[:, 0].astype(np.int32))
        unique_view_ids = np.unique(bzyx[:, 1].astype(np.int32))

        for b in unique_batch_ids:
            b = int(b)
            b_img = min(b, B_img - 1) if B != B_img else b

            for view_idx in unique_view_ids:
                view_idx = int(view_idx)

                mask = (bzyx[:, 0].astype(np.int32) == b) & (bzyx[:, 1].astype(np.int32) == view_idx)
                if mask.sum() == 0:
                    continue

                # 映射到 camera_imgs 的 view 索引（处理 1 x (B*N) 合并）
                if N_img > N:
                    if B_img == 1 and N_img == B * N:
                        view_idx_img = b * N + view_idx
                    else:
                        view_idx_img = view_idx
                    if view_idx_img >= N_img:
                        continue
                else:
                    view_idx_img = view_idx
                    if view_idx_img >= N_img:
                        continue

                # 背景图（反归一化后）
                img_bgr = _get_img_bgr(b_img, view_idx_img)
                vis_img = img_bgr.copy()

                # ====== 1) 画体素投影点（原逻辑：patch->pixel）=====
                proj = bzyx[mask]  # [K,4] b, view, x_patch, y_patch
                x_patch = proj[:, 2]
                y_patch = proj[:, 3]
                x_pix = (x_patch * patch_size).astype(np.int32)
                y_pix = (y_patch * patch_size).astype(np.int32)

                valid = (x_pix >= 0) & (x_pix < W) & (y_pix >= 0) & (y_pix < H)
                x_pix_v = x_pix[valid]
                y_pix_v = y_pix[valid]

                if x_pix_v.shape[0] > 0:
                    if voxel_coords is not None:
                        vox_masked = voxel_coords[mask][valid]  # [K_valid,4]
                        z_coords = vox_masked[:, 1].astype(np.float32)
                        z_min, z_max = float(z_coords.min()), float(z_coords.max())
                        if z_max > z_min:
                            z_norm = (z_coords - z_min) / (z_max - z_min)
                        else:
                            z_norm = np.zeros_like(z_coords, dtype=np.float32)

                        # BGR：近绿 远红
                        colors = np.zeros((len(z_norm), 3), dtype=np.uint8)
                        colors[:, 2] = (z_norm * 255).astype(np.uint8)           # R
                        colors[:, 1] = ((1.0 - z_norm) * 255).astype(np.uint8)   # G
                        colors[:, 0] = 0                                         # B

                        for i in range(len(x_pix_v)):
                            cv2.circle(vis_img, (int(x_pix_v[i]), int(y_pix_v[i])), VOX_RADIUS,
                                    (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])), -1)
                    else:
                        for i in range(len(x_pix_v)):
                            cv2.circle(vis_img, (int(x_pix_v[i]), int(y_pix_v[i])), VOX_RADIUS, (0, 255, 0), -1)

                # ====== 2) 画原始 lidar 点投影（红色）=====
                if raw_pts is not None and proj_mat is not None:
                    # 取本 batch 的 lidar 点
                    if raw_pts.ndim == 3:
                        pts_b = raw_pts[b]  # [M,3/4]
                    else:
                        pts_b = raw_pts     # [M,3/4]（默认属于该 b）

                    # 限制数量，避免太慢
                    if pts_b.shape[0] > MAX_RAW_LIDAR_POINTS:
                        idx = torch.randperm(pts_b.shape[0], device=pts_b.device)[:MAX_RAW_LIDAR_POINTS]
                        pts_b = pts_b[idx]

                    # 选择对应 view 的投影矩阵
                    # proj_mat 可能形状：[B,N,4,4] 或 [1,B*N,4,4]
                    P = None
                    if proj_mat.ndim == 4:
                        # [B,N,4,4]
                        if proj_mat.shape[0] == B and proj_mat.shape[1] >= (view_idx + 1):
                            P = proj_mat[b, view_idx]
                        elif proj_mat.shape[0] == 1 and proj_mat.shape[1] == B * N:
                            P = proj_mat[0, b * N + view_idx]
                    elif proj_mat.ndim == 5:
                        # 不太常见，忽略
                        P = None

                    A = None
                    if img_aug is not None and USE_AUG:
                        # img_aug_matrix 你构造过是 [1, B*N, 4,4]，也可能 [B,N,4,4]
                        if img_aug.ndim == 4:
                            if img_aug.shape[0] == 1 and img_aug.shape[1] == B * N:
                                A = img_aug[0, b * N + view_idx]
                            elif img_aug.shape[0] == B and img_aug.shape[1] >= (view_idx + 1):
                                A = img_aug[b, view_idx]
                        # 否则 A=None

                    if P is not None:
                        pts_b = pts_b.to(P.device).to(torch.float32)
                        if pts_b.shape[1] >= 4:
                            pts_xyz = pts_b[:, :3]
                        else:
                            pts_xyz = pts_b

                        u, v, z = _project_points_to_pixel(pts_xyz, P.to(torch.float32), A.to(torch.float32) if A is not None else None)

                        # 过滤：深度为正 + 在图像范围内
                        u_np = u.detach().cpu().numpy()
                        v_np = v.detach().cpu().numpy()
                        z_np = z.detach().cpu().numpy()
                        m2 = (z_np > 1e-3) & (u_np >= 0) & (u_np < W) & (v_np >= 0) & (v_np < H)

                        u_np = u_np[m2].astype(np.int32)
                        v_np = v_np[m2].astype(np.int32)

                        # 红色点（BGR = (0,0,255)）
                        for i in range(len(u_np)):
                            cv2.circle(vis_img, (int(u_np[i]), int(v_np[i])), RAW_LIDAR_RADIUS, (0, 0, 255), -1)

                # ====== 文本 & 保存 ======
                text = f"Batch:{b} View:{view_idx} VoxPts:{int(mask.sum())} Raw:{raw_key if raw_key else 'None'}"
                cv2.putText(vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                save_path = os.path.join(self.vis_save_dir, f"lidar2image_b{b}_v{view_idx}_iter{self.vis_iter_count}.png")
                cv2.imwrite(save_path, vis_img)

                # stats
                stats_path = os.path.join(self.vis_save_dir, f"lidar2image_stats_b{b}_v{view_idx}_iter{self.vis_iter_count}.txt")
                with open(stats_path, "w") as f:
                    f.write(f"Batch: {b}, View: {view_idx}\n")
                    f.write(f"hw_shape: {hw_shape}, patch_size: {patch_size}, HxW: {H}x{W}\n")
                    f.write(f"scale_x: {scale_x:.3f}, scale_y: {scale_y:.3f}\n")
                    f.write(f"Vox projected points (mask): {int(mask.sum())}\n")
                    f.write(f"Vox valid points: {int(valid.sum()) if 'valid' in locals() else -1}\n")
                    f.write(f"Raw lidar key: {raw_key}\n")
                    f.write(f"USE_AUG: {USE_AUG}, USE_IMAGENET_DENORM: {USE_IMAGENET_DENORM}\n")

        print(f"[MambaFusion] 投影可视化已保存到: {self.vis_save_dir} (iter: {self.vis_iter_count})")

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name:
                nn.init.xavier_uniform_(p)

    def _recover_image(self, pillar_features, coords, indices):
        pillar_features = getattr(self, f'out_norm{indices}')(pillar_features)
        batch_size = coords[:, 0].max().int().item() + 1
        batch_spatial_features = pillar_features.view(
            batch_size, self.patch_size[0], self.patch_size[1], -1).permute(0, 3, 1, 2).contiguous()
        return batch_spatial_features




class UniTRInputLayer(DSVTInputLayer):
    ''' 
    This class converts the output of vfe to unitr input.
    We do in this class:
    1. Window partition: partition voxels to non-overlapping windows.
    2. Set partition: generate non-overlapped and size-equivalent local sets within each window.
    3. Pre-compute the downsample infomation between two consecutive stages.
    4. Pre-compute the position embedding vectors.

    Args:
        sparse_shape (tuple[int, int, int]): Shape of input space (xdim, ydim, zdim).
        window_shape (list[list[int, int, int]]): Window shapes (winx, winy, winz) in different stages. Length: stage_num.
        downsample_stride (list[list[int, int, int]]): Downsample strides between two consecutive stages. 
            Element i is [ds_x, ds_y, ds_z], which is used between stage_i and stage_{i+1}. Length: stage_num - 1.
        d_model (list[int]): Number of input channels for each stage. Length: stage_num.
        set_info (list[list[int, int]]): A list of set config for each stage. Eelement i contains 
            [set_size, block_num], where set_size is the number of voxel in a set and block_num is the
            number of blocks for stage i. Length: stage_num.
        hybrid_factor (list[int, int, int]): Control the window shape in different blocks. 
            e.g. for block_{0} and block_{1} in stage_0, window shapes are [win_x, win_y, win_z] and 
            [win_x * h[0], win_y * h[1], win_z * h[2]] respectively.
        shift_list (list): Shift window. Length: stage_num.
        input_image (bool): whether input modal is image.
    '''

    def __init__(self, model_cfg, accelerate=False):
        # dummy config
        model_cfg.downsample_stride = model_cfg.get('downsample_stride',[])
        model_cfg.normalize_pos = model_cfg.get('normalize_pos',False)
        super().__init__(model_cfg)

        self.input_image = self.model_cfg.get('input_image', False)
        self.key_name = 'patch' if self.input_image else 'voxel'
        # only support image input accelerate
        self.accelerate = self.input_image and accelerate
        self.process_info = None

    def forward(self, batch_dict):
        '''
        Args:
            bacth_dict (dict): 
                The dict contains the following keys
                - voxel_features (Tensor[float]): Voxel features after VFE with shape (N, d_model[0]), 
                    where N is the number of input voxels.
                - voxel_coords (Tensor[int]): Shape of (N, 4), corresponding voxel coordinates of each voxels.
                    Each row is (batch_id, z, y, x). 
                - ...

        Returns:
            voxel_info (dict):
                The dict contains the following keys
                - voxel_coors_stage{i} (Tensor[int]): Shape of (N_i, 4). N is the number of voxels in stage_i.
                    Each row is (batch_id, z, y, x).
                - set_voxel_inds_stage{i}_shift{j} (Tensor[int]): Set partition index with shape (2, set_num, set_info[i][0]).
                    2 indicates x-axis partition and y-axis partition. 
                - set_voxel_mask_stage{i}_shift{i} (Tensor[bool]): Key mask used in set attention with shape (2, set_num, set_info[i][0]).
                - pos_embed_stage{i}_block{i}_shift{i} (Tensor[float]): Position embedding vectors with shape (N_i, d_model[i]). N_i is the 
                    number of remain voxels in stage_i;
                - ...
        '''
        if self.input_image and self.process_info is not None and (batch_dict['patch_coords'][:, 0][-1] == self.process_info['voxel_coors_stage0'][:, 0][-1]):
            patch_info = dict()
            for k in (self.process_info.keys()):
                if torch.is_tensor(self.process_info[k]):
                    patch_info[k] = self.process_info[k].clone()
                else:
                    patch_info[k] = copy.deepcopy(self.process_info[k])
            # accelerate by caching pos embed as patch coords are fixed
            if not self.accelerate:
                for stage_id in range(len(self.downsample_stride)+1): # 1
                    for block_id in range(self.set_info[stage_id][1]): # 4
                        for shift_id in range(self.num_shifts[stage_id]): # 2
                            if not isinstance(self.posembed_layers[stage_id][block_id], nn.Identity):
                                patch_info[f'pos_embed_stage{stage_id}_block{block_id}_shift{shift_id}'] = \
                                    self.get_pos_embed(
                                        patch_info[f'coors_in_win_stage{stage_id}_shift{shift_id}'], stage_id, block_id, shift_id)
                            else:
                                patch_info[f'pos_embed_stage{stage_id}_block{block_id}_shift{shift_id}'] = []
            return patch_info

        key_name = self.key_name # voxel or patch
        coors = batch_dict[f'{key_name}_coords'].long() # [33792, 4]

        info = {}
        # original input voxel coors
        info[f'voxel_coors_stage0'] = coors.clone() # image_id z y x

        for stage_id in range(len(self.downsample_stride)+1): # 1
            # window partition of corrsponding stage-map
            info = self.window_partition(info, stage_id) # 有了shift前后每个voxel or patch对应的window的坐标以及在window中的坐标 4个tensor
            # generate set id of corrsponding stage-map
            info = self.get_set(info, stage_id) # 有了shift前后按照xy排序后每个set的voxel or patch的index 以及mask(重复) 4个tensor 
            for block_id in range(self.set_info[stage_id][1]): # 2  
                for shift_id in range(self.num_shifts[stage_id]): # 2
                    if not isinstance(self.posembed_layers[stage_id][block_id], nn.Identity):
                        info[f'pos_embed_stage{stage_id}_block{block_id}_shift{shift_id}'] = \
                            self.get_pos_embed( # ? 这里embed为啥不引入不同view的camera信息，也没用lidar的z信息
                                info[f'coors_in_win_stage{stage_id}_shift{shift_id}'], stage_id, block_id,  shift_id) # [num_of_voxels, 128]
                    else:
                        info[f'pos_embed_stage{stage_id}_block{block_id}_shift{shift_id}'] = []

        info['sparse_shape_list'] = self.sparse_shape_list # [32, 88, 1] feature的 shape lidar-camera不同

        # save process info for image input as patch coords are fixed
        if self.input_image:  
            self.process_info = {}
            for k in (info.keys()):
                if k != 'patch_feats_stage0':
                    if torch.is_tensor(info[k]):
                        self.process_info[k] = info[k].clone()
                    else:
                        self.process_info[k] = copy.deepcopy(info[k])

        return info
    
    def get_set_single_shift(self, batch_win_inds, stage_id, shift_id=None, coors_in_win=None):
        '''
        voxel_order_list[list]: order respectively sort by x, y, z
        '''

        device = batch_win_inds.device

        # max number of voxel in a window
        voxel_num_set = self.set_info[stage_id][0] # 90 一个set中最多有90个voxel
        max_voxel = self.window_shape[stage_id][shift_id][0] * \
            self.window_shape[stage_id][shift_id][1] * \
            self.window_shape[stage_id][shift_id][2] # 900 一个window中最多有900个voxel

        if self.model_cfg.get('expand_max_voxels', None) is not None:
            max_voxel *= self.model_cfg.get('expand_max_voxels', None)
        contiguous_win_inds = torch.unique(
            batch_win_inds, return_inverse=True)[1] # [num_of_voxels] 每个voxel属于那个window
        voxelnum_per_win = torch.bincount(contiguous_win_inds) # torch.Size([num_of_window])
        win_num = voxelnum_per_win.shape[0] # num_of_window 有150个window

        setnum_per_win_float = voxelnum_per_win / voxel_num_set # torch.Size([num_of_window]) 一个window中有多少个set
        setnum_per_win = torch.ceil(setnum_per_win_float).long() # 向上取整 torch.Size([num_of_window]) 一个window中有多少个set

        set_num = setnum_per_win.sum().item() # 有多少个set num_of_set
        setnum_per_win_cumsum = torch.cumsum(setnum_per_win, dim=0)[:-1] # torch.Size([num_of_window - 1]) 一个window中对应的最后一个voxel的编号

        set_win_inds = torch.full((set_num,), 0, device=device) # torch.Size([num_of_set]) 有308个set
        set_win_inds[setnum_per_win_cumsum] = 1 
        set_win_inds = torch.cumsum(set_win_inds, dim=0) # 每个set所在的window的index torch.Size([num_of_set])

        # input [0,0,0, 1, 2,2]
        roll_set_win_inds_left = torch.roll( 
            set_win_inds, -1)  # [0,0, 1, 2,2,0]
        diff = set_win_inds - roll_set_win_inds_left  # [0, 0, -1, -1, 0, 2] 
        end_pos_mask = diff != 0 # 得到一个表示每个set是否是窗口中的最后一个set的掩码
        template = torch.ones_like(set_win_inds)
        template[end_pos_mask] = (setnum_per_win - 1) * -1  # [1,1,-2, 0, 1,-1]
        set_inds_in_win = torch.cumsum(template, dim=0)  # [1,2,0, 0, 1,0]
        set_inds_in_win[end_pos_mask] = setnum_per_win  # [1,2,3, 1, 1,2]
        set_inds_in_win = set_inds_in_win - 1  # [0,1,2, 0, 0,1] 得到每个set在window中的index

        offset_idx = set_inds_in_win[:, None].repeat(
            1, voxel_num_set) * voxel_num_set # torch.Size([num_of_set, 90]) 每个set在window中的index乘以set中voxel的数量
        base_idx = torch.arange(0, voxel_num_set, 1, device=device)
        base_select_idx = offset_idx + base_idx
        base_select_idx = base_select_idx * \
            voxelnum_per_win[set_win_inds][:, None]
        base_select_idx = base_select_idx.double(
        ) / (setnum_per_win[set_win_inds] * voxel_num_set)[:, None].double()
        base_select_idx = torch.floor(base_select_idx)

        select_idx = base_select_idx
        select_idx = select_idx + set_win_inds.view(-1, 1) * max_voxel # torch.Size([num_of_set, 90]) 每个set在window中的index乘以set中voxel的数量加上window的index乘以window中voxel的数量

        # sort by y
        inner_voxel_inds = get_inner_win_inds_cuda(contiguous_win_inds) # torch.Size([num_of_voxels]) 每个voxel在window中的index 0-899
        global_voxel_inds = contiguous_win_inds * max_voxel + inner_voxel_inds # torch.Size([num_of_voxels]) 每个voxel在window中的index乘以window中voxel的数量加上window的index乘以window中voxel的数量
        _, order1 = torch.sort(global_voxel_inds) # torch.Size([num_of_voxels]) 按照voxel的global index排序
        global_voxel_inds_sorty = contiguous_win_inds * max_voxel + \
            coors_in_win[:, 1] * self.window_shape[stage_id][shift_id][0] * self.window_shape[stage_id][shift_id][2] + \
            coors_in_win[:, 2] * self.window_shape[stage_id][shift_id][2] + \
            coors_in_win[:, 0] # torch.Size([num_of_voxels]) 每个voxel在window中的index乘以window中voxel的数量加上window的index乘以window中voxel的数量
        _, order2 = torch.sort(global_voxel_inds_sorty) # torch.Size([num_of_voxels]) 对于每个window内的voxels按照voxel的y坐标排序

        inner_voxel_inds_sorty = -torch.ones_like(inner_voxel_inds)
        inner_voxel_inds_sorty.scatter_(
            dim=0, index=order2, src=inner_voxel_inds[order1]) # torch.Size([num_of_voxels]) 对于每个window内的voxels按照voxel的y坐标排序
        inner_voxel_inds_sorty_reorder = inner_voxel_inds_sorty 
        voxel_inds_in_batch_sorty = inner_voxel_inds_sorty_reorder + \
            max_voxel * contiguous_win_inds # torch.Size([num_of_voxels]) 得到按照y坐标排序后的内部体素索引
        voxel_inds_padding_sorty = -1 * \
            torch.ones((win_num * max_voxel), dtype=torch.long, device=device) # torch.Size([num_of_window * 900]) 生成一个全-1的tensor
        voxel_inds_padding_sorty[voxel_inds_in_batch_sorty] = torch.arange(
            0, voxel_inds_in_batch_sorty.shape[0], dtype=torch.long, device=device) # torch.Size([num_of_voxels]) 得到按照y坐标排序后的内部体素索引 -1的位置是空的，其他位置是按照y坐标排序后的内部体素索引

        # sort by x
        global_voxel_inds_sorty = contiguous_win_inds * max_voxel + \
            coors_in_win[:, 2] * self.window_shape[stage_id][shift_id][1] * self.window_shape[stage_id][shift_id][2] + \
            coors_in_win[:, 1] * self.window_shape[stage_id][shift_id][2] + \
            coors_in_win[:, 0]
        _, order2 = torch.sort(global_voxel_inds_sorty)

        inner_voxel_inds_sortx = -torch.ones_like(inner_voxel_inds)
        inner_voxel_inds_sortx.scatter_(
            dim=0, index=order2, src=inner_voxel_inds[order1])
        inner_voxel_inds_sortx_reorder = inner_voxel_inds_sortx
        voxel_inds_in_batch_sortx = inner_voxel_inds_sortx_reorder + \
            max_voxel * contiguous_win_inds
        voxel_inds_padding_sortx = -1 * \
            torch.ones((win_num * max_voxel), dtype=torch.long, device=device)
        voxel_inds_padding_sortx[voxel_inds_in_batch_sortx] = torch.arange(
            0, voxel_inds_in_batch_sortx.shape[0], dtype=torch.long, device=device) # torch.Size([num_of_voxels]) 得到按照x坐标排序后的内部体素索引 -1的位置是空的，其他位置是按照x坐标排序后的内部体素索引

        set_voxel_inds_sorty = voxel_inds_padding_sorty[select_idx.long()] # torch.Size([num_of_set, 90]) 得到按照y坐标排序后的内部体素索引
        set_voxel_inds_sortx = voxel_inds_padding_sortx[select_idx.long()] # torch.Size([num_of_set, 90]) 得到按照x坐标排序后的内部体素索引
        all_set_voxel_inds = torch.stack(
            (set_voxel_inds_sorty, set_voxel_inds_sortx), dim=0) # torch.Size([2, num_of_set, 90]) 得到按照y坐标排序后的内部体素索引和按照x坐标排序后的内部体素索引

        return all_set_voxel_inds

    def get_pos_embed(self, coors_in_win, stage_id, block_id, shift_id):
        '''
        Args:
        coors_in_win: shape=[N, 3], order: z, y, x
        '''
        # [N,]
        window_shape = self.window_shape[stage_id][shift_id] # [30, 30, 1]
        embed_layer = self.posembed_layers[stage_id][block_id][shift_id]
        if len(window_shape) == 2:
            ndim = 2
            win_x, win_y = window_shape
            win_z = 0
        elif window_shape[-1] == 1:
            if self.sparse_shape[-1] == 1:
                ndim = 2
            else:
                ndim = 3
            win_x, win_y = window_shape[:2] # (30, 30)
            win_z = 0
        else:
            win_x, win_y, win_z = window_shape
            ndim = 3

        assert coors_in_win.size(1) == 3
        z, y, x = coors_in_win[:, 0] - win_z/2, coors_in_win[:, 1] - win_y/2, coors_in_win[:, 2] - win_x/2

        if self.normalize_pos:
            x = x / win_x * 2 * 3.1415 #[-pi, pi]
            y = y / win_y * 2 * 3.1415 #[-pi, pi]
            z = z / win_z * 2 * 3.1415 #[-pi, pi]
        
        if ndim==2:
            location = torch.stack((x, y), dim=-1)
        else:
            location = torch.stack((x, y, z), dim=-1)
        pos_embed = embed_layer(location) # [num_of_voxels, 2] -> [num_of_voxels, 128]

        return pos_embed



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
    
class DeformableFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels, T, kernel_size=3, stride=1, padding=1, dilation=1, deformable_groups=1):
        super(DeformableFeatureFusion, self).__init__()
        self.offset = nn.Conv2d(in_channels * T, deformable_groups * 2 * kernel_size * kernel_size, kernel_size=3, padding=1)
        self.deform_conv = DeformConv2d(in_channels * T, out_channels * T, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=deformable_groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        # 如果输入和输出通道数或步长不匹配，则使用卷积调整输入形状
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # x 的形状为 [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x_res = x[:, -1]
        x = x.view(B, T * C, H, W)  # 转换形状为 [B, T*C, H, W]
        offset = self.offset(x)  # 生成偏移量 [B, deformable_groups*2*kernel_size*kernel_size, H, W]
        x = self.deform_conv(x, offset)  # 应用 DeformConv2d
        x = x.view(B, T, -1, H, W).mean(dim=1)  # 转换形状为 [B, C, H, W]，并求平均
        x = self.bn(x)
        x += self.shortcut(x_res)
        x = self.relu(x)
        return x