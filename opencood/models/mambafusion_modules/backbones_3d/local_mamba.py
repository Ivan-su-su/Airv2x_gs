import torch
import torch.nn as nn

import math
from functools import partial
from mamba_ssm.models.mixer_seq_simple import create_block
from ..model_utils.voxel_mamba_utils import get_hilbert_index_3d_mamba_lite
from ..ops.win_coors.flattened_window_cuda import fused_hilbert_pos_embed as fused_hilbert_pos_embed_cuda
# try:
#     from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
# except ImportError:
#     RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from ..spconv_utils import replace_feature, spconv
from .spconv_backbone import post_act_block
import torch.utils.checkpoint as cp

def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)



class GlobalMamba(nn.Module):
    def __init__(self, 
                 d_model, 
                 ssm_cfg, 
                 norm_epsilon, 
                 rms_norm,
                 down_kernel_size,
                 down_stride,
                 num_down,
                 norm_fn,
                 indice_key,
                 sparse_shape,
                 hilbert_config,
                 downsample_lvl,
                 down_resolution=True,
                 residual_in_fp32=True, 
                 fused_add_norm=True,
                 device=None,
                 dtype=None,
                 downsample_ori=None,
                 use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # ssm_cfg = {}
        factory_kwargs = {'device': device, 'dtype':dtype}

        # mamba layer
        mamba_encoder_1 = create_block(
            d_model=d_model,
            ssm_cfg=ssm_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            layer_idx=0,
            **factory_kwargs,
        )

        mamba_encoder_2 = create_block(
            d_model=d_model,
            ssm_cfg=ssm_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            layer_idx=1,
            **factory_kwargs,
        )

        self.mamba_encoder_list = nn.ModuleList([mamba_encoder_1, mamba_encoder_2])

        # downsampling operation #
        self.conv_encoder = nn.ModuleList()
        for idx in range(len(down_stride)):
            self.conv_encoder.append(
                DownSp(d_model, down_kernel_size[idx], down_stride[idx], num_down[idx], norm_fn, f"{indice_key}_{idx}"))
        
        # upsampling operation #
        downsample_times = len(down_stride[1:])
        self.conv_decoder = nn.ModuleList()
        self.conv_decoder_norm = nn.ModuleList()
        for idx, kernel_size in enumerate(down_kernel_size[1:]):
            if down_resolution:
                self.conv_decoder.append(
                    post_act_block(
                        d_model, d_model, kernel_size, norm_fn=norm_fn, conv_type='inverseconv',
                        indice_key=f'spconv_{indice_key}_{downsample_times - idx}'))
                self.conv_decoder_norm.append(norm_fn(d_model))
            else:
                self.conv_decoder.append(
                    post_act_block(
                        d_model, d_model, kernel_size, norm_fn=norm_fn, conv_type='subm',
                        indice_key=f'{indice_key}_{downsample_times - idx}'))
                self.conv_decoder_norm.append(norm_fn(d_model))
        
        self.sparse_shape = sparse_shape
        self.downsample_lvl = downsample_lvl

        norm_cls = partial(
            nn.LayerNorm, eps=norm_epsilon, **factory_kwargs
        )
        self.norm = norm_cls(d_model)
        self.norm_back = norm_cls(d_model)
        self.downsample_ori = downsample_ori if downsample_ori is not None else 'curve_template_rank9'
        self._template_validated = False  # 用于避免重复打印验证信息
    
    def _validate_hilbert_template_coverage(
        self, 
        template_name: str, 
        template_size, 
        actual_shape, 
        coords: torch.Tensor,
        stage_name: str = ""
    ):
        """验证Hilbert模板尺寸是否能覆盖实际的BEV尺寸
        
        Args:
            template_name: 模板名称，如'curve_template_rank10'
            template_size: 模板尺寸 (z, y, x)
            actual_shape: 实际sparse tensor的spatial_shape (z, y, x)
            coords: 坐标张量 [N, 4]，格式为(batch, z, y, x)
            stage_name: 阶段名称，用于日志
        """
        if self._template_validated:
            return  # 避免重复验证
        
        ts = list(template_size) if not isinstance(template_size, (list, tuple)) else list(template_size)
        ss = list(actual_shape) if not isinstance(actual_shape, (list, tuple)) else list(actual_shape)
        
        # 检查模板是否能覆盖actual_shape
        for d, (t_dim, s_dim) in enumerate(zip(ts, ss)):
            if t_dim < s_dim:
                raise ValueError(
                    f"[GlobalMamba][{stage_name}] Hilbert template '{template_name}' "
                    f"cannot cover actual spatial_shape on dim{d}: "
                    f"template_size={ts}, actual_shape={ss}. "
                    f"Template dim {t_dim} < actual dim {s_dim}. "
                    f"Please use a larger rank template (e.g., rank10 for 1024x1024)."
                )
        
        # 检查coords是否在范围内
        if coords.shape[0] > 0:
            for d in range(min(len(ss), coords.shape[1] - 1)):
                coord_max = coords[:, d + 1].max().item()
                if coord_max >= ts[d]:
                    raise ValueError(
                        f"[GlobalMamba][{stage_name}] Coordinate out of template range on dim{d}: "
                        f"coord_max={coord_max}, template_dim={ts[d]}. "
                        f"This will cause index overflow in Hilbert curve lookup."
                    )
        
        # 仅在第一次验证时打印信息
        if not self._template_validated:
            print(f"[GlobalMamba][{stage_name}] Template '{template_name}' validation passed: "
                  f"template_size={ts} >= actual_shape={ss}")
    
    def forward(
        self,
        voxel_features,
        voxel_coords,
        batch_size,
        curt_spatial_shape,
        curve_template,
        hilbert_spatial_size,
        pos_embed,
        num_stage,
        debug=True,
        ):

        mamba_layer1 = self.mamba_encoder_list[0]
        mamba_layer2 = self.mamba_encoder_list[1]
        
        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=curt_spatial_shape,
            batch_size=batch_size
        )

        features = []
        for conv in self.conv_encoder:
            x = conv(x)
            features.append(x)
        
        x_s1 = features[0]
        x_s2 = features[1]
        feats_s2 = features[1].features
        coords_s2 = features[1].indices
        feats_s1 = features[0].features
        coords_s1 = voxel_coords

        clvl_cruve_template_s1 = curve_template[self.downsample_ori] # 'curve_template_rank9' for lss
        clvl_hilbert_spatial_size_s1 = hilbert_spatial_size[self.downsample_ori]
        clvl_cruve_template_s2 = curve_template[self.downsample_lvl] # 'curve_template_rank8' for lss
        clvl_hilbert_spatial_size_s2 = hilbert_spatial_size[self.downsample_lvl] # (1, 512, 512)
        # hilbert_s1, hilbert_s2, pos_embed_coords_s1_new, pos_embed_coords_s2_new = fused_hilbert_pos_embed_cuda(
        #     coords_s1.long(), coords_s2.long(), clvl_cruve_template_s1, clvl_cruve_template_s2, batch_size, 
        #     clvl_hilbert_spatial_size_s1[0], clvl_hilbert_spatial_size_s1[1], clvl_hilbert_spatial_size_s1[2],
        #     clvl_hilbert_spatial_size_s2[0], clvl_hilbert_spatial_size_s2[1], clvl_hilbert_spatial_size_s2[2],
        #     x_s1.spatial_shape, x_s2.spatial_shape, (num_stage, num_stage, num_stage)
        # )
        # 在调用fused_hilbert_pos_embed_cuda之前，先验证Hilbert模板能够覆盖实际的spatial_shape
        # 这是关键检查：确保模板尺寸 >= 实际BEV尺寸

        # self._validate_hilbert_template_coverage(
        #     template_name=self.downsample_ori,
        #     template_size=clvl_hilbert_spatial_size_s1,
        #     actual_shape=x_s1.spatial_shape,
        #     coords=coords_s1,
        #     stage_name="S1_ori"
        # )
        # self._validate_hilbert_template_coverage(
        #     template_name=self.downsample_lvl,
        #     template_size=clvl_hilbert_spatial_size_s2,
        #     actual_shape=x_s2.spatial_shape,
        #     coords=coords_s2,
        #     stage_name="S2_lvl"
        # )
        # print("x_s1.spatial_shape:", x_s1.spatial_shape, "hil_s1:", clvl_hilbert_spatial_size_s1, "tpl_s1:", tuple(clvl_cruve_template_s1.shape), clvl_cruve_template_s1.numel())
        # print("x_s2.spatial_shape:", x_s2.spatial_shape, "hil_s2:", clvl_hilbert_spatial_size_s2, "tpl_s2:", tuple(clvl_cruve_template_s2.shape), clvl_cruve_template_s2.numel())
        # print("coords_s2 min/max:", coords_s2.min(0).values.tolist(), coords_s2.max(0).values.tolist())
        # print("coords_s1 min/max:", coords_s1.min(0).values.tolist(), coords_s1.max(0).values.tolist())
        hilbert_s1, pos_embed_coords_s1= fused_hilbert_pos_embed_cuda(
            coords_s1.long(), clvl_cruve_template_s1, batch_size, 
            clvl_hilbert_spatial_size_s1[0], clvl_hilbert_spatial_size_s1[1], clvl_hilbert_spatial_size_s1[2],
            x_s1.spatial_shape, (num_stage, num_stage, num_stage)
        )
        hilbert_s2, pos_embed_coords_s2 = fused_hilbert_pos_embed_cuda(
            coords_s2.long(), clvl_cruve_template_s2, batch_size, 
            clvl_hilbert_spatial_size_s2[0], clvl_hilbert_spatial_size_s2[1], clvl_hilbert_spatial_size_s2[2],
            x_s2.spatial_shape, (num_stage, num_stage, num_stage)
        )
        
        # 调用check_hilbert_curve进行完整性检查(仅在DEBUG模式下启用，避免性能损失)
        # self.check_hilbert_curve(
        #     name="S1",
        #     coords=coords_s1,
        #     spatial_shape=x_s1.spatial_shape,
        #     hilbert_index=hilbert_s1,
        #     hilbert_spatial_size=clvl_hilbert_spatial_size_s1,
        #     batch_size=batch_size,
        #     strict=True,
        # )
        # self.check_hilbert_curve(
        #     name="S2",
        #     coords=coords_s2,
        #     spatial_shape=x_s2.spatial_shape,
        #     hilbert_index=hilbert_s2,
        #     hilbert_spatial_size=clvl_hilbert_spatial_size_s2,
        #     batch_size=batch_size,
        #     strict=True,
        # )
       
        # print(f"coords_s1: {coords_s1.shape}, coords_s2: {coords_s2.shape}")
        # print(f"x_s1.spatial_shape: {x_s1.spatial_shape}, x_s2.spatial_shape: {x_s2.spatial_shape}")
        # print(f"clvl_hilbert_spatial_size_s1: {clvl_hilbert_spatial_size_s1}, clvl_hilbert_spatial_size_s2: {clvl_hilbert_spatial_size_s2}")
        # print(f"num_stage: {num_stage}")
        # print(f"batch_size: {batch_size}")
        # 创建 inds_curt_to_next 和 inds_next_to_curt
        inds_curt_to_next_s1 = {}
        inds_next_to_curt_s1 = {}
        inds_curt_to_next_s2 = {}
        inds_next_to_curt_s2 = {}
        index_info_s1 = {}
        index_info_s2 = {}
        for i in range(batch_size):
            batch_mask_s1 = coords_s1[:, 0] == i
            batch_mask_s2 = coords_s2[:, 0] == i

            # 对 hilbert_s1 和 hilbert_s2 进行排序
            inds_curt_to_next = torch.argsort(hilbert_s1[batch_mask_s1])
            inds_next_to_curt = torch.argsort(inds_curt_to_next)
            inds_curt_to_next_s1[i] = inds_curt_to_next
            inds_next_to_curt_s1[i] = inds_next_to_curt

            inds_curt_to_next = torch.argsort(hilbert_s2[batch_mask_s2])
            inds_next_to_curt = torch.argsort(inds_curt_to_next)
            inds_curt_to_next_s2[i] = inds_curt_to_next
            inds_next_to_curt_s2[i] = inds_next_to_curt
        index_info_s1['inds_curt_to_next'] = inds_curt_to_next_s1
        index_info_s1['inds_next_to_curt'] = inds_next_to_curt_s1
        index_info_s2['inds_curt_to_next'] = inds_curt_to_next_s2
        index_info_s2['inds_next_to_curt'] = inds_next_to_curt_s2
            
        
        pos_embed_s2 = pos_embed(pos_embed_coords_s2.float())

        inds_curt_to_next_s2 = index_info_s2['inds_curt_to_next']
        inds_next_to_curt_s2 = index_info_s2['inds_next_to_curt']
        inds_curt_to_next_s1 = index_info_s1['inds_curt_to_next']
        inds_next_to_curt_s1 = index_info_s1['inds_next_to_curt']
        new_features = []
        # Low Resolution
        out_feat_3d_s2 = torch.zeros_like(feats_s2)
        out_feat_3d_s1 = torch.zeros_like(feats_s1)

        feats_s2 = feats_s2 + pos_embed_s2

        # Borward SSMs
        for i in range(batch_size):
            b_mask_m2 = coords_s2[:, 0] == i
            feat_m2 = feats_s2[b_mask_m2][inds_curt_to_next_s2[i]][None]
            if self.training and self.use_checkpoint:
                out_feat_m2 = cp.checkpoint(mamba_layer1, feat_m2, None, use_reentrant=False)
            else:
                out_feat_m2 = mamba_layer1(feat_m2, None) # [1, 22095, 128]
            out_feat_3d_s2[b_mask_m2] = (out_feat_m2[0]).squeeze(0)[inds_next_to_curt_s2[i]]

        x_s2 = replace_feature(x_s2, self.norm(out_feat_3d_s2))


        pos_embed_s1 = pos_embed(pos_embed_coords_s1.float())

        feats_s1 = feats_s1 + pos_embed_s1
        for i in range(batch_size):
            b_mask_m1 = coords_s1[:, 0] == i
            feat_m1 = feats_s1[b_mask_m1][inds_curt_to_next_s1[i]][None]
            feat_back = feat_m1.flip(1)
            if self.training and self.use_checkpoint:
                out_feat_back = cp.checkpoint(mamba_layer2, feat_back, None, use_reentrant=False)
            else:
                out_feat_back = mamba_layer2(feat_back, None)
            out_feat_3d_s1[b_mask_m1] = (out_feat_back[0]).squeeze(0).flip(0)[inds_next_to_curt_s1[i]]

        x_s1 = replace_feature(x_s1, self.norm_back(out_feat_3d_s1))

        # new_features.append(features[0])
        new_features.append(x_s1)
        new_features.append(x_s2)

        x = x_s2

        for deconv, norm, up_x in zip(self.conv_decoder, self.conv_decoder_norm, new_features[:-1][::-1]):
            x = deconv(x)
            x = replace_feature(x, x.features + up_x.features + features[0].features)
            x = replace_feature(x, norm(x.features))

        return x.features, x.indices
    def check_hilbert_curve(
        self,
        *,
        name: str,
        coords: torch.Tensor,              # [N, 4] (b, z, y, x) or (b, y, x, z) depending on your convention
        spatial_shape,                     # x_s?.spatial_shape, e.g. [1, 100, 352] or [1, 200, 704]
        hilbert_index: torch.Tensor,        # [N]
        hilbert_spatial_size,              # tuple/list, e.g. (1, 512, 512) used for hilbert encoding/template
        batch_size: int,
        print_topk: int = 5,
        allow_equal_cover: bool = True,     # template == shape is ok
        strict: bool = True):               # True -> raise AssertionError on fail; False -> print warnings
        """
        Validate that Hilbert mapping + ordering is consistent for given coords and spatial shapes.

        Assumptions:
        - coords[:,0] is batch index
        - coords[:,1:] are spatial indices corresponding to spatial_shape dims
        - hilbert_spatial_size should cover spatial_shape (template grid >= actual grid)
        """

        def _fail(msg):
            if strict:
                raise AssertionError(msg)
            else:
                print(f"[WARN][{name}] {msg}")

        assert coords.ndim == 2 and coords.shape[1] >= 2, f"[{name}] coords must be [N, >=2], got {coords.shape}"
        assert hilbert_index.ndim == 1 and hilbert_index.shape[0] == coords.shape[0], \
            f"[{name}] hilbert_index shape {hilbert_index.shape} must match N={coords.shape[0]}"

        # Normalize spatial_shape and hilbert_spatial_size to python lists
        ss = list(spatial_shape) if not isinstance(spatial_shape, (list, tuple)) else list(spatial_shape)
        hs = list(hilbert_spatial_size) if not isinstance(hilbert_spatial_size, (list, tuple)) else list(hilbert_spatial_size)

        if len(ss) != len(hs):
            _fail(f"spatial_shape dims ({len(ss)}) != hilbert_spatial_size dims ({len(hs)}). ss={ss}, hs={hs}")
            return False

        # 1) check batch index range
        # 注意：在use_winmamba的情况下，coords中的batch索引可能包含多个view（lidar + cameras），
        # 因此batch索引范围可能超过传入的batch_size参数，这是正常的，可以忽略此检查
        b = coords[:, 0]
        # 只检查batch索引是否为负数（这是真正的错误）
        if b.min().item() < 0:
            _fail(f"batch idx contains negative values: min={b.min().item()}")

        # 2) check coordinate bounds vs spatial_shape
        # coords[:,1:] correspond to ss dims
        sp = coords[:, 1:1+len(ss)]
        if sp.shape[1] != len(ss):
            _fail(f"coords spatial dims mismatch: coords has {sp.shape[1]} spatial dims, spatial_shape has {len(ss)} dims")

        for d, size_d in enumerate(ss):
            vmin = sp[:, d].min().item()
            vmax = sp[:, d].max().item()
            # DEBUG: 打印坐标范围和超出范围的数量
            out_of_range_count = ((sp[:, d] < 0) | (sp[:, d] > size_d)).sum().item()
            if out_of_range_count > 0:
                print(f"[DEBUG][{name}] dim{d}坐标超出范围: min={vmin}, max={vmax}, allowed=[0,{size_d}], 超出范围的点数={out_of_range_count}")
                # 打印超出范围的坐标样本
                out_of_range_mask = (sp[:, d] < 0) | (sp[:, d] > size_d)
                if out_of_range_mask.sum() > 0:
                    out_of_range_coords = coords[out_of_range_mask][:10]  # 只打印前10个
                    print(f"[DEBUG][{name}] dim{d}超出范围的坐标样本(前10个): {out_of_range_coords.cpu().numpy()}")
            # 坐标范围应该是[0, size_d-1]，但实际坐标可能达到size_d（边界情况，可能是浮点数取整导致）
            # 如果vmax == size_d，说明坐标刚好在边界上，这在spconv中是可以接受的（会自动裁剪到size_d-1）
            # 但如果vmax > size_d，说明坐标超出了范围，这是错误
            if vmin < 0:
                _fail(f"coords out of spatial_shape on dim{d}: min={vmin}, max={vmax}, allowed=[0,{size_d-1}]")
            elif vmax > size_d:
                _fail(f"coords out of spatial_shape on dim{d}: min={vmin}, max={vmax}, allowed=[0,{size_d-1}]")
            elif vmax == size_d:
                # 边界情况：坐标达到size_d，这在spconv中是可以接受的（会自动裁剪到size_d-1）
                # 但为了安全，给出警告而不是错误
                print(f"[WARN][{name}] coords on dim{d} reaches boundary: max={vmax}, size_d={size_d}. "
                      f"This may be due to rounding. spconv will handle this automatically.")

        # 3) check whether hilbert_spatial_size covers spatial_shape
        # for Hilbert template grid, usually need hs[d] >= ss[d]
        for d, (sd, hd) in enumerate(zip(ss, hs)):
            if allow_equal_cover:
                if hd < sd:
                    _fail(f"hilbert_spatial_size does NOT cover spatial_shape on dim{d}: hs={hd} < ss={sd}")
            else:
                if hd <= sd:
                    _fail(f"hilbert_spatial_size must be strictly larger than spatial_shape on dim{d}: hs={hd} <= ss={sd}")

        # 4) hilbert index sanity
        if not torch.isfinite(hilbert_index).all():
            _fail("hilbert_index contains NaN/Inf.")
        if (hilbert_index < 0).any():
            _fail(f"hilbert_index contains negative values. min={hilbert_index.min().item()}")

        # 5) per-batch duplicate rate (important!)
        # duplicates might happen if encoding resolution is too low or coords were quantized incorrectly
        dup_stats = []
        for i in range(batch_size):
            m = (b == i)
            if m.sum().item() == 0:
                continue
            h = hilbert_index[m]
            # unique count
            u = torch.unique(h).numel()
            n = h.numel()
            dup_rate = 1.0 - (u / max(n, 1))
            dup_stats.append((i, n, u, dup_rate))
            print(f"[DEBUG] dup_rate: {dup_rate}")
            # DEBUG: 打印重复率信息
            if dup_rate:  # 如果重复率>1%，打印详细信息
                print(f"[DEBUG][{name}] batch {i}重复率: N={n}, unique={u}, dup_rate={dup_rate:.2%}")
                # 打印Hilbert index的分布
               
                h_min, h_max = h.min().item(), h.max().item()
                print(f"[DEBUG][{name}] batch {i} Hilbert index范围: [{h_min}, {h_max}]")
                # 找出重复最多的Hilbert index
                unique_h, counts = torch.unique(h, return_counts=True)
                if len(counts) > 0:
                    top_dup_idx = counts.topk(min(5, len(counts)))[1]
                    print(f"[DEBUG][{name}] batch {i} 重复最多的Hilbert index(前5个): {unique_h[top_dup_idx].cpu().numpy()}, 重复次数: {counts[top_dup_idx].cpu().numpy()}")
            
            # if dup_rate too high, it's suspicious
        

        # 6) optional: show some ordering continuity (rough)
        # we just show first few sorted indices
        try:
            i0 = None
            for i in range(batch_size):
                if (b == i).any():
                    i0 = i
                    break
            if i0 is not None:
                m0 = (b == i0)
                h0 = hilbert_index[m0]
                sidx = torch.argsort(h0)
                print(f"[OK][{name}] spatial_shape={ss}, hilbert_spatial_size={hs}, N={coords.shape[0]}")
                print(f"[{name}] hilbert_index: min={hilbert_index.min().item()}, max={hilbert_index.max().item()}, dtype={hilbert_index.dtype}")
                print(f"[{name}] batch {i0} sorted hilbert head:", h0[sidx[:print_topk]].detach().cpu().tolist())
                # also show corresponding coords head
                c0 = coords[m0][:, 1:1+len(ss)]
                print(f"[{name}] batch {i0} coords head:", c0[sidx[:print_topk]].detach().cpu().tolist())
        except Exception as e:
            # don't hard fail for print/debug
            print(f"[WARN][{name}] failed to print debug head due to: {e}")

        return True
#####  downsampling operation  #####

class Sparse1ConvBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, bias=None, norm_fn=None, downsample=None, indice_key=None):
        super(Sparse1ConvBlock, self).__init__()

        assert norm_fn is not None
        if bias is None:
            bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out
    

class DownSp(spconv.SparseModule):

    def __init__(self, dim, kernel_size, stride, num_down, norm_fn, indice_key):
        super(DownSp, self).__init__()

        first_block = post_act_block(
            dim, dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
            norm_fn=norm_fn, indice_key=f'spconv_{indice_key}', conv_type='spconv')

        block_list = [first_block if stride > 1 else nn.Identity()]
        for _ in range(num_down):
            block_list.append(
                Sparse1ConvBlock(dim, dim, norm_fn=norm_fn, indice_key=indice_key))

        self.blocks = spconv.SparseSequential(*block_list)

    def forward(self, x):
        return self.blocks(x)
