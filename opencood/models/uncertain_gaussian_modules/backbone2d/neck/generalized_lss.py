import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.models.mambafusion_modules.model_utils.basic_block_2d import BasicBlock2D


class GeneralizedLSSFPN(nn.Module):
    """
        This module implements FPN, which creates pyramid features built on top of some input feature maps.
        This code is adapted from https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/necks/fpn.py with minimal modifications.
    """
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        in_channels =  self.model_cfg.IN_CHANNELS
        out_channels = self.model_cfg.OUT_CHANNELS
        num_ins = len(in_channels)
        num_outs = self.model_cfg.NUM_OUTS
        start_level = self.model_cfg.START_LEVEL
        end_level = self.model_cfg.END_LEVEL
        use_bias = self.model_cfg.get('USE_BIAS',False)
        self.align_corners = self.model_cfg.get('ALIGN_CORNERS',False)

        self.in_channels = in_channels

        if end_level == -1:
            self.backbone_end_level = num_ins - 1
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = BasicBlock2D(
                in_channels[i] + (in_channels[i + 1] if i == self.backbone_end_level - 1 else out_channels),
                out_channels, kernel_size=1, bias = use_bias
            )
            fpn_conv = BasicBlock2D(out_channels,out_channels, kernel_size=3, padding=1, bias = use_bias)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                image_features (list[tensor]): Multi-stage features from image backbone.
        Returns:
            batch_dict:
                image_fpn (list(tensor)): FPN features.
        """
        # Helper to run FPN on one agent's inputs
        def run_fpn_single(inputs):
            assert len(inputs) == len(self.in_channels)
            laterals = [inputs[i + self.start_level] for i in range(len(inputs))]
            used_backbone_levels = len(laterals) - 1
            for i in range(used_backbone_levels - 1, -1, -1):
                x = F.interpolate(
                    laterals[i + 1],
                    size=laterals[i].shape[2:],
                    mode='bilinear', align_corners=self.align_corners
                )
                laterals[i] = torch.cat([laterals[i], x], dim=1)
                laterals[i] = self.lateral_convs[i](laterals[i])
                laterals[i] = self.fpn_convs[i](laterals[i])
            outs = [laterals[i] for i in range(used_backbone_levels)]
            return tuple(outs)

        # Case 1: top-level image_features (single-stream)
        if 'image_features' in batch_dict:
            batch_dict['image_fpn'] = run_fpn_single(batch_dict['image_features'])
            return batch_dict

        # Case 2: per-agent image_features; optionally add simple inter-agent interaction
        # Collect agent keys with image_features
        agent_keys = [k for k, v in batch_dict.items() if isinstance(v, dict) and ('image_features' in v)]
        if len(agent_keys) == 0:
            raise KeyError('No image_features found in batch_dict or any agent sub-dicts')
        
        # 过滤掉无效的agent（图像特征为空或全零）
        valid_agent_keys = []
        for agent_name in agent_keys:
            agent_dict = batch_dict[agent_name]
            if 'image_features' in agent_dict:
                image_features = agent_dict['image_features']
                # 检查图像特征是否有效
                is_valid = True
                if isinstance(image_features, (list, tuple)):
                    for feat in image_features:
                        if feat is None or feat.numel() == 0 or torch.count_nonzero(feat).item() == 0:
                            is_valid = False
                            break
                else:
                    if image_features is None or image_features.numel() == 0 or torch.count_nonzero(image_features).item() == 0:
                        is_valid = False
                
                if is_valid:
                    valid_agent_keys.append(agent_name)
                else:
                    print(f"[GeneralizedLSS] 跳过无效的{agent_name} agent图像特征")
        
        if len(valid_agent_keys) == 0:
            raise KeyError('No valid image_features found in any agent sub-dicts')
        
        agent_keys = valid_agent_keys

        do_interact = getattr(self.model_cfg, 'AGENT_INTERACT', False)
        # Ensure all agents have same number of pyramid levels
        num_levels = len(batch_dict[agent_keys[0]]['image_features'])
        for a in agent_keys:
            assert len(batch_dict[a]['image_features']) == num_levels, 'Agents have different FPN input levels.'

        if do_interact and len(agent_keys) > 1:
            # Mean interaction across agents per level (residual add of others' mean)
            # Stack per level: [A, B, C, H, W]
            stacked = [torch.stack([batch_dict[a]['image_features'][lvl] for a in agent_keys], dim=0)
                       for lvl in range(num_levels)]
            sums = [s.sum(dim=0) for s in stacked]
            A = len(agent_keys)
            for ai, a in enumerate(agent_keys):
                fused_inputs = []
                for lvl in range(num_levels):
                    if A > 1:
                        others_mean = (sums[lvl] - stacked[lvl][ai]) / (A - 1)
                        fused = batch_dict[a]['image_features'][lvl] + others_mean
                    else:
                        fused = batch_dict[a]['image_features'][lvl]
                    fused_inputs.append(fused)
                batch_dict[a]['image_fpn'] = run_fpn_single(fused_inputs)
        else:
            # No interaction; process each agent independently
            for a in agent_keys:
                batch_dict[a]['image_fpn'] = run_fpn_single(batch_dict[a]['image_features'])

        return batch_dict


class MY_FPN(nn.Module):
    """
        This module implements FPN, which creates pyramid features built on top of some input feature maps.
        This code is adapted from https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/necks/fpn.py with minimal modifications.
    """
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        in_channels =  self.model_cfg.IN_CHANNELS
        out_channels = self.model_cfg.OUT_CHANNELS
        num_ins = len(in_channels)
        num_outs = self.model_cfg.NUM_OUTS
        start_level = self.model_cfg.START_LEVEL
        end_level = self.model_cfg.END_LEVEL
        use_bias = self.model_cfg.get('USE_BIAS',False)
        self.align_corners = self.model_cfg.get('ALIGN_CORNERS',False)

        self.in_channels = in_channels

        if end_level == -1:
            self.backbone_end_level = num_ins - 1
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = BasicBlock2D(
                in_channels[i] + (in_channels[i + 1] if i == self.backbone_end_level - 1 else out_channels),
                out_channels, kernel_size=1, bias = use_bias
            )
            fpn_conv = BasicBlock2D(out_channels,out_channels, kernel_size=3, padding=1, bias = use_bias)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs):
        """
        Args:
            batch_dict:
                image_features (list[tensor]): Multi-stage features from image backbone.
        Returns:
            batch_dict:
                image_fpn (list(tensor)): FPN features.
        """
        # upsample -> cat -> conv1x1 -> conv3x3
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [inputs[i + self.start_level] for i in range(len(inputs))]

        # build top-down path
        used_backbone_levels = len(laterals) - 1
        for i in range(used_backbone_levels - 1, -1, -1):
            x = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                mode='nearest'

            )
            laterals[i] = torch.cat([laterals[i], x], dim=1) # [12, 256, 32, 88]
            laterals[i] = self.lateral_convs[i](laterals[i])
            laterals[i] = self.fpn_convs[i](laterals[i]) # [12, 256, 32, 88]

        # build outputs
        outs = [laterals[i] for i in range(used_backbone_levels)]
        return outs