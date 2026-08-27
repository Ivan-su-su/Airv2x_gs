"""Joint R90 P1 frontend: shared 64-ch F90 → HeatmapHead + DepthHead."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from opencood.models.gaussian_modules_0822.heatmap.head import build_heatmap_heads
from opencood.models.gaussian_modules_0822.highres_adapter import HighResFusion
from opencood.models.gaussian_modules_0822.image_frontend import (
    AGENT_TYPES,
    ImageFrontend,
    flatten_camera_world_z,
    present_camera_agents,
)
from opencood.models.gaussian_modules_0822.lss.categorical_depth import (
    CategoricalDepthMoments,
)
from opencood.models.gaussian_modules_0822.lss.head import (
    CATEGORICAL_DEPTH_AGENTS,
    DeltaHead,
    HeightEmbedding,
    build_depth_heads,
)
from opencood.models.gaussian_modules_0822.p1_layout import NUM_CLASSES, expected_feature_hw


class Airv2xGaussian0822(nn.Module):
    """Predictions only. GT is built by the trainer via target modules.

    Three independent CamEncode copies (vehicle / RSU / drone). Heatmap and
    Depth share trunk/up1/up2/HighResFusion/F90 within each agent. Does not
    instantiate ``LiftSplatShootEncoder`` or inherit ``Airv2xBase``.
    Official CamEncode depth_head is unused.
    """

    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__()
        self.args = args
        self.frontend = ImageFrontend(args)
        self.highres = nn.ModuleDict()
        self.depth_moments = nn.ModuleDict()
        for agent_type in AGENT_TYPES:
            self.highres[agent_type] = HighResFusion()
            if agent_type not in CATEGORICAL_DEPTH_AGENTS:
                continue
            cam_cfg = args[agent_type]["cam"]
            self.depth_moments[agent_type] = CategoricalDepthMoments(
                ddiscr=cam_cfg["grid_conf"]["ddiscr"],
                mode=cam_cfg["grid_conf"]["mode"],
            )
        self.heatmap_heads = build_heatmap_heads(args)
        self.depth_heads = build_depth_heads(args)
        self.drone_height_embed = HeightEmbedding()
        self.drone_delta_head = DeltaHead()

    def train(self, mode: bool = True) -> "Airv2xGaussian0822":
        """Keep CamEncode.training aligned with ``mode``; freeze EfficientNet BN."""
        super().train(mode)
        self.frontend.apply_train_eval_state(mode)
        self.highres.train(mode)
        self.heatmap_heads.train(mode)
        self.depth_heads.train(mode)
        if mode:
            for agent_type, fusion in self.highres.items():
                if not fusion.training:
                    raise AssertionError(
                        f"{agent_type} HighResFusion.training={fusion.training} "
                        "expected True"
                    )
            for agent_type, head in self.heatmap_heads.items():
                if not head.training:
                    raise AssertionError(
                        f"{agent_type} HeatmapHead.training={head.training} expected True"
                    )
            for agent_type, head in self.depth_heads.items():
                if not head.training:
                    raise AssertionError(
                        f"{agent_type} DepthHead.training={head.training} expected True"
                    )
        self.frontend.assert_train_eval_state(mode)
        return self

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Per-agent joint predictions. Present agents inferred from keys.

        Returns:
            ``{agent: {heatmap_logits, depth_logits, depth_z_mean, depth_z_var}}``
        """
        output: Dict[str, Dict[str, torch.Tensor]] = {}
        feat_h, feat_w = expected_feature_hw()
        for agent_type in present_camera_agents(data_dict):
            imgs = data_dict[agent_type]["batch_merged_cam_inputs"]["imgs"]
            r2, f45 = self.frontend.extract_backbone_features(agent_type, imgs)
            f90 = self.highres[agent_type](r2, f45)
            if tuple(f90.shape[1:]) != (64, feat_h, feat_w):
                raise AssertionError(
                    f"{agent_type} f90 {tuple(f90.shape)} expected "
                    f"[N,64,{feat_h},{feat_w}]"
                )
            heatmap_logits = self.heatmap_heads[agent_type](f90)
            if tuple(heatmap_logits.shape[-3:]) != (NUM_CLASSES, feat_h, feat_w):
                raise AssertionError(
                    f"{agent_type} heatmap_logits {tuple(heatmap_logits.shape)} "
                    f"expected [N,{NUM_CLASSES},{feat_h},{feat_w}]"
                )
            if agent_type == "drone":
                cam_inputs = data_dict[agent_type]["batch_merged_cam_inputs"]
                if "camera_world_z" not in cam_inputs:
                    raise KeyError(
                        "drone batch is missing camera_world_z; height must come "
                        "from odometry.ego_pos + camera.cords"
                    )
                height = flatten_camera_world_z(cam_inputs["camera_world_z"], imgs)
                height = height.to(device=f90.device, dtype=f90.dtype)
                if int(height.shape[0]) != int(f90.shape[0]):
                    raise AssertionError(
                        f"drone height {tuple(height.shape)} vs f90 N={int(f90.shape[0])}"
                    )
                height_embed = self.drone_height_embed(
                    height, (int(f90.shape[2]), int(f90.shape[3]))
                ).to(dtype=f90.dtype)
                delta_pred = self.drone_delta_head(f90, height_embed)
                depth_z_mean = height[:, None, None] + delta_pred
                output[agent_type] = {
                    "heatmap_logits": heatmap_logits,
                    "delta_pred": delta_pred,
                    "depth_z_mean": depth_z_mean,
                    "camera_world_z": height,
                }
                continue
            depth_logits = self.depth_heads[agent_type](f90)
            if tuple(depth_logits.shape[-2:]) != (feat_h, feat_w):
                raise AssertionError(
                    f"{agent_type} depth_logits {tuple(depth_logits.shape)} "
                    f"expected [N,D,{feat_h},{feat_w}]"
                )
            expected_d = int(self.depth_heads[agent_type].num_bins)
            if int(depth_logits.shape[1]) != expected_d:
                raise AssertionError(
                    f"{agent_type} depth_logits D={depth_logits.shape[1]} "
                    f"expected {expected_d}"
                )
            depth_z_mean, depth_z_var = self.depth_moments[agent_type](depth_logits)
            output[agent_type] = {
                "heatmap_logits": heatmap_logits,
                "depth_logits": depth_logits,
                "depth_z_mean": depth_z_mean,
                "depth_z_var": depth_z_var,
            }
        return output
