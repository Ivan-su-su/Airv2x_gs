# -*- coding: utf-8 -*-
"""Two-agent Griffin P1 adaptation model built from AirV2X Gaussian-0822.

The model deliberately reuses the AirV2X 0822 image stack:

    EfficientNet-B0 / CamEncode trunk -> R2 + F45 -> HighResFusion -> heads

but instantiates only two independent agent branches:

* vehicle: V45 (45x80) heatmap + categorical depth;
* drone: bottom-view V90 (90x160) heatmap only.

Drone depth is intentionally absent. Griffin downstream inference uses analytic
bottom-camera ray/ground-plane intersection instead of the AirV2X DeltaHead.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

import torch
from torch import nn

from opencood.models.gaussian_modules_0822.heatmap.head import HeatmapHead
from opencood.models.gaussian_modules_0822.highres_adapter import HighResFusion
from opencood.models.gaussian_modules_0822.image_frontend import (
    _camencode_r2_and_f45,
    apply_camencode_train_eval_state,
    assert_camencode_train_eval_state,
    build_camencode,
    configure_camencode_trainable_state,
    flatten_camera_images,
)
from opencood.models.gaussian_modules_0822.lss.categorical_depth import (
    CategoricalDepthMoments,
)
from opencood.models.gaussian_modules_0822.lss.head import DepthHead
from opencood.models.gaussian_modules_0822.p1_layout import F90_CHANNELS, NUM_CLASSES


R2_CHANNELS = 24
F45_CHANNELS = 256


class TwoAgentImageFrontend(nn.Module):
    """Independent Vehicle/Drone CamEncode backbones with AirV2X train-state rules."""

    def __init__(self, args: Mapping[str, Any]) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict()
        for agent in ("vehicle", "drone"):
            encoder = build_camencode(args[agent]["cam"], agent)
            configure_camencode_trainable_state(encoder)
            self.encoders[agent] = encoder
        print(
            "[Griffin P1] two independent EfficientNet-B0/CamEncode backbones: "
            "vehicle + drone; official CamEncode image/depth heads remain unused"
        )

    def train(self, mode: bool = True) -> "TwoAgentImageFrontend":
        super().train(mode)
        for encoder in self.encoders.values():
            apply_camencode_train_eval_state(encoder, mode)
        return self

    def assert_train_eval_state(self, mode: bool) -> None:
        for agent, encoder in self.encoders.items():
            assert_camencode_train_eval_state(encoder, mode, agent)

    def extract(self, agent: str, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        flat, _ = flatten_camera_images(imgs)
        return _camencode_r2_and_f45(self.encoders[agent], flat[:, :3])


class GriffinGaussianP1(nn.Module):
    """AirV2X-0822 P1 adapted to Griffin vehicle + drone-bottom."""

    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__()
        self.args = dict(args)
        self.frontend = TwoAgentImageFrontend(args)

        self.r2_downsample = nn.Conv2d(
            R2_CHANNELS, R2_CHANNELS, kernel_size=3, stride=2, padding=1
        )
        self.highres = nn.ModuleDict({
            "vehicle": HighResFusion(
                r2_channels=R2_CHANNELS,
                f45_channels=F45_CHANNELS,
                out_channels=F90_CHANNELS,
            ),
            "drone": HighResFusion(
                r2_channels=R2_CHANNELS,
                f45_channels=F45_CHANNELS,
                out_channels=F90_CHANNELS,
            ),
        })
        self.heatmap_heads = nn.ModuleDict({
            "vehicle": HeatmapHead(F90_CHANNELS, NUM_CLASSES),
            "drone": HeatmapHead(F90_CHANNELS, NUM_CLASSES),
        })

        veh_cam = args["vehicle"]["cam"]
        ddiscr = veh_cam["grid_conf"]["ddiscr"]
        mode = str(veh_cam["grid_conf"]["mode"])
        self.depth_heads = nn.ModuleDict({
            "vehicle": DepthHead(num_bins=int(ddiscr[2]), in_channels=F90_CHANNELS)
        })
        self.depth_moments = nn.ModuleDict({
            "vehicle": CategoricalDepthMoments(ddiscr=ddiscr, mode=mode)
        })

        print(
            "[Griffin P1] vehicle: V45 heatmap+depth; "
            "drone-bottom: V90 heatmap only; no learned drone depth"
        )

    def train(self, mode: bool = True) -> "GriffinGaussianP1":
        super().train(mode)
        self.frontend.train(mode)
        return self

    def assert_train_eval_state(self, mode: bool) -> None:
        self.frontend.assert_train_eval_state(mode)

    def _vehicle(self, imgs: torch.Tensor) -> Dict[str, torch.Tensor]:
        r2, f45 = self.frontend.extract("vehicle", imgs)
        r2_down = self.r2_downsample(r2)
        feat = self.highres["vehicle"](r2_down, f45)
        heatmap_logits = self.heatmap_heads["vehicle"](feat)
        depth_logits = self.depth_heads["vehicle"](feat)
        depth_z_mean, depth_z_var = self.depth_moments["vehicle"](depth_logits)
        return {
            "feat": feat,
            "heatmap_logits": heatmap_logits,
            "depth_logits": depth_logits,
            "depth_z_mean": depth_z_mean,
            "depth_z_var": depth_z_var,
        }

    def _drone(self, imgs: torch.Tensor) -> Dict[str, torch.Tensor]:
        r2, f45 = self.frontend.extract("drone", imgs)
        feat = self.highres["drone"](r2, f45)
        heatmap_logits = self.heatmap_heads["drone"](feat)
        return {
            "feat": feat,
            "heatmap_logits": heatmap_logits,
        }

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "vehicle": self._vehicle(batch["vehicle"]["imgs"]),
            "drone": self._drone(batch["drone"]["imgs"]),
        }

    def load_airv2x_v45(self, path: str, device: torch.device) -> Dict[str, Any]:
        """Load matching P1 tensors from AirV2X ``net_final_branch_v45.pth``.

        Keys are intentionally named like the full AirV2X model, so exact
        prefix matching is sufficient. RSU, Gaussian stages, BEV backbone,
        detection heads and drone DeltaHead are ignored.
        """
        ckpt = torch.load(path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
            epoch = ckpt.get("epoch")
        elif isinstance(ckpt, dict):
            state = ckpt
            epoch = None
        else:
            raise TypeError(f"Unsupported checkpoint object: {type(ckpt)}")
        if not state:
            raise KeyError(f"{path} has empty state_dict")

        # Strip common wrappers.
        cleaned: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            while key.startswith("module.") or key.startswith("model."):
                key = key.split(".", 1)[1]
            cleaned[key] = value

        current = self.state_dict()
        matched: Dict[str, torch.Tensor] = {}
        mismatched: List[str] = []
        for key, value in cleaned.items():
            if key not in current:
                continue
            if tuple(value.shape) != tuple(current[key].shape):
                mismatched.append(
                    f"{key}: ckpt={tuple(value.shape)} model={tuple(current[key].shape)}"
                )
                continue
            matched[key] = value

        self.load_state_dict(matched, strict=False)

        # Strong contract for Griffin adaptation: every trainable tensor must
        # be initialized from the already-trained AirV2X V45 branch. Frozen
        # official CamEncode heads / BN bookkeeping are not part of this check.
        trainable_keys = {
            name for name, param in self.named_parameters() if param.requires_grad
        }
        missing_trainable = sorted(trainable_keys.difference(matched.keys()))
        if missing_trainable:
            raise RuntimeError(
                "AirV2X V45 checkpoint did not initialize every trainable Griffin "
                f"P1 tensor. Missing {len(missing_trainable)} trainable keys; "
                f"examples={missing_trainable[:20]}"
            )

        required_families = (
            "frontend.encoders.vehicle.trunk.",
            "frontend.encoders.vehicle.up1.",
            "frontend.encoders.vehicle.up2.",
            "frontend.encoders.drone.trunk.",
            "frontend.encoders.drone.up1.",
            "frontend.encoders.drone.up2.",
            "highres.vehicle.",
            "highres.drone.",
            "heatmap_heads.vehicle.",
            "heatmap_heads.drone.",
            "depth_heads.vehicle.",
            "r2_downsample.",
        )
        missing_families = [
            prefix for prefix in required_families
            if not any(key.startswith(prefix) for key in matched)
        ]
        if missing_families:
            raise RuntimeError(
                "AirV2X V45 checkpoint is missing required Griffin P1 families: "
                f"{missing_families}"
            )

        family_counts = {
            prefix: sum(key.startswith(prefix) for key in matched)
            for prefix in required_families
        }
        report = {
            "path": path,
            "epoch": epoch,
            "matched": len(matched),
            "model_tensors": len(current),
            "match_ratio": len(matched) / max(len(current), 1),
            "trainable_tensors": len(trainable_keys),
            "trainable_initialized": len(trainable_keys),
            "family_counts": family_counts,
            "mismatched": mismatched,
        }
        print(
            f"[Griffin P1] loaded AirV2X V45 init from {path} "
            f"epoch={epoch} matched={len(matched)}/{len(current)} "
            f"({report['match_ratio']:.1%})"
        )
        print(
            f"[Griffin P1] AirV2X initialized ALL trainable tensors: "
            f"{len(trainable_keys)}/{len(trainable_keys)}"
        )
        for prefix, count in family_counts.items():
            print(f"  init {prefix:<42s} {count} tensors")
        if mismatched:
            print("[Griffin P1] shape mismatches:")
            for item in mismatched[:20]:
                print("  ", item)
        return report
