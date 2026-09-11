"""Standalone RSU-only P1 at output stride 8 (45x80).

Does not instantiate Vehicle/Drone, Gaussians, or detection heads.
The only new operator is a trainable stride-2 Conv on R2.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

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
from opencood.models.gaussian_modules_0822.p1_layout import (
    F90_CHANNELS,
    NUM_CLASSES,
    feature_hw_for_stride,
)
from opencood.models.gaussian_modules_0822.vehicle_v45_p1 import (
    F45_CHANNELS,
    OUTPUT_STRIDE,
    R2_CHANNELS,
)

_V90_PREFIX_MAP: Tuple[Tuple[str, str], ...] = (
    ("frontend.encoders.rsu.", "encoder."),
    ("highres.rsu.", "highres."),
    ("heatmap_heads.rsu.", "heatmap_head."),
    ("depth_heads.rsu.", "depth_head."),
    ("depth_moments.rsu.", "depth_moments."),
)


def remap_v90_rsu_key(key: str) -> str:
    """Map a V90 RSU P1 state_dict key onto this standalone model.

    Args:
        key: Checkpoint tensor name, without a ``module.`` prefix.

    Returns:
        Remapped name, or ``""`` if the key is not an RSU P1 tensor.
    """
    for src, dst in _V90_PREFIX_MAP:
        if key.startswith(src):
            return dst + key[len(src) :]
    return ""


class RsuV45P1(nn.Module):
    """RSU CamEncode → R2 stride-2 Conv → HighResFusion → P1 heads.

    Args:
        args: ``hypes['model']['args']``. Reads RSU ``cam`` and optional
            ``output_stride`` (default 8). Vehicle/Drone blocks are ignored.
    """

    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__()
        cam_cfg = args["rsu"]["cam"]
        image_hw = tuple(int(v) for v in cam_cfg["data_aug_conf"]["final_dim"])
        self.output_stride = int(args.get("output_stride", OUTPUT_STRIDE))
        self.feat_h, self.feat_w = feature_hw_for_stride(image_hw, self.output_stride)
        self.num_depth_bins = int(cam_cfg["grid_conf"]["ddiscr"][2])

        self.encoder = build_camencode(cam_cfg, "rsu")
        configure_camencode_trainable_state(self.encoder)
        self.r2_downsample = nn.Conv2d(
            R2_CHANNELS, R2_CHANNELS, kernel_size=3, stride=2, padding=1
        )
        self.highres = HighResFusion(
            r2_channels=R2_CHANNELS,
            f45_channels=F45_CHANNELS,
            out_channels=F90_CHANNELS,
        )
        self.heatmap_head = HeatmapHead(
            in_channels=F90_CHANNELS, num_classes=NUM_CLASSES
        )
        self.depth_head = DepthHead(
            num_bins=self.num_depth_bins, in_channels=F90_CHANNELS
        )
        self.depth_moments = CategoricalDepthMoments(
            ddiscr=cam_cfg["grid_conf"]["ddiscr"],
            mode=str(cam_cfg["grid_conf"]["mode"]),
        )
        print(
            "[V45] RSU-only P1: CamEncode R2/F45, Conv3x3 stride-2 on R2, "
            f"HighResFusion/heads at {self.feat_h}x{self.feat_w} "
            f"(stride={self.output_stride}, D={self.num_depth_bins})"
        )

    def train(self, mode: bool = True) -> "RsuV45P1":
        """Train V45 modules; keep EfficientNet BN frozen in eval."""
        super().train(mode)
        apply_camencode_train_eval_state(self.encoder, mode)
        return self

    def assert_train_eval_state(self, mode: bool) -> None:
        """Assert the RSU CamEncode BN-freeze / fine-tune contract."""
        assert_camencode_train_eval_state(self.encoder, mode, "rsu")
        if mode and not self.r2_downsample.weight.requires_grad:
            raise AssertionError("r2_downsample is not trainable")

    def extract_r2_f45(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """RGB-only CamEncode pass. Depth never enters the trunk.

        Args:
            imgs: ``[B_a, V, C, H, W]`` or ``[N, C, H, W]``, C>=3.

        Returns:
            ``r2`` ``[N, 24, 90, 160]`` and ``f45`` ``[N, 256, 45, 80]``
            for 360x640 inputs.
        """
        flat_imgs, _ = flatten_camera_images(imgs)
        return _camencode_r2_and_f45(self.encoder, flat_imgs[:, :3, :, :])

    def forward_features(self, imgs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the V45 stack and return intermediate tensors.

        Args:
            imgs: RSU camera batch, ``[B_a, V, C, H, W]`` or ``[N, C, H, W]``.

        Returns:
            Dict with ``r2``, ``f45``, ``r2_down``, ``fused``,
            ``heatmap_logits``, ``depth_logits``, ``depth_z_mean``,
            ``depth_z_var``.
        """
        r2, f45 = self.extract_r2_f45(imgs)
        r2_down = self.r2_downsample(r2)
        fused = self.highres(r2_down, f45)
        heatmap_logits = self.heatmap_head(fused)
        depth_logits = self.depth_head(fused)
        depth_z_mean, depth_z_var = self.depth_moments(depth_logits)
        return {
            "r2": r2,
            "f45": f45,
            "r2_down": r2_down,
            "fused": fused,
            "heatmap_logits": heatmap_logits,
            "depth_logits": depth_logits,
            "depth_z_mean": depth_z_mean,
            "depth_z_var": depth_z_var,
        }

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Predict RSU heatmap and categorical depth at stride 8.

        Args:
            data_dict: ``batch['ego']``. Only ``rsu`` cameras are read.

        Returns:
            ``{'rsu': {heatmap_logits, depth_logits, depth_z_mean,
            depth_z_var}}``.
        """
        payload = data_dict["rsu"]
        imgs = payload["batch_merged_cam_inputs"]["imgs"]
        feats = self.forward_features(imgs)
        rsu = {
            "heatmap_logits": feats["heatmap_logits"],
            "depth_logits": feats["depth_logits"],
            "depth_z_mean": feats["depth_z_mean"],
            "depth_z_var": feats["depth_z_var"],
        }
        payload.update(rsu)
        return {"rsu": rsu}

    def load_v90_rsu_weights(
        self, path: str, device: torch.device
    ) -> Dict[str, List[str]]:
        """Load matching V90 RSU P1 tensors. New R2 conv stays random.

        Args:
            path: V90 P1 checkpoint with ``model_state_dict``.
            device: Map-location for ``torch.load``.

        Returns:
            Report dict with ``loaded``, ``new``, ``mismatched``, ``skipped``.
        """
        ckpt = torch.load(path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
            epoch = ckpt.get("epoch")
        elif isinstance(ckpt, dict):
            state = ckpt
            epoch = None
        else:
            raise KeyError(f"{path} is not a state_dict checkpoint")
        if not isinstance(state, dict) or not state:
            raise KeyError(f"{path} has no usable state_dict")
        first = next(iter(state))
        if first.startswith("module."):
            state = {key[7:]: value for key, value in state.items()}

        current = self.state_dict()
        loaded: List[str] = []
        mismatched: List[str] = []
        skipped: List[str] = []
        mapped: Dict[str, torch.Tensor] = {}
        for src_key, value in state.items():
            dst_key = remap_v90_rsu_key(src_key)
            if not dst_key:
                skipped.append(src_key)
                continue
            if dst_key not in current:
                skipped.append(src_key)
                continue
            if tuple(current[dst_key].shape) != tuple(value.shape):
                mismatched.append(
                    f"{src_key} -> {dst_key}: ckpt {tuple(value.shape)} vs "
                    f"model {tuple(current[dst_key].shape)}"
                )
                continue
            mapped[dst_key] = value
            loaded.append(f"{src_key} -> {dst_key}")

        self.load_state_dict(mapped, strict=False)
        new_keys = [key for key in current if key not in mapped]
        report = {
            "loaded": loaded,
            "new": new_keys,
            "mismatched": mismatched,
            "skipped": skipped,
        }
        print(f"Loaded V90 RSU P1 weights from {path} (epoch={epoch})")
        print(f"  loaded: {len(loaded)}")
        print(f"  new: {new_keys}")
        print(f"  mismatched: {len(mismatched)}")
        if mismatched:
            for item in mismatched:
                print(f"    {item}")
        print(f"  skipped (non-rsu / unmapped): {len(skipped)}")
        return report
