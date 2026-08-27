"""Official CamEncode wrapper: ImageNet trunk with BN-frozen fine-tune.

E0 is EfficientNet-B0 created by official
``CamEncode`` → ``EfficientNet.from_pretrained("efficientnet-b0")``.
P1 does not load any HEAL / AirV2X checkpoint overlay.

Official ``image_head`` and ``depth_head`` stay frozen and unused.
P1 depth uses the independent ``DepthHead`` on shared F90.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

import torch
from torch import nn

from opencood.models.sub_modules.lss_submodule import CamEncode

AGENT_TYPES: Tuple[str, ...] = ("vehicle", "rsu", "drone")
_BN_TYPES: Tuple[type, ...] = (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)
_OFFICIAL_UNUSED_HEADS: Tuple[str, ...] = ("image_head", "depth_head")


def flatten_camera_images(imgs: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Flatten ``imgs`` to ``[N, C, H, W]``.

    Args:
        imgs: ``[B_a, V, C, H, W]`` or ``[N, C, H, W]``.

    Returns:
        NCHW images and ``(B_a, V)``.
    """
    if imgs.dim() == 5:
        batch_agents, num_views, channels, height, width = imgs.shape
        flat = imgs.reshape(batch_agents * num_views, channels, height, width)
        return flat, (int(batch_agents), int(num_views))
    if imgs.dim() == 4:
        return imgs, (int(imgs.shape[0]), 1)
    raise ValueError(f"imgs must be 4D or 5D, got {tuple(imgs.shape)}")


def flatten_camera_world_z(
    camera_world_z: torch.Tensor, imgs: torch.Tensor
) -> torch.Tensor:
    """Flatten camera heights to ``[N]`` in the same order as ``imgs``.

    Args:
        camera_world_z: ``[B_a, V]`` or ``[N]``.
        imgs: ``[B_a, V, C, H, W]`` or ``[N, C, H, W]``.

    Returns:
        Height scalars ``[N]`` aligned with flattened images.
    """
    if imgs.dim() == 5:
        batch_agents, num_views = int(imgs.shape[0]), int(imgs.shape[1])
        expected = batch_agents * num_views
        if camera_world_z.dim() == 2:
            if tuple(camera_world_z.shape) != (batch_agents, num_views):
                raise AssertionError(
                    f"camera_world_z {tuple(camera_world_z.shape)} vs imgs "
                    f"{tuple(imgs.shape[:2])}"
                )
            return camera_world_z.reshape(expected)
        if camera_world_z.dim() == 1:
            if int(camera_world_z.shape[0]) != expected:
                raise AssertionError(
                    f"camera_world_z {tuple(camera_world_z.shape)} expected [{expected}]"
                )
            return camera_world_z
        raise ValueError(
            f"camera_world_z must be 1D or 2D, got {tuple(camera_world_z.shape)}"
        )
    if imgs.dim() == 4:
        expected = int(imgs.shape[0])
        flat = camera_world_z.reshape(-1)
        if int(flat.shape[0]) != expected:
            raise AssertionError(
                f"camera_world_z {tuple(camera_world_z.shape)} vs N={expected}"
            )
        return flat
    raise ValueError(f"imgs must be 4D or 5D, got {tuple(imgs.shape)}")


def present_camera_agents(ego_batch: Mapping[str, Any]) -> List[str]:
    """Return agent types that have camera ``imgs`` in this batch.

    Args:
        ego_batch: ``batch['ego']`` dictionary.

    Returns:
        Agent names discovered from ``imgs``, not from LiDAR.
    """
    present: List[str] = []
    for agent_type in AGENT_TYPES:
        agent_batch = ego_batch.get(agent_type)
        if not isinstance(agent_batch, dict):
            continue
        cam_inputs = agent_batch.get("batch_merged_cam_inputs")
        if not isinstance(cam_inputs, dict):
            continue
        imgs = cam_inputs.get("imgs")
        if torch.is_tensor(imgs) and imgs.numel() > 0:
            present.append(agent_type)
    return present


def _is_norm(module: nn.Module) -> bool:
    """True if ``module`` is a BatchNorm family layer."""
    return isinstance(module, _BN_TYPES)


def _camencode_r2_and_f45(
    encoder: CamEncode, rgb: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Official trunk endpoint loop + official up1/up2. One RGB forward.

    Same control flow as ``CamEncode.get_eff_features`` in
    ``lss_submodule.py``. Does not call ``EfficientNet.extract_endpoints``.
    """
    endpoints: Dict[str, torch.Tensor] = {}
    x = encoder.trunk._swish(encoder.trunk._bn0(encoder.trunk._conv_stem(rgb)))
    prev_x = x
    for idx, block in enumerate(encoder.trunk._blocks):
        drop_connect_rate = encoder.trunk._global_params.drop_connect_rate
        if drop_connect_rate:
            drop_connect_rate *= float(idx) / len(encoder.trunk._blocks)
        x = block(x, drop_connect_rate=drop_connect_rate)
        if prev_x.size(2) > x.size(2):
            endpoints["reduction_{}".format(len(endpoints) + 1)] = prev_x
        prev_x = x
    endpoints["reduction_{}".format(len(endpoints) + 1)] = x
    r2 = endpoints["reduction_2"]
    x = encoder.up1(endpoints["reduction_5"], endpoints["reduction_4"])
    if encoder.downsample == 8:
        x = encoder.up2(x, endpoints["reduction_3"])
    return r2, x


class ImageFrontend(nn.Module):
    """Three independent official ``CamEncode`` modules.

    ImageNet EfficientNet-B0 is loaded inside ``CamEncode.__init__``.
    Non-BN trunk weights are fine-tuned. EfficientNet BN stays eval with
    frozen running stats and frozen affine parameters. Official
    ``image_head`` / ``depth_head`` remain unused and frozen.

    Args:
        model_cfg: ``hypes['model']['args']`` including per-agent ``cam``.
    """

    def __init__(self, model_cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.encoders = nn.ModuleDict()
        for agent_type in AGENT_TYPES:
            self.encoders[agent_type] = self._build_camencode(agent_type)
        self._configure_trainable_state()
        print(
            "[E0] ImageNet EfficientNet-B0 via official "
            "CamEncode / EfficientNet.from_pretrained('efficientnet-b0'); "
            "non-BN trunk fine-tuned; EfficientNet BN frozen in eval; "
            "official image_head/depth_head frozen unused; "
            "up1/up2 trainable"
        )

    def _build_camencode(self, agent_type: str) -> CamEncode:
        """Construct official CamEncode. ``use_depth_gt`` must stay false."""
        cam_cfg = self.model_cfg[agent_type]["cam"]
        grid_conf = cam_cfg["grid_conf"]
        ddiscr = list(grid_conf["ddiscr"])
        use_gt_depth = bool(cam_cfg.get("use_depth_gt", False))
        if use_gt_depth:
            raise ValueError(
                f"{agent_type} requires cam.use_depth_gt=false so official "
                "depth_head exists (unused by P1). Do not silently override yaml."
            )
        downsample = int(cam_cfg["img_downsample"])
        if downsample != 8:
            raise ValueError(
                f"{agent_type} img_downsample must be 8 to keep official up2/F45, "
                f"got {downsample}"
            )
        encoder = CamEncode(
            D=int(ddiscr[2]),
            C=int(cam_cfg["img_features"]),
            downsample=downsample,
            ddiscr=ddiscr,
            mode=str(grid_conf["mode"]),
            use_gt_depth=False,
            depth_supervision=bool(cam_cfg.get("depth_supervision", True)),
        )
        return encoder

    def _freeze_efficientnet_bn(self, trunk: nn.Module) -> None:
        """Keep EfficientNet BN in eval with frozen affine and running stats."""
        for module in trunk.modules():
            if not _is_norm(module):
                continue
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def _configure_trainable_state(self) -> None:
        """Fine-tune non-BN trunk; freeze EfficientNet BN and official heads."""
        for encoder in self.encoders.values():
            for param in encoder.trunk.parameters():
                param.requires_grad = True
            self._freeze_efficientnet_bn(encoder.trunk)
            # ImageNet classifier tail is unused by CamEncode feature extraction.
            for unused_name in ("_conv_head", "_fc"):
                unused = getattr(encoder.trunk, unused_name, None)
                if unused is None:
                    continue
                for param in unused.parameters():
                    param.requires_grad = False
            for name in _OFFICIAL_UNUSED_HEADS:
                module = getattr(encoder, name, None)
                if module is None:
                    continue
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
            for name in ("up1", "up2"):
                module = getattr(encoder, name, None)
                if module is None:
                    raise RuntimeError(f"CamEncode missing trainable module {name}")
                for param in module.parameters():
                    param.requires_grad = True

    def apply_train_eval_state(self, mode: bool) -> None:
        """Set train/eval without putting the whole EfficientNet trunk in eval.

        ``CamEncode.training`` follows ``mode`` so
        ``bin_depths(..., target=camencode.training)`` keeps official
        train-time bin clamping. EfficientNet BN stays eval. Official unused
        heads stay eval. up1/up2 (including their BN) follow ``mode``.

        Args:
            mode: ``True`` for train, ``False`` for eval.
        """
        for encoder in self.encoders.values():
            encoder.train(mode)
            for name in _OFFICIAL_UNUSED_HEADS:
                module = getattr(encoder, name, None)
                if module is not None:
                    module.eval()
            self._freeze_efficientnet_bn(encoder.trunk)
            for name in ("up1", "up2"):
                getattr(encoder, name).train(mode)

    def assert_train_eval_state(self, mode: bool) -> None:
        """Assert BN-freeze / fine-tune contract after ``model.train()`` / ``eval()``.

        Args:
            mode: Current training flag of the parent model.
        """
        for agent_type, encoder in self.encoders.items():
            if bool(encoder.training) != bool(mode):
                raise AssertionError(
                    f"{agent_type} CamEncode.training={encoder.training} expected {mode}"
                )
            for name in _OFFICIAL_UNUSED_HEADS:
                module = getattr(encoder, name, None)
                if module is None:
                    continue
                if module.training:
                    raise AssertionError(
                        f"{agent_type}.{name}.training={module.training} expected False"
                    )
                if any(param.requires_grad for param in module.parameters()):
                    raise AssertionError(f"{agent_type}.{name} still requires_grad")
            for module in encoder.trunk.modules():
                if not _is_norm(module):
                    continue
                if module.training:
                    raise AssertionError(
                        f"{agent_type} EfficientNet BN still in train mode"
                    )
                if any(param.requires_grad for param in module.parameters()):
                    raise AssertionError(
                        f"{agent_type} EfficientNet BN affine still trainable"
                    )
            if mode:
                n_trainable_trunk = sum(
                    1 for param in encoder.trunk.parameters() if param.requires_grad
                )
                if n_trainable_trunk == 0:
                    raise AssertionError(
                        f"{agent_type} EfficientNet non-BN trunk has no trainable params"
                    )
            for name in ("up1", "up2"):
                module = getattr(encoder, name)
                if bool(module.training) != bool(mode):
                    raise AssertionError(
                        f"{agent_type}.{name}.training={module.training} expected {mode}"
                    )
                if not all(param.requires_grad for param in module.parameters()):
                    raise AssertionError(f"{agent_type}.{name} is not fully trainable")

    def extract_backbone_features(
        self, agent_type: str, imgs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One RGB pass: official endpoints, then official up1/up2.

        Depth channel never enters the trunk. Returns ``reduction_2`` and
        ``F45`` (``get_eff_features`` output).

        Args:
            agent_type: ``vehicle``, ``rsu``, or ``drone``.
            imgs: ``[B_a, V, C, H, W]`` or ``[N, C, H, W]``, C>=3.

        Returns:
            ``r2`` and ``f45``. For 360x640 / downsample=8: ``[N,24,90,160]``
            and ``[N,256,45,80]``.
        """
        flat_imgs, _ = flatten_camera_images(imgs)
        rgb = flat_imgs[:, :3, :, :]
        return _camencode_r2_and_f45(self.encoders[agent_type], rgb)
