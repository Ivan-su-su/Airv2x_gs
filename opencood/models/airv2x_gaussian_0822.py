"""Joint Gaussian pipeline: image heads → Stage-1/2/3 → BEV backbone → det heads."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
from torch import nn

from opencood.models.common_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.common_modules.downsample_conv import DownsampleConv
from opencood.models.gaussian_modules_0822.base.gaussian_filter import (
    GaussianRangeFilter,
)
from opencood.models.gaussian_modules_0822.base.gaussian_geometry_encoder import (
    GaussianGeometryEncoder,
)
from opencood.models.gaussian_modules_0822.base.projection import CAMERA_GEOMETRY_KEY
from opencood.models.gaussian_modules_0822.cross_agent.adapter import (
    AgentResidualAdapter,
)
from opencood.models.gaussian_modules_0822.cross_agent.interaction import (
    build_cross_agent_interactions,
)
from opencood.models.gaussian_modules_0822.heatmap.head import build_heatmap_heads
from opencood.models.gaussian_modules_0822.highres_adapter import HighResFusion
from opencood.models.gaussian_modules_0822.image_frontend import (
    AGENT_TYPES,
    ImageFrontend,
    flatten_camera_world_z,
    present_camera_agents,
)
from opencood.models.gaussian_modules_0822.initializer import GaussianInitializer
from opencood.models.gaussian_modules_0822.interaction.gaussian_context import (
    build_gaussian_context_interaction,
)
from opencood.models.gaussian_modules_0822.interaction.intra_view import (
    IntraViewInteraction,
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
from opencood.models.gaussian_modules_0822.p1_layout import F90_CHANNELS
from opencood.models.gaussian_modules_0822.projection.mamba_projection_wrapper import (
    MambaProjectionWrapper,
)
from opencood.models.gaussian_modules_0822.refinement.gaussian_refiner import (
    GaussianRefiner,
)
from opencood.models.gaussian_modules_0822.sampling.gaussian_offset_sampler import (
    GaussianOffsetSampler,
)


class Airv2xGaussian0822(nn.Module):
    """Heatmap/depth seeds, Gaussian interaction, then reused BEV det heads.

    Stage-1/2 share the geometry encoder, sampler, and projector.
    Each stage owns its Refiner. Stage-2 and Stage-3 each own an
    AgentResidualAdapter. Interaction blocks are per-stage.
    ``BaseBEVBackbone`` / ``DownsampleConv`` / cls / reg / obj match the
    V2X-ViT detection stack. Intermediate tensors are written onto
    ``data_dict``; the return value is ``psm`` / ``rm`` / ``obj``.
    """

    def __init__(self, args: Dict[str, Any]) -> None:
        super().__init__()
        self.args = args
        self.vehicle_p1_v45 = bool(args.get("vehicle_p1_v45", False))
        self.frontend = ImageFrontend(args)
        self.highres = nn.ModuleDict()
        self.depth_moments = nn.ModuleDict()
        self.r2_downsample = (
            nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1)
            if self.vehicle_p1_v45
            else None
        )
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

        self.initializer = GaussianInitializer(
            z_bins={
                agent: moments.z_bins
                for agent, moments in self.depth_moments.items()
            },
            range_filter=GaussianRangeFilter(
                args["cav_range"],
                enabled=bool(args["gaussian_range_filter"]["enabled"]),
            ),
        )

        refinement_cfg = args["gaussian_refinement"]
        encoder = GaussianGeometryEncoder(
            geo_dim=int(refinement_cfg["geometry"]["geo_dim"])
        )
        sampler = GaussianOffsetSampler(
            feature_dim=F90_CHANNELS, geometry_encoder=encoder
        )
        projector = MambaProjectionWrapper()
        stage1_refiner = GaussianRefiner(
            encoder, feature_dim=F90_CHANNELS, cfg=refinement_cfg
        )
        stage2_refiner = GaussianRefiner(
            encoder, feature_dim=F90_CHANNELS, cfg=refinement_cfg
        )
        stage2_adapter = AgentResidualAdapter(feature_dim=F90_CHANNELS)
        stage3_adapter = AgentResidualAdapter(
            feature_dim=F90_CHANNELS,
            prenorm=True,
        )
        attn_cfg = args.get("gaussian_attention") or {}
        self.intra_view = IntraViewInteraction(
            feature_dim=F90_CHANNELS,
            sampler=sampler,
            projector=projector,
            refiner=stage1_refiner,
            refinement_cfg=refinement_cfg,
            attn_cfg=attn_cfg,
            points_frame="ego",
        )
        self.cross_agent = build_cross_agent_interactions(
            feature_dim=F90_CHANNELS,
            refinement_cfg=refinement_cfg,
            sampler=sampler,
            projector=projector,
            adapter=stage2_adapter,
            refiner=stage2_refiner,
            attn_cfg=attn_cfg,
        )
        self.stage3 = build_gaussian_context_interaction(
            feature_dim=F90_CHANNELS,
            stage3_cfg=args["gaussian_stage3"],
            adapter=stage3_adapter,
            attn_cfg=attn_cfg,
        )

        fusion = args["modality_fusion"]
        self.backbone = BaseBEVBackbone(fusion["base_bev_backbone"], F90_CHANNELS)
        self.shrink_conv = DownsampleConv(fusion["shrink_header"])
        out_c = int(args["outC"])
        self.cls_head = nn.Conv2d(
            out_c, int(args["anchor_number"]) * int(args["num_class"]), kernel_size=1
        )
        self.reg_head = nn.Conv2d(
            out_c, 7 * int(args["anchor_number"]), kernel_size=1
        )
        self.obj_head = nn.Conv2d(out_c, int(args["anchor_number"]), kernel_size=1)
        # Quality head: one IoU-quality logit per anchor, used only to
        # rank overlapping boxes inside NMS (never replaces obj score).
        self.quality_head = nn.Conv2d(
            out_c, int(args["anchor_number"]), kernel_size=1
        )
        nn.init.normal_(self.quality_head.weight, std=0.01)
        nn.init.constant_(self.quality_head.bias, 0.0)
        # Low foreground prior for fresh detector training: the obj head
        # starts predicting p ~ obj_prior_prob instead of p ~ 0.5, so the
        # focal loss does not start from a large negative- dominated
        # logit regime. The P1 frontend checkpoint (verified statically)
        # contains no obj_head keys, so this init is never overwritten by
        # _load_and_freeze_frontend; resuming a full net_epoch*.pth later
        # naturally overrides it.
        obj_prior_prob = float(
            args.get("obj_prior_prob", 0.01)
        )
        if not 0.0 < obj_prior_prob < 1.0:
            raise ValueError(
                f"obj_prior_prob must be in (0,1), got {obj_prior_prob}"
            )
        obj_bias = math.log(
            obj_prior_prob
            / (1.0 - obj_prior_prob)
        )
        nn.init.constant_(
            self.obj_head.bias,
            obj_bias,
        )
        if self.vehicle_p1_v45:
            pretrained = args.get("pretrained_v45") or args["pretrained"]
            print("[V45] Vehicle P1 stride-8 45x80; RSU/Drone stay 90x160")
        else:
            pretrained = args["pretrained"]
        self._load_and_freeze_frontend(str(pretrained))

    _FROZEN_MODULES = (
        "frontend",
        "highres",
        "heatmap_heads",
        "depth_heads",
        "drone_height_embed",
        "drone_delta_head",
        "depth_moments",
    )

    def _load_and_freeze_frontend(self, path: str) -> None:
        """Load the P1 image stack from ``path`` and freeze it.

        Stage-1 and later stay randomly initialized and trainable.
        Frozen modules stay in ``eval`` so BN running stats do not update.
        """
        ckpt = torch.load(path, map_location="cpu")
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise KeyError(f"{path} has no model_state_dict")
        state = ckpt["model_state_dict"]
        first = next(iter(state))
        if first.startswith("module."):
            state = {key[7:]: value for key, value in state.items()}
        current = self.state_dict()
        unexpected = [key for key in state if key not in current]
        if unexpected:
            raise KeyError(f"ckpt keys not in model: {unexpected[:8]}")
        frozen = self._frozen_module_names()
        missing = [
            key
            for key in current
            if key.split(".")[0] in frozen and key not in state
        ]
        if missing:
            raise KeyError(f"frozen keys missing from ckpt: {missing[:8]}")
        for key, value in state.items():
            if tuple(current[key].shape) != tuple(value.shape):
                raise ValueError(
                    f"{key} ckpt {tuple(value.shape)} vs model {tuple(current[key].shape)}"
                )
        self.load_state_dict(state, strict=False)
        for name in self._frozen_module_names():
            module = getattr(self, name)
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
        print(f"loaded frozen frontend from {path} n={len(state)}")

    def _frozen_module_names(self) -> Tuple[str, ...]:
        """P1 modules that stay frozen. ``r2_downsample`` only exists for V45."""
        names = list(self._FROZEN_MODULES)
        if self.vehicle_p1_v45:
            names.append("r2_downsample")
        return tuple(names)

    def train(self, mode: bool = True) -> "Airv2xGaussian0822":
        """Train Stage-1+; keep the P1 image stack frozen in eval (BN included)."""
        super().train(mode)
        for name in self._frozen_module_names():
            getattr(self, name).eval()
        return self

    def _encode_agent(self, agent_type: str, data_dict: Dict[str, Any]) -> None:
        """Write F90, heatmap, and depth tensors onto ``data_dict[agent]``."""
        payload = data_dict[agent_type]
        imgs = payload["batch_merged_cam_inputs"]["imgs"]
        r2, f45 = self.frontend.extract_backbone_features(agent_type, imgs)
        if agent_type == "vehicle" and self.vehicle_p1_v45:
            r2 = self.r2_downsample(r2)
        f90 = self.highres[agent_type](r2, f45)
        heatmap_logits = self.heatmap_heads[agent_type](f90)
        payload["f90"] = f90
        payload["heatmap_logits"] = heatmap_logits
        if agent_type == "drone":
            cam_inputs = payload["batch_merged_cam_inputs"]
            height = flatten_camera_world_z(cam_inputs["camera_world_z"], imgs)
            height = height.to(device=f90.device, dtype=f90.dtype)
            height_embed = self.drone_height_embed(
                height, (int(f90.shape[2]), int(f90.shape[3]))
            ).to(dtype=f90.dtype)
            delta_pred = self.drone_delta_head(f90, height_embed)
            payload["delta_pred"] = delta_pred
            payload["depth_z_mean"] = height[:, None, None] + delta_pred
            payload["camera_world_z"] = height
            return
        depth_logits = self.depth_heads[agent_type](f90)
        depth_z_mean, depth_z_var = self.depth_moments[agent_type](depth_logits)
        payload["depth_logits"] = depth_logits
        payload["depth_z_mean"] = depth_z_mean
        payload["depth_z_var"] = depth_z_var

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Detection heads on the Stage-3 BEV map.

        Intermediate F90 / heatmap / depth / Gaussians / BEV features are
        written onto ``data_dict``.

        Returns:
            ``psm`` / ``rm`` / ``obj``, same keys as V2X-ViT.
        """
        for agent_type in present_camera_agents(data_dict):
            self._encode_agent(agent_type, data_dict)

        gaussians = self.initializer.forward(data_dict)
        for agent, gs in gaussians.items():
            payload = data_dict[agent]
            gaussians[agent] = self.intra_view(
                gs, payload["f90"], payload[CAMERA_GEOMETRY_KEY], agent=agent
            )
        data_dict["gaussians"] = gaussians

        sources = {
            agent: {
                "features": data_dict[agent]["f90"],
                "geometry": data_dict[agent][CAMERA_GEOMETRY_KEY],
            }
            for agent in gaussians
        }
        gaussians = self.cross_agent(gaussians, sources)
        data_dict["gaussians"] = gaussians

        _, bev = self.stage3(gaussians)
        data_dict["spatial_features"] = bev
        data_dict = self.backbone(data_dict)
        fused = self.shrink_conv(data_dict["spatial_features_2d"])
        return {
            "psm": self.cls_head(fused),
            "rm": self.reg_head(fused),
            "obj": self.obj_head(fused),
            "quality": self.quality_head(fused),
        }
