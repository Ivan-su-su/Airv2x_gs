"""Replace cam encoders with frozen-P1 splat. Lidar is not used by Stage-0 yaml."""

from __future__ import annotations

from typing import Any, Dict

from torch import nn

from opencood.models.gaussian_modules_0822.p1_layout import F90_CHANNELS
from opencood.models.lss_pretrain_modules.frozen_p1 import FrozenP1
from opencood.models.lss_pretrain_modules.p1_splat_encoder import P1SplatEncoder


class P1CamMixin:
    """Overrides ``Airv2xBase.init_encoders`` for frozen-P1 camera lift."""

    def init_encoders(self, args: Dict[str, Any]) -> None:
        self.p1_core = FrozenP1(args)
        encode = self.p1_core.encode
        self.veh_models = nn.ModuleList()
        self.rsu_models = nn.ModuleList()
        self.drone_models = nn.ModuleList()
        for agent, bucket in (
            ("vehicle", self.veh_models),
            ("rsu", self.rsu_models),
            ("drone", self.drone_models),
        ):
            if agent not in self.collaborators:
                continue
            for modality in args[agent]["modalities"]:
                if modality != "cam":
                    raise NotImplementedError(
                        f"P1 cam frontend does not wrap modality={modality}"
                    )
                cam_args = dict(args[agent]["cam"])
                cam_args["img_features"] = int(F90_CHANNELS)
                if agent == "vehicle":
                    cam_args["img_downsample"] = (
                        8 if self.p1_core.r2_downsample is not None else 4
                    )
                else:
                    cam_args["img_downsample"] = 4
                if agent == "drone":
                    cam_args["lift"] = str(cam_args.get("lift") or "delta")
                print(
                    f"[stage0-p1] {agent} splat frustum downsample="
                    f"{cam_args['img_downsample']} img_features="
                    f"{cam_args['img_features']} lift={cam_args.get('lift', 'categorical')}"
                )
                bucket.append(P1SplatEncoder(cam_args, agent, encode))
