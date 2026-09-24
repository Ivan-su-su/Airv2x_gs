"""Mixin that replaces camera encoders with frozen Griffin-P1 LSS splat."""

from __future__ import annotations

from typing import Any, Dict

from torch import nn

from opencood.models.gaussian_modules_0822.p1_layout import F90_CHANNELS
from opencood.models.lss_pretrain_modules.frozen_p1 import FrozenP1
from opencood.models.lss_pretrain_modules.p1_splat_encoder import P1SplatEncoder


class P1CamMixin:
    """Build frozen-P1 camera encoders for vehicle/drone Griffin baselines."""

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
                        f"P1 baseline supports camera only, got {agent}:{modality}"
                    )
                bucket.append(
                    make_p1_splat_encoder(args, agent, encode, self.p1_core)
                )


def make_p1_splat_encoder(args, agent, encode, p1_core):
    cam_args = dict(args[agent]["cam"])
    cam_args["img_features"] = int(F90_CHANNELS)
    cam_args["img_downsample"] = (
        8
        if agent == "vehicle" and p1_core.r2_downsample is not None
        else 4
    )
    if agent == "drone":
        cam_args["lift"] = str(cam_args.get("lift") or "ideal_ground")
    else:
        cam_args["lift"] = str(cam_args.get("lift") or "categorical")

    print(
        f"[griffin-p1-lss] {agent}: downsample={cam_args['img_downsample']} "
        f"F90={cam_args['img_features']} lift={cam_args['lift']} "
        f"sample={cam_args.get('feature_sample_mode', 'lss_endpoint')}"
    )
    return P1SplatEncoder(cam_args, agent, encode)
