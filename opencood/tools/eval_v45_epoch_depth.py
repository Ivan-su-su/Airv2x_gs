# -*- coding: utf-8 -*-
"""Compare Vehicle V45 P1 depth on held-out TEST across complete epochs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.gaussian_modules_0822.heatmap.target import binary_objectness_target
from opencood.models.gaussian_modules_0822.image_frontend import present_camera_agents
from opencood.models.gaussian_modules_0822.lss.target import (
    depth_valid_mask,
    extract_camera_z_gt,
)
from opencood.tools import train_utils
from opencood.tools.eval_vehicle_v45_vs_v90 import (
    AGENT,
    FAR_Z_M,
    STRIDE_V45,
    DepthAcc,
    HeatmapAcc,
    load_mix_keys,
    load_v45_state,
    sample_plan_heldout,
)
from opencood.tools.train_gaussian_p1 import _unwrap_model
from opencood.tools.vis_test_heatmap_recall import paint_agent_box_maps


def parse_args() -> argparse.Namespace:
    """CLI for multi-epoch V45 depth compare."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default=(
            "/mnt/home/suyi/AirV2X-Perception_gaussian/opencood/logs/"
            "vehicle_v45_p1/v45_p1_2gpu_2026_09_09_23_25_42"
        ),
    )
    parser.add_argument("--epochs", default="1,4,9,13")
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--frames_per_scene", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    """Load TEST once, score each complete epoch on depth + heatmap."""
    opt = parse_args()
    model_dir = Path(opt.model_dir)
    epochs = [int(x) for x in str(opt.epochs).split(",") if x.strip()]
    hypes = yaml_utils.load_yaml(str(model_dir / "config.yaml"), None)
    hypes["validate_dir"] = hypes["test_dir"]
    hypes["train"] = False
    hypes.pop("extra_train_scenes", None)

    device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    print(f"Building TEST dataset from {hypes['validate_dir']}", flush=True)
    dataset = build_dataset(hypes, visualize=False, train=False)
    mix_keys = load_mix_keys(model_dir / "extra_train_mix.json")
    plan = sample_plan_heldout(dataset, opt.frames_per_scene, mix_keys)
    print(f"sampled {len(plan)} held-out TEST frames", flush=True)

    model = train_utils.create_model(hypes)
    model.to(device)
    model.eval()
    core = _unwrap_model(model)
    d_min = float(core.encoder.d_min)
    d_max = float(core.encoder.d_max)

    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for epoch in epochs:
            ckpt = model_dir / f"net_epoch{epoch}.pth"
            if not ckpt.is_file():
                print(f"skip missing {ckpt}", flush=True)
                continue
            load_v45_state(model, ckpt)
            model.eval()
            heat = HeatmapAcc()
            depth = {
                "valid": DepthAcc(),
                "fg": DepthAcc(),
                "pred_fg": DepthAcc(),
                "far": DepthAcc(),
            }
            for sample_i, (scene, idx, timestamp, mixed) in enumerate(plan):
                sample = dataset[idx]
                batch = dataset.collate_batch_test([sample])
                batch = train_utils.to_device(batch, device)
                ego = batch["ego"]
                if AGENT not in present_camera_agents(ego):
                    continue
                cam = ego[AGENT]["batch_merged_cam_inputs"]
                imgs = cam["imgs"]
                pred = core.forward_features(imgs)
                p_fg = torch.softmax(pred["heatmap_logits"], dim=1)[:, 1]
                z_mean = pred["depth_z_mean"]
                maps = paint_agent_box_maps(ego, AGENT).to(device=p_fg.device)
                if maps.numel() == 0 or maps.shape[0] != p_fg.shape[0]:
                    continue
                gt45 = binary_objectness_target(maps, tau=1, block=STRIDE_V45).gt(0)
                camera_z = extract_camera_z_gt(imgs, spatial_stride=STRIDE_V45)
                valid = depth_valid_mask(camera_z, d_min, d_max)
                heat.add(p_fg, gt45)
                depth["valid"].add(z_mean, camera_z, valid)
                depth["fg"].add(z_mean, camera_z, valid & gt45)
                depth["pred_fg"].add(z_mean, camera_z, valid & p_fg.ge(0.3))
                depth["far"].add(z_mean, camera_z, valid & (camera_z >= FAR_Z_M))
                if sample_i == 0 or (sample_i + 1) == len(plan):
                    print(
                        f"  epoch{epoch} [{sample_i + 1}/{len(plan)}] {scene} ts={timestamp}",
                        flush=True,
                    )
            packed = {
                "epoch": epoch,
                "ckpt": str(ckpt),
                "heatmap": heat.pack(),
                "depth": {key: acc.pack() for key, acc in depth.items()},
            }
            rows.append(packed)
            d = packed["depth"]
            h = packed["heatmap"]
            print(
                f"EPOCH {epoch:2d}  mae_valid={d['valid']['mae']}  "
                f"mae_fg={d['fg']['mae']}  mae_pred_fg={d['pred_fg']['mae']}  "
                f"mae_far={d['far']['mae']}  rec@0.3={h.get('recall@0.3')}  "
                f"mean_p_fg_gt={h.get('mean_p_fg_gt')}",
                flush=True,
            )

    def score(row: Dict[str, Any]) -> float:
        mae = row["depth"]["valid"]["mae"]
        return float("inf") if mae is None else float(mae)

    best = min(rows, key=score) if rows else None
    report = {
        "plan_n": len(plan),
        "best_epoch": None if best is None else best["epoch"],
        "criterion": "lowest TEST mae_valid (held-out 5 frames/scene)",
        "rows": rows,
    }
    out = model_dir / "epoch_depth_compare.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    if best is not None:
        print(f"BEST epoch={best['epoch']} mae_valid={best['depth']['valid']['mae']}")


if __name__ == "__main__":
    main()
