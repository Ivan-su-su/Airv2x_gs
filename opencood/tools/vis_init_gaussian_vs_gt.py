"""Overlay pre-Stage1 Gaussians on ego GT boxes (BEV).

Compares initializer output in ego (after agent→ego) against GT, and
optionally the same seeds left in the agent frame as a transform check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml import yaml_utils
from opencood.models.airv2x_gaussian_0822 import (
    CAMERA_GEOMETRY_KEY,
    present_camera_agents,
)
from opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet
from opencood.tools import train_utils
from opencood.tools.vis_p1_scene_sample import valid_boxes
from opencood.utils.box_utils import boxes_to_corners_3d
from opencood.visualization.simple_plot3d import canvas_bev

AGENT_COLORS = {
    "vehicle": (0, 180, 255),
    "rsu": (255, 160, 0),
    "drone": (255, 60, 180),
}
MAX_POINTS = 8000


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def _points_in_bev_boxes(
    xy: np.ndarray, boxes_hwl: np.ndarray, pad_m: float = 0.5
) -> np.ndarray:
    """Boolean mask: BEV point falls in any GT box (xy rectangle + pad)."""
    n = int(xy.shape[0])
    if n == 0 or boxes_hwl.shape[0] == 0:
        return np.zeros((n,), dtype=bool)
    # hwl → l, w for BEV half-extents
    cx = boxes_hwl[:, 0]
    cy = boxes_hwl[:, 1]
    half_l = 0.5 * boxes_hwl[:, 5] + pad_m
    half_w = 0.5 * boxes_hwl[:, 4] + pad_m
    yaw = boxes_hwl[:, 6]
    c = np.cos(-yaw)
    s = np.sin(-yaw)
    dx = xy[:, 0:1] - cx[None, :]
    dy = xy[:, 1:2] - cy[None, :]
    lx = c[None, :] * dx - s[None, :] * dy
    ly = s[None, :] * dx + c[None, :] * dy
    return ((np.abs(lx) <= half_l[None, :]) & (np.abs(ly) <= half_w[None, :])).any(
        axis=1
    )


def _box_hit_by_points(
    xy: np.ndarray, boxes_hwl: np.ndarray, pad_m: float = 0.5
) -> np.ndarray:
    """Boolean mask: GT box contains at least one BEV point."""
    m = int(boxes_hwl.shape[0])
    if m == 0 or xy.shape[0] == 0:
        return np.zeros((m,), dtype=bool)
    cx = boxes_hwl[:, 0:1]
    cy = boxes_hwl[:, 1:2]
    half_l = 0.5 * boxes_hwl[:, 5:6] + pad_m
    half_w = 0.5 * boxes_hwl[:, 4:5] + pad_m
    yaw = boxes_hwl[:, 6:7]
    c = np.cos(-yaw)
    s = np.sin(-yaw)
    dx = xy[None, :, 0] - cx
    dy = xy[None, :, 1] - cy
    lx = c * dx - s * dy
    ly = s * dx + c * dy
    inside = (np.abs(lx) <= half_l) & (np.abs(ly) <= half_w)
    return inside.any(axis=1)


def _subsample(xy: np.ndarray, max_n: int) -> np.ndarray:
    if xy.shape[0] <= max_n:
        return xy
    rng = np.random.default_rng(0)
    idx = rng.choice(xy.shape[0], size=max_n, replace=False)
    return xy[idx]


def _draw_panel(
    ax_img: Any,
    xy_by_agent: Dict[str, np.ndarray],
    gt_corners: np.ndarray,
    pc_range: List[float],
    title: str,
) -> None:
    canvas = canvas_bev.Canvas_BEV_heading_right(
        canvas_shape=(
            int((pc_range[4] - pc_range[1]) * 10),
            int((pc_range[3] - pc_range[0]) * 10),
        ),
        canvas_x_range=(pc_range[0], pc_range[3]),
        canvas_y_range=(pc_range[1], pc_range[4]),
        left_hand=True,
        canvas_bg_color=(20, 20, 20),
    )
    for agent, xy in xy_by_agent.items():
        if xy.shape[0] == 0:
            continue
        pts = np.concatenate([xy, np.zeros((xy.shape[0], 1))], axis=1)
        canvas_xy, valid = canvas.get_canvas_coords(pts)
        canvas.draw_canvas_points(
            canvas_xy[valid], colors=AGENT_COLORS[agent], radius=1
        )
    if gt_corners.shape[0] > 0:
        canvas.draw_boxes(gt_corners, colors=(0, 255, 0), box_line_thickness=2)
    ax_img.imshow(canvas.canvas)
    ax_img.set_title(title, fontsize=10)
    ax_img.axis("off")


def _collect_agent_frame(
    model: torch.nn.Module, data: Dict[str, Any]
) -> Dict[str, GaussianSet]:
    """Initializer without agent→ego (means stay in each agent's lidar)."""
    out: Dict[str, GaussianSet] = {}
    for agent in present_camera_agents(data):
        payload = data[agent]
        geometry = payload[CAMERA_GEOMETRY_KEY]
        if agent == "drone":
            depth_logits, depth_z = None, payload["depth_z_mean"]
        else:
            depth_logits, depth_z = payload["depth_logits"], None
        out[agent] = model.initializer.forward_agent(
            agent=agent,
            heatmap_logits=payload["heatmap_logits"],
            features=payload["f90"],
            geometry=geometry,
            depth_logits=depth_logits,
            depth_z=depth_z,
            agent_to_ego=None,
        )
    return out


def analyze_frame(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    pc_range: List[float],
    out_png: Path,
) -> Dict[str, Any]:
    data = batch["ego"]
    for agent in present_camera_agents(data):
        model._encode_agent(agent, data)
    ego_sets = model.initializer.forward(data)
    agent_sets = _collect_agent_frame(model, data)
    boxes, _ = valid_boxes(data)
    corners = boxes_to_corners_3d(boxes, "hwl")
    if torch.is_tensor(corners):
        corners = corners.detach().cpu().numpy()

    xy_ego: Dict[str, np.ndarray] = {}
    xy_agent: Dict[str, np.ndarray] = {}
    all_ego = []
    rec: Dict[str, Any] = {
        "n_gt": int(boxes.shape[0]),
        "frame_tag": {a: gs.frame for a, gs in ego_sets.items()},
        "agents": {},
    }
    for agent in ("vehicle", "rsu", "drone"):
        gs_e = ego_sets.get(agent)
        gs_a = agent_sets.get(agent)
        n_e = int(gs_e.n_gaussians) if gs_e is not None else 0
        n_a = int(gs_a.n_gaussians) if gs_a is not None else 0
        xy_e = _to_numpy(gs_e.mean[:, :2]) if n_e else np.zeros((0, 2))
        xy_a = _to_numpy(gs_a.mean[:, :2]) if n_a else np.zeros((0, 2))
        xy_ego[agent] = xy_e
        xy_agent[agent] = xy_a
        if n_e:
            all_ego.append(xy_e)
        in_box = _points_in_bev_boxes(xy_e, boxes) if n_e else np.zeros((0,), dtype=bool)
        rec["agents"][agent] = {
            "n_ego": n_e,
            "n_agent_frame": n_a,
            "ego_xy_mean": xy_e.mean(axis=0).tolist() if n_e else None,
            "agent_xy_mean": xy_a.mean(axis=0).tolist() if n_a else None,
            "frac_in_gt_box": float(in_box.mean()) if n_e else 0.0,
            "n_in_gt_box": int(in_box.sum()),
        }
    stacked = np.concatenate(all_ego, axis=0) if all_ego else np.zeros((0, 2))
    hits = _box_hit_by_points(stacked, boxes) if boxes.shape[0] else np.zeros((0,), bool)
    rec["frac_gt_with_gaussian"] = float(hits.mean()) if boxes.shape[0] else 0.0
    rec["n_gt_with_gaussian"] = int(hits.sum())
    rec["n_gaussian_total"] = int(stacked.shape[0])
    rec["frac_gaussian_in_gt"] = (
        float(_points_in_bev_boxes(stacked, boxes).mean()) if stacked.shape[0] else 0.0
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    _draw_panel(
        axes[0],
        {a: _subsample(xy, MAX_POINTS) for a, xy in xy_ego.items()},
        corners,
        pc_range,
        "Initializer after agent→ego (should overlap GT)",
    )
    _draw_panel(
        axes[1],
        {a: _subsample(xy, MAX_POINTS) for a, xy in xy_agent.items()},
        corners,
        pc_range,
        "Same seeds in agent lidar (no to_ego; should mismatch unless ego)",
    )
    fig.suptitle(
        f"GT={rec['n_gt']}  Gaussians={rec['n_gaussian_total']}  "
        f"in-box={rec['frac_gaussian_in_gt']:.3f}  "
        f"GT hit={rec['frac_gt_with_gaussian']:.3f}",
        fontsize=11,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=120)
    plt.close(fig)
    rec["png"] = str(out_png)
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num", type=int, default=8)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
    )
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    yaml_path = str(
        ROOT / "opencood/hypes_yaml/airv2x/camera/det/airv2x_gaussian_0822.yaml"
    )
    hypes = yaml_utils.load_yaml(
        yaml_path, argparse.Namespace(model_dir="", hypes_yaml=yaml_path)
    )
    hypes["validate_dir"] = hypes["test_dir"]
    dataset = build_dataset(dict(hypes), visualize=False, train=False)
    model = train_utils.create_model(hypes).to(device)
    model.eval()
    pc_range = [float(v) for v in hypes["postprocess"]["anchor_args"]["cav_lidar_range"]]
    out_dir = Path(args.out_dir) if args.out_dir else (
        ROOT
        / "opencood/logs/airv2x_gaussian_0822_camera/v45_scale_asym_2026_09_11_15_54_05"
        / "vis_init_gaussian_gt"
    )
    rows: List[Dict[str, Any]] = []
    n = min(int(args.num), len(dataset) - int(args.index))
    for k in range(n):
        idx = int(args.index) + k
        batch = train_utils.to_device(
            dataset.collate_batch_test([dataset[idx]]), device
        )
        batch["ego"]["epoch"] = 0
        png = out_dir / f"init_vs_gt_{idx:05d}.png"
        with torch.no_grad():
            rec = analyze_frame(model, batch, pc_range, png)
        rec["index"] = idx
        rows.append(rec)
        print(
            f"idx={idx} GT={rec['n_gt']} N={rec['n_gaussian_total']} "
            f"in_box={rec['frac_gaussian_in_gt']:.3f} "
            f"gt_hit={rec['frac_gt_with_gaussian']:.3f} "
            f"agents={ {a: v['n_ego'] for a, v in rec['agents'].items()} }"
        )
    summary = {
        "n_frames": len(rows),
        "mean_frac_gaussian_in_gt": float(
            np.mean([r["frac_gaussian_in_gt"] for r in rows])
        ),
        "mean_frac_gt_with_gaussian": float(
            np.mean([r["frac_gt_with_gaussian"] for r in rows])
        ),
        "frames": rows,
    }
    out_json = out_dir / "init_vs_gt_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print("wrote", out_json)
    print(
        "mean in-box",
        summary["mean_frac_gaussian_in_gt"],
        "mean GT hit",
        summary["mean_frac_gt_with_gaussian"],
    )


if __name__ == "__main__":
    main()
