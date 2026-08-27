#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import box_utils


@dataclass(frozen=True)
class BevSpec:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    voxel_x: float
    voxel_y: float

    @property
    def width(self) -> int:
        return int(round((self.x_max - self.x_min) / self.voxel_x))

    @property
    def height(self) -> int:
        return int(round((self.y_max - self.y_min) / self.voxel_y))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _get_ego_dict(sample: Mapping[str, Any]) -> Mapping[str, Any]:
    if "ego" in sample and isinstance(sample["ego"], Mapping):
        return sample["ego"]
    return sample


def _extract_valid_boxes(sample: Mapping[str, Any]) -> np.ndarray:
    ego = _get_ego_dict(sample)
    boxes = np.asarray(ego["object_bbx_center"])
    mask = np.asarray(ego["object_bbx_mask"]).astype(np.bool_)
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise ValueError(f"object_bbx_center expected [N,7], got {boxes.shape}")
    if mask.ndim != 1 or mask.shape[0] != boxes.shape[0]:
        raise ValueError(f"object_bbx_mask expected [N], got {mask.shape} for boxes {boxes.shape}")
    return boxes[mask]


def _extract_class_ids(sample: Mapping[str, Any], valid_count: int) -> Optional[np.ndarray]:
    ego = _get_ego_dict(sample)
    class_ids = ego.get("class_ids")
    if class_ids is None:
        return None
    class_ids_arr = np.asarray(class_ids)
    if class_ids_arr.ndim != 1:
        return None
    if class_ids_arr.shape[0] < valid_count:
        return None
    return class_ids_arr[:valid_count]


def _polygon_to_grid_xy(
    polygon_xy: np.ndarray,
    bev: BevSpec,
    stride: int = 1,
) -> np.ndarray:
    """Convert polygon corners in meters to pixel indices in BEV grid.

    Args:
        polygon_xy: Array of shape [4, 2] in meters (x, y).
        bev: BEV spec (meters + voxel size).
        stride: Feature stride (1 for voxel grid, 2 for stride-2 feature map).

    Returns:
        Polygon points for cv2.fillPoly, shape [1, 4, 2] int32, in (x_idx, y_idx).
    """
    # opencood's `boxes_to_corners2d` may return [4, 3] (x, y, z). We only use x/y.
    if polygon_xy.shape == (4, 3):
        polygon_xy = polygon_xy[:, :2]
    if polygon_xy.shape != (4, 2):
        raise ValueError(f"polygon_xy expected [4,2] or [4,3], got {polygon_xy.shape}")
    voxel_x = bev.voxel_x * float(stride)
    voxel_y = bev.voxel_y * float(stride)
    x_idx = (polygon_xy[:, 0] - bev.x_min) / voxel_x
    y_idx = (polygon_xy[:, 1] - bev.y_min) / voxel_y
    pts = np.stack([x_idx, y_idx], axis=1)
    pts = np.round(pts).astype(np.int32)
    return pts.reshape(1, 4, 2)


def _occupied_cell_count(
    polygon_xy: np.ndarray,
    bev: BevSpec,
    stride: int = 1,
) -> int:
    """Rasterize rotated box polygon onto BEV grid and count occupied cells."""
    w = int(round(bev.width / float(stride)))
    h = int(round(bev.height / float(stride)))
    pts = _polygon_to_grid_xy(polygon_xy, bev=bev, stride=stride)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, pts, 1)
    return int(mask.sum())


def _render_example(
    polygon_xy: np.ndarray,
    bev: BevSpec,
    stride: int,
    title: str,
    out_path: Path,
) -> None:
    w = int(round(bev.width / float(stride)))
    h = int(round(bev.height / float(stride)))
    pts = _polygon_to_grid_xy(polygon_xy, bev=bev, stride=stride)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.polylines(canvas, pts, isClosed=True, color=(0, 255, 0), thickness=1)
    cv2.fillPoly(canvas, pts, color=(0, 120, 0))
    plt.figure(figsize=(8, 4))
    plt.imshow(canvas[..., ::-1])
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _safe_class_name(class_names: Sequence[str], class_id: int) -> str:
    if class_id < 0:
        return "unknown"
    if class_id >= len(class_names):
        return f"cls_{class_id}"
    return class_names[int(class_id)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hypes_yaml",
        "-y",
        type=str,
        required=True,
        help="Path to a training yaml (same as opencood/tools/train.py).",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Use train split root_dir (default: validate split).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=300,
        help="Max dataset samples to scan.",
    )
    parser.add_argument(
        "--max_boxes",
        type=int,
        default=50000,
        help="Max GT boxes to process (early stop).",
    )
    parser.add_argument(
        "--vis_examples",
        type=int,
        default=30,
        help="How many example polygons to render.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="footprint_stats_out",
        help="Output directory (relative to this script).",
    )
    args = parser.parse_args()

    cfg_path = Path(args.hypes_yaml).expanduser().resolve()
    cfg = yaml_utils.load_yaml(str(cfg_path))
    dataset = build_dataset(cfg, visualize=False, train=bool(args.train))

    pc_range = np.asarray(cfg["POINT_CLOUD_RANGE"], dtype=np.float32)
    voxel_size = np.asarray(cfg["VOXEL_SIZE"], dtype=np.float32)
    bev = BevSpec(
        x_min=float(pc_range[0]),
        y_min=float(pc_range[1]),
        x_max=float(pc_range[3]),
        y_max=float(pc_range[4]),
        voxel_x=float(voxel_size[0]),
        voxel_y=float(voxel_size[1]),
    )

    order = str(cfg.get("postprocess", {}).get("order", "hwl"))
    class_names = cfg.get("CLASS_NAMES") or []
    class_names = [str(x) for x in class_names]

    script_dir = Path(__file__).resolve().parent
    out_dir = (script_dir / args.out_dir).resolve()
    _ensure_dir(out_dir)
    _ensure_dir(out_dir / "examples_stride1")
    _ensure_dir(out_dir / "examples_stride2")

    records: List[Dict[str, Any]] = []
    example_written = 0
    total_boxes = 0

    max_samples = min(int(args.max_samples), len(dataset))
    for sample_idx in range(max_samples):
        sample = dataset[sample_idx]
        boxes = _extract_valid_boxes(sample)
        class_ids = _extract_class_ids(sample, valid_count=boxes.shape[0])

        if boxes.shape[0] == 0:
            continue

        # (N, 4, 2) in meters (x,y)
        corners2d = box_utils.boxes_to_corners2d(boxes, order=order)

        for i in range(corners2d.shape[0]):
            polygon = corners2d[i].astype(np.float32)
            occ1 = _occupied_cell_count(polygon, bev=bev, stride=1)
            occ2 = _occupied_cell_count(polygon, bev=bev, stride=2)
            cid = int(class_ids[i]) if class_ids is not None else -1
            cname = _safe_class_name(class_names, cid)

            records.append(
                {
                    "sample_idx": int(sample_idx),
                    "box_idx": int(i),
                    "class_id": int(cid),
                    "class_name": cname,
                    "occupied_cells_stride1": int(occ1),
                    "occupied_cells_stride2": int(occ2),
                }
            )

            if example_written < int(args.vis_examples):
                _render_example(
                    polygon_xy=polygon,
                    bev=bev,
                    stride=1,
                    title=f"s{sample_idx}_b{i} {cname} stride1 cells={occ1}",
                    out_path=out_dir / "examples_stride1" / f"s{sample_idx}_b{i}.png",
                )
                _render_example(
                    polygon_xy=polygon,
                    bev=bev,
                    stride=2,
                    title=f"s{sample_idx}_b{i} {cname} stride2 cells={occ2}",
                    out_path=out_dir / "examples_stride2" / f"s{sample_idx}_b{i}.png",
                )
                example_written += 1

            total_boxes += 1
            if total_boxes >= int(args.max_boxes):
                break
        if total_boxes >= int(args.max_boxes):
            break

    if not records:
        raise RuntimeError("No GT boxes found. Check dataset path/config and split selection.")

    # Save raw records.
    with (out_dir / "records.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    with (out_dir / "records.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(records)

    # Aggregate stats.
    occ1_all = np.asarray([r["occupied_cells_stride1"] for r in records], dtype=np.int64)
    occ2_all = np.asarray([r["occupied_cells_stride2"] for r in records], dtype=np.int64)
    cname_all = [r["class_name"] for r in records]

    def summarize(x: np.ndarray) -> Dict[str, float]:
        x_sorted = np.sort(x)
        def q(p: float) -> float:
            return float(np.percentile(x_sorted, p))
        return {
            "count": float(x.shape[0]),
            "mean": float(x.mean()),
            "std": float(x.std()),
            "min": float(x.min()),
            "p10": q(10),
            "p25": q(25),
            "p50": q(50),
            "p75": q(75),
            "p90": q(90),
            "max": float(x.max()),
        }

    summary: Dict[str, Any] = {
        "config": {
            "hypes_yaml": str(cfg_path),
            "split": "train" if args.train else "validate",
            "order": order,
            "point_cloud_range": pc_range.tolist(),
            "voxel_size": voxel_size.tolist(),
            "bev_width": bev.width,
            "bev_height": bev.height,
            "stride2_width": int(round(bev.width / 2.0)),
            "stride2_height": int(round(bev.height / 2.0)),
        },
        "overall": {
            "stride1": summarize(occ1_all),
            "stride2": summarize(occ2_all),
        },
        "by_class": {},
    }

    unique_classes = sorted(set(cname_all))
    for cname in unique_classes:
        idxs = [i for i, n in enumerate(cname_all) if n == cname]
        s1 = occ1_all[idxs]
        s2 = occ2_all[idxs]
        summary["by_class"][cname] = {
            "stride1": summarize(s1),
            "stride2": summarize(s2),
        }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Plots.
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(occ1_all, bins=50, color="#2f6f8e")
    plt.title("Occupied cells (stride=1)")
    plt.xlabel("cells")
    plt.ylabel("count")

    plt.subplot(1, 2, 2)
    plt.hist(occ2_all, bins=50, color="#8e2f4f")
    plt.title("Occupied cells (stride=2)")
    plt.xlabel("cells")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_overall.png", dpi=200)
    plt.close()

    # Per-class boxplot (top-K by frequency).
    class_counts = {c: cname_all.count(c) for c in unique_classes}
    top_classes = [c for c, _ in sorted(class_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]]
    data_stride2 = [
        occ2_all[[i for i, n in enumerate(cname_all) if n == c]]
        for c in top_classes
    ]
    plt.figure(figsize=(max(8, len(top_classes) * 1.2), 4))
    plt.boxplot(data_stride2, labels=top_classes, showfliers=False)
    plt.title("Stride=2 occupied cells (top-10 classes)")
    plt.ylabel("cells")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "boxplot_stride2_top10.png", dpi=200)
    plt.close()

    print(f"[OK] Wrote outputs to: {out_dir}")
    print(f"[OK] Processed samples: {max_samples}, boxes: {len(records)}")
    print(f"[OK] Overall stride2 p50={summary['overall']['stride2']['p50']:.1f}, p90={summary['overall']['stride2']['p90']:.1f}")


if __name__ == "__main__":
    main()

