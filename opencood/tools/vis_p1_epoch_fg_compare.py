# -*- coding: utf-8 -*-
"""Compare epoch 1/9/16 predicted FG vs raw RGB, grouped by scene.

Val has a single folder, so TEST supplies the weather/scene diversity.
Val is included as one extra scene. No training/production code is changed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.gaussian_modules_0822.heatmap.metrics import (
    PRIMARY_OBJECTNESS_THRESHOLD,
)
from opencood.models.gaussian_modules_0822.heatmap.target import binary_objectness_target
from opencood.models.gaussian_modules_0822.image_frontend import present_camera_agents
from opencood.tools import train_utils
from opencood.tools.eval_gaussian_p1 import load_epoch_checkpoint
from opencood.tools.train_gaussian_p1 import _unwrap_model
from opencood.tools.vis_p1_scene_sample import (
    denorm_rgb,
    flatten_imgs,
    overlay_seg,
    scene_from_path,
)
from opencood.tools.vis_test_heatmap_recall import paint_agent_box_maps
from opencood.utils.scenario_utils import scenarios_params

EPOCHS = (1, 9, 16)
AGENTS = ("vehicle", "rsu", "drone")
FRAMES_PER_SCENE = 5
SAMPLE_SEED = 42
TAU = float(PRIMARY_OBJECTNESS_THRESHOLD)
OUT = Path("/mnt/home/suyi/visualization/concat128_p1_e1e9e16_fg_vis5")
LOG_DIR = Path(
    "/mnt/home/suyi/AirV2X-Perception_gaussian/opencood/logs/"
    "airv2x_gaussian_p1_joint/concat128_2026_08_29_16_25_05"
)
SCENE_TAG = {
    "2025_05_05_21_46_39": "day_clear",
    "2025_05_05_23_26_24": "fog_dusk",
    "2025_05_06_01_09_15": "dusk",
    "2025_05_10_19_54_35": "night",
    "2025_05_10_20_47_22": "rain",
    "2025_05_06_08_16_50": "val",
}


def scene_tag(scene: str) -> str:
    """Short weather/split label for a scenario folder name."""
    if scene in SCENE_TAG:
        return SCENE_TAG[scene]
    params = scenarios_params.get(scene, {})
    bits = [k for k in ("nighttime", "foggy", "rainy", "dusk", "daytime", "clear") if params.get(k)]
    return "_".join(bits) if bits else scene[-8:]


def sample_plan(
    dataset: Any, frames_per_scene: int, seed: int
) -> List[Tuple[str, int, int]]:
    """Random ``frames_per_scene`` indices per scenario, then sort by time.

    Rank ``0..k-1`` is the k-th earliest among the random subset, so the five
    comparison sheets stay aligned across scenes.
    """
    rng = np.random.RandomState(seed)
    plan: List[Tuple[str, int, int]] = []
    for scene_i, end in enumerate(dataset.len_record):
        start = 0 if scene_i == 0 else int(dataset.len_record[scene_i - 1])
        scene_db = dataset.scenario_database[scene_i]
        first = next(iter(scene_db.values()))
        ts0 = dataset.return_timestamp_key(scene_db, 0)
        scene = scene_from_path(first[ts0]["metadata_path"])
        n = int(end) - start
        k_keep = min(int(frames_per_scene), max(n, 0))
        if k_keep <= 0:
            continue
        local = rng.choice(n, size=k_keep, replace=False)
        chosen = sorted(int(start + x) for x in local)
        for rank, idx in enumerate(chosen):
            plan.append((scene, idx, rank))
    return plan


def pick_view(p_fg: np.ndarray, gt: Optional[np.ndarray]) -> int:
    """Prefer the view with most GT-box pixels, else highest mean p_fg."""
    n_view = int(p_fg.shape[0])
    if gt is not None and int(gt.shape[0]) >= n_view and int(gt.sum()) > 0:
        scores = gt[:n_view].reshape(n_view, -1).sum(axis=1)
        if float(scores.max()) > 0:
            return int(np.argmax(scores))
    return int(np.argmax(p_fg.reshape(n_view, -1).mean(axis=1)))


def box_r90(ego: Dict[str, Any], agent: str, n_view: int) -> np.ndarray:
    """Projected 3D-box occupancy at R90, or zeros if projection is empty."""
    maps = paint_agent_box_maps(ego, agent)
    if maps.numel() == 0 or maps.shape[0] == 0:
        return np.zeros((n_view, 90, 160), dtype=np.uint8)
    occ = binary_objectness_target(maps, tau=1).cpu().numpy().astype(np.uint8)
    if occ.shape[0] < n_view:
        pad = np.zeros((n_view - occ.shape[0], *occ.shape[1:]), dtype=np.uint8)
        occ = np.concatenate([occ, pad], 0)
    return occ[:n_view]


def save_scene_panel(
    path: Path,
    rgb: np.ndarray,
    gt: np.ndarray,
    p_by_epoch: Dict[int, np.ndarray],
    title: str,
) -> None:
    """RAW | GT-box | p_fg×3  /  blank | FG@0.3×3."""
    fig, axes = plt.subplots(2, 5, figsize=(18.5, 6.6))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RAW")
    axes[0, 1].imshow(overlay_seg(rgb, gt))
    axes[0, 1].set_title("GT-box overlay")
    axes[1, 0].axis("off")
    axes[1, 1].axis("off")
    for col, epoch in enumerate(EPOCHS, start=2):
        p_fg = p_by_epoch[epoch]
        im = axes[0, col].imshow(p_fg, cmap="magma", vmin=0.0, vmax=1.0)
        axes[0, col].set_title(f"E{epoch} p_fg")
        fig.colorbar(im, ax=axes[0, col], fraction=0.046)
        axes[1, col].imshow(overlay_seg(rgb, (p_fg >= TAU).astype(np.int64)))
        axes[1, col].set_title(f"E{epoch} FG@{TAU:g}")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_compare_sheet(
    path: Path,
    rows: List[Dict[str, Any]],
    title: str = "",
) -> None:
    """One sheet: each row is scene/agent, columns RAW | E1 | E9 | E16 FG overlay."""
    n_row = len(rows)
    fig, axes = plt.subplots(n_row, 4, figsize=(16.0, 1.55 * n_row + 1.2))
    if n_row == 1:
        axes = np.expand_dims(axes, 0)
    for r, item in enumerate(rows):
        rgb = item["rgb"]
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_ylabel(
            f"{item['tag']}\n{item['agent']}\nidx={item['idx']}",
            fontsize=7,
            rotation=0,
            ha="right",
            va="center",
        )
        axes[r, 0].set_title("RAW" if r == 0 else "")
        for c, epoch in enumerate(EPOCHS, start=1):
            axes[r, c].imshow(overlay_seg(rgb, (item["p"][epoch] >= TAU).astype(np.int64)))
            if r == 0:
                axes[r, c].set_title(f"E{epoch} FG@{TAU:g}")
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        title if title else "epoch 1/9/16 predicted FG overlay  (red = p_fg>=0.3)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.06, 0.01, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def build_split_dataset(hypes: Dict[str, Any], split: str) -> Any:
    """Build train=False dataset for test or val."""
    cfg = dict(hypes)
    cfg["train"] = False
    cfg["validate_dir"] = hypes["test_dir"] if split == "test" else hypes["validate_dir"]
    print(f"Building {split} from {cfg['validate_dir']} ...")
    return build_dataset(cfg, visualize=False, train=False)


def collect_split(
    split: str,
    dataset: Any,
    store: Dict[Tuple[str, int], Dict[str, Any]],
) -> List[Tuple[str, int, int]]:
    """Sample frames and cache GT-box maps. RGB is reloaded at plot time."""
    plan = sample_plan(dataset, FRAMES_PER_SCENE, SAMPLE_SEED)
    print(f"  {split} plan={plan}")
    for scene, idx, rank in plan:
        sample = dataset[idx]
        batch = dataset.collate_batch_test([sample])
        ego = batch["ego"]
        entry: Dict[str, Any] = {
            "split": split,
            "scene": scene,
            "tag": f"{split}:{scene_tag(scene)}",
            "idx": idx,
            "rank": rank,
            "agents": {},
        }
        for agent in present_camera_agents(ego):
            if agent not in AGENTS:
                continue
            imgs = flatten_imgs(ego[agent]["batch_merged_cam_inputs"]["imgs"])
            n_view = int(imgs.shape[0])
            gt = box_r90(ego, agent, n_view)
            entry["agents"][agent] = {"n_view": n_view, "gt": gt, "p_fg": {}}
        store[(split, idx)] = entry
        print(
            f"  cached {split} idx={idx} rank={rank} scene={scene} "
            f"agents={list(entry['agents'])}"
        )
        del batch, sample
    return plan


def load_agent_rgb(dataset: Any, idx: int, agent: str, view: int) -> np.ndarray:
    """Denormalized RGB of one camera view."""
    sample = dataset[idx]
    batch = dataset.collate_batch_test([sample])
    imgs = flatten_imgs(batch["ego"][agent]["batch_merged_cam_inputs"]["imgs"])
    view = min(int(view), int(imgs.shape[0]) - 1)
    return denorm_rgb(imgs[view])


def fill_predictions(
    split: str,
    dataset: Any,
    model: torch.nn.Module,
    device: torch.device,
    epoch: int,
    plan: Sequence[Tuple[str, int, int]],
    store: Dict[Tuple[str, int], Dict[str, Any]],
) -> None:
    """Forward one checkpoint over the cached index list."""
    with torch.no_grad():
        for _scene, idx, _rank in plan:
            sample = dataset[idx]
            batch = dataset.collate_batch_test([sample])
            batch = train_utils.to_device(batch, device)
            ego = batch["ego"]
            pred = model(ego)
            entry = store[(split, idx)]
            for agent, block in entry["agents"].items():
                if agent not in pred:
                    continue
                p_fg = torch.softmax(pred[agent]["heatmap_logits"], dim=1)[:, 1]
                block["p_fg"][epoch] = p_fg.detach().cpu().numpy()
            ratio = {
                agent: round(float((block["p_fg"][epoch] >= TAU).mean()), 4)
                for agent, block in entry["agents"].items()
                if epoch in block["p_fg"]
            }
            print(f"  e{epoch} {split} idx={idx} fg_ratio={ratio}")
            del batch, sample, pred
            torch.cuda.empty_cache()


def main() -> None:
    """Dump five epoch-1/9/16 comparison sheets, one per sampled frame rank."""
    OUT.mkdir(parents=True, exist_ok=True)
    hypes = yaml_utils.load_yaml(str(LOG_DIR / "config.yaml"), None)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    store: Dict[Tuple[str, int], Dict[str, Any]] = {}
    datasets: Dict[str, Any] = {}
    plans: Dict[str, List[Tuple[str, int, int]]] = {}
    for split in ("test", "val"):
        datasets[split] = build_split_dataset(hypes, split)
        plans[split] = collect_split(split, datasets[split], store)

    model = train_utils.create_model(hypes)
    model.to(device)
    model.eval()
    _unwrap_model(model)

    for epoch in EPOCHS:
        print(f"=== epoch {epoch} ===")
        load_epoch_checkpoint(model, str(LOG_DIR), epoch)
        model.to(device)
        model.eval()
        for split, plan in plans.items():
            fill_predictions(split, datasets[split], model, device, epoch, plan, store)

    compare_by_rank: Dict[int, List[Dict[str, Any]]] = {
        r: [] for r in range(FRAMES_PER_SCENE)
    }
    manifest: List[Dict[str, Any]] = []
    for key in sorted(store, key=lambda x: (x[0], store[x]["scene"], store[x]["rank"])):
        entry = store[key]
        scene_dir = OUT / "scenes" / f"{entry['tag'].replace(':', '_')}"
        for agent in AGENTS:
            if agent not in entry["agents"]:
                continue
            block = entry["agents"][agent]
            if any(ep not in block["p_fg"] for ep in EPOCHS):
                continue
            view = pick_view(block["p_fg"][16], block["gt"])
            rgb = load_agent_rgb(datasets[entry["split"]], entry["idx"], agent, view)
            p_view = {ep: block["p_fg"][ep][view] for ep in EPOCHS}
            gt_v = (
                block["gt"][view]
                if view < block["gt"].shape[0]
                else np.zeros_like(p_view[16], dtype=np.uint8)
            )
            save_scene_panel(
                scene_dir / f"r{entry['rank']}_{agent}_idx{entry['idx']:04d}_v{view}.png",
                rgb,
                gt_v,
                p_view,
                f"{entry['tag']} sample{entry['rank']+1}/5 idx={entry['idx']} "
                f"{agent} view={view}",
            )
            ratios = {
                f"e{ep}": round(float((p_view[ep] >= TAU).mean()), 4) for ep in EPOCHS
            }
            manifest.append(
                {
                    "split": entry["split"],
                    "scene": entry["scene"],
                    "tag": entry["tag"],
                    "rank": entry["rank"],
                    "idx": entry["idx"],
                    "agent": agent,
                    "view": view,
                    "fg_ratio": ratios,
                    "gt_box_pixels": int(gt_v.sum()),
                }
            )
            compare_by_rank[int(entry["rank"])].append(
                {
                    "tag": entry["tag"],
                    "agent": agent,
                    "idx": entry["idx"],
                    "rgb": rgb,
                    "p": p_view,
                }
            )

    compare_dir = OUT / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    for rank, rows in compare_by_rank.items():
        if not rows:
            continue
        path = compare_dir / f"compare_sample{rank + 1:02d}_of_5.png"
        save_compare_sheet(
            path,
            rows,
            title=(
                f"sample {rank + 1}/5  (seed={SAMPLE_SEED}, 5 random frames/scene)  "
                "E1/E9/E16 FG@0.3  red=pred"
            ),
        )
        print(f"wrote {path} rows={len(rows)}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote scenes under {OUT / 'scenes'}")
    print(f"n_panels={len(manifest)}")


if __name__ == "__main__":
    main()
