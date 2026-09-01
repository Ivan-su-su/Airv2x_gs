# -*- coding: utf-8 -*-
"""RSU DepthHead-only fine-tune entry. Reuses production model / target / loss.

Does not change Gaussian generation, Heatmap training, Vehicle, or Drone.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from torch.utils.data import DataLoader

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.loss.gaussian_p1_depth_loss import GaussianP1DepthLoss
from opencood.models.gaussian_modules_0822.heatmap.metrics import (
    PRIMARY_OBJECTNESS_THRESHOLD,
)
from opencood.models.gaussian_modules_0822.heatmap.target import build_semantic_target
from opencood.models.gaussian_modules_0822.image_frontend import present_camera_agents
from opencood.models.gaussian_modules_0822.lss.target import (
    depth_valid_mask,
    extract_camera_z_gt,
)
from opencood.tools import train_utils
from opencood.tools.rsu_depth_head_ft_lib import (
    apply_rsu_depth_head_ft_freeze,
    apply_rsu_depth_head_train_eval,
    assert_only_rsu_depth_head_trainable,
    bin_errors,
    build_eval_plans,
    capture_frozen_outputs,
    changed_parameter_names,
    choose_distance_edges,
    collect_rsu_depth_pixels,
    compare_frozen_outputs,
    far_median_ez,
    is_rsu_depth_head_param,
    parameter_checksums,
    plot_pred_vs_gt,
    plot_residual_vs_depth,
    plot_signed_hist,
    plot_train_vs_test,
    rsu_abc_and_aseed,
    rsu_depth_train_forward,
    setup_rsu_depth_head_optimizer,
    summarize_abc,
    write_histogram,
)
from opencood.tools.train_gaussian_p1 import (
    _save_p1_checkpoint,
    _unwrap_model,
    load_pretrained_weights,
)
from opencood.tools.vis_test_heatmap_recall import scene_from_path

VIS_ROOT_DEFAULT = Path("/home/dell/suyi/visualization/rsu_depth_head_ft")
FROZEN_DRIFT_LIMIT = 1.0e-4


def parse_args() -> argparse.Namespace:
    """CLI for RSU DepthHead-only fine-tune."""
    parser = argparse.ArgumentParser(description="RSU DepthHead-only FT")
    parser.add_argument(
        "-y",
        "--hypes_yaml",
        default=(
            "opencood/hypes_yaml/airv2x/camera/gaussian_p1/"
            "airv2x_gaussian_p1_rsu_depth_head_ft.yaml"
        ),
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--tag", default="default")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--frames_per_scene", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fg_tau", type=float, default=PRIMARY_OBJECTNESS_THRESHOLD)
    parser.add_argument("--sigma0", type=float, default=1.0)
    parser.add_argument("--near_m", type=float, default=2.0)
    parser.add_argument("--cover_thresh", type=float, default=0.5)
    parser.add_argument("--out_root", default=str(VIS_ROOT_DEFAULT))
    parser.add_argument(
        "--stage",
        choices=("hist", "train", "eval", "all"),
        default="all",
    )
    parser.add_argument(
        "--model_dir",
        default="",
        help="Existing FT run dir. Required for --stage eval.",
    )
    return parser.parse_args()


def _hypes_for_split(hypes: Dict[str, Any], split: str) -> Dict[str, Any]:
    """Point validate_dir at train/val/test. Always eval-mode (no fog/aug)."""
    local = copy.deepcopy(hypes)
    local["train"] = False
    if split == "train":
        local["validate_dir"] = local["root_dir"]
    elif split == "test":
        local["validate_dir"] = local["test_dir"]
    return local


def build_split_datasets(hypes: Dict[str, Any]) -> Dict[str, Any]:
    """Fixed TRAIN/VAL/TEST datasets with train=False."""
    out = {}
    for split in ("train", "val", "test"):
        local = _hypes_for_split(hypes, split)
        print(f"Building {split} eval dataset from {local['validate_dir']}", flush=True)
        out[split] = build_dataset(local, visualize=False, train=False)
    return out


def _collate_eval(dataset: Any, idx: int, device: torch.device) -> Optional[Dict[str, Any]]:
    """One eval batch on ``device``. Returns None if the frame cannot be read."""
    try:
        sample = dataset[idx]
        batch = dataset.collate_batch_test([sample])
        batch = train_utils.to_device(batch, device)
        return batch["ego"]
    except OSError as exc:
        print(f"skip corrupt eval idx={idx}: {exc}", flush=True)
        return None


def run_histogram(train_dataset: Any, core: torch.nn.Module, vis: Path) -> Dict[str, Any]:
    """TRAIN-only SAM3∩in-range GT depth histogram. No weight updates."""
    z_all: List[np.ndarray] = []
    n_rsu = 0
    n_skip = 0
    for idx in tqdm(range(len(train_dataset)), desc="TRAIN depth-target hist"):
        sample = train_dataset[idx]
        batch = train_dataset.collate_batch_train([sample])
        ego = batch["ego"]
        if "rsu" not in present_camera_agents(ego):
            n_skip += 1
            continue
        n_rsu += 1
        cam = ego["rsu"]["batch_merged_cam_inputs"]
        z_gt = extract_camera_z_gt(cam["imgs"]).detach().cpu()
        enc = core.frontend.encoders["rsu"]
        valid = depth_valid_mask(z_gt, enc.d_min, enc.d_max)
        semantic = build_semantic_target(cam, tau=1).cpu()
        support = valid & semantic.ne(0)
        if int(support.sum().item()) == 0:
            continue
        z_all.append(z_gt[support].numpy().reshape(-1))
    z_cat = np.concatenate(z_all) if z_all else np.zeros((0,), dtype=np.float64)
    z_bins = core.depth_moments["rsu"].z_bins.detach().cpu().numpy()
    summary = write_histogram(
        z_cat,
        z_bins,
        vis / "train_depth_target_hist.csv",
        vis / "train_depth_target_hist.png",
    )
    summary["n_rsu_frames"] = n_rsu
    summary["n_no_rsu"] = n_skip
    (vis / "train_depth_target_hist.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[hist] RSU frames={n_rsu} skip={n_skip} supervised_px={summary['n_supervised_pixels']} "
        f"median_gt={summary['median_gt_m']}",
        flush=True,
    )
    return summary


def _concat_pack(chunks: Sequence[Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Concatenate pred/gt/var lists."""
    if not chunks:
        z = np.zeros((0,), dtype=np.float64)
        return {"pred": z, "gt": z, "var": z}
    return {
        key: np.concatenate([c[key] for c in chunks if c[key].size])
        if any(c[key].size for c in chunks)
        else np.zeros((0,), dtype=np.float64)
        for key in ("pred", "gt", "var")
    }


def evaluate_checkpoint(
    model: torch.nn.Module,
    datasets: Mapping[str, Any],
    plans: Mapping[str, Sequence[Tuple[str, int]]],
    device: torch.device,
    opt: argparse.Namespace,
    distance_edges: Sequence[float],
    source_frozen: Optional[Mapping[str, torch.Tensor]],
    inv_ego: Optional[Dict[str, Any]],
    vis: Path,
    tag: str,
) -> Dict[str, Any]:
    """Fixed-frame TRAIN/VAL/TEST RSU depth + ABC + freeze diffs."""
    core = _unwrap_model(model)
    apply_rsu_depth_head_train_eval(model, False)
    result: Dict[str, Any] = {"tag": tag, "splits": {}, "frozen_output_diff": []}
    if inv_ego is not None and source_frozen is not None:
        current = capture_frozen_outputs(core, inv_ego)
        diffs = compare_frozen_outputs(source_frozen, current)
        result["frozen_output_diff"] = diffs
        for row in diffs:
            if row["allowed"] or row["max_abs"] is None:
                continue
            if float(row["max_abs"]) > FROZEN_DRIFT_LIMIT:
                raise AssertionError(
                    f"frozen output drifted: {row['tensor']} max_abs={row['max_abs']}"
                )
    for split, dataset in datasets.items():
        packs: Dict[str, List[Dict[str, np.ndarray]]] = defaultdict(list)
        abc_rows: List[Dict[str, Any]] = []
        n_rsu = 0
        for scene, idx in tqdm(plans[split], desc=f"eval {tag} {split}", leave=False):
            ego = _collate_eval(dataset, int(idx), device)
            if ego is None or "rsu" not in present_camera_agents(ego):
                continue
            n_rsu += 1
            with torch.no_grad():
                pred = core(ego)
            pixel = collect_rsu_depth_pixels(core, ego, pred, opt.fg_tau)
            for key, pack in pixel.items():
                packs[key].append(pack)
            rows, aseed = rsu_abc_and_aseed(
                core,
                ego,
                pred,
                opt.fg_tau,
                opt.sigma0,
                opt.near_m,
                opt.cover_thresh,
                box_res=5,
                anisotropy_max=4.0,
                orient_window=7,
                eps=1.0e-4,
            )
            for row in rows:
                row.update({"split": split, "scene": scene, "idx": int(idx), "tag": tag})
            abc_rows.extend(rows)
            packs["aseed"].append(aseed)
        split_out: Dict[str, Any] = {"n_rsu_frames": n_rsu, "abc": summarize_abc(abc_rows)}
        for key in ("all_valid", "sam3_fg", "heatmap_seed", "aseed"):
            cat = _concat_pack(packs.get(key, []))
            stats = _error_and_sigma(cat)
            stats["far_median_ez"] = far_median_ez(cat["pred"], cat["gt"], 25.0)
            stats["distance_bins"] = bin_errors(cat["pred"], cat["gt"], distance_edges)
            split_out[key] = stats
            split_out[f"_{key}_arrays"] = cat
        split_out["_abc_rows"] = abc_rows
        result["splits"][split] = split_out
        print(
            f"  [{tag} {split}] RSU frames={n_rsu} "
            f"aseed MAE={split_out['aseed']['mae']} "
            f"median_ez={split_out['aseed']['median_ez']} "
            f"P(A)={split_out['abc']['P(A)']} P(B|A)={split_out['abc']['P(B|A)']}",
            flush=True,
        )
    _write_ckpt_plots(result, vis, tag)
    # Drop bulky arrays before JSON.
    for split in result["splits"]:
        for key in list(result["splits"][split]):
            if key.startswith("_"):
                result["splits"][split].pop(key)
    return result


def _error_and_sigma(pack: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    """MAE / signed error / categorical sigma_z."""
    from opencood.tools.rsu_depth_head_ft_lib import _error_stats, _sigma_stats

    stats = _error_stats(pack["pred"], pack["gt"])
    stats.update(_sigma_stats(pack["pred"], pack["gt"], pack["var"]))
    return stats


def _write_ckpt_plots(result: Dict[str, Any], vis: Path, tag: str) -> None:
    """Per-checkpoint residual / pred-vs-gt / signed-hist for TEST A-seed."""
    test = result["splits"].get("test", {})
    arrays = test.get("_aseed_arrays")
    bins = test.get("aseed", {}).get("distance_bins", [])
    if arrays is not None:
        plot_signed_hist(
            arrays["pred"],
            arrays["gt"],
            vis / "signed_error_hist" / f"{tag}_test_aseed.png",
            f"{tag} TEST A-seed e_z",
        )
        plot_pred_vs_gt(
            arrays["pred"],
            arrays["gt"],
            vis / "pred_vs_gt" / f"{tag}_test_aseed.png",
            f"{tag} TEST A-seed z_pred vs z_gt",
        )
    if bins:
        plot_residual_vs_depth(
            bins,
            vis / "residual_vs_depth" / f"{tag}_test_aseed.png",
            f"{tag} TEST A-seed median e_z vs z_gt",
        )


def _flatten_metrics(ckpt_rows: Sequence[Mapping[str, Any]]) -> Tuple[List[List[Any]], List[List[Any]], List[List[Any]]]:
    """CSV rows for checkpoint / distance-bin / ABC tables."""
    ckpt_csv = []
    bin_csv = []
    abc_csv = []
    for row in ckpt_rows:
        tag = row["tag"]
        for split, block in row["splits"].items():
            for support in ("all_valid", "sam3_fg", "heatmap_seed", "aseed"):
                s = block[support]
                ckpt_csv.append(
                    [
                        tag,
                        split,
                        support,
                        s.get("n"),
                        s.get("mae"),
                        s.get("medae"),
                        s.get("rmse"),
                        s.get("mean_ez"),
                        s.get("median_ez"),
                        s.get("frac_neg"),
                        s.get("frac_pos"),
                        s.get("median_sigma_z"),
                        s.get("median_abs_ez_over_sigma"),
                        s.get("far_median_ez"),
                    ]
                )
                for b in s.get("distance_bins") or []:
                    bin_csv.append(
                        [
                            tag,
                            split,
                            support,
                            b["z_lo"],
                            b["z_hi"],
                            b["n"],
                            b["median_ez"],
                            b["median_abs_ez"],
                            b["mae"],
                        ]
                    )
            abc = block["abc"]
            abc_csv.append(
                [
                    tag,
                    split,
                    abc.get("n_visible"),
                    abc.get("P(A)"),
                    abc.get("P(B|A)"),
                    abc.get("P(C|A)"),
                    abc.get("P(ABC)"),
                ]
            )
    return ckpt_csv, bin_csv, abc_csv


def write_csvs(vis: Path, ckpt_rows: Sequence[Mapping[str, Any]]) -> None:
    """checkpoint_metrics.csv / distance_bin_metrics.csv / abc_metrics.csv."""
    ckpt_csv, bin_csv, abc_csv = _flatten_metrics(ckpt_rows)
    with (vis / "checkpoint_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "tag",
                "split",
                "support",
                "n",
                "mae",
                "medae",
                "rmse",
                "mean_ez",
                "median_ez",
                "frac_neg",
                "frac_pos",
                "median_sigma_z",
                "median_abs_ez_over_sigma",
                "far_median_ez",
            ]
        )
        writer.writerows(ckpt_csv)
    with (vis / "distance_bin_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "tag",
                "split",
                "support",
                "z_lo",
                "z_hi",
                "n",
                "median_ez",
                "median_abs_ez",
                "mae",
            ]
        )
        writer.writerows(bin_csv)
    with (vis / "abc_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tag", "split", "n_visible", "P(A)", "P(B|A)", "P(C|A)", "P(ABC)"])
        writer.writerows(abc_csv)
    with (vis / "frozen_output_diff.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tag", "tensor", "max_abs", "allowed", "status"])
        for row in ckpt_rows:
            for diff in row.get("frozen_output_diff") or []:
                writer.writerow(
                    [row["tag"], diff["tensor"], diff["max_abs"], diff["allowed"], diff["status"]]
                )


def classify_case(ckpt_rows: Sequence[Mapping[str, Any]]) -> str:
    """A/B/C/D from TRAIN vs TEST A-seed MAE and far residual."""
    by_tag = {r["tag"]: r for r in ckpt_rows}
    source = by_tag.get("source")
    if source is None or len(ckpt_rows) < 2:
        return "D"
    train0 = source["splits"]["train"]["aseed"]["mae"]
    test0 = source["splits"]["test"]["aseed"]["mae"]
    far0 = source["splits"]["test"]["aseed"]["far_median_ez"]
    later = [r for r in ckpt_rows if r["tag"] != "source"]
    if not later:
        return "D"
    best_train = min(later, key=lambda r: r["splits"]["train"]["aseed"]["mae"] or 1e9)
    best_test = min(later, key=lambda r: r["splits"]["test"]["aseed"]["mae"] or 1e9)
    last = later[-1]
    train_gain = (train0 or 0) - (best_train["splits"]["train"]["aseed"]["mae"] or 0)
    test_gain = (test0 or 0) - (best_test["splits"]["test"]["aseed"]["mae"] or 0)
    last_test = last["splits"]["test"]["aseed"]["mae"]
    last_train = last["splits"]["train"]["aseed"]["mae"]
    far_last = last["splits"]["test"]["aseed"]["far_median_ez"]
    if train0 is None or train_gain < 0.15:
        return "D"
    overfit = (
        last_train is not None
        and last_test is not None
        and test0 is not None
        and last_train < (best_train["splits"]["train"]["aseed"]["mae"] or last_train) + 0.05
        and last_test > (best_test["splits"]["test"]["aseed"]["mae"] or last_test) + 0.2
    )
    if overfit and test_gain < 0.2:
        return "B"
    far_ok = far0 is not None and far_last is not None and abs(far_last) < abs(far0) * 0.6
    test_ok = test_gain >= 0.5
    if test_ok and far_ok:
        return "A"
    if test_gain >= 0.2:
        return "C"
    if train_gain >= 0.3 and test_gain < 0.15:
        return "B"
    return "C"


def write_report(
    vis: Path,
    meta: Dict[str, Any],
    hist: Optional[Dict[str, Any]],
    ckpt_rows: Sequence[Mapping[str, Any]],
) -> None:
    """report.json + report.txt."""
    case = classify_case(ckpt_rows)
    payload = {
        **meta,
        "train_depth_target": hist,
        "checkpoints": ckpt_rows,
        "case": case,
        "val_sam3_note": (
            "VAL and TEST have no camera *_seg.bin. SAM3 FG support is empty "
            "there; do not select checkpoints from VAL SAM3 depth loss. Use "
            "all_valid / heatmap_seed / A-seed / ABC instead. VAL is still a "
            "held-out Town01 sunny scene, not a fog or far-RSU proxy."
        ),
    }
    (vis / "report.json").write_text(json.dumps(payload, indent=2, default=str))
    lines = [
        "RSU DepthHead-only fine-tune report",
        "===================================",
        f"source checkpoint: {meta.get('source_ckpt')}",
        f"trainable count: {meta.get('trainable_count')}",
        "trainable names:",
    ]
    for name in meta.get("trainable_names") or []:
        lines.append(f"  {name}")
    lines.extend(
        [
            f"optimizer: {meta.get('optimizer')}",
            f"epochs: {meta.get('epochs')}  iters/epoch: {meta.get('n_iter')}",
            f"saves: {meta.get('save_tags')}",
            f"case: {payload['case']}",
            "",
            payload["val_sam3_note"],
            "",
        ]
    )
    if hist:
        lines.append(
            f"TRAIN supervised pixels={hist.get('n_supervised_pixels')} "
            f"median_gt={hist.get('median_gt_m')} p10={hist.get('percentiles_gt_m', {}).get('p10')} "
            f"p90={hist.get('percentiles_gt_m', {}).get('p90')}"
        )
        lines.append(f"near/mid/far: {hist.get('near_mid_far')}")
        lines.append("")
    lines.append("tag split aseed_MAE median_ez far_median_ez P(A) P(B|A)")
    for row in ckpt_rows:
        for split in ("train", "val", "test"):
            block = row["splits"][split]
            a = block["aseed"]
            abc = block["abc"]
            lines.append(
                f"{row['tag']:10s} {split:5s}  "
                f"MAE={a.get('mae')}  med_ez={a.get('median_ez')}  "
                f"far_ez={a.get('far_median_ez')}  "
                f"P(A)={abc.get('P(A)')}  P(B|A)={abc.get('P(B|A)')}"
            )
    (vis / "report.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {vis / 'report.txt'}", flush=True)


def _save_marks(n_iter: int) -> Dict[Tuple[int, int], Tuple[str, str]]:
    """(epoch_index, 1-based iter) → (eval tag, filename)."""
    q25 = max(1, n_iter // 4)
    q50 = max(q25 + 1, n_iter // 2)
    return {
        (0, q25): ("ft_0.25", "net_frac025.pth"),
        (0, q50): ("ft_0.5", "net_frac050.pth"),
        (0, n_iter): ("ft_1", "net_epoch1.pth"),
        (1, n_iter): ("ft_2", "net_epoch2.pth"),
        (2, n_iter): ("ft_3", "net_epoch3.pth"),
        (4, n_iter): ("ft_5", "net_epoch5.pth"),
    }


def main() -> None:
    """Histogram → freeze audit → short FT → fixed-frame eval."""
    opt = parse_args()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, None)
    hypes["tag"] = opt.tag
    ft_cfg = hypes.get("p1_finetune") or {}
    if not ft_cfg.get("rsu_depth_head_only"):
        raise ValueError("yaml p1_finetune.rsu_depth_head_only must be true")
    source_ckpt = str(ft_cfg["pretrained_ckpt"])
    vis = Path(opt.out_root)
    vis.mkdir(parents=True, exist_ok=True)
    for sub in ("signed_error_hist", "residual_vs_depth", "pred_vs_gt", "train_vs_test"):
        (vis / sub).mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    print("Creating model...", flush=True)
    model = train_utils.create_model(hypes)
    load_pretrained_weights(model, source_ckpt, torch.device("cpu"))
    trainable = apply_rsu_depth_head_ft_freeze(model)
    model.to(device)
    apply_rsu_depth_head_train_eval(model, True)
    core = _unwrap_model(model)
    total_n = sum(p.nelement() for p in model.parameters())
    train_n = sum(p.nelement() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_n:,}")
    print(f"Trainable parameters: {train_n:,}")
    print("Trainable parameter names:")
    for name in trainable:
        print(f"  {name} {tuple(dict(model.named_parameters())[name].shape)}")
    assert_only_rsu_depth_head_trainable(model)

    print("Building TRAIN dataset...", flush=True)
    train_dataset = build_dataset(hypes, visualize=False, train=True)
    hist: Optional[Dict[str, Any]] = None
    if opt.stage in ("hist", "all"):
        hist = run_histogram(train_dataset, core, vis)
        if opt.stage == "hist":
            return

    eval_datasets: Dict[str, Any] = {}
    plans: Dict[str, Any] = {}
    inv_ego: Optional[Dict[str, Any]] = None
    source_frozen: Optional[Dict[str, Any]] = None
    edges: List[float] = []
    source_checksums = parameter_checksums(model)

    if opt.stage != "train":
        eval_datasets = build_split_datasets(hypes)
        plans = build_eval_plans(eval_datasets, opt.frames_per_scene, opt.seed)
        (vis / "eval_plan.json").write_text(
            json.dumps({k: v for k, v in plans.items()}, indent=2)
        )
        print(
            "eval frames: "
            + ", ".join(f"{k}={len(v)}" for k, v in plans.items()),
            flush=True,
        )
        for scene, idx in plans["train"]:
            ego = _collate_eval(eval_datasets["train"], int(idx), device)
            if ego is None:
                continue
            present = present_camera_agents(ego)
            if set(present) >= {"vehicle", "rsu", "drone"}:
                inv_ego = ego
                print(
                    f"invariance frame train idx={idx} scene={scene} agents={present}",
                    flush=True,
                )
                break
        if inv_ego is None:
            raise RuntimeError("no TRAIN eval frame has vehicle+rsu+drone")
        apply_rsu_depth_head_train_eval(model, False)
        source_frozen = capture_frozen_outputs(core, inv_ego)
        source_checksums = parameter_checksums(model)
        apply_rsu_depth_head_train_eval(model, False)
        gt_for_edges: List[np.ndarray] = []
        for scene, idx in plans["test"]:
            ego = _collate_eval(eval_datasets["test"], int(idx), device)
            if ego is None or "rsu" not in present_camera_agents(ego):
                continue
            with torch.no_grad():
                pred = core(ego)
            _rows, aseed = rsu_abc_and_aseed(
                core,
                ego,
                pred,
                opt.fg_tau,
                opt.sigma0,
                opt.near_m,
                opt.cover_thresh,
                5,
                4.0,
                7,
                1.0e-4,
            )
            if aseed["gt"].size:
                gt_for_edges.append(aseed["gt"])
        edges = choose_distance_edges(
            np.concatenate(gt_for_edges) if gt_for_edges else np.zeros((0,))
        )
        print(f"distance edges (m): {edges}", flush=True)
        (vis / "distance_edges.json").write_text(json.dumps(edges))

    saved_path = opt.model_dir
    ckpt_specs: List[Tuple[str, str]] = []
    ckpt_rows: List[Dict[str, Any]] = []

    def _eval_current(tag: str) -> None:
        """Evaluate the in-memory weights on the fixed plans."""
        print(f"Evaluating {tag} (in-memory weights)", flush=True)
        row = evaluate_checkpoint(
            model,
            eval_datasets,
            plans,
            device,
            opt,
            edges,
            source_frozen,
            inv_ego,
            vis,
            tag,
        )
        ckpt_rows.append(row)
        write_csvs(vis, ckpt_rows)
        plot_train_vs_test(ckpt_rows, vis / "train_vs_test")

    if opt.stage in ("train", "all"):
        if not saved_path:
            saved_path = train_utils.setup_train(hypes)
        vis_run = vis / Path(saved_path).name
        vis_run.mkdir(parents=True, exist_ok=True)
        plan_path = vis / "eval_plan.json"
        if plan_path.is_file():
            shutil.copy2(plan_path, vis_run / "eval_plan.json")
        (Path(saved_path) / "trainable_parameter_names.txt").write_text(
            "\n".join(trainable) + "\n"
        )
        (Path(saved_path) / "parameter_audit.txt").write_text(
            f"total={total_n}\ntrainable={train_n}\n" + "\n".join(trainable) + "\n"
        )
        opt_cfg = hypes["optimizer"]
        optimizer = setup_rsu_depth_head_optimizer(
            model,
            lr=float(opt_cfg["lr"]),
            eps=float((opt_cfg.get("args") or {}).get("eps", 1e-10)),
            weight_decay=float((opt_cfg.get("args") or {}).get("weight_decay", 1e-4)),
        )
        depth_criterion = GaussianP1DepthLoss(dict(hypes["loss"]["depth"]["args"]))
        writer = SummaryWriter(saved_path)
        step0 = _save_p1_checkpoint(
            saved_path, "net_step0.pth", -1, model, optimizer, None, None
        )
        ckpt_specs.append(("source", step0))
        if opt.stage == "all":
            _eval_current("source")

        if hasattr(train_dataset, "set_fog_epoch"):
            train_dataset.set_fog_epoch(0)
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(hypes["train_params"]["batch_size"]),
            shuffle=True,
            num_workers=int(opt.worker),
            collate_fn=train_dataset.collate_batch_train,
            pin_memory=True,
            drop_last=True,
        )
        n_iter = len(train_loader)
        epochs = int(hypes["train_params"]["epoches"])
        marks = _save_marks(n_iter)
        print(f"Training {epochs} epochs, n_iter={n_iter}, marks={marks}", flush=True)
        global_step = 0
        for epoch in range(epochs):
            if hasattr(train_dataset, "set_fog_epoch"):
                train_dataset.set_fog_epoch(epoch)
            apply_rsu_depth_head_train_eval(model, True)
            pbar = tqdm(enumerate(train_loader), total=n_iter, desc=f"FT epoch {epoch}")
            n_used = 0
            n_skip = 0
            for i, batch_data in pbar:
                if batch_data is None:
                    continue
                batch_data = train_utils.to_device(batch_data, device)
                ego = batch_data["ego"]
                ego["epoch"] = epoch
                optimizer.zero_grad(set_to_none=True)
                loss = rsu_depth_train_forward(core, ego, depth_criterion)
                if loss is None:
                    n_skip += 1
                    continue
                loss.backward()
                optimizer.step()
                n_used += 1
                global_step += 1
                writer.add_scalar("Train/rsu_depth_loss", float(loss.item()), global_step)
                pbar.set_postfix(loss=float(loss.item()), used=n_used, skip=n_skip)
                key = (epoch, i + 1)
                if key in marks:
                    tag, fname = marks[key]
                    path = _save_p1_checkpoint(
                        saved_path, fname, epoch, model, optimizer, None, None
                    )
                    ckpt_specs.append((tag, path))
                    frozen_now = changed_parameter_names(
                        source_checksums, parameter_checksums(model)
                    )
                    illegal = [n for n in frozen_now if not is_rsu_depth_head_param(n)]
                    if illegal:
                        raise AssertionError(f"frozen params changed: {illegal}")
                    if opt.stage == "all":
                        _eval_current(tag)
                        apply_rsu_depth_head_train_eval(model, True)
            print(f"epoch {epoch} used={n_used} skip={n_skip}", flush=True)
        writer.close()
        final_changed = changed_parameter_names(source_checksums, parameter_checksums(model))
        (Path(saved_path) / "changed_parameter_names.txt").write_text(
            "\n".join(final_changed) + "\n"
        )
        shutil.copy2(
            Path(saved_path) / "changed_parameter_names.txt",
            vis / "changed_parameter_names.txt",
        )
        illegal = [n for n in final_changed if not is_rsu_depth_head_param(n)]
        if illegal:
            raise AssertionError(f"experiment invalid, frozen params changed: {illegal}")

    if opt.stage == "eval":
        if not saved_path:
            raise ValueError("--model_dir is required for eval")
        ckpt_specs = _discover_ckpts(saved_path)
        for tag, path in ckpt_specs:
            print(f"Evaluating {tag} {path}", flush=True)
            raw = torch.load(path, map_location="cpu")
            state = raw["model_state_dict"] if "model_state_dict" in raw else raw
            _unwrap_model(model).load_state_dict(state, strict=True)
            apply_rsu_depth_head_ft_freeze(model)
            _eval_current(tag)

    if ckpt_rows:
        unique_tags = [r["tag"] for r in ckpt_rows]
        meta = {
            "source_ckpt": source_ckpt,
            "ft_run_dir": saved_path,
            "trainable_names": trainable,
            "trainable_count": train_n,
            "total_count": total_n,
            "optimizer": {
                "name": "Adam",
                "lr": float(hypes["optimizer"]["lr"]),
                "eps": float((hypes["optimizer"].get("args") or {}).get("eps", 1e-10)),
                "weight_decay": float(
                    (hypes["optimizer"].get("args") or {}).get("weight_decay", 1e-4)
                ),
                "scheduler": "constant",
            },
            "epochs": int(hypes["train_params"]["epoches"]),
            "n_iter": len(train_dataset),
            "save_tags": unique_tags,
            "distance_edges_m": edges,
            "eval_plan": {k: v for k, v in plans.items()},
            "fg_tau": opt.fg_tau,
            "sigma0": opt.sigma0,
            "near_m": opt.near_m,
        }
        write_report(vis, meta, hist, ckpt_rows)


def _discover_ckpts(saved_path: str) -> List[Tuple[str, str]]:
    """Map saved files to eval tags."""
    mapping = [
        ("source", "net_step0.pth"),
        ("ft_0.25", "net_frac025.pth"),
        ("ft_0.5", "net_frac050.pth"),
        ("ft_1", "net_epoch1.pth"),
        ("ft_2", "net_epoch2.pth"),
        ("ft_3", "net_epoch3.pth"),
        ("ft_5", "net_epoch5.pth"),
    ]
    out = []
    for tag, name in mapping:
        path = os.path.join(saved_path, name)
        if os.path.isfile(path):
            out.append((tag, path))
    return out


if __name__ == "__main__":
    main()
