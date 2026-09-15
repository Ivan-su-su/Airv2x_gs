#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose two things on the SAME checkpoint:

1) Where vehicle Gaussian XY sigma becomes large:
   Init -> Stage1 -> Stage2 -> Stage3 voxel pool input.

2) What sigma shrink really does to FINAL post-processed detections,
   not just raw negative-anchor pass rate:
   pred/frame, GT/frame, TP/frame, FP/frame, precision, recall, AP@IoU.

READ ONLY. The sigma multiplier exists only in a temporary pre-hook on
model.stage3.splat and is removed after that inference pass.

Example:
python opencood/tools/diagnose_scale_source_and_final_fp.py \
  --model_dir /path/to/run \
  --epoch 3 \
  --num_frames 50 \
  --workers 4 \
  --sigma_scales 1.0,0.85,0.75,0.65,0.5 \
  --iou 0.3
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import replace
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_THIS = os.path.abspath(__file__)
_REPO = "/".join(_THIS.split("/")[:-3])
if os.path.isdir(os.path.join(_REPO, "opencood")):
    sys.path.insert(0, _REPO)
else:
    sys.path.insert(0, os.getcwd())

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import inference_utils, train_utils
from opencood.utils import eval_utils_opv2v as eval_utils


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--epoch", required=True, type=int)
    p.add_argument("--num_frames", type=int, default=50)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--iou", type=float, default=0.3)
    p.add_argument("--sigma_scales", type=str, default="1.0,0.85,0.75,0.65,0.5")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def qstats(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(x.max()),
        "lt_0p5_frac": float(np.mean(x < 0.5)),
        "0p5_1_frac": float(np.mean((x >= 0.5) & (x < 1.0))),
        "1_2_frac": float(np.mean((x >= 1.0) & (x < 2.0))),
        "2_3_frac": float(np.mean((x >= 2.0) & (x < 3.0))),
        "gt_2_frac": float(np.mean(x >= 2.0)),
        "gt_3_frac": float(np.mean(x >= 3.0)),
        "gt_4_frac": float(np.mean(x >= 4.0)),
    }


@torch.no_grad()
def sigma_xy(gs):
    cov_xy = gs.covariance()[:, :2, :2].float()
    eig = torch.linalg.eigvalsh(cov_xy).clamp_min(0.0)
    return torch.sqrt(eig[:, 0]), torch.sqrt(eig[:, 1])


class VehicleScaleTracer:
    """Capture vehicle geometry without modifying production behavior.

    Important: GaussianInitializer is a plain Python class, NOT nn.Module, so
    PyTorch forward hooks cannot be registered on it. The model calls
    ``self.initializer.forward(data_dict)`` directly. Therefore we temporarily
    wrap that bound ``forward`` method for diagnostics, while Stage1/Stage2 are
    captured with normal PyTorch pre-hooks on nn.Module objects.
    """

    def __init__(self, model):
        self.model = model
        self.enabled = False
        self.current = {}
        self.handles = []

        # GaussianInitializer is NOT nn.Module. Save its original bound method
        # and replace only this instance's forward with a transparent wrapper.
        self._initializer_forward_original = model.initializer.forward

        def _initializer_forward_wrapped(*args, **kwargs):
            output = self._initializer_forward_original(*args, **kwargs)
            self.store("Init", self.vehicle(output))
            return output

        model.initializer.forward = _initializer_forward_wrapped

        # cross_agent input == Stage1 output; stage3 input == Stage2 output.
        if not hasattr(model.cross_agent, "register_forward_pre_hook"):
            raise TypeError(
                f"model.cross_agent is not hookable nn.Module: {type(model.cross_agent)}"
            )
        if not hasattr(model.stage3, "register_forward_pre_hook"):
            raise TypeError(
                f"model.stage3 is not hookable nn.Module: {type(model.stage3)}"
            )

        self.handles.append(
            model.cross_agent.register_forward_pre_hook(self._stage1_hook)
        )
        self.handles.append(
            model.stage3.register_forward_pre_hook(self._stage2_hook)
        )

    def close(self):
        # Always restore the exact original initializer method.
        self.model.initializer.forward = self._initializer_forward_original
        for h in self.handles:
            h.remove()

    def reset(self):
        self.current = {}

    @staticmethod
    def vehicle(d):
        return d.get("vehicle") if isinstance(d, Mapping) else None

    @torch.no_grad()
    def store(self, name, gs):
        if not self.enabled or gs is None or gs.n_gaussians == 0:
            return
        minor, major = sigma_xy(gs)
        self.current[name] = {
            "n": int(gs.n_gaussians),
            "scale_x": gs.scale[:, 0].detach().float().cpu().numpy(),
            "scale_y": gs.scale[:, 1].detach().float().cpu().numpy(),
            "scale_z": gs.scale[:, 2].detach().float().cpu().numpy(),
            "sigma_minor": minor.detach().cpu().numpy(),
            "sigma_major": major.detach().cpu().numpy(),
        }

    def _stage1_hook(self, module, inputs):
        if inputs:
            self.store("Stage1", self.vehicle(inputs[0]))

    def _stage2_hook(self, module, inputs):
        if not inputs:
            return
        veh = self.vehicle(inputs[0])
        self.store("Stage2", veh)
        if (
            self.enabled
            and veh is not None
            and veh.n_gaussians > 0
            and hasattr(self.model.stage3, "pool")
        ):
            self.store("Pool", self.model.stage3.pool(veh))


class SigmaScalePreHook:
    """Temporary scale-only perturbation at splat input."""

    def __init__(self, multiplier):
        self.multiplier = float(multiplier)

    def __call__(self, module, inputs):
        if not inputs or abs(self.multiplier - 1.0) < 1e-12:
            return None
        gs = inputs[0]
        return (replace(gs, scale=gs.scale * self.multiplier),)


def one_frame_stat(pred_box, pred_score, gt_box, iou):
    stat = {iou: {"tp": [], "fp": [], "gt": 0, "score": []}}
    eval_utils.caluclate_tp_fp(pred_box, pred_score, gt_box, stat, iou)
    tp = int(sum(stat[iou]["tp"]))
    fp = int(sum(stat[iou]["fp"]))
    gt = int(stat[iou]["gt"])
    return tp, fp, gt, tp + fp


def new_global_stat(iou):
    return {iou: {"tp": [], "fp": [], "gt": 0, "score": []}}


def safe_ap(stat, iou):
    if stat[iou]["gt"] <= 0 or len(stat[iou]["tp"]) == 0:
        return 0.0
    tmp = copy.deepcopy(stat)
    ap, _, _ = eval_utils.calculate_ap(tmp, iou, global_sort_detections=False)
    return float(ap)


def inference_once(dataset, model, batch_data):
    result = inference_utils.inference_intermediate_fusion(batch_data, model, dataset)
    if len(result) == 4:
        pred_box, pred_score, gt_box, _ = result
    elif len(result) == 3:
        pred_box, pred_score, gt_box = result
    else:
        raise RuntimeError(
            "Unexpected inference_intermediate_fusion return length: " + str(len(result))
        )
    return pred_box, pred_score, gt_box


def main():
    args = parse_args()
    scales = [float(x.strip()) for x in args.sigma_scales.split(",") if x.strip()]
    if not any(abs(x - 1.0) < 1e-12 for x in scales):
        scales.insert(0, 1.0)

    out_dir = args.out or os.path.join(
        args.model_dir, f"scale_source_final_fp_epoch{args.epoch}"
    )
    os.makedirs(out_dir, exist_ok=True)

    hypes = yaml_utils.load_yaml(None, args)
    if "test_dir" in hypes:
        hypes["validate_dir"] = hypes["test_dir"]

    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    model = train_utils.create_model(hypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    _, model = train_utils.load_model(
        args.model_dir, model, args.epoch, start_from_best=False
    )
    model.eval()
    model.zero_grad(set_to_none=True)

    if not hasattr(model, "stage3") or not hasattr(model.stage3, "splat"):
        raise AttributeError("Expected model.stage3.splat")

    tracer = VehicleScaleTracer(model)
    stage_arrays = defaultdict(lambda: defaultdict(list))

    per_scale = {
        s: {"stat": new_global_stat(args.iou), "frames": []}
        for s in scales
    }
    csv_rows = []
    processed = 0

    try:
        for frame_idx, batch_data in tqdm(enumerate(loader)):
            if frame_idx < args.start:
                continue
            if frame_idx >= args.start + args.num_frames:
                break

            batch_data = train_utils.to_device(batch_data, device)

            for s in scales:
                baseline = abs(s - 1.0) < 1e-12
                tracer.enabled = baseline
                tracer.reset()

                scale_hook = None
                if not baseline:
                    scale_hook = model.stage3.splat.register_forward_pre_hook(
                        SigmaScalePreHook(s)
                    )

                try:
                    with torch.no_grad():
                        pred_box, pred_score, gt_box = inference_once(
                            dataset, model, batch_data
                        )
                finally:
                    if scale_hook is not None:
                        scale_hook.remove()

                tp, fp, gt, pred = one_frame_stat(
                    pred_box, pred_score, gt_box, args.iou
                )
                eval_utils.caluclate_tp_fp(
                    pred_box,
                    pred_score,
                    gt_box,
                    per_scale[s]["stat"],
                    args.iou,
                )

                row = {
                    "frame": frame_idx,
                    "sigma_scale": s,
                    "pred": pred,
                    "gt": gt,
                    "tp": tp,
                    "fp": fp,
                    "precision": tp / max(tp + fp, 1),
                    "recall": tp / max(gt, 1),
                }
                per_scale[s]["frames"].append(row)
                csv_rows.append(row)

                if baseline:
                    for stage, dd in tracer.current.items():
                        for key in (
                            "scale_x",
                            "scale_y",
                            "scale_z",
                            "sigma_minor",
                            "sigma_major",
                        ):
                            stage_arrays[stage][key].append(dd[key])

            processed += 1
            pieces = [f"[frame {frame_idx:04d}]"]
            for s in scales:
                r = per_scale[s]["frames"][-1]
                pieces.append(
                    f"x{s:g}: pred={r['pred']} TP={r['tp']} FP={r['fp']} GT={r['gt']}"
                )
            print(" | ".join(pieces))

    finally:
        tracer.close()

    # stage-wise sigma source
    evolution = {}
    for stage, dd in stage_arrays.items():
        evolution[stage] = {}
        for key, chunks in dd.items():
            arr = np.concatenate(chunks) if chunks else np.empty(0)
            evolution[stage][key] = qstats(arr)

    # real final post-process metrics
    final = {}
    for s in scales:
        frames = per_scale[s]["frames"]
        pred = np.asarray([r["pred"] for r in frames], dtype=float)
        gt = np.asarray([r["gt"] for r in frames], dtype=float)
        tp = np.asarray([r["tp"] for r in frames], dtype=float)
        fp = np.asarray([r["fp"] for r in frames], dtype=float)

        total_tp = int(tp.sum())
        total_fp = int(fp.sum())
        total_gt = int(gt.sum())
        final[str(s)] = {
            "sigma_scale": s,
            "frames": len(frames),
            "pred_per_frame": qstats(pred),
            "gt_per_frame": qstats(gt),
            "tp_per_frame": qstats(tp),
            "fp_per_frame": qstats(fp),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_gt": total_gt,
            "precision_iou": total_tp / max(total_tp + total_fp, 1),
            "recall_iou": total_tp / max(total_gt, 1),
            "ap_iou": safe_ap(per_scale[s]["stat"], args.iou),
        }

    base = final[str(1.0)]
    comparisons = {}
    for s in scales:
        cur = final[str(s)]
        bfp = base["fp_per_frame"]["mean"]
        cfp = cur["fp_per_frame"]["mean"]
        bpred = base["pred_per_frame"]["mean"]
        cpred = cur["pred_per_frame"]["mean"]
        comparisons[str(s)] = {
            "delta_fp_per_frame": cfp - bfp,
            "relative_fp_change": (cfp - bfp) / bfp if bfp > 0 else float("nan"),
            "delta_pred_per_frame": cpred - bpred,
            "relative_pred_change": (
                (cpred - bpred) / bpred if bpred > 0 else float("nan")
            ),
            "delta_precision": cur["precision_iou"] - base["precision_iou"],
            "delta_recall": cur["recall_iou"] - base["recall_iou"],
            "delta_ap": cur["ap_iou"] - base["ap_iou"],
        }

    result = {
        "model_dir": args.model_dir,
        "epoch": args.epoch,
        "num_frames_processed": processed,
        "iou": args.iou,
        "note": (
            "FINAL post-process detections are measured through "
            "inference_intermediate_fusion. These are not raw anchor pass rates."
        ),
        "vehicle_scale_evolution": evolution,
        "final_postprocess_sigma_ablation": final,
        "baseline_comparisons": comparisons,
    }

    json_path = os.path.join(out_dir, "scale_source_and_final_fp.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=True)

    csv_path = os.path.join(out_dir, "per_frame_sigma_final_fp.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "sigma_scale",
                "pred",
                "gt",
                "tp",
                "fp",
                "precision",
                "recall",
            ],
        )
        w.writeheader()
        w.writerows(csv_rows)

    print("\n================ VEHICLE SIGMA EVOLUTION ================")
    print(
        f"{'stage':10s} {'med':>8s} {'p90':>8s} {'p99':>8s} "
        f"{'>2m%':>8s} {'>3m%':>8s} {'>4m%':>8s}"
    )
    for stage in ("Init", "Stage1", "Stage2", "Pool"):
        if stage not in evolution:
            continue
        r = evolution[stage]["sigma_major"]
        print(
            f"{stage:10s} "
            f"{r.get('median',float('nan')):8.3f} "
            f"{r.get('p90',float('nan')):8.3f} "
            f"{r.get('p99',float('nan')):8.3f} "
            f"{100*r.get('gt_2_frac',0):8.2f} "
            f"{100*r.get('gt_3_frac',0):8.2f} "
            f"{100*r.get('gt_4_frac',0):8.2f}"
        )

    print("\n================ FINAL BOX SIGMA ABLATION ================")
    print(
        f"{'sigma':>6s} {'pred/fr':>9s} {'GT/fr':>8s} {'TP/fr':>8s} "
        f"{'FP/fr':>8s} {'Prec':>8s} {'Recall':>8s} {'AP':>8s} {'dFP/fr':>9s}"
    )
    for s in scales:
        r = final[str(s)]
        c = comparisons[str(s)]
        print(
            f"{s:6.2f} "
            f"{r['pred_per_frame']['mean']:9.2f} "
            f"{r['gt_per_frame']['mean']:8.2f} "
            f"{r['tp_per_frame']['mean']:8.2f} "
            f"{r['fp_per_frame']['mean']:8.2f} "
            f"{r['precision_iou']:8.4f} "
            f"{r['recall_iou']:8.4f} "
            f"{r['ap_iou']:8.4f} "
            f"{c['delta_fp_per_frame']:9.2f}"
        )

    print("\nKey metric: FINAL FP/frame, not raw negative-anchor pass rate.")
    print("saved:", json_path)
    print("saved:", csv_path)


if __name__ == "__main__":
    main()
