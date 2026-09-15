#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read-only Stage3 black-box diagnostic for AirV2X Gaussian 0911version.

Requires the previously used:
    opencood/tools/analyze_epoch_health.py

Run:
    python opencood/tools/test_stage3_blackbox.py \
      /path/to/run/config.yaml 5 --gpu 0 --n-frames 32

Outputs beside config by default:
    epoch5_stage3_blackbox.json
    epoch5_stage3_blackbox.txt

No production file or checkpoint is modified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opencood.data_utils.datasets import build_dataset
from opencood.models.airv2x_gaussian_0822 import CAMERA_GEOMETRY_KEY, present_camera_agents
from opencood.models.gaussian_modules_0822.interaction.gaussian_context import _blocks_by_batch_agent
from opencood.tools import train_utils
from opencood.tools.analyze_epoch_health import (
    detection_chain,
    get_gt_boxes,
    load_checkpoint,
    load_fixed_batch,
    load_hypes,
    resolve_paths,
)

EPS = 1e-12
IOU = 0.30


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Stage3 black-box scale/distribution diagnostic")
    p.add_argument("config", type=str)
    p.add_argument("epoch", type=int)
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-frames", type=int, default=32)
    p.add_argument("--indices", type=str, default="", help="Optional comma-separated fixed dataset indices; overrides --n-frames.")
    p.add_argument("--out-dir", type=str, default="")
    return p.parse_args()


def tstats(
    x: Optional[torch.Tensor],
    max_quantile_samples: int = 1_000_000,
) -> Dict[str, Any]:
    """
    Full-tensor mean/std/RMS/max, but quantiles use a deterministic subsample
    for very large tensors.

    PyTorch 2.1 CUDA torch.quantile has an input-size limit. AirV2X BEV
    [1,128,200,704] contains 18,022,400 values, which can exceed it.
    """
    if x is None:
        return {"n": 0}
    y = x.detach().float().reshape(-1)
    finite = y[torch.isfinite(y)]
    if finite.numel() == 0:
        return {"n": int(y.numel()), "finite_ratio": 0.0}

    q_src = finite
    if q_src.numel() > int(max_quantile_samples):
        # IMPORTANT: do not use float torch.linspace for indices here.
        # For tensors larger than 2**24, float32 cannot represent every
        # integer exactly and the final index can round to numel() -> OOB.
        n = int(q_src.numel())
        m = int(max_quantile_samples)
        idx = torch.arange(m, device=q_src.device, dtype=torch.long)
        if m > 1:
            idx = torch.div(idx * (n - 1), m - 1, rounding_mode="floor")
        else:
            idx.zero_()
        q_src = q_src[idx]

    q = torch.quantile(
        q_src,
        torch.tensor([0.01, 0.50, 0.99], device=q_src.device),
    )
    return {
        "n": int(y.numel()),
        "finite_ratio": float(finite.numel() / max(y.numel(), 1)),
        "mean": float(finite.mean()),
        "std": float(finite.std()) if finite.numel() > 1 else 0.0,
        "rms": float(finite.square().mean().sqrt()),
        "max_abs": float(finite.abs().max()),
        "p01": float(q[0]),
        "median": float(q[1]),
        "p99": float(q[2]),
        "quantile_samples": int(q_src.numel()),
    }


def rms(x: torch.Tensor) -> float:
    y = x.detach().float().reshape(-1)
    y = y[torch.isfinite(y)]
    return float(y.square().mean().sqrt()) if y.numel() else 0.0


def tensor_diff(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    d = a.detach().float() - b.detach().float()
    return {
        "max_abs": float(d.abs().max()) if d.numel() else 0.0,
        "rms": rms(d),
        "relative_rms": rms(d) / (rms(b) + EPS),
    }


def stage2_forward(model: nn.Module, ego: Dict[str, Any]) -> Mapping[str, Any]:
    for agent in present_camera_agents(ego):
        model._encode_agent(agent, ego)
    gaussians = model.initializer.forward(ego)
    s1: Dict[str, Any] = {}
    for agent, gs in gaussians.items():
        payload = ego[agent]
        s1[agent] = model.intra_view(gs, payload["f90"], payload[CAMERA_GEOMETRY_KEY], agent=agent)
    sources = {
        agent: {"features": ego[agent]["f90"], "geometry": ego[agent][CAMERA_GEOMETRY_KEY]}
        for agent in s1
    }
    return model.cross_agent(s1, sources)


def _adapter_forward_scaled(adapter: nn.Module, feature: torch.Tensor, agent: str,
                            raw_scale: float, standard_raw_residual: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    mlp = adapter.adapters[str(agent).lower()]
    x_in = adapter.input_norm(feature) if adapter.input_norm is not None else feature
    raw = mlp(x_in) * float(raw_scale)
    if standard_raw_residual:
        return feature + raw, raw
    if adapter.delta_norm is None or adapter.gamma is None:
        return feature + raw, raw
    delta = adapter.delta_norm(raw)
    gamma = adapter.gamma.to(device=feature.device, dtype=feature.dtype)
    return feature + gamma * delta, raw


def run_stage3_variant(stage3: nn.Module, gaussians_by_agent: Mapping[str, Any], mode: str,
                       scale_target: Optional[str] = None, scale_value: float = 1.0):
    pooled = {agent: stage3.pool(gs) for agent, gs in gaussians_by_agent.items() if gs.n_gaussians > 0}
    sample = next(iter(gaussians_by_agent.values()))
    if not pooled:
        empty = sample.__class__.empty(sample.mean.device, sample.feature.dtype, stage3.feature_dim)
        bev = stage3.splat(empty)
        return empty, bev, {"empty": True}

    merged, agent_slices = stage3._merge(pooled)
    pooled_feature = merged.feature
    C = int(stage3.feature_dim)

    if mode == "standard_full":
        split = None
        self_feature = pooled_feature
        cross_feature = pooled_feature
    else:
        split_scale = float(scale_value) if mode == "scaled" and scale_target == "split" else 1.0
        split = stage3.split_mlp(pooled_feature) * split_scale
        self_feature, cross_feature = split[:, :C], split[:, C:]

    batch = merged.batch_index
    n_batch = int(batch.max().item()) + 1
    groups = _blocks_by_batch_agent(batch, n_batch, agent_slices)

    self_out = self_feature.clone()
    for _b, _agent, rows in groups:
        x = self_feature[rows].unsqueeze(0)
        for block in stage3.self_blocks:
            x = block(x)
        self_out[rows] = x.squeeze(0)

    adapted = cross_feature.clone()
    raw_adapter_parts: List[torch.Tensor] = []
    adapter_scale = float(scale_value) if mode == "scaled" and scale_target == "adapter" else 1.0
    for agent, (start, end) in agent_slices.items():
        if end <= start:
            continue
        out_a, raw_a = _adapter_forward_scaled(
            stage3.adapter, cross_feature[start:end], agent,
            raw_scale=adapter_scale,
            standard_raw_residual=(mode == "standard_full"),
        )
        adapted[start:end] = out_a
        raw_adapter_parts.append(raw_a)
    raw_adapter_delta = torch.cat(raw_adapter_parts, dim=0) if raw_adapter_parts else pooled_feature.new_empty((0, C))

    cross_out = cross_feature.clone()
    for _b, agent, rows in groups:
        others = [(a2, r2) for b2, a2, r2 in groups if b2 == _b and a2 != agent]
        if not others:
            continue
        query = adapted[rows].unsqueeze(0)
        kv = torch.cat([adapted[r2] for _a2, r2 in others], dim=0).unsqueeze(0)
        for block in stage3.cross_blocks:
            query = block(query, kv)
        cross_out[rows] = query.squeeze(0)

    fusion_input = torch.cat([
        stage3.self_fusion_norm(self_out),
        stage3.cross_fusion_norm(cross_out),
    ], dim=-1)
    fusion_scale = float(scale_value) if mode == "scaled" and scale_target == "fusion" else 1.0
    raw_fusion_delta = stage3.fusion_mlp(fusion_input) * fusion_scale

    if mode in ("standard_final", "standard_full"):
        applied_delta = raw_fusion_delta
    else:
        normed = stage3.fusion_delta_norm(raw_fusion_delta)
        gamma = stage3.fusion_gamma.to(device=normed.device, dtype=normed.dtype)
        applied_delta = gamma * normed

    fused = pooled_feature + applied_delta
    out = merged.replace_feature(fused)
    bev = stage3.splat(out)

    rec = {
        "mode": mode,
        "scale_target": scale_target,
        "scale_value": float(scale_value),
        "counts": {k: int(v.n_gaussians) for k, v in pooled.items()},
        "pooled": tstats(pooled_feature),
        "split": tstats(split),
        "self_feature": tstats(self_feature),
        "cross_feature": tstats(cross_feature),
        "self_out": tstats(self_out),
        "cross_out": tstats(cross_out),
        "adapter_raw_delta": tstats(raw_adapter_delta),
        "fusion_input": tstats(fusion_input),
        "fusion_raw_delta": tstats(raw_fusion_delta),
        "applied_delta": tstats(applied_delta),
        "final": tstats(fused),
        "bev": tstats(bev),
        "ratios": {
            "split_to_pooled_rms": rms(split) / (rms(pooled_feature) + EPS) if split is not None else None,
            "adapter_raw_to_cross_rms": rms(raw_adapter_delta) / (rms(cross_feature) + EPS) if raw_adapter_delta.numel() else None,
            "fusion_raw_to_pooled_rms": rms(raw_fusion_delta) / (rms(pooled_feature) + EPS),
            "applied_to_pooled_rms": rms(applied_delta) / (rms(pooled_feature) + EPS),
            "final_to_pooled_rms": rms(fused) / (rms(pooled_feature) + EPS),
        },
    }
    return out, bev, rec


def heads_from_bev(model: nn.Module, ego: Dict[str, Any], bev: torch.Tensor) -> Dict[str, torch.Tensor]:
    ego["spatial_features"] = bev
    data = model.backbone(ego)
    fused = model.shrink_conv(data["spatial_features_2d"])
    return {"psm": model.cls_head(fused), "rm": model.reg_head(fused), "obj": model.obj_head(fused)}


def final_tp_fp(dataset: Any, ego: Dict[str, Any], output: Mapping[str, torch.Tensor], gt_boxes: Optional[torch.Tensor]):
    det = detection_chain(dataset, ego, output, gt_boxes)
    boxes, scores = det.get("final_boxes"), det.get("final_scores")
    n_gt = int(gt_boxes.shape[0]) if gt_boxes is not None else 0
    if gt_boxes is None or boxes is None or scores is None:
        return det, 0, 0, n_gt
    from opencood.utils import eval_utils_opv2v as eval_utils
    result = {IOU: {"tp": [], "fp": [], "gt": 0, "score": []}}
    eval_utils.caluclate_tp_fp(boxes, scores, gt_boxes, result, IOU)
    return det, int(sum(result[IOU]["tp"])), int(sum(result[IOU]["fp"])), n_gt


def variant_key(mode: str, target: Optional[str] = None, value: float = 1.0) -> str:
    return mode if mode != "scaled" else f"scale_{target}_{value:g}"


def choose_indices(n_dataset: int, n_frames: int, text: str) -> List[int]:
    if text.strip():
        out = [int(x.strip()) for x in text.split(",") if x.strip()]
    else:
        n = min(max(int(n_frames), 1), n_dataset)
        out = np.linspace(0, n_dataset - 1, num=n, dtype=int).tolist()
    for idx in out:
        if idx < 0 or idx >= n_dataset:
            raise IndexError(f"dataset index {idx} outside [0,{n_dataset-1}]")
    return list(dict.fromkeys(out))


def aggregate_metric(rows: List[Mapping[str, Any]], path: Tuple[str, ...]) -> Dict[str, Any]:
    vals = []
    for row in rows:
        cur: Any = row
        ok = True
        for k in path:
            if not isinstance(cur, Mapping) or k not in cur:
                ok = False
                break
            cur = cur[k]
        if ok and cur is not None:
            try:
                v = float(cur)
                if math.isfinite(v):
                    vals.append(v)
            except Exception:
                pass
    if not vals:
        return {"n": 0}
    a = np.asarray(vals, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


def stage3_grad_l2(stage3: nn.Module) -> float:
    sq = 0.0
    for p in stage3.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        g = torch.where(torch.isfinite(g), g, torch.zeros_like(g))
        sq += float((g * g).sum())
    return math.sqrt(max(sq, 0.0))


@contextmanager
def patched_stage3_forward(stage3: nn.Module, mode: str):
    original = stage3.forward
    def _forward(self, gaussians_by_agent):
        out, bev, _ = run_stage3_variant(self, gaussians_by_agent, mode=mode)
        return out, bev
    stage3.forward = types.MethodType(_forward, stage3)
    try:
        yield
    finally:
        stage3.forward = original


def backward_one(model: nn.Module, criterion: nn.Module, batch: Dict[str, Any], epoch: int, mode: str) -> Dict[str, Any]:
    model.zero_grad(set_to_none=True)
    ego = batch["ego"]
    ego["epoch"] = int(epoch)
    model.eval()
    if mode == "current":
        output = model(ego)
    else:
        with patched_stage3_forward(model.stage3, mode):
            output = model(ego)
    loss = criterion(output, ego["label_dict"])
    loss.backward()
    rec = {"loss": float(loss.detach()) if torch.isfinite(loss) else None, "stage3_grad_l2": stage3_grad_l2(model.stage3)}
    model.zero_grad(set_to_none=True)
    return rec


def summarize_variant(rows: List[Mapping[str, Any]], tp: int, fp: int, gt: int) -> Dict[str, Any]:
    return {
        "n_frames": len(rows),
        "tp_iou03": int(tp),
        "fp_iou03": int(fp),
        "gt": int(gt),
        "precision_iou03": float(tp / max(tp + fp, 1)),
        "recall_iou03": float(tp / max(gt, 1)),
        "pooled_rms": aggregate_metric(rows, ("stage3", "pooled", "rms")),
        "split_to_pooled_rms": aggregate_metric(rows, ("stage3", "ratios", "split_to_pooled_rms")),
        "adapter_raw_to_cross_rms": aggregate_metric(rows, ("stage3", "ratios", "adapter_raw_to_cross_rms")),
        "fusion_raw_to_pooled_rms": aggregate_metric(rows, ("stage3", "ratios", "fusion_raw_to_pooled_rms")),
        "applied_to_pooled_rms": aggregate_metric(rows, ("stage3", "ratios", "applied_to_pooled_rms")),
        "final_to_pooled_rms": aggregate_metric(rows, ("stage3", "ratios", "final_to_pooled_rms")),
        "final_count": aggregate_metric(rows, ("final_count",)),
    }


def main() -> None:
    opt = parse_args()
    config, model_dir, ckpt = resolve_paths(opt.config, opt.epoch)
    out_dir = Path(opt.out_dir).expanduser().resolve() if opt.out_dir else model_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"epoch{opt.epoch}_stage3_blackbox.json"
    out_txt = out_dir / f"epoch{opt.epoch}_stage3_blackbox.txt"

    device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    hypes = load_hypes(config, opt.split)
    dataset = build_dataset(hypes, visualize=False, train=False)
    indices = choose_indices(len(dataset), opt.n_frames, opt.indices)
    model = train_utils.create_model(hypes).to(device)
    load_info = load_checkpoint(model, ckpt, device)
    model.eval()

    variants: List[Tuple[str, Optional[str], float]] = [
        ("current", None, 1.0),
        ("scaled", "adapter", 0.1), ("scaled", "adapter", 1.0), ("scaled", "adapter", 10.0),
        ("scaled", "fusion", 0.1), ("scaled", "fusion", 1.0), ("scaled", "fusion", 10.0),
        ("scaled", "split", 0.1), ("scaled", "split", 1.0), ("scaled", "split", 10.0),
        ("standard_final", None, 1.0),
        ("standard_full", None, 1.0),
    ]
    rows: Dict[str, List[Dict[str, Any]]] = {variant_key(*v): [] for v in variants}
    totals = {k: {"tp": 0, "fp": 0, "gt": 0} for k in rows}
    equivalence = {"scale_adapter_1": [], "scale_fusion_1": [], "scale_split_1": []}

    print(f"[blackbox] config={config}")
    print(f"[blackbox] checkpoint={ckpt}")
    print(f"[blackbox] split={opt.split} frames={indices}")

    with torch.no_grad():
        for fi, idx in enumerate(indices, 1):
            batch, frame = load_fixed_batch(dataset, idx, device)
            ego = batch["ego"]
            ego["epoch"] = int(opt.epoch)
            s2 = stage2_forward(model, ego)
            gt_boxes, _ = get_gt_boxes(dataset, {"ego": ego})
            frame_outputs: Dict[str, Dict[str, torch.Tensor]] = {}

            for mode, target, value in variants:
                key = variant_key(mode, target, value)
                _out_g, bev, s3rec = run_stage3_variant(model.stage3, s2, mode, target, value)
                output = heads_from_bev(model, ego, bev)
                det, tp, fp, ngt = final_tp_fp(dataset, ego, output, gt_boxes)
                totals[key]["tp"] += tp
                totals[key]["fp"] += fp
                totals[key]["gt"] += ngt
                rows[key].append({
                    "index": int(idx), "frame": frame, "stage3": s3rec,
                    "final_count": int(det.get("final", {}).get("count", 0)),
                    "recall_iou03": det.get("final", {}).get("recall_iou03"),
                    "output": {"psm": tstats(output["psm"]), "rm": tstats(output["rm"]), "obj": tstats(output["obj"])},
                })
                frame_outputs[key] = {"bev": bev.detach(), "psm": output["psm"].detach(), "rm": output["rm"].detach(), "obj": output["obj"].detach()}

            cur = frame_outputs["current"]
            for eq_key in equivalence:
                got = frame_outputs[eq_key]
                equivalence[eq_key].append({name: tensor_diff(got[name], cur[name]) for name in ("bev", "psm", "rm", "obj")})

            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[blackbox] {fi}/{len(indices)} idx={idx}")

    summary = {key: summarize_variant(rows[key], totals[key]["tp"], totals[key]["fp"], totals[key]["gt"]) for key in rows}

    criterion = train_utils.create_loss(hypes)
    grad_batch_cur, _ = load_fixed_batch(dataset, indices[0], device)
    backward_current = backward_one(model, criterion, grad_batch_cur, opt.epoch, "current")
    grad_batch_std, _ = load_fixed_batch(dataset, indices[0], device)
    backward_standard_final = backward_one(model, criterion, grad_batch_std, opt.epoch, "standard_final")

    def max_eq_rel(eq_key: str) -> float:
        vals = []
        for row in equivalence[eq_key]:
            for name in ("bev", "psm", "rm", "obj"):
                vals.append(float(row[name]["relative_rms"]))
        return max(vals) if vals else float("nan")

    eq_max = {k: max_eq_rel(k) for k in equivalence}
    verdict = {
        "custom_scale1_matches_current": {k: bool(v < 1e-6) for k, v in eq_max.items()},
        "scale1_max_relative_rms_diff_vs_current": eq_max,
        "interpretation": (
            "Adapter/fusion raw-scale degeneracy is supported if raw internal RMS scales with 0.1x/10x "
            "but final/BEV/detection barely move. Split is not expected to be exactly invariant because it "
            "also scales the identity stream entering Transformer residual blocks."
        ),
    }

    payload = {
        "meta": {"config": str(config), "checkpoint": str(ckpt), "epoch": int(opt.epoch), "split": opt.split,
                 "device": str(device), "indices": indices, "checkpoint_load": load_info},
        "summary": summary,
        "equivalence_to_current": equivalence,
        "backward": {
            "current": backward_current,
            "standard_final": backward_standard_final,
            "stage3_grad_ratio_standard_over_current": backward_standard_final["stage3_grad_l2"] / (backward_current["stage3_grad_l2"] + EPS),
        },
        "verdict": verdict,
        "per_frame": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["Stage3 Black-box Diagnostic", "=" * 72, f"checkpoint: {ckpt}", f"frames: {indices}", ""]
    lines.append("variant                     precision@.3  recall@.3   final#   rawFusion/pooled  applied/pooled  final/pooled")
    for key, s in summary.items():
        lines.append(
            f"{key:27s} {s['precision_iou03']:.4f}       {s['recall_iou03']:.4f}     "
            f"{s['final_count'].get('mean', float('nan')):7.2f}   "
            f"{s['fusion_raw_to_pooled_rms'].get('mean', float('nan')):8.4f}          "
            f"{s['applied_to_pooled_rms'].get('mean', float('nan')):8.4f}        "
            f"{s['final_to_pooled_rms'].get('mean', float('nan')):8.4f}"
        )
    lines += ["", "Scale=1 custom path equivalence to production CURRENT"]
    for key, v in eq_max.items():
        lines.append(f"  {key}: max relative RMS diff = {v:.3e}")
    lines += ["", "Backward, one fixed frame, no optimizer step"]
    lines.append(f"  CURRENT:        loss={backward_current['loss']:.6f} stage3_grad_l2={backward_current['stage3_grad_l2']:.6e}")
    lines.append(f"  STANDARD_FINAL: loss={backward_standard_final['loss']:.6f} stage3_grad_l2={backward_standard_final['stage3_grad_l2']:.6e}")
    lines.append(f"  grad ratio std/current = {payload['backward']['stage3_grad_ratio_standard_over_current']:.4f}")
    lines += ["", verdict["interpretation"], "",
              "IMPORTANT: STANDARD_FULL uses the current checkpoint under a different parameterization; its AP is diagnostic only, not a fair trained-model AP."]
    report = "\n".join(lines) + "\n"
    out_txt.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"[blackbox] wrote {out_json}")
    print(f"[blackbox] wrote {out_txt}")


if __name__ == "__main__":
    main()
