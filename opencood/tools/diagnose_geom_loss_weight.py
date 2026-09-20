#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only diagnostic for Stage-2 geometry-loss weight.

Measures, on the same forward graph:
  - detection loss (criterion with geom_weight temporarily disabled)
  - geometry loss
  - gradient norms on Stage-2 mean-head and mean-refinement path
  - gradient cosine
  - effective geometry/detection gradient ratios for candidate lambdas

No optimizer.step() is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

import numpy as np
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hypes-yaml", "-y", required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--num-batches", type=int, default=20)
    p.add_argument("--lambdas", nargs="+", type=float,
                   default=[0.1, 0.25, 0.5, 1.0])
    p.add_argument("--out-dir", default="opencood/logs/geom_weight_diag")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=20260917)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_state_dict(obj):
    if not isinstance(obj, dict):
        return obj
    for key in ("model_state_dict", "state_dict", "model_state", "model"):
        if isinstance(obj.get(key), dict):
            return obj[key]
    return obj


def load_checkpoint(model, path, device):
    if not path:
        return
    obj = torch.load(path, map_location=device)
    state = unwrap_state_dict(obj)
    if state and next(iter(state)).startswith("module."):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    print(f"[checkpoint] {path}")


def build_loader(hypes, worker):
    dataset = build_dataset(hypes, visualize=False, train=True)
    kwargs = dict(
        dataset=dataset,
        batch_size=int(hypes["train_params"]["batch_size"]),
        num_workers=worker,
        collate_fn=dataset.collate_batch_train,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    if worker > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 1
    return DataLoader(**kwargs)


def unique_params(modules: Sequence[torch.nn.Module]):
    out = []
    seen = set()
    for module in modules:
        for p in module.parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            out.append(p)
    return out


def probe_groups(model):
    cross = model.cross_agent
    refiner = cross.refiner
    if refiner is None:
        raise RuntimeError("Stage2 refiner is None.")

    mean_heads = [h.mean_head for h in refiner.heads.heads.values()]
    return {
        "mean_head": unique_params(mean_heads),
        "mean_path": unique_params(
            mean_heads + [refiner.fusion, cross.attention, cross.adapter]
        ),
    }


def grad_vec(loss, params, retain_graph):
    grads = torch.autograd.grad(
        loss, params,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    parts = []
    for p, g in zip(params, grads):
        parts.append(
            (torch.zeros_like(p) if g is None else g.detach())
            .float().reshape(-1)
        )
    return torch.cat(parts) if parts else loss.new_zeros((0,), dtype=torch.float32)


def stats(g_det, g_geo):
    nd = float(torch.linalg.vector_norm(g_det))
    ng = float(torch.linalg.vector_norm(g_geo))
    if nd > 0 and ng > 0:
        cos = float(torch.dot(g_det, g_geo) / (nd * ng + 1e-12))
    else:
        cos = float("nan")
    return nd, ng, ng / max(nd, 1e-12), cos


def med(xs: Iterable[float]):
    xs = [x for x in xs if math.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")


def main():
    opt = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    torch.cuda.set_device(opt.gpu_id)
    device = torch.device(f"cuda:{opt.gpu_id}")
    set_seed(opt.seed)

    hypes = yaml_utils.load_yaml(opt.hypes_yaml)
    geom_cfg = hypes["model"]["args"].get("geometry_supervision") or {}
    if not bool(geom_cfg.get("enabled", False)):
        raise RuntimeError("geometry_supervision.enabled must be true.")

    model = train_utils.create_model(hypes).to(device)
    load_checkpoint(model, opt.checkpoint, device)
    model.train()

    criterion = train_utils.create_loss(hypes).to(device)
    configured_weight = float(getattr(criterion, "geom_weight", 0.0))
    criterion.geom_weight = 0.0  # isolate L_det without changing any other loss

    groups = probe_groups(model)
    loader = build_loader(hypes, opt.worker)
    rows = []

    print(f"[configured geom_weight] {configured_weight}")
    print(f"[beta] {float(geom_cfg.get('beta', 1.0))}")

    for bi, batch in enumerate(loader):
        if bi >= opt.num_batches:
            break
        if batch is None:
            continue

        batch = train_utils.to_device(batch, device)
        ego = batch["ego"]
        ego["epoch"] = 0
        model.zero_grad(set_to_none=True)

        with amp.autocast(enabled=opt.amp):
            out = model(ego)
            lgeo = out.get("geom_loss")
            if not torch.is_tensor(lgeo):
                raise RuntimeError("geom_loss missing from model output.")
            ldet = criterion(out, ego["label_dict"])

        row = {
            "batch": bi,
            "det_loss": float(ldet.detach()),
            "geom_loss": float(lgeo.detach()),
        }
        row["geom_over_det_loss"] = row["geom_loss"] / max(row["det_loss"], 1e-12)

        names = list(groups)
        for gi, name in enumerate(names):
            params = groups[name]
            gd = grad_vec(ldet, params, retain_graph=True)
            gg = grad_vec(lgeo, params, retain_graph=(gi != len(names) - 1))
            nd, ng, ratio, cos = stats(gd, gg)
            row[f"{name}_grad_det"] = nd
            row[f"{name}_grad_geo"] = ng
            row[f"{name}_grad_geo_over_det"] = ratio
            row[f"{name}_grad_cosine"] = cos
            for lam in opt.lambdas:
                row[f"{name}_effective_ratio_lambda_{lam:g}"] = lam * ratio

        rows.append(row)
        print(
            f"[{bi:03d}] "
            f"Ldet={row['det_loss']:.5f} "
            f"Lgeo={row['geom_loss']:.5f} | "
            f"mean_path Ggeo/Gdet={row['mean_path_grad_geo_over_det']:.4f} | "
            f"cos={row['mean_path_grad_cosine']:.4f}"
        )

        del out, lgeo, ldet
        torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("No batches processed.")

    out_dir = Path(opt.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "geom_weight_batches.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "n_batches": len(rows),
        "configured_geom_weight": configured_weight,
        "candidate_lambdas": opt.lambdas,
        "det_loss_median": med(r["det_loss"] for r in rows),
        "geom_loss_median": med(r["geom_loss"] for r in rows),
        "geom_over_det_loss_median": med(r["geom_over_det_loss"] for r in rows),
        "groups": {},
    }

    for name in groups:
        nd = med(r[f"{name}_grad_det"] for r in rows)
        ng = med(r[f"{name}_grad_geo"] for r in rows)
        ratio = med(r[f"{name}_grad_geo_over_det"] for r in rows)
        cos = med(r[f"{name}_grad_cosine"] for r in rows)
        g = {
            "grad_det_median": nd,
            "grad_geo_median": ng,
            "grad_geo_over_det_median": ratio,
            "grad_cosine_median": cos,
            "effective_ratios": {str(l): l * ratio for l in opt.lambdas},
        }
        if nd > 0 and ng > 0:
            g["lambda_for_target_ratio"] = {
                "0.10": 0.10 * nd / ng,
                "0.25": 0.25 * nd / ng,
                "0.50": 0.50 * nd / ng,
                "1.00": 1.00 * nd / ng,
            }
        summary["groups"][name] = g

    json_path = out_dir / "geom_weight_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(
        f"median Ldet={summary['det_loss_median']:.6f}, "
        f"Lgeo={summary['geom_loss_median']:.6f}, "
        f"Lgeo/Ldet={summary['geom_over_det_loss_median']:.6f}"
    )
    for name, g in summary["groups"].items():
        print(
            f"[{name}] median Ggeo/Gdet={g['grad_geo_over_det_median']:.6f}, "
            f"cos={g['grad_cosine_median']:.6f}"
        )
        for lam, ratio in g["effective_ratios"].items():
            print(f"  lambda={lam}: effective ratio={ratio:.6f}")
        if "lambda_for_target_ratio" in g:
            print("  suggested-by-gradient scale:")
            for target, lam in g["lambda_for_target_ratio"].items():
                print(f"    target {target}x det grad -> lambda≈{lam:.6f}")

    print(f"\n[wrote] {csv_path}")
    print(f"[wrote] {json_path}")


if __name__ == "__main__":
    main()
