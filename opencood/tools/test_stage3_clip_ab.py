#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage3 clipping A/B test.

Requires:
    opencood/tools/test_stage3_wd_ab.py

Both variants disable weight decay ONLY for Stage3 core 2-D matrices.
The only intended difference is gradient clipping:

    GPU 4: max_norm = 1
    GPU 5: max_norm = 20

Example:
python opencood/tools/test_stage3_clip_ab.py \
    --config /path/to/airv2x_gaussian_0822.yaml \
    --gpu-a 4 --gpu-b 5 \
    --epochs 1 --steps-per-epoch 100
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opencood.tools.test_stage3_wd_ab import run_variant


WATCH = (
    "stage3.self_blocks.0.q_proj.weight",
    "stage3.cross_blocks.0.q_proj.weight",
    "stage3.self_blocks.0.ffn.2.weight",
    "stage3.cross_blocks.0.ffn.2.weight",
    "stage3.fusion_mlp.weight",
)


def parse_args():
    p = argparse.ArgumentParser(description="Stage3 clip=1 vs clip=20 A/B test")
    p.add_argument("--config", required=True)
    p.add_argument("--gpu-a", type=int, default=4)
    p.add_argument("--gpu-b", type=int, default=5)
    p.add_argument("--clip-a", type=float, default=1.0)
    p.add_argument("--clip-b", type=float, default=20.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260911)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--out-dir", default="")
    return p.parse_args()


def final_row(run):
    hist = run.get("history", [])
    return hist[-1] if hist else {}


def last_grad_row(run):
    for row in reversed(run.get("history", [])):
        if row.get("pre_clip_grad_norm") is not None:
            return row
    return {}


def retention(run, name):
    return final_row(run).get("weights", {}).get(name, {}).get("retention")


def gamma_abs(run):
    return final_row(run).get("fusion_gamma", {}).get("mean_abs")


def stage3_grad(run):
    return last_grad_row(run).get("stage3_grad_l2")


def preclip(run):
    return last_grad_row(run).get("pre_clip_grad_norm")


def clip_factor(run):
    return last_grad_row(run).get("clip_factor")


def loss(run):
    return last_grad_row(run).get("loss")


def fmt(x):
    if x is None:
        return "N/A"
    x = float(x)
    if not math.isfinite(x):
        return str(x)
    if abs(x) >= 1e4 or (abs(x) > 0 and abs(x) < 1e-4):
        return f"{x:.3e}"
    return f"{x:.4f}"


def render(a, b, clip_a, clip_b):
    same_init = a.get("stage3_initial_hash") == b.get("stage3_initial_hash")
    lines = [
        "Stage3 Clipping A/B Test",
        "=" * 72,
        f"A GPU{a.get('gpu')}: Stage3 core WD=0, max_norm={clip_a}",
        f"B GPU{b.get('gpu')}: Stage3 core WD=0, max_norm={clip_b}",
        f"Identical Stage3 initial hash: {same_init}",
        "",
        "Final parameter retention",
    ]
    for name in WATCH:
        lines.append(
            f"  {name}: A={fmt(retention(a, name))}, B={fmt(retention(b, name))}"
        )

    lines += [
        "",
        "Fusion / gradient",
        f"  fusion_gamma mean|x|: A={fmt(gamma_abs(a))}, B={fmt(gamma_abs(b))}",
        f"  Stage3 grad L2:       A={fmt(stage3_grad(a))}, B={fmt(stage3_grad(b))}",
        f"  pre-clip grad L2:     A={fmt(preclip(a))}, B={fmt(preclip(b))}",
        f"  clip factor:          A={fmt(clip_factor(a))}, B={fmt(clip_factor(b))}",
        f"  last logged loss:     A={fmt(loss(a))}, B={fmt(loss(b))}",
        "",
        "AUTOMATED INTERPRETATION",
        "-" * 72,
    ]

    qa = retention(a, "stage3.self_blocks.0.q_proj.weight")
    qb = retention(b, "stage3.self_blocks.0.q_proj.weight")
    ga = gamma_abs(a)
    gb = gamma_abs(b)
    sga = stage3_grad(a)
    sgb = stage3_grad(b)

    if not same_init:
        verdict = "INVALID_AB_INIT_MISMATCH"
        detail = "Initial Stage3 hashes differ; do not interpret this comparison."
    elif qb is not None and qa is not None and qb >= 0.5 and qa >= 0.5:
        if sga and sgb and sgb > 2.0 * sga:
            verdict = "CLIP1_STRONGLY_SUPPRESSES_STAGE3_LEARNING"
            detail = (
                "Both keep Stage3 weights alive because WD=0, but max_norm=20 "
                "preserves substantially larger Stage3 task gradients."
            )
        else:
            verdict = "CLIP1_NOT_A_MAJOR_STAGE3_COLLAPSE_DRIVER_AFTER_WD_FIX"
            detail = (
                "With Stage3 core WD removed, both clipping settings keep the "
                "core matrices healthy over this test window."
            )
    elif qb is not None and qb >= 0.5 and (qa is None or qa < 0.5):
        verdict = "CLIP20_MATERIALLY_IMPROVES_STAGE3_STABILITY"
        detail = (
            "Stage3 survives with max_norm=20 but degrades under max_norm=1."
        )
    else:
        verdict = "STAGE3_STILL_UNHEALTHY_WITH_CLIP20"
        detail = (
            "Even max_norm=20 does not preserve Stage3; inspect task utility, "
            "optimizer dynamics, and loss scaling before full training."
        )

    if gb is not None and gb < 0.02:
        detail += " fusion_gamma is also approaching bypass in the clip20 branch."

    lines += [verdict, detail, ""]
    return "\n".join(lines) + "\n"


def main():
    opt = parse_args()
    config = Path(opt.config).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(config)

    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    out_dir = (
        Path(opt.out_dir).expanduser().resolve()
        if opt.out_dir
        else config.parent / f"stage3_clip_ab_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    a_json = out_dir / f"clip{opt.clip_a:g}_gpu{opt.gpu_a}.json"
    b_json = out_dir / f"clip{opt.clip_b:g}_gpu{opt.gpu_b}.json"

    print("=" * 72)
    print("Stage3 clip A/B")
    print(f"config: {config}")
    print(f"A: GPU {opt.gpu_a}, Stage3 WD=0, clip={opt.clip_a}")
    print(f"B: GPU {opt.gpu_b}, Stage3 WD=0, clip={opt.clip_b}")
    print(f"epochs={opt.epochs}, steps/epoch={opt.steps_per_epoch}")
    print(f"output: {out_dir}")
    print("=" * 72)

    common = dict(
        config_path=str(config),
        epochs=int(opt.epochs),
        steps_per_epoch=int(opt.steps_per_epoch),
        workers=int(opt.workers),
        seed=int(opt.seed),
        log_every=int(opt.log_every),
        use_amp=bool(opt.amp),
    )

    ctx = mp.get_context("spawn")
    pa = ctx.Process(
        target=run_variant,
        kwargs=dict(
            variant="no_stage3_core_wd",
            gpu=int(opt.gpu_a),
            clip_norm=float(opt.clip_a),
            out_json=str(a_json),
            **common,
        ),
    )
    pb = ctx.Process(
        target=run_variant,
        kwargs=dict(
            variant="no_stage3_core_wd",
            gpu=int(opt.gpu_b),
            clip_norm=float(opt.clip_b),
            out_json=str(b_json),
            **common,
        ),
    )

    pa.start()
    pb.start()
    pa.join()
    pb.join()

    if not a_json.exists() or not b_json.exists():
        raise RuntimeError(
            f"Missing A/B JSON. exitcodes: A={pa.exitcode}, B={pb.exitcode}"
        )

    a = json.loads(a_json.read_text(encoding="utf-8"))
    b = json.loads(b_json.read_text(encoding="utf-8"))

    summary = render(a, b, float(opt.clip_a), float(opt.clip_b))
    summary_path = out_dir / "stage3_clip_ab_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    print("\n" + summary)
    print(f"Saved: {summary_path}")
    print(f"Saved: {a_json}")
    print(f"Saved: {b_json}")

    if pa.exitcode != 0 or pb.exitcode != 0:
        raise SystemExit(
            f"Worker failed: A={pa.exitcode}, B={pb.exitcode}. "
            "Inspect JSON traceback."
        )


if __name__ == "__main__":
    main()
