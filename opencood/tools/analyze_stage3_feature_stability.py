"""Diagnose Stage-3 feature amplification without changing production code.

Runs the real ``GaussianContextInteraction`` weights through an isolated
instrumented forward that records activation / residual / parameter /
gradient statistics. Splat is skipped for feature probes and still used
when a detection loss is needed for backward checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml import yaml_utils
from opencood.models.gaussian_modules_0822.attention.interaction_block import (
    GaussianInteractionBlock,
)
from opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet
from opencood.models.gaussian_modules_0822.interaction.gaussian_context import (
    GaussianContextInteraction,
    _blocks_by_batch_agent,
)
from opencood.tools import train_utils

EPS = 1.0e-12
RMS_JUMP = 3.0
MAXABS_JUMP = 5.0
LOGIT_QUERIES = 32
DEFAULT_MODEL_DIR = (
    ROOT
    / "opencood/logs/airv2x_gaussian_0822_camera"
    / "v45_scale_asym_2026_09_10_13_37_23"
)


def _finite(x: torch.Tensor) -> torch.Tensor:
    return x[torch.isfinite(x)]


def tensor_stats(name: str, t: torch.Tensor) -> Dict[str, Any]:
    """Compact finite-aware statistics for one tensor."""
    x = t.detach().reshape(-1).float()
    n = int(x.numel())
    xf = _finite(x)
    nf = int(xf.numel())
    rec: Dict[str, Any] = {
        "name": name,
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "numel": n,
        "finite_ratio": (nf / n) if n else 1.0,
        "mean": None,
        "std": None,
        "mean_abs": None,
        "max_abs": None,
        "rms": None,
        "p90_abs": None,
        "p99_abs": None,
    }
    if nf == 0:
        return rec
    rec["mean"] = float(xf.mean())
    rec["std"] = float(xf.std()) if nf > 1 else 0.0
    rec["mean_abs"] = float(xf.abs().mean())
    rec["max_abs"] = float(xf.abs().max())
    rec["rms"] = float(xf.pow(2).mean().sqrt())
    rec["p90_abs"] = float(torch.quantile(xf.abs(), 0.90))
    rec["p99_abs"] = float(torch.quantile(xf.abs(), 0.99))
    return rec


def residual_ratio(delta: torch.Tensor, base: torch.Tensor) -> Dict[str, float]:
    """RMS and max-abs residual ratios ``delta / base``."""
    d = tensor_stats("delta", delta)
    b = tensor_stats("base", base)
    d_rms = float(d["rms"] or 0.0)
    b_rms = float(b["rms"] or 0.0)
    d_max = float(d["max_abs"] or 0.0)
    b_max = float(b["max_abs"] or 0.0)
    return {
        "rms_ratio": d_rms / (b_rms + EPS),
        "max_ratio": d_max / (b_max + EPS),
        "delta_rms": d_rms,
        "base_rms": b_rms,
        "delta_max_abs": d_max,
        "base_max_abs": b_max,
    }


def jump_flags(src: Dict[str, Any], dst: Dict[str, Any]) -> Dict[str, Any]:
    """Flag a magnitude jump from ``src`` stats to ``dst`` stats."""
    s_rms = float(src.get("rms") or 0.0)
    d_rms = float(dst.get("rms") or 0.0)
    s_max = float(src.get("max_abs") or 0.0)
    d_max = float(dst.get("max_abs") or 0.0)
    rms_r = d_rms / (s_rms + EPS)
    max_r = d_max / (s_max + EPS)
    return {
        "from": src.get("name"),
        "to": dst.get("name"),
        "rms_ratio": rms_r,
        "max_ratio": max_r,
        "flagged": bool(rms_r > RMS_JUMP or max_r > MAXABS_JUMP),
    }


def module_param_stats(module: nn.Module, prefix: str) -> List[Dict[str, Any]]:
    """Weight / bias magnitude stats for Linear and LayerNorm leaves."""
    rows: List[Dict[str, Any]] = []
    for name, child in module.named_modules():
        if not isinstance(child, (nn.Linear, nn.LayerNorm)):
            continue
        full = f"{prefix}.{name}" if name else prefix
        for pname, p in child.named_parameters(recurse=False):
            pf = p.detach().float()
            rows.append(
                {
                    "module": full,
                    "param": pname,
                    "shape": list(p.shape),
                    "abs_max": float(pf.abs().max()),
                    "mean_abs": float(pf.abs().mean()),
                    "frobenius": float(torch.linalg.vector_norm(pf)),
                }
            )
    return rows


def attention_logit_subset(
    q: torch.Tensor,
    k: torch.Tensor,
    n_queries: int = LOGIT_QUERIES,
) -> Dict[str, Any]:
    """Logits ``Q K^T / sqrt(d)`` on a deterministic query subset.

    Args:
        q: ``[B, H, Lq, D]``.
        k: ``[B, H, Lk, D]``.
        n_queries: Number of query indices, evenly spaced.

    Returns:
        Logit stats and mean attention entropy over the subset.
    """
    _b, _h, n_q, dim = q.shape
    n_use = min(int(n_queries), int(n_q))
    idx = torch.linspace(0, n_q - 1, n_use, device=q.device).long()
    q_s = q[:, :, idx, :]
    scale = 1.0 / math.sqrt(float(dim))
    logits = torch.matmul(q_s.float(), k.float().transpose(-2, -1)) * scale
    xf = _finite(logits.reshape(-1))
    rec: Dict[str, Any] = {
        "n_queries": n_use,
        "n_keys": int(k.shape[2]),
        "heads": int(q.shape[1]),
        "mean": None,
        "std": None,
        "max": None,
        "min": None,
        "entropy_mean": None,
    }
    if xf.numel() == 0:
        return rec
    rec["mean"] = float(xf.mean())
    rec["std"] = float(xf.std()) if xf.numel() > 1 else 0.0
    rec["max"] = float(xf.max())
    rec["min"] = float(xf.min())
    attn = torch.softmax(logits, dim=-1)
    ent = -(attn.clamp_min(EPS) * attn.clamp_min(EPS).log()).sum(dim=-1)
    rec["entropy_mean"] = float(_finite(ent.reshape(-1)).mean()) if ent.numel() else None
    return rec


def instrumented_block(
    block: GaussianInteractionBlock,
    query: torch.Tensor,
    context: Optional[torch.Tensor],
    prefix: str,
    activations: List[Dict[str, Any]],
    residuals: Dict[str, Dict[str, float]],
    logits_store: Dict[str, Any],
    collect_logits: bool,
) -> torch.Tensor:
    """Run one production attention block while recording internals.

    Duplicates ``GaussianInteractionBlock.forward`` math using the same
    parameters. Does not modify the module.
    """
    if query.numel() == 0:
        return query
    squeeze = query.dim() == 2
    if squeeze:
        query = query.unsqueeze(0)
    if context is None:
        context = query
    elif context.dim() == 2:
        context = context.unsqueeze(0)

    activations.append(tensor_stats(f"{prefix}.block_input", query))
    q_in = block.query_norm(query)
    kv_in = block.context_norm(context)
    activations.append(tensor_stats(f"{prefix}.query_ln_out", q_in))
    activations.append(tensor_stats(f"{prefix}.context_ln_out", kv_in))
    batch, n_q, dim = q_in.shape
    n_kv = kv_in.shape[1]
    q = block.q_proj(q_in).view(batch, n_q, block.heads, block.head_dim).transpose(1, 2)
    k = block.k_proj(kv_in).view(batch, n_kv, block.heads, block.head_dim).transpose(1, 2)
    v = block.v_proj(kv_in).view(batch, n_kv, block.heads, block.head_dim).transpose(1, 2)
    activations.append(tensor_stats(f"{prefix}.Q", q))
    activations.append(tensor_stats(f"{prefix}.K", k))
    activations.append(tensor_stats(f"{prefix}.V", v))
    if collect_logits and prefix not in logits_store:
        logits_store[prefix] = attention_logit_subset(q, k)

    attn = F.scaled_dot_product_attention(q, k, v)
    activations.append(tensor_stats(f"{prefix}.attn_pre_out_proj", attn))
    attn_flat = attn.transpose(1, 2).reshape(batch, n_q, dim)
    attn_out = block.out_proj(attn_flat)
    activations.append(tensor_stats(f"{prefix}.attn_post_out_proj", attn_out))
    residuals[f"{prefix}.attn"] = residual_ratio(attn_out, query)
    x1 = query + attn_out
    activations.append(tensor_stats(f"{prefix}.after_attn_residual", x1))

    ffn_in = block.ffn_norm(x1)
    activations.append(tensor_stats(f"{prefix}.ffn_ln_out", ffn_in))
    fc1 = block.ffn[0](ffn_in)
    gelu = block.ffn[1](fc1)
    fc2 = block.ffn[2](gelu)
    activations.append(tensor_stats(f"{prefix}.ffn_fc1", fc1))
    activations.append(tensor_stats(f"{prefix}.ffn_gelu", gelu))
    activations.append(tensor_stats(f"{prefix}.ffn_out", fc2))
    residuals[f"{prefix}.ffn"] = residual_ratio(fc2, x1)
    out = x1 + fc2
    activations.append(tensor_stats(f"{prefix}.after_ffn_residual", out))
    if squeeze:
        out = out.squeeze(0)
    return out


def instrumented_stage3(
    stage3: GaussianContextInteraction,
    gaussians_by_agent: Dict[str, GaussianSet],
    skip_splat: bool,
    collect_logits: bool,
    self_keep: Optional[int] = None,
) -> Tuple[Optional[GaussianSet], Optional[torch.Tensor], Dict[str, Any]]:
    """Instrumented Stage-3 core. Stops before splat when ``skip_splat``.

    Args:
        stage3: Production Stage-3 module.
        gaussians_by_agent: Stage-2 Gaussians.
        skip_splat: If True, do not call BEV splat.
        collect_logits: Record subset attention logits.
        self_keep: If set, keep only the first N vehicle self tokens.

    Returns:
        ``(out_gaussians, bev_or_none, probe_dict)``.
    """
    activations: List[Dict[str, Any]] = []
    residuals: Dict[str, Dict[str, float]] = {}
    logits_store: Dict[str, Any] = {}
    dtypes: Dict[str, str] = {}
    pooled = {
        agent: stage3.pool(gs)
        for agent, gs in gaussians_by_agent.items()
        if gs.n_gaussians > 0
    }
    probe: Dict[str, Any] = {
        "pooled_counts": {a: gs.n_gaussians for a, gs in pooled.items()},
        "autocast": bool(torch.is_autocast_enabled()),
    }
    if not pooled:
        probe["activations"] = activations
        probe["residuals"] = residuals
        return None, None, probe

    for agent, gs in pooled.items():
        activations.append(tensor_stats(f"pooled_feature/{agent}", gs.feature))
    merged, agent_slices = stage3._merge(pooled)
    pooled_feature = merged.feature
    activations.append(tensor_stats("pooled_feature", pooled_feature))
    dtypes["pooled_feature"] = str(pooled_feature.dtype)

    split = stage3.split_mlp(pooled_feature)
    activations.append(tensor_stats("split_mlp.out", split))
    dtypes["split_mlp.out"] = str(split.dtype)
    self_feature = split[:, : stage3.feature_dim]
    cross_feature = split[:, stage3.feature_dim :]
    activations.append(tensor_stats("self_feature", self_feature))
    activations.append(tensor_stats("cross_feature", cross_feature))
    residuals["split_mlp_vs_pooled"] = residual_ratio(
        self_feature - pooled_feature, pooled_feature
    )

    batch = merged.batch_index
    n_batch = int(batch.max().item()) + 1
    groups = _blocks_by_batch_agent(batch, n_batch, agent_slices)

    self_out = self_feature.clone()
    for _b, agent, rows in groups:
        use_rows = rows
        if self_keep is not None and agent == "vehicle":
            use_rows = rows[: int(self_keep)]
        x = self_feature[use_rows].unsqueeze(0)
        for li, block in enumerate(stage3.self_blocks):
            x = instrumented_block(
                block,
                x,
                None,
                f"self/{agent}/block{li}",
                activations,
                residuals,
                logits_store,
                collect_logits=collect_logits and agent == "vehicle" and li == 0,
            )
        self_out[use_rows] = x.squeeze(0)
    activations.append(tensor_stats("self_context", self_out))

    cross_out = cross_feature.clone()
    for _b, agent, rows in groups:
        others = [(a2, r2) for b2, a2, r2 in groups if b2 == _b and a2 != agent]
        if not others:
            continue
        raw_q = cross_feature[rows]
        adapted_q = stage3.adapter(raw_q, agent)
        activations.append(tensor_stats(f"cross/{agent}.raw", raw_q))
        activations.append(tensor_stats(f"cross/{agent}.adapter_out", adapted_q))
        residuals[f"cross/{agent}.adapter"] = residual_ratio(adapted_q - raw_q, raw_q)
        query = adapted_q.unsqueeze(0)
        kv_parts = [stage3.adapter(cross_feature[r2], a2) for a2, r2 in others]
        kv = torch.cat(kv_parts, dim=0).unsqueeze(0)
        activations.append(tensor_stats(f"cross/{agent}.kv", kv))
        for li, block in enumerate(stage3.cross_blocks):
            query = instrumented_block(
                block,
                query,
                kv,
                f"cross/{agent}/block{li}",
                activations,
                residuals,
                logits_store,
                collect_logits=collect_logits and agent == "vehicle" and li == 0,
            )
        cross_out[rows] = query.squeeze(0)
    activations.append(tensor_stats("cross_context", cross_out))

    concat = torch.cat([self_out, cross_out], dim=-1)
    activations.append(tensor_stats("fusion.concat", concat))
    delta_fusion = stage3.fusion_mlp(concat)
    activations.append(tensor_stats("fusion.delta", delta_fusion))
    dtypes["fusion.delta"] = str(delta_fusion.dtype)
    residuals["fusion"] = residual_ratio(delta_fusion, pooled_feature)
    fused = pooled_feature + delta_fusion
    activations.append(tensor_stats("final_feature", fused))
    dtypes["final_feature"] = str(fused.dtype)

    out = merged.replace_feature(fused)
    bev = None if skip_splat else stage3.splat(out)
    by_name = {r["name"]: r for r in activations}
    jumps: List[Dict[str, Any]] = []
    pairs = [
        ("pooled_feature", "self_feature"),
        ("pooled_feature", "cross_feature"),
        ("self_feature", "self/vehicle/block0.block_input"),
        ("self/vehicle/block0.query_ln_out", "self/vehicle/block0.Q"),
        ("self/vehicle/block0.Q", "self/vehicle/block0.attn_pre_out_proj"),
        ("self/vehicle/block0.attn_pre_out_proj", "self/vehicle/block0.attn_post_out_proj"),
        ("self/vehicle/block0.block_input", "self/vehicle/block0.after_attn_residual"),
        ("self/vehicle/block0.ffn_ln_out", "self/vehicle/block0.ffn_fc1"),
        ("self/vehicle/block0.ffn_fc1", "self/vehicle/block0.ffn_out"),
        ("self/vehicle/block0.after_attn_residual", "self/vehicle/block0.after_ffn_residual"),
        ("cross/vehicle.raw", "cross/vehicle.adapter_out"),
        ("cross/vehicle/block0.query_ln_out", "cross/vehicle/block0.Q"),
        ("cross/vehicle/block0.attn_pre_out_proj", "cross/vehicle/block0.attn_post_out_proj"),
        ("cross/vehicle/block0.block_input", "cross/vehicle/block0.after_attn_residual"),
        ("cross/vehicle/block0.after_attn_residual", "cross/vehicle/block0.after_ffn_residual"),
        ("fusion.concat", "fusion.delta"),
        ("pooled_feature", "final_feature"),
    ]
    for a, b in pairs:
        if a in by_name and b in by_name:
            jumps.append(jump_flags(by_name[a], by_name[b]))
    first = next((j for j in jumps if j["flagged"]), None)
    probe.update(
        {
            "activations": activations,
            "residuals": residuals,
            "logits": logits_store,
            "jumps": jumps,
            "first_major_amplification": first,
            "dtypes": dtypes,
            "agent_slices": {k: list(v) for k, v in agent_slices.items()},
        }
    )
    return out, bev, probe


def load_epoch_weights(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    """Load ``model_state_dict`` from a training checkpoint."""
    blob = torch.load(str(ckpt_path), map_location=device)
    src = blob["model_state_dict"] if "model_state_dict" in blob else blob
    state = {}
    for key, value in src.items():
        state[key[7:] if key.startswith("module.") else key] = value
    model.load_state_dict(state, strict=False)


def collect_stage3_params(stage3: GaussianContextInteraction) -> List[Dict[str, Any]]:
    """Named Linear / LayerNorm stats for every Stage-3 submodule."""
    rows: List[Dict[str, Any]] = []
    rows.extend(module_param_stats(stage3.split_mlp, "split_mlp"))
    for i, block in enumerate(stage3.self_blocks):
        rows.extend(module_param_stats(block, f"self_blocks.{i}"))
    rows.extend(module_param_stats(stage3.adapter, "adapter"))
    for i, block in enumerate(stage3.cross_blocks):
        rows.extend(module_param_stats(block, f"cross_blocks.{i}"))
    rows.extend(module_param_stats(stage3.fusion_mlp, "fusion_mlp"))
    return rows


def collect_stage3_grads(stage3: GaussianContextInteraction) -> List[Dict[str, Any]]:
    """Gradient stats for Stage-3 parameters after backward."""
    rows: List[Dict[str, Any]] = []
    for name, param in stage3.named_parameters():
        rec: Dict[str, Any] = {
            "name": name,
            "requires_grad": bool(param.requires_grad),
            "grad_is_none": param.grad is None,
        }
        if param.grad is None:
            rows.append(rec)
            continue
        rec.update(tensor_stats("grad", param.grad))
        rec["name"] = name
        rec["l2"] = float(torch.linalg.vector_norm(param.grad.detach().float()))
        rows.append(rec)
    return rows


@contextmanager
def sdp_backend(mem_efficient: bool):
    """Temporarily select mem-efficient or math SDPA. Flash stays off."""
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_math=not mem_efficient,
        enable_mem_efficient=mem_efficient,
    ):
        yield


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-3 feature stability diagnostic")
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=str, default="", help="Single ckpt path")
    parser.add_argument(
        "--epochs",
        type=str,
        default="1,4,7,10,12,28",
        help="Checkpoint file numbers net_epochN.pth",
    )
    parser.add_argument("--index", type=int, default=0, help="Dataset index")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output JSON path. Default: <model_dir>/stage3_feature_stability.json",
    )
    return parser.parse_args()


def load_fixed_batch(
    hypes: Dict[str, Any],
    split: str,
    index: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Load one deterministic sample. Does not shuffle."""
    if split == "test":
        hypes = dict(hypes)
        hypes["validate_dir"] = hypes["test_dir"]
    dataset = build_dataset(hypes, visualize=False, train=False)
    if index < 0 or index >= len(dataset):
        raise IndexError(f"index {index} out of range 0..{len(dataset) - 1}")
    raw = dataset[index]
    batch = dataset.collate_batch_test([raw])
    batch = train_utils.to_device(batch, device)
    meta = batch["ego"].get("metadata_path_list", ["?"])
    print("frame:", meta[0] if isinstance(meta, (list, tuple)) else meta)
    print("dataset_len:", len(dataset), "index:", index)
    return batch


def capture_stage2(
    model: nn.Module, batch: Dict[str, Any]
) -> Dict[str, GaussianSet]:
    """Run production encode/init/stage1/stage2 and return Stage-2 Gaussians."""
    from opencood.models.airv2x_gaussian_0822 import (
        CAMERA_GEOMETRY_KEY,
        present_camera_agents,
    )

    data = batch["ego"]
    for agent in present_camera_agents(data):
        model._encode_agent(agent, data)
    gaussians = model.initializer.forward(data)
    for agent, gs in gaussians.items():
        gaussians[agent] = model.intra_view(
            gs, data[agent]["f90"], data[agent][CAMERA_GEOMETRY_KEY], agent=agent
        )
    sources = {
        agent: {
            "features": data[agent]["f90"],
            "geometry": data[agent][CAMERA_GEOMETRY_KEY],
        }
        for agent in gaussians
    }
    return model.cross_agent(gaussians, sources)


def summarize_first_amp(probe: Dict[str, Any]) -> Dict[str, Any]:
    first = probe.get("first_major_amplification")
    key_stats = {}
    want = [
        "pooled_feature",
        "self_feature",
        "self/vehicle/block0.Q",
        "self/vehicle/block0.attn_post_out_proj",
        "self/vehicle/block0.ffn_out",
        "self/vehicle/block0.after_ffn_residual",
        "cross/vehicle.adapter_out",
        "cross/vehicle/block0.ffn_out",
        "fusion.delta",
        "final_feature",
    ]
    by_name = {r["name"]: r for r in probe.get("activations", [])}
    for key in want:
        if key in by_name:
            key_stats[key] = {
                "rms": by_name[key]["rms"],
                "max_abs": by_name[key]["max_abs"],
                "mean_abs": by_name[key]["mean_abs"],
            }
    return {"first_major_amplification": first, "key_stats": key_stats}


def run_backward(
    model: nn.Module,
    batch: Dict[str, Any],
    criterion: nn.Module,
    mem_efficient: bool,
    detect_anomaly: bool,
) -> Dict[str, Any]:
    """One detection-loss backward on the current weights. No optimizer step."""
    model.zero_grad(set_to_none=True)
    rec: Dict[str, Any] = {
        "backend": "mem-efficient" if mem_efficient else "math",
        "anomaly": detect_anomaly,
        "forward_finite": None,
        "loss": None,
        "loss_finite": None,
        "backward_ok": False,
        "first_nan": None,
        "grads": [],
    }
    torch.autograd.set_detect_anomaly(detect_anomaly)
    try:
        with sdp_backend(mem_efficient):
            out = model(batch["ego"])
            loss = criterion(out, batch["ego"]["label_dict"])
        rec["loss"] = float(loss.detach()) if torch.isfinite(loss) else None
        rec["loss_finite"] = bool(torch.isfinite(loss))
        rec["forward_finite"] = all(
            bool(torch.isfinite(out[k]).all())
            for k in ("psm", "rm", "obj")
            if k in out
        )
        loss.backward()
        rec["backward_ok"] = True
        rec["grads"] = collect_stage3_grads(model.stage3)
    except RuntimeError as exc:
        rec["first_nan"] = str(exc)
        rec["backward_ok"] = False
    finally:
        torch.autograd.set_detect_anomaly(False)
        model.zero_grad(set_to_none=True)
    return rec


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    opt = parse_args()
    device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model_dir = Path(opt.model_dir)
    out_json = Path(opt.out) if opt.out else model_dir / "stage3_feature_stability.json"
    out_dir = out_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    dummy = argparse.Namespace(model_dir=str(model_dir), hypes_yaml=None)
    hypes = yaml_utils.load_yaml(None, dummy)
    batch = load_fixed_batch(hypes, opt.split, opt.index, device)
    batch["ego"]["epoch"] = 0
    criterion = train_utils.create_loss(hypes)
    model = train_utils.create_model(hypes).to(device)

    if opt.checkpoint:
        ckpts = [("custom", Path(opt.checkpoint))]
    else:
        ckpts = []
        for token in opt.epochs.split(","):
            n = int(token.strip())
            path = model_dir / f"net_epoch{n}.pth"
            if path.is_file():
                ckpts.append((n, path))
            else:
                print(f"missing {path}")
    if not ckpts:
        raise FileNotFoundError("no checkpoints to run")

    payload: Dict[str, Any] = {
        "model_dir": str(model_dir),
        "split": opt.split,
        "index": opt.index,
        "device": str(device),
        "cc": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "stage3_audit": {
            "split_mlp": "Linear(C, 2C), no activation",
            "self": "PreNorm SDPA + FFN residual, 1 layer",
            "cross": "AgentResidualAdapter then PreNorm SDPA+FFN, 1 layer",
            "fusion_mlp": "Linear(2C, C), no activation",
            "final": "pooled_feature + fusion_mlp(cat(self, cross))",
            "splat": "not included in feature probes",
        },
        "epochs": [],
    }
    activation_rows: List[Dict[str, Any]] = []
    param_rows: List[Dict[str, Any]] = []

    detail_epoch = ckpts[-2][0] if len(ckpts) >= 2 else ckpts[0][0]
    # Prefer file 12 for the known frame comparison.
    for tag, _path in ckpts:
        if tag == 12:
            detail_epoch = 12
            break

    for tag, path in ckpts:
        print(f"\n===== checkpoint {tag} {path.name} =====")
        load_epoch_weights(model, path, device)
        model.eval()
        with torch.no_grad():
            g2 = capture_stage2(model, batch)
            stage2_stats = {
                agent: tensor_stats(f"stage2/{agent}", gs.feature)
                for agent, gs in g2.items()
            }
            _out, _bev, probe = instrumented_stage3(
                model.stage3,
                g2,
                skip_splat=True,
                collect_logits=(tag == detail_epoch),
            )
        params = collect_stage3_params(model.stage3)
        for row in params:
            row["epoch_file"] = tag
            param_rows.append(row)
        for rec in probe["activations"]:
            activation_rows.append({"epoch_file": tag, **rec})

        epoch_rec: Dict[str, Any] = {
            "epoch_file": tag,
            "path": str(path),
            "stage2": stage2_stats,
            "pooled_counts": probe["pooled_counts"],
            "summary": summarize_first_amp(probe),
            "residuals": probe["residuals"],
            "jumps": probe["jumps"],
            "dtypes": probe["dtypes"],
            "autocast": probe["autocast"],
            "logits": probe.get("logits"),
            "params": params,
        }

        if tag == detail_epoch:
            seq_rows = []
            with torch.no_grad():
                vehicle_n = int(probe["pooled_counts"].get("vehicle", 0))
                lengths = [n for n in (256, 512, 1024, vehicle_n) if n <= vehicle_n]
                if vehicle_n not in lengths:
                    lengths.append(vehicle_n)
                for length in lengths:
                    _o, _b, sub = instrumented_stage3(
                        model.stage3,
                        g2,
                        skip_splat=True,
                        collect_logits=False,
                        self_keep=length,
                    )
                    seq_rows.append(
                        {
                            "length": length,
                            "attn_ratio": sub["residuals"].get(
                                "self/vehicle/block0.attn"
                            ),
                            "ffn_ratio": sub["residuals"].get("self/vehicle/block0.ffn"),
                            "after_ffn": next(
                                (
                                    a
                                    for a in sub["activations"]
                                    if a["name"]
                                    == "self/vehicle/block0.after_ffn_residual"
                                ),
                                None,
                            ),
                        }
                    )
            epoch_rec["seq_length"] = seq_rows

            model.train()
            for name, module in model.named_modules():
                if "frontend" in name or "heatmap" in name or "depth" in name:
                    module.eval()
            epoch_rec["backward"] = {
                "mem_efficient": run_backward(
                    model, batch, criterion, True, detect_anomaly=True
                ),
                "math": run_backward(
                    model, batch, criterion, False, detect_anomaly=True
                ),
            }
            model.eval()

        payload["epochs"].append(epoch_rec)
        first = probe.get("first_major_amplification")
        print("pooled", probe["pooled_counts"])
        print("first_amp", first)
        print(
            "final",
            next(
                (a for a in probe["activations"] if a["name"] == "final_feature"),
                None,
            ),
        )

    # Last checkpoint always gets a backend A/B if it was not the detail epoch.
    last_tag, last_path = ckpts[-1]
    if last_tag != detail_epoch:
        print(f"\n===== extra backward on {last_tag} =====")
        load_epoch_weights(model, last_path, device)
        model.train()
        payload["epochs"][-1]["backward"] = {
            "mem_efficient": run_backward(
                model, batch, criterion, True, detect_anomaly=True
            ),
            "math": run_backward(
                model, batch, criterion, False, detect_anomaly=True
            ),
        }

    out_json.write_text(json.dumps(payload, indent=2, default=str))
    write_csv(out_dir / "stage3_activation_stats.csv", activation_rows)
    write_csv(out_dir / "stage3_parameter_stats.csv", param_rows)
    print("wrote", out_json)
    print("wrote", out_dir / "stage3_activation_stats.csv")
    print("wrote", out_dir / "stage3_parameter_stats.csv")


if __name__ == "__main__":
    main()
