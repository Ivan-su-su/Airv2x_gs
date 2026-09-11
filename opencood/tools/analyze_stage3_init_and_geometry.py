"""Offline Stage-3 / geometry audit. Standalone: no production hooks.

Modes:
  forward : activation magnitude table on a fixed frame (fresh or ckpt Stage-3)
  scale   : Stage-1/2 delta_log_scale + cumulative ratio statistics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from opencood.hypes_yaml import yaml_utils
from opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet
from opencood.tools import train_utils

DEFAULT_MODEL_DIR = (
    ROOT / "opencood/logs/airv2x_gaussian_0822_camera"
    / "v45_scale_asym_2026_09_10_13_37_23"
)


def _stat(t: torch.Tensor) -> Dict[str, float]:
    x = t.detach().float().reshape(-1)
    if x.numel() == 0:
        return {"rms": 0.0, "meanabs": 0.0, "maxabs": 0.0}
    return {
        "rms": float(x.pow(2).mean().sqrt()),
        "meanabs": float(x.abs().mean()),
        "maxabs": float(x.abs().max()),
    }


def _load_weights(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    blob = torch.load(str(ckpt_path), map_location=device)
    src = blob.get("model_state_dict", blob)
    state = {k[7:] if k.startswith("module.") else k: v for k, v in src.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded {ckpt_path.name}: missing={len(missing)} unexpected={len(unexpected)}")


def _forward_table(model: nn.Module, batch: Dict[str, Any]) -> Dict[str, Any]:
    """Instrumented Stage-3 pass mirroring production math exactly."""
    stage3 = model.stage3
    data = batch["ego"]
    from opencood.models.airv2x_gaussian_0822 import (
        CAMERA_GEOMETRY_KEY,
        present_camera_agents,
    )

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
    g2 = model.cross_agent(gaussians, sources)

    table: Dict[str, Any] = {}
    pooled_sets = {a: stage3.pool(gs) for a, gs in g2.items() if gs.n_gaussians > 0}
    merged, slices = stage3._merge(pooled_sets)
    table["pooled_feature"] = _stat(merged.feature)

    split = stage3.split_mlp(merged.feature)
    self_feature = split[:, : stage3.feature_dim]
    cross_feature = split[:, stage3.feature_dim :]
    table["self_feature"] = _stat(self_feature)
    table["cross_feature"] = _stat(cross_feature)

    table["adapter_input"] = _stat(cross_feature)
    adapted = cross_feature.clone()
    delta_parts = []
    gamma = stage3.adapter.gamma
    for agent, (s, e) in slices.items():
        if e <= s:
            continue
        raw = cross_feature[s:e]
        delta_raw = stage3.adapter.adapters[agent](stage3.adapter.input_norm(raw))
        delta_n = stage3.adapter.delta_norm(delta_raw)
        adapted[s:e] = raw + gamma.to(raw.dtype) * delta_n
        delta_parts.append(gamma.to(raw.dtype) * delta_n)
        table[f"adapter_delta_raw/{agent}"] = _stat(delta_raw)
        table[f"adapter_delta_norm/{agent}"] = _stat(delta_n)
    if delta_parts:
        table["adapter_effective_delta"] = _stat(torch.cat(delta_parts))
        table["adapter_output"] = _stat(adapted)

    # replay attention stacks on the adapted tensor (production path)
    from opencood.models.gaussian_modules_0822.interaction.gaussian_context import (
        _blocks_by_batch_agent,
    )

    batch_idx = merged.batch_index
    n_batch = int(batch_idx.max().item()) + 1
    groups = _blocks_by_batch_agent(batch_idx, n_batch, slices)
    self_out = self_feature.clone()
    for _b, _ag, rows in groups:
        x = self_feature[rows].unsqueeze(0)
        for block in stage3.self_blocks:
            x = block(x)
        self_out[rows] = x.squeeze(0)
    cross_out = cross_feature.clone()
    for _b, ag, rows in groups:
        others = [(a2, r2) for b2, a2, r2 in groups if b2 == _b and a2 != ag]
        if not others:
            continue
        q = adapted[rows].unsqueeze(0)
        kv = torch.cat([adapted[r2] for _a2, r2 in others], dim=0).unsqueeze(0)
        for block in stage3.cross_blocks:
            q = block(q, kv)
        cross_out[rows] = q.squeeze(0)
    table["self_context"] = _stat(self_out)
    table["cross_context"] = _stat(cross_out)

    fusion_input = torch.cat(
        [stage3.self_fusion_norm(self_out), stage3.cross_fusion_norm(cross_out)],
        dim=-1,
    )
    delta_raw = stage3.fusion_mlp(fusion_input)
    delta = stage3.fusion_delta_norm(delta_raw)
    fg = stage3.fusion_gamma.to(delta.dtype)
    scaled = fg * delta
    fused = merged.feature + scaled
    table["fusion_delta_raw"] = _stat(delta_raw)
    table["fusion_delta_norm"] = _stat(delta)
    table["fusion_scaled_delta"] = _stat(scaled)
    table["final_feature"] = _stat(fused)

    base = table["pooled_feature"]["rms"] + 1e-12
    table["_ratios"] = {
        "adapter_residual_over_input": table.get("adapter_effective_delta", {}).get(
            "rms", 0.0
        )
        / (table["adapter_input"]["rms"] + 1e-12),
        "fusion_residual_over_pooled": table["fusion_scaled_delta"]["rms"] / base,
        "final_over_pooled": table["final_feature"]["rms"] / base,
    }
    return table


def _axis_stats(values: torch.Tensor, pos_bound: float) -> Dict[str, Any]:
    v = values.detach().float()
    q = torch.quantile(v, torch.tensor([0.5, 0.9, 0.99], device=v.device))
    return {
        "median": float(q[0]),
        "p90": float(q[1]),
        "p99": float(q[2]),
        "max": float(v.max()),
        "frac_pos": float((v > 0).float().mean()),
        "frac_near_pos_bound": float((v > 0.9 * pos_bound).float().mean()),
    }


def _scale_table(model: nn.Module, batch: Dict[str, Any]) -> Dict[str, Any]:
    """Stage-1 then Stage-2 scale stats, including per-step delta_log_scale."""
    data = batch["ego"]
    from opencood.models.airv2x_gaussian_0822 import (
        CAMERA_GEOMETRY_KEY,
        present_camera_agents,
    )

    deltas: Dict[tuple, torch.Tensor] = {}

    def _wrap(refiner: nn.Module, tag: str) -> None:
        orig = refiner.heads.forward

        def hooked(fusion_feature, mean, scale, quaternion, agent):
            out = orig(fusion_feature, mean, scale, quaternion, agent)
            deltas[(tag, str(agent).lower())] = out["delta_log_scale"].detach()
            return out

        refiner.heads.forward = hooked

    _wrap(model.intra_view.refiner, "stage1")
    _wrap(model.cross_agent.refiner, "stage2")

    for agent in present_camera_agents(data):
        model._encode_agent(agent, data)
    init_sets = model.initializer.forward(data)
    s1_sets = {}
    for agent, gs0 in init_sets.items():
        s1_sets[agent] = model.intra_view(
            gs0, data[agent]["f90"], data[agent][CAMERA_GEOMETRY_KEY], agent=agent
        )
    sources = {
        agent: {
            "features": data[agent]["f90"],
            "geometry": data[agent][CAMERA_GEOMETRY_KEY],
        }
        for agent in s1_sets
    }
    s2_sets = model.cross_agent(s1_sets, sources)

    table: Dict[str, Any] = {}
    for agent, gs0 in init_sets.items():
        if gs0.n_gaussians == 0:
            continue
        pos_bound = float(
            model.intra_view.refiner.heads.heads[agent].scale_delta_range[1]
        )
        init = gs0.scale.detach()
        entry: Dict[str, Any] = {"n": int(gs0.n_gaussians), "pos_bound": pos_bound}
        entry["initializer_scale"] = {
            ax: _stat(init[:, i]) for i, ax in enumerate("xyz")
        }
        for tag, gs in (("stage1", s1_sets[agent]), ("stage2", s2_sets[agent])):
            ratio = gs.scale.detach() / init.clamp_min(1e-12)
            entry[f"{tag}_scale_over_init"] = {
                ax: _axis_stats(ratio[:, i], 1e9) for i, ax in enumerate("xyz")
            }
            dlog = deltas.get((tag, agent))
            if dlog is not None:
                entry[f"{tag}_delta_log_scale"] = {
                    ax: _axis_stats(dlog[:, i], pos_bound)
                    for i, ax in enumerate("xyz")
                }
        table[agent] = entry
    return table


def _param_health(stage3: nn.Module) -> Dict[str, Any]:
    block = stage3.self_blocks[0]
    g = block.query_norm.weight.detach()
    return {
        "query_norm_gamma_mean": float(g.mean()),
        "query_norm_gamma_min": float(g.min()),
        "query_norm_gamma_max": float(g.max()),
        "q_proj_norm": float(torch.linalg.vector_norm(block.q_proj.weight.detach())),
        "ffn_fc2_norm": float(torch.linalg.vector_norm(block.ffn[2].weight.detach())),
        "adapter_gamma_mean": float(stage3.adapter.gamma.detach().mean()),
        "fusion_gamma_mean": float(stage3.fusion_gamma.detach().mean()),
    }


def _smoke(
    hypes: Dict[str, Any], device: torch.device, iters: int, log_every: int
) -> Dict[str, Any]:
    """Fresh-run training smoke. Does not load Stage-3 checkpoints."""
    import math
    import os
    from torch.utils.data import DataLoader, DistributedSampler
    from opencood.data_utils.datasets import build_dataset

    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world > 1
    if distributed:
        torch.cuda.set_device(local)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        device = torch.device("cuda", local)

    dataset = build_dataset(hypes, visualize=False, train=True)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=int(hypes["train_params"]["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        collate_fn=dataset.collate_batch_train,
        drop_last=True,
    )
    model = train_utils.create_model(hypes).to(device)
    model.train()
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], output_device=device.index,
            find_unused_parameters=True,
        )
    criterion = train_utils.create_loss(hypes)
    optimizer = train_utils.setup_optimizer(hypes, model)
    records: List[Dict[str, Any]] = []
    it = 0
    nonfinite = 0
    while it < iters:
        if sampler is not None:
            sampler.set_epoch(it)
        for batch in loader:
            if batch is None:
                continue
            it += 1
            batch = train_utils.to_device(batch, device)
            batch["ego"]["epoch"] = 0
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["ego"])
            loss = criterion(output, batch["ego"]["label_dict"])
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            gn = float(grad_norm)
            lv = float(loss.item())
            finite = bool(math.isfinite(gn) and math.isfinite(lv))
            if not finite:
                nonfinite += 1
            log_now = it == 1 or it % log_every == 0 or it == iters
            if rank == 0 and log_now:
                core = model.module if hasattr(model, "module") else model
                rec: Dict[str, Any] = {
                    "iter": it,
                    "loss": lv,
                    "grad_norm": gn,
                    "finite": finite,
                    "params": _param_health(core.stage3),
                }
                if it == iters:
                    core.eval()
                    with torch.no_grad():
                        mag = _forward_table(core, batch)
                        rec["pooled_rms"] = mag["pooled_feature"]["rms"]
                        rec["adapter_residual_ratio"] = mag["_ratios"][
                            "adapter_residual_over_input"
                        ]
                        rec["fusion_residual_ratio"] = mag["_ratios"][
                            "fusion_residual_over_pooled"
                        ]
                        rec["final_over_pooled"] = mag["_ratios"]["final_over_pooled"]
                        rec["scale"] = _scale_table(core, batch)
                    core.train()
                records.append(rec)
                extra = ""
                if "pooled_rms" in rec:
                    extra = (
                        f" pooled={rec['pooled_rms']:.3f}"
                        f" adapter_r={rec['adapter_residual_ratio']:.3f}"
                        f" fusion_r={rec['fusion_residual_ratio']:.3f}"
                    )
                print(
                    f"[smoke] iter={it} loss={lv:.4f} grad={gn:.4f} "
                    f"finite={int(finite)}{extra}"
                )
            if it >= iters:
                break
    if distributed:
        torch.distributed.barrier()
    return {"iters": it, "nonfinite": nonfinite, "world_size": world, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["forward", "scale", "smoke"], default="forward")
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    yaml_path = str(
        ROOT / "opencood/hypes_yaml/airv2x/camera/det/airv2x_gaussian_0822.yaml"
    )
    hypes = yaml_utils.load_yaml(
        yaml_path, argparse.Namespace(model_dir="", hypes_yaml=yaml_path)
    )

    if args.mode == "smoke":
        import os
        report = _smoke(hypes, torch.device("cpu"), int(args.iters), int(args.log_every))
        if int(os.environ.get("RANK", "0")) == 0:
            out = Path(args.out) if args.out else Path(args.model_dir) / "stage3_smoke.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2, default=str))
            print("wrote", out)
        return

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    from opencood.data_utils.datasets import build_dataset

    ds_hypes = dict(hypes)
    if args.split == "test":
        ds_hypes["validate_dir"] = hypes["test_dir"]
    dataset = build_dataset(ds_hypes, visualize=False, train=False)
    batch = dataset.collate_batch_test([dataset[args.index]])
    batch = train_utils.to_device(batch, device)
    batch["ego"]["epoch"] = 0

    model = train_utils.create_model(hypes).to(device)
    if args.checkpoint:
        _load_weights(model, Path(args.checkpoint), device)
    model.eval()
    with torch.no_grad():
        if args.mode == "forward":
            report: Dict[str, Any] = _forward_table(model, batch)
        else:
            report = _scale_table(model, batch)

    out = Path(args.out) if args.out else Path(args.model_dir) / f"stage3_{args.mode}_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
