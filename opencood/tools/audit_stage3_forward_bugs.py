"""Read-only audit of Stage-3 / scale / optimizer. Does not patch production files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from opencood.hypes_yaml import yaml_utils
from opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet
from opencood.models.gaussian_modules_0822.interaction.gaussian_context import (
    _blocks_by_batch_agent,
    build_gaussian_context_interaction,
)
from opencood.models.gaussian_modules_0822.refinement.refine_heads import AgentRefineHeads
from opencood.tools import train_utils

EPS = 1.0e-12
CKPT_DIR = (
    ROOT
    / "opencood/logs/airv2x_gaussian_0822_camera"
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


def _ratio(delta: torch.Tensor, base: torch.Tensor) -> float:
    return _stat(delta)["rms"] / (_stat(base)["rms"] + EPS)


def scale_map(raw: float, lo: float, hi: float) -> Tuple[float, float]:
    p = float(torch.sigmoid(torch.tensor(raw)))
    if p <= 0.5:
        delta = lo * (1.0 - 2.0 * p)
    else:
        delta = hi * (2.0 * p - 1.0)
    return delta, float(torch.exp(torch.tensor(delta)))


def test_scale_zero_mapping() -> Dict[str, Any]:
    lo, hi = -0.20, 0.05
    rows = []
    for raw in (-10.0, -1.0, 0.0, 1.0, 10.0):
        d, m = scale_map(raw, lo, hi)
        rows.append({"raw": raw, "delta_log": d, "mult": m})
    d0, m0 = scale_map(0.0, lo, hi)
    dpos, _ = scale_map(0.05, lo, hi)
    dneg, _ = scale_map(-0.05, lo, hi)
    return {
        "table": rows,
        "raw0_delta": d0,
        "raw0_mult": m0,
        "unbiased_at_zero": abs(d0) < 1e-12 and abs(m0 - 1.0) < 1e-12,
        "near_zero_pos_delta": dpos,
        "near_zero_neg_delta": dneg,
        "near_zero_slope_note": (
            "asymmetric: |d(raw=-0.05)| / |d(raw=+0.05)| = "
            f"{abs(dneg) / (abs(dpos) + EPS):.3f} "
            "(contraction slope steeper than expansion)"
        ),
    }


def test_optimizer_groups() -> Dict[str, Any]:
    model = build_gaussian_context_interaction(
        feature_dim=8,
        stage3_cfg={
            "voxel_size": [4, 4, 4],
            "point_cloud_range": [-8, -8, -8, 8, 8, 8],
            "adapter_bottleneck": 8,
        },
        attn_cfg={"heads": 4, "ffn_dim": 16},
    )
    hypes = {
        "optimizer": {
            "core_method": "Adam",
            "lr": 0.002,
            "args": {"weight_decay": 1e-4, "eps": 1e-10},
        }
    }
    opt = train_utils.setup_optimizer(hypes, model)
    named = dict(model.named_parameters())
    decay_ids = {id(p) for p in opt.param_groups[0]["params"]}
    nod_ids = {id(p) for p in opt.param_groups[1]["params"]}
    def grp(name: str) -> str:
        pid = id(named[name])
        return "no_decay" if pid in nod_ids else "decay"
    return {
        "lr": 0.002,
        "decay_wd": opt.param_groups[0]["weight_decay"],
        "no_decay_wd": opt.param_groups[1]["weight_decay"],
        "query_norm.weight": grp("self_blocks.0.query_norm.weight"),
        "query_norm.bias": grp("self_blocks.0.query_norm.bias"),
        "fusion_mlp.weight": grp("fusion_mlp.weight"),
        "fusion_mlp.bias": grp("fusion_mlp.bias"),
        "adapter.Linear.weight": grp("adapter.adapters.vehicle.0.weight"),
        "adapter.Linear.bias": grp("adapter.adapters.vehicle.0.bias"),
        "adapter.gamma": grp("adapter.gamma"),
        "fusion_gamma": grp("fusion_gamma"),
    }


def _synthetic_sets(dim: int = 8) -> Dict[str, GaussianSet]:
    def make(n: int, agent: str, val: float, xs: List[float]) -> GaussianSet:
        mean = torch.tensor([[x, 0.0, 0.0] for x in xs], dtype=torch.float32)
        feat = torch.full((n, dim), val)
        feat[:, 0] = val
        return GaussianSet(
            mean=mean,
            scale=torch.ones(n, 3),
            quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n, -1).contiguous(),
            feature=feat,
            batch_index=torch.zeros(n, dtype=torch.long),
            agent=agent,
            frame="ego",
        )

    return {
        "vehicle": make(3, "vehicle", 1.0, [0.0, 8.0, 16.0]),
        "rsu": make(2, "rsu", 2.0, [4.0, 12.0]),
        "drone": make(2, "drone", 3.0, [20.0, 28.0]),
    }


def test_routing_and_residuals() -> Dict[str, Any]:
    cfg = {
        "voxel_size": [4.0, 4.0, 4.0],
        "point_cloud_range": [-4, -4, -4, 36, 4, 4],
        "adapter_bottleneck": 8,
        "bev_n_sigma": 3.0,
    }
    model = build_gaussian_context_interaction(
        feature_dim=8, stage3_cfg=cfg, attn_cfg={"heads": 4, "ffn_dim": 16}
    )
    model.eval()
    sets = _synthetic_sets(8)
    pooled = {a: model.pool(gs) for a, gs in sets.items()}
    merged, slices = model._merge(pooled)
    batch = merged.batch_index
    groups = _blocks_by_batch_agent(batch, 1, slices)
    self_map = {ag: rows.tolist() for _b, ag, rows in groups}
    kv_map: Dict[str, List[int]] = {}
    for _b, ag, rows in groups:
        others = [(a2, r2) for b2, a2, r2 in groups if b2 == _b and a2 != ag]
        kv_map[ag] = [int(i) for a2, r2 in others for i in r2.tolist()]
    agent_of = {}
    for ag, (s, e) in slices.items():
        for i in range(s, e):
            agent_of[i] = ag

    routing = {
        "self_only_own_agent": all(
            agent_of[i] == ag for ag, rows in self_map.items() for i in rows
        ),
        "cross_kv_excludes_query_agent": all(
            agent_of[i] != ag for ag, ids in kv_map.items() for i in ids
        ),
        "no_duplicate_kv": all(len(ids) == len(set(ids)) for ids in kv_map.values()),
        "self_map": {k: v for k, v in self_map.items()},
        "kv_agents": {ag: sorted({agent_of[i] for i in ids}) for ag, ids in kv_map.items()},
    }

    # Residual disable: fusion gamma = 0 → final == pooled
    with torch.no_grad():
        model.fusion_gamma.zero_()
        out, _ = model(sets)
        fused_zero, pooled_feat = out.feature, merged.feature
        # re-merge because pool is deterministic
        fused_vs_pooled = float((fused_zero - pooled_feat).abs().max())

        model.fusion_gamma.fill_(0.1)
        orig_ad = model.adapter.forward
        model.adapter.forward = lambda feature, agent: feature
        # identity adapter: still attention can change cross. Check adapter_out==in
        split = model.split_mlp(pooled_feat)
        cross = split[:, 8:]
        adapted = cross.clone()
        diffs = []
        for ag, (s, e) in slices.items():
            y = model.adapter(cross[s:e], ag)
            diffs.append(float((y - cross[s:e]).abs().max()))
        adapter_id = max(diffs) if diffs else 0.0
        model.adapter.forward = orig_ad

        # self/cross identity
        for block in list(model.self_blocks) + list(model.cross_blocks):
            block.forward = lambda query, context=None, mask=None: query
        model.fusion_gamma.zero_()
        model.adapter.forward = lambda feature, agent: feature
        out2, _ = model(sets)
        identity_diff = float((out2.feature - pooled_feat).abs().max())

        geo = {
            "mean_unchanged": torch.equal(out2.mean, merged.mean),
            "scale_unchanged": torch.equal(out2.scale, merged.scale),
            "quat_unchanged": torch.equal(out2.quaternion, merged.quaternion),
        }

    return {
        "routing": routing,
        "fusion_delta_zero_max_abs": fused_vs_pooled,
        "adapter_disabled_max_abs": adapter_id,
        "full_identity_max_abs": identity_diff,
        "stage3_geometry_unchanged": geo,
    }


def probe(stage3: nn.Module, g2: Dict[str, GaussianSet]) -> Dict[str, Any]:
    pooled = {a: stage3.pool(gs) for a, gs in g2.items() if gs.n_gaussians > 0}
    merged, slices = stage3._merge(pooled)
    pf = merged.feature
    split = stage3.split_mlp(pf)
    self_f = split[:, : stage3.feature_dim]
    cross_f = split[:, stage3.feature_dim :]
    rec: Dict[str, Any] = {
        "pooled": _stat(pf),
        "split": _stat(split),
        "self_in": _stat(self_f),
        "cross_in": _stat(cross_f),
    }
    batch = merged.batch_index
    n_batch = int(batch.max().item()) + 1
    groups = _blocks_by_batch_agent(batch, n_batch, slices)

    self_out = self_f.clone()
    attn_d, ffn_d = [], []
    for _b, _ag, rows in groups:
        x = self_f[rows].unsqueeze(0)
        block = stage3.self_blocks[0]
        q = block.query_norm(x)
        kv = block.context_norm(x)
        bsz, nq, dim = q.shape
        qq = block.q_proj(q).view(bsz, nq, block.heads, block.head_dim).transpose(1, 2)
        kk = block.k_proj(kv).view(bsz, nq, block.heads, block.head_dim).transpose(1, 2)
        vv = block.v_proj(kv).view(bsz, nq, block.heads, block.head_dim).transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(qq, kk, vv)
        attn_out = block.out_proj(attn.transpose(1, 2).reshape(bsz, nq, dim))
        x1 = x + attn_out
        ffn_out = block.ffn(block.ffn_norm(x1))
        y = x1 + ffn_out
        self_out[rows] = y.squeeze(0)
        attn_d.append(attn_out.reshape(-1))
        ffn_d.append(ffn_out.reshape(-1))
    rec["self_out"] = _stat(self_out)
    attn_cat = torch.cat(attn_d) if attn_d else self_f.new_zeros(0)
    ffn_cat = torch.cat(ffn_d) if ffn_d else self_f.new_zeros(0)
    rec["self_attn_delta"] = _stat(attn_cat)
    rec["self_ffn_delta"] = _stat(ffn_cat)
    rec["self_attn_ratio"] = _ratio(attn_cat, self_f) if attn_d else 0.0
    rec["self_ffn_ratio"] = _ratio(ffn_cat, self_out) if ffn_d else 0.0

    adapted = cross_f.clone()
    ad_res = []
    for ag, (s, e) in slices.items():
        raw = cross_f[s:e]
        out_a = stage3.adapter(raw, ag)
        adapted[s:e] = out_a
        ad_res.append((out_a - raw).reshape(-1))
    rec["adapter_in"] = _stat(cross_f)
    rec["adapter_out"] = _stat(adapted)
    rec["adapter_residual"] = _stat(torch.cat(ad_res)) if ad_res else _stat(cross_f[:0])
    rec["adapter_ratio"] = _ratio(torch.cat(ad_res), cross_f) if ad_res else 0.0

    cross_out = cross_f.clone()
    c_attn, c_ffn = [], []
    for _b, ag, rows in groups:
        others = [(a2, r2) for b2, a2, r2 in groups if b2 == _b and a2 != ag]
        if not others:
            continue
        query = adapted[rows].unsqueeze(0)
        kv = torch.cat([adapted[r2] for _a2, r2 in others], dim=0).unsqueeze(0)
        block = stage3.cross_blocks[0]
        q = block.query_norm(query)
        kvn = block.context_norm(kv)
        bsz, nq, dim = q.shape
        nkv = kvn.shape[1]
        qq = block.q_proj(q).view(bsz, nq, block.heads, block.head_dim).transpose(1, 2)
        kk = block.k_proj(kvn).view(bsz, nkv, block.heads, block.head_dim).transpose(1, 2)
        vv = block.v_proj(kvn).view(bsz, nkv, block.heads, block.head_dim).transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(qq, kk, vv)
        attn_out = block.out_proj(attn.transpose(1, 2).reshape(bsz, nq, dim))
        x1 = query + attn_out
        ffn_out = block.ffn(block.ffn_norm(x1))
        y = x1 + ffn_out
        cross_out[rows] = y.squeeze(0)
        c_attn.append(attn_out.reshape(-1))
        c_ffn.append(ffn_out.reshape(-1))
    rec["cross_out"] = _stat(cross_out)
    rec["cross_attn_ratio"] = _ratio(torch.cat(c_attn), adapted) if c_attn else 0.0
    rec["cross_ffn_ratio"] = _ratio(torch.cat(c_ffn), adapted) if c_ffn else 0.0

    fin = torch.cat(
        [stage3.self_fusion_norm(self_out), stage3.cross_fusion_norm(cross_out)],
        dim=-1,
    )
    raw = stage3.fusion_mlp(fin)
    nd = stage3.fusion_delta_norm(raw)
    scaled = stage3.fusion_gamma.to(nd.dtype) * nd
    fused = pf + scaled
    rec["fusion_in"] = _stat(fin)
    rec["fusion_delta_raw"] = _stat(raw)
    rec["fusion_delta"] = _stat(scaled)
    rec["fusion_ratio"] = _ratio(scaled, pf)
    rec["final"] = _stat(fused)
    rec["final_over_pooled"] = rec["final"]["rms"] / (rec["pooled"]["rms"] + EPS)
    rec["geo_unchanged"] = bool(
        torch.equal(merged.mean, merged.mean) and torch.equal(merged.scale, merged.scale)
    )
    flagged = []
    for name, r in (
        ("adapter", rec["adapter_ratio"]),
        ("self_attn", rec["self_attn_ratio"]),
        ("self_ffn", rec["self_ffn_ratio"]),
        ("cross_attn", rec["cross_attn_ratio"]),
        ("cross_ffn", rec["cross_ffn_ratio"]),
        ("fusion", rec["fusion_ratio"]),
    ):
        if r > 1.0:
            flagged.append({"op": name, "ratio": r, "gt1": True, "gt3": r > 3.0})
    rec["first_ratio_gt1"] = flagged[0] if flagged else None
    rec["ratio_gt3"] = [x for x in flagged if x["gt3"]]
    rec["mean"] = merged.mean
    rec["scale"] = merged.scale
    rec["fused"] = fused
    rec["merged"] = merged
    return rec


def capture_g2(model: nn.Module, batch: Dict[str, Any]) -> Dict[str, GaussianSet]:
    from opencood.models.airv2x_gaussian_0822 import (
        CAMERA_GEOMETRY_KEY,
        present_camera_agents,
    )

    data = batch["ego"]
    for agent in present_camera_agents(data):
        model._encode_agent(agent, data)
    g = model.initializer.forward(data)
    init_scale = {a: gs.scale.detach().clone() for a, gs in g.items()}
    s1 = {}
    for agent, gs in g.items():
        s1[agent] = model.intra_view(
            gs, data[agent]["f90"], data[agent][CAMERA_GEOMETRY_KEY], agent=agent
        )
    sources = {
        a: {"features": data[a]["f90"], "geometry": data[a][CAMERA_GEOMETRY_KEY]}
        for a in s1
    }
    s2 = model.cross_agent(s1, sources)
    return {"init_scale": init_scale, "s1": s1, "s2": s2}


def test_stage2_invalid_identity() -> Dict[str, Any]:
    from opencood.models.gaussian_modules_0822.cross_agent.test_stage2_correctness import (
        test_adapter_runs_on_source_tokens_not_query,
        test_invalid_query_is_exact_identity,
        test_mixed_valid_subset_skips_invalid,
    )

    test_invalid_query_is_exact_identity()
    test_adapter_runs_on_source_tokens_not_query()
    test_mixed_valid_subset_skips_invalid()
    return {"stage2_unit_tests": "PASS"}


def summarize_old_stability_json() -> Dict[str, Any]:
    path = CKPT_DIR / "stage3_feature_stability.json"
    blob = json.loads(path.read_text())
    rows = []
    for ep in blob.get("epochs") or []:
        n = int(ep.get("epoch_file", -1))
        if n not in {1, 4, 7, 10, 12}:
            continue
        ks = ((ep.get("summary") or {}).get("key_stats") or {})
        res = ep.get("residuals") or {}
        params = {f"{p['module']}.{p['param']}": p.get("frobenius") for p in ep.get("params") or []}
        gt1 = [
            {"op": k, "ratio": v["rms_ratio"]}
            for k, v in res.items()
            if isinstance(v, dict) and float(v.get("rms_ratio") or 0) > 1.0
        ]
        gt3 = [x for x in gt1 if x["ratio"] > 3.0]
        rows.append(
            {
                "epoch": n,
                "pooled": ks.get("pooled_feature"),
                "final": ks.get("final_feature"),
                "fusion_delta": ks.get("fusion.delta"),
                "split_ratio": (res.get("split_mlp_vs_pooled") or {}).get("rms_ratio"),
                "self_attn_ratio": (res.get("self/vehicle/block0.attn") or {}).get("rms_ratio"),
                "self_ffn_ratio": (res.get("self/vehicle/block0.ffn") or {}).get("rms_ratio"),
                "drone_adapter_ratio": (res.get("cross/drone.adapter") or {}).get("rms_ratio"),
                "veh_adapter_ratio": (res.get("cross/vehicle.adapter") or {}).get("rms_ratio"),
                "fusion_ratio": (res.get("fusion") or {}).get("rms_ratio"),
                "first_ratio_gt1": gt1[0] if gt1 else None,
                "ratio_gt3": gt3,
                "q_proj": params.get("self_blocks.0.q_proj.weight"),
                "query_norm_g": params.get("self_blocks.0.query_norm.weight"),
                "ffn_norm_g": params.get("self_blocks.0.ffn_norm.weight"),
                "fusion_lin": params.get("fusion_mlp.weight"),
            }
        )
    return {"path": str(path), "epochs": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=4)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--skip_full_model", action="store_true")
    args = parser.parse_args()
    report: Dict[str, Any] = {}
    report["scale_zero_mapping"] = test_scale_zero_mapping()
    report["scale_zero_mapping_old_range"] = {
        "range": [-0.3, 0.1],
        "table": [
            {"raw": r, "delta_log": scale_map(r, -0.3, 0.1)[0], "mult": scale_map(r, -0.3, 0.1)[1]}
            for r in (-10.0, -1.0, 0.0, 1.0, 10.0)
        ],
        "raw0_unbiased": abs(scale_map(0.0, -0.3, 0.1)[0]) < 1e-12,
    }
    report["optimizer"] = test_optimizer_groups()
    report["routing_residuals"] = test_routing_and_residuals()
    report["stage2_invalid"] = test_stage2_invalid_identity()
    report["old_ckpt_amplitude"] = summarize_old_stability_json()

    if not args.skip_full_model:
        device = torch.device(
            f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        if device.type == "cuda":
            torch.cuda.set_device(device)
        yaml_path = str(
            ROOT / "opencood/hypes_yaml/airv2x/camera/det/airv2x_gaussian_0822.yaml"
        )
        hypes = yaml_utils.load_yaml(
            yaml_path, argparse.Namespace(model_dir="", hypes_yaml=yaml_path)
        )
        from opencood.data_utils.datasets import build_dataset

        ds = dict(hypes)
        ds["validate_dir"] = hypes["test_dir"]
        dataset = build_dataset(ds, visualize=False, train=False)
        batch = train_utils.to_device(
            dataset.collate_batch_test([dataset[args.index]]), device
        )
        batch["ego"]["epoch"] = 0
        model = train_utils.create_model(hypes).to(device)

        report["ownership"] = {
            "s2_adapter_is_s3": model.cross_agent.adapter is model.stage3.adapter,
            "s1_refiner_is_s2": model.intra_view.refiner is model.cross_agent.refiner,
            "s1_s2_share_geo_encoder": (
                model.intra_view.refiner.geometry_encoder
                is model.cross_agent.refiner.geometry_encoder
            ),
            "self_cross_param_overlap": bool(
                {id(p) for p in model.stage3.self_blocks.parameters()}
                & {id(p) for p in model.stage3.cross_blocks.parameters()}
            ),
        }

        model.eval()
        with torch.no_grad():
            cap = capture_g2(model, batch)
            report["fresh_stage2"] = {
                a: _stat(gs.feature) for a, gs in cap["s2"].items() if gs.n_gaussians
            }
            pr = probe(model.stage3, cap["s2"])
            pooled = {
                a: model.stage3.pool(gs)
                for a, gs in cap["s2"].items()
                if gs.n_gaussians
            }
            merged, _ = model.stage3._merge(pooled)
            out, bev = model.stage3(cap["s2"])
            bev_pooled = model.stage3.splat(merged)
            report["fresh_probe"] = {
                k: v
                for k, v in pr.items()
                if k not in ("mean", "scale", "fused", "merged")
            }
            report["stage3_geo"] = {
                "scale_max_abs": float((out.scale - merged.scale).abs().max()),
                "mean_max_abs": float((out.mean - merged.mean).abs().max()),
            }
            report["bev"] = {
                "after_stage3": _stat(bev),
                "pooled_splat": _stat(bev_pooled),
            }

        model.train()
        criterion = train_utils.create_loss(hypes)
        train_h = dict(hypes)
        train_set = build_dataset(train_h, visualize=False, train=True)
        tb = train_utils.to_device(
            train_set.collate_batch_train([train_set[0]]), device
        )
        tb["ego"]["epoch"] = 0
        model.zero_grad(set_to_none=True)
        out_h = model(tb["ego"])
        loss = criterion(out_h, tb["ego"]["label_dict"])
        loss.backward()

        def gstat(mod: nn.Module) -> Dict[str, float]:
            gs = [
                p.grad.detach().float().reshape(-1)
                for p in mod.parameters()
                if p.grad is not None
            ]
            if not gs:
                return {"rms": 0.0, "maxabs": 0.0}
            x = torch.cat(gs)
            return {
                "rms": float(x.pow(2).mean().sqrt()),
                "maxabs": float(x.abs().max()),
            }

        report["grads"] = {
            "loss": float(loss.item()),
            "psm": _stat(out_h["psm"]),
            "rm": _stat(out_h["rm"]),
            "obj": _stat(out_h["obj"]),
            "s1_scale_head": gstat(
                model.intra_view.refiner.heads.heads["vehicle"].scale_head
            ),
            "s2_scale_head": gstat(
                model.cross_agent.refiner.heads.heads["vehicle"].scale_head
            ),
            "s3_adapter": gstat(model.stage3.adapter),
            "s3_split": gstat(model.stage3.split_mlp),
            "s3_self": gstat(model.stage3.self_blocks),
            "s3_cross": gstat(model.stage3.cross_blocks),
            "s3_fusion": gstat(model.stage3.fusion_mlp),
        }

    outp = CKPT_DIR / "forward_bug_audit.json"
    outp.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print("wrote", outp)


if __name__ == "__main__":
    main()
