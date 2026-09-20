#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only Stage-2 geometry diagnostic for AirV2X Gaussian.

Diagnoses, per agent:
1) actual geometry-vs-detection gradient strength after current coord normalization;
2) detection/geometry gradient conflict (cosine);
3) GT residual distribution vs bounded mean-head range;
4) dependence on cross-agent source image tokens via a zero-token ablation.

No optimizer.step() and no parameter updates.
"""
from __future__ import annotations

import argparse, csv, json, math, random, re, sys, types
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.gaussian_modules_0822.refinement import geometry_supervision as geom_impl
from opencood.tools import train_utils

AGENTS = ("vehicle", "rsu", "drone")
INTERACTION_BY_AGENT = {
    "vehicle": "vehicle_from_drone",
    "rsu": "rsu_from_drone",
    "drone": "drone_from_ground",
}
AXES = ("x", "y", "z")
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="")
    p.add_argument("--hypes-yaml", "-y", default="")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--checkpoint-epoch", type=int, default=None)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--num-batches", type=int, default=20)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--out-dir", default="opencood/logs/stage2_geom_diagnosis")
    p.add_argument("--pred-saturation-frac", type=float, default=0.95)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def resolve_paths(opt):
    if opt.model_dir:
        model_dir = Path(opt.model_dir)
        cfg = model_dir / "config.yaml"
        if not cfg.exists(): raise FileNotFoundError(cfg)
        if opt.checkpoint:
            ckpt = Path(opt.checkpoint)
        elif opt.checkpoint_epoch is not None:
            ckpt = model_dir / f"net_epoch{opt.checkpoint_epoch}.pth"
        else:
            pat = re.compile(r"^net_epoch(\d+)\.pth$")
            cands = [(int(m.group(1)), p) for p in model_dir.glob("net_epoch*.pth") if (m := pat.match(p.name))]
            if not cands: raise FileNotFoundError(f"No net_epoch*.pth in {model_dir}")
            ckpt = max(cands, key=lambda x: x[0])[1]
        return str(cfg), str(ckpt)
    if not opt.hypes_yaml: raise ValueError("Use --model-dir or -y/--hypes-yaml")
    return opt.hypes_yaml, opt.checkpoint


def unwrap_state(obj):
    if not isinstance(obj, dict): raise TypeError(type(obj))
    for key in ("model_state_dict", "model_state", "state_dict", "model"):
        if isinstance(obj.get(key), dict): return obj[key]
    return obj


def load_checkpoint(model, path, device):
    if not path:
        print("[checkpoint] none")
        return
    state = dict(unwrap_state(torch.load(path, map_location=device)))
    if state and next(iter(state)).startswith("module."):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    print(f"[checkpoint] {path}")


def build_loader(hypes, worker):
    ds = build_dataset(hypes, visualize=False, train=True)
    kw = dict(dataset=ds, batch_size=int(hypes["train_params"]["batch_size"]), shuffle=False,
              num_workers=worker, collate_fn=ds.collate_batch_train, pin_memory=True, drop_last=True)
    if worker > 0:
        kw.update(persistent_workers=True, prefetch_factor=1)
    return DataLoader(**kw)


def unique_params(modules: Sequence[torch.nn.Module]):
    out, seen = [], set()
    for module in modules:
        for p in module.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p)); out.append(p)
    return out


def probe_groups(model, agent):
    cross = model.cross_agent
    refiner = cross.refiner
    if refiner is None: raise RuntimeError("Stage2 refiner is None")
    itype = INTERACTION_BY_AGENT[agent]
    head = refiner.heads.heads[agent]
    return {
        "mean_head": unique_params([head.mean_head]),
        "query_path": unique_params([
            head.mean_head,
            refiner.fusion.fusions[agent],
            cross.attention[itype],
            cross.sampler.offset_fusion.fusions[agent],
            cross.sampler.offset_heads[agent],
        ]),
    }


class CrossCapture:
    def __init__(self, cross):
        self.stage1 = self.sources = self.stage2 = self.valid_masks = None
        self.pre = cross.register_forward_pre_hook(self._pre)
        self.post = cross.register_forward_hook(self._post)
    def _pre(self, module, inputs):
        self.stage1, self.sources = inputs[0], inputs[1]
    def _post(self, module, inputs, output):
        self.stage2 = output
        self.valid_masks = {k: v.detach().clone() for k, v in module.last_valid_masks.items()}
    def clear(self):
        self.stage1 = self.sources = self.stage2 = self.valid_masks = None
    def close(self):
        self.pre.remove(); self.post.remove()


@contextmanager
def zero_source_tokens(cross):
    """Keep real projection/masks, but replace sampled source feature values by zero."""
    original = cross._sample_tokens
    def _zeroed(_self, points, source_features, source_geometry):
        tokens, mask = original(points, source_features, source_geometry)
        return torch.zeros_like(tokens), mask
    cross._sample_tokens = types.MethodType(_zeroed, cross)
    try:
        yield
    finally:
        cross._sample_tokens = original


def flat_grad(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False, allow_unused=True)
    parts = [(torch.zeros_like(p) if g is None else g.detach()).float().reshape(-1) for p, g in zip(params, grads)]
    return torch.cat(parts) if parts else torch.zeros((0,), device=loss.device)


def grad_stats(gd, gg):
    nd, ng = float(torch.linalg.vector_norm(gd)), float(torch.linalg.vector_norm(gg))
    cos = float(torch.dot(gd, gg) / (nd * ng + EPS)) if nd > 0 and ng > 0 else float("nan")
    return nd, ng, ng / max(nd, EPS), cos


def vec_cos(a, b):
    na, nb = torch.linalg.vector_norm(a, dim=-1), torch.linalg.vector_norm(b, dim=-1)
    den = na * nb
    out = torch.full_like(den, float("nan"))
    good = den > 1e-8
    out[good] = (a[good] * b[good]).sum(-1) / den[good].clamp_min(EPS)
    return out


def q(x, p):
    x = x.detach().float(); x = x[torch.isfinite(x)]
    return float(torch.quantile(x, p)) if x.numel() else float("nan")


def finite_mean(x):
    x = x.detach().float(); x = x[torch.isfinite(x)]
    return float(x.mean()) if x.numel() else float("nan")


def build_info(model, agent, s1, s2, ego, cross_valid):
    """Expose the exact intermediates used by current production geometry_supervision.py."""
    if s1 is None or s2 is None or s2.n_gaussians == 0 or s1.uv is None or s1.view_index is None:
        return None
    payload = ego.get(agent)
    if not isinstance(payload, dict): return None
    cam_inputs = payload.get("batch_merged_cam_inputs") or {}
    imgs = cam_inputs.get("imgs")
    geometry = payload.get(geom_impl.CAMERA_GEOMETRY_KEY)
    if imgs is None or int(imgs.shape[-3]) < 4 or geometry is None: return None

    imgs_flat = geom_impl._flatten_imgs(imgs)
    gt_depth, view, uv_px = geom_impl._sample_gt_depth(s1, imgs_flat)
    dmin, dmax = model.depth_ranges.get(agent, (float("nan"), float("nan")))
    depth_valid = torch.isfinite(gt_depth) & (gt_depth >= dmin) & (gt_depth <= dmax)
    if cross_valid is None:
        cross_valid = torch.zeros(s2.n_gaussians, dtype=torch.bool, device=s2.mean.device)
    else:
        cross_valid = cross_valid.to(device=s2.mean.device, dtype=torch.bool)
    if cross_valid.numel() != s2.n_gaussians:
        raise RuntimeError(f"{agent}: cross_valid size mismatch")
    mask = depth_valid & cross_valid
    idx = mask.nonzero(as_tuple=False).squeeze(1)
    if idx.numel() == 0: return None

    gt_xyz = geom_impl._gt_xyz_ego(s1, geometry, uv_px, view, gt_depth, idx)
    base = s1.mean.detach()[idx]
    pred = s2.mean[idx] - base
    target = gt_xyz - base
    beta = float(model.geometry_supervision_beta)
    loss_xyz = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    return dict(idx=idx, base=base, gt_xyz=gt_xyz, pred=pred, target=target,
                diff=pred-target, loss_xyz=loss_xyz, loss_mean=loss_xyz.mean(),
                loss_sum=loss_xyz.sum(), n_valid=int(idx.numel()), n_coords=int(idx.numel())*3,
                depth_valid_count=int(depth_valid.sum()), cross_valid_count=int(cross_valid.sum()), beta=beta)


def unit_xyz(model, agent, like):
    return model.cross_agent.refiner.heads.heads[agent].unit_xyz.to(device=like.device, dtype=like.dtype)


def distribution_metrics(model, agent, info, sat_frac):
    pred, target, diff = info["pred"], info["target"], info["diff"]
    base, gt_xyz = info["base"], info["gt_xyz"]
    bounds = unit_xyz(model, agent, pred)
    pred_norm, target_norm = torch.linalg.vector_norm(pred, dim=-1), torch.linalg.vector_norm(target, dim=-1)
    cosine = vec_cos(pred, target)
    e1 = torch.linalg.vector_norm(base - gt_xyz, dim=-1)
    e2 = torch.linalg.vector_norm(base + pred - gt_xyz, dim=-1)
    exceed = target.abs() > bounds[None]
    near = pred.abs() >= sat_frac * bounds[None]
    beta = info["beta"]
    linear = diff.abs() >= beta if beta > 0 else torch.ones_like(diff, dtype=torch.bool)
    gradmag = (diff.abs()/beta).clamp(max=1.0) if beta > 0 else torch.ones_like(diff)
    out = {
        "n_valid": info["n_valid"], "n_coords": info["n_coords"],
        "geom_loss_mean": float(info["loss_mean"].detach()),
        "stage1_error_mean_m": float(e1.mean()), "stage2_error_mean_m": float(e2.mean()),
        "stage2_improvement_mean_m": float((e1-e2).mean()),
        "delta_gt_norm_p50_m": q(target_norm,.5), "delta_gt_norm_p90_m": q(target_norm,.9), "delta_gt_norm_p95_m": q(target_norm,.95),
        "delta_pred_norm_p50_m": q(pred_norm,.5), "delta_pred_norm_p90_m": q(pred_norm,.9), "delta_pred_norm_p95_m": q(pred_norm,.95),
        "pred_gt_cos_mean": finite_mean(cosine), "pred_gt_cos_p50": q(cosine,.5),
        "gt_bound_exceed_any_rate": float(exceed.any(-1).float().mean()),
        "pred_near_bound_any_rate": float(near.any(-1).float().mean()),
        "smoothl1_linear_coord_rate": float(linear.float().mean()),
        "smoothl1_gradmag_mean": float(gradmag.mean()),
    }
    for i, axis in enumerate(AXES):
        agt, apr = target[:,i].abs(), pred[:,i].abs()
        out.update({
            f"bound_{axis}_m": float(bounds[i]),
            f"delta_gt_abs_{axis}_p50_m": q(agt,.5), f"delta_gt_abs_{axis}_p90_m": q(agt,.9), f"delta_gt_abs_{axis}_p95_m": q(agt,.95),
            f"delta_pred_abs_{axis}_p50_m": q(apr,.5), f"delta_pred_abs_{axis}_p90_m": q(apr,.9),
            f"gt_bound_exceed_{axis}_rate": float(exceed[:,i].float().mean()),
            f"pred_near_bound_{axis}_rate": float(near[:,i].float().mean()),
        })
    return out


def ablation_metrics(info, zero_s2):
    idx, base, target, gt_xyz = info["idx"], info["base"], info["target"], info["gt_xyz"]
    normal = info["pred"]
    zero = zero_s2.mean[idx] - base
    change = torch.linalg.vector_norm(normal-zero, dim=-1)
    normal_norm = torch.linalg.vector_norm(normal, dim=-1)
    e1 = torch.linalg.vector_norm(base-gt_xyz, dim=-1)
    en = torch.linalg.vector_norm(base+normal-gt_xyz, dim=-1)
    ez = torch.linalg.vector_norm(base+zero-gt_xyz, dim=-1)
    return {
        "zero_source_delta_change_mean_m": float(change.mean()),
        "zero_source_relative_delta_change": float(change.mean()/normal_norm.mean().clamp_min(EPS)),
        "normal_vs_zero_delta_cos_mean": finite_mean(vec_cos(normal,zero)),
        "normal_pred_gt_cos_mean": finite_mean(vec_cos(normal,target)),
        "zero_source_pred_gt_cos_mean": finite_mean(vec_cos(zero,target)),
        "normal_improvement_mean_m": float((e1-en).mean()),
        "zero_source_improvement_mean_m": float((e1-ez).mean()),
        "source_visual_gain_mean_m": float((ez-en).mean()),
    }


def med(values: Iterable[float]):
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(vals)) if vals else float("nan")


def summarize(rows):
    if not rows: return {}
    out = {"n_batches_with_agent": len(rows), "n_valid_total": int(sum(r["n_valid"] for r in rows))}
    keys = set().union(*(r.keys() for r in rows))
    for key in sorted(keys):
        if key in ("batch","agent","n_valid","n_coords"): continue
        vals = [float(r[key]) for r in rows if isinstance(r.get(key),(int,float))]
        if vals: out[f"{key}_median"] = med(vals)
    return out


def flags(s):
    out = {}
    ratio = s.get("mean_head_actual_aux_over_det_median", float("nan"))
    cos = s.get("mean_head_grad_cosine_median", float("nan"))
    exceed = s.get("gt_bound_exceed_any_rate_median", float("nan"))
    src = s.get("zero_source_relative_delta_change_median", float("nan"))
    if math.isfinite(ratio): out["aux_strength"] = "very_weak" if ratio<.02 else ("weak_to_moderate" if ratio<.10 else "substantial")
    if math.isfinite(cos): out["gradient_relation"] = "conflict" if cos<-.10 else ("aligned" if cos>.10 else "orthogonal")
    if math.isfinite(exceed): out["mean_bound"] = "likely_limiting" if exceed>.10 else "not_major_for_most_targets"
    if math.isfinite(src): out["source_token_use"] = "low_sensitivity" if src<.10 else "material_sensitivity"
    return out


def main():
    opt = parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    torch.cuda.set_device(opt.gpu_id); device = torch.device(f"cuda:{opt.gpu_id}"); seed_all(opt.seed)
    cfg_path, ckpt_path = resolve_paths(opt)
    hypes = yaml_utils.load_yaml(cfg_path)
    if not bool((((hypes.get("model") or {}).get("args") or {}).get("geometry_supervision") or {}).get("enabled",False)):
        raise RuntimeError("geometry_supervision.enabled must be true")

    model = train_utils.create_model(hypes).to(device); load_checkpoint(model, ckpt_path, device); model.train()
    criterion = train_utils.create_loss(hypes).to(device)
    geom_weight = float(getattr(criterion,"geom_weight",0.0)); criterion.geom_weight = 0.0
    loader = build_loader(hypes,opt.worker); cap = CrossCapture(model.cross_agent)
    groups = {a: probe_groups(model,a) for a in AGENTS}
    rows, prod_diffs = [], []

    print(f"config={cfg_path}\ncheckpoint={ckpt_path}\ngeom_weight={geom_weight}\nbeta={model.geometry_supervision_beta}")
    for a in AGENTS: print(f"{a} unit_xyz={model.cross_agent.refiner.heads.heads[a].unit_xyz.detach().cpu().tolist()}")

    try:
        for bi,batch in enumerate(loader):
            if bi >= opt.num_batches: break
            if batch is None: continue
            batch = train_utils.to_device(batch,device); ego = batch["ego"]; ego["epoch"] = 0
            cap.clear(); model.zero_grad(set_to_none=True); criterion.zero_grad(set_to_none=True)
            with amp.autocast(enabled=bool(opt.amp)):
                output = model(ego)
                det_loss = criterion(output,ego["label_dict"])
            if cap.stage1 is None or cap.stage2 is None or cap.valid_masks is None: raise RuntimeError("cross-agent capture failed")
            prod_geom = output.get("geom_loss")
            if not torch.is_tensor(prod_geom): raise RuntimeError("geom_loss missing")

            infos = {}
            for a in AGENTS:
                if a in cap.stage1 and a in cap.stage2:
                    info = build_info(model,a,cap.stage1[a],cap.stage2[a],ego,cap.valid_masks.get(a))
                    if info is not None: infos[a]=info
            total_coords = sum(x["n_coords"] for x in infos.values())
            if not total_coords: continue
            recomposed = sum(x["loss_sum"] for x in infos.values())/total_coords
            diff = abs(float(recomposed.detach())-float(prod_geom.detach())); prod_diffs.append(diff)
            if diff > 1e-5: raise RuntimeError(f"production geom loss mismatch: {diff}")

            normal_valid = {k:v.clone() for k,v in cap.valid_masks.items()}
            with torch.no_grad(), zero_source_tokens(model.cross_agent):
                zero_s2 = model.cross_agent(cap.stage1,cap.sources)
                zero_valid = {k:v.clone() for k,v in model.cross_agent.last_valid_masks.items()}
            for a in infos:
                if a in normal_valid and a in zero_valid and not torch.equal(normal_valid[a],zero_valid[a]):
                    raise RuntimeError(f"{a}: zero-token ablation changed valid mask")

            compact=[]
            for a,info in infos.items():
                row = {"batch":bi,"agent":a,"det_loss":float(det_loss.detach()),"production_geom_loss":float(prod_geom.detach()),
                       "n_valid":info["n_valid"],"n_coords":info["n_coords"],"coord_share":info["n_coords"]/total_coords,
                       "configured_geom_weight":geom_weight}
                row.update(distribution_metrics(model,a,info,opt.pred_saturation_frac))
                if a in zero_s2: row.update(ablation_metrics(info,zero_s2[a]))
                share=float(row["coord_share"])
                for gname,params in groups[a].items():
                    gd,gg = flat_grad(det_loss,params), flat_grad(info["loss_mean"],params)
                    nd,ng,raw_ratio,cos = grad_stats(gd,gg)
                    scale=geom_weight*share
                    row.update({f"{gname}_grad_det_norm":nd,f"{gname}_grad_geo_mean_norm":ng,
                                f"{gname}_grad_geo_mean_over_det":raw_ratio,f"{gname}_grad_cosine":cos,
                                f"{gname}_actual_aux_over_det":scale*ng/max(nd,EPS),
                                f"{gname}_combined_over_det":float(torch.linalg.vector_norm(gd+scale*gg))/max(nd,EPS)})
                rows.append(row)
                compact.append(f"{a}:n={row['n_valid']} share={row['coord_share']:.2f} Lg={row['geom_loss_mean']:.3f} "
                               f"G={row['mean_head_actual_aux_over_det']:.3f} cos={row['mean_head_grad_cosine']:.3f} "
                               f"bound={row['gt_bound_exceed_any_rate']:.1%} srcΔ={row.get('zero_source_relative_delta_change',float('nan')):.2f} "
                               f"srcGain={row.get('source_visual_gain_mean_m',float('nan')):.4f}m")
            print(f"[batch {bi:03d}] "+" | ".join(compact))
            del output,det_loss,prod_geom,recomposed,zero_s2,infos
            cap.clear(); torch.cuda.empty_cache()
    finally:
        cap.close()

    if not rows: raise RuntimeError("No diagnostic rows")
    out_dir=Path(opt.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    csv_path=out_dir/"stage2_geom_diagnosis_batches.csv"
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with csv_path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    by=defaultdict(list)
    for r in rows: by[r["agent"]].append(r)
    summary={"config":cfg_path,"checkpoint":ckpt_path,"configured_geom_weight":geom_weight,
             "geom_beta":float(model.geometry_supervision_beta),"num_rows":len(rows),
             "max_production_geom_loss_recomposition_diff":max(prod_diffs) if prod_diffs else float("nan"),"agents":{}}
    for a in AGENTS:
        s=summarize(by[a]); s["flags"]=flags(s) if s else {}; summary["agents"][a]=s
    json_path=out_dir/"stage2_geom_diagnosis_summary.json"; json_path.write_text(json.dumps(summary,indent=2,allow_nan=True))

    print("\n=== FINAL SUMMARY ===")
    print(f"production geom-loss reconstruction max diff={summary['max_production_geom_loss_recomposition_diff']:.3e}")
    for a in AGENTS:
        s=summary["agents"][a]
        if not s: print(f"[{a}] no valid rows"); continue
        print(f"\n[{a}] share={s.get('coord_share_median',float('nan')):.3f} Lgeo={s.get('geom_loss_mean_median',float('nan')):.5f}")
        print(f"  mean-head actual aux/det={s.get('mean_head_actual_aux_over_det_median',float('nan')):.4f}, grad cos={s.get('mean_head_grad_cosine_median',float('nan')):.4f}")
        print(f"  query-path actual aux/det={s.get('query_path_actual_aux_over_det_median',float('nan')):.4f}, grad cos={s.get('query_path_grad_cosine_median',float('nan')):.4f}")
        print(f"  GT bound exceed={s.get('gt_bound_exceed_any_rate_median',float('nan')):.2%}, pred near-bound={s.get('pred_near_bound_any_rate_median',float('nan')):.2%}")
        print(f"  SmoothL1 linear coords={s.get('smoothl1_linear_coord_rate_median',float('nan')):.2%}, pred/GT cos={s.get('pred_gt_cos_mean_median',float('nan')):.4f}")
        print(f"  zero-source relative delta change={s.get('zero_source_relative_delta_change_median',float('nan')):.4f}")
        print(f"  source visual gain={s.get('source_visual_gain_mean_m_median',float('nan')):.6f} m (positive=real source tokens help)")
        print(f"  flags={s.get('flags',{})}")
    print(f"\n[wrote] {csv_path}\n[wrote] {json_path}")


if __name__ == "__main__":
    main()
