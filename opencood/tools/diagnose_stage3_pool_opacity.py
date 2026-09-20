#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
READ-ONLY Stage3 pooling / opacity diagnostic for AirV2X Gaussian.

A) Stage2 -> Stage3 voxel-pool compression + multiplicity
B) Within-voxel heterogeneity
C) Same-checkpoint AP A/B:
   current splat: feature * GaussianKernel
   opacity splat: p_fg * feature * GaussianKernel

Default checkpoint:
 /mnt/home/suyi/AirV2X-Perception_gaussian/opencood/logs/
 airv2x_gaussian_0822_geom_sup/geom_sup_2026_09_17_00_34_27/net_epoch25.pth
"""
from __future__ import annotations
import argparse, contextlib, copy, csv, json, math, random, sys, types
from collections import OrderedDict, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Sequence

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
from opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet
from opencood.tools import train_utils
from opencood.utils import eval_utils_opv2v as eval_utils

DEFAULT_CKPT = (
    "/mnt/home/suyi/AirV2X-Perception_gaussian/opencood/logs/"
    "airv2x_gaussian_0822_geom_sup/geom_sup_2026_09_17_00_34_27/net_epoch25.pth"
)
AGENTS = ("vehicle", "rsu", "drone")
EPS = 1e-12

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--config", default="")
    p.add_argument("--num-frames", type=int, default=100)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--worker", type=int, default=2)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--eval-ious", default="0.3,0.5,0.7")
    p.add_argument("--max-hetero-voxels", type=int, default=5000)
    p.add_argument("--out-dir", default="opencood/logs/diagnose_stage3_pool_opacity_ep25")
    return p.parse_args()

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def resolve_config(ckpt, explicit):
    p = Path(explicit) if explicit else ckpt.parent / "config.yaml"
    if not p.exists(): raise FileNotFoundError(p)
    return p

def unwrap(blob):
    if not isinstance(blob, dict): raise TypeError(type(blob))
    for k in ("model_state_dict","model_state","state_dict","model"):
        if isinstance(blob.get(k), dict): return blob[k]
    return blob

def load_ckpt(model, ckpt, device):
    state = dict(unwrap(torch.load(str(ckpt), map_location=device)))
    if state and next(iter(state)).startswith("module."):
        state = {k[7:]:v for k,v in state.items()}
    cur = model.state_dict()
    bad = [k for k,v in state.items() if k in cur and tuple(v.shape)!=tuple(cur[k].shape)]
    if bad: raise RuntimeError(f"shape mismatch: {bad[:10]}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"missing={missing[:20]} unexpected={unexpected[:20]}")
    print(f"[ckpt] exact-compatible: {ckpt}")

def make_loader(hypes, worker):
    ds = build_dataset(hypes, visualize=False, train=False)
    collate = getattr(ds, "collate_batch_test", ds.collate_batch_train)
    return ds, DataLoader(ds, batch_size=1, shuffle=False, num_workers=worker,
                          collate_fn=collate, pin_memory=True, drop_last=False)

def stats(x):
    a = np.asarray(list(x), dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {k:float("nan") for k in ["mean","p50","p75","p90","p95","p99","max"]} | {"n":0}
    return {"n":int(len(a)),"mean":float(a.mean()),"p50":float(np.quantile(a,.5)),
            "p75":float(np.quantile(a,.75)),"p90":float(np.quantile(a,.9)),
            "p95":float(np.quantile(a,.95)),"p99":float(np.quantile(a,.99)),
            "max":float(a.max())}

def voxel_grouping(pool, gs):
    if gs.n_gaussians == 0: return None
    dev, dt = gs.mean.device, gs.mean.dtype
    vs = pool.voxel_size.to(dev, dt); mn = pool.pc_min.to(dev, dt); grid = pool.grid.to(dev)
    idx = torch.floor((gs.mean-mn)/vs).long()
    valid = ((idx>=0)&(idx<grid)).all(-1)
    kept = gs.index_select(valid)
    if kept.n_gaussians == 0: return None
    idx = idx[valid]
    batch = kept.batch_index if kept.batch_index is not None else torch.zeros(kept.n_gaussians, dtype=torch.long, device=dev)
    gx,gy,gz = [int(v) for v in grid.tolist()]
    code = ((batch*gx+idx[:,0])*gy+idx[:,1])*gz+idx[:,2]
    uniq, inv, cnt = torch.unique(code, return_inverse=True, return_counts=True)
    return kept, inv.long(), cnt.long()

def quat_spread_deg(q):
    q = F.normalize(q.float(), dim=-1)
    ref = q[:1]
    q = q * torch.where((q*ref).sum(-1,keepdim=True)<0,-1.,1.)
    qm = F.normalize(q.mean(0,keepdim=True), dim=-1)
    dot = (q*qm).sum(-1).abs().clamp(0,1)
    return torch.rad2deg(2*torch.acos(dot))

def hetero(grouping, max_vox, rng):
    out = defaultdict(list)
    if grouping is None: return out
    kept, inv, cnt = grouping
    gids = (cnt>1).nonzero(as_tuple=False).squeeze(1).cpu().numpy()
    if len(gids)>max_vox: gids = rng.choice(gids, max_vox, replace=False)
    for gid in gids.tolist():
        rows = (inv==int(gid)).nonzero(as_tuple=False).squeeze(1)
        feat = kept.feature[rows].float()
        fm = F.normalize(feat.mean(0,keepdim=True),dim=-1)
        out["feature_cos_to_mean"] += (F.normalize(feat,dim=-1)*fm).sum(-1).cpu().tolist()
        xyz = kept.mean[rows].float()
        out["xyz_spread_m"] += torch.linalg.vector_norm(xyz-xyz.mean(0,keepdim=True),dim=-1).cpu().tolist()
        sc = kept.scale[rows].float()
        out["scale_spread_m"] += torch.linalg.vector_norm(sc-sc.mean(0,keepdim=True),dim=-1).cpu().tolist()
        out["quat_angle_deg"] += quat_spread_deg(kept.quaternion[rows]).cpu().tolist()
        if kept.p_fg is not None:
            p = kept.p_fg[rows].float()
            out["pfg_std"].append(float(p.std(unbiased=False)))
            out["pfg_range"].append(float(p.max()-p.min()))
            out["pfg_mean"].append(float(p.mean()))
    return out

class Capture:
    def __init__(self, model):
        self.stage2 = None
        self.h = model.stage3.register_forward_pre_hook(self.pre)
    def pre(self, module, inputs): self.stage2 = inputs[0]
    def clear(self): self.stage2 = None
    def close(self): self.h.remove()

@contextlib.contextmanager
def carry_pfg(stage3):
    """
    Preserve pooled p_fg across Stage3._merge() using the real baseline API:
      _merge(pooled) -> (merged: GaussianSet, agent_slices: dict)

    GaussianSet has no generic .replace() method, so use dataclasses.replace.
    """
    orig = stage3._merge

    def patched(self, pooled):
        merged, agent_slices = orig(pooled)

        # gaussian_context._merge concatenates list(pooled.values()) in this order.
        p_list = []
        for gs in pooled.values():
            if gs.n_gaussians == 0:
                continue
            if gs.p_fg is None:
                p = torch.ones(
                    gs.n_gaussians,
                    device=gs.mean.device,
                    dtype=gs.mean.dtype,
                )
            else:
                p = gs.p_fg
            p_list.append(p)

        if merged.n_gaussians > 0:
            if not p_list:
                raise RuntimeError(
                    "Stage3 merged non-empty Gaussians but no pooled p_fg was available"
                )
            p_fg = torch.cat(p_list, dim=0)
            if int(p_fg.shape[0]) != merged.n_gaussians:
                raise RuntimeError(
                    f"p_fg length {int(p_fg.shape[0])} != merged N {merged.n_gaussians}"
                )
            merged = replace(merged, p_fg=p_fg)

        return merged, agent_slices

    stage3._merge = types.MethodType(patched, stage3)
    try:
        yield
    finally:
        stage3._merge = orig


@contextlib.contextmanager
def opacity_splat(stage3):
    """
    Real GaussianBEVSplat.forward signature is forward(gaussians).
    Since splatting is linear in feature, feature <- feature * p_fg is exactly
    the desired p_fg * feature * GaussianKernel ablation.
    """
    splat = stage3.splat
    orig = splat.forward

    def patched(_self, gs):
        if gs.p_fg is None:
            raise RuntimeError("p_fg missing at Stage3 splat")
        p = gs.p_fg.to(
            device=gs.feature.device,
            dtype=gs.feature.dtype,
        ).unsqueeze(-1)
        g2 = gs.replace_feature(gs.feature * p)
        return orig(g2)

    splat.forward = types.MethodType(patched, splat)
    try:
        yield
    finally:
        splat.forward = orig


def forward(model, ego, opacity=False):
    if not opacity: return model(ego)
    with carry_pfg(model.stage3), opacity_splat(model.stage3):
        return model(ego)

def collect_pool(model, loader, device, use_amp, start, nframes, max_vox, seed):
    cap = Capture(model); rng = np.random.default_rng(seed)
    rows=[]; mult={a:[] for a in AGENTS}; hh={a:defaultdict(list) for a in AGENTS}
    done=0; model.eval()
    try:
        with torch.no_grad():
            for fi,batch in enumerate(loader):
                if fi<start: continue
                if done>=nframes: break
                if batch is None: continue
                batch=train_utils.to_device(batch,device); ego=batch["ego"]; ego["epoch"]=0
                cap.clear()
                with amp.autocast(enabled=use_amp): model(ego)
                if cap.stage2 is None: raise RuntimeError("Stage2 capture failed")
                for a in AGENTS:
                    gs=cap.stage2.get(a)
                    if gs is None: continue
                    g=voxel_grouping(model.stage3.pool,gs)
                    if g is None:
                        kept=pooled=0; cnt=torch.empty(0,dtype=torch.long,device=device)
                    else:
                        kept_obj, inv, cnt = g; kept=int(cnt.sum()); pooled=int(cnt.numel())
                        mult[a]+=cnt.cpu().tolist()
                        h=hetero(g,max_vox,rng)
                        for k,v in h.items(): hh[a][k]+=v
                    rows.append({"frame":fi,"agent":a,"stage2_n":int(gs.n_gaussians),
                                 "in_range_n":kept,"pooled_n":pooled,
                                 "stage2_over_pooled":float(gs.n_gaussians)/pooled if pooled else float("nan"),
                                 "multi_voxel_rate":float((cnt>1).float().mean()) if cnt.numel() else float("nan"),
                                 "max_multiplicity":int(cnt.max()) if cnt.numel() else 0})
                done+=1
    finally: cap.close()
    summ={"agents":{}}
    for a in AGENTS:
        ar=[r for r in rows if r["agent"]==a]
        if not ar: summ["agents"][a]={}; continue
        st=sum(r["stage2_n"] for r in ar); po=sum(r["pooled_n"] for r in ar); ke=sum(r["in_range_n"] for r in ar)
        summ["agents"][a]={"frames":len(ar),"stage2_n_total":st,"in_range_n_total":ke,
            "pooled_n_total":po,"global_stage2_over_pooled":st/po if po else float("nan"),
            "multiplicity":stats(mult[a]),"frame_multi_voxel_rate":stats(r["multi_voxel_rate"] for r in ar),
            "heterogeneity":{k:stats(v) for k,v in hh[a].items()}}
    return rows,summ

def init_stat(ious):
    return {
        float(i): {"tp": [], "fp": [], "gt": 0, "score": []}
        for i in ious
    }


def add_ap(stat, pred, score, gt, ious):
    ngt = 0 if gt is None else int(len(gt))
    for iou in ious:
        iou = float(iou)
        if pred is None or score is None or len(pred) == 0:
            stat[iou]["gt"] += ngt
        else:
            eval_utils.caluclate_tp_fp(pred, score, gt, stat, iou)


def ap_safe(stat, iou):
    # calculate_ap mutates tp/fp into cumulative arrays, so evaluate a deepcopy.
    s = copy.deepcopy(stat)
    iou = float(iou)
    if s[iou]["gt"] <= 0:
        return float("nan")
    if len(s[iou]["tp"]) == 0:
        return 0.0
    ap, _, _ = eval_utils.calculate_ap(
        s, iou, global_sort_detections=False
    )
    return float(ap)


def parse_pp(result):
    """
    Match the repository's existing diagnose_three_key_questions.py helper.
    Do not assume a fixed 3-return or 7-return layout; locate GT by shape.
    """
    if not isinstance(result, (tuple, list)):
        raise RuntimeError(f"unexpected post_process type: {type(result)}")
    pred, score = result[0], result[1]
    gt = None
    for item in result[2:]:
        if (
            torch.is_tensor(item)
            and item.ndim == 3
            and tuple(item.shape[-2:]) == (8, 3)
        ):
            gt = item
            break
        if (
            isinstance(item, np.ndarray)
            and item.ndim == 3
            and tuple(item.shape[-2:]) == (8, 3)
        ):
            gt = item
            break
    if gt is None:
        raise RuntimeError("Cannot locate GT [N,8,3] in post_process result")
    return pred, score, gt


def official_pp(dataset, batch_data, output):
    return parse_pp(
        dataset.post_process(
            batch_data,
            OrderedDict(ego=output),
        )
    )


def eval_variant(
    ds,
    loader,
    model,
    device,
    use_amp,
    opacity,
    start,
    nframes,
    ious,
):
    stat = init_stat(ious)
    done = 0
    model.eval()

    with torch.no_grad():
        for fi, batch in enumerate(loader):
            if fi < start:
                continue
            if done >= nframes:
                break
            if batch is None:
                continue

            batch = train_utils.to_device(batch, device)
            ego = batch["ego"]
            ego["epoch"] = 0

            with amp.autocast(enabled=use_amp):
                out = forward(model, ego, opacity)

            pred_box, pred_score, gt_box = official_pp(ds, batch, out)
            add_ap(stat, pred_box, pred_score, gt_box, ious)
            done += 1

    ans = {"n_frames": done}
    for iou in ious:
        iou = float(iou)
        ans[f"AP@{iou}"] = ap_safe(stat, iou)
        ans[f"GT@{iou}"] = int(stat[iou]["gt"])
        ans[f"TP@{iou}"] = int(sum(stat[iou]["tp"]))
        ans[f"FP@{iou}"] = int(sum(stat[iou]["fp"]))

    return ans


def main():
    opt=args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    torch.cuda.set_device(opt.gpu_id); device=torch.device(f"cuda:{opt.gpu_id}"); seed_all(opt.seed)
    ckpt=Path(opt.checkpoint); cfg=resolve_config(ckpt,opt.config)
    hypes=yaml_utils.load_yaml(str(cfg))
    print("[config]",cfg); print("[checkpoint]",ckpt)
    ds,loader=make_loader(hypes,opt.worker)
    model=train_utils.create_model(hypes).to(device); load_ckpt(model,ckpt,device); model.eval()
    if not hasattr(model.stage3,"pool") or not hasattr(model.stage3,"splat"):
        raise RuntimeError("Expected stage3.pool and stage3.splat")
    outdir=Path(opt.out_dir); outdir.mkdir(parents=True,exist_ok=True)
    rows,pool=collect_pool(model,loader,device,opt.amp,opt.start,opt.num_frames,opt.max_hetero_voxels,opt.seed)
    csvp=outdir/"pool_frame_agent.csv"
    with csvp.open("w",newline="") as f:
        fields=sorted({k for r in rows for k in r}); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    ious=[float(x) for x in opt.eval_ious.split(",") if x.strip()]
    ds,loader=make_loader(hypes,opt.worker)
    cur=eval_variant(ds,loader,model,device,opt.amp,False,opt.start,opt.num_frames,ious)
    ds,loader=make_loader(hypes,opt.worker)
    opa=eval_variant(ds,loader,model,device,opt.amp,True,opt.start,opt.num_frames,ious)
    ab={f"AP@{i}":{"current":cur[f"AP@{i}"],"opacity":opa[f"AP@{i}"],
                     "delta":opa[f"AP@{i}"]-cur[f"AP@{i}"]} for i in ious}
    rep={"checkpoint":str(ckpt),"config":str(cfg),"pooling":pool,
         "current_ap":cur,"opacity_ap":opa,"ab":ab}
    jp=outdir/"stage3_pool_opacity_summary.json"; jp.write_text(json.dumps(rep,indent=2,allow_nan=True))
    print("\n=== POOLING ===")
    for a in AGENTS:
        s=pool["agents"].get(a) or {}
        if not s: continue
        m=s["multiplicity"]; print(f"[{a}] {s['stage2_n_total']} -> {s['pooled_n_total']} ratio={s['global_stage2_over_pooled']:.3f}x")
        print(f"  multiplicity p50={m['p50']:.2f} p90={m['p90']:.2f} p95={m['p95']:.2f} p99={m['p99']:.2f} max={m['max']:.0f}")
        h=s["heterogeneity"]
        if "feature_cos_to_mean" in h:
            z=h["feature_cos_to_mean"]; print(f"  feature cos mean={z['mean']:.3f} p50={z['p50']:.3f} p90={z['p90']:.3f}")
        if "xyz_spread_m" in h:
            z=h["xyz_spread_m"]; print(f"  xyz spread mean={z['mean']:.3f}m p90={z['p90']:.3f}m")
        if "pfg_range" in h:
            z=h["pfg_range"]; print(f"  p_fg range mean={z['mean']:.3f} p90={z['p90']:.3f}")
    print("\n=== AP A/B ===")
    for k,v in ab.items(): print(f"{k}: current={v['current']:.4f} opacity={v['opacity']:.4f} delta={v['delta']:+.4f}")
    print("[wrote]",csvp); print("[wrote]",jp)

if __name__=="__main__":
    main()
