#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified AirV2X diagnostic for three questions:

Q1. Does objectness focal loss need fg/bg re-weighting?
    - pos/explicit-neg obj distributions
    - focal loss / gradient pressure for current + counterfactual alpha
    - obj threshold sweep with final AP/TP/FP/recall

Q2. Should semantic/class score be used as an enhancement/filter?
    - correct class layout: [B,A*C,H,W] -> [B,H,W,A,C]
    - compare obj-only ranking, obj*class, sqrt(obj*class), hard class gates

Q3. Are model parameters unhealthy, and are Stage1/Stage2 interactions useful?
    - per-parameter / per-group finite, RMS, max, near-zero audit
    - Stage1/Stage2 feature-change hooks
    - black-box ablations: full / no_stage1 / no_stage2 / no_stage1_no_stage2
    - optional Stage3 residual-alpha=0 only if source explicitly supports it

READ-ONLY: no checkpoint writes, no optimizer step, all temporary monkey patches restored.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import inspect
import json
import math
import os
import sys
from collections import OrderedDict, defaultdict
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
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
from opencood.tools import train_utils
from opencood.utils import box_utils, common_utils
from opencood.utils import eval_utils_opv2v as eval_utils

DEFAULT_MODEL_DIR = (
    "/mnt/home/suyi/AirV2X-Perception_gaussian/opencood/logs/"
    "airv2x_gaussian_0822_objfocal/v45_objfocal_2026_09_13_21_11_11"
)


def trms(x):
    if x is None or not torch.is_tensor(x) or x.numel() == 0:
        return None

    y = x.detach().float().reshape(-1)
    y = y[torch.isfinite(y)]

    if y.numel() == 0:
        return None

    return float(torch.sqrt(torch.mean(y * y)).item())


def safe_delta_ratio(x, y):
    if (
        x is None
        or y is None
        or not torch.is_tensor(x)
        or not torch.is_tensor(y)
    ):
        return None

    if x.shape != y.shape or x.numel() == 0:
        return None

    base = trms(x)
    delta = trms(y - x)

    if base is None or delta is None:
        return None

    if not math.isfinite(base) or not math.isfinite(delta):
        return None

    return delta / max(base, 1e-12)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--epoch", type=int, default=13)
    p.add_argument("--num_frames", type=int, default=50)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--eval_ious", default="0.3,0.5,0.7")

    # Q1
    p.add_argument("--obj_thresholds", default="0.05,0.08,0.10,0.15,0.20,0.25")
    p.add_argument("--obj_alphas", default="0.25,0.35,0.50")

    # Q2
    p.add_argument("--semantic_obj_thresholds", default="0.10,0.15")
    p.add_argument("--semantic_class_thresholds", default="0.03,0.05,0.08,0.10")

    # Q3
    p.add_argument("--interaction_frames", type=int, default=12)
    p.add_argument("--interaction_obj_threshold", type=float, default=None)

    p.add_argument("--out", default=None)
    return p.parse_args()


def flist(s):
    return sorted({float(x.strip()) for x in s.split(",") if x.strip()})


def find_label_dict(obj):
    if isinstance(obj, Mapping):
        if "pos_equal_one" in obj and "neg_equal_one" in obj:
            return obj
        for v in obj.values():
            f = find_label_dict(v)
            if f is not None:
                return f
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            f = find_label_dict(v)
            if f is not None:
                return f
    return None


def loss_args_from_hypes(hypes):
    loss = hypes.get("loss", {})
    if isinstance(loss, Mapping) and isinstance(loss.get("args"), Mapping):
        return loss["args"]
    return loss if isinstance(loss, Mapping) else {}


def stats_np(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "p10": float(np.quantile(x, .10)),
        "median": float(np.quantile(x, .50)),
        "p90": float(np.quantile(x, .90)),
        "p99": float(np.quantile(x, .99)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def init_stat(ious):
    return {float(i): {"tp": [], "fp": [], "gt": 0, "score": []} for i in ious}


def add_ap(stat, pred, score, gt, ious):
    ngt = 0 if gt is None else int(len(gt))
    for iou in ious:
        iou = float(iou)
        if pred is None or score is None or len(pred) == 0:
            stat[iou]["gt"] += ngt
        else:
            eval_utils.caluclate_tp_fp(pred, score, gt, stat, iou)


def ap_safe(stat, iou):
    s = copy.deepcopy(stat)
    iou = float(iou)
    if s[iou]["gt"] <= 0:
        return float("nan")
    if len(s[iou]["tp"]) == 0:
        return 0.0
    try:
        ap, _, _ = eval_utils.calculate_ap(s, iou, global_sort_detections=False)
        return float(ap)
    except Exception:
        return float("nan")


def parse_pp(result):
    if not isinstance(result, (tuple, list)):
        raise RuntimeError(type(result))
    pred, score = result[0], result[1]
    gt = None
    for item in result[2:]:
        if torch.is_tensor(item) and item.ndim == 3 and tuple(item.shape[-2:]) == (8, 3):
            gt = item
            break
        if isinstance(item, np.ndarray) and item.ndim == 3 and tuple(item.shape[-2:]) == (8, 3):
            gt = item
            break
    if gt is None:
        raise RuntimeError("Cannot locate GT [N,8,3] in post_process result")
    return pred, score, gt


def official_pp(dataset, batch_data, output):
    return parse_pp(dataset.post_process(batch_data, OrderedDict(ego=output)))


def greedy_counts(pred, score, gt, iou):
    ngt = 0 if gt is None else int(gt.shape[0])
    if pred is None or score is None or int(pred.shape[0]) == 0:
        return 0, 0, ngt
    det_np = common_utils.torch_tensor_to_numpy(pred)
    score_np = common_utils.torch_tensor_to_numpy(score)
    gt_np = common_utils.torch_tensor_to_numpy(gt)
    det_polys = list(common_utils.convert_format(det_np))
    gt_polys = list(common_utils.convert_format(gt_np))
    available = list(range(len(gt_polys)))
    tp = fp = 0
    for di in np.argsort(-score_np):
        if not available:
            fp += 1
            continue
        cand = [gt_polys[j] for j in available]
        ious = common_utils.compute_iou(det_polys[int(di)], cand)
        if len(ious) == 0 or float(np.max(ious)) < float(iou):
            fp += 1
            continue
        available.pop(int(np.argmax(ious)))
        tp += 1
    return tp, fp, ngt


# ------------------------------ Q1 -----------------------------------
def focal_pressure(obj_logits, pos_mask, neg_mask, alpha, gamma):
    z = obj_logits.detach().float().clone().requires_grad_(True)
    pos = (pos_mask > .5).to(z.device, z.dtype)
    neg = (neg_mask > .5).to(z.device, z.dtype)
    valid = (pos + neg).clamp(max=1.)
    target = pos
    bce = F.binary_cross_entropy_with_logits(z, target, reduction="none")
    p = torch.sigmoid(z)
    alpha_t = target * alpha + (1. - target) * (1. - alpha)
    pt_err = target * (1. - p) + (1. - target) * p
    raw = alpha_t * pt_err.pow(gamma) * bce * valid
    npos = pos.sum().clamp(min=1.)
    loss = raw.sum() / npos
    grad = torch.autograd.grad(loss, z)[0].detach()
    pb, nb = pos > .5, neg > .5
    pl = float(raw[pb].sum().item() / npos.item())
    nl = float(raw[nb].sum().item() / npos.item())
    pg, ng = grad[pb].abs(), grad[nb].abs()
    return {
        "loss_total": float(loss.detach().item()),
        "loss_pos": pl,
        "loss_neg": nl,
        "grad_mass_pos": float(pg.sum().item()) if pg.numel() else 0.,
        "grad_mass_neg": float(ng.sum().item()) if ng.numel() else 0.,
        "grad_mean_pos": float(pg.mean().item()) if pg.numel() else 0.,
        "grad_mean_neg": float(ng.mean().item()) if ng.numel() else 0.,
        "num_pos": int(pb.sum().item()),
        "num_neg": int(nb.sum().item()),
    }


# ------------------------------ Q2 -----------------------------------
def class_scores_correct(output, num_class):
    psm = output["psm"]
    B, AC, H, W = psm.shape
    C = int(num_class)
    A = AC // C
    if A * C != AC:
        raise RuntimeError(f"AC={AC}, C={C}")
    prob = torch.sigmoid(
        psm.permute(0, 2, 3, 1).contiguous().view(B, H, W, A, C)
    )
    fg, lab0 = torch.max(prob[..., 1:], dim=-1)
    return fg.reshape(B, -1), (lab0 + 1).reshape(B, -1), prob[..., 0].reshape(B, -1)


def trace_postprocess(batch_data, output, pp, obj_t, rank_mode="obj", cls_t=None):
    cav = batch_data["ego"]
    device = output["obj"].device
    T = cav["transformation_matrix"]
    anchors = cav["anchor_box"]
    if torch.is_tensor(T): T = T.to(device)
    if torch.is_tensor(anchors): anchors = anchors.to(device)

    obj = torch.sigmoid(output["obj"].permute(0, 2, 3, 1).contiguous()).reshape(1, -1)
    fg, labels, _ = class_scores_correct(output, int(pp.num_class))
    gate = obj[0] > float(obj_t)
    if cls_t is not None:
        gate = gate & (fg[0] > float(cls_t))

    if rank_mode == "obj":
        rank = obj[0]
    elif rank_mode == "product":
        rank = obj[0] * fg[0]
    elif rank_mode == "sqrt_product":
        rank = torch.sqrt((obj[0] * fg[0]).clamp_min(0.))
    else:
        raise ValueError(rank_mode)

    decoded = pp.delta_to_boxes3d(output["rm"], anchors)[0]
    decoded, score, lab = decoded[gate], rank[gate], labels[0][gate]
    cand = int(gate.sum().item())
    if decoded.numel() == 0:
        return None, None, cand, 0, 0

    corners = box_utils.boxes_to_corners_3d(decoded, order=pp.params["order"])
    proj = box_utils.project_box3d(corners, T)
    k1 = box_utils.remove_large_pred_bbx(proj, pp.dataset)
    zmin, zmax = pp.lidar_range[2], pp.lidar_range[5]
    try:
        k2 = box_utils.remove_bbx_abnormal_z(proj, z_min=zmin, z_max=zmax)
    except TypeError:
        k2 = box_utils.remove_bbx_abnormal_z(proj)
    keep = torch.logical_and(k1, k2)
    proj, score, lab = proj[keep], score[keep], lab[keep]
    pre = int(proj.shape[0])
    if pre:
        kn = box_utils.nms_rotated(proj, score, pp.params["nms_thresh"])
        proj, score, lab = proj[kn], score[kn], lab[kn]
    post = int(proj.shape[0])
    if post:
        kr = box_utils.get_mask_for_boxes_within_range_torch(proj, pp.lidar_range)
        proj, score, lab = proj[kr], score[kr], lab[kr]
    return proj, score, cand, pre, int(proj.shape[0])


# ------------------------------ Q3 -----------------------------------
def param_group(name):
    for prefix, group in [
        ("intra_view", "stage1_intra_view"),
        ("cross_agent", "stage2_cross_agent"),
        ("stage3", "stage3"),
        ("backbone", "bev_backbone"),
        ("shrink_conv", "shrink_conv"),
        ("cls_head", "cls_head"),
        ("reg_head", "reg_head"),
        ("obj_head", "obj_head"),
    ]:
        if name.startswith(prefix): return group
    if name.startswith(("frontend", "highres", "heatmap_heads", "depth_heads", "depth_moments", "drone_")):
        return "p1_frontend"
    return name.split(".")[0]


def audit_params(model):
    rows, groups = [], defaultdict(lambda: dict(n=0, train=0, ss=0., sa=0., max=0., nz=0, finite=0, total=0, suspicious=0, tensors=0))
    for name, p in model.named_parameters():
        x = p.detach().float().reshape(-1)
        fin = torch.isfinite(x)
        xf = x[fin]
        n, nf = x.numel(), int(fin.sum().item())
        if nf:
            rms = float(torch.sqrt(torch.mean(xf * xf)).item())
            ma = float(xf.abs().mean().item())
            mx = float(xf.abs().max().item())
            nz = float((xf.abs() < 1e-8).float().mean().item())
        else:
            rms = ma = mx = nz = float("nan")
        suspicious = (nf != n) or (p.requires_grad and nf and rms < 1e-8) or (nf and mx > 1e3)
        g = param_group(name)
        rows.append({
            "name": name, "group": g, "shape": str(tuple(p.shape)), "numel": int(n),
            "requires_grad": bool(p.requires_grad), "finite_fraction": float(nf/max(n,1)),
            "rms": rms, "mean_abs": ma, "max_abs": mx, "near_zero_fraction_1e8": nz,
            "suspicious": bool(suspicious),
        })
        a = groups[g]
        a["n"] += n; a["train"] += n if p.requires_grad else 0; a["tensors"] += 1
        a["finite"] += nf; a["total"] += n; a["suspicious"] += int(suspicious)
        if nf:
            a["ss"] += float((xf*xf).sum().item()); a["sa"] += float(xf.abs().sum().item())
            a["max"] = max(a["max"], mx); a["nz"] += int((xf.abs()<1e-8).sum().item())
    grows = []
    for g, a in sorted(groups.items()):
        nf = max(a["finite"], 1)
        grows.append({
            "group": g, "tensor_count": a["tensors"], "numel": a["n"], "trainable_numel": a["train"],
            "finite_fraction": float(a["finite"]/max(a["total"],1)),
            "rms": float(math.sqrt(a["ss"]/nf)), "mean_abs": float(a["sa"]/nf),
            "max_abs": float(a["max"]), "near_zero_fraction_1e8": float(a["nz"]/nf),
            "suspicious_tensor_count": a["suspicious"],
        })
    return rows, grows


def gs_feature(x):
    if x is None: return None
    for k in ("feature", "features"):
        v = getattr(x, k, None)
        if torch.is_tensor(v): return v
    return None


def trms(x):
    if x is None or not torch.is_tensor(x) or x.numel()==0: return None
    y = x.detach().float()
    return float(torch.sqrt(torch.mean(y*y)).item())


def activation_hooks(model, raw):
    hs = []
    def h1(m, inp, out):
        x, y = gs_feature(inp[0]) if inp else None, gs_feature(out)
        d = None
        if x is not None and y is not None and x.shape == y.shape:
            d = safe_delta_ratio(x, y)
        raw.append({"module":"stage1_intra_view","in_rms":trms(x),"out_rms":trms(y),"delta_over_input_rms":d})
    def h2(m, inp, out):
        a = inp[0] if inp else None
        if not isinstance(a, Mapping) or not isinstance(out, Mapping): return
        for k in set(a.keys()) & set(out.keys()):
            x, y = gs_feature(a[k]), gs_feature(out[k])
            if x is None or y is None: continue
            d = safe_delta_ratio(x, y) if x.shape==y.shape else None
            raw.append({"module":f"stage2_cross_agent:{k}","in_rms":trms(x),"out_rms":trms(y),"delta_over_input_rms":d})
    if hasattr(model,"intra_view") and isinstance(model.intra_view, torch.nn.Module): hs.append(model.intra_view.register_forward_hook(h1))
    if hasattr(model,"cross_agent") and isinstance(model.cross_agent, torch.nn.Module): hs.append(model.cross_agent.register_forward_hook(h2))
    return hs


def aggregate_acts(raw):
    d = defaultdict(list)
    for r in raw: d[r["module"]].append(r)
    out=[]
    for mod, rr in sorted(d.items()):
        def av(k):
            x=[float(r[k]) for r in rr if r.get(k) is not None and math.isfinite(float(r[k]))]
            return float(np.mean(x)) if x else None
        out.append({"module":mod,"calls":len(rr),"in_rms_mean":av("in_rms"),"out_rms_mean":av("out_rms"),"delta_over_input_rms_mean":av("delta_over_input_rms")})
    return out


@contextlib.contextmanager
def patch_interaction(model, variant):
    oi = oc = None
    old_alpha_exists = False; old_alpha = None
    try:
        if variant in ("no_stage1","no_stage1_no_stage2"):
            oi = model.intra_view.forward
            model.intra_view.forward = lambda gs, *a, **k: gs
        if variant in ("no_stage2","no_stage1_no_stage2"):
            oc = model.cross_agent.forward
            model.cross_agent.forward = lambda gs, *a, **k: gs
        if variant == "stage3_residual_alpha0":
            src = inspect.getsource(model.stage3.forward)
            if "debug_residual_alpha" not in src:
                raise RuntimeError("stage3 does not explicitly support debug_residual_alpha")
            old_alpha_exists = hasattr(model.stage3,"debug_residual_alpha")
            old_alpha = getattr(model.stage3,"debug_residual_alpha",None)
            model.stage3.debug_residual_alpha = 0.0
        yield
    finally:
        if oi is not None: model.intra_view.forward = oi
        if oc is not None: model.cross_agent.forward = oc
        if variant == "stage3_residual_alpha0" and hasattr(model,"stage3"):
            if old_alpha_exists: model.stage3.debug_residual_alpha = old_alpha
            elif hasattr(model.stage3,"debug_residual_alpha"): delattr(model.stage3,"debug_residual_alpha")


def aggregate_rows(rows, ious):
    n=max(len(rows),1)
    out={"frames":len(rows),"candidate_per_frame":float(np.mean([r["candidate_count"] for r in rows])) if rows else 0.,"pred_per_frame":float(np.mean([r["pred_count"] for r in rows])) if rows else 0.}
    for i in ious:
        k=f"{i:.2f}"; tp=sum(r[f"tp_{k}"] for r in rows); fp=sum(r[f"fp_{k}"] for r in rows); gt=sum(r[f"gt_{k}"] for r in rows)
        out[f"tp_per_frame_iou{k}"]=tp/n; out[f"fp_per_frame_iou{k}"]=fp/n
        out[f"precision_iou{k}"]=tp/max(tp+fp,1); out[f"recall_iou{k}"]=tp/max(gt,1)
    return out


def write_csv(path, rows):
    if not rows: return
    keys=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); keys.append(k)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    args=parse_args()
    ious=flist(args.eval_ious)
    obj_ts=flist(args.obj_thresholds)
    alpha_list=flist(args.obj_alphas)
    sem_obj_ts=flist(args.semantic_obj_thresholds)
    sem_cls_ts=flist(args.semantic_class_thresholds)
    outdir=args.out or os.path.join(args.model_dir,f"three_questions_epoch{args.epoch}")
    os.makedirs(outdir,exist_ok=True)

    hypes=yaml_utils.load_yaml(None,args)
    if "test_dir" in hypes: hypes["validate_dir"]=hypes["test_dir"]
    largs=loss_args_from_hypes(hypes)
    cfg_alpha=float(largs.get("obj_alpha",.25)); cfg_gamma=float(largs.get("obj_gamma",2.)); cfg_ow=float(largs.get("obj_weight",1.))

    dataset=build_dataset(hypes,visualize=False,train=False)
    loader=DataLoader(dataset,batch_size=1,num_workers=args.workers,collate_fn=dataset.collate_batch_test,shuffle=False,pin_memory=False,drop_last=False)
    model=train_utils.create_model(hypes)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); _,model=train_utils.load_model(args.model_dir,model,args.epoch,start_from_best=False); model.eval(); model.zero_grad(set_to_none=True)
    pp=dataset.post_processor
    prod_t=float(pp.params["target_args"]["obj_threshold"])
    inter_t=prod_t if args.interaction_obj_threshold is None else float(args.interaction_obj_threshold)

    # Q3 static + full-model activation hooks
    p_tensor,p_group=audit_params(model)
    act_raw=[]; hs=activation_hooks(model,act_raw)

    # Q1/Q2 storage
    pos_chunks=[]; neg_chunks=[]; ign_chunks=[]
    press_sum=defaultdict(lambda:defaultdict(float)); press_n=defaultdict(int)
    q1_frames=defaultdict(list); q1_stats={t:init_stat(ious) for t in obj_ts}

    q2_specs=[]
    for ot in sem_obj_ts:
        q2_specs += [("obj_rank",ot,"obj",None),("obj_x_cls_rank",ot,"product",None),("sqrt_obj_x_cls_rank",ot,"sqrt_product",None)]
        q2_specs += [(f"hard_cls_gate_{ct:.2f}",ot,"obj",ct) for ct in sem_cls_ts]
    q2_frames=defaultdict(list); q2_stats={(v,ot,ct):init_stat(ious) for v,ot,rm,ct in q2_specs}

    end=args.start+args.num_frames
    try:
        for fi,batch in tqdm(enumerate(loader),total=min(len(loader),end),desc="Q1/Q2"):
            if fi<args.start: continue
            if fi>=end: break
            batch=train_utils.to_device(batch,device)
            with torch.no_grad(): out=model(batch["ego"])
            pp.params["target_args"]["obj_threshold"]=prod_t
            _,_,gt=official_pp(dataset,batch,out)

            ld=find_label_dict(batch["ego"])
            if ld is None: raise RuntimeError("label dict not found")
            pos,neg=ld["pos_equal_one"],ld["neg_equal_one"]
            logits=out["obj"].permute(0,2,3,1).contiguous()
            prob=torch.sigmoid(logits).reshape(-1)
            pf=pos.reshape(-1)>.5; nf=neg.reshape(-1)>.5; ig=(~pf)&(~nf)
            pos_chunks.append(prob[pf].detach().cpu().numpy()); neg_chunks.append(prob[nf].detach().cpu().numpy()); ign_chunks.append(prob[ig].detach().cpu().numpy())

            for a in sorted(set(alpha_list+[cfg_alpha])):
                z=focal_pressure(logits,pos,neg,a,cfg_gamma); press_n[a]+=1
                for k,v in z.items(): press_sum[a][k]+=float(v)

            for t in obj_ts:
                pred,score,cand,pre,final=trace_postprocess(batch,out,pp,t,"obj",None)
                add_ap(q1_stats[t],pred,score,gt,ious)
                r={"candidate_count":cand,"pred_count":final}
                for i in ious:
                    tp,fp,ng=greedy_counts(pred,score,gt,i); k=f"{i:.2f}"; r[f"tp_{k}"]=tp; r[f"fp_{k}"]=fp; r[f"gt_{k}"]=ng
                q1_frames[t].append(r)

            for v,ot,rm,ct in q2_specs:
                pred,score,cand,pre,final=trace_postprocess(batch,out,pp,ot,rm,ct)
                key=(v,ot,ct); add_ap(q2_stats[key],pred,score,gt,ious)
                r={"candidate_count":cand,"pred_count":final}
                for i in ious:
                    tp,fp,ng=greedy_counts(pred,score,gt,i); k=f"{i:.2f}"; r[f"tp_{k}"]=tp; r[f"fp_{k}"]=fp; r[f"gt_{k}"]=ng
                q2_frames[key].append(r)
    finally:
        pp.params["target_args"]["obj_threshold"]=prod_t
        for h in hs: h.remove()

    cat=lambda xs: np.concatenate([x for x in xs if x.size]) if any(x.size for x in xs) else np.empty(0)
    obj_dist={"positive":stats_np(cat(pos_chunks)),"explicit_negative":stats_np(cat(neg_chunks)),"ignore":stats_np(cat(ign_chunks))}

    q1_pressure=[]
    for a in sorted(press_n):
        n=max(press_n[a],1); row={"alpha":a,"gamma":cfg_gamma,"frames":press_n[a],"is_configured_alpha":abs(a-cfg_alpha)<1e-12}
        for k,v in press_sum[a].items(): row[k]=v/n
        row["grad_mass_pos_over_neg"]=row.get("grad_mass_pos",0.)/max(row.get("grad_mass_neg",0.),1e-12)
        row["grad_mean_pos_over_neg"]=row.get("grad_mean_pos",0.)/max(row.get("grad_mean_neg",0.),1e-12)
        row["loss_pos_over_neg"]=row.get("loss_pos",0.)/max(row.get("loss_neg",0.),1e-12)
        q1_pressure.append(row)

    q1=[]
    for t in obj_ts:
        r={"obj_threshold":t,**aggregate_rows(q1_frames[t],ious)}
        for i in ious: r[f"ap_iou{i:.2f}"]=ap_safe(q1_stats[t],i)
        q1.append(r)

    q2=[]
    seen=set()
    for v,ot,rm,ct in q2_specs:
        key=(v,ot,ct)
        if key in seen: continue
        seen.add(key)
        r={"variant":v,"obj_threshold":ot,"class_threshold":ct,"rank_mode":rm,**aggregate_rows(q2_frames[key],ious)}
        for i in ious: r[f"ap_iou{i:.2f}"]=ap_safe(q2_stats[key],i)
        q2.append(r)

    # Q3 interaction ablation in second pass
    variants=["full","no_stage1","no_stage2","no_stage1_no_stage2"]
    stage3_supported=False
    if hasattr(model,"stage3"):
        try: stage3_supported="debug_residual_alpha" in inspect.getsource(model.stage3.forward)
        except Exception: stage3_supported=False
    if stage3_supported: variants.append("stage3_residual_alpha0")
    q3s={v:init_stat(ious) for v in variants}; q3f=defaultdict(list)
    loader2=DataLoader(dataset,batch_size=1,num_workers=args.workers,collate_fn=dataset.collate_batch_test,shuffle=False,pin_memory=False,drop_last=False)
    stop=args.start+min(args.interaction_frames,args.num_frames)
    try:
        for fi,batch in tqdm(enumerate(loader2),total=min(len(loader2),stop),desc="Q3 interactions"):
            if fi<args.start: continue
            if fi>=stop: break
            batch=train_utils.to_device(batch,device)
            with torch.no_grad(): full=model(batch["ego"])
            pp.params["target_args"]["obj_threshold"]=prod_t
            _,_,gt=official_pp(dataset,batch,full)
            for v in variants:
                if v=="full": out=full
                else:
                    with patch_interaction(model,v):
                        with torch.no_grad(): out=model(batch["ego"])
                pred,score,cand,pre,final=trace_postprocess(batch,out,pp,inter_t,"obj",None)
                add_ap(q3s[v],pred,score,gt,ious)
                r={"candidate_count":cand,"pred_count":final}
                for i in ious:
                    tp,fp,ng=greedy_counts(pred,score,gt,i); k=f"{i:.2f}"; r[f"tp_{k}"]=tp; r[f"fp_{k}"]=fp; r[f"gt_{k}"]=ng
                q3f[v].append(r)
    finally:
        pp.params["target_args"]["obj_threshold"]=prod_t

    q3=[]
    for v in variants:
        r={"variant":v,"obj_threshold":inter_t,**aggregate_rows(q3f[v],ious)}
        for i in ious: r[f"ap_iou{i:.2f}"]=ap_safe(q3s[v],i)
        q3.append(r)

    acts=aggregate_acts(act_raw)
    fullrow=next((r for r in q3 if r["variant"]=="full"),None)
    q3_delta=[]
    if fullrow:
        for r in q3:
            if r["variant"]=="full": continue
            q3_delta.append({"variant":r["variant"],"delta_ap30_vs_full":r.get("ap_iou0.30",float("nan"))-fullrow.get("ap_iou0.30",float("nan")),"delta_ap50_vs_full":r.get("ap_iou0.50",float("nan"))-fullrow.get("ap_iou0.50",float("nan")),"delta_recall30_vs_full":r.get("recall_iou0.30",float("nan"))-fullrow.get("recall_iou0.30",float("nan"))})

    summary={
        "model_dir":args.model_dir,"epoch":args.epoch,"production_obj_threshold":prod_t,
        "configured_obj_loss":{"obj_alpha":cfg_alpha,"obj_gamma":cfg_gamma,"obj_weight":cfg_ow},
        "Q1_obj_internal_weighting":{
            "question":"Does objectness focal fg/bg weighting need adjustment?",
            "obj_distributions":obj_dist,"focal_pressure":q1_pressure,"threshold_sweep":q1,
            "notes":["Counterfactual alpha pressure is evaluated on current logits only; it is not a prediction of retrained AP.","Use negative p99/FP together with positive recall and configured-alpha grad mass before changing alpha."]},
        "Q2_semantic_filtering":{
            "question":"Should corrected semantic/class score be used as ranking/filtering enhancement?",
            "class_layout":"[B,A*C,H,W] -> permute(B,H,W,A*C) -> view(B,H,W,A,C)","results":q2,
            "notes":["Prefer reranking if AP30/AP50 rises without recall loss.","Use hard gate only if TP/recall loss is small."]},
        "Q3_model_and_interactions":{
            "question":"Are model parameters unhealthy and are interaction modules causally useful?",
            "parameter_groups":p_group,"suspicious_parameters":[r for r in p_tensor if r["suspicious"]],"activations":acts,
            "interaction_ablation":q3,"ablation_deltas_vs_full":q3_delta,"stage3_alpha0_supported":stage3_supported,
            "notes":["Parameter norms alone do not prove interaction quality; black-box ablation is primary.","If removing a stage improves AP materially, that stage is causally harmful on tested frames.","If removing a stage barely changes AP and feature delta is tiny, it may be weak/bypassed."]}
    }

    with open(os.path.join(outdir,"summary.json"),"w",encoding="utf-8") as f: json.dump(summary,f,indent=2,ensure_ascii=False,allow_nan=True)
    write_csv(os.path.join(outdir,"q1_obj_threshold_sweep.csv"),q1)
    write_csv(os.path.join(outdir,"q1_obj_pressure.csv"),q1_pressure)
    write_csv(os.path.join(outdir,"q2_semantic_ablation.csv"),q2)
    write_csv(os.path.join(outdir,"q3_parameter_groups.csv"),p_group)
    write_csv(os.path.join(outdir,"q3_parameter_tensors.csv"),p_tensor)
    write_csv(os.path.join(outdir,"q3_activation_stats.csv"),acts)
    write_csv(os.path.join(outdir,"q3_interaction_ablation.csv"),q3)

    print("\n=== Q1 OBJ WEIGHTING ===")
    print("positive:",obj_dist["positive"]); print("negative:",obj_dist["explicit_negative"])
    for r in q1_pressure:
        mark=" *configured" if r["is_configured_alpha"] else ""
        print(f"alpha={r['alpha']:.2f} gradMass(pos/neg)={r['grad_mass_pos_over_neg']:.4f} gradMean(pos/neg)={r['grad_mean_pos_over_neg']:.2f} loss(pos/neg)={r['loss_pos_over_neg']:.4f}{mark}")
    for r in q1:
        print(f"T={r['obj_threshold']:.2f} FP/fr={r.get('fp_per_frame_iou0.30',float('nan')):.2f} R30={r.get('recall_iou0.30',float('nan')):.3f} AP30={r.get('ap_iou0.30',float('nan')):.3f} AP50={r.get('ap_iou0.50',float('nan')):.3f}")

    print("\n=== Q2 SEMANTIC ===")
    for r in q2:
        ct="-" if r["class_threshold"] is None else f"{r['class_threshold']:.2f}"
        print(f"{r['variant']:>22s} objT={r['obj_threshold']:.2f} clsT={ct:>4s} FP/fr={r.get('fp_per_frame_iou0.30',float('nan')):.2f} R30={r.get('recall_iou0.30',float('nan')):.3f} AP30={r.get('ap_iou0.30',float('nan')):.3f} AP50={r.get('ap_iou0.50',float('nan')):.3f}")

    print("\n=== Q3 PARAMETERS / INTERACTIONS ===")
    print("suspicious parameter tensors:",sum(int(r["suspicious"]) for r in p_tensor))
    for r in q3:
        print(f"{r['variant']:>24s} AP30={r.get('ap_iou0.30',float('nan')):.3f} AP50={r.get('ap_iou0.50',float('nan')):.3f} R30={r.get('recall_iou0.30',float('nan')):.3f} FP/fr={r.get('fp_per_frame_iou0.30',float('nan')):.2f}")
    print("\nSaved:",outdir)


if __name__ == "__main__":
    main()
