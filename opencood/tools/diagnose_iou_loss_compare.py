#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive OLD(no IoU loss) vs NEW(+IoU loss) AirV2X Gaussian diagnostic.

Default checkpoints
-------------------
OLD:
  /mnt/home/suyi/AirV2X-Perception_gaussian/opencood/logs/
  airv2x_gaussian_0822_objfocal/v45_objfocal_2026_09_13_21_11_11/net_epoch13.pth
NEW:
  /mnt/home/suyi/AirV2X-Perception_gaussian/opencood/logs/
  airv2x_gaussian_0822_objfocal/v45_objfocal_2026_09_15_11_59_01/net_epoch17.pth

READ-ONLY. It does not update checkpoints, optimizer state, or source code.

Diagnostics
-----------
A) IoU-loss training-path / gradient audit
   * configured reg / IoU weights
   * exact anchor_box visibility to the active loss branch
   * raw + weighted regression / IoU losses
   * IoU requires_grad / grad_fn
   * grad norm to rm and reg_head.weight
   * IoU/reg gradient ratio and cosine

B) Localization OLD vs NEW
   * paired positive-anchor 3D IoU
   * paired positive-anchor BEV IoU
   * each GT's best BEV IoU among label-positive decoded boxes
   * positive-decoded GT coverage @ .3/.5/.7

C) Score-quality alignment
   * Spearman(obj, max-GT-IoU)
   * Spearman(foreground-class-score, max-GT-IoU)
   * score distributions by IoU bin
   * corrected class channel layout [B,A*C,H,W] -> [B,H,W,A,C]

D) NMS/ranking bottleneck
   * normal objectness NMS
   * obj*class NMS
   * hard semantic gate
   * oracle-NMS selection using true max-GT-IoU, final score still obj
   * full oracle selection+ranking upper bound

E) Regression component oracle
   * base, GT XY, GT Z, GT WL, GT HWL, GT yaw,
     GT XY+yaw, GT XY+WL, full GT

F) Gaussian geometry/update audit
   * Init / Stage1 / Stage2 sigma and scale stats
   * Init->Stage1 and Stage1->Stage2 mean / log-scale / rotation updates
   * configured delta-range proximity
   * Stage1 / Stage2 feature delta RMS ratio

G) Parameter health + config-diff control

Recommended command:
  cd /data2/home/suyi/AirV2X-Perception_gaussian
  python opencood/tools/diagnose_iou_loss_compare.py \
    --num_frames 50 --grad_frames 3 --geometry_frames 20 --workers 4
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
from collections import OrderedDict, defaultdict
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from opencood.loss.point_pillar_loss_multiclass import PointPillarLossMultiClass

OLD_DIR = (
    "/mnt/home/suyi/AirV2X-Perception_gaussian/opencood/logs/"
    "airv2x_gaussian_0822_objfocal/v45_objfocal_2026_09_13_21_11_11"
)
NEW_DIR = (
    "/mnt/home/suyi/AirV2X-Perception_gaussian/opencood/logs/"
    "airv2x_gaussian_0822_objfocal/v45_objfocal_2026_09_15_11_59_01"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--old_dir", default=OLD_DIR)
    p.add_argument("--old_epoch", type=int, default=13)
    p.add_argument("--new_dir", default=NEW_DIR)
    p.add_argument("--new_epoch", type=int, default=17)
    p.add_argument("--num_frames", type=int, default=50)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--quality_obj_thresholds", default="0.10,0.15,0.20")
    p.add_argument("--eval_ious", default="0.3,0.5,0.7")
    p.add_argument("--hard_class_threshold", type=float, default=0.10)
    p.add_argument("--grad_frames", type=int, default=3)
    p.add_argument("--geometry_frames", type=int, default=20)
    p.add_argument("--delta_bound_tol", type=float, default=0.01)
    p.add_argument("--out", default=None)
    return p.parse_args()


def flist(s):
    return sorted({float(x.strip()) for x in str(s).split(",") if x.strip()})


def write_csv(path, rows):
    if not rows:
        return
    keys, seen = [], set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def finite_np(x):
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    return a[np.isfinite(a)]


def stats_np(x):
    a = finite_np(x)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size), "mean": float(a.mean()),
        "p10": float(np.quantile(a, .10)), "p25": float(np.quantile(a, .25)),
        "median": float(np.quantile(a, .50)), "p75": float(np.quantile(a, .75)),
        "p90": float(np.quantile(a, .90)), "p99": float(np.quantile(a, .99)),
        "min": float(a.min()), "max": float(a.max()),
    }


def safe_mean(xs):
    a = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(np.mean(a)) if a else None


def safe_ratio(a, b):
    if a is None or b is None:
        return None
    a, b = float(a), float(b)
    if not math.isfinite(a) or not math.isfinite(b):
        return None
    return float(a / max(abs(b), 1e-12))


def tensor_rms(x):
    if x is None or not torch.is_tensor(x) or x.numel() == 0:
        return None
    y = x.detach().float().reshape(-1)
    y = y[torch.isfinite(y)]
    if y.numel() == 0:
        return None
    return float(torch.sqrt(torch.mean(y * y)).item())


def grad_norm(g):
    if g is None or not torch.is_tensor(g) or g.numel() == 0:
        return 0.0
    x = g.detach().float().reshape(-1)
    x = x[torch.isfinite(x)]
    return 0.0 if x.numel() == 0 else float(torch.linalg.vector_norm(x).item())


def cosine_flat(a, b):
    if a is None or b is None:
        return None
    a, b = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    m = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[m], b[m]
    if a.numel() == 0:
        return None
    na, nb = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    if float(na.item()) < 1e-12 or float(nb.item()) < 1e-12:
        return None
    return float(torch.dot(a, b).item() / (float(na.item()) * float(nb.item())))


def find_label_dict(obj):
    if isinstance(obj, Mapping):
        if "pos_equal_one" in obj and "neg_equal_one" in obj and "targets" in obj:
            return obj
        for v in obj.values():
            z = find_label_dict(v)
            if z is not None:
                return z
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            z = find_label_dict(v)
            if z is not None:
                return z
    return None


def local_args(base, model_dir):
    x = copy.copy(base)
    x.model_dir = model_dir
    return x


def load_hypes(base, model_dir):
    h = yaml_utils.load_yaml(None, local_args(base, model_dir))
    if "test_dir" in h:
        h["validate_dir"] = h["test_dir"]
    return h


def loss_args(h):
    x = h.get("loss", {})
    if not isinstance(x, Mapping):
        return {}
    if "det" in x and isinstance(x["det"], Mapping):
        d = x["det"]
        return d.get("args", d) if isinstance(d.get("args", d), Mapping) else {}
    if isinstance(x.get("args"), Mapping):
        return x["args"]
    return x


def model_args(h):
    x = h.get("model", {})
    return x.get("args", {}) if isinstance(x, Mapping) and isinstance(x.get("args"), Mapping) else {}


def flatten_config(obj, prefix=""):
    out = {}
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            p = (prefix + "." + str(k)) if prefix else str(k)
            out.update(flatten_config(v, p))
    elif isinstance(obj, (list, tuple)):
        out[prefix] = json.dumps(obj, default=str)
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        out[prefix] = obj
    return out


def config_diff(a, b):
    a, b = flatten_config(a), flatten_config(b)
    rows = []
    for k in sorted(set(a) | set(b)):
        av, bv = a.get(k, "<MISSING>"), b.get(k, "<MISSING>")
        if av != bv:
            rows.append({"key": k, "old": av, "new": bv})
    return rows


def parse_pp(result):
    if not isinstance(result, (tuple, list)):
        raise RuntimeError("unexpected post_process result %r" % type(result))
    pred, score, gt = result[0], result[1], None
    for x in result[2:]:
        if hasattr(x, "shape") and len(x.shape) == 3 and tuple(x.shape[-2:]) == (8, 3):
            gt = x
            break
    if gt is None:
        raise RuntimeError("GT [N,8,3] not found in post_process output")
    return pred, score, gt


def official_gt(dataset, batch, out):
    return parse_pp(dataset.post_process(batch, OrderedDict(ego=out)))[2]


def init_ap(ious):
    return {float(i): {"tp": [], "fp": [], "gt": 0, "score": []} for i in ious}


def add_ap(stat, pred, score, gt, ious):
    ng = 0 if gt is None else int(len(gt))
    for i in ious:
        i = float(i)
        if pred is None or score is None or len(pred) == 0:
            stat[i]["gt"] += ng
        else:
            eval_utils.caluclate_tp_fp(pred, score, gt, stat, i)


def ap_safe(stat, iou):
    iou = float(iou)
    st = copy.deepcopy(stat)
    if st[iou]["gt"] <= 0:
        return float("nan")
    if len(st[iou]["tp"]) == 0:
        return 0.0
    try:
        return float(eval_utils.calculate_ap(st, iou, global_sort_detections=False)[0])
    except Exception:
        return float("nan")


def greedy_counts(pred, score, gt, thr):
    ng = 0 if gt is None else int(gt.shape[0])
    if pred is None or score is None or int(pred.shape[0]) == 0:
        return 0, 0, ng
    dp = list(common_utils.convert_format(common_utils.torch_tensor_to_numpy(pred)))
    gp = list(common_utils.convert_format(common_utils.torch_tensor_to_numpy(gt)))
    s = common_utils.torch_tensor_to_numpy(score)
    avail, tp, fp = list(range(len(gp))), 0, 0
    for di in np.argsort(-s):
        if not avail:
            fp += 1
            continue
        vals = common_utils.compute_iou(dp[int(di)], [gp[j] for j in avail])
        if len(vals) == 0 or float(np.max(vals)) < float(thr):
            fp += 1
        else:
            avail.pop(int(np.argmax(vals)))
            tp += 1
    return tp, fp, ng


def gt_coverage(pred, gt, thr):
    if gt is None or int(gt.shape[0]) == 0:
        return 0, 0
    ng = int(gt.shape[0])
    if pred is None or int(pred.shape[0]) == 0:
        return 0, ng
    pp = list(common_utils.convert_format(common_utils.torch_tensor_to_numpy(pred)))
    gp = list(common_utils.convert_format(common_utils.torch_tensor_to_numpy(gt)))
    hit = 0
    for g in gp:
        v = common_utils.compute_iou(g, pp)
        hit += int(len(v) and float(np.max(v)) >= float(thr))
    return hit, ng


def correct_class_scores(out, num_class):
    psm = out["psm"]
    B, AC, H, W = psm.shape
    C = int(num_class)
    if AC % C:
        raise RuntimeError("psm channels not divisible by num_class")
    A = AC // C
    prob = torch.sigmoid(psm.permute(0, 2, 3, 1).contiguous().view(B, H, W, A, C))
    fg, lab = torch.max(prob[..., 1:], dim=-1)
    return fg.reshape(B, -1), (lab + 1).reshape(B, -1), prob[..., 0].reshape(B, -1)


def max_gt_bev_iou(pred, gt):
    n = 0 if pred is None else int(pred.shape[0])
    if n == 0:
        return np.empty(0, np.float32)
    if gt is None or int(gt.shape[0]) == 0:
        return np.zeros(n, np.float32)
    pp = list(common_utils.convert_format(common_utils.torch_tensor_to_numpy(pred)))
    gp = list(common_utils.convert_format(common_utils.torch_tensor_to_numpy(gt)))
    out = np.zeros(n, np.float32)
    for i, p in enumerate(pp):
        v = common_utils.compute_iou(p, gp)
        out[i] = float(np.max(v)) if len(v) else 0.0
    return out


def paired_bev_iou(pred, gt):
    n = min(0 if pred is None else int(pred.shape[0]), 0 if gt is None else int(gt.shape[0]))
    if n == 0:
        return np.empty(0, np.float32)
    pp = list(common_utils.convert_format(common_utils.torch_tensor_to_numpy(pred[:n])))
    gp = list(common_utils.convert_format(common_utils.torch_tensor_to_numpy(gt[:n])))
    out = np.zeros(n, np.float32)
    for i in range(n):
        v = common_utils.compute_iou(pp[i], [gp[i]])
        out[i] = float(v[0]) if len(v) else 0.0
    return out


def rankdata(x):
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(x.size, np.float64)
    i = 0
    while i < x.size:
        j = i + 1
        while j < x.size and x[order[j]] == x[order[i]]:
            j += 1
        r[order[i:j]] = .5 * ((i + 1) + j)
        i = j
    return r


def spearman(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return None
    x, y = rankdata(x), rankdata(y)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _project_box3d_dtype_safe(corners, transformation_matrix):
    """Project corners after forcing calibration matrix to corners dtype/device.

    AirV2X calibration matrices can arrive from the dataloader as float64 while
    model-decoded boxes/corners are float32. ``torch.matmul`` inside
    ``box_utils.project_box3d`` requires identical dtypes, so diagnostics must
    normalize the matrix dtype explicitly. This is a read-only cast and does not
    change any model prediction or production source code.
    """
    if corners is None or not torch.is_tensor(corners):
        return corners
    if torch.is_tensor(transformation_matrix):
        transformation_matrix = transformation_matrix.to(
            device=corners.device, dtype=corners.dtype
        )
    else:
        transformation_matrix = torch.as_tensor(
            transformation_matrix, device=corners.device, dtype=corners.dtype
        )
    return box_utils.project_box3d(corners, transformation_matrix)


def candidate_pack(batch, out, pp, min_t):
    cav, device = batch["ego"], out["obj"].device
    T, anchors = cav["transformation_matrix"], cav["anchor_box"]
    if torch.is_tensor(T): T = T.to(device)
    if torch.is_tensor(anchors): anchors = anchors.to(device)
    obj = torch.sigmoid(out["obj"].permute(0, 2, 3, 1).contiguous()).reshape(1, -1)[0]
    fg, labels, bg = correct_class_scores(out, int(pp.num_class))
    fg, labels, bg = fg[0], labels[0], bg[0]
    decoded = pp.delta_to_boxes3d(out["rm"], anchors)[0]
    gate = obj > float(min_t)
    decoded, obj, fg, labels, bg = decoded[gate], obj[gate], fg[gate], labels[gate], bg[gate]
    if decoded.numel() == 0:
        return {"corners": None, "obj": obj, "fg": fg, "labels": labels, "bg": bg}
    corners = box_utils.boxes_to_corners_3d(decoded, order=pp.params["order"])
    c = _project_box3d_dtype_safe(corners, T)
    k1 = box_utils.remove_large_pred_bbx(c, pp.dataset)
    try:
        k2 = box_utils.remove_bbx_abnormal_z(c, z_min=pp.lidar_range[2], z_max=pp.lidar_range[5])
    except TypeError:
        k2 = box_utils.remove_bbx_abnormal_z(c)
    k = torch.logical_and(k1, k2)
    return {"corners": c[k], "obj": obj[k], "fg": fg[k], "labels": labels[k], "bg": bg[k]}


def final_range(c, s, pp):
    if c is None or int(c.shape[0]) == 0:
        return c, s
    k = box_utils.get_mask_for_boxes_within_range_torch(c, pp.lidar_range)
    return c[k], s[k]


def nms_variant(pack, pp, gt, t, variant, cls_t):
    if pack["corners"] is None or int(pack["corners"].shape[0]) == 0:
        return None, None, 0, 0, 0
    gate = pack["obj"] > float(t)
    if variant == "hard_cls_0.10":
        gate = gate & (pack["fg"] > float(cls_t))
    c, o, f = pack["corners"][gate], pack["obj"][gate], pack["fg"][gate]
    if int(c.shape[0]) == 0:
        return None, None, 0, 0, 0
    q = torch.as_tensor(max_gt_bev_iou(c, gt), device=c.device, dtype=o.dtype)
    if variant == "normal_obj": nscore, fscore = o, o
    elif variant == "obj_x_cls": nscore, fscore = o * f, o * f
    elif variant == "hard_cls_0.10": nscore, fscore = o, o
    elif variant == "oracle_nms_objscore": nscore, fscore = q, o
    elif variant == "oracle_nms_oraclescore": nscore, fscore = q, q
    else: raise ValueError(variant)
    pre = int(c.shape[0])
    k = box_utils.nms_rotated(c, nscore, pp.params["nms_thresh"])
    c, fscore = c[k], fscore[k]
    post = int(c.shape[0])
    c, fscore = final_range(c, fscore, pp)
    return c, fscore, pre, post, 0 if c is None else int(c.shape[0])


def make_criterion(h, device):
    c = PointPillarLossMultiClass(dict(loss_args(h)))
    c.to(device)
    return c


def positive_decode(criterion, out, label, anchor):
    """Decode positive predicted/GT boxes in one canonical dtype/device.

    AirV2X labels / anchors can be float64 while model regression logits are
    float32. For diagnostics we normalize TARGETS and ANCHORS to the model rm
    dtype/device before decoding. This avoids PyTorch promotion to float64 in
    the GT path and makes later component replacement / projection safe.
    It does not alter model predictions or checkpoint state.
    """
    rm = out["rm"].permute(0, 2, 3, 1).contiguous().view(out["rm"].shape[0], -1, 7)
    target = label["targets"].to(device=rm.device, dtype=rm.dtype).view(label["targets"].shape[0], -1, 7)
    pos = label["pos_equal_one"].to(device=rm.device).view(rm.shape[0], -1) > 0
    if anchor is None:
        return None, None, pos
    if torch.is_tensor(anchor):
        anchor = anchor.to(device=rm.device, dtype=rm.dtype)
    pred = criterion._decode_delta_to_boxes(rm, anchor, pos)
    gt = criterion._decode_delta_to_boxes(target, anchor, pos)
    # Final guard in case a repo-specific decoder internally promotes dtype.
    gt = gt.to(device=pred.device, dtype=pred.dtype)
    return pred, gt, pos


def projected_boxes(boxes, T, pp):
    if boxes is None or boxes.numel() == 0:
        if torch.is_tensor(boxes):
            return torch.empty((0, 8, 3), device=boxes.device, dtype=boxes.dtype)
        device = T.device if torch.is_tensor(T) else "cpu"
        dtype = T.dtype if torch.is_tensor(T) else torch.float32
        return torch.empty((0, 8, 3), device=device, dtype=dtype)
    corners = box_utils.boxes_to_corners_3d(boxes, order=pp.params["order"])
    return _project_box3d_dtype_safe(corners, T)


def component_oracle(pred, gt, T, pp):
    if pred is None or gt is None or int(pred.shape[0]) == 0:
        return {}
    # Defensive dtype/device normalization. ``gt`` may originate from dataset
    # float64 targets while ``pred`` comes from float32 model logits. Index-put
    # assignment requires an exact dtype match.
    gt = gt.to(device=pred.device, dtype=pred.dtype)
    variants = OrderedDict(base=pred)
    def repl(idx):
        z = pred.clone()
        z[:, idx] = gt[:, idx]
        return z
    variants["gt_xy"] = repl([0, 1])
    variants["gt_z"] = repl([2])
    variants["gt_wl"] = repl([4, 5])
    variants["gt_hwl"] = repl([3, 4, 5])
    variants["gt_yaw"] = repl([6])
    variants["gt_xy_yaw"] = repl([0, 1, 6])
    variants["gt_xy_wl"] = repl([0, 1, 4, 5])
    variants["full_gt"] = gt
    gc = projected_boxes(gt, T, pp)
    return {k: paired_bev_iou(projected_boxes(v, T, pp), gc) for k, v in variants.items()}


def exact_reg_loss(c, out, label):
    rm = out["rm"].permute(0, 2, 3, 1).contiguous().view(out["rm"].shape[0], -1, 7)
    tgt = label["targets"].to(device=rm.device, dtype=rm.dtype).view(label["targets"].shape[0], -1, 7)
    pos = label["pos_equal_one"].to(device=rm.device).view(rm.shape[0], -1) > 0
    w = pos.float(); w /= torch.clamp(pos.sum(1, keepdim=True).float(), min=1.)
    ps, ts = c.add_sin_difference(rm, tgt)
    raw = c.reg_loss_func(ps, ts, weights=w).sum() / rm.shape[0]
    return raw, raw * float(c.reg_coe), rm, tgt, pos


def train_anchor(label, out):
    a = label.get("anchor_box", None) if isinstance(label, Mapping) else None
    if a is not None: return a, "target_dict.anchor_box"
    a = out.get("anchor_box", None) if isinstance(out, Mapping) else None
    return (a, "output_dict.anchor_box") if a is not None else (None, "MISSING")


def iou_loss_tensor(c, rm, tgt, pos, anchor):
    if anchor is None or int(pos.sum().item()) == 0:
        return None, None
    p = c._decode_delta_to_boxes(rm, anchor, pos)
    g = c._decode_delta_to_boxes(tgt, anchor, pos)
    if int(p.shape[0]) == 0 or p.shape != g.shape:
        return None, None
    from opencood.utils.iou3d_nms import iou3d_nms_utils
    iou = iou3d_nms_utils.paired_boxes_iou3d_gpu(p.float(), g.float())
    return (1. - iou).sum() / max(int(p.shape[0]), 1), iou


def agrad(loss, target, retain=True):
    if loss is None or not torch.is_tensor(loss) or not loss.requires_grad:
        return None, None
    try:
        return torch.autograd.grad(loss, target, retain_graph=retain, allow_unused=True)[0], None
    except Exception as e:
        return None, type(e).__name__ + ": " + str(e)


def gradient_audit(name, model, dataset, h, args, device):
    if args.grad_frames <= 0: return []
    c = make_criterion(h, device); la = loss_args(h)
    iw = float(la.get("iou_weight", 0.)); rw = float(la.get("reg", getattr(c, "reg_coe", 1.)))
    loader = DataLoader(dataset, batch_size=1, num_workers=args.workers, collate_fn=dataset.collate_batch_test, shuffle=False, pin_memory=False, drop_last=False)
    rows, stop = [], args.start + args.grad_frames
    for fi, batch in tqdm(enumerate(loader), total=min(len(loader), stop), desc=name + " grad"):
        if fi < args.start: continue
        if fi >= stop: break
        batch = train_utils.to_device(batch, device); model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            out = model(batch["ego"]); label = find_label_dict(batch["ego"])
            rawreg, wreg, rm, tgt, pos = exact_reg_loss(c, out, label)
            ea, src = train_anchor(label, out)
            exact_iou, _ = iou_loss_tensor(c, rm, tgt, pos, ea)
            ba = batch["ego"].get("anchor_box", None)
            diag_iou, _ = iou_loss_tensor(c, rm, tgt, pos, ba)
            grm, er = agrad(wreg, out["rm"], True)
            gim, ei = agrad(diag_iou, out["rm"], True)
            gp = model.reg_head.weight if hasattr(model, "reg_head") else None
            grw, erw = agrad(wreg, gp, True) if gp is not None else (None, None)
            giw, eiw = agrad(diag_iou, gp, True) if gp is not None else (None, None)
            ratio = safe_ratio(grad_norm(gim), grad_norm(grm))
            rows.append({
                "run": name, "frame": fi, "configured_reg_weight": rw, "configured_iou_weight": iw,
                "num_positive": int(pos.sum().item()), "raw_reg_loss": float(rawreg.detach().item()), "weighted_reg_loss": float(wreg.detach().item()),
                "exact_training_anchor_source": src, "exact_training_iou_branch_has_anchor": bool(ea is not None),
                "exact_training_iou_loss": None if exact_iou is None else float(exact_iou.detach().item()),
                "exact_weighted_iou_loss": None if exact_iou is None else float(exact_iou.detach().item()) * iw,
                "exact_iou_requires_grad": None if exact_iou is None else bool(exact_iou.requires_grad),
                "exact_iou_grad_fn": None if exact_iou is None or exact_iou.grad_fn is None else type(exact_iou.grad_fn).__name__,
                "diagnostic_batch_anchor_available": bool(ba is not None), "diagnostic_iou_loss": None if diag_iou is None else float(diag_iou.detach().item()),
                "diagnostic_iou_requires_grad": None if diag_iou is None else bool(diag_iou.requires_grad),
                "diagnostic_iou_grad_fn": None if diag_iou is None or diag_iou.grad_fn is None else type(diag_iou.grad_fn).__name__,
                "grad_reg_rm_l2": grad_norm(grm), "grad_iou_rm_l2_raw": grad_norm(gim), "grad_iou_over_reg_rm_raw": ratio,
                "configured_weighted_iou_over_reg_rm": None if ratio is None else ratio * iw,
                "grad_cosine_reg_vs_iou_rm": cosine_flat(grm, gim),
                "grad_reg_head_weight_l2": grad_norm(grw), "grad_iou_reg_head_weight_l2_raw": grad_norm(giw),
                "grad_iou_over_reg_head_weight_raw": safe_ratio(grad_norm(giw), grad_norm(grw)),
                "grad_cosine_reg_vs_iou_reg_head_weight": cosine_flat(grw, giw),
                "reg_rm_grad_error": er, "iou_rm_grad_error": ei, "reg_weight_grad_error": erw, "iou_weight_grad_error": eiw,
            })
        del out; model.zero_grad(set_to_none=True)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return rows


def param_group(name):
    if name.startswith("intra_view"): return "stage1_intra_view"
    if name.startswith("cross_agent"): return "stage2_cross_agent"
    if name.startswith("stage3"): return "stage3"
    if name.startswith("backbone"): return "bev_backbone"
    if name.startswith("shrink_conv"): return "shrink_conv"
    if name.startswith("cls_head"): return "cls_head"
    if name.startswith("reg_head"): return "reg_head"
    if name.startswith("obj_head"): return "obj_head"
    if name.startswith(("frontend", "highres", "heatmap_heads", "depth_heads", "drone_height_embed", "drone_delta_head", "depth_moments", "r2_downsample")): return "p1_frontend"
    return name.split(".")[0]


def parameter_audit(model, run):
    acc = defaultdict(lambda: dict(n=0, finite=0, ss=0., sa=0., mx=0., nz=0, train=0, tensors=0))
    for name, p in model.named_parameters():
        g = param_group(name); x = p.detach().float().reshape(-1); fin = torch.isfinite(x); xf = x[fin]; a = acc[g]
        a["n"] += x.numel(); a["finite"] += int(fin.sum()); a["train"] += x.numel() if p.requires_grad else 0; a["tensors"] += 1
        if xf.numel():
            a["ss"] += float((xf*xf).sum()); a["sa"] += float(xf.abs().sum()); a["mx"] = max(a["mx"], float(xf.abs().max())); a["nz"] += int((xf.abs()<1e-8).sum())
    rows=[]
    for g,a in sorted(acc.items()):
        nf=max(a["finite"],1); rows.append({"run":run,"group":g,"tensor_count":a["tensors"],"numel":a["n"],"trainable_numel":a["train"],"finite_fraction":a["finite"]/max(a["n"],1),"rms":math.sqrt(a["ss"]/nf),"mean_abs":a["sa"]/nf,"max_abs":a["mx"],"near_zero_fraction_1e8":a["nz"]/nf})
    return rows


def gs_n(gs):
    if gs is None: return 0
    try: return int(gs.n_gaussians)
    except Exception: pass
    for k in ("mean","scale","feature","features"):
        x=getattr(gs,k,None)
        if torch.is_tensor(x): return int(x.shape[0])
    return 0


def quat(gs):
    for k in ("quaternion","rotation","quat"):
        x=getattr(gs,k,None)
        if torch.is_tensor(x) and x.ndim==2 and x.shape[-1]==4: return x
    return None


def qstats_t(x):
    if x is None or not torch.is_tensor(x) or x.numel()==0: return {"n":0}
    y=x.detach().float().reshape(-1); y=y[torch.isfinite(y)]
    if y.numel()==0: return {"n":0}
    q=torch.quantile(y, torch.tensor([.1,.5,.9,.99],device=y.device)); return {"n":int(y.numel()),"mean":float(y.mean()),"p10":float(q[0]),"median":float(q[1]),"p90":float(q[2]),"p99":float(q[3]),"max":float(y.max())}


def snap(gs):
    if gs is None or gs_n(gs)==0: return None
    m,s,q=getattr(gs,"mean",None),getattr(gs,"scale",None),quat(gs)
    f=getattr(gs,"feature",None); f=f if f is not None else getattr(gs,"features",None)
    z={"n":gs_n(gs),"mean":None if m is None else m.detach().float().cpu().clone(),"scale":None if s is None else s.detach().float().cpu().clone(),"quat":None if q is None else q.detach().float().cpu().clone(),"feature_rms":tensor_rms(f)}
    try:
        e=torch.linalg.eigvalsh(gs.covariance()[:,:2,:2].detach().float()).clamp_min(0.).sqrt(); z["sigma_minor"]=qstats_t(e[:,0]); z["sigma_major"]=qstats_t(e[:,1])
    except Exception: z["sigma_minor"],z["sigma_major"]={"n":0},{"n":0}
    return z


def rotation_stats(a,b):
    if a is None or b is None or a.shape!=b.shape or a.numel()==0: return None
    a=a.float()/a.float().norm(dim=-1,keepdim=True).clamp_min(1e-12); b=b.float()/b.float().norm(dim=-1,keepdim=True).clamp_min(1e-12)
    return qstats_t(2*torch.acos((a*b).sum(-1).abs().clamp(0,1)))


def transition_row(run,frame,agent,a_name,b_name,a,b,bounds,tol):
    r={"run":run,"frame":frame,"agent":agent,"transition":a_name+"->"+b_name,"n0":0 if a is None else a["n"],"n1":0 if b is None else b["n"],"same_count":False}
    if a is None or b is None or a["n"]!=b["n"]: return r
    r["same_count"]=True
    if a["mean"] is not None and b["mean"] is not None and a["mean"].shape==b["mean"].shape:
        st=qstats_t(torch.linalg.vector_norm(b["mean"]-a["mean"],dim=-1)); r.update({"mean_delta_"+k:v for k,v in st.items()})
    if a["scale"] is not None and b["scale"] is not None and a["scale"].shape==b["scale"].shape:
        ld=torch.log(b["scale"].clamp_min(1e-12)/a["scale"].clamp_min(1e-12)); lo,hi=bounds
        for j,ax in enumerate(("x","y","z")):
            if j>=ld.shape[1]: break
            st=qstats_t(ld[:,j]); r.update({"logscale_%s_%s"%(ax,k):v for k,v in st.items()}); r["logscale_%s_near_lower"%ax]=float(((ld[:,j]-lo).abs()<=tol).float().mean()); r["logscale_%s_near_upper"%ax]=float(((ld[:,j]-hi).abs()<=tol).float().mean())
    st=rotation_stats(a["quat"],b["quat"])
    if st: r.update({"rotation_rad_"+k:v for k,v in st.items()})
    return r


class GeometryTracer:
    def __init__(self,model,run,h,max_frame,tol):
        self.model,self.run,self.max_frame,self.tol=model,run,max_frame,tol; self.frame=-1; self.enabled=False; self.cur={}; self.stage=[]; self.updates=[]; self.features=[]; self.handles=[]
        try:
            dr=model_args(h)["gaussian_refinement"]["scale"]["delta_range"]; self.bounds=(float(dr[0]),float(dr[1]))
        except Exception: self.bounds=(-.2,.05)
        self.orig=model.initializer.forward
        def wrap(*a,**k):
            out=self.orig(*a,**k)
            if self.enabled: self.cur["Init"]={str(n):snap(g) for n,g in out.items()}
            return out
        model.initializer.forward=wrap
        self.handles.append(model.cross_agent.register_forward_pre_hook(self.h_stage1)); self.handles.append(model.stage3.register_forward_pre_hook(self.h_stage2)); self.handles.append(model.intra_view.register_forward_hook(self.h_feat1)); self.handles.append(model.cross_agent.register_forward_hook(self.h_feat2))
    def begin(self,f): self.frame=int(f); self.cur={}; self.enabled=(f<self.max_frame)
    def close(self):
        self.model.initializer.forward=self.orig
        for h in self.handles: h.remove()
    def record(self,name,mapping):
        if not isinstance(mapping,Mapping): return
        sm={str(n):snap(g) for n,g in mapping.items()}; self.cur[name]=sm
        for agent,s in sm.items():
            if s is None: continue
            r={"run":self.run,"frame":self.frame,"stage":name,"agent":agent,"n":s["n"],"feature_rms":s["feature_rms"]}
            for pn in ("sigma_minor","sigma_major"):
                for k,v in s[pn].items(): r[pn+"_"+k]=v
            if s["scale"] is not None:
                for j,ax in enumerate(("x","y","z")):
                    if j<s["scale"].shape[1]:
                        for k,v in qstats_t(s["scale"][:,j]).items(): r["scale_%s_%s"%(ax,k)]=v
            self.stage.append(r)
    def compare(self,a,b):
        aa,bb=self.cur.get(a,{}),self.cur.get(b,{})
        for ag in sorted(set(aa)&set(bb)): self.updates.append(transition_row(self.run,self.frame,ag,a,b,aa[ag],bb[ag],self.bounds,self.tol))
    def h_stage1(self,m,inp):
        if self.enabled and inp: self.record("Stage1",inp[0]); self.compare("Init","Stage1")
    def h_stage2(self,m,inp):
        if self.enabled and inp: self.record("Stage2",inp[0]); self.compare("Stage1","Stage2")
    def h_feat1(self,m,inp,out):
        if not self.enabled or not inp: return
        f0=getattr(inp[0],"feature",None); f0=f0 if f0 is not None else getattr(inp[0],"features",None); f1=getattr(out,"feature",None); f1=f1 if f1 is not None else getattr(out,"features",None)
        if torch.is_tensor(f0) and torch.is_tensor(f1) and f0.shape==f1.shape and f0.numel(): self.features.append({"run":self.run,"frame":self.frame,"module":"stage1_intra_view","agent":"per_call","input_rms":tensor_rms(f0),"output_rms":tensor_rms(f1),"delta_over_input_rms":safe_ratio(tensor_rms(f1-f0),tensor_rms(f0))})
    def h_feat2(self,m,inp,out):
        if not self.enabled or not inp or not isinstance(inp[0],Mapping) or not isinstance(out,Mapping): return
        for ag in sorted(set(inp[0])&set(out)):
            g0,g1=inp[0][ag],out[ag]; f0=getattr(g0,"feature",None); f0=f0 if f0 is not None else getattr(g0,"features",None); f1=getattr(g1,"feature",None); f1=f1 if f1 is not None else getattr(g1,"features",None)
            if torch.is_tensor(f0) and torch.is_tensor(f1) and f0.shape==f1.shape and f0.numel(): self.features.append({"run":self.run,"frame":self.frame,"module":"stage2_cross_agent","agent":str(ag),"input_rms":tensor_rms(f0),"output_rms":tensor_rms(f1),"delta_over_input_rms":safe_ratio(tensor_rms(f1-f0),tensor_rms(f0))})


def aggregate(rows,keys):
    g=defaultdict(list)
    for r in rows: g[tuple(r[k] for k in keys)].append(r)
    out=[]
    for kk,rr in sorted(g.items()):
        z={k:v for k,v in zip(keys,kk)}; z["count"]=len(rr); fields=set().union(*(r.keys() for r in rr))
        for f in fields:
            if f in keys or f in ("frame","same_count"): continue
            vals=[float(r[f]) for r in rr if isinstance(r.get(f),(int,float)) and math.isfinite(float(r[f]))]
            if vals: z[f+"_mean"]=float(np.mean(vals))
        if "same_count" in fields: z["same_count_fraction"]=float(np.mean([1. if r.get("same_count") else 0. for r in rr]))
        out.append(z)
    return out


def load_run(name,run_dir,epoch,args,device):
    ck=os.path.join(run_dir,"net_epoch%d.pth"%epoch)
    if not os.path.exists(ck): raise FileNotFoundError(ck)
    h=load_hypes(args,run_dir); ds=build_dataset(h,visualize=False,train=False); model=train_utils.create_model(h); model.to(device); _,model=train_utils.load_model(run_dir,model,epoch,start_from_best=False); model.eval(); model.zero_grad(set_to_none=True); return h,ds,model


def evaluate(name,run_dir,epoch,h,ds,model,args,device):
    ts,ious=flist(args.quality_obj_thresholds),flist(args.eval_ious); mint=min(ts); c=make_criterion(h,device); pp=ds.post_processor; prod=float(pp.params["target_args"]["obj_threshold"])
    loader=DataLoader(ds,batch_size=1,num_workers=args.workers,collate_fn=ds.collate_batch_test,shuffle=False,pin_memory=False,drop_last=False)
    variants=["normal_obj","obj_x_cls","hard_cls_0.10","oracle_nms_objscore","oracle_nms_oraclescore"]
    ap={(t,v):init_ap(ious) for t in ts for v in variants}; frames=defaultdict(list); paired3=[]; pairedbev=[]; gtbest=[]; cover={i:[0,0] for i in ious}; qbuf={t:{k:[] for k in ("iou","obj","cls","bg")} for t in ts}; comp=defaultdict(list); gtcounts={}
    tracer=GeometryTracer(model,name,h,args.start+args.geometry_frames,args.delta_bound_tol); end=args.start+args.num_frames
    try:
        for fi,batch in tqdm(enumerate(loader),total=min(len(loader),end),desc=name+" eval"):
            if fi<args.start: continue
            if fi>=end: break
            batch=train_utils.to_device(batch,device); tracer.begin(fi)
            with torch.no_grad(): out=model(batch["ego"])
            oldt=float(pp.params["target_args"]["obj_threshold"]); pp.params["target_args"]["obj_threshold"]=prod; gt=official_gt(ds,batch,out); pp.params["target_args"]["obj_threshold"]=oldt; gtcounts[fi]=int(gt.shape[0]); label=find_label_dict(batch["ego"]); anchor=batch["ego"].get("anchor_box",None); T=batch["ego"]["transformation_matrix"]; T=T.to(device) if torch.is_tensor(T) else T
            predp,gtp,_=positive_decode(c,out,label,anchor)
            if predp is not None and int(predp.shape[0])>0:
                try:
                    from opencood.utils.iou3d_nms import iou3d_nms_utils
                    paired3.append(iou3d_nms_utils.paired_boxes_iou3d_gpu(predp.float(),gtp.float()).detach().cpu().numpy())
                except Exception: pass
                pc,gc=projected_boxes(predp,T,pp),projected_boxes(gtp,T,pp); pairedbev.append(paired_bev_iou(pc,gc))
                if int(gt.shape[0]) and int(pc.shape[0]):
                    ppol=list(common_utils.convert_format(common_utils.torch_tensor_to_numpy(pc))); gpol=list(common_utils.convert_format(common_utils.torch_tensor_to_numpy(gt))); vals=[]
                    for gp in gpol:
                        vv=common_utils.compute_iou(gp,ppol); vals.append(float(np.max(vv)) if len(vv) else 0.)
                    vals=np.asarray(vals,np.float32); gtbest.append(vals)
                    for i in ious: cover[i][0]+=int((vals>=i).sum()); cover[i][1]+=int(vals.size)
                for k,v in component_oracle(predp,gtp,T,pp).items():
                    if v.size: comp[k].append(v)
            pack=candidate_pack(batch,out,pp,mint)
            if pack["corners"] is not None and int(pack["corners"].shape[0]):
                qq=max_gt_bev_iou(pack["corners"],gt); oo=pack["obj"].detach().cpu().numpy(); cc=pack["fg"].detach().cpu().numpy(); bb=pack["bg"].detach().cpu().numpy()
                for t in ts:
                    m=oo>t
                    for k,v in (("iou",qq),("obj",oo),("cls",cc),("bg",bb)): qbuf[t][k].append(v[m])
            for t in ts:
                pre=None if pack["corners"] is None else pack["corners"][pack["obj"]>t]
                for v in variants:
                    pred,score,pn,postn,fn=nms_variant(pack,pp,gt,t,v,args.hard_class_threshold); add_ap(ap[(t,v)],pred,score,gt,ious); r={"run":name,"frame":fi,"obj_threshold":t,"variant":v,"candidate_pre_nms":pn,"post_nms":postn,"final":fn}
                    for i in ious:
                        k="%.2f"%i; tp,fp,ng=greedy_counts(pred,score,gt,i); hp,gp=gt_coverage(pre,gt,i); hn,gn=gt_coverage(pred,gt,i); r.update({"tp_"+k:tp,"fp_"+k:fp,"gt_"+k:ng,"pre_hit_"+k:hp,"pre_gt_"+k:gp,"post_hit_"+k:hn,"post_gt_"+k:gn})
                    frames[(t,v)].append(r)
    finally:
        pp.params["target_args"]["obj_threshold"]=prod; tracer.close()
    cat=lambda xs: np.concatenate([x for x in xs if isinstance(x,np.ndarray) and x.size]) if any(isinstance(x,np.ndarray) and x.size for x in xs) else np.empty(0,np.float32)
    loc={"run":name,"paired_positive_3d_iou":stats_np(cat(paired3)),"paired_positive_bev_iou":stats_np(cat(pairedbev)),"gt_best_positive_decoded_bev_iou":stats_np(cat(gtbest)),"positive_decoded_gt_coverage":{"%.2f"%i:{"hit":cover[i][0],"gt":cover[i][1],"coverage":cover[i][0]/max(cover[i][1],1)} for i in ious}}
    comps=[]; base=cat(comp.get("base",[]))
    for k in ("base","gt_xy","gt_z","gt_wl","gt_hwl","gt_yaw","gt_xy_yaw","gt_xy_wl","full_gt"):
        x=cat(comp.get(k,[])); r={"run":name,"variant":k}; r.update({"bev_iou_"+a:b for a,b in stats_np(x).items()})
        if k!="base" and x.size and x.size==base.size: r.update({"delta_vs_base_"+a:b for a,b in stats_np(x-base).items()})
        comps.append(r)
    qrows=[]; brows=[]; bins=[(0,.3,"[0,.3)"),(.3,.5,"[.3,.5)"),(.5,.7,"[.5,.7)"),(.7,1.000001,"[.7,1]")]
    for t in ts:
        q,o,cl,bg=[cat(qbuf[t][k]) for k in ("iou","obj","cls","bg")]; qrows.append({"run":name,"obj_threshold":t,"n_candidates":int(q.size),"spearman_obj_vs_iou":spearman(o,q),"spearman_cls_vs_iou":spearman(cl,q),"spearman_bg_vs_iou":spearman(bg,q),"quality_iou_mean":None if not q.size else float(q.mean()),"quality_iou_median":None if not q.size else float(np.median(q))})
        for lo,hi,bn in bins:
            m=(q>=lo)&(q<hi); brows.append({"run":name,"obj_threshold":t,"iou_bin":bn,"count":int(m.sum()),"obj_mean":None if not m.any() else float(o[m].mean()),"obj_median":None if not m.any() else float(np.median(o[m])),"cls_mean":None if not m.any() else float(cl[m].mean()),"cls_median":None if not m.any() else float(np.median(cl[m])),"bg_mean":None if not m.any() else float(bg[m].mean()),"bg_median":None if not m.any() else float(np.median(bg[m]))})
    finals=[]
    for t in ts:
        for v in variants:
            rr=frames[(t,v)]; n=max(len(rr),1); r={"run":name,"obj_threshold":t,"variant":v,"frames":len(rr),"candidate_pre_nms_per_frame":safe_mean([x["candidate_pre_nms"] for x in rr]),"post_nms_per_frame":safe_mean([x["post_nms"] for x in rr]),"final_per_frame":safe_mean([x["final"] for x in rr])}
            for i in ious:
                k="%.2f"%i; tp=sum(x["tp_"+k] for x in rr); fp=sum(x["fp_"+k] for x in rr); ng=sum(x["gt_"+k] for x in rr); ph=sum(x["pre_hit_"+k] for x in rr); pg=sum(x["pre_gt_"+k] for x in rr); nh=sum(x["post_hit_"+k] for x in rr); ng2=sum(x["post_gt_"+k] for x in rr); r.update({"tp_per_frame_iou"+k:tp/n,"fp_per_frame_iou"+k:fp/n,"precision_iou"+k:tp/max(tp+fp,1),"recall_iou"+k:tp/max(ng,1),"ap_iou"+k:ap_safe(ap[(t,v)],i),"pre_nms_coverage_iou"+k:ph/max(pg,1),"post_nms_coverage_iou"+k:nh/max(ng2,1)})
            finals.append(r)
    return {"run":name,"run_dir":run_dir,"epoch":epoch,"production_obj_threshold":prod,"configured_loss":dict(loss_args(h)),"gt_counts":gtcounts,"localization":loc,"final_metrics":finals,"score_quality":qrows,"score_quality_bins":brows,"component_oracle":comps,"geometry_stage_summary":aggregate(tracer.stage,["run","stage","agent"]),"geometry_update_summary":aggregate(tracer.updates,["run","transition","agent"]),"feature_update_summary":aggregate(tracer.features,["run","module","agent"]),"parameter_groups":parameter_audit(model,name)}


def find_final(rows,run,t,v):
    return next((x for x in rows if x["run"]==run and abs(x["obj_threshold"]-t)<1e-12 and x["variant"]==v),None)


def flags(old,new,grad,ts):
    r={}; ng=[x for x in grad if x["run"]=="new_iou"]
    r["new_configured_iou_weight"]=None if not ng else ng[0]["configured_iou_weight"]
    r["new_exact_training_anchor_available_fraction"]=None if not ng else float(np.mean([1. if x["exact_training_iou_branch_has_anchor"] else 0. for x in ng]))
    vals=[x["diagnostic_iou_requires_grad"] for x in ng if x["diagnostic_iou_requires_grad"] is not None]; r["new_diagnostic_iou_requires_grad_fraction"]=None if not vals else float(np.mean([1. if x else 0. for x in vals]))
    r["new_raw_iou_over_reg_rm_grad_mean"]=safe_mean([x["grad_iou_over_reg_rm_raw"] for x in ng]); r["new_configured_iou_over_reg_rm_grad_mean"]=safe_mean([x["configured_weighted_iou_over_reg_rm"] for x in ng]); r["new_reg_iou_grad_cosine_mean"]=safe_mean([x["grad_cosine_reg_vs_iou_rm"] for x in ng])
    for metric in ("paired_positive_bev_iou","gt_best_positive_decoded_bev_iou"):
        a=old["localization"][metric].get("median"); b=new["localization"][metric].get("median"); r["delta_%s_median"%metric]=None if a is None or b is None else b-a
    for i in (.3,.5,.7):
        k="%.2f"%i; a=old["localization"]["positive_decoded_gt_coverage"].get(k,{}).get("coverage"); b=new["localization"]["positive_decoded_gt_coverage"].get(k,{}).get("coverage"); r["delta_positive_coverage_"+k]=None if a is None or b is None else b-a
    fr=old["final_metrics"]+new["final_metrics"]; gaps=[]
    for run in ("old_no_iou","new_iou"):
        for t in ts:
            a=find_final(fr,run,t,"normal_obj"); b=find_final(fr,run,t,"oracle_nms_objscore"); c=find_final(fr,run,t,"oracle_nms_oraclescore")
            if a: gaps.append({"run":run,"obj_threshold":t,"oracle_selection_gap_ap30":None if b is None else b.get("ap_iou0.30",0)-a.get("ap_iou0.30",0),"oracle_selection_gap_ap50":None if b is None else b.get("ap_iou0.50",0)-a.get("ap_iou0.50",0),"oracle_selection_gap_ap70":None if b is None else b.get("ap_iou0.70",0)-a.get("ap_iou0.70",0),"oracle_total_gap_ap50":None if c is None else c.get("ap_iou0.50",0)-a.get("ap_iou0.50",0)})
    r["oracle_nms_gaps"]=gaps; w=[]
    if r["new_exact_training_anchor_available_fraction"]==0.: w.append("CRITICAL: exact IoU-loss branch cannot see anchor_box on audited batches; if training used the same contract, IoU loss was a no-op.")
    if r["new_diagnostic_iou_requires_grad_fraction"]==0.: w.append("CRITICAL: numeric IoU loss does not require grad on audited batches; paired_boxes_iou3d_gpu may be non-differentiable in this path.")
    if r["new_configured_iou_over_reg_rm_grad_mean"] is not None and r["new_configured_iou_over_reg_rm_grad_mean"]<.05: w.append("Configured IoU gradient pressure is <5% of SmoothL1 at rm output.")
    if r.get("delta_positive_coverage_0.50") is not None and r.get("delta_positive_coverage_0.70") is not None and abs(r["delta_positive_coverage_0.50"])<.01 and abs(r["delta_positive_coverage_0.70"])<.01: w.append("Positive decoded-box coverage changed by <1pp at IoU .5 and .7; localization upper bound barely moved.")
    r["warnings"]=w; return r


def plot_loc(old,new,path):
    labels=["paired BEV median","GT-best median","coverage .5","coverage .7"]
    def v(x): return [x["localization"]["paired_positive_bev_iou"].get("median",np.nan),x["localization"]["gt_best_positive_decoded_bev_iou"].get("median",np.nan),x["localization"]["positive_decoded_gt_coverage"]["0.50"]["coverage"],x["localization"]["positive_decoded_gt_coverage"]["0.70"]["coverage"]]
    a,b=v(old),v(new); x=np.arange(4); w=.36; fig,ax=plt.subplots(figsize=(8.5,5.2)); ax.bar(x-w/2,a,w,label="old no-IoU"); ax.bar(x+w/2,b,w,label="new +IoU"); ax.set_xticks(x); ax.set_xticklabels(labels,rotation=20,ha="right"); ax.set_ylim(0,1.02); ax.set_ylabel("IoU / coverage"); ax.grid(True,axis="y",alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=170); plt.close(fig)


def main():
    args=parse_args(); ts=flist(args.quality_obj_thresholds); out=args.out or os.path.join(args.new_dir,"iou_compare_epoch%d_vs_epoch%d"%(args.new_epoch,args.old_epoch)); os.makedirs(out,exist_ok=True); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("device",device); print("out",out)
    oh0,nh0=load_hypes(args,args.old_dir),load_hypes(args,args.new_dir); cd=config_diff(oh0,nh0); write_csv(os.path.join(out,"config_differences.csv"),cd)
    oh,ods,om=load_run("old_no_iou",args.old_dir,args.old_epoch,args,device); ores=evaluate("old_no_iou",args.old_dir,args.old_epoch,oh,ods,om,args,device); og=gradient_audit("old_no_iou",om,ods,oh,args,device); del om; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    nh,nds,nm=load_run("new_iou",args.new_dir,args.new_epoch,args,device); nres=evaluate("new_iou",args.new_dir,args.new_epoch,nh,nds,nm,args,device); ng=gradient_audit("new_iou",nm,nds,nh,args,device); del nm; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    grad=og+ng; finals=ores["final_metrics"]+nres["final_metrics"]; q=ores["score_quality"]+nres["score_quality"]; qb=ores["score_quality_bins"]+nres["score_quality_bins"]; co=ores["component_oracle"]+nres["component_oracle"]; gs=ores["geometry_stage_summary"]+nres["geometry_stage_summary"]; gu=ores["geometry_update_summary"]+nres["geometry_update_summary"]; fu=ores["feature_update_summary"]+nres["feature_update_summary"]; pg=ores["parameter_groups"]+nres["parameter_groups"]
    loc=[]
    for res in (ores,nres):
        r={"run":res["run"]}
        for block in ("paired_positive_3d_iou","paired_positive_bev_iou","gt_best_positive_decoded_bev_iou"):
            for k,v in res["localization"][block].items(): r[block+"_"+k]=v
        for k,v in res["localization"]["positive_decoded_gt_coverage"].items(): r["positive_coverage_"+k]=v["coverage"]
        loc.append(r)
    for fn,rows in (("final_variant_metrics.csv",finals),("loss_gradient_audit.csv",grad),("localization_summary.csv",loc),("score_quality_summary.csv",q),("score_quality_bins.csv",qb),("component_oracle.csv",co),("geometry_stage_stats.csv",gs),("geometry_update_stats.csv",gu),("feature_update_stats.csv",fu),("parameter_groups.csv",pg)): write_csv(os.path.join(out,fn),rows)
    f=flags(ores,nres,grad,ts); common=sorted(set(ores["gt_counts"])&set(nres["gt_counts"])); f["same_frame_gt_count_match"]=all(ores["gt_counts"][k]==nres["gt_counts"][k] for k in common); f["config_difference_count"]=len(cd)
    with open(os.path.join(out,"comparison_flags.json"),"w") as z: json.dump(f,z,indent=2,ensure_ascii=False,allow_nan=True)
    summary={"old":ores,"new":nres,"config_differences":cd,"loss_gradient_audit":grad,"comparison_flags":f,"definitions":{"oracle_nms_objscore":"GT max-IoU used only for NMS selection; final score remains objectness.","oracle_nms_oraclescore":"GT max-IoU used for NMS and final ranking; diagnostic upper bound only.","component_oracle":"Paired positive-anchor BEV IoU after replacing selected predicted box components by paired GT components."}}
    with open(os.path.join(out,"summary.json"),"w") as z: json.dump(summary,z,indent=2,ensure_ascii=False,allow_nan=True)
    plot_loc(ores,nres,os.path.join(out,"localization_compare.png"))
    print("\n=== AUTO FLAGS ===")
    for w in f["warnings"]: print("WARNING:",w)
    print("\n=== LOCALIZATION ===")
    for res in (ores,nres): print(res["run"],res["localization"])
    print("\n=== SCORE/NMS ===")
    for run in ("old_no_iou","new_iou"):
        for t in ts:
            qr=next((x for x in q if x["run"]==run and abs(x["obj_threshold"]-t)<1e-12),None); a=find_final(finals,run,t,"normal_obj"); b=find_final(finals,run,t,"oracle_nms_objscore")
            if qr: print(run,"T",t,"rho(obj,IoU)",qr["spearman_obj_vs_iou"],"rho(cls,IoU)",qr["spearman_cls_vs_iou"])
            if a and b: print("  normal AP30/50/70",a.get("ap_iou0.30"),a.get("ap_iou0.50"),a.get("ap_iou0.70"),"oracleNMS",b.get("ap_iou0.30"),b.get("ap_iou0.50"),b.get("ap_iou0.70"))
    print("\nSaved:",out)


if __name__ == "__main__":
    main()