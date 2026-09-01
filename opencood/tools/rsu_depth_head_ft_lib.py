# -*- coding: utf-8 -*-
"""RSU DepthHead-only fine-tune helpers. Does not change production forward."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.path import Path as MplPath
from torch import nn

from opencood.loss.gaussian_p1_depth_loss import GaussianP1DepthLoss
from opencood.models.gaussian_modules_0822.heatmap.metrics import (
    PRIMARY_OBJECTNESS_THRESHOLD,
)
from opencood.models.gaussian_modules_0822.heatmap.target import build_semantic_target
from opencood.models.gaussian_modules_0822.image_frontend import present_camera_agents
from opencood.models.gaussian_modules_0822.lss.target import (
    build_depth_class_target,
    depth_valid_mask,
    extract_camera_z_gt,
)
from opencood.tools.audit_gaussian_scale_coverage import (
    _as_numpy,
    _record_len,
    _slice_cam,
    flatten_imgs,
    gt_boxes_from_ego,
    n_views_of,
    pairwise_for_agent,
    projected_polygons,
)
from opencood.tools.eval_heatmap_ft_abc import obb_surface_distance
from opencood.tools.gaussian_scale_audit.coverage import (
    coverage_fraction,
    sample_box_points,
)
from opencood.tools.gaussian_scale_audit.geometry import (
    invert_spd,
    r90_pixel_centers,
    view_gaussians,
)
from opencood.tools.gaussian_scale_audit.orientation import local_orientation
from opencood.tools.train_gaussian_p1 import _unwrap_model
from opencood.tools.vis_test_heatmap_recall import sample_plan, scene_from_path

RSU_DEPTH_HEAD_KEY = "depth_heads.rsu"
AGENT_ORDER = ("vehicle", "rsu", "drone")
DEFAULT_DISTANCE_EDGES_M = (2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0)
NEAR_MID_FAR = ((2.0, 10.0), (10.0, 25.0), (25.0, 50.0))


def is_rsu_depth_head_param(name: str) -> bool:
    """True iff ``name`` belongs to the live RSU ``DepthHead``."""
    stripped = name[7:] if name.startswith("module.") else name
    return stripped.startswith(f"{RSU_DEPTH_HEAD_KEY}.")


def apply_rsu_depth_head_ft_freeze(model: nn.Module) -> List[str]:
    """Freeze every parameter except ``depth_heads.rsu``. Return trainable names.

    Raises:
        AssertionError: If any non-RSU-DepthHead parameter is still trainable,
            or if the RSU DepthHead has no trainable tensors.
    """
    core = _unwrap_model(model)
    core.rsu_depth_head_only = True
    core.heatmap_only = False
    core.frontend.freeze_backbone = True
    for param in core.parameters():
        param.requires_grad = False
    if "rsu" not in core.depth_heads:
        raise AssertionError("model has no depth_heads.rsu")
    for param in core.depth_heads["rsu"].parameters():
        param.requires_grad = True
    return assert_only_rsu_depth_head_trainable(model)


def assert_only_rsu_depth_head_trainable(model: nn.Module) -> List[str]:
    """Stop if anything other than RSU DepthHead is trainable."""
    trainable: List[str] = []
    illegal: List[str] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_rsu_depth_head_param(name):
            trainable.append(name)
        else:
            illegal.append(name)
    if illegal:
        raise AssertionError(
            "RSU DepthHead-only FT: illegal trainable params: " + ", ".join(illegal)
        )
    if not trainable:
        raise AssertionError("RSU DepthHead-only FT: no trainable DepthHead params")
    return trainable


def apply_rsu_depth_head_train_eval(model: nn.Module, training: bool) -> None:
    """Train only RSU DepthHead; keep frozen BN / modules in eval."""
    core = _unwrap_model(model)
    core.rsu_depth_head_only = True
    core.heatmap_only = False
    core.frontend.freeze_backbone = True
    core.train(training)
    assert_only_rsu_depth_head_trainable(model)


def setup_rsu_depth_head_optimizer(
    model: nn.Module,
    lr: float,
    eps: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Adam over RSU DepthHead parameters only."""
    names = assert_only_rsu_depth_head_trainable(model)
    params = []
    named = dict(model.named_parameters())
    for name in names:
        params.append(named[name])
    frozen_ids = {id(p) for n, p in model.named_parameters() if not p.requires_grad}
    opt_ids = {id(p) for p in params}
    if frozen_ids & opt_ids:
        raise AssertionError("frozen params entered the RSU DepthHead optimizer")
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    if trainable_ids != opt_ids:
        raise AssertionError("optimizer params do not match trainable RSU DepthHead")
    optimizer = torch.optim.Adam(params, lr=float(lr), eps=float(eps), weight_decay=float(weight_decay))
    print(
        f"RSU DepthHead-only Adam: n_tensors={len(params)} lr={lr} "
        f"eps={eps} weight_decay={weight_decay}"
    )
    return optimizer


def parameter_checksums(model: nn.Module) -> Dict[str, str]:
    """SHA1 of each parameter tensor (cpu float bytes)."""
    out: Dict[str, str] = {}
    for name, param in model.named_parameters():
        payload = param.detach().float().cpu().numpy().tobytes()
        out[name] = hashlib.sha1(payload).hexdigest()
    return out


def changed_parameter_names(
    before: Mapping[str, str], after: Mapping[str, str]
) -> List[str]:
    """Names whose checksums differ."""
    keys = sorted(set(before) | set(after))
    changed: List[str] = []
    for key in keys:
        if before.get(key) != after.get(key):
            changed.append(key)
    return changed


def rsu_depth_train_forward(
    core: nn.Module,
    ego: Dict[str, Any],
    depth_criterion: GaussianP1DepthLoss,
) -> Optional[torch.Tensor]:
    """RSU depth Focal only. Frozen F90 is computed under ``no_grad``.

    Production ``forward`` is not modified. Returns None if RSU is absent or
    the SAM3∩in-range support is empty.
    """
    if "rsu" not in present_camera_agents(ego):
        return None
    cam = ego["rsu"]["batch_merged_cam_inputs"]
    imgs = cam["imgs"]
    with torch.no_grad():
        r2, f45 = core.frontend.extract_backbone_features("rsu", imgs)
        f90 = core.highres["rsu"](r2, f45)
    depth_logits = core.depth_heads["rsu"](f90)
    depth_z_mean, depth_z_var = core.depth_moments["rsu"](depth_logits)
    predictions = {
        "rsu": {
            "depth_logits": depth_logits,
            "depth_z_mean": depth_z_mean,
            "depth_z_var": depth_z_var,
        }
    }
    semantic = build_semantic_target(cam, tau=1).to(device=depth_logits.device)
    depth_gt = build_depth_class_target(core.frontend.encoders["rsu"], imgs)
    z_gt = extract_camera_z_gt(imgs)
    enc = core.frontend.encoders["rsu"]
    valid = depth_valid_mask(z_gt, enc.d_min, enc.d_max)
    loss = depth_criterion(
        predictions,
        {"rsu": depth_gt},
        {"rsu": semantic},
        {"rsu": valid},
    )
    n_valid = int(depth_criterion.loss_dict.get("rsu_valid_cells", 0.0))
    if n_valid <= 0:
        return None
    return loss


def capture_frozen_outputs(
    core: nn.Module, ego: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    """Production forward + F90 hooks for the invariance test."""
    captured: Dict[str, torch.Tensor] = {}
    handles = []

    def _save(name: str):
        def _hook(_mod: nn.Module, _inp: Any, out: torch.Tensor) -> None:
            captured[name] = out.detach().float().cpu().clone()

        return _hook

    for agent in AGENT_ORDER:
        if agent not in present_camera_agents(ego):
            continue
        handles.append(core.highres[agent].register_forward_hook(_save(f"{agent}.f90")))
    with torch.no_grad():
        pred = core(ego)
    for handle in handles:
        handle.remove()
    for agent, out in pred.items():
        captured[f"{agent}.heatmap_logits"] = out["heatmap_logits"].detach().float().cpu().clone()
        if "depth_logits" in out:
            captured[f"{agent}.depth_logits"] = out["depth_logits"].detach().float().cpu().clone()
        if "depth_z_mean" in out:
            captured[f"{agent}.depth_z_mean"] = out["depth_z_mean"].detach().float().cpu().clone()
        if "delta_pred" in out:
            captured[f"{agent}.delta_pred"] = out["delta_pred"].detach().float().cpu().clone()
    return captured


def compare_frozen_outputs(
    source: Mapping[str, torch.Tensor],
    current: Mapping[str, torch.Tensor],
    allowed_change: Sequence[str] = ("rsu.depth_logits", "rsu.depth_z_mean"),
) -> List[Dict[str, Any]]:
    """Max-abs diffs. Keys in ``allowed_change`` may move; others must stay ~0."""
    rows: List[Dict[str, Any]] = []
    keys = sorted(set(source) | set(current))
    for key in keys:
        if key not in source or key not in current:
            rows.append(
                {
                    "tensor": key,
                    "max_abs": None,
                    "allowed": key in allowed_change,
                    "status": "missing",
                }
            )
            continue
        delta = float((source[key] - current[key]).abs().max().item())
        rows.append(
            {
                "tensor": key,
                "max_abs": delta,
                "allowed": key in allowed_change,
                "status": "ok",
            }
        )
    return rows


def _error_stats(pred: np.ndarray, gt: np.ndarray) -> Dict[str, Optional[float]]:
    """Signed / absolute error summaries. Empty → None."""
    if pred.size == 0:
        return {
            "n": 0,
            "mae": None,
            "medae": None,
            "rmse": None,
            "mean_ez": None,
            "median_ez": None,
            "frac_neg": None,
            "frac_pos": None,
        }
    ez = pred.astype(np.float64) - gt.astype(np.float64)
    abs_e = np.abs(ez)
    return {
        "n": int(ez.size),
        "mae": float(np.mean(abs_e)),
        "medae": float(np.median(abs_e)),
        "rmse": float(np.sqrt(np.mean(ez * ez))),
        "mean_ez": float(np.mean(ez)),
        "median_ez": float(np.median(ez)),
        "frac_neg": float(np.mean(ez < 0.0)),
        "frac_pos": float(np.mean(ez > 0.0)),
    }


def _sigma_stats(
    pred: np.ndarray, gt: np.ndarray, var: np.ndarray
) -> Dict[str, Optional[float]]:
    """Categorical sigma_z summaries. Not calibrated uncertainty."""
    if pred.size == 0:
        return {"median_sigma_z": None, "median_abs_ez_over_sigma": None}
    sigma = np.sqrt(np.clip(var.astype(np.float64), 0.0, None))
    abs_e = np.abs(pred.astype(np.float64) - gt.astype(np.float64))
    ratio = abs_e / np.clip(sigma, 1.0e-6, None)
    return {
        "median_sigma_z": float(np.median(sigma)),
        "median_abs_ez_over_sigma": float(np.median(ratio)),
    }


def collect_rsu_depth_pixels(
    core: nn.Module,
    ego: Dict[str, Any],
    pred: Mapping[str, Any],
    fg_tau: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    """RSU pixel packs: all_valid, sam3_fg, heatmap_seed."""
    empty = {
        "pred": np.zeros((0,), dtype=np.float64),
        "gt": np.zeros((0,), dtype=np.float64),
        "var": np.zeros((0,), dtype=np.float64),
    }
    if "rsu" not in pred or "rsu" not in ego:
        return {"all_valid": empty, "sam3_fg": empty, "heatmap_seed": empty}
    cam = ego["rsu"]["batch_merged_cam_inputs"]
    z_gt = extract_camera_z_gt(cam["imgs"]).detach().cpu().numpy()
    z_pred = pred["rsu"]["depth_z_mean"].detach().cpu().numpy()
    z_var = pred["rsu"]["depth_z_var"].detach().cpu().numpy()
    enc = core.frontend.encoders["rsu"]
    valid = (
        np.isfinite(z_gt)
        & (z_gt >= float(enc.d_min))
        & (z_gt <= float(enc.d_max))
    )
    semantic = build_semantic_target(cam, tau=1).detach().cpu().numpy() > 0
    p_fg = (
        torch.softmax(pred["rsu"]["heatmap_logits"], dim=1)[:, 1]
        .detach()
        .cpu()
        .numpy()
    )
    seed = p_fg >= float(fg_tau)
    packs = {}
    for name, mask in (
        ("all_valid", valid),
        ("sam3_fg", valid & semantic),
        ("heatmap_seed", valid & seed),
    ):
        packs[name] = {
            "pred": z_pred[mask].reshape(-1),
            "gt": z_gt[mask].reshape(-1),
            "var": z_var[mask].reshape(-1),
        }
    return packs


def rsu_abc_and_aseed(
    core: nn.Module,
    ego: Dict[str, Any],
    pred: Mapping[str, Any],
    fg_tau: float,
    sigma0: float,
    near_m: float,
    cover_thresh: float,
    box_res: int,
    anisotropy_max: float,
    orient_window: int,
    eps: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """RSU A/B/C rows plus A-seed camera-z arrays."""
    empty = {
        "pred": np.zeros((0,), dtype=np.float64),
        "gt": np.zeros((0,), dtype=np.float64),
        "var": np.zeros((0,), dtype=np.float64),
    }
    if "rsu" not in pred:
        return [], empty
    cam = ego["rsu"]["batch_merged_cam_inputs"]
    if not torch.is_tensor(cam.get("imgs")):
        return [], empty
    imgs = flatten_imgs(cam["imgs"])
    n_cav = _record_len(ego, "rsu")
    n_view = n_views_of(imgs, n_cav, "rsu")
    n_flat = int(imgs.shape[0])
    p_fg = torch.softmax(pred["rsu"]["heatmap_logits"], dim=1)[:, 1].detach().cpu().numpy()
    pred_fg = p_fg >= float(fg_tau)
    boxes, class_ids = gt_boxes_from_ego(ego)
    if boxes.shape[0] == 0:
        return [], empty
    image_hw = (int(imgs.shape[-2]), int(imgs.shape[-1]))
    polygons = projected_polygons(ego, "rsu", boxes, image_hw)
    pairwise = pairwise_for_agent(ego, "rsu")
    intrins = _as_numpy(cam["intrinsics"])
    extrinsics = _as_numpy(cam["extrinsics"])
    post_rots = _as_numpy(cam["post_rots"])
    post_trans = _as_numpy(cam["post_trans"])
    z_mean = pred["rsu"]["depth_z_mean"].detach().cpu().numpy()
    z_var = pred["rsu"]["depth_z_var"].detach().cpu().numpy()
    z_gt = extract_camera_z_gt(cam["imgs"]).detach().cpu().numpy()
    depth_prob = (
        torch.softmax(pred["rsu"]["depth_logits"], dim=1).detach().cpu().numpy()
    )
    z_bins = core.depth_moments["rsu"].z_bins.detach().cpu().numpy().astype(np.float64)
    u_map, v_map = r90_pixel_centers()
    n_box = int(boxes.shape[0])
    visible = np.zeros((n_box,), dtype=bool)
    a_flags = np.zeros((n_box,), dtype=bool)
    b_flags = np.zeros((n_box,), dtype=bool)
    c_flags = np.zeros((n_box,), dtype=bool)
    box_mu: List[List[np.ndarray]] = [[] for _ in range(n_box)]
    box_sig: List[List[np.ndarray]] = [[] for _ in range(n_box)]
    aseed_pred: List[np.ndarray] = []
    aseed_gt: List[np.ndarray] = []
    aseed_var: List[np.ndarray] = []
    sigma0_list = [float(sigma0)]
    for flat in range(n_flat):
        local = flat // n_view
        view = flat % n_view
        ys, xs = np.where(pred_fg[flat])
        n_seed = int(ys.size)
        seed_u = u_map[ys, xs] if n_seed else np.zeros((0,), dtype=np.float64)
        seed_v = v_map[ys, xs] if n_seed else np.zeros((0,), dtype=np.float64)
        seed_uv = (
            np.stack([seed_u, seed_v], axis=1) if n_seed else np.zeros((0, 2), dtype=np.float64)
        )
        theta = None
        k = _slice_cam(intrins, local, view, n_view)
        ext = _slice_cam(extrinsics, local, view, n_view)
        prot = _slice_cam(post_rots, local, view, n_view)
        ptra = np.asarray(_slice_cam(post_trans, local, view, n_view)).reshape(-1)
        t_cav2ego = pairwise[local] if pairwise is not None else np.eye(4, dtype=np.float64)
        for box_i in range(n_box):
            poly = polygons[flat][box_i]
            if poly is None or len(poly) < 3:
                continue
            visible[box_i] = True
            if n_seed == 0:
                continue
            inside = MplPath(np.asarray(poly, dtype=np.float64)).contains_points(seed_uv)
            hit = np.flatnonzero(inside)
            if hit.size == 0:
                continue
            a_flags[box_i] = True
            aseed_pred.append(z_mean[flat, ys[hit], xs[hit]].reshape(-1))
            aseed_gt.append(z_gt[flat, ys[hit], xs[hit]].reshape(-1))
            aseed_var.append(z_var[flat, ys[hit], xs[hit]].reshape(-1))
            if theta is None:
                theta, aniso, _l1, _l2 = local_orientation(
                    p_fg[flat],
                    ys,
                    xs,
                    window=int(orient_window),
                    anisotropy_max=float(anisotropy_max),
                )
            z_cells = z_mean[flat, ys[hit], xs[hit]]
            prob_sel = depth_prob[flat][:, ys[hit], xs[hit]]
            mu_view, sig_view, _diff = view_gaussians(
                seed_u[hit],
                seed_v[hit],
                z_cells,
                theta[hit],
                aniso[hit],
                k,
                prot,
                ptra,
                ext,
                t_cav2ego,
                "rsu",
                sigma0_list,
                z_bins,
                prob_sel,
                2.0,
                float(eps),
            )
            if mu_view.shape[0]:
                box_mu[box_i].append(mu_view)
                box_sig[box_i].append(sig_view[float(sigma0)])
    rows: List[Dict[str, Any]] = []
    for box_i in range(n_box):
        if not visible[box_i]:
            continue
        if box_mu[box_i]:
            mu = np.concatenate(box_mu[box_i], axis=0)
            sig = np.concatenate(box_sig[box_i], axis=0)
            dist = obb_surface_distance(mu, boxes[box_i])
            b_flags[box_i] = bool(np.min(dist) <= float(near_m))
            pts = sample_box_points(boxes[box_i], int(box_res))
            prec = invert_spd(sig)
            cov = coverage_fraction(pts, mu, prec, k=2.0)
            c_flags[box_i] = bool(cov >= float(cover_thresh))
        rows.append(
            {
                "box_id": int(box_i),
                "class_id": int(class_ids[box_i]) if box_i < class_ids.size else 1,
                "A": bool(a_flags[box_i]),
                "B": bool(b_flags[box_i]),
                "C": bool(c_flags[box_i]),
            }
        )
    if aseed_pred:
        aseed = {
            "pred": np.concatenate(aseed_pred),
            "gt": np.concatenate(aseed_gt),
            "var": np.concatenate(aseed_var),
        }
    else:
        aseed = empty
    return rows, aseed


def summarize_abc(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    """P(A), P(B|A), P(C|A), P(ABC) on visible RSU GT."""
    n = len(rows)
    n_a = sum(1 for r in rows if r["A"])
    n_b = sum(1 for r in rows if r["A"] and r["B"])
    n_c = sum(1 for r in rows if r["A"] and r["C"])
    n_abc = sum(1 for r in rows if r["A"] and r["B"] and r["C"])

    def _rate(num: int, den: int) -> Optional[float]:
        if den <= 0:
            return None
        return float(num) / float(den)

    return {
        "n_visible": n,
        "P(A)": _rate(n_a, n),
        "P(B|A)": _rate(n_b, n_a),
        "P(C|A)": _rate(n_c, n_a),
        "P(ABC)": _rate(n_abc, n),
    }


def bin_errors(
    pred: np.ndarray,
    gt: np.ndarray,
    edges: Sequence[float],
) -> List[Dict[str, Any]]:
    """Median e_z / |e_z| per GT-depth bin. Same edges across checkpoints."""
    rows: List[Dict[str, Any]] = []
    edge_arr = np.asarray(edges, dtype=np.float64)
    if pred.size == 0:
        for i in range(len(edge_arr) - 1):
            rows.append(
                {
                    "z_lo": float(edge_arr[i]),
                    "z_hi": float(edge_arr[i + 1]),
                    "n": 0,
                    "median_ez": None,
                    "median_abs_ez": None,
                    "mae": None,
                }
            )
        return rows
    ez = pred.astype(np.float64) - gt.astype(np.float64)
    for i in range(len(edge_arr) - 1):
        lo, hi = float(edge_arr[i]), float(edge_arr[i + 1])
        if i == len(edge_arr) - 2:
            mask = (gt >= lo) & (gt <= hi)
        else:
            mask = (gt >= lo) & (gt < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "z_lo": lo,
                    "z_hi": hi,
                    "n": 0,
                    "median_ez": None,
                    "median_abs_ez": None,
                    "mae": None,
                }
            )
            continue
        ez_b = ez[mask]
        rows.append(
            {
                "z_lo": lo,
                "z_hi": hi,
                "n": n,
                "median_ez": float(np.median(ez_b)),
                "median_abs_ez": float(np.median(np.abs(ez_b))),
                "mae": float(np.mean(np.abs(ez_b))),
            }
        )
    return rows


def far_median_ez(
    pred: np.ndarray, gt: np.ndarray, far_lo: float = 25.0
) -> Optional[float]:
    """Median signed error on GT depth >= ``far_lo``."""
    if pred.size == 0:
        return None
    mask = gt >= float(far_lo)
    if int(mask.sum()) == 0:
        return None
    return float(np.median(pred[mask].astype(np.float64) - gt[mask].astype(np.float64)))


def choose_distance_edges(gt_depths: np.ndarray) -> List[float]:
    """Fixed metric-depth edges from source GT, always covering [2, 50]."""
    if gt_depths.size == 0:
        return list(DEFAULT_DISTANCE_EDGES_M)
    qs = np.quantile(gt_depths, [0.1, 0.25, 0.5, 0.75, 0.9])
    edges = [2.0]
    for value in qs:
        rounded = float(np.round(value, 1))
        if rounded > edges[-1] + 1.0 and rounded < 50.0:
            edges.append(rounded)
    if 25.0 not in edges and 25.0 > edges[-1]:
        edges.append(25.0)
    edges.append(50.0)
    # Keep DEFAULT edges as a superset so far-depth is never dropped.
    merged = sorted(set(float(x) for x in list(DEFAULT_DISTANCE_EDGES_M) + edges))
    return merged


def write_histogram(
    z_gt: np.ndarray,
    z_bins: np.ndarray,
    out_csv: Path,
    out_png: Path,
) -> Dict[str, Any]:
    """TRAIN SAM3∩in-range depth histogram. Diagnostic only."""
    n = int(z_gt.size)
    summary: Dict[str, Any] = {
        "n_supervised_pixels": n,
        "median_gt_m": None,
        "percentiles_gt_m": {},
        "near_mid_far": {},
        "bin_counts": [],
    }
    if n == 0:
        out_csv.write_text("bin,z_lo,z_hi,count\n")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("TRAIN RSU SAM3∩in-range GT depth (empty)")
        fig.savefig(out_png, dpi=130)
        plt.close(fig)
        return summary
    pct = {
        f"p{p}": float(np.percentile(z_gt, p)) for p in (10, 25, 50, 75, 90, 95)
    }
    summary["median_gt_m"] = pct["p50"]
    summary["percentiles_gt_m"] = pct
    for lo, hi in NEAR_MID_FAR:
        frac = float(np.mean((z_gt >= lo) & (z_gt < hi)))
        summary["near_mid_far"][f"{lo:g}-{hi:g}m"] = {
            "fraction": frac,
            "count": int(np.sum((z_gt >= lo) & (z_gt < hi))),
        }
    # Assign each GT z to nearest LID bin index via searchsorted on bin centers.
    centers = np.asarray(z_bins, dtype=np.float64)
    n_bins = int(centers.size)
    idx = np.clip(np.searchsorted(centers, z_gt, side="left"), 0, n_bins - 1)
    # Snap to closer neighbor.
    prev = np.clip(idx - 1, 0, n_bins - 1)
    use_prev = np.abs(z_gt - centers[prev]) < np.abs(z_gt - centers[idx])
    idx = np.where(use_prev, prev, idx)
    counts = np.bincount(idx, minlength=n_bins)
    lines = ["bin,z_center_m,count"]
    for i, count in enumerate(counts.tolist()):
        summary["bin_counts"].append(
            {"bin": i, "z_center_m": float(centers[i]), "count": int(count)}
        )
        lines.append(f"{i},{centers[i]:.6f},{int(count)}")
    out_csv.write_text("\n".join(lines) + "\n")
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(centers, counts, color="#3b82f6", linewidth=1.5)
    ax.fill_between(centers, counts, color="#3b82f6", alpha=0.25)
    ax.set_xlabel("GT camera-z (m)")
    ax.set_ylabel("Supervised pixel count")
    ax.set_title("TRAIN RSU depth target (SAM3 FG ∩ in-range)")
    ax.set_xlim(float(centers[0]), float(centers[-1]))
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return summary


def plot_residual_vs_depth(
    bin_rows: Sequence[Mapping[str, Any]],
    out_path: Path,
    title: str,
) -> None:
    """Median e_z vs GT-depth bin centers."""
    xs = []
    ys = []
    for row in bin_rows:
        if row["median_ez"] is None:
            continue
        xs.append(0.5 * (float(row["z_lo"]) + float(row["z_hi"])))
        ys.append(float(row["median_ez"]))
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    if xs:
        ax.plot(xs, ys, marker="o", color="#dc2626")
    ax.axhline(0.0, color="#6b7280", linewidth=1.0)
    ax.set_xlabel("GT camera-z bin center (m)")
    ax.set_ylabel("median e_z = z_pred - z_gt (m)")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_pred_vs_gt(
    pred: np.ndarray, gt: np.ndarray, out_path: Path, title: str
) -> None:
    """Scatter z_pred vs z_gt (subsampled)."""
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    if pred.size:
        n = pred.size
        if n > 8000:
            rng = np.random.RandomState(0)
            pick = rng.choice(n, size=8000, replace=False)
            pred_s = pred[pick]
            gt_s = gt[pick]
        else:
            pred_s, gt_s = pred, gt
        ax.scatter(gt_s, pred_s, s=4, alpha=0.25, c="#2563eb", linewidths=0)
        lo = min(float(gt.min()), float(pred.min()), 2.0)
        hi = max(float(gt.max()), float(pred.max()), 50.0)
        ax.plot([lo, hi], [lo, hi], color="#6b7280", linewidth=1.0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xlabel("z_gt (m)")
    ax.set_ylabel("z_pred (m)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_signed_hist(
    pred: np.ndarray, gt: np.ndarray, out_path: Path, title: str
) -> None:
    """Histogram of e_z."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    if pred.size:
        ez = pred.astype(np.float64) - gt.astype(np.float64)
        ax.hist(ez, bins=60, color="#7c3aed", alpha=0.85)
        ax.axvline(0.0, color="#6b7280", linewidth=1.0)
        ax.axvline(float(np.median(ez)), color="#dc2626", linewidth=1.2, linestyle="--")
    ax.set_xlabel("e_z = z_pred - z_gt (m)")
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_train_vs_test(rows: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    """TRAIN/VAL/TEST MAE and median e_z vs checkpoint order."""
    tags = [r["tag"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for split, color in (("train", "#2563eb"), ("val", "#ca8a04"), ("test", "#dc2626")):
        mae = [r["splits"][split]["aseed"]["mae"] if r["splits"][split]["aseed"]["mae"] is not None else float("nan") for r in rows]
        med = [r["splits"][split]["aseed"]["median_ez"] if r["splits"][split]["aseed"]["median_ez"] is not None else float("nan") for r in rows]
        axes[0].plot(tags, mae, marker="o", label=split, color=color)
        axes[1].plot(tags, med, marker="o", label=split, color=color)
    axes[0].set_ylabel("A-seed MAE (m)")
    axes[0].set_title("A-seed MAE by split")
    axes[1].axhline(0.0, color="#6b7280", linewidth=1.0)
    axes[1].set_ylabel("A-seed median e_z (m)")
    axes[1].set_title("A-seed median signed error")
    for ax in axes:
        ax.set_xlabel("checkpoint")
        ax.legend()
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "mae_and_signed.png", dpi=140)
    plt.close(fig)


def build_eval_plans(
    datasets: Mapping[str, Any], frames_per_scene: int, seed: int
) -> Dict[str, List[Tuple[str, int]]]:
    """Fixed per-split frame lists."""
    return {
        split: sample_plan(dataset, frames_per_scene, seed)
        for split, dataset in datasets.items()
    }


def scene_of_sample(dataset: Any, idx: int) -> str:
    """Scenario timestamp from a dataset index."""
    sample = dataset[idx]
    # dataset[idx] may already be ego or wrapped; collate later in caller.
    return ""
