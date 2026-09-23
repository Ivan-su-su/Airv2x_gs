#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose two things in Airv2x_gs / baseline:

1) Geometry mapping consistency
   A. Verify the current Gaussian seed lift follows AirV2X/LSS's
      linspace(0, W-1, fW) / linspace(0, H-1, fH) convention.
   B. Diagnose the later 3D->image->feature sampling path:
      current MambaProjectionWrapper uses uv_grid = uv_norm * 2 - 1 while
      torch.grid_sample uses align_corners=False.
      Compare current ("before") against a candidate corrected mapping ("after")
      WITHOUT modifying the repository.

2) Gaussian sampling cost
   Read DEFAULT_NUM_OFFSETS and AirV2X camera counts from the repository,
   then report:
     unique 3D points/Gaussian = 2*K+1
     Stage-1 token slots/Gaussian
     Stage-2 token slots/Gaussian
     rough sampled-feature payload per 1000 Gaussians.

Run from repository root:
    cd /mnt/home/suyi/AirV2X-Perception_gaussian
    python diagnose_geometry_sampling.py

This file DOES NOT patch any source.
Python 3.8 compatible.
"""

from __future__ import print_function

import ast
import math
import re
from pathlib import Path

import torch


ROOT = Path.cwd()

PROJECTION = ROOT / "opencood/models/gaussian_modules_0822/projection/mamba_projection_wrapper.py"
HEATMAP_GEOM = ROOT / "opencood/models/gaussian_modules_0822/heatmap/geometry.py"
SAMPLER = ROOT / "opencood/models/gaussian_modules_0822/sampling/gaussian_offset_sampler.py"
AIRV2X_UTILS = ROOT / "opencood/utils/airv2x_utils.py"
BASE_YAML = ROOT / "opencood/hypes_yaml/airv2x/camera/det/airv2x_gaussian_0822.yaml"
P1_LAYOUT = ROOT / "opencood/models/gaussian_modules_0822/p1_layout.py"


def require(path):
    if not path.exists():
        raise FileNotFoundError(
            "%s not found. Run this script from the AirV2X-Perception_gaussian repo root."
            % path
        )


for _p in [PROJECTION, HEATMAP_GEOM, SAMPLER, AIRV2X_UTILS, BASE_YAML, P1_LAYOUT]:
    require(_p)


def ast_constant(path, name):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                return ast.literal_eval(node.value)
    raise KeyError("%s not found in %s" % (name, path))


def parse_max_cav_and_v45():
    # Avoid depending on the repo's YAML loader; these patterns are stable in current config.
    txt = BASE_YAML.read_text()

    def grab_int(pattern, default=None):
        m = re.search(pattern, txt)
        if m is None:
            if default is not None:
                return default
            raise RuntimeError("Pattern not found: %s" % pattern)
        return int(m.group(1))

    max_cav = {}
    block = re.search(
        r"max_cav:\s*&max_cav\s*\n(?P<body>(?:\s+\w+:\s*\d+\s*\n)+)", txt
    )
    if block:
        for k, v in re.findall(r"^\s+(vehicle|rsu|drone):\s*(\d+)\s*$",
                               block.group("body"), flags=re.M):
            max_cav[k] = int(v)

    if set(max_cav) != {"vehicle", "rsu", "drone"}:
        # Current defaults, only used if YAML formatting changes.
        max_cav = {"vehicle": 3, "rsu": 2, "drone": 2}

    m = re.search(r"vehicle_p1_v45:\s*(true|false)", txt, flags=re.I)
    vehicle_v45 = (m is not None and m.group(1).lower() == "true")
    return max_cav, vehicle_v45


def read_current_source_contracts():
    proj = PROJECTION.read_text()
    sampler = SAMPLER.read_text()
    geom = HEATMAP_GEOM.read_text()

    print("=" * 88)
    print("SOURCE CONTRACT CHECK")
    print("=" * 88)

    print("[projection]")
    print("  align_corners default False:",
          "align_corners: bool = False" in proj)
    print("  current uv_grid = uv_norm*2-1:",
          '"uv_grid": uv_norm * 2.0 - 1.0' in proj)

    print("[seed lift]")
    lss_formula_present = (
        "float(image_width - 1) / float(feature_width - 1)" in geom
        and "float(image_height - 1) / float(feature_height - 1)" in geom
    )
    print("  feature cell -> image pixel uses LSS endpoint/linspace convention:",
          lss_formula_present)

    k = int(ast_constant(SAMPLER, "DEFAULT_NUM_OFFSETS"))
    symmetric = (
        "torch.cat([center + delta, center - delta, center], dim=1)" in sampler
    )
    print("[sampler]")
    print("  DEFAULT_NUM_OFFSETS =", k)
    print("  symmetric {+delta, -delta, center} implementation:", symmetric)
    print("  unique xyz samples/Gaussian = 2*K+1 =", 2 * k + 1)
    print()
    return k


def feature_to_lss_image_xy(ix, iy, image_hw, feature_hw):
    """Exact convention used by AirV2X LSS create_frustum."""
    H, W = image_hw
    fH, fW = feature_hw
    ix = ix.float()
    iy = iy.float()

    if fW > 1:
        u = ix * float(W - 1) / float(fW - 1)
    else:
        u = torch.zeros_like(ix)
    if fH > 1:
        v = iy * float(H - 1) / float(fH - 1)
    else:
        v = torch.zeros_like(iy)
    return u, v


def grid_to_feature_xy_align_false(gx, gy, feature_hw):
    """Inverse coordinate relation of grid_sample(... align_corners=False)."""
    fH, fW = feature_hw
    x = ((gx + 1.0) * float(fW) - 1.0) / 2.0
    y = ((gy + 1.0) * float(fH) - 1.0) / 2.0
    return x, y


def current_projected_grid(u, v, image_hw):
    """Current repo behavior: uv_norm=(u/W,v/H), uv_grid=2*uv_norm-1."""
    H, W = image_hw
    gx = 2.0 * (u / float(W)) - 1.0
    gy = 2.0 * (v / float(H)) - 1.0
    return gx, gy


def candidate_fixed_grid(u, v, image_hw, feature_hw):
    """
    Candidate correction while KEEPING align_corners=False.

    First map image coordinates back to the LSS feature coordinate system:
        xf = u * (fW-1)/(W-1)
        yf = v * (fH-1)/(H-1)
    then convert feature coordinates to grid_sample's align_corners=False grid:
        gx = 2*(xf+0.5)/fW - 1
        gy = 2*(yf+0.5)/fH - 1
    """
    H, W = image_hw
    fH, fW = feature_hw

    if W > 1 and fW > 1:
        xf = u * float(fW - 1) / float(W - 1)
    else:
        xf = torch.zeros_like(u)
    if H > 1 and fH > 1:
        yf = v * float(fH - 1) / float(H - 1)
    else:
        yf = torch.zeros_like(v)

    gx = 2.0 * (xf + 0.5) / float(fW) - 1.0
    gy = 2.0 * (yf + 0.5) / float(fH) - 1.0
    return gx, gy


def stats(name, ex, ey):
    dist = torch.sqrt(ex.square() + ey.square())
    print("  %-11s | mean|dx|=%8.6f  max|dx|=%8.6f  "
          "mean|dy|=%8.6f  max|dy|=%8.6f  mean2D=%8.6f  max2D=%8.6f"
          % (
              name,
              ex.abs().mean().item(),
              ex.abs().max().item(),
              ey.abs().mean().item(),
              ey.abs().max().item(),
              dist.mean().item(),
              dist.max().item(),
          ))


def geometry_case(label, image_hw, feature_hw):
    H, W = image_hw
    fH, fW = feature_hw
    yy, xx = torch.meshgrid(
        torch.arange(fH, dtype=torch.float64),
        torch.arange(fW, dtype=torch.float64),
        indexing="ij",
    )

    # Step A: seed lift convention.
    u, v = feature_to_lss_image_xy(xx, yy, image_hw, feature_hw)
    lss_x_ref = torch.linspace(0.0, float(W - 1), fW, dtype=torch.float64)
    lss_y_ref = torch.linspace(0.0, float(H - 1), fH, dtype=torch.float64)
    lift_err_x = (u[0] - lss_x_ref).abs().max().item()
    lift_err_y = (v[:, 0] - lss_y_ref).abs().max().item()

    # Step B: project those exact seed-center pixels back to F90 via old/new grid rules.
    gx_old, gy_old = current_projected_grid(u, v, image_hw)
    x_old, y_old = grid_to_feature_xy_align_false(gx_old, gy_old, feature_hw)

    gx_new, gy_new = candidate_fixed_grid(u, v, image_hw, feature_hw)
    x_new, y_new = grid_to_feature_xy_align_false(gx_new, gy_new, feature_hw)

    print("=" * 88)
    print("GEOMETRY TEST:", label)
    print("image HxW=%dx%d, feature HxW=%dx%d" % (H, W, fH, fW))
    print("=" * 88)
    print("[A] Current seed lift vs official AirV2X/LSS frustum linspace")
    print("  max pixel error: x=%.9f px, y=%.9f px" % (lift_err_x, lift_err_y))
    print("  -> PASS" if max(lift_err_x, lift_err_y) < 1e-9 else "  -> FAIL")

    print("[B] Round-trip: feature cell -> LSS image pixel -> sampled feature coordinate")
    stats("BEFORE", x_old - xx, y_old - yy)
    stats("AFTER", x_new - xx, y_new - yy)

    # Corners are the most intuitive evidence.
    corners = [
        (0, 0),
        (0, fW - 1),
        (fH - 1, 0),
        (fH - 1, fW - 1),
        (fH // 2, fW // 2),
    ]
    print("  selected points: expected -> BEFORE -> AFTER")
    for y0, x0 in corners:
        print(
            "    (%3d,%3d) -> (%8.4f,%8.4f) -> (%8.4f,%8.4f)"
            % (
                x0, y0,
                x_old[y0, x0].item(), y_old[y0, x0].item(),
                x_new[y0, x0].item(), y_new[y0, x0].item(),
            )
        )
    print()


def sampled_payload_mib(tokens_per_gaussian, channels, n_gaussians=1000, bytes_per=4):
    return (
        float(tokens_per_gaussian)
        * float(channels)
        * float(n_gaussians)
        * float(bytes_per)
        / (1024.0 ** 2)
    )


def sampling_cost(k):
    camera_keys = ast_constant(AIRV2X_UTILS, "CAMERA_KEYS_BY_AGENT")
    cams_per_cav = {k_: len(v_) for k_, v_ in camera_keys.items()}
    max_cav, vehicle_v45 = parse_max_cav_and_v45()

    try:
        channels = int(ast_constant(P1_LAYOUT, "F90_CHANNELS"))
    except Exception:
        channels = 128

    S = 2 * k + 1

    print("=" * 88)
    print("SAMPLING / TOKEN COST")
    print("=" * 88)
    print("K =", k)
    print("Unique 3D sample points per Gaussian = 2*K+1 =", S)
    print("Important: this is NOT 2^3+1. Each of the K outputs is already a 3D")
    print("vector (a,b,c); the code takes its +/- pair plus one center.")
    print()
    print("AirV2X cameras/CAV:", cams_per_cav)
    print("Current max_cav:", max_cav)
    print("vehicle_p1_v45:", vehicle_v45)
    print("F90 channels:", channels)
    print()

    print("[Stage-1: query Gaussian samples only cameras of its own CAV]")
    for agent in ("vehicle", "rsu", "drone"):
        views = cams_per_cav[agent]
        tokens = S * views
        mem = sampled_payload_mib(tokens, channels)
        print(
            "  %-7s: %d xyz -> %d cameras -> %d token slots/Gaussian"
            " | sampled feature payload ~= %.2f MiB / 1000 Gaussians (FP32)"
            % (agent, S, views, tokens, mem)
        )

    # Stage-2 uses all flattened cameras from source agent types.
    nveh = max_cav["vehicle"] * cams_per_cav["vehicle"]
    nrsu = max_cav["rsu"] * cams_per_cav["rsu"]
    ndro = max_cav["drone"] * cams_per_cav["drone"]

    stage2 = [
        ("drone <- vehicle+rsu", nveh + nrsu),
        ("vehicle <- drone", ndro),
        ("rsu <- drone", ndro),
    ]
    print()
    print("[Stage-2: current worst-case camera slots under max_cav]")
    for name, views in stage2:
        tokens = S * views
        mem = sampled_payload_mib(tokens, channels)
        print(
            "  %-24s: %d xyz x %d source cameras = %d token slots/Gaussian"
            " | ~= %.2f MiB / 1000 Gaussians (FP32)"
            % (name, S, views, tokens, mem)
        )

    print()
    print("Interpretation:")
    print("  * The Gaussian itself still has only %d sampled xyz points." % S)
    print("  * Projection/sampling cost multiplies by the number of source cameras.")
    print("  * Invalid projected tokens are masked later, but they are still materialized")
    print("    by projection/grid_sample before attention masking.")
    print("  * The values above are raw sampled-feature tensors only; Q/K/V, masks,")
    print("    adapters and autograd activations add extra memory/compute.")
    print()


def main():
    k = read_current_source_contracts()

    # Current base config uses 360x640 images. RSU/Drone F90 are 90x160;
    # Vehicle is 45x80 when vehicle_p1_v45=true.
    max_cav, vehicle_v45 = parse_max_cav_and_v45()
    del max_cav

    geometry_case(
        "RSU/Drone F90",
        image_hw=(360, 640),
        feature_hw=(90, 160),
    )
    geometry_case(
        "Vehicle F90 (V45)" if vehicle_v45 else "Vehicle F90",
        image_hw=(360, 640),
        feature_hw=(45, 80) if vehicle_v45 else (90, 160),
    )

    sampling_cost(k)

    print("=" * 88)
    print("DIAGNOSIS")
    print("=" * 88)
    print("1) Seed lift/backprojection mapping itself is expected to PASS: it matches")
    print("   AirV2X LSS's torch.linspace(0, W-1, fW) convention.")
    print("2) If BEFORE shows ~0.2-cell mean error / 0.5-cell max error while AFTER")
    print("   is ~0, the inconsistency is specifically in the later projected-feature")
    print("   sampling coordinate conversion (uv_norm -> grid_sample), not in LSS lift.")
    print("3) This candidate AFTER result is diagnostic only; this script does not edit")
    print("   mamba_projection_wrapper.py.")
    print("4) Sampling count is read from the current repo, so K=3 should report 7 xyz")
    print("   samples/Gaussian and the camera-multiplied token costs above.")


if __name__ == "__main__":
    main()
