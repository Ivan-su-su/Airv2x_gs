"""RSU variance is centered on the global mean, not the peak."""

from __future__ import annotations

import torch

from opencood.models.gaussian_modules_0822.depth.depth_initializer import (
    categorical_depth_moments,
)
from opencood.models.gaussian_modules_0822.heatmap.orientation import pca_cell_radius


def test_rsu_window_follows_global_mean_not_peak() -> None:
    bins = torch.linspace(1.0, 49.0, 49)
    logits = torch.full((1, 49, 1, 1), -20.0)
    logits[0, 8:18, 0, 0] = 2.0
    logits[0, 40, 0, 0] = 3.0
    view = torch.zeros(1, dtype=torch.long)
    y = torch.zeros(1, dtype=torch.long)
    x = torch.zeros(1, dtype=torch.long)
    rsu = categorical_depth_moments(logits, bins, view, y, x, agent="rsu", window_m=10.0)
    veh = categorical_depth_moments(logits, bins, view, y, x, agent="vehicle", window_m=10.0)
    prob = torch.softmax(logits[0, :, 0, 0], dim=0)
    mu_global = float((prob * bins).sum())
    peak = float(bins[40])
    assert abs(float(rsu["depth_mean"][0]) - mu_global) < 1.0e-5
    assert abs(float(veh["depth_mean"][0]) - peak) < 2.0
    assert float(rsu["depth_var"][0]) < 40.0
    old = (peak - mu_global) ** 2
    assert float(rsu["depth_var"][0]) < 0.2 * old


def test_pca_cell_radius_uses_feature_stride() -> None:
    assert pca_cell_radius((360, 640), (90, 160), 40.0) == 10
    assert pca_cell_radius((360, 640), (45, 80), 40.0) == 5


if __name__ == "__main__":
    test_rsu_window_follows_global_mean_not_peak()
    test_pca_cell_radius_uses_feature_stride()
    print("rsu variance / pca radius ok")
