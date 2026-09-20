#!/usr/bin/env python3
from pathlib import Path
import py_compile

ROOT = Path.cwd()
POOL = ROOT / "opencood/models/gaussian_modules_0822/fusion/gaussian_voxel_pool.py"
CONTEXT = ROOT / "opencood/models/gaussian_modules_0822/interaction/gaussian_context.py"

POOL_CODE = '"""Per-agent opacity-aware voxel pooling for Stage-3.\n\nBackward compatible:\n- weights=None: exact legacy mean pooling behavior.\n- weights=[N]: normalized opacity-weighted pooling.\n\nGeometry stays in scale/quaternion form. We deliberately do NOT pool a full\ncovariance and eigendecompose it during training, because backward through\neigenvectors becomes unstable for repeated / nearly repeated eigenvalues.\n"""\n\nfrom __future__ import annotations\n\nfrom typing import Any, Optional\n\nimport torch\n\nfrom opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet\nfrom opencood.models.gaussian_modules_0822.base.gaussian_transform import (\n    normalize_quaternion,\n)\n\nDEFAULT_VOXEL_SIZE = (0.4, 0.4, 0.125)\nDEFAULT_POINT_CLOUD_RANGE = (-140.8, -40.0, -150.0, 140.8, 40.0, 30.0)\n\n\ndef scatter_mean(\n    values: torch.Tensor,\n    inverse: torch.Tensor,\n    n_groups: int,\n) -> torch.Tensor:\n    sums = values.new_zeros((n_groups, values.shape[1]))\n    sums.index_add_(0, inverse, values)\n    counts = torch.bincount(inverse, minlength=n_groups).to(values.dtype)\n    return sums / counts.clamp_min(1).unsqueeze(-1)\n\n\ndef scatter_weighted_mean(\n    values: torch.Tensor,\n    weights: torch.Tensor,\n    inverse: torch.Tensor,\n    n_groups: int,\n) -> torch.Tensor:\n    if weights.dim() != 1 or int(weights.shape[0]) != int(values.shape[0]):\n        raise ValueError(\n            f"weights must be [N], got {tuple(weights.shape)} "\n            f"for values {tuple(values.shape)}"\n        )\n\n    w = weights.to(device=values.device, dtype=values.dtype)\n    sums = values.new_zeros((n_groups, values.shape[1]))\n    sums.index_add_(0, inverse, values * w.unsqueeze(-1))\n\n    denom = values.new_zeros((n_groups,))\n    denom.index_add_(0, inverse, w)\n    return sums / denom.clamp_min(1.0e-8).unsqueeze(-1)\n\n\nclass GaussianVoxelPool:\n    def __init__(\n        self,\n        voxel_size: Any = DEFAULT_VOXEL_SIZE,\n        point_cloud_range: Any = DEFAULT_POINT_CLOUD_RANGE,\n    ) -> None:\n        self.voxel_size = torch.tensor([float(v) for v in voxel_size])\n        self.pc_min = torch.tensor([float(v) for v in point_cloud_range[:3]])\n        pc_max = torch.tensor([float(v) for v in point_cloud_range[3:]])\n        self.grid = torch.floor(\n            (pc_max - self.pc_min) / self.voxel_size\n        ).long()\n\n    def __call__(\n        self,\n        gaussians: GaussianSet,\n        weights: Optional[torch.Tensor] = None,\n    ) -> GaussianSet:\n        n = gaussians.n_gaussians\n        if n == 0:\n            return gaussians\n\n        if weights is not None:\n            if weights.dim() != 1 or int(weights.shape[0]) != n:\n                raise ValueError(\n                    f"weights must be [N={n}], got {tuple(weights.shape)}"\n                )\n\n        device = gaussians.mean.device\n        dtype = gaussians.mean.dtype\n        voxel_size = self.voxel_size.to(device=device, dtype=dtype)\n        pc_min = self.pc_min.to(device=device, dtype=dtype)\n        grid = self.grid.to(device)\n\n        idx = torch.floor(\n            (gaussians.mean - pc_min) / voxel_size\n        ).long()\n        valid = ((idx >= 0) & (idx < grid)).all(dim=-1)\n\n        kept = gaussians.index_select(valid)\n        if kept.n_gaussians == 0:\n            return kept\n\n        kept_weights = None if weights is None else weights[valid]\n        idx = idx[valid]\n\n        batch = (\n            kept.batch_index\n            if kept.batch_index is not None\n            else torch.zeros(\n                kept.n_gaussians,\n                dtype=torch.long,\n                device=device,\n            )\n        )\n\n        gx, gy, gz = (int(v) for v in grid.tolist())\n        code = (\n            (batch * gx + idx[:, 0]) * gy + idx[:, 1]\n        ) * gz + idx[:, 2]\n\n        unq_code, inverse = torch.unique(\n            code,\n            return_inverse=True,\n        )\n        inverse = inverse.long()\n        m = int(unq_code.numel())\n        pooled_batch = unq_code // (gx * gy * gz)\n\n        def _pool(values: torch.Tensor) -> torch.Tensor:\n            if kept_weights is None:\n                return scatter_mean(values, inverse, m)\n            return scatter_weighted_mean(\n                values,\n                kept_weights,\n                inverse,\n                m,\n            )\n\n        # Stable quaternion pooling:\n        # align q/-q against the first quaternion in each voxel,\n        # then take the same normalized opacity-weighted mean.\n        q = kept.quaternion\n        order = torch.argsort(inverse, stable=True)\n        sorted_inv = inverse[order]\n        boundary = torch.ones_like(\n            sorted_inv,\n            dtype=torch.bool,\n        )\n        boundary[1:] = sorted_inv[1:] != sorted_inv[:-1]\n        first_rows = order[boundary]\n        reference = q[first_rows][inverse]\n\n        sign = torch.where(\n            (q * reference).sum(-1, keepdim=True) < 0.0,\n            -1.0,\n            1.0,\n        )\n        pooled_quat = normalize_quaternion(\n            _pool(q * sign)\n        )\n\n        # Scale is pooled directly, exactly like the old stable baseline,\n        # except opacity supplies normalized weights when enabled.\n        # This remains a convex mean, so the number of Gaussians in a voxel\n        # cannot by itself make scale explode.\n        pooled_scale = _pool(kept.scale)\n\n        return GaussianSet(\n            mean=_pool(kept.mean),\n            scale=pooled_scale,\n            quaternion=pooled_quat,\n            feature=_pool(kept.feature),\n            p_fg=(\n                _pool(kept.p_fg.unsqueeze(-1)).squeeze(-1)\n                if kept.p_fg is not None\n                else None\n            ),\n            batch_index=pooled_batch,\n            agent=kept.agent,\n            frame=kept.frame,\n        )\n'

def main():
    if not POOL.exists() or not CONTEXT.exists():
        raise FileNotFoundError(
            "Run from /mnt/home/suyi/AirV2X-Perception_gaussian"
        )

    # Replace the unstable covariance->eigh pooling implementation completely.
    POOL.write_text(POOL_CODE)

    context = CONTEXT.read_text()
    old = """                pooled_gs = self.pool(
                    gs,
                    weights=alpha_pre,
                    covariance_pool=True,
                )
"""
    new = """                pooled_gs = self.pool(
                    gs,
                    weights=alpha_pre,
                )
"""
    if old in context:
        context = context.replace(old, new, 1)
    elif "covariance_pool=True" in context:
        context = context.replace(
            ",\n                    covariance_pool=True",
            "",
            1,
        )
    elif "weights=alpha_pre" not in context:
        raise RuntimeError(
            "Could not find opacity-aware pooling call in gaussian_context.py"
        )

    CONTEXT.write_text(context)

    py_compile.compile(str(POOL), doraise=True)
    py_compile.compile(str(CONTEXT), doraise=True)

    print("[ok] removed covariance eigendecomposition from trainable pooling")
    print("[ok] opacity still weights mean / scale / quaternion / feature / p_fg")
    print("[ok] scale uses normalized weighted mean")
    print("[ok] quaternion uses antipodal aligned normalized weighted mean")
    print("[ok] py_compile passed")

if __name__ == "__main__":
    main()
