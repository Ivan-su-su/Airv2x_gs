#!/usr/bin/env python3
from pathlib import Path
import py_compile

ROOT = Path.cwd()
OPACITY = ROOT / "opencood/models/gaussian_modules_0822/fusion/gaussian_opacity.py"
CONTEXT = ROOT / "opencood/models/gaussian_modules_0822/interaction/gaussian_context.py"
YAML = ROOT / "opencood/hypes_yaml/airv2x/camera/det/airv2x_gaussian_0822_geom_sup_opacity.yaml"

OPACITY_CODE = '# Agent-specific learned Gaussian opacity for Stage-3.\n\nfrom __future__ import annotations\n\nimport math\n\nimport torch\nfrom torch import nn\n\nfrom opencood.models.gaussian_modules_0822.base.gaussian import GaussianSet\nfrom opencood.models.gaussian_modules_0822.base.gaussian_transform import (\n    reconstruct_covariance,\n)\n\n\nclass GaussianOpacityCalibrator(nn.Module):\n    def __init__(\n        self,\n        hidden_dim: int = 16,  # YAML compatibility only\n        init_prob: float = 0.95,\n        detach_covariance: bool = True,\n        cell_x: float = 0.4,\n        cell_y: float = 0.4,\n        prob_eps: float = 1.0e-4,\n        var_eps: float = 1.0e-6,\n    ) -> None:\n        super().__init__()\n        del hidden_dim\n\n        if not 0.0 < init_prob < 1.0:\n            raise ValueError(f"init_prob must be in (0,1), got {init_prob}")\n\n        self.detach_covariance = bool(detach_covariance)\n        self.cell_x = float(cell_x)\n        self.cell_y = float(cell_y)\n        self.prob_eps = float(prob_eps)\n        self.var_eps = float(var_eps)\n\n        self.net = nn.Sequential(\n            nn.Linear(4, 16),\n            nn.SiLU(),\n            nn.Linear(16, 32),\n            nn.SiLU(),\n            nn.Linear(32, 8),\n            nn.SiLU(),\n            nn.Linear(8, 1),\n        )\n\n        last = self.net[-1]\n        nn.init.normal_(last.weight, mean=0.0, std=1.0e-3)\n        nn.init.constant_(\n            last.bias,\n            math.log(float(init_prob) / (1.0 - float(init_prob))),\n        )\n\n    def forward(self, gaussians: GaussianSet) -> torch.Tensor:\n        if gaussians.n_gaussians == 0:\n            return gaussians.feature.new_empty((0,))\n        if gaussians.p_fg is None:\n            raise ValueError("opacity calibration requires GaussianSet.p_fg")\n\n        p = gaussians.p_fg.float().clamp(\n            min=self.prob_eps,\n            max=1.0 - self.prob_eps,\n        )\n        p_logit = torch.log(p) - torch.log1p(-p)\n\n        scale = gaussians.scale.float()\n        quaternion = gaussians.quaternion.float()\n        if self.detach_covariance:\n            scale = scale.detach()\n            quaternion = quaternion.detach()\n\n        cov = reconstruct_covariance(scale, quaternion)\n        xx = cov[:, 0, 0].clamp_min(self.var_eps)\n        yy = cov[:, 1, 1].clamp_min(self.var_eps)\n        xy = cov[:, 0, 1]\n\n        log_var_x = torch.log(xx / (self.cell_x * self.cell_x))\n        log_var_y = torch.log(yy / (self.cell_y * self.cell_y))\n        corr_xy = xy / torch.sqrt((xx * yy).clamp_min(self.var_eps))\n        corr_xy = corr_xy.clamp(-1.0, 1.0)\n\n        x = torch.stack(\n            [p_logit, log_var_x, log_var_y, corr_xy],\n            dim=-1,\n        )\n\n        if not torch.isfinite(x).all():\n            n_bad = int((~torch.isfinite(x)).any(dim=-1).sum().item())\n            raise RuntimeError(f"non-finite opacity inputs: n_bad={n_bad}")\n\n        opacity = torch.sigmoid(self.net(x).squeeze(-1))\n        return opacity.to(dtype=gaussians.feature.dtype)\n'
OLD_CTOR = '                    hidden_dim=int(opacity.get("hidden_dim", 8)),\n                    init_prob=float(opacity.get("init_prob", 0.95)),\n                    detach_covariance=bool(opacity.get("detach_covariance", True)),\n                    cell_area=cell_area,\n'
NEW_CTOR = '                    hidden_dim=int(opacity.get("hidden_dim", 16)),\n                    init_prob=float(opacity.get("init_prob", 0.95)),\n                    detach_covariance=bool(opacity.get("detach_covariance", True)),\n                    cell_x=float(voxel_size[0]),\n                    cell_y=float(voxel_size[1]),\n'

def main():
    context = CONTEXT.read_text()

    required = [
        "self.opacity_heads = nn.ModuleDict()",
        'keys = agents if self.opacity_agent_specific else ["shared"]',
        'key = agent if self.opacity_agent_specific else "shared"',
    ]
    missing = [m for m in required if m not in context]
    if missing:
        raise RuntimeError(
            "gaussian_context.py is not the expected agent-specific version; "
            f"missing={missing}"
        )

    if OLD_CTOR in context:
        context = context.replace(OLD_CTOR, NEW_CTOR, 1)
        context = context.replace(
            "            cell_area = float(voxel_size[0]) * float(voxel_size[1])\n",
            "",
            1,
        )
        CONTEXT.write_text(context)
    elif "cell_x=float(voxel_size[0])" not in context:
        raise RuntimeError("opacity constructor block not recognized")

    OPACITY.write_text(OPACITY_CODE)

    if YAML.exists():
        y = YAML.read_text()
        y = y.replace("agent_specific: false", "agent_specific: true")
        if "agent_specific: true" not in y:
            y = y.replace(
                "detach_covariance: true\n",
                "detach_covariance: true\n        agent_specific: true\n",
                1,
            )
        y = y.replace("hidden_dim: 8", "hidden_dim: 16")
        YAML.write_text(y)

    py_compile.compile(str(OPACITY), doraise=True)
    py_compile.compile(str(CONTEXT), doraise=True)

    print("[ok] removed determinant/area opacity logic")
    print("[ok] input = logit(p_fg), log(var_x), log(var_y), corr_xy")
    print("[ok] MLP = 4 -> 16 -> 32 -> 8 -> 1")
    print("[ok] vehicle/rsu/drone use separate opacity heads")
    print("[ok] py_compile passed")

if __name__ == "__main__":
    main()
