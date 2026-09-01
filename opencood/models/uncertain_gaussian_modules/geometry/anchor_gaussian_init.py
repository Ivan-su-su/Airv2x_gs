from typing import Dict, List, Optional

import torch.nn as nn


class AnchorGaussianInitModule(nn.Module):
    """根据全局体素结果初始化 anchor gaussian。"""

    def __init__(self, model_cfg=None):
        super().__init__()
        self.model_cfg = model_cfg or {}

    def forward(
        self, batch_dict: Dict, available_agents: Optional[List[str]] = None
    ) -> Dict:
        gp = batch_dict.setdefault("gaussian_pipeline", {})
        gp["available_agents"] = available_agents or gp.get("available_agents", [])
        gp.setdefault("anchor_gaussians", None)
        gp.setdefault("gaussian_candidates", {})
        gp.setdefault("projection_masks", {})
        return batch_dict
