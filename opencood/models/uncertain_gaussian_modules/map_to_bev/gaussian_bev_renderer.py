from typing import Dict, List, Optional

import torch.nn as nn


class GaussianBEVRenderer(nn.Module):
    """将最终 gaussian 集合渲染到 BEV 特征图。"""

    def __init__(self, model_cfg=None):
        super().__init__()
        self.model_cfg = model_cfg or {}

    def render_to_bev(
        self, batch_dict: Dict, available_agents: Optional[List[str]] = None
    ) -> Dict:
        """预留 gaussian splatting 到 BEV 的主接口。"""
        gp = batch_dict.setdefault("gaussian_pipeline", {})
        gp.setdefault("bev_render_output", None)
        return batch_dict

    def forward(
        self, batch_dict: Dict, available_agents: Optional[List[str]] = None
    ) -> Dict:
        gp = batch_dict.setdefault("gaussian_pipeline", {})
        gp["available_agents"] = available_agents or gp.get("available_agents", [])
        batch_dict = self.render_to_bev(batch_dict, available_agents)
        return batch_dict
