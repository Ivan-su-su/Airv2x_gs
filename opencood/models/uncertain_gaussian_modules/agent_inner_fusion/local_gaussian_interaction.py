from typing import Dict, List, Optional

import torch.nn as nn


class LocalGaussianInteractionModule(nn.Module):
    """完成 dual-path split 与局部 self/cross interaction。"""

    def __init__(self, model_cfg=None):
        super().__init__()
        self.model_cfg = model_cfg or {}

    def split_dual_path_features(
        self, batch_dict: Dict, available_agents: Optional[List[str]] = None
    ) -> Dict:
        """预留 self feature 与 cross feature 拆分接口。"""
        gp = batch_dict.setdefault("gaussian_pipeline", {})
        gp.setdefault("dual_path_features", None)
        return batch_dict

    def run_local_interaction(
        self, batch_dict: Dict, available_agents: Optional[List[str]] = None
    ) -> Dict:
        """预留局部窗口内的高斯交互接口。"""
        gp = batch_dict.setdefault("gaussian_pipeline", {})
        gp.setdefault("local_interaction_features", None)
        return batch_dict

    def forward(
        self, batch_dict: Dict, available_agents: Optional[List[str]] = None
    ) -> Dict:
        gp = batch_dict.setdefault("gaussian_pipeline", {})
        gp["available_agents"] = available_agents or gp.get("available_agents", [])
        batch_dict = self.split_dual_path_features(batch_dict, available_agents)
        batch_dict = self.run_local_interaction(batch_dict, available_agents)
        return batch_dict
