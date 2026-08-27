from typing import Dict, List, Optional

import torch.nn as nn


class GlobalGaussianInteractionModule(nn.Module):
    """执行跨 agent 的全局交互与特征融合更新。"""

    def __init__(self, model_cfg=None):
        super().__init__()
        self.model_cfg = model_cfg or {}

    def run_global_interaction(
        self, batch_dict: Dict, available_agents: Optional[List[str]] = None
    ) -> Dict:
        """预留全局 self/cross attention 接口。"""
        gp = batch_dict.setdefault("gaussian_pipeline", {})
        gp.setdefault("global_interaction_features", None)
        return batch_dict

    def fuse_interaction_features(
        self, batch_dict: Dict, available_agents: Optional[List[str]] = None
    ) -> Dict:
        """预留交互后高斯更新与 render feature 接口。"""
        gp = batch_dict.setdefault("gaussian_pipeline", {})
        gp.setdefault("render_features", None)
        return batch_dict

    def forward(
        self, batch_dict: Dict, available_agents: Optional[List[str]] = None
    ) -> Dict:
        gp = batch_dict.setdefault("gaussian_pipeline", {})
        gp["available_agents"] = available_agents or gp.get("available_agents", [])
        batch_dict = self.run_global_interaction(batch_dict, available_agents)
        batch_dict = self.fuse_interaction_features(batch_dict, available_agents)
        return batch_dict
