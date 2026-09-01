from typing import Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentDepthHead(nn.Module):
    """Predict agent-specific depth logits from sampled view features."""

    def __init__(self, input_dim: int, hidden_dim: int, depth_num: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, depth_num),
        )

    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        """Return depth logits for one set of sampled view features."""
        return self.net(view_features)


class SharedDepthPredictor(nn.Module):
    """Shared predictor with agent-specific LSS depth settings."""

    AGENT_ORDER = ("vehicle", "rsu", "drone")

    def __init__(self, model_cfg=None):
        super().__init__()
        self.model_cfg = model_cfg or {}
        self.agent_cfg = self.build_agent_specific_configs()
        self.depth_heads = self.build_agent_specific_depth_heads()
        self._depth_bin_buffer_names = {}
        for agent_name, agent_cfg in self.agent_cfg.items():
            buffer_name = f"depth_bins_{agent_name}"
            self.register_buffer(
                buffer_name,
                self.init_depth_bins(agent_cfg),
                persistent=False,
            )
            self._depth_bin_buffer_names[agent_name] = buffer_name

    def _normalize_agent_cfg(self, cfg_update: Mapping) -> Dict[str, float]:
        """Normalize one agent's explicit depth config."""
        normalized_cfg: Dict[str, float] = {}
        dbound_cfg = cfg_update.get("DBOUND")
        if isinstance(dbound_cfg, (list, tuple)) and len(dbound_cfg) == 3:
            depth_start = float(dbound_cfg[0])
            depth_max = float(dbound_cfg[1])
            step = float(dbound_cfg[2])
            normalized_cfg.update(
                {
                    "depth_start": depth_start,
                    "depth_max": depth_max,
                    "depth_num": int(round((depth_max - depth_start) / step)),
                }
            )
        for key in ("depth_num", "depth_start", "depth_max", "depth_input_dim", "depth_hidden_dim"):
            if key in cfg_update:
                value = cfg_update[key]
                if key in ("depth_num", "depth_input_dim", "depth_hidden_dim"):
                    normalized_cfg[key] = int(value)
                else:
                    normalized_cfg[key] = float(value)
        required_keys = {
            "depth_num",
            "depth_start",
            "depth_max",
            "depth_input_dim",
            "depth_hidden_dim",
        }
        missing_keys = required_keys.difference(normalized_cfg.keys())
        if len(missing_keys) > 0:
            raise KeyError(
                "AGENT_DEPTH_CONFIG is missing required keys for one agent: "
                f"{sorted(missing_keys)}"
            )
        return normalized_cfg

    def build_agent_specific_configs(self) -> Dict[str, Dict[str, float]]:
        """Build one LSS-style depth config for each heterogeneous agent."""
        agent_depth_cfg = self.model_cfg
        if "AGENT_DEPTH_CONFIG" in agent_depth_cfg and isinstance(
            agent_depth_cfg.get("AGENT_DEPTH_CONFIG"), Mapping
        ):
            agent_depth_cfg = agent_depth_cfg.get("AGENT_DEPTH_CONFIG")
        if not isinstance(agent_depth_cfg, Mapping) or len(agent_depth_cfg) == 0:
            raise KeyError("SharedDepthPredictor requires AGENT_DEPTH_CONFIG.")

        agent_cfg: Dict[str, Dict[str, float]] = {}
        for agent_name in self.AGENT_ORDER:
            current_cfg = agent_depth_cfg.get(agent_name)
            if current_cfg is None:
                continue
            if not isinstance(current_cfg, Mapping):
                raise TypeError(
                    f"AGENT_DEPTH_CONFIG['{agent_name}'] must be a mapping, got {type(current_cfg)}."
                )
            agent_cfg[agent_name] = self._normalize_agent_cfg(current_cfg)
        if len(agent_cfg) == 0:
            raise KeyError("AGENT_DEPTH_CONFIG must contain at least one agent entry.")
        return agent_cfg

    def init_depth_bins(self, agent_cfg: Dict[str, float]) -> torch.Tensor:
        """Initialize GaussianLSS-style depth bin centers."""
        depth_num = int(agent_cfg["depth_num"])
        depth_start = float(agent_cfg["depth_start"])
        depth_max = float(agent_cfg["depth_max"])
        depth_range = depth_max - depth_start
        interval = depth_range / depth_num
        interval = interval * torch.ones((depth_num + 1), dtype=torch.float32)
        interval[0] = depth_start
        bin_edges = torch.cumsum(interval, dim=0)
        return 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def build_agent_specific_depth_heads(self) -> nn.ModuleDict:
        """Build one depth head for each heterogeneous agent type."""
        return nn.ModuleDict(
            {
                agent_name: AgentDepthHead(
                    int(agent_cfg["depth_input_dim"]),
                    int(agent_cfg["depth_hidden_dim"]),
                    int(agent_cfg["depth_num"]),
                )
                for agent_name, agent_cfg in self.agent_cfg.items()
            }
        )

    def get_agent_depth_bins(self, agent_name: str) -> torch.Tensor:
        """Return the depth bins that correspond to the given agent."""
        if agent_name not in self._depth_bin_buffer_names:
            raise KeyError(f"Unsupported agent type for depth bins: {agent_name}")
        return getattr(self, self._depth_bin_buffer_names[agent_name])

    def predict_agent_depth_logits(
        self, view_features: torch.Tensor, agent_name: str
    ) -> torch.Tensor:
        """Route sampled view features to the corresponding agent-specific depth head."""
        if agent_name not in self.depth_heads:
            raise KeyError(f"Unsupported agent type for depth prediction: {agent_name}")
        return self.depth_heads[agent_name](view_features)

    def predict_depth_distribution(
        self, depth_logits: torch.Tensor, agent_name: str
    ) -> Dict[str, torch.Tensor]:
        """Convert depth logits into GaussianLSS-style depth statistics."""
        depth_bins = self.get_agent_depth_bins(agent_name).to(depth_logits.device)
        depth_prob = F.softmax(depth_logits, dim=-1)
        bins = depth_bins.view(*([1] * (depth_prob.ndim - 1)), -1)
        soft_depth_mean = (depth_prob * bins).sum(dim=-1, keepdim=True)
        depth_variance = (depth_prob * (bins - soft_depth_mean).pow(2)).sum(
            dim=-1, keepdim=True
        )
        depth_entropy = -(depth_prob * torch.log(depth_prob.clamp_min(1e-8))).sum(
            dim=-1, keepdim=True
        )
        return {
            "depth_prob": depth_prob,
            "soft_depth_mean": soft_depth_mean,
            "depth_variance": depth_variance,
            "depth_entropy": depth_entropy,
        }

    def forward(self, view_features: torch.Tensor, agent_name: str) -> Dict[str, torch.Tensor]:
        """Predict depth logits and summarized statistics for one agent."""
        depth_logits = self.predict_agent_depth_logits(view_features, agent_name)
        depth_stats = self.predict_depth_distribution(depth_logits, agent_name)
        depth_stats["depth_logits"] = depth_logits
        return depth_stats
