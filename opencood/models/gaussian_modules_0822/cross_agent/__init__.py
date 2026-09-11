"""Stage-2 cross-agent Gaussian interaction."""

from opencood.models.gaussian_modules_0822.cross_agent.adapter import (
    AgentResidualAdapter,
)
from opencood.models.gaussian_modules_0822.cross_agent.interaction import (
    INTERACTION_TYPES,
    CrossAgentInteraction,
    build_cross_agent_interactions,
    interaction_sources,
)

__all__ = [
    "AgentResidualAdapter",
    "CrossAgentInteraction",
    "INTERACTION_TYPES",
    "build_cross_agent_interactions",
    "interaction_sources",
]
