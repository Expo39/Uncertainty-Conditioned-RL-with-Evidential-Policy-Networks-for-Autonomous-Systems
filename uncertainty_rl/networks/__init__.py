"""
@file __init__.py
@brief Evidential deep learning policy networks.
"""

from uncertainty_rl.networks.evidential_policy import (
    EvidentialLayer,
    EvidentialPolicyNetwork,
    UncertaintyConditionedActor,
)
from uncertainty_rl.networks.sb3_integration import (
    EvidentialActorCriticPolicy,
    EvidentialDistribution,
    EvidentialPPO,
    LayerNormActorCriticPolicy,
    ScheduledEntCoefPPO,
)

__all__ = [
    "EvidentialLayer",
    "EvidentialPolicyNetwork",
    "UncertaintyConditionedActor",
    "EvidentialActorCriticPolicy",
    "EvidentialDistribution",
    "EvidentialPPO",
    "LayerNormActorCriticPolicy",
    "ScheduledEntCoefPPO",
]
