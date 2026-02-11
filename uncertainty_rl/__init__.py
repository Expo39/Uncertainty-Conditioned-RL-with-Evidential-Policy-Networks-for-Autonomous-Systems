"""Uncertainty-Conditioned RL with Evidential Policy Networks for Autonomous Vehicles.

This package implements uncertainty-aware reinforcement learning for autonomous parking
using CARLA simulator with SLAM localisation uncertainty propagation through 
evidential deep learning policies.
"""

__version__ = "0.1.0"
__author__ = "Uncertainty-Conditioned RL Team"

from uncertainty_rl.networks.evidential_policy import (
    EvidentialPolicyNetwork,
    EvidentialLayer,
    UncertaintyConditionedActor,
)
from uncertainty_rl.envs.carla_parking import CARLAParkingEnv

__all__ = [
    "EvidentialPolicyNetwork",
    "EvidentialLayer",
    "UncertaintyConditionedActor",
    "CARLAParkingEnv",
]
