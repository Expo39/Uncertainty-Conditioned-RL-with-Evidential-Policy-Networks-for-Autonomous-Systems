"""
@file __init__.py
@brief Uncertainty-Conditioned RL with Evidential Policy Networks.

Package marker only - no eager imports of environment or network modules.
Those require gymnasium/torch which are not installed in the local .venv.
Import directly from sub-modules where needed:

    from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv
    from uncertainty_rl.networks.evidential_policy import EvidentialPolicyNetwork
"""

__version__ = "0.1.0"
__author__ = "Antonio Galdes"
