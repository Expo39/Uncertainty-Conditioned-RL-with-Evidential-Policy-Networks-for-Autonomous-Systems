"""
@file conftest.py
@brief Shared pytest fixtures for the uncertainty_rl test suite.

Provides reusable fixtures for network instantiation, dummy states,
environment configuration, and logging setup.
"""

from typing import Any, Dict

import pytest

try:
    import torch
except ImportError:
    # Network fixtures are unavailable without torch; such tests should use
    # pytest.importorskip.
    torch = None  # type: ignore[assignment]

# Network tests use a small arbitrary state dim (not the full 13-dim env obs)
# for fast unit test execution. Env obs space tests in test_carla_parking.py
# use _compute_obs_dim() directly.
STATE_DIM = 15
BATCH_SIZE = 8
HIDDEN_DIMS = [64, 64]  # Smaller than production for fast tests


@pytest.fixture
def state_batch():
    """
    @brief Random state batch shaped (BATCH_SIZE, STATE_DIM).
    """
    pytest.importorskip("torch")
    return torch.randn(BATCH_SIZE, STATE_DIM)


@pytest.fixture
def single_state():
    """
    @brief Single random state shaped (1, STATE_DIM).
    """
    pytest.importorskip("torch")
    return torch.randn(1, STATE_DIM)


@pytest.fixture
def train_config() -> Dict[str, Any]:
    """
    @brief Minimal merged configuration for tests (train_config + env_config combined).

    Represents the merged dict that train() and make_env() receive after
    merge_configs(train_config, env_config) is called in main(). Tests
    that need only training keys or only env keys can read from this dict
    as both key sets are present.
    """
    return {
        "carla_host": "localhost",
        "carla_port": 2000,
        "town": "FlatPlane",
        "max_steps": 50,
        "learning_rate": 3e-4,
        "n_steps": 128,
        "batch_size": 32,
        "n_epochs": 4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "net_arch": [64, 64],
        "checkpoint_freq": 500,
        "ros2": {
            "covariance_topic": "/ekf_uncertainty/covariance",
            "covariance_timeout": 10.0,
        },
        "carla_sensors": {
            "imu": {
                "noise_accel_stddev_x": 0.1,
                "noise_accel_stddev_y": 0.1,
                "noise_accel_stddev_z": 0.1,
                "noise_gyro_stddev_x": 0.01,
                "noise_gyro_stddev_y": 0.01,
                "noise_gyro_stddev_z": 0.01,
                "sensor_tick": 0.05,
            },
        },
        "parking_scenarios": {
            "bay_occupancy_min": 0.3,
            "bay_occupancy_max": 0.8,
            "num_patrol_vehicles_max": 1,
            "pedestrian_spawn_probability": 0.8,
        },
    }


@pytest.fixture
def eval_config() -> Dict[str, Any]:
    """
    @brief Minimal evaluation configuration for tests.
    """
    return {
        "carla_host": "localhost",
        "carla_port": 2000,
        "town": "FlatPlane",
        "max_steps": 50,
        "n_episodes": 3,
        "ros2": {
            "covariance_topic": "/ekf_uncertainty/covariance",
            "covariance_timeout": 10.0,
        },
        "eval_conditions": [
            {
                "name": "nominal_empty",
                "description": "RTK fixed, empty lot",
                "held_gnss_tier": "rtk_fixed",
                "imu_noise_multiplier": 1.0,
                "num_patrol_vehicles": 0,
                "pedestrian_spawn_probability": 0.0,
                "bay_occupancy_rate": 0.6,
            },
            {
                "name": "rtk_float",
                "description": "RTK float, moderate traffic",
                "held_gnss_tier": "rtk_float",
                "imu_noise_multiplier": 1.5,
                "num_patrol_vehicles": 1,
                "pedestrian_spawn_probability": 1.0,
                "bay_occupancy_rate": 0.6,
            },
        ],
    }
