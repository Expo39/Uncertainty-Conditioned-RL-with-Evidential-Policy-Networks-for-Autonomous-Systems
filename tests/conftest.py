"""
@file conftest.py
@brief Shared pytest fixtures for the uncertainty_rl test suite.

Provides reusable fixtures for network instantiation, dummy states,
environment configuration, and logging setup.
"""

from typing import Any, Dict

import pytest
import torch

# ---------------------------------------------------------------------------
# Constants for network tests. Network tests use a small arbitrary state dim
# (not the full 21-dim env obs) for fast unit test execution. Env obs space
# tests in test_carla_parking.py use _compute_obs_dim() directly.
# ---------------------------------------------------------------------------
STATE_DIM = 15
ACTION_DIM = 3
BATCH_SIZE = 8
HIDDEN_DIMS = [64, 64]  # Smaller than production for fast tests


# ---------------------------------------------------------------------------
# Tensor fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def state_batch() -> torch.Tensor:
    """
    @brief Random state batch shaped (BATCH_SIZE, STATE_DIM).
    """
    return torch.randn(BATCH_SIZE, STATE_DIM)


@pytest.fixture
def single_state() -> torch.Tensor:
    """
    @brief Single random state shaped (1, STATE_DIM).
    """
    return torch.randn(1, STATE_DIM)


@pytest.fixture
def action_batch() -> torch.Tensor:
    """
    @brief Random action batch shaped (BATCH_SIZE, ACTION_DIM).
    """
    return torch.randn(BATCH_SIZE, ACTION_DIM)


@pytest.fixture
def low_uncertainty_state() -> torch.Tensor:
    """
    @brief State with very low EKF localisation uncertainty (indices 6-8 near zero).
    """
    state = torch.randn(1, STATE_DIM)
    state[0, 6:9] = 0.01
    state[0, 9:15] = 0.0001
    return state


@pytest.fixture
def high_uncertainty_state() -> torch.Tensor:
    """
    @brief State with high EKF localisation uncertainty (indices 6-8 large).
    """
    state = torch.randn(1, STATE_DIM)
    state[0, 6:9] = 1.0
    state[0, 9:12] = 1.0
    state[0, 12:15] = 0.5
    return state


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def train_config() -> Dict[str, Any]:
    """
    @brief Minimal training configuration for tests.
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
        "carla_conditions": {
            "weather_presets": ["ClearNoon", "HardRainNoon"],
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
                "name": "clear_low_noise",
                "description": "Clear weather, low IMU noise, no traffic",
                "weather_preset": "ClearNoon",
                "imu_noise_multiplier": 0.5,
                "num_patrol_vehicles": 0,
                "num_pedestrians": 0,
            },
            {
                "name": "rain_degraded",
                "description": "Heavy rain, 2x IMU noise, moderate traffic",
                "weather_preset": "HardRainNoon",
                "imu_noise_multiplier": 2.0,
                "num_patrol_vehicles": 2,
                "num_pedestrians": 3,
            },
        ],
    }
