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
# Constants matching the project's 15-dim state / 3-dim action convention
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
    @brief State with very low SLAM uncertainty (indices 6-8 near zero).
    """
    state = torch.randn(1, STATE_DIM)
    state[0, 6:9] = 0.01
    state[0, 9:15] = 0.0001
    return state


@pytest.fixture
def high_uncertainty_state() -> torch.Tensor:
    """
    @brief State with high SLAM uncertainty (indices 6-8 large).
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
        "town": "Town01",
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
            "covariance_topic": "/slam_uncertainty/covariance",
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
            "gnss": {
                "noise_alt_stddev": 0.5,
                "noise_lat_stddev": 0.00001,
                "noise_lon_stddev": 0.00001,
                "sensor_tick": 0.1,
            },
        },
        "carla_conditions": {
            "weather_presets": ["ClearNoon"],
            "fog_density_range": [0.0, 0.0],
            "fog_distance_range": [50.0, 50.0],
            "num_vehicles": 0,
            "num_pedestrians": 0,
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
        "town": "Town01",
        "max_steps": 50,
        "n_episodes": 3,
        "ros2": {
            "covariance_topic": "/slam_uncertainty/covariance",
            "covariance_timeout": 10.0,
        },
        "eval_conditions": [
            {
                "name": "clear_low_noise",
                "description": "Clear weather, low sensor noise",
                "weather_preset": "ClearNoon",
                "fog_density": 0.0,
                "imu_noise_multiplier": 0.5,
                "gnss_noise_multiplier": 0.5,
                "num_vehicles": 0,
                "num_pedestrians": 0,
            },
            {
                "name": "fog_moderate",
                "description": "Moderate fog, moderate noise",
                "weather_preset": "CloudyNoon",
                "fog_density": 50.0,
                "imu_noise_multiplier": 2.0,
                "gnss_noise_multiplier": 5.0,
                "num_vehicles": 10,
                "num_pedestrians": 5,
            },
        ],
    }
