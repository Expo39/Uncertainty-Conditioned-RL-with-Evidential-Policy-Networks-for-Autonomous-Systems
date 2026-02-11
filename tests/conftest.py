"""
@file conftest.py
@brief Shared pytest fixtures for the uncertainty_rl test suite.

Provides reusable fixtures for network instantiation, dummy states,
environment configuration, and logging setup.
"""
from typing import Dict, Any

import numpy as np
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
        "uncertainty_noise_std": 0.1,
        "max_steps": 50,
        "learning_rate": 3e-4,
        "buffer_size": 1000,
        "learning_starts": 100,
        "batch_size": 32,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "net_arch": [64, 64],
        "checkpoint_freq": 500,
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
        "noise_levels": [0.05, 0.1, 0.5],
        "n_episodes": 3,
    }
