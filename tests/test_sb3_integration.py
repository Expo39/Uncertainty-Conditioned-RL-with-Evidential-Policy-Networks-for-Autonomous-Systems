"""
@file test_sb3_integration.py
@brief Unit tests for SB3 evidential actor-critic integration.

Tests the EvidentialDistribution, EvidentialActorCriticPolicy, and
EvidentialPPO classes without requiring CARLA, ROS 2, or a GPU.
"""

import tempfile
from typing import Tuple

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

from uncertainty_rl.networks.sb3_integration import (
    EvidentialActorCriticPolicy,
    EvidentialDistribution,
    EvidentialPPO,
)
from uncertainty_rl.utils.constants import TOTAL_OBS_DIM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STATE_DIM = TOTAL_OBS_DIM
ACTION_DIM = 3
BATCH_SIZE = 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def obs_space() -> spaces.Box:
    """
    @brief 15-dim continuous observation space.
    """
    return spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(STATE_DIM,),
        dtype=np.float32,
    )


@pytest.fixture
def act_space() -> spaces.Box:
    """
    @brief 3-dim continuous action space matching parking env.
    """
    return spaces.Box(
        low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        dtype=np.float32,
    )


@pytest.fixture
def nig_params() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    @brief Valid NIG parameters for testing.
    """
    gamma = torch.randn(BATCH_SIZE, ACTION_DIM)
    nu = torch.abs(torch.randn(BATCH_SIZE, ACTION_DIM)) + 0.1
    alpha = torch.abs(torch.randn(BATCH_SIZE, ACTION_DIM)) + 1.5
    beta = torch.abs(torch.randn(BATCH_SIZE, ACTION_DIM)) + 0.1
    return gamma, nu, alpha, beta


@pytest.fixture
def evidential_dist() -> EvidentialDistribution:
    """
    @brief EvidentialDistribution instance.
    """
    return EvidentialDistribution(action_dim=ACTION_DIM)


@pytest.fixture
def policy(obs_space: spaces.Box, act_space: spaces.Box) -> EvidentialActorCriticPolicy:
    """
    @brief EvidentialActorCriticPolicy instance with small network.
    """
    return EvidentialActorCriticPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: 3e-4,
        net_arch=[64, 64],
        lambda_reg=0.01,
    )


# ===========================================================================
# TestEvidentialDistribution
# ===========================================================================


class TestEvidentialDistribution:
    """
    @class TestEvidentialDistribution
    @brief Tests for the SB3-compatible evidential distribution.
    """

    def test_proba_distribution_net_returns_module(
        self, evidential_dist: EvidentialDistribution
    ) -> None:
        """
        @brief proba_distribution_net returns an nn.Module.
        """
        module = evidential_dist.proba_distribution_net(latent_dim=64)
        assert isinstance(module, torch.nn.Module)

    def test_proba_distribution_sets_normal(
        self,
        evidential_dist: EvidentialDistribution,
        nig_params: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> None:
        """
        @brief proba_distribution creates a Normal distribution.
        """
        gamma, nu, alpha, beta = nig_params
        evidential_dist.proba_distribution(gamma, nu, alpha, beta)
        assert evidential_dist.distribution is not None

    def test_log_prob_shape(
        self,
        evidential_dist: EvidentialDistribution,
        nig_params: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> None:
        """
        @brief log_prob returns shape (batch_size,).
        """
        gamma, nu, alpha, beta = nig_params
        evidential_dist.proba_distribution(gamma, nu, alpha, beta)
        actions = torch.randn(BATCH_SIZE, ACTION_DIM)
        log_prob = evidential_dist.log_prob(actions)
        assert log_prob.shape == (BATCH_SIZE,)

    def test_entropy_shape(
        self,
        evidential_dist: EvidentialDistribution,
        nig_params: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> None:
        """
        @brief entropy returns shape (batch_size,).
        """
        gamma, nu, alpha, beta = nig_params
        evidential_dist.proba_distribution(gamma, nu, alpha, beta)
        entropy = evidential_dist.entropy()
        assert entropy is not None
        assert entropy.shape == (BATCH_SIZE,)

    def test_sample_shape(
        self,
        evidential_dist: EvidentialDistribution,
        nig_params: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> None:
        """
        @brief sample returns shape (batch_size, action_dim).
        """
        gamma, nu, alpha, beta = nig_params
        evidential_dist.proba_distribution(gamma, nu, alpha, beta)
        sample = evidential_dist.sample()
        assert sample.shape == (BATCH_SIZE, ACTION_DIM)

    def test_mode_equals_gamma(
        self,
        evidential_dist: EvidentialDistribution,
        nig_params: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> None:
        """
        @brief mode() returns gamma (the NIG mean).
        """
        gamma, nu, alpha, beta = nig_params
        evidential_dist.proba_distribution(gamma, nu, alpha, beta)
        mode = evidential_dist.mode()
        assert torch.allclose(mode, gamma)

    def test_nig_params_cached(
        self,
        evidential_dist: EvidentialDistribution,
        nig_params: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> None:
        """
        @brief nig_params property returns cached parameters.
        """
        gamma, nu, alpha, beta = nig_params
        evidential_dist.proba_distribution(gamma, nu, alpha, beta)
        cached = evidential_dist.nig_params
        assert torch.allclose(cached[0], gamma)
        assert torch.allclose(cached[1], nu)
        assert torch.allclose(cached[2], alpha)
        assert torch.allclose(cached[3], beta)

    def test_log_prob_is_finite(
        self,
        evidential_dist: EvidentialDistribution,
        nig_params: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> None:
        """
        @brief log_prob values are finite (no NaN or Inf).
        """
        gamma, nu, alpha, beta = nig_params
        evidential_dist.proba_distribution(gamma, nu, alpha, beta)
        actions = torch.randn(BATCH_SIZE, ACTION_DIM)
        log_prob = evidential_dist.log_prob(actions)
        assert torch.all(torch.isfinite(log_prob))

    def test_gradient_flows_through_sample(
        self,
        evidential_dist: EvidentialDistribution,
    ) -> None:
        """
        @brief Gradients flow through sample via rsample.
        """
        gamma = torch.randn(BATCH_SIZE, ACTION_DIM, requires_grad=True)
        nu = torch.abs(torch.randn(BATCH_SIZE, ACTION_DIM)) + 0.1
        alpha = torch.abs(torch.randn(BATCH_SIZE, ACTION_DIM)) + 1.5
        beta_leaf = torch.randn(BATCH_SIZE, ACTION_DIM, requires_grad=True)
        beta = torch.abs(beta_leaf) + 0.1
        evidential_dist.proba_distribution(gamma, nu, alpha, beta)
        sample = evidential_dist.sample()
        loss = sample.sum()
        loss.backward()
        assert gamma.grad is not None
        assert beta_leaf.grad is not None


# ===========================================================================
# TestEvidentialActorCriticPolicy
# ===========================================================================


class TestEvidentialActorCriticPolicy:
    """
    @class TestEvidentialActorCriticPolicy
    @brief Tests for the evidential actor-critic policy.
    """

    def test_instantiation(self, policy: EvidentialActorCriticPolicy) -> None:
        """
        @brief Policy instantiates without error.
        """
        assert isinstance(policy, EvidentialActorCriticPolicy)

    def test_forward_returns_correct_shapes(
        self, policy: EvidentialActorCriticPolicy
    ) -> None:
        """
        @brief forward() returns (actions, values, log_prob) with correct shapes.
        """
        obs = torch.randn(BATCH_SIZE, STATE_DIM)
        actions, values, log_prob = policy.forward(obs)
        assert actions.shape == (BATCH_SIZE, ACTION_DIM)
        assert values.shape == (BATCH_SIZE, 1)
        assert log_prob.shape == (BATCH_SIZE,)

    def test_evaluate_actions_returns_correct_shapes(
        self, policy: EvidentialActorCriticPolicy
    ) -> None:
        """
        @brief evaluate_actions() returns correct shapes.
        """
        obs = torch.randn(BATCH_SIZE, STATE_DIM)
        actions = torch.randn(BATCH_SIZE, ACTION_DIM)
        values, log_prob, entropy = policy.evaluate_actions(obs, actions)
        assert values.shape == (BATCH_SIZE, 1)
        assert log_prob.shape == (BATCH_SIZE,)
        assert entropy is not None
        assert entropy.shape == (BATCH_SIZE,)

    def test_evaluate_actions_caches_nig_params(
        self, policy: EvidentialActorCriticPolicy
    ) -> None:
        """
        @brief evaluate_actions() caches NIG params for evidential loss.
        """
        obs = torch.randn(BATCH_SIZE, STATE_DIM)
        actions = torch.randn(BATCH_SIZE, ACTION_DIM)
        policy.evaluate_actions(obs, actions)
        assert policy._cached_nig_params is not None
        gamma, nu, alpha, beta = policy._cached_nig_params
        assert gamma.shape == (BATCH_SIZE, ACTION_DIM)
        assert nu.shape == (BATCH_SIZE, ACTION_DIM)
        assert alpha.shape == (BATCH_SIZE, ACTION_DIM)
        assert beta.shape == (BATCH_SIZE, ACTION_DIM)

    def test_nig_constraints_maintained(
        self, policy: EvidentialActorCriticPolicy
    ) -> None:
        """
        @brief NIG constraints hold: nu > 0, alpha > 1, beta > 0.
        """
        obs = torch.randn(BATCH_SIZE, STATE_DIM)
        actions = torch.randn(BATCH_SIZE, ACTION_DIM)
        policy.evaluate_actions(obs, actions)
        assert policy._cached_nig_params is not None
        _, nu, alpha, beta = policy._cached_nig_params
        assert torch.all(nu > 0)
        assert torch.all(alpha > 1.0)
        assert torch.all(beta > 0)

    def test_get_action_with_uncertainty_interface(
        self, policy: EvidentialActorCriticPolicy
    ) -> None:
        """
        @brief get_action_with_uncertainty returns required dict keys.
        """
        obs = torch.randn(1, STATE_DIM)
        action, uncertainty_dict = policy.get_action_with_uncertainty(obs)
        assert action.shape == (1, ACTION_DIM)
        required_keys = {
            "epistemic",
            "aleatoric",
            "total",
            "gamma",
            "nu",
            "alpha",
            "beta",
        }
        assert required_keys == set(uncertainty_dict.keys())

    def test_predict_returns_numpy(self, policy: EvidentialActorCriticPolicy) -> None:
        """
        @brief predict() returns numpy array.
        """
        obs = np.random.randn(1, STATE_DIM).astype(np.float32)
        action, _ = policy.predict(obs, deterministic=True)
        assert isinstance(action, np.ndarray)
        assert action.shape == (1, ACTION_DIM)

    def test_gradient_flows_through_action_net(
        self, policy: EvidentialActorCriticPolicy
    ) -> None:
        """
        @brief Gradients reach the EvidentialLayer parameters.
        """
        obs = torch.randn(BATCH_SIZE, STATE_DIM)
        actions = torch.randn(BATCH_SIZE, ACTION_DIM)
        values, log_prob, _ = policy.evaluate_actions(obs, actions)
        loss = log_prob.mean() + values.mean()
        loss.backward()

        # Check EvidentialLayer (action_net) has gradients
        for param in policy.action_net.parameters():
            assert param.grad is not None

    def test_value_net_independent_of_evidential(
        self, policy: EvidentialActorCriticPolicy
    ) -> None:
        """
        @brief Value predictions use the standard critic, not evidential.
        """
        assert isinstance(policy.value_net, torch.nn.Linear)
        # Critic output dim should be 1
        assert policy.value_net.out_features == 1

    def test_save_load_roundtrip(
        self,
        policy: EvidentialActorCriticPolicy,
        obs_space: spaces.Box,
        act_space: spaces.Box,
    ) -> None:
        """
        @brief Save and load preserves lambda_reg.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/test_policy"
            policy.save(path)
            loaded = EvidentialActorCriticPolicy.load(path)
            assert loaded.lambda_reg == policy.lambda_reg


# ===========================================================================
# TestEvidentialPPO
# ===========================================================================


class TestEvidentialPPO:
    """
    @class TestEvidentialPPO
    @brief Tests for the EvidentialPPO training algorithm.
    """

    @pytest.fixture
    def dummy_env(self) -> gym.Env:
        """
        @brief Simple continuous-action environment for testing.
        """
        return gym.make("Pendulum-v1")

    def test_instantiation(self, dummy_env: gym.Env) -> None:
        """
        @brief EvidentialPPO instantiates without error.
        """
        model = EvidentialPPO(
            policy=EvidentialActorCriticPolicy,
            env=dummy_env,
            lambda_reg=0.01,
            n_steps=64,
            batch_size=32,
        )
        assert isinstance(model, EvidentialPPO)

    def test_lambda_reg_passed_to_policy(self, dummy_env: gym.Env) -> None:
        """
        @brief lambda_reg propagated from EvidentialPPO to policy.
        """
        model = EvidentialPPO(
            policy=EvidentialActorCriticPolicy,
            env=dummy_env,
            lambda_reg=0.05,
            n_steps=64,
            batch_size=32,
        )
        assert model.policy.lambda_reg == 0.05

    def test_train_step_runs(self, dummy_env: gym.Env) -> None:
        """
        @brief Short training run completes without error.
        """
        model = EvidentialPPO(
            policy=EvidentialActorCriticPolicy,
            env=dummy_env,
            lambda_reg=0.01,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
        )
        model.learn(total_timesteps=128)

    def test_evidential_metrics_logged(self, dummy_env: gym.Env) -> None:
        """
        @brief Evidential metrics appear in the logger after training.
        """
        model = EvidentialPPO(
            policy=EvidentialActorCriticPolicy,
            env=dummy_env,
            lambda_reg=0.01,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
        )
        model.learn(total_timesteps=128)

        # Check that the evidential keys were recorded
        name_to_value = model.logger.name_to_value
        assert "train/evidential_reg_loss" in name_to_value
        assert "train/epistemic_uncertainty" in name_to_value
        assert "train/aleatoric_uncertainty" in name_to_value
