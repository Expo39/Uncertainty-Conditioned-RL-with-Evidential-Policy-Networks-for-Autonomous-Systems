"""
@file test_sb3_integration.py
@brief Unit tests for SB3 evidential actor-critic integration.

Tests the EvidentialDistribution, EvidentialActorCriticPolicy, and
EvidentialPPO classes without requiring CARLA, ROS 2, or a GPU.
"""

import tempfile
from typing import Tuple

import numpy as np
import pytest

# Skip entire module if torch is not available (CI without training deps)
torch = pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")

from gymnasium import spaces  # noqa: E402

from uncertainty_rl.networks.evidential_policy import (  # noqa: E402
    UncertaintyConditionedActor,
)
from uncertainty_rl.networks.sb3_integration import (  # noqa: E402
    EvidentialActorCriticPolicy,
    EvidentialDistribution,
    EvidentialPPO,
    LayerNormActorCriticPolicy,
    ScheduledEntCoefPPO,
)
from uncertainty_rl.training.train_ppo import linear_schedule  # noqa: E402
from uncertainty_rl.utils.constants import ACTION_DIM, TOTAL_OBS_DIM  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STATE_DIM = TOTAL_OBS_DIM
BATCH_SIZE = 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def obs_space() -> spaces.Box:
    """
    @brief Full observation space (TOTAL_OBS_DIM, both ablation flags on).
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
    @brief 3-dim continuous action space matching parking env. All axes are
           uniformly [-1, 1]: the env applies a tanh squash in the policy and
           remaps throttle / brake to [0, 1] inside step().
    """
    return spaces.Box(
        low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
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
        @brief log_prob returns shape (batch_size,). Actions must be in (-1, 1).
        """
        gamma, nu, alpha, beta = nig_params
        evidential_dist.proba_distribution(gamma, nu, alpha, beta)
        actions = evidential_dist.sample()
        log_prob = evidential_dist.log_prob(actions)
        assert log_prob.shape == (BATCH_SIZE,)

    def test_entropy_none_before_distribution_set(
        self, evidential_dist: EvidentialDistribution
    ) -> None:
        """
        @brief entropy() returns None until proba_distribution() is called.
        """
        assert evidential_dist.entropy() is None

    def test_entropy_finite_and_correct_shape(
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
        @brief entropy() returns a finite per-sample tensor of shape (batch_size,).
        """
        gamma, nu, alpha, beta = nig_params
        evidential_dist.proba_distribution(gamma, nu, alpha, beta)
        entropy = evidential_dist.entropy()
        assert entropy is not None
        assert entropy.shape == (BATCH_SIZE,)
        assert torch.all(torch.isfinite(entropy))

    def test_entropy_increases_with_std(self) -> None:
        """
        @brief A wider predictive (larger aleatoric) yields higher entropy.

        Holds nu/alpha fixed and raises beta (so aleatoric = beta/(alpha-1) and the
        sampling std both grow); the closed-form Gaussian entropy must increase.
        """
        gamma = torch.zeros(1, ACTION_DIM)
        nu = torch.full((1, ACTION_DIM), 1.0)
        alpha = torch.full((1, ACTION_DIM), 2.0)
        dist = EvidentialDistribution(action_dim=ACTION_DIM)

        narrow = dist.proba_distribution(
            gamma, nu, alpha, torch.full((1, ACTION_DIM), 0.05)
        ).entropy()
        wide = dist.proba_distribution(
            gamma, nu, alpha, torch.full((1, ACTION_DIM), 0.5)
        ).entropy()
        assert narrow is not None and wide is not None
        assert float(wide.item()) > float(narrow.item())

    def test_sample_shape_and_squashed(
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
        @brief sample returns shape (batch_size, action_dim) and values in (-1, 1).
        """
        gamma, nu, alpha, beta = nig_params
        evidential_dist.proba_distribution(gamma, nu, alpha, beta)
        sample = evidential_dist.sample()
        assert sample.shape == (BATCH_SIZE, ACTION_DIM)
        assert torch.all(sample > -1.0)
        assert torch.all(sample < 1.0)

    def test_aleatoric_floor_clamps_sampling_std(self) -> None:
        """
        @brief A collapsing NIG (beta -> 0) cannot drive the sampling std below
               sqrt(aleatoric_floor).

        Feeds near-zero beta (aleatoric = beta/(alpha-1) -> ~0) with a non-trivial
        floor and asserts the resulting Normal std equals sqrt(floor), not ~0.
        """
        floor = 0.04
        dist = EvidentialDistribution(action_dim=ACTION_DIM, aleatoric_floor=floor)
        gamma = torch.zeros(BATCH_SIZE, ACTION_DIM)
        nu = torch.full((BATCH_SIZE, ACTION_DIM), 1.0)
        alpha = torch.full((BATCH_SIZE, ACTION_DIM), 2.0)
        beta = torch.full((BATCH_SIZE, ACTION_DIM), 1e-8)  # collapsed evidence
        dist.proba_distribution(gamma, nu, alpha, beta)
        std = dist.distribution.stddev
        assert torch.allclose(
            std, torch.full_like(std, float(np.sqrt(floor))), atol=1e-6
        )

    def test_default_floor_allows_small_std(self) -> None:
        """
        @brief With the default (~1e-6) floor, a collapsing NIG yields a near-zero
               std, confirming the floor is what gates collapse, not a side effect.
        """
        dist = EvidentialDistribution(action_dim=ACTION_DIM)  # default floor 1e-6
        gamma = torch.zeros(BATCH_SIZE, ACTION_DIM)
        nu = torch.full((BATCH_SIZE, ACTION_DIM), 1.0)
        alpha = torch.full((BATCH_SIZE, ACTION_DIM), 2.0)
        beta = torch.full((BATCH_SIZE, ACTION_DIM), 1e-8)
        dist.proba_distribution(gamma, nu, alpha, beta)
        assert torch.all(dist.distribution.stddev < 0.01)

    def test_mode_equals_tanh_gamma(
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
        @brief mode() returns tanh(gamma) - squashed deterministic action.
        """
        gamma, nu, alpha, beta = nig_params
        evidential_dist.proba_distribution(gamma, nu, alpha, beta)
        mode = evidential_dist.mode()
        assert torch.allclose(mode, torch.tanh(gamma))

    def test_log_prob_includes_tanh_jacobian(
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
        @brief log_prob applies the tanh change-of-variables correction.
        """
        from torch.distributions import Normal

        gamma, nu, alpha, beta = nig_params
        evidential_dist.proba_distribution(gamma, nu, alpha, beta)
        gaussian_actions = evidential_dist.sample()
        # Reuse the cached pre-squash sample to avoid the atanh fallback.
        pre_squash = evidential_dist._gaussian_actions
        assert pre_squash is not None
        log_prob = evidential_dist.log_prob(
            gaussian_actions, gaussian_actions=pre_squash
        )

        # Manual reference: Normal log_prob minus the Jacobian term.
        std = torch.sqrt(torch.clamp(beta / (alpha - 1), min=1e-6, max=1.0))
        normal_lp = Normal(gamma, std).log_prob(pre_squash).sum(dim=-1)
        jacobian = torch.log(1.0 - torch.tanh(pre_squash) ** 2 + 1e-6).sum(dim=-1)
        expected = normal_lp - jacobian
        assert torch.allclose(log_prob, expected, atol=1e-5)

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
        actions = evidential_dist.sample()
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
        @brief evaluate_actions() returns correct shapes. Entropy is the closed-form
               pre-squash Gaussian entropy, shape (batch_size,).
        """
        obs = torch.randn(BATCH_SIZE, STATE_DIM)
        # Squashed actions in (-1, 1); torch.tanh of randn keeps them off the boundary.
        actions = torch.tanh(torch.randn(BATCH_SIZE, ACTION_DIM))
        values, log_prob, entropy = policy.evaluate_actions(obs, actions)
        assert values.shape == (BATCH_SIZE, 1)
        assert log_prob.shape == (BATCH_SIZE,)
        assert entropy is not None
        assert entropy.shape == (BATCH_SIZE,)
        assert torch.all(torch.isfinite(entropy))

    def test_evaluate_actions_caches_nig_params(
        self, policy: EvidentialActorCriticPolicy
    ) -> None:
        """
        @brief evaluate_actions() caches NIG params for evidential loss.
        """
        obs = torch.randn(BATCH_SIZE, STATE_DIM)
        actions = torch.tanh(torch.randn(BATCH_SIZE, ACTION_DIM))
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
        actions = torch.tanh(torch.randn(BATCH_SIZE, ACTION_DIM))
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
        actions = torch.tanh(torch.randn(BATCH_SIZE, ACTION_DIM))
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

    def test_save_load_preserves_aleatoric_floor(
        self, obs_space: spaces.Box, act_space: spaces.Box
    ) -> None:
        """
        @brief aleatoric_floor round-trips through save/load (so a resumed stage
               keeps the same exploration floor) and reaches the distribution.
        """
        floor = 0.04
        policy = EvidentialActorCriticPolicy(
            observation_space=obs_space,
            action_space=act_space,
            lr_schedule=lambda _: 3e-4,
            net_arch=[64, 64],
            aleatoric_floor=floor,
        )
        assert policy.action_dist.aleatoric_floor == floor
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/test_policy"
            policy.save(path)
            loaded = EvidentialActorCriticPolicy.load(path)
            assert loaded.aleatoric_floor == floor
            assert loaded.action_dist.aleatoric_floor == floor


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

    def test_aleatoric_floor_passed_to_policy(self, dummy_env: gym.Env) -> None:
        """
        @brief aleatoric_floor propagates from EvidentialPPO to the policy's
               distribution.
        """
        model = EvidentialPPO(
            policy=EvidentialActorCriticPolicy,
            env=dummy_env,
            aleatoric_floor=0.04,
            n_steps=64,
            batch_size=32,
        )
        assert model.policy.aleatoric_floor == 0.04
        assert model.policy.action_dist.aleatoric_floor == 0.04

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

    def test_lambda_reg_logged(self, dummy_env: gym.Env) -> None:
        """
        @brief train/lambda_reg is logged after training steps.
        """
        model = EvidentialPPO(
            policy=EvidentialActorCriticPolicy,
            env=dummy_env,
            lambda_reg=0.01,
            lambda_reg_warmup_steps=50000,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
        )
        model.learn(total_timesteps=128)
        assert "train/lambda_reg" in model.logger.name_to_value

    def test_lambda_reg_annealing_starts_near_zero(self, dummy_env: gym.Env) -> None:
        """
        @brief At step 0, effective lambda_reg should be 0.0 (ramp = 0/warmup = 0).
        @note num_timesteps is 0 before any learning, so ramp = min(1, 0/50000) = 0.
        """
        model = EvidentialPPO(
            policy=EvidentialActorCriticPolicy,
            env=dummy_env,
            lambda_reg=0.5,
            lambda_reg_warmup_steps=50000,
            n_steps=64,
            batch_size=32,
        )
        # Before any learning, num_timesteps == 0
        ramp = min(
            1.0,
            float(model.num_timesteps) / float(model.lambda_reg_warmup_steps),
        )
        effective_lambda = model.lambda_reg * ramp
        assert effective_lambda == 0.0

    def test_lambda_reg_annealing_reaches_full_after_warmup(
        self, dummy_env: gym.Env
    ) -> None:
        """
        @brief After warmup_steps, effective lambda equals lambda_reg.
        """
        warmup = 100
        model = EvidentialPPO(
            policy=EvidentialActorCriticPolicy,
            env=dummy_env,
            lambda_reg=0.5,
            lambda_reg_warmup_steps=warmup,
            n_steps=64,
            batch_size=32,
        )
        # Simulate being past warmup
        model.num_timesteps = warmup + 1
        ramp = min(
            1.0,
            float(model.num_timesteps) / float(model.lambda_reg_warmup_steps),
        )
        effective_lambda = model.lambda_reg * ramp
        assert effective_lambda == model.lambda_reg

    def test_evidential_reg_loss_bounded(self, dummy_env: gym.Env) -> None:
        """
        @brief Prior-anchoring evidential reg loss stays bounded even with large NIG params.

        Tests that the new log-penalty regulariser (replacing Amini et al. supervised term)
        produces bounded values and stays finite across training steps.
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

        # Check that evidential_reg_loss is finite and not explosively large
        reg_loss = model.logger.name_to_value.get("train/evidential_reg_loss")
        assert reg_loss is not None, "evidential_reg_loss not logged"
        assert torch.isfinite(
            torch.tensor(reg_loss)
        ).item(), f"evidential_reg_loss is not finite: {reg_loss}"
        # Prior-anchoring should produce values typically < 50 even with clamped params
        assert (
            reg_loss < 50.0
        ), f"evidential_reg_loss unexpectedly large: {reg_loss} (expected < 50)"


# ===========================================================================
# TestScheduledEntCoefPPO
# ===========================================================================


class TestScheduledEntCoefPPO:
    """
    @class TestScheduledEntCoefPPO
    @brief Tests the standard-PPO subclass that accepts a callable ent_coef.

    This is the path the vanilla / input-uncertainty baselines take. Stock PPO
    cannot multiply a callable ent_coef by the loss tensor; this subclass must
    resolve the schedule per train() call and leave it callable afterwards.
    """

    @pytest.fixture
    def dummy_env(self) -> gym.Env:
        """
        @brief Simple continuous-action environment for testing.
        """
        return gym.make("Pendulum-v1")

    def test_train_runs_with_callable_ent_coef(self, dummy_env: gym.Env) -> None:
        """
        @brief A short run with a scheduled ent_coef completes without the
               'function * Tensor' TypeError that bare PPO raises.
        """
        model = ScheduledEntCoefPPO(
            policy="MlpPolicy",
            env=dummy_env,
            ent_coef=linear_schedule(0.02, 0.006),
            n_steps=64,
            batch_size=32,
            n_epochs=2,
        )
        model.learn(total_timesteps=128)
        # ent_coef must remain the callable after training so the schedule keeps
        # advancing on subsequent updates.
        assert callable(model.ent_coef)

    def test_constant_ent_coef_still_works(self, dummy_env: gym.Env) -> None:
        """
        @brief A plain float ent_coef is passed straight through (no regression
               for the non-scheduled case).
        """
        model = ScheduledEntCoefPPO(
            policy="MlpPolicy",
            env=dummy_env,
            ent_coef=0.01,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
        )
        model.learn(total_timesteps=128)
        assert model.ent_coef == 0.01

    def test_bare_ppo_rejects_callable_ent_coef(self, dummy_env: gym.Env) -> None:
        """
        @brief Regression guard documenting WHY the subclass exists: stock PPO
               raises on a callable ent_coef, which is what crashed the vanilla
               curriculum run.
        """
        from stable_baselines3.ppo import PPO

        model = PPO(
            policy="MlpPolicy",
            env=dummy_env,
            ent_coef=linear_schedule(0.02, 0.006),
            n_steps=64,
            batch_size=32,
            n_epochs=2,
        )
        with pytest.raises(TypeError):
            model.learn(total_timesteps=128)


# ===========================================================================
# TestLayerNormActorCriticPolicy
# ===========================================================================


class TestLayerNormActorCriticPolicy:
    """
    @class TestLayerNormActorCriticPolicy
    @brief The standard-head policy must match the evidential backbone + prior.

    The 2x2 ablation needs the standard and evidential baselines identical except
    for the head and the observation. These tests pin the two non-head matches:
    the LayerNorm-equipped MLP extractor, and the default-forward action-mean bias
    (steer 0, throttle +0.5, brake -1.0) that the evidential head already sets.
    """

    def _policy(
        self, obs_space: spaces.Box, act_space: spaces.Box
    ) -> LayerNormActorCriticPolicy:
        return LayerNormActorCriticPolicy(
            observation_space=obs_space,
            action_space=act_space,
            lr_schedule=lambda _: 3e-4,
            net_arch=[64, 64],
        )

    def test_mlp_extractor_has_layernorm(
        self, obs_space: spaces.Box, act_space: spaces.Box
    ) -> None:
        """
        @brief The policy/value MLPs must contain LayerNorm, matching the
               evidential policy backbone (stock MlpPolicy has none).
        """
        from torch import nn

        pol = self._policy(obs_space, act_space)
        has_ln = any(isinstance(m, nn.LayerNorm) for m in pol.mlp_extractor.policy_net)
        assert has_ln, "policy_net is missing LayerNorm"

    def test_action_mean_bias_matches_evidential_prior(
        self, obs_space: spaces.Box, act_space: spaces.Box
    ) -> None:
        """
        @brief The Gaussian action-mean bias must be the same default-forward prior
               the evidential head uses (steer 0, throttle +0.5, brake -1.0).
        """
        pol = self._policy(obs_space, act_space)
        bias = pol.action_net.bias.detach()
        assert float(bias[0]) == pytest.approx(0.0)
        assert float(bias[1]) == pytest.approx(0.5)
        assert float(bias[2]) == pytest.approx(-1.0)

    def test_trains_with_scheduled_ent_coef(
        self, obs_space: spaces.Box, act_space: spaces.Box
    ) -> None:
        """
        @brief End-to-end: the policy runs under ScheduledEntCoefPPO on a 3-action
               env without error (the real vanilla/input-uncertainty path).
        """
        env = gym.make("Pendulum-v1")  # 1-action smoke env; bias branch is no-op
        model = ScheduledEntCoefPPO(
            policy=LayerNormActorCriticPolicy,
            env=env,
            ent_coef=linear_schedule(0.02, 0.006),
            n_steps=64,
            batch_size=32,
            n_epochs=2,
        )
        model.learn(total_timesteps=128)


# ===========================================================================
# TestNIGInit
# ===========================================================================


class TestNIGInit:
    """
    @class TestNIGInit
    @brief Tests for NIG hyperprior initialisation in EvidentialActorCriticPolicy.
    """

    def test_nig_init_nu_and_alpha_values(
        self,
        obs_space: spaces.Box,
        act_space: spaces.Box,
    ) -> None:
        """
        @brief After build, nu bias approx 0.31 (sub-1) and alpha bias approx 2.24.

        With weight scaled by 0.01 and zero input the raw output equals the bias.
        softplus(-1.0) + 1e-6 ~ 0.3133 (nu): a sub-1 prior keeps the no-evidence
        regime epistemic-dominant, since epistemic = aleatoric / nu.
        softplus(0.9) + 1.0 ~ 1.2411 + 1.0 ~ 2.2411 (alpha).
        """
        import torch.nn.functional as F

        policy = EvidentialActorCriticPolicy(
            observation_space=obs_space,
            action_space=act_space,
            lr_schedule=lambda _: 3e-4,
            net_arch=[64, 64],
            lambda_reg=0.01,
        )

        n = act_space.shape[0]  # 3
        with torch.no_grad():
            # Pass zero input so output ≈ bias (weight scaled by 0.01 -> ~0)
            latent_dim = policy.action_net.linear.in_features
            raw = policy.action_net.linear(torch.zeros(1, latent_dim))
            # raw shape: (1, 4*n) = (1, 12)
            nu_raw = raw[0, 1 * n : 2 * n]
            alpha_raw = raw[0, 2 * n : 3 * n]

            nu_activated = F.softplus(nu_raw) + 1e-6
            alpha_activated = F.softplus(alpha_raw) + 1.0

        # nu ~ softplus(-1.0) + 1e-6 ~ 0.3133. Must be < 1 so epistemic =
        # aleatoric / nu starts epistemic-dominant before evidence accrues.
        assert torch.all(nu_activated > 0.25), f"nu too small: {nu_activated}"
        assert torch.all(nu_activated < 1.0), f"nu not sub-1: {nu_activated}"

        # alpha ~ softplus(0.9) + 1.0 ~ 2.2411
        assert torch.all(alpha_activated > 2.1), f"alpha too small: {alpha_activated}"
        assert torch.all(alpha_activated < 2.4), f"alpha too large: {alpha_activated}"

    def test_layernorm_in_mlp_extractor(
        self,
        obs_space: spaces.Box,
        act_space: spaces.Box,
    ) -> None:
        """
        @brief _build_mlp_extractor inserts LayerNorm after each Linear.
        """
        policy = EvidentialActorCriticPolicy(
            observation_space=obs_space,
            action_space=act_space,
            lr_schedule=lambda _: 3e-4,
            net_arch=[64, 64],
            lambda_reg=0.01,
        )

        # policy_net and value_net should each contain at least one LayerNorm
        policy_ln_count = sum(
            1
            for m in policy.mlp_extractor.policy_net
            if isinstance(m, torch.nn.LayerNorm)
        )
        value_ln_count = sum(
            1
            for m in policy.mlp_extractor.value_net
            if isinstance(m, torch.nn.LayerNorm)
        )
        assert policy_ln_count > 0, "No LayerNorm found in policy_net"
        assert value_ln_count > 0, "No LayerNorm found in value_net"


# ===========================================================================
# TestUncertaintyConditionedActorWiring
# ===========================================================================


class TestUncertaintyConditionedActorWiring:
    """
    @class TestUncertaintyConditionedActorWiring
    @brief Tests for the dual-encoder vs flat MLP pathway in
           EvidentialActorCriticPolicy.

    Verifies that use_uncertainty_conditioning=True wires UncertaintyConditionedActor
    as the action_net and routes observations correctly, while False keeps the
    flat EvidentialLayer pathway unchanged.
    """

    @pytest.fixture
    def obs_space(self) -> spaces.Box:
        """
        @brief 13-dim observation space (full obs with covariance + obstacle dims).
        """
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(TOTAL_OBS_DIM,),
            dtype=np.float32,
        )

    @pytest.fixture
    def act_space(self) -> spaces.Box:
        """
        @brief 3-dim action space [steering, throttle, brake].
        """
        return spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def test_flat_mlp_path_uses_evidential_layer(
        self, obs_space: spaces.Box, act_space: spaces.Box
    ) -> None:
        """
        @brief use_uncertainty_conditioning=False -> action_net is EvidentialLayer.
        """
        from uncertainty_rl.networks.evidential_policy import EvidentialLayer

        policy = EvidentialActorCriticPolicy(
            observation_space=obs_space,
            action_space=act_space,
            lr_schedule=lambda _: 3e-4,
            net_arch=[64, 64],
            lambda_reg=0.01,
            use_uncertainty_conditioning=False,
        )
        assert isinstance(policy.action_net, EvidentialLayer)

    def test_dual_encoder_path_uses_uncertainty_conditioned_actor(
        self, obs_space: spaces.Box, act_space: spaces.Box
    ) -> None:
        """
        @brief use_uncertainty_conditioning=True wires UncertaintyConditionedActor.
        @note This is the dual-encoder pathway.
        """
        policy = EvidentialActorCriticPolicy(
            observation_space=obs_space,
            action_space=act_space,
            lr_schedule=lambda _: 3e-4,
            net_arch=[64, 64],
            lambda_reg=0.01,
            use_uncertainty_conditioning=True,
        )
        assert isinstance(policy.action_net, UncertaintyConditionedActor)

    def test_dual_encoder_forward_returns_correct_shapes(
        self, obs_space: spaces.Box, act_space: spaces.Box
    ) -> None:
        """
        @brief Dual-encoder forward pass returns (actions, values, log_prob)
               with correct shapes.
        """
        policy = EvidentialActorCriticPolicy(
            observation_space=obs_space,
            action_space=act_space,
            lr_schedule=lambda _: 3e-4,
            net_arch=[64, 64],
            lambda_reg=0.01,
            use_uncertainty_conditioning=True,
        )
        obs = torch.randn(BATCH_SIZE, TOTAL_OBS_DIM)
        actions, values, log_prob = policy.forward(obs)
        assert actions.shape == (BATCH_SIZE, ACTION_DIM)
        assert values.shape == (BATCH_SIZE, 1)
        assert log_prob.shape == (BATCH_SIZE,)

    def test_dual_encoder_evaluate_actions_caches_nig_params(
        self, obs_space: spaces.Box, act_space: spaces.Box
    ) -> None:
        """
        @brief evaluate_actions caches NIG params when using dual-encoder.
        """
        policy = EvidentialActorCriticPolicy(
            observation_space=obs_space,
            action_space=act_space,
            lr_schedule=lambda _: 3e-4,
            net_arch=[64, 64],
            lambda_reg=0.01,
            use_uncertainty_conditioning=True,
        )
        obs = torch.randn(BATCH_SIZE, TOTAL_OBS_DIM)
        actions = torch.randn(BATCH_SIZE, ACTION_DIM)
        policy.evaluate_actions(obs, actions)
        assert policy._cached_nig_params is not None
        gamma, nu, alpha, beta = policy._cached_nig_params
        assert gamma.shape == (BATCH_SIZE, ACTION_DIM)

    def test_dual_encoder_nig_constraints_hold(
        self, obs_space: spaces.Box, act_space: spaces.Box
    ) -> None:
        """
        @brief NIG constraints (nu > 0, alpha > 1, beta > 0) hold in dual-encoder mode.
        """
        policy = EvidentialActorCriticPolicy(
            observation_space=obs_space,
            action_space=act_space,
            lr_schedule=lambda _: 3e-4,
            net_arch=[64, 64],
            lambda_reg=0.01,
            use_uncertainty_conditioning=True,
        )
        obs = torch.randn(BATCH_SIZE, TOTAL_OBS_DIM)
        actions = torch.randn(BATCH_SIZE, ACTION_DIM)
        policy.evaluate_actions(obs, actions)
        assert policy._cached_nig_params is not None
        _, nu, alpha, beta = policy._cached_nig_params
        assert torch.all(nu > 0)
        assert torch.all(alpha > 1.0)
        assert torch.all(beta > 0)

    def test_dual_encoder_get_action_with_uncertainty(
        self, obs_space: spaces.Box, act_space: spaces.Box
    ) -> None:
        """
        @brief get_action_with_uncertainty returns correct keys in dual-encoder mode.
        """
        policy = EvidentialActorCriticPolicy(
            observation_space=obs_space,
            action_space=act_space,
            lr_schedule=lambda _: 3e-4,
            net_arch=[64, 64],
            lambda_reg=0.01,
            use_uncertainty_conditioning=True,
        )
        obs = torch.randn(1, TOTAL_OBS_DIM)
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

    def test_dual_encoder_save_load_preserves_flag(
        self,
        obs_space: spaces.Box,
        act_space: spaces.Box,
    ) -> None:
        """
        @brief Save and load roundtrip preserves use_uncertainty_conditioning flag.
        """
        policy = EvidentialActorCriticPolicy(
            observation_space=obs_space,
            action_space=act_space,
            lr_schedule=lambda _: 3e-4,
            net_arch=[64, 64],
            lambda_reg=0.01,
            use_uncertainty_conditioning=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/test_dual_policy"
            policy.save(path)
            loaded = EvidentialActorCriticPolicy.load(path)
            assert loaded.use_uncertainty_conditioning is True

    def test_gradient_flows_through_dual_encoder(
        self, obs_space: spaces.Box, act_space: spaces.Box
    ) -> None:
        """
        @brief Gradients reach UncertaintyConditionedActor parameters in
               dual-encoder mode.
        """
        policy = EvidentialActorCriticPolicy(
            observation_space=obs_space,
            action_space=act_space,
            lr_schedule=lambda _: 3e-4,
            net_arch=[64, 64],
            lambda_reg=0.01,
            use_uncertainty_conditioning=True,
        )
        obs = torch.randn(BATCH_SIZE, TOTAL_OBS_DIM)
        actions = torch.randn(BATCH_SIZE, ACTION_DIM)
        values, log_prob, _ = policy.evaluate_actions(obs, actions)
        loss = log_prob.mean() + values.mean()
        loss.backward()

        # Gradients must reach the dual-encoder action_net parameters
        for param in policy.action_net.parameters():
            assert param.grad is not None, "No gradient in dual-encoder action_net"
