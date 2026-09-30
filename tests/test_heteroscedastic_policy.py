"""
@file test_heteroscedastic_policy.py
@brief Unit tests for the heteroscedastic Gaussian control head.

Pins the contract that makes it a clean control for the evidential head: the
same initial action distribution, clamp and squashing, a state-dependent std,
the plain PPO loss, and the training/evaluation wiring. No CARLA, ROS 2 or GPU.
"""

import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")

from gymnasium import spaces  # noqa: E402

from uncertainty_rl.evaluation.evaluate import evaluate_agent  # noqa: E402
from uncertainty_rl.networks.heteroscedastic import (  # noqa: E402
    _LOG_STD_BIAS,
    HeteroscedasticActorCriticPolicy,
    HeteroscedasticDistribution,
    HeteroscedasticLayer,
    HeteroscedasticPPO,
)
from uncertainty_rl.networks.sb3_integration import (  # noqa: E402
    EvidentialActorCriticPolicy,
    EvidentialDistribution,
    LayerNormActorCriticPolicy,
)
from uncertainty_rl.training import train_ppo  # noqa: E402
from uncertainty_rl.training.train_ppo import linear_schedule  # noqa: E402
from uncertainty_rl.utils.constants import ACTION_DIM, TOTAL_OBS_DIM  # noqa: E402

BATCH_SIZE = 8
FLOOR = 0.04


@pytest.fixture
def obs_space() -> spaces.Box:
    """
    @brief Full observation space (TOTAL_OBS_DIM, both ablation flags on).
    """
    return spaces.Box(
        low=-np.inf, high=np.inf, shape=(TOTAL_OBS_DIM,), dtype=np.float32
    )


@pytest.fixture
def act_space() -> spaces.Box:
    """
    @brief ACTION_DIM continuous action space in [-1, 1], as in the parking env.
    """
    return spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)


@pytest.fixture
def policy(
    obs_space: spaces.Box, act_space: spaces.Box
) -> HeteroscedasticActorCriticPolicy:
    """
    @brief Heteroscedastic policy with a small backbone and the production floor.
    """
    return HeteroscedasticActorCriticPolicy(
        obs_space,
        act_space,
        lr_schedule=lambda _: 3e-4,
        net_arch=[64, 64],
        aleatoric_floor=FLOOR,
    )


def _sampling_std(pol: Any, obs: torch.Tensor) -> torch.Tensor:
    """
    @brief Pre-squash sampling std of a policy's action distribution.
    @param pol: Any SB3 actor-critic policy.
    @param obs: Observation batch.
    @return Std tensor of shape (batch_size, action_dim).
    """
    return pol.get_distribution(obs).distribution.stddev


class TestShapes:
    """
    @class TestShapes
    @brief Output shapes of the layer, policy and uncertainty interface.
    """

    def test_layer_outputs(self) -> None:
        """
        @brief The layer returns (mean, log_std), each (batch_size, action_dim).
        """
        layer = HeteroscedasticLayer(16, ACTION_DIM)
        mean, log_std = layer(torch.randn(BATCH_SIZE, 16))
        assert mean.shape == (BATCH_SIZE, ACTION_DIM)
        assert log_std.shape == (BATCH_SIZE, ACTION_DIM)

    def test_policy_forward_and_evaluate(
        self, policy: HeteroscedasticActorCriticPolicy
    ) -> None:
        """
        @brief forward and evaluate_actions give per-sample actions, values,
               log-probabilities and entropies.
        """
        obs = torch.randn(BATCH_SIZE, TOTAL_OBS_DIM)
        actions, values, log_prob = policy(obs)
        assert actions.shape == (BATCH_SIZE, ACTION_DIM)
        assert values.shape == (BATCH_SIZE, 1)
        assert log_prob.shape == (BATCH_SIZE,)
        assert torch.all(actions.abs() < 1.0)

        values, log_prob, entropy = policy.evaluate_actions(obs, actions)
        assert values.shape == (BATCH_SIZE, 1)
        assert log_prob.shape == (BATCH_SIZE,)
        assert entropy is not None and entropy.shape == (BATCH_SIZE,)

    def test_uncertainty_dict(self, policy: HeteroscedasticActorCriticPolicy) -> None:
        """
        @brief get_action_with_uncertainty reports the variance with no epistemic.
        """
        action, unc = policy.get_action_with_uncertainty(
            torch.randn(BATCH_SIZE, TOTAL_OBS_DIM)
        )
        assert action.shape == (BATCH_SIZE, ACTION_DIM)
        assert set(unc) == {
            "aleatoric",
            "epistemic",
            "total",
            "mean",
            "log_std",
            "action_std",
        }
        for value in unc.values():
            assert value.shape == (BATCH_SIZE, ACTION_DIM)
        assert torch.isnan(unc["epistemic"]).all()
        torch.testing.assert_close(unc["total"], unc["aleatoric"])
        torch.testing.assert_close(unc["action_std"], unc["aleatoric"].sqrt())


class TestInitialisation:
    """
    @class TestInitialisation
    @brief The head must start from the evidential head's action distribution.
    """

    def test_log_std_bias_matches_evidential_prior(self) -> None:
        """
        @brief exp(2 * bias) equals the evidential prior variance
               softplus(0.0) / (softplus(0.9) + 1.5 - 1) ~= 0.398.
        """
        softplus = torch.nn.functional.softplus
        prior = float(softplus(torch.tensor(0.0)) / (softplus(torch.tensor(0.9)) + 0.5))
        assert math.exp(2.0 * _LOG_STD_BIAS) == pytest.approx(prior)
        assert _LOG_STD_BIAS == pytest.approx(-0.4605, abs=1e-4)

    def test_initial_std_matches_evidential(
        self,
        policy: HeteroscedasticActorCriticPolicy,
        obs_space: spaces.Box,
        act_space: spaces.Box,
    ) -> None:
        """
        @brief At a zero observation the latent is zero, so both heads output
               their biases: the sampling stds agree within 1e-4.
        """
        evidential = EvidentialActorCriticPolicy(
            obs_space,
            act_space,
            lr_schedule=lambda _: 3e-4,
            net_arch=[64, 64],
            aleatoric_floor=FLOOR,
        )
        obs = torch.zeros(1, TOTAL_OBS_DIM)
        with torch.no_grad():
            torch.testing.assert_close(
                _sampling_std(policy, obs),
                _sampling_std(evidential, obs),
                atol=1e-4,
                rtol=0.0,
            )

    def test_mean_bias(self, policy: HeteroscedasticActorCriticPolicy) -> None:
        """
        @brief The mean bias survives ortho_init: steer 0, throttle +0.5, brake -1.
        """
        bias = policy.action_net.linear.bias.detach()
        torch.testing.assert_close(
            bias[:ACTION_DIM], torch.tensor([0.0, 0.5, -1.0]), atol=1e-7, rtol=0.0
        )
        torch.testing.assert_close(
            bias[ACTION_DIM:],
            torch.full((ACTION_DIM,), _LOG_STD_BIAS),
            atol=1e-6,
            rtol=0.0,
        )


class TestDistribution:
    """
    @class TestDistribution
    @brief Clamp bounds and parity with the evidential distribution.
    """

    def test_variance_clamp(self) -> None:
        """
        @brief The sampling variance stays in [aleatoric_floor, 1.0] for any
               log_std.
        """
        dist = HeteroscedasticDistribution(ACTION_DIM, aleatoric_floor=FLOOR)
        log_std = torch.linspace(-20.0, 20.0, BATCH_SIZE * ACTION_DIM).reshape(
            BATCH_SIZE, ACTION_DIM
        )
        dist.proba_distribution(torch.zeros_like(log_std), log_std)
        variance = dist.distribution.stddev**2
        assert torch.all(variance >= FLOOR - 1e-7)
        assert torch.all(variance <= 1.0 + 1e-7)

    def test_parity_with_evidential(self) -> None:
        """
        @brief For identical mean and std, log_prob, entropy, mode and the
               squashed sample agree with EvidentialDistribution.
        """
        torch.manual_seed(0)
        mean = torch.randn(BATCH_SIZE, ACTION_DIM)
        std = torch.empty(BATCH_SIZE, ACTION_DIM).uniform_(0.3, 0.9)
        het = HeteroscedasticDistribution(ACTION_DIM, aleatoric_floor=FLOOR)
        het.proba_distribution(mean, torch.log(std))
        # beta / (alpha - 1) = std^2 with alpha = 2.
        evid = EvidentialDistribution(ACTION_DIM, aleatoric_floor=FLOOR)
        evid.proba_distribution(
            mean, torch.ones_like(mean), torch.full_like(mean, 2.0), std**2
        )

        actions = torch.tanh(torch.randn(BATCH_SIZE, ACTION_DIM))
        torch.testing.assert_close(het.log_prob(actions), evid.log_prob(actions))
        torch.testing.assert_close(het.entropy(), evid.entropy())
        torch.testing.assert_close(het.mode(), evid.mode())

        torch.manual_seed(1)
        het_sample = het.sample()
        torch.manual_seed(1)
        evid_sample = evid.sample()
        torch.testing.assert_close(het_sample, evid_sample)
        torch.testing.assert_close(het_sample, torch.tanh(het._gaussian_actions))


class TestStateDependence:
    """
    @class TestStateDependence
    @brief The head under test has a state-dependent std; the standard head not.
    """

    def test_std_gradient_wrt_obs(
        self,
        policy: HeteroscedasticActorCriticPolicy,
        obs_space: spaces.Box,
        act_space: spaces.Box,
    ) -> None:
        """
        @brief d std / d obs is non-zero for the new head and zero for
               LayerNormActorCriticPolicy.
        """
        obs = torch.randn(BATCH_SIZE, TOTAL_OBS_DIM, requires_grad=True)
        _sampling_std(policy, obs).sum().backward()
        assert obs.grad is not None and obs.grad.abs().sum() > 0.0

        standard = LayerNormActorCriticPolicy(
            obs_space, act_space, lr_schedule=lambda _: 3e-4, net_arch=[64, 64]
        )
        obs = torch.randn(BATCH_SIZE, TOTAL_OBS_DIM, requires_grad=True)
        _sampling_std(standard, obs).sum().backward()
        assert obs.grad is None or obs.grad.abs().sum() == 0.0


class TestSaveLoad:
    """
    @class TestSaveLoad
    @brief Save and load round trip of the policy.
    """

    def test_round_trip(self, policy: HeteroscedasticActorCriticPolicy) -> None:
        """
        @brief Weights and aleatoric_floor survive save and load.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "policy.pth")
            policy.save(path)
            loaded = HeteroscedasticActorCriticPolicy.load(path)
        assert loaded.aleatoric_floor == FLOOR
        assert loaded.action_dist.aleatoric_floor == FLOOR
        for key, value in policy.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], value)


class TestHeteroscedasticPPO:
    """
    @class TestHeteroscedasticPPO
    @brief One PPO update logs the variance under the evidential tag.
    """

    def test_train_logs_aleatoric_and_entropy(self) -> None:
        """
        @brief learn() with a callable ent_coef runs one update and records
               train/aleatoric_uncertainty and train/entropy_loss.
        """
        model = HeteroscedasticPPO(
            policy=HeteroscedasticActorCriticPolicy,
            env=gym.make("Pendulum-v1"),
            ent_coef=linear_schedule(0.02, 0.006),
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            policy_kwargs={"aleatoric_floor": FLOOR},
        )
        model.learn(total_timesteps=64)
        logged = model.logger.name_to_value
        assert "train/aleatoric_uncertainty" in logged
        assert "train/entropy_loss" in logged
        assert logged["train/aleatoric_uncertainty"] > 0.0
        # The trace is a train()-only side channel.
        assert model.policy._aleatoric_trace is None
        assert callable(model.ent_coef)


class _Stop(Exception):
    """
    @class _Stop
    @brief Raised by the patched SB3 logger to end train() right after the model
           is built, so no rollout or CARLA env is needed.
    """


class TestTrainPPOWiring:
    """
    @class TestTrainPPOWiring
    @brief train_ppo.train() builds and resumes the heteroscedastic model.
    """

    def _config(self, tmp_path: Path, policy_type: str) -> Dict[str, Any]:
        """
        @brief Minimal train config for a model-construction-only run.
        """
        return {
            "policy_type": policy_type,
            "baseline_name": "heteroscedastic",
            "log_dir": str(tmp_path / "logs"),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "evidential": {"aleatoric_floor": FLOOR},
        }

    def test_fresh_model(self, tmp_path: Path) -> None:
        """
        @brief A fresh run builds HeteroscedasticPPO with the matched policy and
               the evidential aleatoric_floor.
        """
        with (
            patch.object(train_ppo, "HeteroscedasticPPO") as ppo_cls,
            patch.object(train_ppo, "configure", side_effect=_Stop),
        ):
            with pytest.raises(_Stop):
                train_ppo.train(
                    self._config(tmp_path, "heteroscedastic"), env=MagicMock()
                )
        kwargs = ppo_cls.call_args.kwargs
        assert kwargs["policy"] is HeteroscedasticActorCriticPolicy
        assert kwargs["policy_kwargs"]["aleatoric_floor"] == FLOOR
        assert callable(kwargs["ent_coef"])

    def test_resume(self, tmp_path: Path) -> None:
        """
        @brief A resumed run loads HeteroscedasticPPO and re-binds the schedules.
        """
        resume_dir = tmp_path / "resume"
        resume_dir.mkdir()
        (resume_dir / "final_model.zip").touch()
        with (
            patch.object(train_ppo, "HeteroscedasticPPO") as ppo_cls,
            patch.object(train_ppo, "configure", side_effect=_Stop),
        ):
            with pytest.raises(_Stop):
                train_ppo.train(
                    self._config(tmp_path, "heteroscedastic"),
                    env=MagicMock(),
                    resume_from=str(resume_dir),
                )
        ppo_cls.load.assert_called_once_with(str(resume_dir / "final_model"))
        loaded = ppo_cls.load.return_value
        assert callable(loaded.ent_coef)
        assert callable(loaded.lr_schedule)

    def test_unknown_policy_type_raises(self, tmp_path: Path) -> None:
        """
        @brief An unknown policy_type is rejected before any model is built.
        """
        with pytest.raises(ValueError, match="Unknown policy_type"):
            train_ppo.train(self._config(tmp_path, "bogus"), env=MagicMock())


class _TinyEnv(gym.Env):
    """
    @class _TinyEnv
    @brief One-dimensional observation, ACTION_DIM action env for model building.
    """

    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

    def reset(
        self, *, seed: Any = None, options: Any = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        @brief Return a zero observation.
        """
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        @brief Terminate immediately.
        """
        return np.zeros(1, dtype=np.float32), 0.0, True, False, {}


class _OneStepVecEnv:
    """
    @class _OneStepVecEnv
    @brief DummyVecEnv stand-in whose every episode terminates on its first step.
    """

    def __init__(self, infos: List[Dict[str, Any]]) -> None:
        """
        @brief Store one terminal info per episode.
        """
        self._infos = infos
        self._episode = -1

    def reset(self) -> np.ndarray:
        """
        @brief Advance to the next episode.
        """
        self._episode += 1
        return np.zeros((1, 1), dtype=np.float32)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        @brief Terminate with this episode's info.
        """
        return (
            np.zeros((1, 1), dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([True]),
            [self._infos[self._episode]],
        )


class TestEvaluationPath:
    """
    @class TestEvaluationPath
    @brief evaluate_agent records the heteroscedastic variance, epistemic NaN.
    """

    def test_episode_record(self) -> None:
        """
        @brief The episode record carries a finite mean_action_std and a NaN
               mean_epistemic.
        """
        model = HeteroscedasticPPO(
            policy=HeteroscedasticActorCriticPolicy,
            env=_TinyEnv(),
            n_steps=8,
            batch_size=8,
            device="cpu",
            policy_kwargs={"aleatoric_floor": FLOOR},
        )
        metrics = evaluate_agent(
            model=model,
            env=_OneStepVecEnv([{"success": True}]),  # type: ignore[arg-type]
            n_episodes=1,
        )
        record = metrics.episode_records[0]
        assert np.isfinite(record["mean_action_std"])
        assert record["mean_action_std"] > 0.0
        assert np.isfinite(record["mean_aleatoric"])
        assert np.isnan(record["mean_epistemic"])
