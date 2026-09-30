"""
@file heteroscedastic.py
@brief Heteroscedastic Gaussian actor head: the evidential head without the NIG.

Control arm for the head ablation. The actor outputs a state-dependent
(mean, log_std) pair, the soft actor-critic parameterisation, and matches the
evidential head in backbone, initial action distribution, variance clamp,
squashing and entropy, so the only differences are the variance
parameterisation and the absence of the NIG prior anchor.
"""

import math
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import nn
from torch.distributions import Normal

from uncertainty_rl.networks.sb3_integration import (
    EvidentialDistribution,
    ScheduledEntCoefPPO,
    _insert_layernorm,
)
from uncertainty_rl.utils.constants import ACTION_DIM


def _softplus(x: float) -> float:
    """
    @brief Scalar softplus, log(1 + exp(x)), matching torch.nn.functional.softplus.
    @param x: Pre-activation value.
    @return Softplus of x.
    """
    return math.log1p(math.exp(x))


# Pre-tanh action-mean bias [steer, throttle, brake], shared with the evidential
# and standard heads: steer bipolar, throttle default-on, brake default-off.
_MEAN_BIAS: Tuple[float, ...] = (0.0, 0.5, -1.0)
# Evidential alpha/beta bias priors and alpha offset (@see EvidentialLayer). The
# log_std bias is derived from them so both heads start from the same action
# variance, softplus(beta) / (softplus(alpha) + offset - 1) ~= 0.398.
_ALPHA_BIAS_PRIOR: float = 0.9
_ALPHA_OFFSET: float = 1.5
_BETA_BIAS_PRIOR: float = 0.0
_LOG_STD_BIAS: float = 0.5 * math.log(
    _softplus(_BETA_BIAS_PRIOR) / (_softplus(_ALPHA_BIAS_PRIOR) + _ALPHA_OFFSET - 1.0)
)
# Sampling-variance ceiling: std capped at the action half-range, as in
# EvidentialDistribution.proba_distribution.
_VARIANCE_CAP: float = 1.0


class HeteroscedasticLayer(nn.Module):
    """
    @class HeteroscedasticLayer
    @brief Output layer emitting a state-dependent Gaussian (mean, log_std).
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        @brief Constructor for HeteroscedasticLayer.
        @param input_dim: Dimension of the latent actor features.
        @param output_dim: Action dimension. ACTION_DIM gets the per-axis mean
               bias; any other size falls back to a zero mean.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim * 2)
        # Weights scaled by 0.01 so the prior biases dominate at step 0.
        with th.no_grad():
            self.linear.weight.mul_(0.01)
        self.reset_bias()

    def reset_bias(self) -> None:
        """
        @brief Write the prior bias layout [mean | log_std].
        @note Called again by the policy after ortho_init, which zeroes biases.
        """
        n = self.output_dim
        with th.no_grad():
            bias = self.linear.bias
            if n == ACTION_DIM:
                bias[:n] = th.tensor(_MEAN_BIAS, dtype=bias.dtype)
            else:
                bias[:n].fill_(0.0)
            bias[n:].fill_(_LOG_STD_BIAS)

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        @brief Forward pass to compute the Gaussian parameters.
        @param x: Input features of shape (batch_size, input_dim).
        @return Tuple of (mean, log_std), each (batch_size, output_dim).
        """
        out = self.linear(x)
        return out[:, : self.output_dim], out[:, self.output_dim :]


class HeteroscedasticDistribution(EvidentialDistribution):
    """
    @class HeteroscedasticDistribution
    @brief Tanh-squashed Gaussian with a state-dependent variance exp(2 * log_std).

    Inherits log_prob, entropy, sample and mode from EvidentialDistribution so the
    squashing and Jacobian code is shared rather than copied.
    """

    def __init__(self, action_dim: int, aleatoric_floor: float = 1e-6) -> None:
        """
        @brief Initialise the heteroscedastic distribution.
        @param action_dim: Dimension of the action space.
        @param aleatoric_floor: Lower clamp on the sampling variance.
        @see EvidentialDistribution for the aleatoric_floor rationale.
        """
        super().__init__(action_dim, aleatoric_floor=aleatoric_floor)
        self._log_std: Optional[th.Tensor] = None

    def proba_distribution_net(self, latent_dim: int, **kwargs: Any) -> nn.Module:
        """
        @brief Create the heteroscedastic output layer.
        @param latent_dim: Dimension of the latent actor features.
        @return HeteroscedasticLayer module.
        """
        return HeteroscedasticLayer(latent_dim, self.action_dim)

    def proba_distribution(  # type: ignore[override]
        self, mean: th.Tensor, log_std: th.Tensor
    ) -> "HeteroscedasticDistribution":
        """
        @brief Set distribution parameters from the actor output.
        @param mean: Pre-squash mean, shape (batch_size, action_dim).
        @param log_std: Pre-clamp log standard deviation, same shape.
        @return Self with updated distribution.
        """
        self._log_std = log_std
        self._gaussian_actions = None
        # Same clamp as the evidential head: floor for exploration, cap at the
        # action half-range.
        variance = th.clamp(
            th.exp(2.0 * log_std), min=self.aleatoric_floor, max=_VARIANCE_CAP
        )
        self.distribution = Normal(mean, th.sqrt(variance))
        return self

    def actions_from_params(  # type: ignore[override]
        self, mean: th.Tensor, log_std: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        """
        @brief Get actions directly from the Gaussian parameters.
        @param mean: Pre-squash mean.
        @param log_std: Log standard deviation.
        @param deterministic: If True, return tanh(mean).
        @return Actions tensor.
        """
        self.proba_distribution(mean, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(  # type: ignore[override]
        self, mean: th.Tensor, log_std: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        @brief Get actions and log probabilities from the Gaussian parameters.
        @param mean: Pre-squash mean.
        @param log_std: Log standard deviation.
        @return Tuple of (actions, log_prob).
        """
        actions = self.actions_from_params(mean, log_std)
        return actions, self.log_prob(actions)

    @property
    def log_std(self) -> th.Tensor:
        """
        @brief Pre-clamp log standard deviation from the last proba_distribution().
        @return Log std of shape (batch_size, action_dim).
        """
        assert self._log_std is not None
        return self._log_std

    @property
    def raw_variance(self) -> th.Tensor:
        """
        @brief Unclamped predicted variance exp(2 * log_std), the reported signal.
        @return Variance of shape (batch_size, action_dim).
        """
        return th.exp(2.0 * self.log_std)


class HeteroscedasticActorCriticPolicy(ActorCriticPolicy):
    """
    @class HeteroscedasticActorCriticPolicy
    @brief Actor-critic with a state-dependent Gaussian actor and standard critic.

    Mirrors the flat-MLP path of EvidentialActorCriticPolicy (LayerNorm backbone,
    module gains, prior biases) with the NIG head replaced by HeteroscedasticLayer.
    log_std_init is accepted through the shared policy_kwargs and ignored: the
    initial std comes from the matched prior bias.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        aleatoric_floor: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        """
        @brief Initialise the heteroscedastic actor-critic policy.
        @param observation_space: Observation space.
        @param action_space: Action space.
        @param lr_schedule: Learning rate schedule.
        @param aleatoric_floor: Exploration floor on the sampling variance.
        """
        self.aleatoric_floor = aleatoric_floor
        # The state-dependent std replaces gSDE, as in the evidential head.
        kwargs["use_sde"] = False
        # Per-minibatch mean variance, collected only while HeteroscedasticPPO
        # trains; None disables collection.
        self._aleatoric_trace: Optional[List[th.Tensor]] = None
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self) -> None:
        """
        @brief Build MLP extractor with LayerNorm after each hidden Linear layer.
        """
        super()._build_mlp_extractor()
        self.mlp_extractor.policy_net = _insert_layernorm(self.mlp_extractor.policy_net)
        self.mlp_extractor.value_net = _insert_layernorm(self.mlp_extractor.value_net)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        @brief Build networks with the heteroscedastic actor head and standard critic.
        @param lr_schedule: Learning rate schedule.
        """
        self._build_mlp_extractor()

        action_dim = get_action_dim(self.action_space)
        self.action_dist = HeteroscedasticDistribution(
            action_dim, aleatoric_floor=self.aleatoric_floor
        )
        self.action_net = self.action_dist.proba_distribution_net(
            latent_dim=self.mlp_extractor.latent_dim_pi
        )
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains: Dict[nn.Module, float] = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

            # ortho_init zeroes every bias, so the prior must be written after it.
            cast(HeteroscedasticLayer, self.action_net).reset_bias()

        optimizer_kwargs = dict(lr=cast(float, lr_schedule(1)), **self.optimizer_kwargs)
        self.optimizer = self.optimizer_class(self.parameters(), **optimizer_kwargs)

    def _latent_pi(self, obs: PyTorchObs) -> th.Tensor:
        """
        @brief Actor latent features for the given observations.
        @param obs: Observations.
        @return Latent actor features, shape (batch_size, latent_dim_pi).
        """
        features = self.extract_features(obs, self.pi_features_extractor)
        return cast(th.Tensor, self.mlp_extractor.forward_actor(features))

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        @brief Forward pass for rollout collection.
        @param obs: Observation tensor.
        @param deterministic: Whether to use deterministic actions.
        @return Tuple of (actions, values, log_prob).
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = cast(Tuple[th.Tensor, th.Tensor], features)
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        values = self.value_net(latent_vf)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        action_shape = cast(Tuple[int, ...], self.action_space.shape)
        return actions.reshape(-1, action_shape[0]), values, log_prob

    def get_distribution(self, obs: PyTorchObs) -> HeteroscedasticDistribution:
        """
        @brief Get the action distribution for given observations.
        @param obs: Observations.
        @return Heteroscedastic distribution.
        """
        return self._get_action_dist_from_latent(self._latent_pi(obs))

    def _get_action_dist_from_latent(
        self, latent_pi: th.Tensor
    ) -> HeteroscedasticDistribution:
        """
        @brief Get the heteroscedastic distribution from latent actor features.
        @param latent_pi: Latent actor features.
        @return Heteroscedastic distribution.
        """
        mean, log_std = cast(HeteroscedasticLayer, self.action_net)(latent_pi)
        dist = cast(HeteroscedasticDistribution, self.action_dist)
        return dist.proba_distribution(mean, log_std)

    def evaluate_actions(
        self, obs: PyTorchObs, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        @brief Evaluate actions, recording the mean variance when collection is on.
        @param obs: Observations.
        @param actions: Actions to evaluate.
        @return Tuple of (values, log_prob, entropy).
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = cast(Tuple[th.Tensor, th.Tensor], features)
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if self._aleatoric_trace is not None:
            # Detached: a logging side channel that never enters the loss.
            self._aleatoric_trace.append(distribution.raw_variance.detach().mean())
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def get_action_with_uncertainty(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        """
        @brief Get action with its predicted action variance.
        @param obs: Observation tensor.
        @param deterministic: If True, return tanh(mean); otherwise sample.
        @return Tuple of (action, uncertainty_dict) holding aleatoric (raw
                exp(2 * log_std)), epistemic (NaN: no epistemic channel), total
                (= aleatoric), mean, log_std and action_std (sqrt(aleatoric)).
        """
        if self.training:
            self.set_training_mode(False)
        with th.no_grad():
            distribution = self.get_distribution(obs)
            # Sampled through the training distribution so the clamp matches.
            action = distribution.get_actions(deterministic=deterministic)
            # Reported variance is the raw output; only sampling is clamped.
            aleatoric = th.clamp(distribution.raw_variance, min=1e-6)
            mean = cast(Normal, distribution.distribution).mean

        return action, {
            "aleatoric": aleatoric,
            "epistemic": th.full_like(aleatoric, float("nan")),
            "total": aleatoric,
            "mean": mean,
            "log_std": distribution.log_std,
            "action_std": th.sqrt(aleatoric),
        }

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """
        @brief Include aleatoric_floor in saved parameters.
        @return Dictionary of constructor parameters for save/load.
        """
        data: Dict[str, Any] = cast(
            Dict[str, Any], super()._get_constructor_parameters()
        )
        data["aleatoric_floor"] = self.aleatoric_floor
        return data


class HeteroscedasticPPO(ScheduledEntCoefPPO):
    """
    @class HeteroscedasticPPO
    @brief Scheduled-entropy PPO that also logs the heteroscedastic action variance.

    The loss is exactly ScheduledEntCoefPPO's: no prior anchor, since the anchor
    belongs to the NIG mechanism under test. train/aleatoric_uncertainty uses the
    evidential tag so both heads' curves overlay in TensorBoard.
    """

    def train(self) -> None:
        """
        @brief Run one PPO update and log the mean predicted action variance.
        """
        policy = cast(HeteroscedasticActorCriticPolicy, self.policy)
        trace: List[th.Tensor] = []
        policy._aleatoric_trace = trace
        try:
            super().train()
        finally:
            policy._aleatoric_trace = None
        if trace:
            self.logger.record(
                "train/aleatoric_uncertainty", float(th.stack(trace).mean().item())
            )
