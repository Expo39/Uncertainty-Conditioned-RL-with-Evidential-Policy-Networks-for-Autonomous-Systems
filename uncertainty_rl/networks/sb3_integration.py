"""
@file sb3_integration.py
@brief Stable-Baselines3 integration for evidential actor-critic policy.

Provides SB3-compatible wrappers around the evidential deep learning
components: a custom Distribution, a custom ActorCriticPolicy with
evidential actor head, and a custom PPO subclass that adds evidential
regularisation loss and uncertainty logging.
"""

from functools import partial
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, cast

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution, sum_independent_dims
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.type_aliases import GymEnv, PyTorchObs, Schedule
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.ppo import PPO
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from uncertainty_rl.networks.evidential_policy import EvidentialLayer

SelfEvidentialPPO = TypeVar("SelfEvidentialPPO", bound="EvidentialPPO")


class EvidentialDistribution(Distribution):
    """
    @class EvidentialDistribution
    @brief SB3-compatible distribution using Gaussian approximation of NIG predictive.

    The evidential network outputs NIG parameters (gamma, nu, alpha, beta).
    For sampling and log_prob, we approximate with Normal(gamma, std) where
    std = sqrt(total_uncertainty) and total = epistemic + aleatoric.
    NIG parameters are cached for the evidential regularisation loss.
    """

    def __init__(self, action_dim: int) -> None:
        """
        @brief Initialise the evidential distribution.
        @param action_dim: Dimension of the action space.
        """
        super().__init__()
        self.action_dim = action_dim
        self._gamma: Optional[th.Tensor] = None
        self._nu: Optional[th.Tensor] = None
        self._alpha: Optional[th.Tensor] = None
        self._beta: Optional[th.Tensor] = None

    def proba_distribution_net(self, latent_dim: int, **kwargs: Any) -> nn.Module:
        """
        @brief Create the evidential output layer.
        @param latent_dim: Dimension of the latent actor features.
        @return EvidentialLayer module.
        """
        return EvidentialLayer(latent_dim, self.action_dim)

    def proba_distribution(
        self,
        gamma: th.Tensor,
        nu: th.Tensor,
        alpha: th.Tensor,
        beta: th.Tensor,
    ) -> "EvidentialDistribution":
        """
        @brief Set distribution parameters from NIG output.
        @param gamma: Mean parameter (batch_size, action_dim).
        @param nu: Precision parameter (batch_size, action_dim).
        @param alpha: Shape parameter (batch_size, action_dim).
        @param beta: Rate parameter (batch_size, action_dim).
        @return Self with updated distribution.
        """
        self._gamma = gamma
        self._nu = nu
        self._alpha = alpha
        self._beta = beta

        # Gaussian approximation of the NIG predictive distribution
        epistemic = beta / (alpha - 1)
        aleatoric = beta / (nu * (alpha - 1))
        total = epistemic + aleatoric
        std = th.sqrt(total)

        self.distribution = Normal(gamma, std)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """
        @brief Compute log probability of actions under Gaussian approximation.
        @param actions: Actions tensor of shape (batch_size, action_dim).
        @return Log probability summed over action dimensions, shape (batch_size,).
        """
        assert self.distribution is not None
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> Optional[th.Tensor]:
        """
        @brief Compute entropy of the Gaussian approximation.
        @return Entropy summed over action dimensions, shape (batch_size,).
        """
        assert self.distribution is not None
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> th.Tensor:
        """
        @brief Sample actions using the reparameterisation trick.
        @return Sampled actions of shape (batch_size, action_dim).
        """
        assert self.distribution is not None
        return cast(th.Tensor, self.distribution.rsample())

    def mode(self) -> th.Tensor:
        """
        @brief Return the deterministic action (mean = gamma).
        @return Mean actions of shape (batch_size, action_dim).
        """
        assert self.distribution is not None
        return cast(th.Tensor, self.distribution.mean)

    def actions_from_params(
        self,
        gamma: th.Tensor,
        nu: th.Tensor,
        alpha: th.Tensor,
        beta: th.Tensor,
        deterministic: bool = False,
    ) -> th.Tensor:
        """
        @brief Get actions directly from NIG parameters.
        @param gamma: Mean parameter.
        @param nu: Precision parameter.
        @param alpha: Shape parameter.
        @param beta: Rate parameter.
        @param deterministic: If True, return mean action.
        @return Actions tensor.
        """
        self.proba_distribution(gamma, nu, alpha, beta)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self,
        gamma: th.Tensor,
        nu: th.Tensor,
        alpha: th.Tensor,
        beta: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        @brief Get actions and log probabilities from NIG parameters.
        @param gamma: Mean parameter.
        @param nu: Precision parameter.
        @param alpha: Shape parameter.
        @param beta: Rate parameter.
        @return Tuple of (actions, log_prob).
        """
        actions = self.actions_from_params(gamma, nu, alpha, beta)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    @property
    def nig_params(
        self,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        @brief Access cached NIG parameters for evidential loss.
        @return Tuple of (gamma, nu, alpha, beta).
        """
        assert self._gamma is not None
        assert self._nu is not None
        assert self._alpha is not None
        assert self._beta is not None
        return self._gamma, self._nu, self._alpha, self._beta


class EvidentialActorCriticPolicy(ActorCriticPolicy):
    """
    @class EvidentialActorCriticPolicy
    @brief Actor-critic policy with evidential actor and standard critic.

    Replaces the standard DiagGaussianDistribution actor head with an
    EvidentialLayer that outputs NIG parameters (gamma, nu, alpha, beta).
    The critic remains a standard MLP value head. Evidential deep learning
    applies to the actor only.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        lambda_reg: float = 0.01,
        **kwargs: Any,
    ) -> None:
        """
        @brief Initialise the evidential actor-critic policy.
        @param observation_space: Observation space.
        @param action_space: Action space.
        @param lr_schedule: Learning rate schedule.
        @param lambda_reg: Evidential regularisation weight.
        """
        self.lambda_reg = lambda_reg
        # Evidential distribution is incompatible with SDE
        kwargs["use_sde"] = False
        # Cache for NIG params set during evaluate_actions()
        self._cached_nig_params: Optional[
            Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]
        ] = None
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        @brief Build networks with evidential actor head and standard critic.
        @param lr_schedule: Learning rate schedule.
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Evidential distribution and action network
        self.action_dist = EvidentialDistribution(get_action_dim(self.action_space))
        self.action_net = self.action_dist.proba_distribution_net(
            latent_dim=latent_dim_pi
        )

        # Standard value head (critic unchanged)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        # Orthogonal initialisation
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

        # Set up optimiser
        optimizer_kwargs = dict(lr=cast(float, lr_schedule(1)), **self.optimizer_kwargs)
        self.optimizer = self.optimizer_class(self.parameters(), **optimizer_kwargs)

    def _get_action_dist_from_latent(
        self, latent_pi: th.Tensor
    ) -> EvidentialDistribution:
        """
        @brief Get evidential distribution from latent actor features.
        @param latent_pi: Latent features from the actor MLP.
        @return Evidential distribution with NIG parameters set.
        """
        gamma, nu, alpha, beta = self.action_net(latent_pi)
        return cast(
            EvidentialDistribution,
            self.action_dist.proba_distribution(gamma, nu, alpha, beta),
        )

    def evaluate_actions(
        self, obs: PyTorchObs, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        @brief Evaluate actions and cache NIG params for evidential loss.
        @param obs: Observations.
        @param actions: Actions to evaluate.
        @return Tuple of (values, log_prob, entropy).
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()

        # Cache NIG params for EvidentialPPO.train()
        self._cached_nig_params = distribution.nig_params

        return values, log_prob, entropy

    def get_action_with_uncertainty(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        """
        @brief Get action with uncertainty estimates.

        Matches the get_action() -> (action, uncertainty_dict) interface
        used throughout the project for evaluation and deployment.

        @param obs: Observation tensor.
        @param deterministic: Whether to use deterministic actions.
        @return Tuple of (action, uncertainty_dict) where uncertainty_dict
                contains epistemic, aleatoric, total, gamma, nu, alpha, beta.
        """
        self.set_training_mode(False)
        with th.no_grad():
            features = self.extract_features(obs)
            if self.share_features_extractor:
                latent_pi, _ = self.mlp_extractor(features)
            else:
                pi_features, _ = features
                latent_pi = self.mlp_extractor.forward_actor(pi_features)

            gamma, nu, alpha, beta = self.action_net(latent_pi)

            epistemic = beta / (alpha - 1)
            aleatoric = beta / (nu * (alpha - 1))
            total = epistemic + aleatoric

            if deterministic:
                action = gamma
            else:
                std = th.sqrt(total)
                dist = Normal(gamma, std)
                action = dist.sample()

        return action, {
            "epistemic": epistemic,
            "aleatoric": aleatoric,
            "total": total,
            "gamma": gamma,
            "nu": nu,
            "alpha": alpha,
            "beta": beta,
        }

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """
        @brief Include lambda_reg in saved constructor parameters.
        @return Dictionary of constructor parameters for save/load.
        """
        data = super()._get_constructor_parameters()
        data["lambda_reg"] = self.lambda_reg
        return data


class EvidentialPPO(PPO):
    """
    @class EvidentialPPO
    @brief PPO with evidential regularisation loss on the actor.

    Adds the evidential regression regularisation term to the PPO loss
    during training. The NIG NLL is handled via the Gaussian approximation
    in log_prob; this class adds the evidential regularisation penalty
    that penalises high evidence on incorrect predictions.

    Logs epistemic and aleatoric uncertainty to TensorBoard alongside
    standard PPO metrics.
    """

    def __init__(
        self,
        policy: Union[str, type],
        env: Union[GymEnv, str],
        lambda_reg: float = 0.01,
        **kwargs: Any,
    ) -> None:
        """
        @brief Initialise EvidentialPPO.
        @param policy: Policy class or string.
        @param env: Environment.
        @param lambda_reg: Evidential regularisation weight.
        """
        self.lambda_reg = lambda_reg
        # Pass lambda_reg to policy_kwargs
        policy_kwargs = kwargs.get("policy_kwargs") or {}
        policy_kwargs["lambda_reg"] = lambda_reg
        kwargs["policy_kwargs"] = policy_kwargs
        super().__init__(policy=policy, env=env, **kwargs)

    def train(self) -> None:
        """
        @brief PPO training step with evidential regularisation.

        Reproduces the standard PPO training loop but adds the evidential
        regularisation term: lambda_reg * mean(|actions - gamma| * (2*nu + alpha))
        to the combined loss. Also logs epistemic and aleatoric uncertainty.
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range_fn = cast(Schedule, self.clip_range)
        clip_range = clip_range_fn(self._current_progress_remaining)
        clip_range_vf: Optional[float] = None
        if self.clip_range_vf is not None:
            clip_range_vf_fn = cast(Schedule, self.clip_range_vf)
            clip_range_vf = clip_range_vf_fn(self._current_progress_remaining)

        entropy_losses: List[float] = []
        pg_losses: List[float] = []
        value_losses: List[float] = []
        evidential_reg_losses: List[float] = []
        clip_fractions: List[float] = []
        epistemic_uncertainties: List[float] = []
        aleatoric_uncertainties: List[float] = []

        assert self.rollout_buffer is not None
        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs: List[float] = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                # evaluate_actions caches NIG params
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                # Evidential regularisation from cached NIG params
                ev_policy = cast(EvidentialActorCriticPolicy, self.policy)
                assert ev_policy._cached_nig_params is not None
                gamma, nu, alpha, beta = ev_policy._cached_nig_params
                error = th.abs(actions - gamma)
                evidential_reg = (error * (2 * nu + alpha)).mean()

                # Log uncertainties
                with th.no_grad():
                    epistemic = (beta / (alpha - 1)).mean()
                    aleatoric = (beta / (nu * (alpha - 1))).mean()
                    epistemic_uncertainties.append(epistemic.item())
                    aleatoric_uncertainties.append(aleatoric.item())

                # Normalise advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # Policy loss (clipped surrogate)
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # Value loss
                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # Combined loss with evidential regularisation
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.lambda_reg * evidential_reg
                )
                evidential_reg_losses.append(evidential_reg.item())

                # KL divergence for early stopping
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(float(approx_kl_div))

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} "
                            f"due to reaching max kl: "
                            f"{approx_kl_div:.2f}"
                        )
                    break

                # Optimisation step
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.max_grad_norm,
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        # Standard PPO logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record(
            "train/n_updates",
            self._n_updates,
            exclude="tensorboard",
        )
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        # Evidential-specific logs
        self.logger.record(
            "train/evidential_reg_loss",
            np.mean(evidential_reg_losses),
        )
        self.logger.record(
            "train/epistemic_uncertainty",
            np.mean(epistemic_uncertainties),
        )
        self.logger.record(
            "train/aleatoric_uncertainty",
            np.mean(aleatoric_uncertainties),
        )
