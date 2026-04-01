"""
@file sb3_integration.py
@brief Stable-Baselines3 integration for evidential actor-critic policy.

Provides SB3-compatible wrappers around the evidential deep learning
components: a custom Distribution, a custom ActorCriticPolicy with
evidential actor head, and a custom PPO subclass that adds evidential
regularisation loss and uncertainty logging.
"""

import logging
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union, cast

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

from uncertainty_rl.networks.evidential_policy import (
    EvidentialLayer,
    UncertaintyConditionedActor,
)
from uncertainty_rl.utils.constants import COVARIANCE_FEATURES_DIM, VEHICLE_STATE_DIM

logger = logging.getLogger("uncertainty_rl.networks.sb3_integration")


def _insert_layernorm(seq: nn.Sequential) -> nn.Sequential:
    """
    @brief Rebuild an nn.Sequential, inserting LayerNorm after each Linear layer.
    @param seq: Original sequential module from MlpExtractor.
    @return New sequential with LayerNorm inserted after every nn.Linear.
    @note Used by EvidentialActorCriticPolicy._build_mlp_extractor() to match
          the LayerNorm-equipped EvidentialPolicyNetwork architecture.
    """
    layers: List[nn.Module] = []
    for layer in seq:
        layers.append(layer)
        if isinstance(layer, nn.Linear):
            layers.append(nn.LayerNorm(layer.out_features))
    return nn.Sequential(*layers)


class EvidentialDistribution(Distribution):
    """
    @class EvidentialDistribution
    @brief SB3-compatible distribution using Gaussian approximation of NIG predictive.

    The evidential network outputs NIG parameters (gamma, nu, alpha, beta).
    For sampling and log_prob, we approximate with Normal(gamma, std) where
    std = sqrt(aleatoric) = sqrt(beta / (nu * (alpha - 1))), the NIG predictive std.
    Epistemic uncertainty is not added to the action std -- it characterises
    model uncertainty over gamma, not per-sample action noise.
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

        # Gaussian approximation: std = sqrt(beta / (nu*(alpha-1))) = sqrt(aleatoric).
        # Epistemic uncertainty is not folded into action std -- it quantifies
        # model uncertainty over gamma, not per-sample noise.
        aleatoric = beta / (nu * (alpha - 1))
        std = th.sqrt(aleatoric)

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
        use_uncertainty_conditioning: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        @brief Initialise the evidential actor-critic policy.
        @param observation_space: Observation space.
        @param action_space: Action space.
        @param lr_schedule: Learning rate schedule.
        @param lambda_reg: Evidential regularisation weight.
        @param use_uncertainty_conditioning: If True, replace the flat MLP actor
               with UncertaintyConditionedActor (dual-encoder). The observation
               is split into vehicle state (indices 0-5) and covariance features
               (indices 6-14) and processed through separate encoder branches
               before fusion. Requires include_covariance=True in the env config.
        """
        self.lambda_reg = lambda_reg
        self.use_uncertainty_conditioning = use_uncertainty_conditioning
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

    def _build_mlp_extractor(self) -> None:
        """
        @brief Build MLP extractor with LayerNorm after each hidden Linear layer.

        Calls the parent implementation then injects nn.LayerNorm into the
        policy and value MLP sequences. This matches the LayerNorm architecture
        used in EvidentialPolicyNetwork for RL training stability (Dohare et al. 2024).
        """
        super()._build_mlp_extractor()
        self.mlp_extractor.policy_net = _insert_layernorm(self.mlp_extractor.policy_net)
        self.mlp_extractor.value_net = _insert_layernorm(self.mlp_extractor.value_net)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        @brief Build networks with evidential actor head and standard critic.

        When use_uncertainty_conditioning=True, wires UncertaintyConditionedActor
        as the action network. The MLP extractor's policy_net is bypassed; the
        dual-encoder receives the raw observation split into vehicle state and
        covariance features. When False, uses the standard flat MLP + EvidentialLayer.

        @param lr_schedule: Learning rate schedule.
        """
        self._build_mlp_extractor()

        action_dim = get_action_dim(self.action_space)
        self.action_dist = EvidentialDistribution(action_dim)

        if self.use_uncertainty_conditioning:
            # Dual-encoder actor: separate pathways for state and covariance.
            # net_arch hidden dims are taken from the MLP extractor latent dim as
            # a proxy for the configured hidden size.
            hidden_dim = self.mlp_extractor.latent_dim_pi
            self.action_net: nn.Module = UncertaintyConditionedActor(
                state_dim=VEHICLE_STATE_DIM,
                uncertainty_dim=COVARIANCE_FEATURES_DIM,
                action_dim=action_dim,
                hidden_dims=[hidden_dim, hidden_dim],
            )
        else:
            # Flat MLP actor: EvidentialLayer on top of MLP extractor latent.
            latent_dim_pi = self.mlp_extractor.latent_dim_pi
            self.action_net = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi
            )

        # Standard value head (critic unchanged)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.use_uncertainty_conditioning:
            logger.info(
                "EvidentialActorCriticPolicy: dual-encoder actor "
                "(state_dim=%d, uncertainty_dim=%d, action_dim=%d, hidden_dim=%d)",
                VEHICLE_STATE_DIM,
                COVARIANCE_FEATURES_DIM,
                action_dim,
                hidden_dim,
            )
        else:
            logger.info(
                "EvidentialActorCriticPolicy: flat MLP actor "
                "(latent_dim=%d, action_dim=%d)",
                self.mlp_extractor.latent_dim_pi,
                action_dim,
            )

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

            # Re-apply NIG hyperprior biases: ortho_init zeroes all biases,
            # overwriting the values set in EvidentialLayer.__init__.
            # Must come AFTER the init_weights loop above.
            # For dual-encoder, the EvidentialLayer lives inside action_net.
            with th.no_grad():
                n = action_dim
                if self.use_uncertainty_conditioning:
                    evid = cast(UncertaintyConditionedActor, self.action_net)
                    evid.evidential_layer.linear.bias[0 * n : 1 * n].fill_(0.0)
                    evid.evidential_layer.linear.bias[1 * n : 2 * n].fill_(0.9)
                    evid.evidential_layer.linear.bias[2 * n : 3 * n].fill_(0.9)
                    evid.evidential_layer.linear.bias[3 * n : 4 * n].fill_(0.0)
                else:
                    flat = cast(EvidentialLayer, self.action_net)
                    flat.linear.bias[0 * n : 1 * n].fill_(0.0)
                    flat.linear.bias[1 * n : 2 * n].fill_(0.9)
                    flat.linear.bias[2 * n : 3 * n].fill_(0.9)
                    flat.linear.bias[3 * n : 4 * n].fill_(0.0)

        # Set up optimiser
        optimizer_kwargs = dict(lr=cast(float, lr_schedule(1)), **self.optimizer_kwargs)
        self.optimizer = self.optimizer_class(self.parameters(), **optimizer_kwargs)

    def _get_nig_from_obs(
        self, obs: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        @brief Run the dual-encoder actor on raw observations.

        Splits the observation into vehicle state (indices 0 to VEHICLE_STATE_DIM-1)
        and covariance features (indices VEHICLE_STATE_DIM to
        VEHICLE_STATE_DIM+COVARIANCE_FEATURES_DIM-1) and passes them through
        the UncertaintyConditionedActor.

        @param obs: Observation tensor of shape (batch, obs_dim).
        @return Tuple (gamma, nu, alpha, beta) of NIG parameters.
        @warning Only valid when use_uncertainty_conditioning=True.
        """
        state = obs[:, :VEHICLE_STATE_DIM]
        uncertainty = obs[
            :, VEHICLE_STATE_DIM : VEHICLE_STATE_DIM + COVARIANCE_FEATURES_DIM
        ]
        dual = cast(UncertaintyConditionedActor, self.action_net)
        result = dual(state, uncertainty)
        return cast(Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor], result)

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        @brief Forward pass for rollout collection.

        SB3's default forward() calls _get_action_dist_from_latent(latent_pi),
        which only works for the flat MLP path. For the dual-encoder path,
        we split the observation into state and covariance features and pass
        them through UncertaintyConditionedActor directly.

        @param obs: Observation tensor.
        @param deterministic: Whether to use deterministic actions.
        @return Tuple of (actions, values, log_prob).
        """
        features = self.extract_features(obs, self.pi_features_extractor)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features = features
            vf_features = self.extract_features(obs, self.vf_features_extractor)
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        values = self.value_net(latent_vf)

        if self.use_uncertainty_conditioning:
            # latent_pi not used for the actor here; dual-encoder reads raw obs.
            gamma, nu, alpha, beta = self._get_nig_from_obs(obs)
            distribution = cast(
                EvidentialDistribution,
                self.action_dist.proba_distribution(gamma, nu, alpha, beta),
            )
        else:
            distribution = self._get_action_dist_from_latent(latent_pi)

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape(-1, self.action_space.shape[0])
        return actions, values, log_prob

    def get_distribution(self, obs: th.Tensor) -> EvidentialDistribution:
        """
        @brief Get the action distribution for given observations.

        Overrides SB3's default which calls _get_action_dist_from_latent().
        For the dual-encoder path, splits observations and passes them
        through UncertaintyConditionedActor directly.

        @param obs: Observation tensor.
        @return Evidential distribution.
        """
        if self.use_uncertainty_conditioning:
            gamma, nu, alpha, beta = self._get_nig_from_obs(obs)
            return cast(
                EvidentialDistribution,
                self.action_dist.proba_distribution(gamma, nu, alpha, beta),
            )
        # Flat MLP path: MLP extractor latent -> EvidentialLayer.
        features = self.extract_features(obs, self.pi_features_extractor)
        if self.share_features_extractor:
            latent_pi, _ = self.mlp_extractor(features)
        else:
            latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def _get_action_dist_from_latent(
        self, latent_pi: th.Tensor
    ) -> EvidentialDistribution:
        """
        @brief Get evidential distribution from latent actor features (flat MLP path).

        Only called when use_uncertainty_conditioning=False. For the dual-encoder
        path, _get_nig_from_obs() is used directly.

        @param latent_pi: Latent features from the actor MLP.
        @return Evidential distribution with NIG parameters set.
        """
        flat = cast(EvidentialLayer, self.action_net)
        gamma, nu, alpha, beta = flat(latent_pi)
        return cast(
            EvidentialDistribution,
            self.action_dist.proba_distribution(gamma, nu, alpha, beta),
        )

    def evaluate_actions(
        self, obs: PyTorchObs, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        @brief Evaluate actions and cache NIG params for evidential loss.

        For the dual-encoder path (use_uncertainty_conditioning=True), the
        UncertaintyConditionedActor receives the raw observation split into
        vehicle state and covariance features, bypassing the MLP extractor's
        policy_net. The critic path is unchanged regardless of mode.

        @param obs: Observations.
        @param actions: Actions to evaluate.
        @return Tuple of (values, log_prob, entropy).
        """
        # No extractor argument: when share_features_extractor=False, SB3 returns
        # a (pi_features, vf_features) tuple which we unpack below.
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        if self.use_uncertainty_conditioning:
            # Dual-encoder: raw obs -> split -> UncertaintyConditionedActor.
            # latent_pi is not used for the actor in this mode.
            gamma, nu, alpha, beta = self._get_nig_from_obs(cast(th.Tensor, obs))
            distribution = cast(
                EvidentialDistribution,
                self.action_dist.proba_distribution(gamma, nu, alpha, beta),
            )
        else:
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
            if self.use_uncertainty_conditioning:
                gamma, nu, alpha, beta = self._get_nig_from_obs(obs)
            else:
                features = self.extract_features(obs)
                if self.share_features_extractor:
                    latent_pi, _ = self.mlp_extractor(features)
                else:
                    pi_features, _ = features
                    latent_pi = self.mlp_extractor.forward_actor(pi_features)
                flat = cast(EvidentialLayer, self.action_net)
                gamma, nu, alpha, beta = flat(latent_pi)

            epistemic = beta / (alpha - 1)
            aleatoric = beta / (nu * (alpha - 1))
            total = epistemic + aleatoric

            if deterministic:
                action = gamma
            else:
                # Consistent with EvidentialDistribution.proba_distribution:
                # use aleatoric std only, not total.
                std = th.sqrt(aleatoric)
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
        @brief Include lambda_reg and use_uncertainty_conditioning in saved parameters.
        @return Dictionary of constructor parameters for save/load.
        """
        data: Dict[str, Any] = cast(
            Dict[str, Any], super()._get_constructor_parameters()
        )
        data["lambda_reg"] = self.lambda_reg
        data["use_uncertainty_conditioning"] = self.use_uncertainty_conditioning
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
        lambda_reg_warmup_steps: int = 50000,
        **kwargs: Any,
    ) -> None:
        """
        @brief Initialise EvidentialPPO.
        @param policy: Policy class or string.
        @param env: Environment.
        @param lambda_reg: Evidential regularisation weight (target value after warmup).
        @param lambda_reg_warmup_steps: Number of environment steps over which
               lambda_reg is linearly annealed from 0 to lambda_reg. Allows the NLL to
               establish good predictions before the regularisation term fires.
        """
        self.lambda_reg = lambda_reg
        self.lambda_reg_warmup_steps = lambda_reg_warmup_steps
        logger.info(
            "EvidentialPPO: lambda_reg=%.4f, warmup_steps=%d",
            lambda_reg,
            lambda_reg_warmup_steps,
        )
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

        # Linearly anneal lambda_reg from 0 to self.lambda_reg over the first
        # lambda_reg_warmup_steps environment steps. This lets the NLL loss
        # establish good predictions before the evidential regularisation fires.
        ramp = min(1.0, float(self.num_timesteps) / float(self.lambda_reg_warmup_steps))
        current_lambda_reg = self.lambda_reg * ramp

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

                # Combined loss with annealed evidential regularisation
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + current_lambda_reg * evidential_reg
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

        # Standard PPO logs. `loss` is the last-batch tensor -- SB3 convention.
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
        self.logger.record("train/lambda_reg", current_lambda_reg)
