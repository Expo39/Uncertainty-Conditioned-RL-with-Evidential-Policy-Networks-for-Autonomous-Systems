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
    @brief SB3-compatible tanh-squashed Gaussian approximation of NIG predictive.
    """

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    # Numerical floor for the tanh Jacobian term; matches SB3.
    _SQUASH_EPS: float = 1e-6

    def __init__(self, action_dim: int, aleatoric_floor: float = 1e-6) -> None:
        """
        @brief Initialise the evidential distribution.
        @param action_dim: Dimension of the action space.
        @param aleatoric_floor: Minimum predictive aleatoric (action variance) before
               the sqrt, so the action sampling std cannot fall below
               sqrt(aleatoric_floor). The NIG variance is the policy's exploration
               noise, and beta is softplus-unbounded toward 0, so without a floor the
               reward gradient can shrink it to a Dirac delta and exploration collapses
               (a documented DER pathology). This is the evidential-actor analogue of
               the SAC log_std clamp.
               @see documentation/detailed_notes/evidential_actor_variance_collapse.md
        """
        super().__init__()
        self.action_dim = action_dim
        self.aleatoric_floor = aleatoric_floor
        self._gamma: Optional[th.Tensor] = None
        self._nu: Optional[th.Tensor] = None
        self._alpha: Optional[th.Tensor] = None
        self._beta: Optional[th.Tensor] = None
        # Pre-squash sample retained for the Jacobian correction.
        self._gaussian_actions: Optional[th.Tensor] = None

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
        self._gaussian_actions = None

        # Clamp before sqrt. The min is an exploration floor: the NIG variance is the
        # action sampling std, and beta is softplus-unbounded toward 0, so the reward
        # gradient can otherwise collapse it to a Dirac delta (DER variance-minimisation
        # pathology). The max is a hard ceiling at the action half-range; the alpha >= 1.5
        # construction bound constrains only the denominator.
        aleatoric = th.clamp(beta / (alpha - 1), min=self.aleatoric_floor, max=1.0)
        std = th.sqrt(aleatoric)

        self.distribution = Normal(gamma, std)
        return self

    def log_prob(
        self, actions: th.Tensor, gaussian_actions: Optional[th.Tensor] = None
    ) -> th.Tensor:
        """
        @brief Log probability of squashed actions, with tanh Jacobian correction.
        @param actions: Squashed actions in (-1, 1), shape (batch_size, action_dim).
        @param gaussian_actions: Optional pre-squash sample retained from sample().
               When None, recovered via atanh on the clipped input.
        @return Log probability summed over action dimensions, shape (batch_size,).
        """
        assert self.distribution is not None
        dist = cast(Normal, self.distribution)
        if gaussian_actions is None:
            gaussian_actions = self._gaussian_actions
        if gaussian_actions is None:
            # atanh is unstable at +-1; clamp by the same epsilon used below.
            clipped = th.clamp(actions, -1.0 + self._SQUASH_EPS, 1.0 - self._SQUASH_EPS)
            gaussian_actions = th.atanh(clipped)
        log_prob_gaussian = sum_independent_dims(dist.log_prob(gaussian_actions))
        # Jacobian of y = tanh(x): dy/dx = 1 - tanh(x)^2.
        jacobian = th.log(1.0 - th.tanh(gaussian_actions) ** 2 + self._SQUASH_EPS)
        return log_prob_gaussian - jacobian.sum(dim=-1)

    def entropy(self) -> Optional[th.Tensor]:
        """
        @brief Differential entropy of the pre-squash Gaussian, summed over action dims.
        @return Entropy of shape (batch_size,), or None if no distribution is set.

        The exact entropy of the tanh-squashed distribution has no closed form (it needs
        the Jacobian expectation), so this returns the base Gaussian entropy
        0.5*log(2*pi*e*std^2) summed over axes - a tractable proxy that, unlike the
        -log_prob fallback, depends DIRECTLY on std. The entropy bonus then acts on the
        action std (the exploration knob), restoring the standard PPO guard against the
        Gaussian shrinking prematurely (Schulman et al. 2017). std reflects the
        aleatoric_floor, so the bonus and the floor reinforce the same exploration level.
        """
        dist = getattr(self, "distribution", None)
        if dist is None:
            return None
        return sum_independent_dims(cast(Normal, dist).entropy())

    def sample(self) -> th.Tensor:
        """
        @brief Sample squashed actions via reparameterisation.
        @return Squashed actions in (-1, 1), shape (batch_size, action_dim).
        """
        assert self.distribution is not None
        dist = cast(Normal, self.distribution)
        gaussian_actions = cast(th.Tensor, dist.rsample())
        self._gaussian_actions = gaussian_actions
        return th.tanh(gaussian_actions)

    def mode(self) -> th.Tensor:
        """
        @brief Deterministic squashed action (tanh of the NIG mean gamma).
        @return Squashed mean actions of shape (batch_size, action_dim).
        """
        assert self.distribution is not None
        dist = cast(Normal, self.distribution)
        gaussian_actions = cast(th.Tensor, dist.mean)
        self._gaussian_actions = gaussian_actions
        return th.tanh(gaussian_actions)

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

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        lambda_reg: float = 0.01,
        aleatoric_floor: float = 1e-6,
        use_uncertainty_conditioning: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        @brief Initialise the evidential actor-critic policy.
        @param observation_space: Observation space.
        @param action_space: Action space.
        @param lr_schedule: Learning rate schedule.
        @param lambda_reg: Evidential regularisation weight.
        @param aleatoric_floor: Minimum predictive aleatoric (action variance); the
               action sampling std cannot fall below sqrt(aleatoric_floor). Passed to
               EvidentialDistribution to stop exploration collapse. @see that class.
        @param use_uncertainty_conditioning: If True, replace the flat MLP actor
               with UncertaintyConditionedActor (dual-encoder). The covariance
               block (obs indices VEHICLE_STATE_DIM through VEHICLE_STATE_DIM +
               COVARIANCE_FEATURES_DIM - 1) is routed through a dedicated
               uncertainty encoder; the rest of the observation (speed, yaw
               rate, relative target pose, LiDAR clearances) is routed through
               the state encoder. The two encoded representations are fused
               before the EvidentialLayer head. Requires include_covariance=True
               in the env config.
        """
        self.lambda_reg = lambda_reg
        self.aleatoric_floor = aleatoric_floor
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
        policy and value MLP sequences.
        """
        super()._build_mlp_extractor()
        self.mlp_extractor.policy_net = _insert_layernorm(self.mlp_extractor.policy_net)
        self.mlp_extractor.value_net = _insert_layernorm(self.mlp_extractor.value_net)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        @brief Build networks with evidential actor head and standard critic.

        When use_uncertainty_conditioning=True, wires UncertaintyConditionedActor
        as the action network. The MLP extractor's policy_net is bypassed; the
        dual-encoder receives the raw observation split into a navigation-state
        block (everything except the covariance dims) and the covariance block.
        When False, uses the standard flat MLP + EvidentialLayer.

        @param lr_schedule: Learning rate schedule.
        """
        self._build_mlp_extractor()

        action_dim = get_action_dim(self.action_space)
        self.action_dist = EvidentialDistribution(
            action_dim, aleatoric_floor=self.aleatoric_floor
        )

        if self.use_uncertainty_conditioning:
            # Dual-encoder actor. The state encoder receives the full
            # observation MINUS the covariance block, so the actor sees
            # speed, yaw rate, the relative target bay pose (dx, dy, dyaw)
            # and the LiDAR clearances. The covariance block (std_x, std_y,
            # std_yaw) flows into the uncertainty encoder only. An actor
            # that cannot see dx / dy / dyaw has no goal-direction signal
            # and cannot learn to drive to the bay; the covariance is a
            # confidence input that modulates the action, not a replacement
            # for the navigation features. The hidden width is taken from
            # the MLP extractor's latent_dim_pi so net_arch in the YAML
            # controls both encoders symmetrically.
            obs_shape = cast(Tuple[int, ...], self.observation_space.shape)
            obs_dim = obs_shape[0]
            state_dim = obs_dim - COVARIANCE_FEATURES_DIM
            hidden_dim = self.mlp_extractor.latent_dim_pi
            self.action_net: nn.Module = UncertaintyConditionedActor(
                state_dim=state_dim,
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
                state_dim,
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
            # Per-axis gamma bias must match EvidentialLayer.__init__ (steer
            # bipolar, throttle default-on, brake default-off) when
            # action_dim=3; symmetric gamma=0 fallback otherwise.
            with th.no_grad():
                n = action_dim
                if self.use_uncertainty_conditioning:
                    evid = cast(UncertaintyConditionedActor, self.action_net)
                    bias = evid.evidential_layer.linear.bias
                else:
                    flat = cast(EvidentialLayer, self.action_net)
                    bias = flat.linear.bias
                if action_dim == 3:
                    bias[0] = 0.0  # gamma steer (bipolar)
                    bias[1] = 0.5  # gamma throttle (default-on)
                    bias[2] = -1.0  # gamma brake (default-off)
                else:
                    bias[0 * n : 1 * n].fill_(0.0)
                bias[1 * n : 2 * n].fill_(0.9)  # nu
                bias[2 * n : 3 * n].fill_(0.9)  # alpha
                bias[3 * n : 4 * n].fill_(0.0)  # beta

        # Set up optimiser
        optimizer_kwargs = dict(lr=cast(float, lr_schedule(1)), **self.optimizer_kwargs)
        self.optimizer = self.optimizer_class(self.parameters(), **optimizer_kwargs)

    # -----------------------------------------------------------------------
    # SB3 overrides
    # -----------------------------------------------------------------------

    def _get_nig_from_obs(
        self, obs: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        @brief Run the dual-encoder actor on raw observations.

        Splits the observation into a navigation-state block and a covariance
        block. The covariance block sits at obs[:, VEHICLE_STATE_DIM:
        VEHICLE_STATE_DIM+COVARIANCE_FEATURES_DIM]; everything else (speed,
        yaw rate, relative target pose, LiDAR clearances) is concatenated
        and forwarded to the state encoder. The covariance block is sent to
        the uncertainty encoder. Both pathways feed into the
        UncertaintyConditionedActor.

        @param obs: Observation tensor of shape (batch, obs_dim).
        @return Tuple (gamma, nu, alpha, beta) of NIG parameters.
        @warning Only valid when use_uncertainty_conditioning=True. Requires
                 include_covariance=True in the env config so the covariance
                 block is actually present at the expected indices.
        """
        cov_start = VEHICLE_STATE_DIM
        cov_end = VEHICLE_STATE_DIM + COVARIANCE_FEATURES_DIM
        # Navigation block: speed and yaw rate before the covariance, plus
        # target pose and LiDAR clearances after it.
        state = th.cat([obs[:, :cov_start], obs[:, cov_end:]], dim=-1)
        uncertainty = obs[:, cov_start:cov_end]
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
        if self.use_uncertainty_conditioning:
            # Dual-encoder: actor reads raw obs directly; only critic MLP is needed.
            if self.share_features_extractor:
                _, latent_vf = self.mlp_extractor(features)
            else:
                vf_features = self.extract_features(obs, self.vf_features_extractor)
                latent_vf = self.mlp_extractor.forward_critic(
                    cast(th.Tensor, vf_features)
                )
            gamma, nu, alpha, beta = self._get_nig_from_obs(obs)
            distribution = cast(
                EvidentialDistribution,
                self.action_dist.proba_distribution(gamma, nu, alpha, beta),
            )
        else:
            if self.share_features_extractor:
                latent_pi, latent_vf = self.mlp_extractor(features)
            else:
                pi_features = cast(th.Tensor, features)
                vf_features = self.extract_features(obs, self.vf_features_extractor)
                latent_pi = self.mlp_extractor.forward_actor(pi_features)
                latent_vf = self.mlp_extractor.forward_critic(
                    cast(th.Tensor, vf_features)
                )
            distribution = self._get_action_dist_from_latent(latent_pi)

        values = self.value_net(latent_vf)

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        action_shape = cast(Tuple[int, ...], self.action_space.shape)
        actions = actions.reshape(-1, action_shape[0])
        return actions, values, log_prob

    def get_distribution(  # type: ignore[override]
        self, obs: th.Tensor
    ) -> EvidentialDistribution:
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
        # Flat MLP path: actor MLP latent -> EvidentialLayer.
        features = self.extract_features(obs, self.pi_features_extractor)
        if self.share_features_extractor:
            latent_pi = self.mlp_extractor.forward_actor(features)
        else:
            latent_pi = self.mlp_extractor.forward_actor(cast(th.Tensor, features))
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
        the navigation-state block (everything except the covariance dims)
        and the covariance block, bypassing the MLP extractor's policy_net.
        The critic path is unchanged regardless of mode.

        @param obs: Observations.
        @param actions: Actions to evaluate.
        @return Tuple of (values, log_prob, entropy).
        """
        # No extractor argument: when share_features_extractor=False, SB3 returns
        # a (pi_features, vf_features) tuple which we unpack below.
        features = self.extract_features(obs)
        if self.use_uncertainty_conditioning:
            # Dual-encoder: only the critic MLP is needed; actor reads raw obs directly.
            if self.share_features_extractor:
                _, latent_vf = self.mlp_extractor(features)
            else:
                _, vf_features = cast(Tuple[th.Tensor, th.Tensor], features)
                latent_vf = self.mlp_extractor.forward_critic(vf_features)
            gamma, nu, alpha, beta = self._get_nig_from_obs(cast(th.Tensor, obs))
            distribution = cast(
                EvidentialDistribution,
                self.action_dist.proba_distribution(gamma, nu, alpha, beta),
            )
        else:
            if self.share_features_extractor:
                latent_pi, latent_vf = self.mlp_extractor(features)
            else:
                pi_features, vf_features = cast(Tuple[th.Tensor, th.Tensor], features)
                latent_pi = self.mlp_extractor.forward_actor(pi_features)
                latent_vf = self.mlp_extractor.forward_critic(vf_features)
            distribution = self._get_action_dist_from_latent(latent_pi)
            gamma, nu, alpha, beta = distribution.nig_params

        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()

        # Cache NIG params for EvidentialPPO.train()
        self._cached_nig_params = (gamma, nu, alpha, beta)

        return values, log_prob, entropy

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

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
        if self.training:
            self.set_training_mode(False)
        with th.no_grad():
            if self.use_uncertainty_conditioning:
                gamma, nu, alpha, beta = self._get_nig_from_obs(obs)
            else:
                features = self.extract_features(obs, self.pi_features_extractor)
                latent_pi = self.mlp_extractor.forward_actor(cast(th.Tensor, features))
                flat = cast(EvidentialLayer, self.action_net)
                gamma, nu, alpha, beta = flat(latent_pi)

            alpha_m1 = alpha - 1
            # Reported aleatoric is the TRUE model output (the uncertainty signal),
            # not floored - only the SAMPLING std below is floored, to match training.
            aleatoric = th.clamp(beta / alpha_m1, min=1e-6)
            epistemic = beta / (nu * alpha_m1)
            total = epistemic + aleatoric

            if deterministic:
                action = th.tanh(gamma)
            else:
                # Match EvidentialDistribution.proba_distribution exactly: aleatoric std
                # only (not total), floored at aleatoric_floor so eval/deployment
                # exploration matches training. Squash with tanh into [-1, 1] per axis.
                sampling_aleatoric = th.clamp(
                    beta / alpha_m1, min=self.aleatoric_floor, max=1.0
                )
                std = th.sqrt(sampling_aleatoric)
                dist = Normal(gamma, std)
                action = th.tanh(dist.sample())

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
        @brief Include lambda_reg, aleatoric_floor and use_uncertainty_conditioning in
               saved parameters.
        @return Dictionary of constructor parameters for save/load.
        """
        data: Dict[str, Any] = cast(
            Dict[str, Any], super()._get_constructor_parameters()
        )
        data["lambda_reg"] = self.lambda_reg
        data["aleatoric_floor"] = self.aleatoric_floor
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

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def __init__(
        self,
        policy: Union[str, type],
        env: Union[GymEnv, str],
        lambda_reg: float = 0.01,
        lambda_reg_warmup_steps: int = 50000,
        lambda_evidence: float = 0.0,
        lambda_evidence_warmup_steps: int = 50000,
        aleatoric_floor: float = 1e-6,
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
        @param lambda_evidence: Evidence-accrual weight (target value after warmup) for
               the nu-only advantage-gated term in train(). Drives nu off its prior so
               the epistemic estimate carries state-dependent signal. 0.0 (the default)
               disables the term, recovering the prior-anchored-nu behaviour.
        @param lambda_evidence_warmup_steps: Number of environment steps over which
               lambda_evidence is linearly annealed from 0 to lambda_evidence, so PPO
               settles the action mean (gamma) before evidence accrual begins.
        @param aleatoric_floor: Minimum predictive aleatoric (action variance) floor for
               the policy's sampling std; forwarded to EvidentialActorCriticPolicy to
               prevent exploration collapse. @see EvidentialDistribution.
        """
        self.lambda_reg = lambda_reg
        self.lambda_reg_warmup_steps = lambda_reg_warmup_steps
        self.lambda_evidence = lambda_evidence
        self.lambda_evidence_warmup_steps = lambda_evidence_warmup_steps
        logger.info(
            "EvidentialPPO: lambda_reg=%.4f, warmup_steps=%d, "
            "lambda_evidence=%.4f, evidence_warmup_steps=%d, aleatoric_floor=%.4g",
            lambda_reg,
            lambda_reg_warmup_steps,
            lambda_evidence,
            lambda_evidence_warmup_steps,
            aleatoric_floor,
        )
        # Pass evidential policy params to policy_kwargs.
        policy_kwargs = kwargs.get("policy_kwargs") or {}
        policy_kwargs["lambda_reg"] = lambda_reg
        policy_kwargs["aleatoric_floor"] = aleatoric_floor
        kwargs["policy_kwargs"] = policy_kwargs
        super().__init__(policy=policy, env=env, **kwargs)

    def train(self) -> None:
        """
        @brief PPO training step with evidential regularisation.

        Reproduces the standard PPO training loop but adds the evidential
        regularisation term to the combined loss. Also logs epistemic and
        aleatoric uncertainty.
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        # Linearly anneal lambda_reg from 0 to self.lambda_reg over the first
        # lambda_reg_warmup_steps environment steps. This lets the NLL loss
        # establish good predictions before the evidential regularisation fires.
        if self.lambda_reg_warmup_steps > 0:
            ramp = min(
                1.0, float(self.num_timesteps) / float(self.lambda_reg_warmup_steps)
            )
        else:
            ramp = 1.0
        current_lambda_reg = self.lambda_reg * ramp

        # Same linear warmup for the evidence-accrual term: defer it until PPO has
        # settled the action mean (gamma), so the residual it reads is meaningful.
        if self.lambda_evidence_warmup_steps > 0:
            evidence_ramp = min(
                1.0,
                float(self.num_timesteps) / float(self.lambda_evidence_warmup_steps),
            )
        else:
            evidence_ramp = 1.0
        current_lambda_evidence = self.lambda_evidence * evidence_ramp

        clip_range_fn = cast(Schedule, self.clip_range)
        clip_range = clip_range_fn(self._current_progress_remaining)

        # ent_coef may be a float (constant) or a Schedule callable (decay).
        # Unlike clip_range / learning_rate, SB3 does NOT wrap ent_coef into a
        # schedule internally - it is stored verbatim as passed to the
        # constructor - so this override must resolve a callable itself.
        ent_coef_attr: Any = getattr(self, "ent_coef")
        if callable(ent_coef_attr):
            ent_coef = float(ent_coef_attr(self._current_progress_remaining))
        else:
            ent_coef = float(ent_coef_attr)

        clip_range_vf: Optional[float] = None
        if self.clip_range_vf is not None:
            clip_range_vf_fn = cast(Schedule, self.clip_range_vf)
            clip_range_vf = clip_range_vf_fn(self._current_progress_remaining)

        # Accumulate as detached tensors.
        # approx_kl_div is synced per-batch only when target_kl early-stopping is active.
        entropy_losses: List[th.Tensor] = []
        pg_losses: List[th.Tensor] = []
        value_losses: List[th.Tensor] = []
        evidential_reg_losses: List[th.Tensor] = []
        evidence_losses: List[th.Tensor] = []
        # clip_fraction as a running float sum.
        clip_fraction_sum: float = 0.0
        clip_fraction_count: int = 0
        epistemic_uncertainties: List[th.Tensor] = []
        aleatoric_uncertainties: List[th.Tensor] = []

        assert self.rollout_buffer is not None
        ev_policy = cast(EvidentialActorCriticPolicy, self.policy)
        # NIG hyperprior targets match EvidentialLayer.__init__ bias values.
        # alpha: softplus(0.9) + 1.5 offset. beta: softplus(0.0). nu is NOT anchored
        # here when the evidence term is active (lambda_evidence > 0): the evidence
        # term governs nu instead, so anchoring it as well would fight that gradient
        # and re-pin nu (collapsing epistemic onto a rescaled aleatoric).
        _alpha_prior = 2.741
        _beta_prior = 0.693
        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs: List[float] = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                # evaluate_actions caches NIG params
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                # Prior-anchoring penalty on the NIG evidence parameters as
                # a raw-ratio quadratic (x / prior - 1)^2 - zero at the prior,
                # with a restoring gradient that grows linearly with distance.
                # Only alpha and beta are anchored; nu is left to the evidence term.
                assert ev_policy._cached_nig_params is not None
                gamma, nu, alpha, beta = ev_policy._cached_nig_params

                evidential_reg = (alpha / _alpha_prior - 1.0).pow(2).mean() + (
                    beta / _beta_prior - 1.0
                ).pow(2).mean()

                # Advantage-gated evidence accrual on nu only. The action mean
                # (gamma) is detached so this term routes gradient solely into the
                # evidence mass nu, never competing with the PPO surrogate over the
                # mean. The residual is measured in the pre-squash space the actor
                # parameterises (actions are tanh-squashed, so atanh maps them back).
                # The advantage weight is clamped to be non-negative so only
                # better-than-baseline actions are allowed to ACCRUE evidence.
                # The per-element term 0.5*sq_resid*nu - 0.5*log(nu) is the nu-only
                # Gaussian-precision negative log-likelihood; its stationary point
                # nu* = 1 / sq_resid raises nu where the action was well-predicted
                # (confident/familiar) and lowers it where it was surprising, so
                # epistemic = beta / (nu * (alpha - 1)) becomes state-dependent.
                # alpha and beta are absent, so this cannot collapse them.
                if self.lambda_evidence > 0.0:
                    raw_advantages = rollout_data.advantages
                    weight = th.clamp(raw_advantages, min=0.0).unsqueeze(-1)
                    # Same atanh-stability epsilon the distribution uses to invert
                    # the squash, so the residual is measured on a matching scale.
                    atanh_eps = EvidentialDistribution._SQUASH_EPS
                    with th.no_grad():
                        clipped_actions = th.clamp(
                            actions,
                            -1.0 + atanh_eps,
                            1.0 - atanh_eps,
                        )
                        pre_squash = th.atanh(clipped_actions)
                        sq_resid = (pre_squash - gamma.detach()).pow(2)
                    evidence_loss = (
                        weight * (0.5 * sq_resid * nu - 0.5 * th.log(nu))
                    ).mean()
                else:
                    evidence_loss = th.zeros((), device=nu.device)

                with th.no_grad():
                    alpha_m1 = alpha - 1
                    aleatoric = (beta / alpha_m1).mean()
                    epistemic = (beta / (nu * alpha_m1)).mean()
                    epistemic_uncertainties.append(epistemic.detach())
                    aleatoric_uncertainties.append(aleatoric.detach())

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

                pg_losses.append(policy_loss.detach())
                # Sync clip fraction to CPU immediately as a float.
                clip_fraction_sum += float(
                    th.mean((th.abs(ratio - 1) > clip_range).float())
                )
                clip_fraction_count += 1

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
                value_losses.append(value_loss.detach())

                # EvidentialDistribution.entropy() returns the closed-form pre-squash
                # Gaussian entropy, so the bonus acts directly on the action std. The
                # -mean(log_prob) branch is a defensive fallback for a None entropy.
                if entropy is None:
                    entropy_loss = -th.mean(log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.detach())

                # Combined loss with annealed evidential regularisation and the
                # annealed evidence-accrual term (both warmed up independently).
                loss = (
                    policy_loss
                    + ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + current_lambda_reg * evidential_reg
                    + current_lambda_evidence * evidence_loss
                )
                evidential_reg_losses.append(evidential_reg.detach())
                evidence_losses.append(evidence_loss.detach())

                # KL divergence for early stopping - only sync to CPU when target_kl is
                # set.
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    kl_tensor = th.mean((th.exp(log_ratio) - 1) - log_ratio)
                    if self.target_kl is not None:
                        approx_kl_div = float(kl_tensor)
                        approx_kl_divs.append(approx_kl_div)
                    else:
                        approx_kl_div = 0.0

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

        def _mean(tensors: List[th.Tensor]) -> float:
            return float(th.stack(tensors).mean().item())

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        # Standard PPO logs. `loss` is the last-batch tensor - SB3 convention.
        self.logger.record("train/entropy_loss", _mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", _mean(pg_losses))
        self.logger.record("train/value_loss", _mean(value_losses))
        kl_mean = float(np.mean(approx_kl_divs)) if approx_kl_divs else 0.0
        self.logger.record("train/approx_kl", kl_mean)
        clip_frac = (
            clip_fraction_sum / clip_fraction_count if clip_fraction_count > 0 else 0.0
        )
        self.logger.record("train/clip_fraction", clip_frac)
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record(
            "train/n_updates",
            self._n_updates,
            exclude="tensorboard",
        )
        self.logger.record("train/clip_range", clip_range)
        # Logged so the decay schedule is visible in TensorBoard.
        self.logger.record("train/ent_coef", ent_coef)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        # Evidential-specific logs
        self.logger.record("train/evidential_reg_loss", _mean(evidential_reg_losses))
        self.logger.record("train/evidence_loss", _mean(evidence_losses))
        self.logger.record(
            "train/epistemic_uncertainty", _mean(epistemic_uncertainties)
        )
        self.logger.record(
            "train/aleatoric_uncertainty", _mean(aleatoric_uncertainties)
        )
        self.logger.record("train/lambda_reg", current_lambda_reg)
        self.logger.record("train/lambda_evidence", current_lambda_evidence)


class ScheduledEntCoefPPO(PPO):
    """
    @class ScheduledEntCoefPPO
    @brief Standard SB3 PPO that accepts a callable (scheduled) ent_coef.

    SB3 wraps learning_rate and clip_range into internal schedules but stores
    ent_coef verbatim, so stock PPO.train() does `self.ent_coef * entropy_loss`
    and raises `unsupported operand type(s) for *: 'function' and 'Tensor'` when
    ent_coef is a linear-decay closure. The evidential path avoids this because
    EvidentialPPO.train() resolves the callable itself; the standard (vanilla /
    input-uncertainty) baselines use this subclass so the SAME ent_coef decay
    schedule drives every baseline - a fairness requirement for the ablation.

    @note Resolves ent_coef(progress_remaining) to a float for the duration of
          one train() call, then restores the callable so the schedule keeps
          advancing on the next update. The whole rollout's epochs share one
          ent_coef value, matching SB3's per-update (not per-epoch) convention.
    """

    def train(self) -> None:
        """
        @brief Resolve a callable ent_coef to a float, then run standard PPO.train().
        """
        ent_coef_attr: Any = getattr(self, "ent_coef")
        if callable(ent_coef_attr):
            self.ent_coef = float(ent_coef_attr(self._current_progress_remaining))
            try:
                super().train()
            finally:
                # Restore the callable so the next update re-resolves the schedule.
                self.ent_coef = ent_coef_attr
        else:
            super().train()


class LayerNormActorCriticPolicy(ActorCriticPolicy):
    """
    @class LayerNormActorCriticPolicy
    @brief Standard Gaussian actor-critic matched to the evidential policy's
           backbone and action prior.

    The 2x2 ablation requires the four baselines to differ ONLY on the two axes
    under test: the actor HEAD (Gaussian vs evidential NIG) and the OBSERVATION
    (covariance present or not). Everything else - the feature backbone and the
    initial action prior - must be identical, or it becomes a confound. Stock
    SB3 MlpPolicy differs from EvidentialActorCriticPolicy in two ways that have
    nothing to do with the head, both removed here:

    1. LayerNorm: EvidentialActorCriticPolicy inserts nn.LayerNorm after every
       hidden Linear in the policy/value MLPs (RL stability); MlpPolicy does not.
       This policy injects the same LayerNorm so the backbones match.
    2. Action prior: the evidential head biases the action mean to a gentle
       default-forward (steer 0, throttle +0.5, brake -1.0 pre-tanh) so the car
       drives at init; MlpPolicy starts at mean 0 (throttle 0 -> pedal-off, the
       car defaults to doing nothing while the brake axis dominates by noise).
       This policy sets the SAME action-mean bias so both heads start from the
       same forward-leaning prior.

    The Gaussian log_std remains the standard learned per-axis parameter - that
    IS the head difference the ablation tests, so it is left as SB3 default.
    """

    def _build_mlp_extractor(self) -> None:
        """
        @brief Build the MLP extractor with LayerNorm after each hidden Linear,
               matching EvidentialActorCriticPolicy.
        """
        super()._build_mlp_extractor()
        self.mlp_extractor.policy_net = _insert_layernorm(self.mlp_extractor.policy_net)
        self.mlp_extractor.value_net = _insert_layernorm(self.mlp_extractor.value_net)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        @brief Build the policy, then set the default-forward action-mean bias.
        @param lr_schedule: Learning rate schedule.

        After the standard build (which ortho-inits and zeroes action_net bias),
        overwrite the Gaussian mean bias to match the evidential head's per-axis
        action prior when action_dim == 3 (steer 0, throttle +0.5, brake -1.0).
        Any other action_dim keeps the symmetric zero default (smoke tests).
        """
        super()._build(lr_schedule)
        action_dim = get_action_dim(self.action_space)
        if action_dim == 3:
            with th.no_grad():
                bias = self.action_net.bias
                bias[0] = 0.0  # steer (bipolar)
                bias[1] = 0.5  # throttle (default-on, gentle forward)
                bias[2] = -1.0  # brake (default-off)
