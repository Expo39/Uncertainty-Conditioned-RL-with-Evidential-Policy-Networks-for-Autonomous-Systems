"""
@file evidential_policy.py
@brief Evidential policy network for uncertainty quantification.

This module implements an evidential deep learning policy network that quantifies
both epistemic (model) uncertainty and aleatoric (data) uncertainty using
evidential distributions.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

logger = logging.getLogger("uncertainty_rl.networks.evidential_policy")


class EvidentialLayer(nn.Module):
    """
    @class EvidentialLayer
    @brief Evidential output layer for uncertainty quantification.

    This layer outputs the parameters of an evidential Normal-Inverse-Gamma (NIG)
    distribution for epistemic and aleatoric uncertainty quantification.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        @brief Constructor for EvidentialLayer.
        @param input_dim: Dimension of input features.
        @param output_dim: Dimension of output (action space).
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Output 4 parameters per action: gamma, nu, alpha, beta
        self.linear = nn.Linear(input_dim, output_dim * 4)

        # Bias layout [gamma | nu | alpha | beta] keeps NIG parameters in a stable
        # prior region at step 0. Weights scaled by 0.01 so biases dominate init.
        # See documentation/detailed_notes/evidential_nig_initialisation.md for derivation.
        with torch.no_grad():
            self.linear.weight.mul_(0.01)
            n = self.output_dim
            self.linear.bias[0 * n : 1 * n].fill_(0.0)  # gamma: zero mean
            self.linear.bias[1 * n : 2 * n].fill_(0.9)  # nu
            self.linear.bias[2 * n : 3 * n].fill_(0.9)  # alpha
            self.linear.bias[3 * n : 4 * n].fill_(0.0)  # beta

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        @brief Forward pass to compute evidential parameters.
        @param x: Input features of shape (batch_size, input_dim).
        @return Tuple of (gamma, nu, alpha, beta) evidential parameters.
                - gamma: Mean of the Gaussian (batch_size, output_dim)
                - nu: Precision parameter (batch_size, output_dim)
                - alpha: Shape parameter (batch_size, output_dim)
                - beta: Rate parameter (batch_size, output_dim)
        """
        out = self.linear(x)
        # Reshape to (batch_size, output_dim, 4)
        out = out.view(-1, self.output_dim, 4)

        # Split into 4 parameters
        gamma = out[..., 0]  # Mean (no constraint)
        nu = F.softplus(out[..., 1]) + 1e-6  # Precision (positive)
        alpha = F.softplus(out[..., 2]) + 1.0  # Shape (> 1 for finite variance)
        beta = F.softplus(out[..., 3]) + 1e-6  # Rate (positive)

        # Clamp NIG parameters to prevent divergence in RL training.
        # Unbounded growth of nu/alpha/beta destabilises the log-penalty regularisation
        # (EvidentialPPO) and the NLL loss. max=100.0 is a conservative ceiling.
        nu = torch.clamp(nu, max=100.0)
        alpha = torch.clamp(alpha, max=100.0)
        beta = torch.clamp(beta, max=100.0)

        return gamma, nu, alpha, beta


class EvidentialPolicyNetwork(nn.Module):
    """
    @class EvidentialPolicyNetwork
    @brief Standalone evidential policy network for testing and experiments.

    @see EvidentialActorCriticPolicy in sb3_integration.py for the RL path.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
    ) -> None:
        """
        @brief Constructor for EvidentialPolicyNetwork.
        @param state_dim: Dimension of state space (including uncertainty features).
        @param action_dim: Dimension of action space.
        @param hidden_dims: List of hidden layer dimensions.
        @param activation: Activation function ('relu', 'tanh', 'elu', 'leaky_relu').
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

        # Factory so each layer gets its own activation module instance.
        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU,
        }
        act_cls = activation_map.get(activation.lower(), nn.ReLU)

        # Build feature extraction layers
        layers: List[nn.Module] = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    act_cls(),
                    nn.LayerNorm(hidden_dim),  # Normalisation for stability
                ]
            )
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Evidential output layer
        self.evidential_layer = EvidentialLayer(prev_dim, action_dim)

        logger.debug(
            "EvidentialPolicyNetwork: state_dim=%d, action_dim=%d, hidden_dims=%s",
            state_dim,
            action_dim,
            hidden_dims,
        )

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        @brief Forward pass through the network.
        @param state: State tensor of shape (batch_size, state_dim).
        @return Tuple of (gamma, nu, alpha, beta) evidential parameters.
        """
        features = self.feature_extractor(state)
        gamma, nu, alpha, beta = self.evidential_layer(features)
        return gamma, nu, alpha, beta

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        @brief Get action from the policy with uncertainty estimates.
        @param state: State tensor of shape (batch_size, state_dim).
        @param deterministic: If True, return mean action. Otherwise, sample.
        @return Tuple of (action, uncertainty_dict) where uncertainty_dict contains
                epistemic and aleatoric uncertainty estimates.
        """
        gamma, nu, alpha, beta = self.forward(state)
        
        aleatoric_uncertainty = beta / (alpha - 1)
        epistemic_uncertainty = beta / (nu * (alpha - 1))
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty

        # For deterministic action, use mean
        if deterministic:
            action = gamma
        else:
            # Epistemic uncertainty (Var[mu]) is not added to action noise --
            # it quantifies model uncertainty over gamma, not per-sample noise.
            std = torch.sqrt(torch.clamp(aleatoric_uncertainty, min=1e-6))
            dist = Normal(gamma, std)
            action = dist.sample()

        uncertainty_dict = {
            "epistemic": epistemic_uncertainty,
            "aleatoric": aleatoric_uncertainty,
            "total": total_uncertainty,
            "gamma": gamma,
            "nu": nu,
            "alpha": alpha,
            "beta": beta,
        }

        return action, uncertainty_dict

    def compute_evidential_loss(
        self,
        gamma: torch.Tensor,
        nu: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        target: torch.Tensor,
        lambda_reg: float = 0.01,
    ) -> Dict[str, torch.Tensor]:
        """
        @brief Compute the full evidential regression loss (NLL + regularisation).
        @param gamma: Mean parameter.
        @param nu: Precision parameter.
        @param alpha: Shape parameter.
        @param beta: Rate parameter.
        @param target: Target values.
        @param lambda_reg: Regularisation coefficient.
        @return Dictionary containing loss components.

        @note This is a standalone supervised regression loss used in unit tests
              and standalone experiments. 
        """
        omega = 2 * beta * (1 + nu)
        nll = (
            0.5 * torch.log(torch.pi / nu)
            - alpha * torch.log(omega)
            + (alpha + 0.5) * torch.log(nu * (target - gamma) ** 2 + omega)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        # Regularisation term to penalise high evidence on wrong predictions
        error = torch.abs(target - gamma)
        reg = error * (2 * nu + alpha)

        loss = nll.mean() + lambda_reg * reg.mean()

        return {
            "loss": loss,
            "nll": nll.mean(),
            "regularisation": reg.mean(),
        }


class UncertaintyConditionedActor(nn.Module):
    """
    @class UncertaintyConditionedActor
    @brief Actor network that conditions on state uncertainty.

    This network explicitly uses uncertainty information in the state to make
    more cautious decisions under high uncertainty.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        state_dim: int,
        uncertainty_dim: int,
        action_dim: int,
        hidden_dims: Optional[List[int]] = None,
    ) -> None:
        """
        @brief Constructor for UncertaintyConditionedActor.
        @param state_dim: Dimension of state (excluding uncertainty features).
        @param uncertainty_dim: Dimension of uncertainty features (e.g., covariance).
        @param action_dim: Dimension of action space.
        @param hidden_dims: List of hidden layer dimensions.
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        # Separate processing for state and uncertainty
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0] // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dims[0] // 2),
        )

        self.uncertainty_encoder = nn.Sequential(
            nn.Linear(uncertainty_dim, hidden_dims[0] // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dims[0] // 2),
        )

        # Combined processing
        combined_layers = []
        prev_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            combined_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                ]
            )
            prev_dim = hidden_dim

        self.combined_layers = nn.Sequential(*combined_layers)

        # Evidential output
        self.evidential_layer = EvidentialLayer(prev_dim, action_dim)

        logger.debug(
            "UncertaintyConditionedActor: state_dim=%d, uncertainty_dim=%d, "
            "action_dim=%d, hidden_dims=%s",
            state_dim,
            uncertainty_dim,
            action_dim,
            hidden_dims,
        )

    def forward(
        self, state: torch.Tensor, uncertainty: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        @brief Forward pass with separate state and uncertainty inputs.
        @param state: State tensor without uncertainty.
        @param uncertainty: Uncertainty features (e.g., covariance matrix elements).
        @return Evidential parameters (gamma, nu, alpha, beta).
        """
        state_features = self.state_encoder(state)
        uncertainty_features = self.uncertainty_encoder(uncertainty)

        # Concatenate and process
        combined = torch.cat([state_features, uncertainty_features], dim=-1)
        features = self.combined_layers(combined)

        gamma, nu, alpha, beta = self.evidential_layer(features)
        return gamma, nu, alpha, beta
