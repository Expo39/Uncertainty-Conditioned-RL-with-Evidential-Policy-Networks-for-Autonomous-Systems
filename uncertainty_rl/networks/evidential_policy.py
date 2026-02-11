"""
@file evidential_policy.py
@brief Evidential policy network for uncertainty quantification.

This module implements an evidential deep learning policy network that quantifies
both epistemic (model) uncertainty and aleatoric (data) uncertainty using
evidential distributions.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class EvidentialLayer(nn.Module):
    """
    @class EvidentialLayer
    @brief Evidential output layer for uncertainty quantification.

    This layer outputs the parameters of an evidential Normal-Inverse-Gamma (NIG)
    distribution, which can be used to quantify both epistemic and aleatoric uncertainty.
    """

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

        return gamma, nu, alpha, beta


class EvidentialPolicyNetwork(nn.Module):
    """
    @class EvidentialPolicyNetwork
    @brief Evidential policy network for actor-critic RL.

    This network outputs action distributions with epistemic and aleatoric uncertainty
    estimates using evidential deep learning.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Optional[list] = None,
        activation: str = "relu",
    ) -> None:
        """
        @brief Constructor for EvidentialPolicyNetwork.
        @param state_dim: Dimension of state space (including uncertainty features).
        @param action_dim: Dimension of action space.
        @param hidden_dims: List of hidden layer dimensions.
        @param activation: Activation function to use.
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

        # Select activation function
        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU,
        }
        self.activation = activation_map.get(activation.lower(), nn.ReLU)()

        # Build feature extraction layers
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    self.activation,
                    nn.LayerNorm(hidden_dim),  # Normalisation for stability
                ]
            )
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Evidential output layer
        self.evidential_layer = EvidentialLayer(prev_dim, action_dim)

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

        # Compute uncertainties
        epistemic_uncertainty = beta / (alpha - 1)  # Epistemic (model) uncertainty
        aleatoric_uncertainty = beta / (
            nu * (alpha - 1)
        )  # Aleatoric (data) uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty

        # For deterministic action, use mean
        if deterministic:
            action = gamma
        else:
            # Sample from Student-t distribution (predictive distribution)
            # Approximate with Gaussian for simplicity
            std = torch.sqrt(total_uncertainty)
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
        @brief Compute evidential regression loss.
        @param gamma: Mean parameter.
        @param nu: Precision parameter.
        @param alpha: Shape parameter.
        @param beta: Rate parameter.
        @param target: Target values.
        @param lambda_reg: Regularisation coefficient.
        @return Dictionary containing loss components.
        """
        # NLL term
        two_beta_lambda = 2 * beta * (1 + nu)
        nll = (
            0.5 * torch.log(np.pi / nu)
            - alpha * torch.log(two_beta_lambda)
            + (alpha + 0.5) * torch.log(nu * (target - gamma) ** 2 + two_beta_lambda)
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

    def __init__(
        self,
        state_dim: int,
        uncertainty_dim: int,
        action_dim: int,
        hidden_dims: Optional[list] = None,
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

        return self.evidential_layer(features)
