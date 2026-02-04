## @file evidential_network_demo.py
#  @brief Example: Using the evidential policy network.
#
#  This script demonstrates how to use the evidential policy network for
#  uncertainty-aware action selection.
import torch
import numpy as np
from uncertainty_rl.networks.evidential_policy import EvidentialPolicyNetwork


## @brief Demonstrate evidential policy network usage.
def main():
    print("Creating evidential policy network...")
    
    # Network parameters
    state_dim = 15  # Including uncertainty features
    action_dim = 3  # Steering, throttle, brake
    hidden_dims = [256, 256]
    
    # Create network
    network = EvidentialPolicyNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        activation="relu"
    )
    
    print(f"Network created with {sum(p.numel() for p in network.parameters())} parameters")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Hidden layers: {hidden_dims}")
    
    # Create sample state (batch of 5)
    batch_size = 5
    state = torch.randn(batch_size, state_dim)
    
    print(f"\nSample state batch shape: {state.shape}")
    
    # Forward pass to get evidential parameters
    print("\nForward pass through network...")
    gamma, nu, alpha, beta = network.forward(state)
    
    print(f"Output shapes:")
    print(f"  gamma (mean): {gamma.shape}")
    print(f"  nu (precision): {nu.shape}")
    print(f"  alpha (shape): {alpha.shape}")
    print(f"  beta (rate): {beta.shape}")
    
    # Get action with uncertainty estimates
    print("\nGetting actions with uncertainty...")
    action, uncertainty_dict = network.get_action(state, deterministic=False)
    
    print(f"Action shape: {action.shape}")
    print(f"\nUncertainty estimates (first sample):")
    print(f"  Epistemic: {uncertainty_dict['epistemic'][0].mean().item():.6f}")
    print(f"  Aleatoric: {uncertainty_dict['aleatoric'][0].mean().item():.6f}")
    print(f"  Total: {uncertainty_dict['total'][0].mean().item():.6f}")
    
    # Compare deterministic vs stochastic actions
    print("\nComparing deterministic vs stochastic actions...")
    action_det, _ = network.get_action(state, deterministic=True)
    action_stoch, _ = network.get_action(state, deterministic=False)
    
    print(f"Deterministic action (first sample): {action_det[0].detach().numpy()}")
    print(f"Stochastic action (first sample): {action_stoch[0].detach().numpy()}")
    
    # Demonstrate uncertainty with different input uncertainties
    print("\n\nDemonstrating effect of input uncertainty...")
    
    # Low uncertainty state
    state_low_unc = torch.randn(1, state_dim)
    state_low_unc[0, 6:9] = 0.01  # Very low position uncertainty
    
    # High uncertainty state
    state_high_unc = torch.randn(1, state_dim)
    state_high_unc[0, 6:9] = 0.5  # High position uncertainty
    
    _, unc_low = network.get_action(state_low_unc, deterministic=False)
    _, unc_high = network.get_action(state_high_unc, deterministic=False)
    
    print(f"\nLow input uncertainty:")
    print(f"  Input std: {state_low_unc[0, 6:9].numpy()}")
    print(f"  Output epistemic: {unc_low['epistemic'][0].mean().item():.6f}")
    print(f"  Output aleatoric: {unc_low['aleatoric'][0].mean().item():.6f}")
    
    print(f"\nHigh input uncertainty:")
    print(f"  Input std: {state_high_unc[0, 6:9].numpy()}")
    print(f"  Output epistemic: {unc_high['epistemic'][0].mean().item():.6f}")
    print(f"  Output aleatoric: {unc_high['aleatoric'][0].mean().item():.6f}")
    
    # Demonstrate loss computation
    print("\n\nDemonstrating evidential loss computation...")
    target = torch.randn(batch_size, action_dim)
    
    loss_dict = network.compute_evidential_loss(
        gamma=gamma,
        nu=nu,
        alpha=alpha,
        beta=beta,
        target=target,
        lambda_reg=0.01
    )
    
    print(f"Loss components:")
    print(f"  Total loss: {loss_dict['loss'].item():.6f}")
    print(f"  NLL: {loss_dict['nll'].item():.6f}")
    print(f"  Regularisation: {loss_dict['regularisation'].item():.6f}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
