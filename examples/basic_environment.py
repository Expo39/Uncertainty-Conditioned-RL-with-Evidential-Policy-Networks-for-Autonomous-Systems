"""Example: Basic usage of the CARLA parking environment.

This script demonstrates how to create and interact with the CARLA parking
environment with SLAM uncertainty.
"""
import numpy as np
from uncertainty_rl.envs.carla_parking import CARLAParkingEnv


def main():
    """Run a simple episode in the CARLA parking environment."""
    print("Creating CARLA parking environment...")
    
    # Create environment with moderate uncertainty
    env = CARLAParkingEnv(
        carla_host="localhost",
        carla_port=2000,
        town="Town01",
        uncertainty_noise_std=0.1,  # 10cm standard deviation
        max_steps=500,
    )
    
    print("Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Reset environment
    print("\nResetting environment...")
    state, info = env.reset()
    
    print(f"Initial state shape: {state.shape}")
    print(f"Initial position: x={state[0]:.2f}, y={state[1]:.2f}, yaw={np.rad2deg(state[2]):.1f}°")
    print(f"Initial uncertainty: σx={state[6]:.4f}, σy={state[7]:.4f}, σyaw={np.rad2deg(state[8]):.4f}°")
    
    # Run episode with random actions
    print("\nRunning episode with random actions...")
    episode_reward = 0.0
    steps = 0
    
    for step in range(100):
        # Sample random action
        action = env.action_space.sample()
        
        # Step environment
        next_state, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        steps += 1
        
        if step % 10 == 0:
            print(f"Step {step}: reward={reward:.2f}, pos=({next_state[0]:.2f}, {next_state[1]:.2f})")
        
        if terminated or truncated:
            print(f"\nEpisode ended after {steps} steps")
            print(f"Terminated: {terminated}, Truncated: {truncated}")
            break
            
        state = next_state
    
    print(f"\nTotal episode reward: {episode_reward:.2f}")
    print(f"Final position: x={state[0]:.2f}, y={state[1]:.2f}, yaw={np.rad2deg(state[2]):.1f}°")
    
    # Clean up
    env.close()
    print("\nEnvironment closed.")


if __name__ == "__main__":
    main()
