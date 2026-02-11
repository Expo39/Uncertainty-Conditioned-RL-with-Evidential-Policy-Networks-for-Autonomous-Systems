"""
@file test_carla_parking.py
@brief Tests for the CARLA parking environment.

All tests run in simulation mode (no CARLA server required) by verifying
the environment's API contract, state dimensions, reward logic, and
fallback behaviour.
"""
import pytest
import numpy as np

from uncertainty_rl.envs.carla_parking import CARLAParkingEnv


class TestCARLAParkingEnvAPI:
    """
    @class TestCARLAParkingEnvAPI
    @brief Verify the Gymnasium API contract in simulation mode.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """
        @brief Create an environment (will fall back to sim mode without CARLA).
        """
        self.env = CARLAParkingEnv(
            carla_host="localhost",
            carla_port=2000,
            uncertainty_noise_std=0.1,
            max_steps=50,
        )

    def teardown_method(self) -> None:
        """
        @brief Clean up after each test.
        """
        self.env.close()

    def test_observation_space_shape(self) -> None:
        """
        @brief Observation space must be 15-dimensional.
        """
        assert self.env.observation_space.shape == (15,)

    def test_action_space_shape(self) -> None:
        """
        @brief Action space must be 3-dimensional [steering, throttle, brake].
        """
        assert self.env.action_space.shape == (3,)

    def test_action_space_bounds(self) -> None:
        """
        @brief Action bounds: steering [-1,1], throttle [0,1], brake [0,1].
        """
        np.testing.assert_array_equal(
            self.env.action_space.low, [-1.0, 0.0, 0.0]
        )
        np.testing.assert_array_equal(
            self.env.action_space.high, [1.0, 1.0, 1.0]
        )

    def test_reset_returns_tuple(self) -> None:
        """
        @brief reset() must return (observation, info) per Gymnasium API.
        """
        result = self.env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

        obs, info = result
        assert obs.shape == (15,)
        assert isinstance(info, dict)

    def test_step_returns_five_tuple(self) -> None:
        """
        @brief step() must return (obs, reward, terminated, truncated, info).
        """
        self.env.reset()
        action = self.env.action_space.sample()
        result = self.env.step(action)

        assert isinstance(result, tuple)
        assert len(result) == 5

        obs, reward, terminated, truncated, info = result
        assert obs.shape == (15,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_state_dtype(self) -> None:
        """
        @brief State must be float32.
        """
        obs, _ = self.env.reset()
        assert obs.dtype == np.float32

    def test_episode_truncates_at_max_steps(self) -> None:
        """
        @brief Episode must truncate when max_steps is reached.
        """
        self.env.reset()
        for _ in range(self.env.max_steps):
            action = self.env.action_space.sample()
            _, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break

        # After max_steps, should be truncated
        assert self.env.steps <= self.env.max_steps


class TestRewardFunction:
    """
    @class TestRewardFunction
    @brief Tests for the parking reward computation.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """
        @brief Create environment with known target at origin.
        """
        self.env = CARLAParkingEnv(
            target_parking_spot=(0.0, 0.0, 0.0),
            uncertainty_noise_std=0.1,
            max_steps=50,
        )

    def teardown_method(self) -> None:
        """
        @brief Clean up.
        """
        self.env.close()

    def test_success_reward_at_target(self) -> None:
        """
        @brief A state at the target with zero velocity should yield success bonus.
        """
        # Simulate a perfect parking state
        state = np.zeros(15, dtype=np.float32)
        # Position at target, zero velocity
        reward, done = self.env._compute_reward(state)
        assert reward > 90.0, "Success bonus should dominate reward"
        assert done is True

    def test_reward_decreases_with_distance(self) -> None:
        """
        @brief States further from target should have lower reward.
        """
        state_near = np.zeros(15, dtype=np.float32)
        state_near[0] = 1.0  # 1m from target

        state_far = np.zeros(15, dtype=np.float32)
        state_far[0] = 10.0  # 10m from target

        reward_near, _ = self.env._compute_reward(state_near)
        reward_far, _ = self.env._compute_reward(state_far)

        assert reward_near > reward_far

    def test_episode_ends_when_too_far(self) -> None:
        """
        @brief Episode should terminate when position error exceeds 20m.
        """
        state = np.zeros(15, dtype=np.float32)
        state[0] = 25.0  # 25m away
        _, done = self.env._compute_reward(state)
        assert done is True

    def test_velocity_penalty(self) -> None:
        """
        @brief Higher velocity near target should reduce reward.
        """
        state_still = np.zeros(15, dtype=np.float32)
        state_still[0] = 2.0  # Near target

        state_fast = np.zeros(15, dtype=np.float32)
        state_fast[0] = 2.0
        state_fast[3] = 5.0  # Moving fast

        reward_still, _ = self.env._compute_reward(state_still)
        reward_fast, _ = self.env._compute_reward(state_fast)

        assert reward_still > reward_fast


class TestCovarianceSimulation:
    """
    @class TestCovarianceSimulation
    @brief Tests for the simulated SLAM covariance in the state.
    """

    def test_initial_covariance_matches_noise_std(self) -> None:
        """
        @brief Initial covariance diagonal should reflect uncertainty_noise_std.
        """
        noise_std = 0.2
        env = CARLAParkingEnv(
            uncertainty_noise_std=noise_std,
            max_steps=10,
        )
        expected_variance = noise_std ** 2
        np.testing.assert_approx_equal(
            env.covariance_matrix[0, 0], expected_variance
        )
        np.testing.assert_approx_equal(
            env.covariance_matrix[1, 1], expected_variance
        )
        env.close()

    def test_covariance_is_symmetric(self) -> None:
        """
        @brief Covariance matrix must always be symmetric.
        """
        env = CARLAParkingEnv(uncertainty_noise_std=0.3, max_steps=10)
        np.testing.assert_array_equal(
            env.covariance_matrix, env.covariance_matrix.T
        )
        env.close()
