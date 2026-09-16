"""
@file test_train_ppo.py
@brief Unit tests for training script helpers.

Tests the pure helper functions in train_ppo.py that do not require CARLA,
ROS 2, or a GPU. The main training entry point is excluded (requires full
Docker stack).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from uncertainty_rl.training.train_ppo import (
    EnvDiagnosticsCallback,
    _apply_stage_training_overrides,
    linear_schedule,
    load_env_config,
    make_env,
)


class TestLinearSchedule:
    """
    @class TestLinearSchedule
    @brief Tests for the linear learning-rate schedule helper.
    """

    def test_returns_callable(self) -> None:
        """
        @brief linear_schedule() must return a callable.
        """
        schedule = linear_schedule(3e-4)
        assert callable(schedule)

    def test_progress_one_gives_initial_value(self) -> None:
        """
        @brief progress_remaining=1.0 is the start of training - LR equals initial.
        """
        initial = 3e-4
        schedule = linear_schedule(initial)
        assert schedule(1.0) == pytest.approx(initial)

    def test_progress_zero_gives_zero(self) -> None:
        """
        @brief progress_remaining=0.0 is end of training - LR decays to zero.
        """
        schedule = linear_schedule(1e-3)
        assert schedule(0.0) == pytest.approx(0.0)

    def test_progress_half_gives_half_initial(self) -> None:
        """
        @brief progress_remaining=0.5 gives half the initial learning rate.
        """
        initial = 2e-4
        schedule = linear_schedule(initial)
        assert schedule(0.5) == pytest.approx(initial * 0.5)

    def test_linear_interpolation(self) -> None:
        """
        @brief LR decreases linearly: value at p equals initial * p.
        """
        initial = 1.0
        schedule = linear_schedule(initial)

        for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            assert schedule(p) == pytest.approx(initial * p, rel=1e-6)

    def test_independent_schedules_dont_interfere(self) -> None:
        """
        @brief Two schedules with different initial values are independent.
        """
        s1 = linear_schedule(1e-3)
        s2 = linear_schedule(5e-4)

        assert s1(0.5) == pytest.approx(5e-4)
        assert s2(0.5) == pytest.approx(2.5e-4)


class TestEnvDiagnosticsCallback:
    """
    @class TestEnvDiagnosticsCallback
    @brief Tests for the EnvDiagnosticsCallback TensorBoard helper.

    SB3's BaseCallback.logger is a read-only property, so tests use
    patch.object to inject a MagicMock logger rather than direct assignment.
    """

    def test_on_step_accumulates_pos_error(self) -> None:
        """
        @brief _on_step() accumulates pos_error into a running sum and step count.
        """
        cb = EnvDiagnosticsCallback()
        cb.locals = {"infos": [{"pos_error": 3.0}, {"pos_error": 7.0}]}
        mock_property = property(lambda self: MagicMock())
        with patch.object(type(cb), "logger", new_callable=lambda: mock_property):
            cb._on_step()
        assert cb._pos_sum == pytest.approx(10.0)
        assert cb._step_count == 2

    def test_on_rollout_end_records_mean_pos_error(self) -> None:
        """
        @brief _on_rollout_end() calls logger.record with the rolling mean pos_error.
        """
        cb = EnvDiagnosticsCallback()
        cb.locals = {
            "infos": [
                {
                    "pos_error": 4.0,
                    "orientation_error": 0.1,
                    "speed": 1.0,
                    "progress_reward": 0.1,
                },
                {
                    "pos_error": 6.0,
                    "orientation_error": 0.2,
                    "speed": 2.0,
                    "progress_reward": 0.2,
                },
            ]
        }
        mock_logger = MagicMock()
        with patch.object(  # noqa: E501
            type(cb), "logger", new_callable=lambda: property(lambda self: mock_logger)
        ):
            cb._on_step()
            cb._on_rollout_end()

        recorded_keys = [call.args[0] for call in mock_logger.record.call_args_list]
        assert "env/mean_pos_error_m" in recorded_keys
        for call in mock_logger.record.call_args_list:
            if call.args[0] == "env/mean_pos_error_m":
                assert call.args[1] == pytest.approx(5.0)

    def test_on_rollout_end_clears_accumulators(self) -> None:
        """
        @brief After _on_rollout_end(), all running accumulators are reset to zero.
        """
        cb = EnvDiagnosticsCallback()
        cb.locals = {
            "infos": [
                {
                    "pos_error": 1.0,
                    "orientation_error": 0.0,
                    "speed": 0.0,
                    "progress_reward": 0.0,
                },
            ]
        }
        mock_logger = MagicMock()
        with patch.object(  # noqa: E501
            type(cb), "logger", new_callable=lambda: property(lambda self: mock_logger)
        ):
            cb._on_step()
            cb._on_rollout_end()

        assert cb._pos_sum == 0.0
        assert cb._ori_sum == 0.0
        assert cb._spd_sum == 0.0
        assert cb._prog_sum == 0.0
        assert cb._step_count == 0

    def test_episode_outcome_rates_logged_on_terminal_step(self) -> None:
        """
        @brief success_rate, collision_rate, timeout_rate are recorded when an
               episode ends (success, collision, or timeout flag is set).
        """
        cb = EnvDiagnosticsCallback()
        cb.locals = {
            "infos": [
                {
                    "pos_error": 0.1,
                    "orientation_error": 0.0,
                    "speed": 0.0,
                    "progress_reward": 0.0,
                    "success": True,
                    "collision": False,
                    "timeout": False,
                },
            ]
        }
        mock_logger = MagicMock()
        with patch.object(  # noqa: E501
            type(cb), "logger", new_callable=lambda: property(lambda self: mock_logger)
        ):
            cb._on_step()
            cb._on_rollout_end()

        recorded_keys = [call.args[0] for call in mock_logger.record.call_args_list]
        assert "env/success_rate" in recorded_keys
        assert "env/collision_rate" in recorded_keys
        assert "env/timeout_rate" in recorded_keys

    def test_success_rate_one_when_all_success(self) -> None:
        """
        @brief success_rate == 1.0 when every terminal step was a success.
        """
        cb = EnvDiagnosticsCallback()
        cb.locals = {
            "infos": [
                {
                    "pos_error": 0.0,
                    "orientation_error": 0.0,
                    "speed": 0.0,
                    "progress_reward": 0.0,
                    "success": True,
                    "collision": False,
                    "timeout": False,
                },
                {
                    "pos_error": 0.0,
                    "orientation_error": 0.0,
                    "speed": 0.0,
                    "progress_reward": 0.0,
                    "success": True,
                    "collision": False,
                    "timeout": False,
                },
            ]
        }
        mock_logger = MagicMock()
        with patch.object(  # noqa: E501
            type(cb), "logger", new_callable=lambda: property(lambda self: mock_logger)
        ):
            cb._on_step()
            cb._on_rollout_end()

        for call in mock_logger.record.call_args_list:
            if call.args[0] == "env/success_rate":
                assert call.args[1] == pytest.approx(1.0)

    def test_no_record_when_no_steps_accumulated(self) -> None:
        """
        @brief If _on_rollout_end() is called with empty accumulators, nothing
               is recorded (no KeyError, no spurious log entries).
        """
        cb = EnvDiagnosticsCallback()
        mock_logger = MagicMock()
        with patch.object(  # noqa: E501
            type(cb), "logger", new_callable=lambda: property(lambda self: mock_logger)
        ):
            cb._on_rollout_end()
        mock_logger.record.assert_not_called()


class TestMakeEnvParallel:
    """
    @class TestMakeEnvParallel
    @brief Tests for make_env() parallel port and EKF path derivation.

    Verifies that each rank gets the correct CARLA port (base + rank*1000) and
    the correct per-instance EKF state file path. Does not require CARLA or
    a GPU - CARLAParkingEnv construction is mocked.
    """

    _BASE_CONFIG = {
        "carla_host": "carla-server",
        "carla_port": 2000,
        "town": "FlatPlane",
        "max_steps": 100,
        "ros2": {"covariance_topic": "/odometry/filtered"},
        "carla_sensors": {},
        "parking_scenarios": {},
        "include_covariance": True,
        "include_obstacle_obs": True,
        "carla_timestep": 0.05,
        "debug": False,
        "map_load_sleep": 0.0,
        "action_repeat": 1,
        "no_rendering_mode": False,
    }

    def test_rank0_uses_base_port(self) -> None:
        """
        @brief Rank 0 connects to the base CARLA port (backward compatible).
        """
        captured: dict = {}

        def _fake_env(**kwargs: object) -> MagicMock:
            captured.update(kwargs)
            return MagicMock()

        with patch(
            "uncertainty_rl.envs.factory.CARLAParkingEnv", side_effect=_fake_env
        ):
            make_env(self._BASE_CONFIG, bay_margin=0.0, rank=0)()

        assert captured["carla_port"] == 2000

    def test_rank1_uses_port_3000(self) -> None:
        """
        @brief Rank 1 connects to base port + 1000 = 3000.
        """
        captured: dict = {}

        def _fake_env(**kwargs: object) -> MagicMock:
            captured.update(kwargs)
            return MagicMock()

        with patch(
            "uncertainty_rl.envs.factory.CARLAParkingEnv", side_effect=_fake_env
        ):
            make_env(self._BASE_CONFIG, bay_margin=0.0, rank=1)()

        assert captured["carla_port"] == 3000

    def test_rank2_uses_port_4000(self) -> None:
        """
        @brief Rank 2 connects to base port + 2000 = 4000.
        """
        captured: dict = {}

        def _fake_env(**kwargs: object) -> MagicMock:
            captured.update(kwargs)
            return MagicMock()

        with patch(
            "uncertainty_rl.envs.factory.CARLAParkingEnv", side_effect=_fake_env
        ):
            make_env(self._BASE_CONFIG, bay_margin=0.0, rank=2)()

        assert captured["carla_port"] == 4000

    def test_rank0_no_ekf_state_file_override(self) -> None:
        """
        @brief Rank 0 does not inject ekf_state_file into ros2_config.

        Backward compatible: single-instance deployments should see the same
        ros2_config they always passed (no extra keys added).
        """
        captured: dict = {}

        def _fake_env(**kwargs: object) -> MagicMock:
            captured.update(kwargs)
            return MagicMock()

        with patch(
            "uncertainty_rl.envs.factory.CARLAParkingEnv", side_effect=_fake_env
        ):
            make_env(self._BASE_CONFIG, bay_margin=0.0, rank=0)()

        assert "ekf_state_file" not in captured["ros2_config"]

    def test_rank1_ekf_state_file_is_ekf_state_1(self) -> None:
        """
        @brief Rank 1 gets ekf_state_1.json derived from the default path.
        """
        captured: dict = {}

        def _fake_env(**kwargs: object) -> MagicMock:
            captured.update(kwargs)
            return MagicMock()

        with patch(
            "uncertainty_rl.envs.factory.CARLAParkingEnv", side_effect=_fake_env
        ):
            make_env(self._BASE_CONFIG, bay_margin=0.0, rank=1)()

        assert (
            captured["ros2_config"]["ekf_state_file"]
            == "/workspace/outputs/ekf_state_1.json"
        )

    def test_rank2_ekf_state_file_is_ekf_state_2(self) -> None:
        """
        @brief Rank 2 gets ekf_state_2.json derived from the default path.
        """
        captured: dict = {}

        def _fake_env(**kwargs: object) -> MagicMock:
            captured.update(kwargs)
            return MagicMock()

        with patch(
            "uncertainty_rl.envs.factory.CARLAParkingEnv", side_effect=_fake_env
        ):
            make_env(self._BASE_CONFIG, bay_margin=0.0, rank=2)()

        assert (
            captured["ros2_config"]["ekf_state_file"]
            == "/workspace/outputs/ekf_state_2.json"
        )


class TestLoadEnvConfigStage:
    """
    @class TestLoadEnvConfigStage
    @brief Tests the curriculum stage override deep-merge in load_env_config().
    """

    def _write_base(self, sim_dir: Path) -> Path:
        """@brief Write a minimal env_config.yaml and return its path."""
        env_path = sim_dir / "env_config.yaml"
        env_path.write_text(
            yaml.safe_dump(
                {
                    "use_extra_spawns": True,
                    "bay_margin": -0.25,
                    "parking_scenarios": {
                        "fixed_floor_plan": "rectangle",
                        "bay_occupancy_min": 0.2,
                        "bay_occupancy_max": 0.8,
                    },
                }
            )
        )
        return env_path

    def test_no_stage_leaves_base_config(self, tmp_path: Path) -> None:
        """
        @brief Without a stage, the base env_config values are unchanged.
        """
        sim_dir = tmp_path / "deployment" / "sim"
        sim_dir.mkdir(parents=True)
        env_path = self._write_base(sim_dir)

        cfg = load_env_config(str(env_path), stage=None)

        assert cfg["use_extra_spawns"] is True
        assert cfg["bay_margin"] == -0.25
        assert cfg["parking_scenarios"]["bay_occupancy_max"] == 0.8

    def test_stage_override_deep_merges_and_wins(self, tmp_path: Path) -> None:
        """
        @brief A stage file overrides top-level and nested keys (deep merge),
               while leaving non-overridden nested keys intact.
        """
        sim_dir = tmp_path / "deployment" / "sim"
        sim_dir.mkdir(parents=True)
        env_path = self._write_base(sim_dir)
        curr_dir = sim_dir / "curriculum"
        curr_dir.mkdir()
        (curr_dir / "stage1.yaml").write_text(
            yaml.safe_dump(
                {
                    "use_extra_spawns": False,
                    "bay_margin": -0.75,
                    "parking_scenarios": {
                        "fixed_target_bay_id": "perpendicular_5",
                        "bay_occupancy_min": 0.0,
                        "bay_occupancy_max": 0.0,
                    },
                }
            )
        )

        cfg = load_env_config(str(env_path), stage=1)

        # Overridden keys take the stage value.
        assert cfg["use_extra_spawns"] is False
        assert cfg["bay_margin"] == -0.75
        assert cfg["parking_scenarios"]["bay_occupancy_max"] == 0.0
        assert cfg["parking_scenarios"]["fixed_target_bay_id"] == "perpendicular_5"
        # Non-overridden nested key from the base survives the deep merge.
        assert cfg["parking_scenarios"]["fixed_floor_plan"] == "rectangle"

    def test_missing_stage_file_raises(self, tmp_path: Path) -> None:
        """
        @brief Requesting a stage with no matching file raises FileNotFoundError.
        """
        sim_dir = tmp_path / "deployment" / "sim"
        sim_dir.mkdir(parents=True)
        env_path = self._write_base(sim_dir)

        with pytest.raises(FileNotFoundError):
            load_env_config(str(env_path), stage=99)


class TestStageTrainingOverrides:
    """
    @class TestStageTrainingOverrides
    @brief Tests the per-stage training_overrides application + allowlist.
    """

    def _write_stage(self, sim_dir: Path, stage: int, overrides: dict) -> str:
        """@brief Write a stage file with a training_overrides block; return env path."""
        env_path = sim_dir / "env_config.yaml"
        env_path.write_text(yaml.safe_dump({"use_extra_spawns": True}))
        curr = sim_dir / "curriculum"
        curr.mkdir(exist_ok=True)
        (curr / f"stage{stage}.yaml").write_text(
            yaml.safe_dump({"training_overrides": overrides})
        )
        return str(env_path)

    def test_allowlisted_overrides_applied(self, tmp_path: Path) -> None:
        """
        @brief Allowlisted keys (stage_timesteps, learning_rate, ent_coef) override
               the merged config.
        """
        sim_dir = tmp_path / "deployment" / "sim"
        sim_dir.mkdir(parents=True)
        env_path = self._write_stage(
            sim_dir,
            1,
            {
                "stage_timesteps": 2000000,
                "learning_rate": 0.0003,
                "ent_coef": 0.02,
            },
        )
        config = {"learning_rate": 0.0001, "ent_coef": 0.005}  # train_config values

        _apply_stage_training_overrides(config, env_path, stage=1)

        assert config["stage_timesteps"] == 2000000
        assert config["learning_rate"] == 0.0003
        assert config["ent_coef"] == 0.02

    def test_architecture_key_rejected(self, tmp_path: Path) -> None:
        """
        @brief A non-allowlisted (architecture) key in training_overrides raises,
               so a stage cannot break weight loading on resume.
        """
        sim_dir = tmp_path / "deployment" / "sim"
        sim_dir.mkdir(parents=True)
        env_path = self._write_stage(sim_dir, 2, {"net_arch": [512, 512]})
        config: dict = {}

        with pytest.raises(ValueError, match="not allowed"):
            _apply_stage_training_overrides(config, env_path, stage=2)

    def test_absent_block_is_noop(self, tmp_path: Path) -> None:
        """
        @brief A stage with no training_overrides block leaves config unchanged.
        """
        sim_dir = tmp_path / "deployment" / "sim"
        sim_dir.mkdir(parents=True)
        env_path = sim_dir / "env_config.yaml"
        env_path.write_text(yaml.safe_dump({"use_extra_spawns": True}))
        curr = sim_dir / "curriculum"
        curr.mkdir()
        (curr / "stage3.yaml").write_text(yaml.safe_dump({"bay_margin": -0.35}))
        config = {"learning_rate": 0.0001}

        _apply_stage_training_overrides(config, str(env_path), stage=3)

        assert config == {"learning_rate": 0.0001}

    def test_training_overrides_stripped_from_env(self, tmp_path: Path) -> None:
        """
        @brief load_env_config() strips training_overrides so it never reaches the
               env dict (it is a training-side block, applied separately).
        """
        sim_dir = tmp_path / "deployment" / "sim"
        sim_dir.mkdir(parents=True)
        env_path = sim_dir / "env_config.yaml"
        env_path.write_text(yaml.safe_dump({"use_extra_spawns": True}))
        curr = sim_dir / "curriculum"
        curr.mkdir()
        (curr / "stage1.yaml").write_text(
            yaml.safe_dump(
                {
                    "bay_margin": -0.75,
                    "training_overrides": {"stage_timesteps": 2000000},
                }
            )
        )

        env_cfg = load_env_config(str(env_path), stage=1)

        assert "training_overrides" not in env_cfg
        assert env_cfg["bay_margin"] == -0.75


class TestBaselineOverlay:
    """
    @class TestBaselineOverlay
    @brief The --baseline overlay flips the train_config defaults to the named
           ablation baseline.

    main() merges train_config (whose defaults ARE the full method) then overlays
    the baseline file with {**config, **baseline} so the run trains that baseline.
    These tests pin the overlay semantics against the real config files - notably
    that the vanilla pathfinder baseline switches the policy to standard PPO and
    drops the covariance observation.
    """

    _REPO = Path(__file__).parent.parent
    _TRAIN_CONFIG = _REPO / "configs" / "train_config.yaml"
    _BASELINES = _REPO / "configs" / "baselines"

    def _overlay(self, baseline_file: str) -> dict:
        """
        @brief Reproduce main()'s {**train_config, **baseline} overlay.
        """
        train_cfg = yaml.safe_load(self._TRAIN_CONFIG.read_text())
        baseline = yaml.safe_load((self._BASELINES / baseline_file).read_text())
        return {**train_cfg, **baseline}

    def test_train_config_default_is_evidential_head(self) -> None:
        """
        @brief The DEFAULT baseline (full_method) is the evidential head - which is
               why an explicit baseline overlay is needed to run the standard-head
               baselines. policy_type is owned by the baseline files, not
               train_config, so the default is asserted via the default overlay.
        """
        assert self._overlay("full_method.yaml")["policy_type"] == "evidential"

    def test_dual_encoder_off_for_clean_ablation(self) -> None:
        """
        @brief use_uncertainty_conditioning must be False so covariance enters
               identically (as obs dims) for both heads - no dual-encoder confound
               on the covariance axis of the 2x2 ablation.
        """
        train_cfg = yaml.safe_load(self._TRAIN_CONFIG.read_text())
        assert train_cfg["evidential"]["use_uncertainty_conditioning"] is False

    def test_vanilla_overlay_selects_standard_no_covariance(self) -> None:
        """
        @brief vanilla_ppo.yaml flips policy_type to standard and covariance off.
        """
        cfg = self._overlay("vanilla_ppo.yaml")
        assert cfg["policy_type"] == "standard"
        assert cfg["include_covariance"] is False
        assert cfg["include_obstacle_obs"] is True
        assert cfg["baseline_name"] == "vanilla_ppo"

    def test_baseline_inherits_ppo_hyperparams(self) -> None:
        """
        @brief The baseline overlay leaves the shared PPO hyperparameters from
               train_config intact (the baseline files override only obs/policy).
        """
        cfg = self._overlay("vanilla_ppo.yaml")
        train_cfg = yaml.safe_load(self._TRAIN_CONFIG.read_text())
        assert cfg["learning_rate"] == train_cfg["learning_rate"]
        assert cfg["n_steps"] == train_cfg["n_steps"]
        assert cfg["gamma"] == train_cfg["gamma"]
