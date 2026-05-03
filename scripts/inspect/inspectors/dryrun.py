"""
@file dryrun.py
@brief DryRunInspector and KeyboardController for the full training pipeline dry-run.
"""

import math
import threading
import time
from typing import Any, List, Optional

import numpy as np

try:
    import carla
except ImportError:
    import sys
    print("ERROR: carla Python package not found.  Run inside the training container.")
    sys.exit(1)

from scripts.inspect.inspectors.base import _Inspector, _read_live_tier
from scripts.inspect._drawing import _draw_layout_overlays
from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv


# ---------------------------------------------------------------------------
# Keyboard controller for manual dryrun mode
# ---------------------------------------------------------------------------


class KeyboardController:
    """
    @class KeyboardController
    @brief Latching TTY keyboard input for manual dryrun control.

    Each key latches its axis on until the opposite key or a neutral press
    cancels it.  Up/Down = throttle/brake, Left/Right = steer, Space = stop.
    """

    _STEER_STEP: float = 0.1
    _THROTTLE_STEP: float = 0.1
    _BRAKE_STEP: float = 0.2

    def __init__(self) -> None:
        """@brief Initialise controller with zeroed latched state."""
        import tty  # noqa: F401 
        import termios  # noqa: F401

        self._steer: float = 0.0
        self._longitudinal: float = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """@brief Start background stdin reader thread."""
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print(
            "  Keyboard control (incremental): Up=+throttle  Down=+brake"
            "  Left/Right=steer  Space=stop  Ctrl+C=quit"
        )

    def stop(self) -> None:
        """@brief Signal the reader thread to stop."""
        self._running = False

    def get_action(self) -> np.ndarray:
        """
        @brief Return the current latched [steer, longitudinal] action.
        @return numpy array of shape (2,) with values in [-1.0, 1.0].
        """
        with self._lock:
            return np.array([self._steer, self._longitudinal], dtype=np.float32)

    def _read_loop(self) -> None:
        """@brief Read raw escape sequences from stdin and update latched state."""
        import os
        import select
        import signal
        import sys
        import termios
        import tty

        fd = sys.stdin.fileno()
        try:
            old = termios.tcgetattr(fd)
        except termios.error:
            print(
                "  WARNING: stdin is not a TTY - keyboard control disabled. "
                "Run with MANUAL=true so the container gets an interactive stdin."
            )
            return

        try:
            tty.setraw(fd)
            while self._running:
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not ready:
                    continue
                ch = sys.stdin.read(1)
                if ch == "\x03":
                    os.kill(os.getpid(), signal.SIGINT)
                    break
                if ch == " ":
                    with self._lock:
                        self._steer = 0.0
                        self._longitudinal = 0.0
                elif ch == "\x1b":
                    rest = sys.stdin.read(2)
                    seq = ch + rest
                    with self._lock:
                        if seq == "\x1b[A":
                            self._longitudinal = min(
                                1.0, self._longitudinal + self._THROTTLE_STEP
                            )
                        elif seq == "\x1b[B":
                            self._longitudinal = max(
                                -1.0, self._longitudinal - self._BRAKE_STEP
                            )
                        elif seq == "\x1b[D":
                            self._steer = max(
                                -1.0, self._steer - self._STEER_STEP
                            )
                        elif seq == "\x1b[C":
                            self._steer = min(
                                1.0, self._steer + self._STEER_STEP
                            )
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except termios.error:
                pass


# ---------------------------------------------------------------------------
# Dry-run inspector
# ---------------------------------------------------------------------------


class DryRunInspector(_Inspector):
    """
    @class DryRunInspector
    @brief Runs the full training environment (reset + step loop) with a constant
           forward action or keyboard control, no model.  Spectator follows the ego.
    """

    _LOG_INTERVAL: int = 50  # steps between console obs prints

    def __init__(
        self,
        env: CARLAParkingEnv,
        duration: int,
        n_episodes: Optional[int] = None,
        dryrun_action: Optional[List[float]] = None,
        initial_view: str = "third_person",
        termination_pause: float = 3.0,
        manual: bool = False,
    ) -> None:
        """
        @brief Construct the dry-run inspector.
        @param env: Pre-reset CARLAParkingEnv with include_covariance=True.
        @param duration: Maximum wall-clock seconds to run (across all episodes).
        @param n_episodes: Stop after this many episodes; None = run until duration.
        @param dryrun_action: Fixed [steer, longitudinal] to apply each step.
               None = action_space.sample(). Ignored when manual=True.
        @param initial_view: One of 'third_person', 'side', 'back', 'front', 'free'.
        @param termination_pause: Seconds to hold scene after episode ends.
        @param manual: If True, use keyboard arrow keys instead of random/fixed action.
        """
        super().__init__(env, duration)
        self._n_episodes = n_episodes
        self._dryrun_action: Optional[np.ndarray] = (
            np.array(dryrun_action, dtype=np.float32)
            if dryrun_action is not None
            else None
        )
        self._view: str = (
            initial_view
            if initial_view in ("third_person", "side", "back", "front", "free")
            else "third_person"
        )
        self._termination_pause = termination_pause
        self._keyboard: Optional[KeyboardController] = (
            KeyboardController() if manual else None
        )

    # ------------------------------------------------------------------
    # Spectator placement
    # ------------------------------------------------------------------

    def place_spectator(self) -> None:
        """
        @brief Position spectator for the active view.

        'free' mode: placed once at z=0 beside the vehicle, not touched again.
        All other views: delegates to _update_spectator().
        """
        if self._env.vehicle is None or self._env.world is None:
            return
        if self._view == "free":
            vt = self._env.vehicle.get_transform()
            yaw_rad = math.radians(vt.rotation.yaw)
            sx = vt.location.x - 20.0 * math.sin(yaw_rad)
            sy = vt.location.y + 20.0 * math.cos(yaw_rad)
            look_yaw = math.degrees(math.atan2(vt.location.y - sy, vt.location.x - sx))
            self._env.world.get_spectator().set_transform(
                carla.Transform(
                    carla.Location(x=sx, y=sy, z=0.0),
                    carla.Rotation(pitch=10.0, yaw=look_yaw, roll=0.0),
                )
            )
            print(
                "  Free view: spectator at z=0, pitched up. "
                "Use CARLA controls to fly."
            )
            return
        self._update_spectator()

    def _update_spectator(self) -> None:
        """
        @brief Move spectator to follow the ego vehicle.
        """
        if self._env.vehicle is None or self._env.world is None or self._view == "free":
            return
        vt = self._env.vehicle.get_transform()
        yaw_rad = math.radians(vt.rotation.yaw)
        spectator = self._env.world.get_spectator()

        if self._view == "third_person":
            spectator.set_transform(
                carla.Transform(
                    carla.Location(
                        x=vt.location.x - 20.0 * math.cos(yaw_rad),
                        y=vt.location.y - 20.0 * math.sin(yaw_rad),
                        z=vt.location.z + 10.0,
                    ),
                    carla.Rotation(pitch=-25.0, yaw=vt.rotation.yaw),
                )
            )
        elif self._view == "side":
            sx = vt.location.x - 15.0 * math.sin(yaw_rad)
            sy = vt.location.y + 15.0 * math.cos(yaw_rad)
            spectator.set_transform(
                carla.Transform(
                    carla.Location(x=sx, y=sy, z=vt.location.z),
                    carla.Rotation(
                        pitch=0.0,
                        yaw=math.degrees(
                            math.atan2(vt.location.y - sy, vt.location.x - sx)
                        ),
                    ),
                )
            )
        elif self._view == "back":
            spectator.set_transform(
                carla.Transform(
                    carla.Location(
                        x=vt.location.x - 20.0 * math.cos(yaw_rad),
                        y=vt.location.y - 20.0 * math.sin(yaw_rad),
                        z=vt.location.z,
                    ),
                    carla.Rotation(pitch=0.0, yaw=vt.rotation.yaw),
                )
            )
        else:  # front
            fx = vt.location.x + 15.0 * math.cos(yaw_rad)
            fy = vt.location.y + 15.0 * math.sin(yaw_rad)
            spectator.set_transform(
                carla.Transform(
                    carla.Location(x=fx, y=fy, z=vt.location.z),
                    carla.Rotation(
                        pitch=0.0,
                        yaw=math.degrees(
                            math.atan2(vt.location.y - fy, vt.location.x - fx)
                        ),
                    ),
                )
            )

    # ------------------------------------------------------------------
    # Observation logging
    # ------------------------------------------------------------------

    def _print_obs(self, obs: Any, step: int, episode: int) -> None:
        """
        @brief Print key observation values to the console for diagnosis.

        WHITE = model inputs, YELLOW = CARLA ground truth, RED = EKF diagnostic.

        @param obs: Observation array from env.step() or env.reset().
        @param step: Current step within the episode.
        @param episode: Current episode index.
        """
        _WHITE = "\033[97m"
        _YELLOW = "\033[33m"
        _RED = "\033[31m"
        _RESET = "\033[0m"
        _CYAN = "\033[36m"

        if obs is None or len(obs) < 3:
            return

        W = _WHITE
        Y = _YELLOW
        R = _RED
        X = _RESET
        lines = []

        _live_tier = _read_live_tier()
        lines.append(
            f"--- ep={episode}  step={step}  "
            + _CYAN + f"tier={_live_tier}" + X
            + " " + "-" * 28
        )

        lines.append(W + f"vel  vyaw={math.degrees(obs[0]):+.1f}deg/s" + X)

        if self._env.vehicle is not None:
            t = self._env.vehicle.get_transform()
            v = self._env.vehicle.get_velocity()
            spd = math.sqrt(v.x**2 + v.y**2)
            lines.append(
                Y + f"GT   pos=({t.location.x:.2f},{t.location.y:.2f})"
                f"  yaw={t.rotation.yaw:+.1f}deg  spd={spd:.2f}m/s" + X
            )

        if self._env._cov_subscriber is not None:
            ekf_pose = self._env._cov_subscriber.get_latest_pose()
            if ekf_pose is not None:
                ekf_x = float(ekf_pose[0])
                ekf_y = -float(ekf_pose[1])
                ekf_yaw = float(ekf_pose[2])
                lines.append(
                    R + f"EKF(odom)   x={ekf_x:.2f}  y={ekf_y:.2f}"
                    f"  yaw={math.degrees(ekf_yaw):+.1f}deg" + X
                )
                tx, ty, cos_r, sin_r, r = self._env._ekf_odom_offset
                wx = cos_r * ekf_x - sin_r * ekf_y + tx
                wy = sin_r * ekf_x + cos_r * ekf_y + ty
                wyaw_rad = math.atan2(
                    math.sin(ekf_yaw + r), math.cos(ekf_yaw + r)
                )
                wyaw = math.degrees(wyaw_rad)
                lines.append(
                    R + f"EKF(world)  x={wx:.2f}  y={wy:.2f}"
                    f"  yaw={wyaw:+.1f}deg" + X
                )

        if len(obs) >= 4:
            lines.append(
                W + f"cov  std=({obs[1]:.3f},{obs[2]:.3f},{obs[3]:.3f})" + X
            )

        if len(obs) >= 7:
            lines.append(
                W + f"tgt  dx={obs[4]:+.2f}m  dy={obs[5]:+.2f}m"
                f"  dyaw={math.degrees(obs[6]):+.1f}deg" + X
            )

        gt = self._env._target_bay
        _, _, _, _, r = self._env._ekf_odom_offset

        ekf_std = 0.0
        if self._env._cov_subscriber is not None:
            _, unc = self._env._cov_subscriber.get_latest_state()
            if unc is not None:
                ekf_std = float(max(unc[0], unc[1]))

        lines.append(
            Y + f"bay  world=({gt['x']:.2f},{gt['y']:.2f})"
            f"  yaw={math.degrees(gt['yaw']):+.1f}deg"
            f"  ekf_std={ekf_std:.3f}m  r={math.degrees(r):+.1f}deg" + X
        )

        if len(obs) >= 12:
            lines.append(
                W + f"obs  L={obs[7]:.2f}m({math.degrees(obs[8]):+.1f}deg)"
                f"  R={obs[9]:.2f}m({math.degrees(obs[10]):+.1f}deg)"
                f"  F={obs[11]:.2f}m" + X
            )

        print("\n" + "\n".join(lines))

    # ------------------------------------------------------------------
    # Overlay drawing
    # ------------------------------------------------------------------

    def _draw_overlays(self, life_time: float) -> None:
        """
        @brief Draw target bay highlight and lot geometry in the CARLA window.
        @param life_time: Primitive lifetime in seconds.
        """
        if self._env.world is None or not self._env._current_layout:
            return
        _draw_layout_overlays(
            self._env.world,
            self._env._current_layout,
            self._env._target_bay.get("bay_id", ""),
            life_time,
        )

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        @brief Run the dry-run episode loop.

        Each episode: reset() -> step loop -> termination_pause hold.
        Spectator follows ego every step; obs printed every _LOG_INTERVAL steps.
        """
        if self._env.world is None:
            print("ERROR: CARLA world not available.  Aborting.")
            return

        if self._keyboard is not None:
            self._keyboard.start()

        deadline = time.monotonic() + self._duration
        episode = 0
        total_steps = 0
        _latest_obs = None

        mode_desc = (
            "keyboard control"
            if self._keyboard is not None
            else f"constant action {self._dryrun_action.tolist()}"
        )
        print(
            f"Dry-run: {mode_desc} for up to {self._duration}s"
            + (f" / {self._n_episodes} episodes." if self._n_episodes else ".")
        )
        print(
            "\nModel inputs per step (12-dim obs):"
            "\n  [0]    vel: vyaw"
            "\n  [1-3]  cov: std(x,y,yaw)"
            "\n  [4-6]  tgt: dx dy dyaw (ego-relative)"
            "\n  [7-11] obs: L(dist,bear)  R(dist,bear)  F(dist)"
            "\n  bay/EKF lines are diagnostic only (not fed to model)"
        )

        try:
            while time.monotonic() < deadline:
                if self._n_episodes is not None and episode >= self._n_episodes:
                    break

                obs, _ = self._env.reset()
                episode += 1
                step = 0
                _rmse_pos_sq = 0.0
                _rmse_yaw_sq = 0.0
                _rmse_n = 0
                print(f"\n--- Episode {episode}  [view: {self._view}] ---")
                self._update_spectator()
                self._draw_overlays(life_time=self._OVERLAY_LIFE)
                self._print_obs(obs, step, episode)

                terminated = truncated = False
                while not (terminated or truncated):
                    if time.monotonic() >= deadline:
                        break
                    if self._keyboard is not None:
                        action = self._keyboard.get_action()
                    else:
                        action = self._dryrun_action
                    obs, reward, terminated, truncated, info = self._env.step(action)
                    step += 1
                    total_steps += 1

                    if self._env._cov_subscriber is not None and self._env.vehicle is not None:
                        _ep = self._env._cov_subscriber.get_latest_pose()
                        if _ep is not None:
                            _tx, _ty, _cr, _sr, _rr = self._env._ekf_odom_offset
                            _wx = _cr * float(_ep[0]) - _sr * (-float(_ep[1])) + _tx
                            _wy = _sr * float(_ep[0]) + _cr * (-float(_ep[1])) + _ty
                            _gt = self._env.vehicle.get_transform()
                            _pos_err = math.sqrt(
                                (_wx - _gt.location.x) ** 2
                                + (_wy - _gt.location.y) ** 2
                            )
                            _ekf_wyaw = math.atan2(
                                math.sin(float(_ep[2]) + _rr),
                                math.cos(float(_ep[2]) + _rr),
                            )
                            _gt_yaw = math.radians(_gt.rotation.yaw)
                            _yaw_err = math.atan2(
                                math.sin(_ekf_wyaw - _gt_yaw),
                                math.cos(_ekf_wyaw - _gt_yaw),
                            )
                            _rmse_pos_sq += _pos_err ** 2
                            _rmse_yaw_sq += _yaw_err ** 2
                            _rmse_n += 1

                    time.sleep(
                        self._env._action_repeat * self._env._carla_timestep
                    )
                    self._update_spectator()

                    _gt_done = False
                    if self._env.vehicle is not None:
                        _vt = self._env.vehicle.get_transform()
                        _tgt = self._env._target_bay
                        _gt_dist = math.sqrt(
                            (_vt.location.x - _tgt["x"]) ** 2
                            + (_vt.location.y - _tgt["y"]) ** 2
                        )
                        if _gt_dist <= 0.5:
                            print(
                                f"\n  [GT proximity] {_gt_dist:.2f}m from target "
                                f"-- ending episode early"
                            )
                            self._print_obs(obs, step, episode)
                            _gt_done = True
                            terminated = True

                    if not _gt_done and step % self._LOG_INTERVAL == 0:
                        self._draw_overlays(life_time=self._OVERLAY_LIFE)
                        self._print_obs(obs, step, episode)
                    _latest_obs = obs

                reason = (
                    "gt_proximity"
                    if _gt_done
                    else info.get(
                        "termination_reason",
                        "truncated" if truncated else "terminated",
                    )
                )
                _C = "\033[36m"
                _R = "\033[0m"
                if _rmse_n > 0:
                    _pos_rmse = math.sqrt(_rmse_pos_sq / _rmse_n)
                    _yaw_rmse = math.degrees(math.sqrt(_rmse_yaw_sq / _rmse_n))
                    _rmse_str = (
                        _C + f"pos_rmse={_pos_rmse:.3f}m  yaw_rmse={_yaw_rmse:.2f}deg" + _R
                    )
                else:
                    _rmse_str = _C + "pos_rmse=n/a  yaw_rmse=n/a" + _R
                print(
                    f"  Episode {episode} ended: {reason}"
                    f"  steps={step}  total_steps={total_steps}  "
                    + _rmse_str
                )
                if not _gt_done:
                    print("--- Final observation ---")
                    self._print_obs(obs, step, episode)

                if self._termination_pause > 0:
                    print(
                        f"  Pausing {self._termination_pause:.1f}s "
                        "(Ctrl+C to skip) ..."
                    )
                    pause_end = time.monotonic() + self._termination_pause
                    while time.monotonic() < pause_end:
                        self._env.world.tick()
                        self._update_spectator()
                        time.sleep(1.0 / self._TICK_HZ)

        except KeyboardInterrupt:
            print("\nInterrupted.")
            if _latest_obs is not None:
                print("--- Final observation at interruption ---")
                self._print_obs(_latest_obs, step, episode)
        finally:
            if self._keyboard is not None:
                self._keyboard.stop()

        print(f"\nDry-run complete: {episode} episodes, {total_steps} steps.")
