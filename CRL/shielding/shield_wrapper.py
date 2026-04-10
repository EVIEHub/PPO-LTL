import gym
import numpy as np
from typing import Optional, Dict, Any


class ShieldWrapper(gym.Wrapper):
    """
    Reactive action shielding for CARLA-like envs (e.g., CarlaRouteEnv).

    Enable by wrapping your env:
        env = ShieldWrapper(env, mode="replace", max_throttle_safe=0.2, ...)

    This does not modify rewards; it only intercepts actions under unsafe conditions.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        mode: str = "replace",           # 'replace' or 'none'
        max_throttle_safe: float = 0.2,   # throttle cap when unsafe
        max_steer_abs_safe: float = 0.4,  # |steer| cap when unsafe
        max_steer_delta: float = 0.15,    # per-step |Δsteer| cap
        overspeed_kmh: float = 30.0,      # km/h threshold considered unsafe
    ) -> None:
        super().__init__(env)
        assert mode in ("replace", "none")
        self.mode = mode
        self.max_throttle_safe = float(max_throttle_safe)
        self.max_steer_abs_safe = float(max_steer_abs_safe)
        self.max_steer_delta = float(max_steer_delta)
        self.overspeed_kmh = float(overspeed_kmh)

        self._last_steer: float = 0.0
        self._last_throttle: float = 0.0

    def reset(self, **kwargs):
        self._last_steer = 0.0
        self._last_throttle = 0.0
        return self.env.reset(**kwargs)

    def _is_unsafe(self, last_info: Optional[Dict[str, Any]]) -> (bool, str):
        if not isinstance(last_info, dict):
            return False, ""

        reasons = []
        if bool(last_info.get("collision_state", False)):
            reasons.append("collision")
        try:
            avg_dev = float(last_info.get("avg_center_dev", 0.0))
            if avg_dev > 1.5:
                reasons.append("center_deviation")
        except Exception:
            pass
        try:
            speed = float(last_info.get("speed", 0.0))
            if speed > self.overspeed_kmh:
                reasons.append("overspeed")
        except Exception:
            pass
        try:
            if float(last_info.get("collision_rate", 0.0)) > 0.1:
                reasons.append("recent_collisions")
        except Exception:
            pass

        return (len(reasons) > 0), ",".join(reasons)

    def _apply_shield(self, action: np.ndarray) -> np.ndarray:
        try:
            steer = float(action[0])
            throttle = float(action[1])
        except Exception:
            return action

        delta = np.clip(steer - self._last_steer, -self.max_steer_delta, self.max_steer_delta)
        steer = self._last_steer + delta
        steer *= 0.7
        steer = float(np.clip(steer, -self.max_steer_abs_safe, self.max_steer_abs_safe))
        throttle = float(np.clip(throttle, 0.0, self.max_throttle_safe))
        return np.array([steer, throttle], dtype=np.float32)

    def step(self, action):
        orig_action = action
        unsafe = False
        reason = ""

        if abs(self._last_steer) > self.max_steer_abs_safe * 0.9:
            unsafe = True
            reason = (reason + "," if reason else "") + "high_steer"

        last_info = getattr(self.env, "_last_info", None)
        is_unsafe, rs = self._is_unsafe(last_info)
        if is_unsafe:
            unsafe = True
            reason = (reason + "," if reason else "") + rs

        new_action = orig_action
        if unsafe and self.mode == "replace":
            try:
                new_action = self._apply_shield(np.asarray(orig_action, dtype=np.float32))
            except Exception:
                new_action = orig_action

        out = self.env.step(new_action)
        # Support both Gym(4) and Gymnasium(5) style returns, and forward same arity
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:
            obs, reward, done, info = out
            terminated, truncated = bool(done), False

        try:
            self._last_steer = float(new_action[0])
            self._last_throttle = float(new_action[1])
        except Exception:
            pass

        try:
            setattr(self.env, "_last_info", info)
        except Exception:
            pass

        if isinstance(info, dict):
            info["shield/activated"] = bool(unsafe and self.mode == "replace")
            info["shield/mode"] = self.mode
            if reason:
                info["shield/reason"] = reason
            try:
                info["shield/orig_action"] = np.asarray(orig_action).tolist()
                info["shield/new_action"] = np.asarray(new_action).tolist()
            except Exception:
                pass

        # Return with same arity as underlying env
        if isinstance(out, tuple) and len(out) == 5:
            return obs, reward, terminated, truncated, info
        else:
            return obs, reward, bool(terminated or truncated), info


