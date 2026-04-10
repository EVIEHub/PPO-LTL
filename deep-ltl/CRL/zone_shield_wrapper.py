# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class ShieldWrapper(gym.Wrapper):
    """
    A lightweight action-shielding wrapper for ZoneEnv/Safety-Gymnasium-like tasks.

    Policy:
      - Probabilistically applies conservative action clipping to reduce collision risk
      - Uses both deterministic (action norm-based) and stochastic (shield_prob) strategies

    This wrapper is environment-agnostic and makes minimal assumptions:
      - Action space is Box (continuous). The wrapper clips action norm.
    """

    def __init__(
        self,
        env: gym.Env,
        shield_prob: float = 0.3,          # Probability of applying shield
        max_action_norm: float = 0.7,      # Clip large actions to this norm
        damping_factor: float = 0.8,       # Damp actions by this factor when shielding
    ) -> None:
        super().__init__(env)
        self.shield_prob = float(shield_prob)
        self.max_action_norm = float(max_action_norm)
        self.damping_factor = float(damping_factor)
        assert isinstance(self.action_space, gym.spaces.Box), "ShieldWrapper requires continuous actions (Box)."
        self._rng = np.random.RandomState(None)  # Will be seeded on reset
        self._shield_count = 0
        self._total_steps = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Any, Dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        # Reseed RNG if seed provided
        if seed is not None:
            self._rng = np.random.RandomState(seed + 12345)  # Offset to differ from env seed
        self._shield_count = 0
        self._total_steps = 0
        return obs, info

    def _apply_shield(self, action: np.ndarray) -> np.ndarray:
        """Apply conservative action clipping"""
        a = np.asarray(action, dtype=np.float32).copy()
        
        # Strategy 1: Damp the action
        a = self.damping_factor * a
        
        # Strategy 2: Clip action norm
        norm = np.linalg.norm(a)
        if norm > 1e-8 and norm > self.max_action_norm:
            a = a * (self.max_action_norm / norm)
        
        return a

    def step(self, action: Any):
        self._total_steps += 1
        
        # Decide whether to apply shield
        # Two triggers: (1) probabilistic, (2) deterministic if action is too large
        action_array = np.asarray(action, dtype=np.float32)
        action_norm = np.linalg.norm(action_array)
        
        # Apply shield if: (a) random trigger, OR (b) action norm exceeds threshold
        should_shield = (self._rng.random() < self.shield_prob) or (action_norm > 1.0)
        
        safe_action = action
        if should_shield:
            safe_action = self._apply_shield(action_array)
            self._shield_count += 1
        
        out = self.env.step(safe_action)
        # Normalize to Gymnasium 5-tuple
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:
            obs, reward, terminated, truncated, info = out[0], out[1], out[-3], out[-2], out[-1]
        
        # Bookkeeping
        if isinstance(info, dict):
            info["shielded"] = should_shield
            info["shield_rate"] = self._shield_count / max(1, self._total_steps)
        
        return obs, reward, terminated, truncated, info


