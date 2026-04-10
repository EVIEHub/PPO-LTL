# src/safety_gym_wrapper.py
from __future__ import annotations
from typing import Any, List, Tuple, Set

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperObsType


class SafetyGymWrapper(gym.Wrapper):
    """
    作用：
      - 在 obs["goal"]["propositions"] 暴露颜色 AP 列表（供 LDBA 构造和一致性检查）
      - 在 info["propositions"] 暴露“当前为真”的颜色命题集合（从 info['cost_zones_*'] 推断）
      - 在 info["hit_wall"] 暴露是否撞墙（从 info['cost_ltl_walls']/info['cost_walls'] 推断）
    注意：
      - 不要在这里做安全惩罚的 reward shaping（让 PPO-Lagrange 通过 cost 约束）
      - 如需“撞墙立即终止”，可在这里把 terminated=True
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.colors: List[str] = []
        # 仅当底层是 Safety* 环境时，启用启发式映射；ZoneEnv 不启用
        try:
            env_id = getattr(getattr(self.env, "spec", None), "id", "") or ""
        except Exception:
            env_id = ""
        self._enable_safetygym_mapping = env_id.startswith("Safety")

    # ---------- 工具 ----------
    @staticmethod
    def _infer_colors_from_info(info: dict) -> List[str]:
        keys = [k for k in info.keys() if k.startswith("cost_zones_")]
        return sorted(k.replace("cost_zones_", "", 1) for k in keys)

    # ---------- Gym API ----------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)

        # 1) 推断颜色 AP 列表（推断失败则用默认兜底）
        if not self.colors:
            inferred = self._infer_colors_from_info(info)
            if not inferred and hasattr(self.env, "colors"):
                try:
                    inferred = list(getattr(self.env, "colors"))
                except Exception:
                    inferred = []
            if not inferred:
                inferred = ["blue", "green", "yellow", "red"]
            self.colors = inferred

        # 2) 确保观察空间是字典格式，并在 obs["goal"]["propositions"] 暴露 AP 列表
        if not isinstance(obs, dict):
            obs = {"obs": obs}
        
        # 确保 goal 字段存在并包含 propositions
        if "goal" not in obs:
            obs["goal"] = {}
        goal = dict(obs["goal"])
        goal["propositions"] = list(self.colors)
        obs["goal"] = goal
        
        # 确保 wall_sensor 字段存在
        if "wall_sensor" not in obs:
            obs["wall_sensor"] = np.array([0, 0, 0, 0], dtype=np.int64)

        # 3) 初始化 info 字段
        info["propositions"] = set()
        info.setdefault("hit_wall", False)
        return obs, info

    def step(self, action):
        out = self.env.step(action)
        # 兼容两种返回格式：5元组和6元组（obs, reward[, cost], terminated, truncated, info）
        if len(out) == 6:
            obs, reward, cost, terminated, truncated, info = out
            # 统一把 cost 写入 info
            try:
                info["cost"] = float(cost)
            except Exception:
                info["cost"] = 0.0
        else:
            obs, reward, terminated, truncated, info = out
            info.setdefault("cost", 0.0)

        # 强制 reward 为标量
        try:
            import numpy as _np
            if not _np.isscalar(reward):
                reward = float(_np.asarray(reward).squeeze().item())
            else:
                reward = float(reward)
        except Exception:
            reward = float(reward)

        # 确保观察空间是字典格式
        if not isinstance(obs, dict):
            obs = {"obs": obs}
        
        # 确保 goal 字段存在并包含 propositions
        if "goal" not in obs:
            obs["goal"] = {}
        goal = dict(obs["goal"])
        goal["propositions"] = list(self.colors)
        obs["goal"] = goal
        
        # 确保 wall_sensor 字段存在
        if "wall_sensor" not in obs:
            obs["wall_sensor"] = np.array([0, 0, 0, 0], dtype=np.int64)

        # (A) 当前为真的颜色命题（进入某色圆区）
        active: Set[str] = set()
        # 优先：原本来自 ZoneEnv 的 cost_zones_* 信号
        if self.colors:
            for c in self.colors:
                if info.get(f"cost_zones_{c}", 0) > 0:
                    active.add(c)
        # 仅对 Safety* 环境启用启发式映射
        if self._enable_safetygym_mapping:
            # - 任意 cost_* > 0 视作 blue（不安全）
            try:
                any_cost = any((k.startswith("cost_") and float(info.get(k, 0)) > 0) for k in info.keys())
            except Exception:
                any_cost = False
            # 明确兼容常见键：cost_hazards / cost_sum
            try:
                any_cost = any_cost or float(info.get("cost_hazards", 0)) > 0 or float(info.get("cost_sum", 0)) > 0
            except Exception:
                pass
            if any_cost:
                active.add("blue")
            # - 成功/接近目标 视作 green（安全达成）
            goal_hit = bool(info.get("success", False) or info.get("goal_met", False))
            goal_close = False
            try:
                gd = info.get("goal_dist", None)
                if gd is not None:
                    goal_close = float(gd) < 0.2
            except Exception:
                goal_close = False
            if goal_hit or goal_close:
                active.add("green")
        info["propositions"] = active

        # (B) 硬安全：撞墙（或通用碰撞）
        hit_wall = bool(
            info.get("cost_ltl_walls", 0) > 0
            or info.get("cost_walls", 0) > 0
            or (self._enable_safetygym_mapping and (
                info.get("cost_collision", 0) > 0 or info.get("collision", 0) > 0
            ))
            or info.get("cost_hazards", 0) > 0
            or info.get("cost_sum", 0) > 0
        )
        info["hit_wall"] = hit_wall
        # 改进：撞墙不立即终止，让agent有更多学习机会
        # 通过cost信号让Lagrangian学习避免撞墙，而不是简单地终止episode
        # if hit_wall:
        #     terminated = True

        # 不要在这里修改 reward（安全惩罚走 cost 通道）
        return obs, reward, terminated, truncated, info

    # ---------- LTL 辅助 ----------
    def get_propositions(self) -> list[str]:
        return sorted(self.colors)

    def get_possible_assignments(self):
        # 若项目里用到了 Assignment，可在此返回 0/1 组合；多数场景不需要。
        props = self.get_propositions()
        return [{p} for p in props]