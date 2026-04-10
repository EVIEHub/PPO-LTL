# src/envs/ldba_wrapper.py
from __future__ import annotations
from typing import Any, Tuple, Set, List

import re
import gymnasium as gym

# 项目里的 LTL->LDBA 构造函数（保持原路径/导入）
# 如果原本有 joblib.memory 的 cache，照旧保留
from ltl.automata import ltl2ldba


# ---- 从公式里提取原子命题名（排除操作符） ----
_ATOM_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_LTL_OPS = {"X","F","G","U","R","W","true","false","and","or","not","AND","OR","NOT"}
def _parse_atoms(formula: str) -> List[str]:
    tokens = set(_ATOM_RE.findall(formula))
    return sorted(t for t in tokens if t not in _LTL_OPS)


class LDBAWrapper(gym.Wrapper):
    """
    职责：
      - reset: 读取 obs['goal']['propositions']，做“公式-AP 一致性检查”，然后构造 LDBA
      - step: 依据 info['propositions'] 与当前 LTL 步的 A^- + info['hit_wall'] 合成 info['cost']
    可以把“自动机状态推进/奖励设计/complete_observation”等逻辑继续放在这里（下面留接口）。
    """

    def __init__(self, env: gym.Env, *, formula_str: str, finite: bool = False,
                 w_logic: float = 1.0, w_wall: float = 1.0):
        super().__init__(env)
        self.formula_str = formula_str
        self.finite = bool(finite)
        # 成本权重（论文默认 1, 1）
        self.w_logic = float(w_logic)
        self.w_wall = float(w_wall)

        # 运行时状态
        self.propositions: List[str] = []
        self.ldba = None
        self.terminate_on_acceptance = False
        self.ldba_state = None
        self.num_accepting_visits = 0

    # ---------- 原来用于补观测/推进自动机的接口（按需保留/实现） ----------
    def complete_observation(self, obs, info):
        # 如项目已有具体实现，可覆盖/替换此方法
        pass

    def advance_automaton(self, active_props: Set[str]):
        """
        根据 active_props 推进自动机状态。
        兼容多种常见 LDBA 接口：优先使用 ldba.step(state, label_set)；
        其次尝试 ldba.delta(state, label_set) / ldba.transition(...)/ ldba.get_next_state(...)/ ldba.next(...)
        若找不到接口，则保持原状态（使逻辑成本仅依赖撞墙）。
        """
        if self.ldba is None:
            return
        label = set(active_props) if active_props else set()

        candidates = [
            "step", "delta", "transition", "get_next_state", "next"
        ]
        next_state = None
        for name in candidates:
            fn = getattr(self.ldba, name, None)
            if callable(fn):
                try:
                    # 常见签名：(state, label_set)
                    next_state = fn(self.ldba_state, label)
                except TypeError:
                    try:
                        # 备选签名：(label_set, state)
                        next_state = fn(label, self.ldba_state)
                    except Exception:
                        next_state = None
                except Exception:
                    next_state = None
            if next_state is not None:
                break

        if next_state is not None:
            # 接受状态统计（若接口可用）
            prev_state = self.ldba_state
            self.ldba_state = next_state
            try:
                in_accepting = False
                if hasattr(self.ldba, "is_accepting") and callable(getattr(self.ldba, "is_accepting")):
                    in_accepting = bool(self.ldba.is_accepting(self.ldba_state))
                elif hasattr(self.ldba, "accepting_states"):
                    in_accepting = self.ldba_state in getattr(self.ldba, "accepting_states")
                if in_accepting:
                    self.num_accepting_visits += 1
            except Exception:
                pass
            # 标记自动机状态变化（便于日志）
            if prev_state is not next_state:
                # 在 info 中标注由 step() 设置，这里只更新内部标志位
                setattr(self, "_ldba_state_changed_step", True)
        else:
            # 找不到可用接口：保持原状态
            setattr(self, "_ldba_state_changed_step", False)

    # ---------- 当前 LTL 步的 A^-（负命题集合们） ----------
    def _current_negative_sets(self) -> List[Set[str]]:
        """
        返回当前 LTL 步的 A^-：list[set[str]]
        例：[{'yellow'}, {'blue','red'}] 表示：避免 yellow 或避免同时出现 blue & red。
        优先尝试从 LDBA 暴露的接口读取；若无可用信息，则返回空（仅保留撞墙成本）。
        """
        if self.ldba is None or self.ldba_state is None:
            return []

        # 常见约定：
        # 1) ldba.get_negative_sets(state) -> Iterable[Iterable[str]]
        # 2) state.negative_sets / state.A_neg / state.forbidden / state.forbidden_sets
        # 3) ldba.labels.get(state, {}).get("A_neg", ...)
        try:
            fn = getattr(self.ldba, "get_negative_sets", None)
            if callable(fn):
                neg = fn(self.ldba_state)
                return [set(s) for s in neg] if neg is not None else []
        except Exception:
            pass

        # 尝试从状态对象读属性
        for attr in ("negative_sets", "A_neg", "forbidden", "forbidden_sets"):
            try:
                neg = getattr(self.ldba_state, attr, None)
                if neg:
                    return [set(s) for s in neg]
            except Exception:
                continue

        # 尝试从 ldba 元数据字典读取
        try:
            labels = getattr(self.ldba, "labels", None)
            if isinstance(labels, dict):
                meta = labels.get(self.ldba_state, {})
                for key in ("A_neg", "negative_sets", "forbidden", "forbidden_sets"):
                    if key in meta and meta[key]:
                        return [set(s) for s in meta[key]]
        except Exception:
            pass

        return []

    # ---------- Gym API ----------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Tuple[Any, dict]:
        obs, info = self.env.reset(seed=seed, options=options)

        # 1) 读取 AP 列表（ZoneEnv: 颜色）
        props = (obs.get("goal", {}) or {}).get("propositions", None) if isinstance(obs, dict) else None
        if not props:
            raise AssertionError("Env must provide obs['goal']['propositions'] (list[str]).")
        self.propositions = list(props)

        # 2) 公式-AP 一致性检查（避免 HOA 晦涩报错）
        atoms = _parse_atoms(self.formula_str)
        missing = [a for a in atoms if a not in self.propositions]
        if missing:
            raise ValueError(
                f"Formula uses AP {missing} not in env propositions {self.propositions}. "
                f"For ZoneEnv, APs must be color names from this list."
            )

        # 3) 构造 LDBA（保持你原有的缓存/实现）
        self.ldba = ltl2ldba(self.formula_str, self.propositions, simplify_labels=False)

        # 4) 初始化自动机运行状态（按你项目逻辑）
        #    下面两行是典型写法；若你的 LDBA 接口不同，请替换为等价调用
        self.terminate_on_acceptance = getattr(self.ldba, "is_finite_specification", lambda: self.finite)()
        self.ldba_state = getattr(self.ldba, "initial_state", None)
        self.num_accepting_visits = 0

        # 5) 可选：补全观测
        self.complete_observation(obs, info)

        # 6) 标志：本回合自动机状态刷新
        info["ldba_state_changed"] = True
        info.setdefault("cost", 0.0)
        info.setdefault("cost_logic", 0.0)
        info.setdefault("cost_wall", 0.0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # A) 当前为真的命题集合
        active = info.get("propositions", set())
        if not isinstance(active, (set, frozenset)):
            try:
                active = set(active)
            except Exception:
                active = set()

        # B) 可选：推进自动机
        setattr(self, "_ldba_state_changed_step", False)
        self.advance_automaton(active)

        # C) 判定 A^- 违例
        neg_sets = self._current_negative_sets()
        violated_neg = any(neg.issubset(active) for neg in neg_sets)

        # D) 硬安全：撞墙
        hit_wall = bool(info.get("hit_wall", False))

        # E) 合成成本（含权重与分量日志）
        c_logic = 1.0 if violated_neg else 0.0
        c_wall = 1.0 if hit_wall else 0.0
        cost = self.w_logic * c_logic + self.w_wall * c_wall
        info["cost_logic"] = float(c_logic)
        info["cost_wall"] = float(c_wall)
        info["cost"] = float(cost)

        # F) 标记自动机状态变化（供上游记录）
        if getattr(self, "_ldba_state_changed_step", False):
            info["ldba_state_changed"] = True

        # G) 可选：若在 finite 规格下需要到达接受就终止，可在此你项目逻辑终止
        # if self.terminate_on_acceptance and self._in_accepting_state():
        #     terminated = True

        # H) 可选：补全观测
        self.complete_observation(obs, info)

        return obs, reward, terminated, truncated, info