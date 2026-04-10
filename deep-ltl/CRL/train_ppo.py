# -*- coding: utf-8 -*-
"""
ZoneEnv + PPO-Lagrange training script (train_ppo0.py)

Usage (one line):
PYTHONPATH=src/ python src/training/train_ppo0.py --env ZoneEnv-v0 --exp ppolag_zone --seed 1 --n-envs 8 --total-steps 2000000 --cost-limit 0.005 --lagrangian-lr 0.005 --gamma 0.99 --gae-lambda 0.95 --clip 0.2 --lr 3e-4 --ent-coef 0.005 --vf-coef 0.5 --formula '(!blue U green) & F yellow' --finite
"""
from __future__ import annotations

import os
import time
import argparse
from typing import Callable, Optional, Tuple, Any

import numpy as np
import torch
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
# === Project-local imports (make sure these files存在于src/对应路径) ===
from ppo_lag import PPOLagrangian              # PPO-Lagrange 实现
#from envs.zones.safety_gym_wrapper import SafetyGymWrapper
from safety_gym_wrapper import SafetyGymWrapper
from ldba_wrapper import LDBAWrapper
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from zone_shield_wrapper import ShieldWrapper as ZoneShieldWrapper
from stable_baselines3.common.callbacks import BaseCallback
import csv

# ===== paste this near the top of src/train/train_ppo.py (after imports) =====
#import gymnasium as gym
import safety_gymnasium 

def _ensure_zoneenv_registered():
    """Register ZoneEnv-v0 by pointing it to a valid base env if needed."""
    # already registered?
    try:
        gym.spec("ZoneEnv-v0")
        return
    except Exception:
        pass

    # 1) Prefer your own custom ZoneEnv if it exists
    try:
        from envs.zones.zone_env import ZoneEnv  # your custom class (if present)
        gym.register(id="ZoneEnv-v0", entry_point="envs.zones.zone_env:ZoneEnv")
        print("[INFO] ZoneEnv-v0 registered using your custom envs.zones.zone_env:ZoneEnv")
        return
    except Exception:
        pass

    # 2) Fallback: use an installed Safety-Gymnasium base env
    CANDIDATES = [
        "SafetyPointGoal1-v0",
        "SafetyPointGoal2-v0",
        "SafetyCarGoal1-v0",
        "SafetyCarGoal2-v0",
        "SafetyAntGoal1-v0",
        "SafetyAntGoal2-v0",
    ]
    base_id = None
    for cid in CANDIDATES:
        try:
            gym.spec(cid)
            base_id = cid
            break
        except Exception:
            continue

    if base_id is None:
        # show helpful message with what *is* installed
        available = [s.id for s in gym.registry.values() if "Safety" in s.id or "safety" in s.id]
        raise RuntimeError(
            "No Safety-Gymnasium base env found. Please install safety-gymnasium, e.g.:\n"
            "  pip install safety-gymnasium\n"
            "Then re-run. Currently available matching envs: " + repr(available)
        )

    def _make_base():
        # Ensure base env is created without EnvChecker to avoid 6-tuple check;
        # we normalize signatures in our wrappers instead.
        try:
            return gym.make(base_id, disable_env_checker=True)
        except TypeError:
            return gym.make(base_id)

    gym.register(id="ZoneEnv-v0", entry_point=_make_base)
    print(f"[INFO] ZoneEnv-v0 registered by aliasing to base env: {base_id}")

# call once before building vector envs
#_ensure_zoneenv_registered()
# ===== end of paste =====
# ------------------------ Argparse ------------------------ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("ZoneEnv + PPO/PPO-Lagrange/PPO-Shielding (train_ppo0.py)")
    # Env / run
    p.add_argument("--env", type=str, default="ZoneEnv-v0", help="Gym env id (default: ZoneEnv-v0)")
    p.add_argument("--exp", type=str, default="ppolag_zone", help="experiment name (folder under runs/)")
    p.add_argument("--algorithm", type=str, default="PPOLag", choices=["PPOLag", "PPO", "PPOShield"], help="PPOLag (LTL), PPO (baseline), PPOShield (action shielding)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--log-interval", type=int, default=10_000)
    p.add_argument("--save-dir", type=str, default="runs")

    # LTL / LDBA
    p.add_argument("--formula", type=str, default="(!blue U green) & F yellow")
    p.add_argument("--finite", action="store_true", help="Terminate episode upon acceptance (finite spec)")

    # PPO hyperparams
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=5e-3)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--n-steps", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=128)

    # Lagrange hyperparams
    p.add_argument("--cost-limit", type=float, default=0.005)
    p.add_argument("--lagrangian-lr", type=float, default=5e-3)
    p.add_argument("--cost-gamma", type=float, default=0.99)
    p.add_argument("--cost-gae-lambda", type=float, default=0.95)
    p.add_argument("--normalize-cost-adv", action="store_true")

    # Vec env
    p.add_argument("--sync-env", action="store_true", help="Use SyncVectorEnv instead of AsyncVectorEnv")
    return p.parse_args()


# ------------------------ Env builders ------------------------ #
def make_env_fn(env_id: str, seed: int, formula: str, finite: bool, use_shield: bool = False) -> Callable[[], gym.Env]:
    """
    Returns a thunk to build ONE environment instance:
        Gym(make) -> SafetyGymWrapper -> LDBAWrapper
    """
    class ZeroCostWrapper(gym.Wrapper):
        """Make info['cost']=0.0 to emulate vanilla PPO with PPOLagrangian."""
        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            # FORCE cost to 0.0 (not setdefault, which doesn't overwrite existing values)
            try:
                if isinstance(info, dict):
                    info["cost"] = 0.0
            except Exception:
                pass
            return obs, reward, terminated, truncated, info

        def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Any, dict]:
            obs, info = self.env.reset(seed=seed, options=options)
            try:
                if isinstance(info, dict):
                    info["cost"] = 0.0
            except Exception:
                pass
            return obs, info

    class DictifyObsWrapper(gym.ObservationWrapper):
        """Convert Box observation to Dict({'obs': Box}) for PPOLag buffer compatibility."""
        def __init__(self, env: gym.Env):
            super().__init__(env)
            from gymnasium import spaces
            # 如果已经是字典格式，保持原样；否则包装成 {'obs': ...}
            if isinstance(env.observation_space, spaces.Dict):
                self.observation_space = env.observation_space
            else:
                self.observation_space = spaces.Dict({'obs': env.observation_space})

        def observation(self, observation):
            # 如果已经是字典格式，保持原样；否则包装成 {'obs': ...}
            if isinstance(observation, dict):
                return observation
            else:
                return {'obs': observation}

    class RewardScalarWrapper(gym.Wrapper):
        """Force reward to float scalar for SB3 Monitor compatibility."""
        def step(self, action):
            out = self.env.step(action)
            if len(out) == 5:
                obs, reward, terminated, truncated, info = out
            else:
                # Should not happen here; higher wrappers normalize to 5, but keep fallback
                obs, reward, terminated, truncated, info = out[0], out[1], out[-3], out[-2], out[-1]
            try:
                import numpy as _np
                reward = float(_np.asarray(reward).squeeze().item())
            except Exception:
                reward = float(reward)
            return obs, reward, terminated, truncated, info

    def thunk() -> gym.Env:
        # Disable Gymnasium env checker so we can normalize 6-tuple -> 5-tuple ourselves
        try:
            env = gym.make(env_id, disable_env_checker=True)
        except TypeError:
            env = gym.make(env_id)
        # Wrap DEEPEST base env with a 6->5 adapter so upstream Gym wrappers always see 5-tuple
        class _SixToFiveAdapter(gym.Wrapper):
            def step(self, action):
                out = self.env.step(action)
                try:
                    if len(out) == 6:
                        obs, reward, cost, terminated, truncated, info = out
                        try:
                            if isinstance(info, dict):
                                info.setdefault("cost", float(cost))
                        except Exception:
                            pass
                        return obs, reward, terminated, truncated, info
                except Exception:
                    pass
                return out
        # Inject adapter at deepest .env level (below TimeLimit/OrderEnforcing)
        _parent = None
        _node = env
        while hasattr(_node, "env"):
            _parent = _node
            _node = _node.env
        if _parent is not None:
            _parent.env = _SixToFiveAdapter(_node)
        else:
            env = _SixToFiveAdapter(env)

        env = SafetyGymWrapper(env)
        # lazily decide LTL vs zero-cost by reading closure-captured 'formula' value at higher level
        if formula == "__PPO_BASELINE_NO_LTL__":
            base_env = ZeroCostWrapper(env)
        else:
            base_env = LDBAWrapper(env, formula_str=formula, finite=finite)
        # optional shielding (no LTL cost)
        if use_shield:
            env = ZoneShieldWrapper(base_env)
        else:
            env = base_env
        # Always dictify obs so buffers/policies expect a Dict observation space
        env = DictifyObsWrapper(env)
        # Always scalarize reward for SB3 Monitor
        env = RewardScalarWrapper(env)
        env.reset(seed=seed)
        return env
    return thunk


def make_vec_envs(env_id: str, n_envs: int, seed: int, formula: str, finite: bool, sync_env: bool, use_shield: bool):
    """
    Build SB3-compatible VecEnv. Use DummyVecEnv to ensure compatibility with VecMonitor/logger.
    """
    thunks = [make_env_fn(env_id, seed + i * 131, formula, finite, use_shield) for i in range(n_envs)]
    return DummyVecEnv(thunks)


class EpisodeCSVLogger(BaseCallback):
    """Log per-episode reward and cost into progress.csv under run_dir."""
    def __init__(self, run_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.run_dir = run_dir
        # write to a dedicated episodes file to avoid clashing with SB3 progress.csv
        self.csv_path = os.path.join(run_dir, "episodes.csv")
        self._ep_returns = None
        self._ep_costs = None
        self._ep_lens = None
        self._header_written = False

    def _on_training_start(self) -> None:
        num_envs = self.training_env.num_envs if hasattr(self.training_env, "num_envs") else 1
        self._ep_returns = [0.0] * num_envs
        self._ep_costs = [0.0] * num_envs
        self._ep_lens = [0] * num_envs
        # ensure directory
        os.makedirs(self.run_dir, exist_ok=True)
        # write header
        if not os.path.isfile(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "time/total_timesteps",
                    "episode_reward",
                    "episode_cost",
                    "episode_length",
                    "success",
                    "hit_wall",
                    "lambda",
                ])
            self._header_written = True

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if rewards is None or infos is None or dones is None:
            return True
        num_envs = len(dones)
        # accumulate
        for i in range(num_envs):
            r = float(rewards[i])
            self._ep_returns[i] += r
            c = 0.0
            try:
                info = infos[i] if isinstance(infos, (list, tuple)) else infos
                if isinstance(info, dict):
                    cv = info.get("cost")
                    if cv is not None:
                        c = float(cv)
            except Exception:
                pass
            self._ep_costs[i] += c
            self._ep_lens[i] += 1

        # write when done
        if any(dones):
            with open(self.csv_path, "a", newline="") as f:
                w = csv.writer(f)
                for i, done in enumerate(dones):
                    if done:
                        # success/hit_wall flags from last infos when available
                        info = infos[i] if isinstance(infos, (list, tuple)) else infos
                        success = 1 if (isinstance(info, dict) and bool(info.get("success", False))) else 0
                        hit_wall = 1 if (isinstance(info, dict) and bool(info.get("hit_wall", False))) else 0
                        # current lambda (if PPOLagrangian)
                        lmbda_val = None
                        try:
                            lmbda_val = float(getattr(self.model, "lmbda").item())
                        except Exception:
                            lmbda_val = ""
                        w.writerow([
                            int(self.num_timesteps),
                            self._ep_returns[i],
                            self._ep_costs[i],
                            self._ep_lens[i],
                            success,
                            hit_wall,
                            lmbda_val,
                        ])
                        self._ep_returns[i] = 0.0
                        self._ep_costs[i] = 0.0
                        self._ep_lens[i] = 0
        return True


# ------------------------ Utils ------------------------ #
def select_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


# ------------------------ Main ------------------------ #
def main():
    args = parse_args()

    # Register ZoneEnv-v0 only when requested, and only if not already present
    if args.env == "ZoneEnv-v0":
        try:
            gym.spec("ZoneEnv-v0")
        except Exception:
            _ensure_zoneenv_registered()

    # seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # dirs
    run_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(run_dir, exist_ok=True)

    # device
    device = select_device(args.device)
    print(f"[INFO] Device: {device}")

    # build envs
    algo_name = args.algorithm.upper()
    is_baseline_ppo = (algo_name == "PPO")
    use_shield = (algo_name == "PPOSHIELD")
    # PPOShield 不使用 LTL（与 PPO 一样通过 ZeroCostWrapper 走无约束），仅加动作屏蔽
    formula_for_env = "__PPO_BASELINE_NO_LTL__" if (is_baseline_ppo or use_shield) else args.formula
    if is_baseline_ppo:
        print(f"[INFO] Building {args.n_envs}× {args.env} as PPO baseline (zero-cost).")
    elif use_shield:
        print(f"[INFO] Building {args.n_envs}× {args.env} with Zone Shielding …")
    else:
        print(f"[INFO] Building {args.n_envs}× {args.env} with LDBA(formula='{args.formula}', finite={args.finite}) …")
    vec_env = make_vec_envs(
        env_id=args.env,
        n_envs=args.n_envs,
        seed=args.seed,
        formula=formula_for_env,
        finite=(False if (is_baseline_ppo or use_shield) else args.finite),
        sync_env=args.sync_env,
        use_shield=use_shield,
    )
    # Add VecMonitor to record episode stats and enable CSV/TensorBoard logging
    vec_env = VecMonitor(vec_env)

    # model
    print("[INFO] Creating PPOLagrangian … (PPO baseline if algorithm=PPO; Shielding uses same PPO core)")
    common_kwargs = dict(
        policy="MultiInputPolicy",
        env=vec_env,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        verbose=1,
        device=str(device),
        tensorboard_log=run_dir,
    )
    if is_baseline_ppo:
        # pass only PPO-compatible kwargs
        model = PPOLagrangian(**common_kwargs)
    else:
        # include Lagrange-specific args
        model = PPOLagrangian(
            **common_kwargs,
            cost_limit=args.cost_limit,
            lagrangian_lr=args.lagrangian_lr,
            cost_gamma=args.cost_gamma,
            cost_gae_lambda=args.cost_gae_lambda,
            normalize_cost_adv=args.normalize_cost_adv,
        )

    # train
    total_steps = int(args.total_steps)
    t0 = time.time()
    print(f"[INFO] Start training for {total_steps:,} steps …")
    # Configure SB3 logger to save CSV and TensorBoard files into run_dir
    new_logger = configure(run_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    model.learn(total_timesteps=total_steps, callback=EpisodeCSVLogger(run_dir))

    # save
    save_path = os.path.join(run_dir, f"final_model_{args.total_steps}.zip")
    model.save(save_path)
    dt = time.time() - t0
    print(f"[INFO] Done. Saved to: {save_path}  (time: {dt/60:.1f} min)")


if __name__ == "__main__":
    main()