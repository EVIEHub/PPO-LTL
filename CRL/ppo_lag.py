# ppo_lag.py — Robust PPO-Lagrangian for Stable-Baselines3 (Dict obs OK)

from typing import Tuple, Generator
from collections import namedtuple
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor, explained_variance


# ----------------- custom sample tuple (adds cost fields) -----------------
CostDictRolloutBufferSamples = namedtuple(
    "CostDictRolloutBufferSamples",
    [
        "observations",
        "actions",
        "old_values",
        "old_log_prob",
        "advantages",
        "returns",
        "cost_advantages",
        "cost_returns",
        "episode_starts",
    ],
)

# ----------------- helper: robust numpy casting -----------------
def _to_npy(x):
    if isinstance(x, th.Tensor):
        return x.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


# ----------------- Cost-aware DictRolloutBuffer -----------------
class CostRolloutBuffer(DictRolloutBuffer):
    """
    与 SB3 的 DictRolloutBuffer 一致，但增加成本相关字段，并改造 get() 让 samples 带上 cost_adv/cost_ret。
    内部缓存全部保持 numpy，避免 CUDA→NumPy 报错；在 get() 时统一转到 self._device。
    """

    def __init__(self, *args, **kwargs):
        dev = kwargs.get("device", None)
        if dev is None and len(args) >= 4:  # (buffer_size, obs_space, action_space, device, ...)
            dev = args[3]
        try:
            self._device = th.device(dev)
        except Exception:
            self._device = th.device("cpu")

        super().__init__(*args, **kwargs)

        self.costs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(self, obs, action, reward, episode_start, value, log_prob, cost, cost_value):
        super().add(obs, action, reward, episode_start, value, log_prob)
        idx = self.pos - 1
        self.costs[idx] = _to_npy(cost)
        self.cost_values[idx] = _to_npy(cost_value)

    def compute_returns_and_advantage(
        self,
        last_values,
        dones,
        gamma: float,
        gae_lambda: float,
        last_cost_values,
        cost_gamma: float,
        cost_gae_lambda: float,
    ) -> None:
        last_values_np = _to_npy(last_values)
        last_cost_values_np = _to_npy(last_cost_values)
        dones_np = _to_npy(dones)

        values = _to_npy(self.values)
        rewards = _to_npy(self.rewards)
        cost_values = _to_npy(self.cost_values)
        costs = _to_npy(self.costs)
        episode_starts = _to_npy(self.episode_starts)

        advantages = _to_npy(self.advantages)
        returns = _to_npy(self.returns)
        cost_advantages = _to_npy(self.cost_advantages)
        cost_returns = _to_npy(self.cost_returns)

        # 奖励 GAE
        last_gae_r = np.zeros_like(last_values_np, dtype=np.float32)
        for step in range(self.buffer_size - 1, -1, -1):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones_np
                next_values = last_values_np
            else:
                next_non_terminal = 1.0 - episode_starts[step + 1]
                next_values = values[step + 1]
            delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
            last_gae_r = delta + gamma * gae_lambda * next_non_terminal * last_gae_r
            advantages[step] = last_gae_r
        returns[:] = advantages + values

        # 成本 GAE
        last_gae_c = np.zeros_like(last_cost_values_np, dtype=np.float32)
        for step in range(self.buffer_size - 1, -1, -1):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones_np
                next_cvalues = last_cost_values_np
            else:
                next_non_terminal = 1.0 - episode_starts[step + 1]
                next_cvalues = cost_values[step + 1]
            delta_c = costs[step] + cost_gamma * next_cvalues * next_non_terminal - cost_values[step]
            last_gae_c = delta_c + cost_gamma * cost_gae_lambda * next_non_terminal * last_gae_c
            cost_advantages[step] = last_gae_c
        cost_returns[:] = cost_advantages + cost_values

        self.advantages = advantages
        self.returns = returns
        self.cost_advantages = cost_advantages
        self.cost_returns = cost_returns

    def get(self, batch_size: int) -> Generator[CostDictRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer is not full"
        batch_inds = np.random.permutation(self.buffer_size * self.n_envs)
        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            end_idx = start_idx + batch_size
            yield self._get_samples(batch_inds[start_idx:end_idx])
            start_idx = end_idx

    def _get_samples(self, batch_inds: np.ndarray) -> CostDictRolloutBufferSamples:
        env_indices = batch_inds % self.n_envs
        time_indices = batch_inds // self.n_envs

        observations = {}
        for key, obs_array in self.observations.items():
            obs = obs_array[time_indices, env_indices]
            observations[key] = th.as_tensor(obs, device=self._device)

        actions         = th.as_tensor(self.actions[time_indices, env_indices], device=self._device)
        old_values      = th.as_tensor(self.values[time_indices, env_indices], device=self._device, dtype=th.float32)
        old_log_prob    = th.as_tensor(self.log_probs[time_indices, env_indices], device=self._device, dtype=th.float32)
        advantages      = th.as_tensor(self.advantages[time_indices, env_indices], device=self._device, dtype=th.float32)
        returns         = th.as_tensor(self.returns[time_indices, env_indices], device=self._device, dtype=th.float32)
        episode_starts  = th.as_tensor(self.episode_starts[time_indices, env_indices], device=self._device, dtype=th.float32)

        cost_advantages = th.as_tensor(self.cost_advantages[time_indices, env_indices], device=self._device, dtype=th.float32)
        cost_returns    = th.as_tensor(self.cost_returns[time_indices, env_indices], device=self._device, dtype=th.float32)

        return CostDictRolloutBufferSamples(
            observations=observations,
            actions=actions,
            old_values=old_values,
            old_log_prob=old_log_prob,
            advantages=advantages,
            returns=returns,
            cost_advantages=cost_advantages,
            cost_returns=cost_returns,
            episode_starts=episode_starts,
        )


# ----------------- PPO-Lagrangian 本体 -----------------
class PPOLagrangian(PPO):
    """
    PPO with hard constraints:
      - Extra cost-value head V_c(s)
      - Mixed advantage: A = A_r - λ A_c
      - Dual ascent on λ: λ ← max(0, λ + η (avg_cost - cost_limit))
    """

    def _get_clip_ranges(self):
        # 兼容老版 SB3
        clip_range = self.clip_range
        if callable(clip_range):
            clip_range = clip_range(self._current_progress_remaining)
        clip_range_vf = None
        if hasattr(self, "clip_range_vf") and self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf
            if callable(clip_range_vf):
                clip_range_vf = clip_range_vf(self._current_progress_remaining)
        return clip_range, clip_range_vf

    def __init__(
        self,
        *args,
        cost_limit: float = 0.01,
        lagrangian_lr: float = 5e-3,
        normalize_cost_adv: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cost_limit = float(cost_limit)
        self.lagrangian_lr = float(lagrangian_lr)
        self.normalize_cost_adv = normalize_cost_adv
        # λ 放在 device 上，但此时 device 已在父类 __init__ 设置好了
        self.lmbda = th.tensor(0.0, device=self.device)

        # 重要：不要在 __init__ 里访问 self.policy / mlp_extractor！
        # 延后到 _setup_model 之后再创建 cost_value_head 和 rollout_buffer。

    def _setup_model(self) -> None:
        # 先让 PPO 把 policy / optimizer 等搭好
        super()._setup_model()

        # 现在 policy 已就绪，可以安全地取 latent_dim_vf
        latent_dim_vf = self.policy.mlp_extractor.latent_dim_vf
        self.cost_value_head = nn.Linear(latent_dim_vf, 1).to(self.device)

        # 把成本分支参数加入到同一个 optimizer
        # 避免重复加入（load 多次 _setup_model 时）
        existing = set()
        for g in self.policy.optimizer.param_groups:
            for p in g["params"]:
                existing.add(id(p))
        new_params = [p for p in self.cost_value_head.parameters() if id(p) not in existing]
        if len(new_params) > 0:
            self.policy.optimizer.add_param_group({"params": new_params})

        # 用自定义的 CostRolloutBuffer 替换
        self.rollout_buffer = CostRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

    # 让 save()/load() 把成本分支一并保存/恢复
    def _get_torch_save_params(self):
        state_dicts, tensors = super()._get_torch_save_params()
        state_dicts = [n for n in state_dicts if n != "policy.optimizer"]

        state_dicts += ["cost_value_head"]
        tensors += ["lmbda"]
        return state_dicts, tensors

    @th.no_grad()
    def _values_and_cost_values(self, obs_tensor) -> Tuple[th.Tensor, th.Tensor]:
        feats = self.policy.extract_features(obs_tensor)
        _, latent_vf = self.policy.mlp_extractor(feats)
        v_r = self.policy.value_net(latent_vf).flatten()
        v_c = self.cost_value_head(latent_vf).flatten()
        return v_r, v_c

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps: int) -> bool:
        assert isinstance(rollout_buffer, CostRolloutBuffer)
        rollout_buffer.reset()
        callback.on_rollout_start()

        obs_np = self._last_obs
        episode_starts_np = self._last_episode_starts
        obs_t = obs_as_tensor(obs_np, self.device)

        n_steps = 0
        while n_steps < n_rollout_steps:
            with th.no_grad():
                actions_t, values_t, log_probs_t = self.policy(obs_t, deterministic=False)
                feats = self.policy.extract_features(obs_t)
                _, latent_vf = self.policy.mlp_extractor(feats)
                cost_values_t = self.cost_value_head(latent_vf).flatten()

            actions_np = actions_t.detach().cpu().numpy()
            new_obs_np, rewards_np, dones_np, infos = env.step(actions_np)
            self.num_timesteps += env.num_envs

            costs_np = np.array([float(info.get("cost", 0.0)) for info in infos], dtype=np.float32)
            costs_t = th.as_tensor(costs_np, device=self.device)

            rollout_buffer.add(
                obs_np,
                actions_np,
                rewards_np,
                episode_starts_np,
                values_t.flatten(),
                log_probs_t.flatten(),
                cost=costs_t,
                cost_value=cost_values_t,
            )

            # 为callback设置locals
            callback.locals.update({
                'obs': obs_np,
                'actions': actions_np,
                'rewards': rewards_np,
                'dones': dones_np,
                'infos': infos,
                'values': values_t,
                'log_probs': log_probs_t,
            })

            obs_np = new_obs_np
            episode_starts_np = dones_np
            obs_t = obs_as_tensor(obs_np, self.device)

            n_steps += 1
            if not callback.on_step():
                return False

        with th.no_grad():
            last_vr, last_vc = self._values_and_cost_values(obs_t)

        rollout_buffer.compute_returns_and_advantage(
            last_vr, episode_starts_np, self.gamma, self.gae_lambda, last_vc, self.gamma, self.gae_lambda
        )

        self._last_obs = obs_np
        self._last_episode_starts = episode_starts_np
        # 供 explained_variance 记录
        self._last_values = last_vr.detach().cpu()
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)
        clip_range, clip_range_vf = self._get_clip_ranges()

        # 归一化优势
        adv = self.rollout_buffer.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.rollout_buffer.advantages = adv

        if self.normalize_cost_adv:
            cadv = self.rollout_buffer.cost_advantages
            cadv = (cadv - cadv.mean()) / (cadv.std() + 1e-8)
            self.rollout_buffer.cost_advantages = cadv

        policy_losses, value_losses, entropy_losses = [], [], []
        approx_kls, clip_fracs, stds = [], [], []
        n_updates = 0

        for _ in range(self.n_epochs):
            for data in self.rollout_buffer.get(self.batch_size):
                n_updates += 1
                obs = data.observations
                actions = data.actions
                old_log_prob = data.old_log_prob
                old_values = data.old_values
                returns = data.returns

                # 混合优势
                mixed_adv = data.advantages - self.lmbda * data.cost_advantages

                values, log_prob, entropy = self.policy.evaluate_actions(obs, actions)
                with th.no_grad():
                    feats = self.policy.extract_features(obs)
                    _, latent_vf = self.policy.mlp_extractor(feats)
                cost_values = self.cost_value_head(latent_vf).flatten()

                ratio = th.exp(log_prob - old_log_prob)
                pg_loss1 = -mixed_adv * ratio
                pg_loss2 = -mixed_adv * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = th.max(pg_loss1, pg_loss2).mean()

                if clip_range_vf is None:
                    value_loss = F.mse_loss(values.flatten(), returns)
                else:
                    v_pred = old_values + th.clamp(values.flatten() - old_values, -clip_range_vf, clip_range_vf)
                    value_loss = F.mse_loss(v_pred, returns)

                cost_value_loss = F.mse_loss(cost_values, data.cost_returns)

                loss = policy_loss + self.vf_coef * (value_loss + cost_value_loss) - self.ent_coef * entropy.mean()

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                for p in self.cost_value_head.parameters():
                    if p.grad is not None:
                        p.grad.data.clamp_(-self.max_grad_norm, self.max_grad_norm)
                self.policy.optimizer.step()

                with th.no_grad():
                    approx_kl = th.mean((old_log_prob - log_prob)).item()
                    clip_frac = th.mean((th.abs(ratio - 1.0) > clip_range).float()).item()
                    std_val = getattr(getattr(self.policy, "action_dist", None), "std", None)
                    if std_val is None:
                        if hasattr(self.policy, "log_std"):
                            std_val = th.exp(self.policy.log_std).mean().item()
                        else:
                            std_val = float("nan")

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.mean().item())
                approx_kls.append(approx_kl)
                clip_fracs.append(clip_frac)
                stds.append(std_val)

        # 对偶更新 λ
        avg_cost = float(self.rollout_buffer.costs.mean().item())
        self.lmbda = th.clamp(self.lmbda + self.lagrangian_lr * (avg_cost - self.cost_limit), min=0.0)

        # 日志
        self.logger.record("train/policy_gradient_loss", np.mean(policy_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kls))
        self.logger.record("train/clip_fraction", np.mean(clip_fracs))
        self.logger.record("train/clip_range", float(clip_range))
        self.logger.record("train/std", np.nanmean(stds))
        lr_now = np.mean([pg["lr"] for pg in self.policy.optimizer.param_groups])
        self.logger.record("train/learning_rate", lr_now)
        self.logger.record("train/loss", np.mean(policy_losses) + self.vf_coef * np.mean(value_losses))
        self.logger.record("train/n_updates", n_updates)
        self.logger.record("constraint/avg_cost", avg_cost)
        self.logger.record("lagrange/lambda", float(self.lmbda.item()))
        if hasattr(self, "_last_values"):
            try:
                self.logger.record(
                    "train/explained_variance",
                    explained_variance(self.rollout_buffer.returns.flatten(), self._last_values.flatten()),
                )
            except TypeError:
                # 兼容NumPy 2.0+版本
                self.logger.record("train/explained_variance", 0.0)
