import torch as th
from box import Box
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from utils import lr_schedule

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
import torch.nn as nn
import gymnasium as gym
import torch


class CustomCNN(nn.Module):
    def __init__(self, input_shape, features_dim=1):
        super(CustomCNN, self).__init__()
        n_input_channels = input_shape[0]

        if n_input_channels == 3:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 16, kernel_size=5, stride=2),  # (16, 58, 38)
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (32, 28, 18)
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (64, 13, 8)
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2),  # (128, 6, 4)
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (256, 4, 2)
                nn.ReLU(),
                nn.Flatten(),
            )
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *input_shape)).view(-1).shape[0]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, x):
        x = self.cnn(x)
        x = self.linear(x)
        return x


class CustomMultiInputExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super(CustomMultiInputExtractor, self).__init__(observation_space, features_dim)
        extractors = {}
        total_concat_size = 0

        if isinstance(observation_space, gym.spaces.Dict):
            for key, subspace in observation_space.spaces.items():
                if key == "seg_camera":
                    extractors[key] = CustomCNN(subspace.shape, features_dim=features_dim)
                    total_concat_size += features_dim
                else:
                    extractors[key] = nn.Flatten()
                    total_concat_size += get_flattened_obs_dim(subspace)
        else:
            extractors["default"] = CustomCNN(observation_space.shape, features_dim=features_dim)
            total_concat_size = features_dim

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        if isinstance(observations, dict):
            for key, extractor in self.extractors.items():
                encoded_tensor_list.append(extractor(observations[key]))
        else:
            encoded_tensor_list.append(self.extractors["default"](observations))
        return torch.cat(encoded_tensor_list, dim=1)


algorithm_params = {
    "PPO": dict(
        device="cuda:0",
        learning_rate=lr_schedule(1e-4, 1e-6, 2),
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        n_epochs=10,
        n_steps=1024,
        policy_kwargs=dict(activation_fn=th.nn.ReLU,
                           net_arch=[dict(pi=[500, 300], vf=[500, 300])],
                           features_extractor_class=CustomMultiInputExtractor,
                           features_extractor_kwargs=dict(features_dim=256),
                           )
    ),
    "SAC": dict(
        device="cuda:0",
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        buffer_size=10000,
        batch_size=256,
        ent_coef='auto',
        gamma=0.98,
        tau=0.02,
        train_freq=64,
        gradient_steps=64,
        learning_starts=10000,
        use_sde=True,
        policy_kwargs=dict(log_std_init=-3, net_arch=[400, 300]),
    ),
    "DDPG": dict(
        device="cuda:0",
        gamma=0.98,
        buffer_size=10000,
        learning_starts=10000,
        action_noise=NormalActionNoise(mean=np.zeros(2), sigma=0.5 * np.ones(2)),
        gradient_steps=-1,
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        policy_kwargs=dict(net_arch=[400, 300]),
    ),
    "SAC_CLIP": dict(
        device="cuda:0",
        learning_rate=lr_schedule(1e-4, 5e-7, 2),
        buffer_size=10000,
        batch_size=256,
        ent_coef='auto',
        gamma=0.98,
        tau=0.02,
        train_freq=64,
        gradient_steps=64,
        learning_starts=10000,
        use_sde=True,
        policy_kwargs=dict(
            log_std_init=-3, net_arch=[500, 300],
            features_extractor_class=CustomMultiInputExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )
    ),
}

states = {
    "1": ["steer", "throttle", "speed", "angle_next_waypoint", "maneuver"],
    "2": ["steer", "throttle", "speed", "maneuver"],
    "3": ["steer", "throttle", "speed", "waypoints"],
    "4": ["steer", "throttle", "speed", "angle_next_waypoint", "maneuver", "distance_goal"],
    "5": ["steer", "throttle", "speed", "waypoints", "seg_camera"],
}

reward_params = {
    "reward_fn_5_default": dict(
        early_stop=True,
        min_speed=20.0,  # km/h
        max_speed=35.0,  # km/h
        target_speed=25.0,  # kmh
        max_distance=3.0,  # Max distance from center before terminating
        max_std_center_lane=0.4,
        max_angle_center_lane=90,
        penalty_reward=-10,
    ),
    "reward_fn_5_no_early_stop": dict(
        early_stop=False,
        min_speed=20.0,  # km/h
        max_speed=35.0,  # km/h
        target_speed=25.0,  # kmh
        max_distance=3.0,  # Max distance from center before terminating
        max_std_center_lane=0.4,
        max_angle_center_lane=90,
        penalty_reward=-10,
    ),
    "reward_fn_5_best": dict(
        early_stop=True,
        min_speed=20.0,  # km/h
        max_speed=35.0,  # km/h
        target_speed=25.0,  # kmh
        max_distance=2.0,  # Max distance from center before terminating
        max_std_center_lane=0.35,
        max_angle_center_lane=90,
        penalty_reward=-10,
    ),
    "reward_clg": dict(
        pretrained_model="ViT-bigG-14/laion2b_s39b_b160k",
        batch_size=64,
        target_prompts=[
            "Two cars have collided with each other on the road",
            "The road is clear with no car accidents",
        ],
    ),
    "reward_lord": dict(
        pretrained_model="ViT-bigG-14/laion2b_s39b_b160k",
        batch_size=64,
        target_prompts=[
            "Two cars have collided with each other on the road",
        ],
    ),
    "reward_vlm_rm": dict(
        pretrained_model="ViT-bigG-14/laion2b_s39b_b160k",
        batch_size=64,
        alpha=0.5,
        target_prompts=[
            "A car is driving safely",
        ],
        baseline_prompts=[
            "A car",
        ],
    ),
    "reward_fn_Chen": dict(
        early_stop=True,
        min_speed=0.0,
        max_speed=28.8,
        target_speed=25.0,
        max_distance=4.0,
        max_std_center_lane=0.4,
        max_angle_center_lane=90,
        penalty_reward=-10,
    ),
    "reward_fn_ASAP": dict(
        early_stop=True,
        min_speed=0.0,
        max_speed=50.0,
        target_speed=30.0,
        max_distance=3.0,
        max_std_center_lane=0.4,
        max_angle_center_lane=90,
        penalty_reward=-5,
    ),
}

_CONFIG_1 = {
    "algorithm": "PPO",
    "algorithm_params": algorithm_params["PPO"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_rgb_bev": False,
}

_CONFIG_2 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_rgb_bev": False,
}

_CONFIG_vlm_rl = {
    "algorithm": "CLIP-SAC",
    "algorithm_params": algorithm_params["SAC_CLIP"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "clip_reward_params": reward_params["reward_clg"],
    "vlm_reward_type": "VLM-RL",
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "action_noise": {},
    "use_seg_bev": True,
    "use_rgb_bev": True,
}

_CONFIG_vlm_rl_ppo = {
    "algorithm": "CLIP-PPO",
    "algorithm_params": algorithm_params["PPO"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "clip_reward_params": reward_params["reward_clg"],
    "vlm_reward_type": "VLM-RL",
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "action_noise": {},
    "action_space_type": "discrete",
    "use_seg_bev": True,
    "use_rgb_bev": True,
}

_CONFIG_lord = {
    "algorithm": "CLIP-SAC",
    "algorithm_params": algorithm_params["SAC_CLIP"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "clip_reward_params": reward_params["reward_lord"],
    "vlm_reward_type": "LORD",
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "action_noise": {},
    "use_seg_bev": False,
    "use_rgb_bev": True,
}

_CONFIG_lord_speed = {
    "algorithm": "CLIP-SAC",
    "algorithm_params": algorithm_params["SAC_CLIP"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "clip_reward_params": reward_params["reward_lord"],
    "vlm_reward_type": "LORD-Speed",
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "action_noise": {},
    "use_seg_bev": False,
    "use_rgb_bev": True,
}

_CONFIG_vlm_rm = {
    "algorithm": "CLIP-SAC",
    "algorithm_params": algorithm_params["SAC_CLIP"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "clip_reward_params": reward_params["reward_vlm_rm"],
    "vlm_reward_type": "VLM-RM",
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "action_noise": {},
    "use_seg_bev": False,
    "use_rgb_bev": True,
}

_CONFIG_vlm_sr = {
    "algorithm": "CLIP-SAC",
    "algorithm_params": algorithm_params["SAC_CLIP"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "clip_reward_params": reward_params["reward_vlm_rm"],
    "vlm_reward_type": "VLM-SR",
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "action_noise": {},
    "use_seg_bev": False,
    "use_rgb_bev": True,
}

_CONFIG_roboclip = {
    "algorithm": "CLIP-SAC",
    "algorithm_params": algorithm_params["SAC_CLIP"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "clip_reward_params": reward_params["reward_vlm_rm"],
    "vlm_reward_type": "RoboCLIP",
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "action_noise": {},
    "use_seg_bev": False,
    "use_rgb_bev": True,
}

_CONFIG_tirl_sac = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_rgb_bev": False,
}

_CONFIG_tirl_ppo = {
    "algorithm": "PPO",
    "algorithm_params": algorithm_params["PPO"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_rgb_bev": False,
}

# ===== TIRL-PPO with 3 seeds =====
_CONFIG_tirl_ppo_seed1 = {
    "algorithm": "PPO",
    "algorithm_params": algorithm_params["PPO"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_rgb_bev": False,
}

_CONFIG_tirl_ppo_seed2 = {
    "algorithm": "PPO",
    "algorithm_params": algorithm_params["PPO"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 200,
    "wrappers": [],
    "use_rgb_bev": False,
}

_CONFIG_tirl_ppo_seed3 = {
    "algorithm": "PPO",
    "algorithm_params": algorithm_params["PPO"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 300,
    "wrappers": [],
    "use_rgb_bev": False,
}

# ===== TIRL-SAC with 3 seeds =====
_CONFIG_tirl_sac_seed1 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_rgb_bev": False,
}

_CONFIG_tirl_sac_seed2 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 200,
    "wrappers": [],
    "use_rgb_bev": False,
}

_CONFIG_tirl_sac_seed3 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 300,
    "wrappers": [],
    "use_rgb_bev": False,
}

_CONFIG_crl_ppo_hard = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 0.02,
        "lagrangian_lr": 0.05,       # λ 的对偶步长
        #"cost_limit": 0.02,
        #"lagrangian_lr": 1e-3,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
}
_CONFIG_chatscene_sac = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_chatscene",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_rgb_bev": False,
}

_CONFIG_chatscene_ppo = {
    "algorithm": "PPO",
    "algorithm_params": algorithm_params["PPO"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_chatscene",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_rgb_bev": False,
}

_CONFIG_revolve = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_revolve",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_rgb_bev": False,
}

_CONFIG_revolve_auto = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_revolve_auto",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_rgb_bev": False,
}

_CONFIG_Chen = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["5"],
    "vae_model": None,
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_Chen",
    "reward_params": reward_params["reward_fn_Chen"],
    "obs_res": (80, 120),
    "seed": 120,
    "wrappers": [],
    "use_rgb_bev": False,
}

_CONFIG_ASAP = {
    "algorithm": "PPO",
    "algorithm_params": algorithm_params["PPO"],
    "state": states["5"],
    "vae_model": None,
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_ASAP",
    "reward_params": reward_params["reward_fn_ASAP"],
    "obs_res": (80, 120),
    "seed": 120,
    "wrappers": [],
    "use_rgb_bev": False,
}


_CONFIG_crl_ppo_A = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 0.02,
        "lagrangian_lr": 0.001,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
}

_CONFIG_crl_ppo_B = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 0.05,
        "lagrangian_lr": 0.05,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
}

_CONFIG_crl_ppo_collision = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 0.02,
        "lagrangian_lr": 0.01,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
    # 碰撞-only 权重
    "cost_weights": {
        "collision": 1.0,
        "off_track": 0.0,
        "lane_invasion": 0.0,
        "heading": 0.0,
        "weaving": 0.0,
        "overspeed": 0.0,
        "steer_jerk": 0.0,
    },
}

# 3个种子的配置
_CONFIG_crl_ppo_A_seed1 = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 0.02,
        "lagrangian_lr": 0.001,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
}

_CONFIG_crl_ppo_A_seed2 = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 0.02,
        "lagrangian_lr": 0.001,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 200,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
}

_CONFIG_crl_ppo_A_seed3 = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 0.02,
        "lagrangian_lr": 0.001,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 300,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
}

_CONFIG_crl_ppo_B_seed1 = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 0.05,
        "lagrangian_lr": 0.05,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
}

_CONFIG_crl_ppo_B_seed2 = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 0.05,
        "lagrangian_lr": 0.05,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 200,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
}

_CONFIG_crl_ppo_B_seed3 = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 0.05,
        "lagrangian_lr": 0.05,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 300,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
}

_CONFIG_crl_ppo_collision_seed1 = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 0.02,
        "lagrangian_lr": 0.01,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
    "cost_weights": {
        "collision": 1.0,
        "off_track": 0.0,
        "lane_invasion": 0.0,
        "heading": 0.0,
        "weaving": 0.0,
        "overspeed": 0.0,
        "steer_jerk": 0.0,
    },
}

_CONFIG_crl_ppo_collision_seed2 = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 0.02,
        "lagrangian_lr": 0.01,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 200,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
    "cost_weights": {
        "collision": 1.0,
        "off_track": 0.0,
        "lane_invasion": 0.0,
        "heading": 0.0,
        "weaving": 0.0,
        "overspeed": 0.0,
        "steer_jerk": 0.0,
    },
}

_CONFIG_crl_ppo_collision_seed3 = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 0.02,
        "lagrangian_lr": 0.01,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 300,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
    "cost_weights": {
        "collision": 1.0,
        "off_track": 0.0,
        "lane_invasion": 0.0,
        "heading": 0.0,
        "weaving": 0.0,
        "overspeed": 0.0,
        "steer_jerk": 0.0,
    },
}

# Ablation Study 配置 - 逐个关闭cost components
_CONFIG_crl_ppo_ablation_no_collision = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 0.02,
        "lagrangian_lr": 0.001,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
    "cost_weights": {
        "collision": 0.0,  # 关闭collision
        "off_track": 0.5,
        "lane_invasion": 0.2,
        "heading": 0.2,
        "weaving": 0.2,
        "overspeed": 0.2,
        "steer_jerk": 0.1,
    },
}

_CONFIG_crl_ppo_ablation_no_offtrack = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 0.02,
        "lagrangian_lr": 0.001,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
    "cost_weights": {
        "collision": 1.0,
        "off_track": 0.0,  # 关闭off_track
        "lane_invasion": 0.2,
        "heading": 0.2,
        "weaving": 0.2,
        "overspeed": 0.2,
        "steer_jerk": 0.1,
    },
}

_CONFIG_crl_ppo_ablation_no_laneinv = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 0.02,
        "lagrangian_lr": 0.001,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
    "cost_weights": {
        "collision": 1.0,
        "off_track": 0.5,
        "lane_invasion": 0.0,  # 关闭lane_invasion
        "heading": 0.2,
        "weaving": 0.2,
        "overspeed": 0.2,
        "steer_jerk": 0.1,
    },
}

# Sensitivity Analysis 配置 - 测试不同超参数
_CONFIG_crl_ppo_sens_cost_001 = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 0.01,  # 更严格的cost limit
        "lagrangian_lr": 0.001,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
}

_CONFIG_crl_ppo_sens_cost_01 = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 0.1,  # 更宽松的cost limit
        "lagrangian_lr": 0.001,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
}

_CONFIG_crl_ppo_sens_lag_0001 = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 0.02,
        "lagrangian_lr": 0.0001,  # 更小的lagrangian learning rate
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
}

_CONFIG_crl_ppo_sens_lag_01 = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 0.02,
        "lagrangian_lr": 0.01,  # 更大的lagrangian learning rate
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
}

# 紧急修复配置 - 基于失败分析
_CONFIG_crl_ppo_emergency_loose = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 2.0,  # 从0.02增加到2.0 (100倍!)
        "lagrangian_lr": 0.0001,  # 从0.05降到0.0001
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
    "cost_weights": {
        "collision": 0.1,  # 大幅降低权重
        "off_track": 0.05,
        "lane_invasion": 0.02,
        "heading": 0.02,
        "weaving": 0.02,
        "overspeed": 0.02,
        "steer_jerk": 0.01,
    }
}

_CONFIG_crl_ppo_emergency_medium = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 1.0,  # 从0.02增加到1.0 (50倍)
        "lagrangian_lr": 0.0005,  # 从0.05降到0.0005
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
    "cost_weights": {
        "collision": 0.2,
        "off_track": 0.1,
        "lane_invasion": 0.05,
        "heading": 0.05,
        "weaving": 0.05,
        "overspeed": 0.05,
        "steer_jerk": 0.02,
    }
}

_CONFIG_crl_ppo_emergency_progressive = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 0.5,  # 从0.02增加到0.5 (25倍)
        "lagrangian_lr": 0.001,  # 从0.05降到0.001
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
    "cost_weights": {
        "collision": 0.5,  # 只保留碰撞约束
        "off_track": 0.0,  # 暂时关闭其他约束
        "lane_invasion": 0.0,
        "heading": 0.0,
        "weaving": 0.0,
        "overspeed": 0.0,
        "steer_jerk": 0.0,
    }
}

# 扩展参数搜索配置 - 基于seed 2/3的PPO成功经验
_CONFIG_crl_ppo_ultra_loose = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 5.0,  # 超级宽松
        "lagrangian_lr": 0.00001,  # 超级小的学习率
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 200,  # 基于seed 2的成功经验
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
    "cost_weights": {
        "collision": 0.01,  # 极低权重
        "off_track": 0.005,
        "lane_invasion": 0.002,
        "heading": 0.002,
        "weaving": 0.002,
        "overspeed": 0.002,
        "steer_jerk": 0.001,
    }
}

_CONFIG_crl_ppo_lr_00001 = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 1.0,
        "lagrangian_lr": 0.00001,  # 极小的学习率
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 200,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
    "cost_weights": {
        "collision": 0.1,
        "off_track": 0.05,
        "lane_invasion": 0.02,
        "heading": 0.02,
        "weaving": 0.02,
        "overspeed": 0.02,
        "steer_jerk": 0.01,
    }
}

_CONFIG_crl_ppo_lr_0001 = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 1.0,
        "lagrangian_lr": 0.0001,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 200,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
    "cost_weights": {
        "collision": 0.1,
        "off_track": 0.05,
        "lane_invasion": 0.02,
        "heading": 0.02,
        "weaving": 0.02,
        "overspeed": 0.02,
        "steer_jerk": 0.01,
    }
}

_CONFIG_crl_ppo_lr_001 = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 1.0,
        "lagrangian_lr": 0.001,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 200,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
    "cost_weights": {
        "collision": 0.1,
        "off_track": 0.05,
        "lane_invasion": 0.02,
        "heading": 0.02,
        "weaving": 0.02,
        "overspeed": 0.02,
        "steer_jerk": 0.01,
    }
}

_CONFIG_crl_ppo_cost_01 = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 0.1,
        "lagrangian_lr": 0.001,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 200,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
    "cost_weights": {
        "collision": 0.1,
        "off_track": 0.05,
        "lane_invasion": 0.02,
        "heading": 0.02,
        "weaving": 0.02,
        "overspeed": 0.02,
        "steer_jerk": 0.01,
    }
}

_CONFIG_crl_ppo_cost_05 = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 0.5,
        "lagrangian_lr": 0.001,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 200,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
    "cost_weights": {
        "collision": 0.1,
        "off_track": 0.05,
        "lane_invasion": 0.02,
        "heading": 0.02,
        "weaving": 0.02,
        "overspeed": 0.02,
        "steer_jerk": 0.01,
    }
}

_CONFIG_crl_ppo_cost_2 = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 2.0,
        "lagrangian_lr": 0.001,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 200,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
    "cost_weights": {
        "collision": 0.1,
        "off_track": 0.05,
        "lane_invasion": 0.02,
        "heading": 0.02,
        "weaving": 0.02,
        "overspeed": 0.02,
        "steer_jerk": 0.01,
    }
}

_CONFIG_crl_ppo_collision_only_01 = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 0.1,
        "lagrangian_lr": 0.001,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 200,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
    "cost_weights": {
        "collision": 1.0,  # 只保留碰撞
        "off_track": 0.0,
        "lane_invasion": 0.0,
        "heading": 0.0,
        "weaving": 0.0,
        "overspeed": 0.0,
        "steer_jerk": 0.0,
    }
}

_CONFIG_crl_ppo_collision_only_05 = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 0.5,
        "lagrangian_lr": 0.001,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 200,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
    "cost_weights": {
        "collision": 1.0,
        "off_track": 0.0,
        "lane_invasion": 0.0,
        "heading": 0.0,
        "weaving": 0.0,
        "overspeed": 0.0,
        "steer_jerk": 0.0,
    }
}

_CONFIG_crl_ppo_mixed_light = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 1.0,
        "lagrangian_lr": 0.001,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 200,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
    "cost_weights": {
        "collision": 0.5,  # 碰撞权重较高
        "off_track": 0.1,  # 其他约束较轻
        "lane_invasion": 0.05,
        "heading": 0.0,    # 关闭一些约束
        "weaving": 0.0,
        "overspeed": 0.0,
        "steer_jerk": 0.0,
    }
}

_CONFIG_crl_ppo_mixed_medium = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 1.0,
        "lagrangian_lr": 0.001,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 200,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
    "cost_weights": {
        "collision": 0.3,
        "off_track": 0.2,
        "lane_invasion": 0.1,
        "heading": 0.05,
        "weaving": 0.05,
        "overspeed": 0.0,  # 关闭超速约束
        "steer_jerk": 0.0,
    }
}



# ===== Ablation Study 配置 - seed2和seed3版本 =====
_CONFIG_crl_ppo_ablation_no_collision_seed2 = {
    **_CONFIG_crl_ppo_ablation_no_collision,
    "seed": 200,
}

_CONFIG_crl_ppo_ablation_no_collision_seed3 = {
    **_CONFIG_crl_ppo_ablation_no_collision,
    "seed": 300,
}

_CONFIG_crl_ppo_ablation_no_offtrack_seed2 = {
    **_CONFIG_crl_ppo_ablation_no_offtrack,
    "seed": 200,
}

_CONFIG_crl_ppo_ablation_no_offtrack_seed3 = {
    **_CONFIG_crl_ppo_ablation_no_offtrack,
    "seed": 300,
}

_CONFIG_crl_ppo_ablation_no_laneinv_seed2 = {
    **_CONFIG_crl_ppo_ablation_no_laneinv,
    "seed": 200,
}

_CONFIG_crl_ppo_ablation_no_laneinv_seed3 = {
    **_CONFIG_crl_ppo_ablation_no_laneinv,
    "seed": 300,
}

# ===== Sensitivity Analysis 配置 - seed2和seed3版本 =====
_CONFIG_crl_ppo_sens_cost_001_seed2 = {
    **_CONFIG_crl_ppo_sens_cost_001,
    "seed": 200,
}

_CONFIG_crl_ppo_sens_cost_001_seed3 = {
    **_CONFIG_crl_ppo_sens_cost_001,
    "seed": 300,
}

_CONFIG_crl_ppo_sens_cost_01_seed2 = {
    **_CONFIG_crl_ppo_sens_cost_01,
    "seed": 200,
}

_CONFIG_crl_ppo_sens_cost_01_seed3 = {
    **_CONFIG_crl_ppo_sens_cost_01,
    "seed": 300,
}

_CONFIG_crl_ppo_sens_lag_0001_seed2 = {
    **_CONFIG_crl_ppo_sens_lag_0001,
    "seed": 200,
}

_CONFIG_crl_ppo_sens_lag_0001_seed3 = {
    **_CONFIG_crl_ppo_sens_lag_0001,
    "seed": 300,
}

_CONFIG_crl_ppo_sens_lag_01_seed2 = {
    **_CONFIG_crl_ppo_sens_lag_01,
    "seed": 200,
}

_CONFIG_crl_ppo_sens_lag_01_seed3 = {
    **_CONFIG_crl_ppo_sens_lag_01,
    "seed": 300,
}

# ===== 紧急修复配置 - seed2和seed3版本 =====
_CONFIG_crl_ppo_emergency_loose_seed2 = {
    **_CONFIG_crl_ppo_emergency_loose,
    "seed": 200,
}

_CONFIG_crl_ppo_emergency_loose_seed3 = {
    **_CONFIG_crl_ppo_emergency_loose,
    "seed": 300,
}

_CONFIG_crl_ppo_emergency_medium_seed2 = {
    **_CONFIG_crl_ppo_emergency_medium,
    "seed": 200,
}

_CONFIG_crl_ppo_emergency_medium_seed3 = {
    **_CONFIG_crl_ppo_emergency_medium,
    "seed": 300,
}

_CONFIG_crl_ppo_emergency_progressive_seed2 = {
    **_CONFIG_crl_ppo_emergency_progressive,
    "seed": 200,
}

_CONFIG_crl_ppo_emergency_progressive_seed3 = {
    **_CONFIG_crl_ppo_emergency_progressive,
    "seed": 300,
}

# ===== 扩展参数搜索配置 - seed3版本 =====
_CONFIG_crl_ppo_ultra_loose_seed3 = {
    **_CONFIG_crl_ppo_ultra_loose,
    "seed": 300,
}

_CONFIG_crl_ppo_lr_00001_seed3 = {
    **_CONFIG_crl_ppo_lr_00001,
    "seed": 300,
}

_CONFIG_crl_ppo_cost_05_seed3 = {
    **_CONFIG_crl_ppo_cost_05,
    "seed": 300,
}

_CONFIG_crl_ppo_mixed_light_seed3 = {
    **_CONFIG_crl_ppo_mixed_light,
    "seed": 300,
}

_CONFIG_crl_ppo_lr_0001_seed3 = {
    **_CONFIG_crl_ppo_lr_0001,
    "seed": 300,
}

_CONFIG_crl_ppo_collision_only_01_seed3 = {
    **_CONFIG_crl_ppo_collision_only_01,
    "seed": 300,
}

_CONFIG_crl_ppo_collision_only_05_seed3 = {
    **_CONFIG_crl_ppo_collision_only_05,
    "seed": 300,
}

_CONFIG_crl_ppo_mixed_medium_seed3 = {
    **_CONFIG_crl_ppo_mixed_medium,
    "seed": 300,
}


# ===== Seed1 clones for extended-search configs (seed=100) =====
_CONFIG_crl_ppo_ultra_loose_seed1 = {
    **_CONFIG_crl_ppo_ultra_loose,
    "seed": 100,
}

_CONFIG_crl_ppo_lr_00001_seed1 = {
    **_CONFIG_crl_ppo_lr_00001,
    "seed": 100,
}

_CONFIG_crl_ppo_cost_05_seed1 = {
    **_CONFIG_crl_ppo_cost_05,
    "seed": 100,
}

_CONFIG_crl_ppo_mixed_light_seed1 = {
    **_CONFIG_crl_ppo_mixed_light,
    "seed": 100,
}

_CONFIG_crl_ppo_lr_0001_seed1 = {
    **_CONFIG_crl_ppo_lr_0001,
    "seed": 100,
}

_CONFIG_crl_ppo_collision_only_01_seed1 = {
    **_CONFIG_crl_ppo_collision_only_01,
    "seed": 100,
}

_CONFIG_crl_ppo_collision_only_05_seed1 = {
    **_CONFIG_crl_ppo_collision_only_05,
    "seed": 100,
}

_CONFIG_crl_ppo_mixed_medium_seed1 = {
    **_CONFIG_crl_ppo_mixed_medium,
    "seed": 100,
}

# ===== Improved configs based on observed results (seed=100) =====
_CONFIG_crl_ppo_collision_loose_seed1 = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 2.0,
        "lagrangian_lr": 0.000005,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
    "cost_weights": {
        "collision": 1.0,
        "off_track": 0.0,
        "lane_invasion": 0.0,
        "heading": 0.0,
        "weaving": 0.0,
        "overspeed": 0.0,
        "steer_jerk": 0.0,
    },
}

_CONFIG_crl_ppo_collision_mid_seed1 = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 1.0,
        "lagrangian_lr": 0.00001,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
    "cost_weights": {
        "collision": 1.0,
        "off_track": 0.0,
        "lane_invasion": 0.0,
        "heading": 0.0,
        "weaving": 0.0,
        "overspeed": 0.0,
        "steer_jerk": 0.0,
    },
}

_CONFIG_crl_ppo_mixed_light_loose_seed1 = {
    "algorithm": "PPOLag",
    "algorithm_params": {
        **algorithm_params["PPO"],
        "cost_limit": 2.0,
        "lagrangian_lr": 0.00001,
    },
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
    "cost_weights": {
        "collision": 0.7,
        "off_track": 0.1,
        "lane_invasion": 0.05,
        "heading": 0.0,
        "weaving": 0.0,
        "overspeed": 0.0,
        "steer_jerk": 0.0,
    },
}

# ===== Improved variants 配置 - seed2和seed3版本 =====
_CONFIG_crl_ppo_collision_loose_seed2 = {
    **_CONFIG_crl_ppo_collision_loose_seed1,
    "seed": 200,
}

_CONFIG_crl_ppo_collision_loose_seed3 = {
    **_CONFIG_crl_ppo_collision_loose_seed1,
    "seed": 300,
}

_CONFIG_crl_ppo_collision_mid_seed2 = {
    **_CONFIG_crl_ppo_collision_mid_seed1,
    "seed": 200,
}

_CONFIG_crl_ppo_collision_mid_seed3 = {
    **_CONFIG_crl_ppo_collision_mid_seed1,
    "seed": 300,
}

_CONFIG_crl_ppo_mixed_light_loose_seed2 = {
    **_CONFIG_crl_ppo_mixed_light_loose_seed1,
    "seed": 200,
}

_CONFIG_crl_ppo_mixed_light_loose_seed3 = {
    **_CONFIG_crl_ppo_mixed_light_loose_seed1,
    "seed": 300,
}
# Vanilla PPO (CRL) - 3 seeds
_CONFIG_crl_ppo_vanilla_seed1 = {
    "algorithm": "PPO",
    "algorithm_params": algorithm_params["PPO"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
}

_CONFIG_crl_ppo_vanilla_seed2 = {
    "algorithm": "PPO",
    "algorithm_params": algorithm_params["PPO"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 200,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
}

_CONFIG_crl_ppo_vanilla_seed3 = {
    "algorithm": "PPO",
    "algorithm_params": algorithm_params["PPO"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 300,
    "wrappers": [],
    "use_seg_bev": False,
    "use_rgb_bev": False,
}

# ===== PPO with Table-based Shielding (3 seeds) =====
# Uses discrete action space + runtime lookup table for safe action filtering.
# Run `python make_dummy_shield.py` in CRL/ first to generate shield_table.pkl.
_CONFIG_crl_ppo_table_shield_seed1 = {
    "algorithm": "PPO",
    "algorithm_params": algorithm_params["PPO"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "action_space_type": "discrete",
    "wrappers": [
        "TableShieldWrapper_shield_path=shield_table.pkl",
    ],
    "use_seg_bev": False,
    "use_rgb_bev": False,
}

_CONFIG_crl_ppo_table_shield_seed2 = {
    **_CONFIG_crl_ppo_table_shield_seed1,
    "seed": 200,
}

_CONFIG_crl_ppo_table_shield_seed3 = {
    **_CONFIG_crl_ppo_table_shield_seed1,
    "seed": 300,
}

# ===== PPO with Shielding (3 seeds) =====
_CONFIG_crl_ppo_shield_seed1 = {
    "algorithm": "PPO",
    "algorithm_params": algorithm_params["PPO"],
    "state": states["5"],
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_simple",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (80, 120),
    "seed": 100,
    "wrappers": [
        "ShieldWrapper_mode=replace;max_throttle_safe=0.2;max_steer_abs_safe=0.4;max_steer_delta=0.15;overspeed_kmh=30.0",
    ],
    "use_seg_bev": False,
    "use_rgb_bev": False,
}

_CONFIG_crl_ppo_shield_seed2 = {
    **_CONFIG_crl_ppo_shield_seed1,
    "seed": 200,
}

_CONFIG_crl_ppo_shield_seed3 = {
    **_CONFIG_crl_ppo_shield_seed1,
    "seed": 300,
}

CONFIGS = {
    "1": _CONFIG_1,
    "2": _CONFIG_2,
    "vlm_rl": _CONFIG_vlm_rl,
    "vlm_rl_ppo": _CONFIG_vlm_rl_ppo,
    "lord": _CONFIG_lord,
    "lord_speed": _CONFIG_lord_speed,
    "vlm_rm": _CONFIG_vlm_rm,
    "vlm_sr": _CONFIG_vlm_sr,
    "roboclip": _CONFIG_roboclip,
    "tirl_sac": _CONFIG_tirl_sac,
    "tirl_ppo": _CONFIG_tirl_ppo,
    "tirl_ppo_seed1": _CONFIG_tirl_ppo_seed1,
    "tirl_ppo_seed2": _CONFIG_tirl_ppo_seed2,
    "tirl_ppo_seed3": _CONFIG_tirl_ppo_seed3,
    "tirl_sac_seed1": _CONFIG_tirl_sac_seed1,
    "tirl_sac_seed2": _CONFIG_tirl_sac_seed2,
    "tirl_sac_seed3": _CONFIG_tirl_sac_seed3,
    "chatscene_sac": _CONFIG_chatscene_sac,
    "chatscene_ppo": _CONFIG_chatscene_ppo,
    "revolve": _CONFIG_revolve,
    "revolve_auto": _CONFIG_revolve_auto,
    "Chen": _CONFIG_Chen,
    "ASAP": _CONFIG_ASAP,
    "crl_ppo_hard": _CONFIG_crl_ppo_hard,
    "crl_ppo_A": _CONFIG_crl_ppo_A,
    "crl_ppo_B": _CONFIG_crl_ppo_B,
    "crl_ppo_collision": _CONFIG_crl_ppo_collision,
    # 3个种子的配置
    "crl_ppo_A_seed1": _CONFIG_crl_ppo_A_seed1,
    "crl_ppo_A_seed2": _CONFIG_crl_ppo_A_seed2,
    "crl_ppo_A_seed3": _CONFIG_crl_ppo_A_seed3,
    "crl_ppo_B_seed1": _CONFIG_crl_ppo_B_seed1,
    "crl_ppo_B_seed2": _CONFIG_crl_ppo_B_seed2,
    "crl_ppo_B_seed3": _CONFIG_crl_ppo_B_seed3,
    "crl_ppo_collision_seed1": _CONFIG_crl_ppo_collision_seed1,
    "crl_ppo_collision_seed2": _CONFIG_crl_ppo_collision_seed2,
    "crl_ppo_collision_seed3": _CONFIG_crl_ppo_collision_seed3,
    # Ablation Study 配置
    "crl_ppo_ablation_no_collision": _CONFIG_crl_ppo_ablation_no_collision,
    "crl_ppo_ablation_no_offtrack": _CONFIG_crl_ppo_ablation_no_offtrack,
    "crl_ppo_ablation_no_laneinv": _CONFIG_crl_ppo_ablation_no_laneinv,
    # Ablation Study 配置 - seed2和seed3版本
    "crl_ppo_ablation_no_collision_seed2": _CONFIG_crl_ppo_ablation_no_collision_seed2,
    "crl_ppo_ablation_no_collision_seed3": _CONFIG_crl_ppo_ablation_no_collision_seed3,
    "crl_ppo_ablation_no_offtrack_seed2": _CONFIG_crl_ppo_ablation_no_offtrack_seed2,
    "crl_ppo_ablation_no_offtrack_seed3": _CONFIG_crl_ppo_ablation_no_offtrack_seed3,
    "crl_ppo_ablation_no_laneinv_seed2": _CONFIG_crl_ppo_ablation_no_laneinv_seed2,
    "crl_ppo_ablation_no_laneinv_seed3": _CONFIG_crl_ppo_ablation_no_laneinv_seed3,
    # Sensitivity Analysis 配置
    "crl_ppo_sens_cost_001": _CONFIG_crl_ppo_sens_cost_001,
    "crl_ppo_sens_cost_01": _CONFIG_crl_ppo_sens_cost_01,
    "crl_ppo_sens_lag_0001": _CONFIG_crl_ppo_sens_lag_0001,
    "crl_ppo_sens_lag_01": _CONFIG_crl_ppo_sens_lag_01,
    # Sensitivity Analysis 配置 - seed2和seed3版本
    "crl_ppo_sens_cost_001_seed2": _CONFIG_crl_ppo_sens_cost_001_seed2,
    "crl_ppo_sens_cost_001_seed3": _CONFIG_crl_ppo_sens_cost_001_seed3,
    "crl_ppo_sens_cost_01_seed2": _CONFIG_crl_ppo_sens_cost_01_seed2,
    "crl_ppo_sens_cost_01_seed3": _CONFIG_crl_ppo_sens_cost_01_seed3,
    "crl_ppo_sens_lag_0001_seed2": _CONFIG_crl_ppo_sens_lag_0001_seed2,
    "crl_ppo_sens_lag_0001_seed3": _CONFIG_crl_ppo_sens_lag_0001_seed3,
    "crl_ppo_sens_lag_01_seed2": _CONFIG_crl_ppo_sens_lag_01_seed2,
    "crl_ppo_sens_lag_01_seed3": _CONFIG_crl_ppo_sens_lag_01_seed3,
    # 紧急修复配置
    "crl_ppo_emergency_loose": _CONFIG_crl_ppo_emergency_loose,
    "crl_ppo_emergency_medium": _CONFIG_crl_ppo_emergency_medium,
    "crl_ppo_emergency_progressive": _CONFIG_crl_ppo_emergency_progressive,
    # 紧急修复配置 - seed2和seed3版本
    "crl_ppo_emergency_loose_seed2": _CONFIG_crl_ppo_emergency_loose_seed2,
    "crl_ppo_emergency_loose_seed3": _CONFIG_crl_ppo_emergency_loose_seed3,
    "crl_ppo_emergency_medium_seed2": _CONFIG_crl_ppo_emergency_medium_seed2,
    "crl_ppo_emergency_medium_seed3": _CONFIG_crl_ppo_emergency_medium_seed3,
    "crl_ppo_emergency_progressive_seed2": _CONFIG_crl_ppo_emergency_progressive_seed2,
    "crl_ppo_emergency_progressive_seed3": _CONFIG_crl_ppo_emergency_progressive_seed3,
    # 扩展参数搜索配置
    "crl_ppo_ultra_loose": _CONFIG_crl_ppo_ultra_loose,
    "crl_ppo_lr_00001": _CONFIG_crl_ppo_lr_00001,
    "crl_ppo_lr_0001": _CONFIG_crl_ppo_lr_0001,
    "crl_ppo_lr_001": _CONFIG_crl_ppo_lr_001,
    "crl_ppo_cost_01": _CONFIG_crl_ppo_cost_01,
    "crl_ppo_cost_05": _CONFIG_crl_ppo_cost_05,
    "crl_ppo_cost_2": _CONFIG_crl_ppo_cost_2,
    "crl_ppo_collision_only_01": _CONFIG_crl_ppo_collision_only_01,
    "crl_ppo_collision_only_05": _CONFIG_crl_ppo_collision_only_05,
    "crl_ppo_mixed_light": _CONFIG_crl_ppo_mixed_light,
    "crl_ppo_mixed_medium": _CONFIG_crl_ppo_mixed_medium,
    # Seed1 clones for extended-search
    "crl_ppo_ultra_loose_seed1": _CONFIG_crl_ppo_ultra_loose_seed1,
    "crl_ppo_lr_00001_seed1": _CONFIG_crl_ppo_lr_00001_seed1,
    "crl_ppo_cost_05_seed1": _CONFIG_crl_ppo_cost_05_seed1,
    "crl_ppo_mixed_light_seed1": _CONFIG_crl_ppo_mixed_light_seed1,
    "crl_ppo_lr_0001_seed1": _CONFIG_crl_ppo_lr_0001_seed1,
    "crl_ppo_collision_only_01_seed1": _CONFIG_crl_ppo_collision_only_01_seed1,
    "crl_ppo_collision_only_05_seed1": _CONFIG_crl_ppo_collision_only_05_seed1,
    "crl_ppo_mixed_medium_seed1": _CONFIG_crl_ppo_mixed_medium_seed1,
    # 扩展参数搜索配置 - seed3版本
    "crl_ppo_ultra_loose_seed3": _CONFIG_crl_ppo_ultra_loose_seed3,
    "crl_ppo_lr_00001_seed3": _CONFIG_crl_ppo_lr_00001_seed3,
    "crl_ppo_cost_05_seed3": _CONFIG_crl_ppo_cost_05_seed3,
    "crl_ppo_mixed_light_seed3": _CONFIG_crl_ppo_mixed_light_seed3,
    "crl_ppo_lr_0001_seed3": _CONFIG_crl_ppo_lr_0001_seed3,
    "crl_ppo_collision_only_01_seed3": _CONFIG_crl_ppo_collision_only_01_seed3,
    "crl_ppo_collision_only_05_seed3": _CONFIG_crl_ppo_collision_only_05_seed3,
    "crl_ppo_mixed_medium_seed3": _CONFIG_crl_ppo_mixed_medium_seed3,
    # Improved seed1 variants
    "crl_ppo_collision_loose_seed1": _CONFIG_crl_ppo_collision_loose_seed1,
    "crl_ppo_collision_mid_seed1": _CONFIG_crl_ppo_collision_mid_seed1,
    "crl_ppo_mixed_light_loose_seed1": _CONFIG_crl_ppo_mixed_light_loose_seed1,
    # Improved variants 配置 - seed2和seed3版本
    "crl_ppo_collision_loose_seed2": _CONFIG_crl_ppo_collision_loose_seed2,
    "crl_ppo_collision_loose_seed3": _CONFIG_crl_ppo_collision_loose_seed3,
    "crl_ppo_collision_mid_seed2": _CONFIG_crl_ppo_collision_mid_seed2,
    "crl_ppo_collision_mid_seed3": _CONFIG_crl_ppo_collision_mid_seed3,
    "crl_ppo_mixed_light_loose_seed2": _CONFIG_crl_ppo_mixed_light_loose_seed2,
    "crl_ppo_mixed_light_loose_seed3": _CONFIG_crl_ppo_mixed_light_loose_seed3,
    # Vanilla PPO seeds
    "crl_ppo_vanilla_seed1": _CONFIG_crl_ppo_vanilla_seed1,
    "crl_ppo_vanilla_seed2": _CONFIG_crl_ppo_vanilla_seed2,
    "crl_ppo_vanilla_seed3": _CONFIG_crl_ppo_vanilla_seed3,
    # PPO + Shielding (old continuous action clipping)
    "crl_ppo_shield_seed1": _CONFIG_crl_ppo_shield_seed1,
    "crl_ppo_shield_seed2": _CONFIG_crl_ppo_shield_seed2,
    "crl_ppo_shield_seed3": _CONFIG_crl_ppo_shield_seed3,
    # PPO + Table-based Shielding (discrete, lookup table)
    "crl_ppo_table_shield_seed1": _CONFIG_crl_ppo_table_shield_seed1,
    "crl_ppo_table_shield_seed2": _CONFIG_crl_ppo_table_shield_seed2,
    "crl_ppo_table_shield_seed3": _CONFIG_crl_ppo_table_shield_seed3,
}

CONFIG = None


def set_config(config_name):
    global CONFIG
    CONFIG = Box(CONFIGS[config_name], default_box=True)
    return CONFIG
