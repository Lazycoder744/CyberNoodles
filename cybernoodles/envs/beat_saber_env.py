from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from cybernoodles.core.gpu_simulator import GPUBeatSaberSimulator
from cybernoodles.core.network import ACTION_DIM, INPUT_DIM
from cybernoodles.envs.presets import SimulatorTuning, apply_simulator_tuning

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    gym = None
    spaces = None


def _tensor_to_cpu_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _info_value_at_index(value, index):
    if isinstance(value, torch.Tensor):
        item = value.detach().cpu()
        if item.ndim > 0:
            item = item[index]
        return item.item() if item.ndim == 0 else item.numpy()
    if isinstance(value, np.ndarray):
        item = value
        if item.ndim > 0:
            item = item[index]
        return item.item() if np.ndim(item) == 0 else item
    return value


@dataclass
class BeatSaberEnvConfig:
    num_envs: int
    device: str = "cuda"
    penalty_weights: tuple = (0.5, 0.0, 0.0, 0.0)
    dense_reward_scale: float = 0.0
    training_wheels: float = 0.0
    rehab_assists: float = 0.0
    survival_assistance: float = 0.0
    stability_assistance: float = 0.0
    style_guidance_level: float = 0.0
    hit_timing_profile: str = "default"
    saber_inertia: float = 0.0
    rot_clamp: float = 0.07
    pos_clamp: float = 0.12
    fail_enabled: bool = True
    score_only_mode: bool = False
    external_pose_passthrough: bool = False


class BeatSaberVectorEnv:
    """
    Thin environment boundary around the custom GPU simulator.

    Beat Saber gameplay semantics remain custom; this layer only centralizes
    construction and configuration so trainer code stops owning simulator setup.
    """

    def __init__(self, config: BeatSaberEnvConfig):
        self.config = config
        self.simulator = GPUBeatSaberSimulator(num_envs=int(config.num_envs), device=config.device)
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

    def _build_tuning(self) -> SimulatorTuning:
        return SimulatorTuning(
            penalty_weights=tuple(self.config.penalty_weights),
            dense_reward_scale=float(self.config.dense_reward_scale),
            training_wheels=float(self.config.training_wheels),
            rehab_assists=float(self.config.rehab_assists),
            survival_assistance=float(self.config.survival_assistance),
            stability_assistance=float(self.config.stability_assistance),
            style_guidance_level=float(self.config.style_guidance_level),
            hit_timing_profile=str(self.config.hit_timing_profile),
            fail_enabled=bool(self.config.fail_enabled),
            saber_inertia=float(self.config.saber_inertia),
            rot_clamp=float(self.config.rot_clamp),
            pos_clamp=float(self.config.pos_clamp),
            score_only_mode=bool(self.config.score_only_mode),
            external_pose_passthrough=bool(self.config.external_pose_passthrough),
        )

    def _config_ready(self) -> bool:
        required_attrs = (
            "_w_miss",
            "_good_hitbox_scale",
            "_start_energy",
            "_delta_clamp",
            "_fail_enabled",
        )
        return all(hasattr(self.simulator, attr) for attr in required_attrs)

    def _apply_config(self) -> bool:
        if not self._config_ready():
            return False
        apply_simulator_tuning(self.simulator, self._build_tuning())
        return True

    def _build_observation_space(self) -> Optional[object]:
        if spaces is None:
            return None
        return spaces.Box(low=-float("inf"), high=float("inf"), shape=(INPUT_DIM,), dtype=np.float32)

    def _build_action_space(self) -> Optional[object]:
        if spaces is None:
            return None
        return spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(self.config, key):
                raise AttributeError(f"Unknown BeatSaberEnvConfig field: {key}")
            setattr(self.config, key, value)
        self._apply_config()
        return self

    def load_maps(self, beatmaps, bpms, capacity=None):
        self.simulator.load_maps(beatmaps, bpms, capacity=capacity)
        if not self._apply_config():
            raise RuntimeError("BeatSaberVectorEnv failed to apply simulator config after load_maps().")

    def reset(self, start_times=None):
        self.simulator.reset(start_times=start_times)
        return self.simulator.get_states()

    def get_states(self):
        return self.simulator.get_states()

    def step(self, actions, dt=1.0 / 60.0):
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.simulator.device)
        if actions.ndim == 1:
            if int(self.config.num_envs) != 1:
                raise ValueError(
                    f"Expected batched actions with shape ({int(self.config.num_envs)}, {ACTION_DIM}); "
                    f"got unbatched shape {tuple(actions.shape)}."
                )
            actions = actions.unsqueeze(0)
        expected_shape = (int(self.config.num_envs), ACTION_DIM)
        if tuple(actions.shape) != expected_shape:
            raise ValueError(f"Expected actions with shape {expected_shape}; got {tuple(actions.shape)}.")

        rewards, _ = self.simulator.step(actions, dt=dt)
        observations = self.simulator.get_states()
        done = self.simulator.episode_done.clone()
        if hasattr(self.simulator, "_terminal_reason"):
            terminal_reason = self.simulator._terminal_reason.clone()
            truncated = terminal_reason == 3
        else:
            terminal_reason = torch.zeros_like(done, dtype=torch.long)
            truncated = torch.zeros_like(done)
        terminated = done & ~truncated
        info = {
            "done": done,
            "terminal_reason": terminal_reason,
        }
        if hasattr(self.simulator, "_completion_ratio"):
            info["completion_ratio"] = self.simulator._completion_ratio.clone()
        return observations, rewards, terminated, truncated, info

    def __getattr__(self, item):
        return getattr(self.simulator, item)


class BeatSaberGymEnv(gym.Env if gym is not None else object):
    """
    Single-env Gymnasium-compatible wrapper.

    This is intentionally thin. It gives future SB3/Gym integrations a stable
    landing zone without replacing the custom simulator semantics.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: BeatSaberEnvConfig):
        if gym is None or spaces is None:
            raise RuntimeError("Gymnasium is not installed. Install `gymnasium` to use BeatSaberGymEnv.")
        config.num_envs = 1
        self.vector_env = BeatSaberVectorEnv(config)
        self.observation_space = self.vector_env.observation_space
        self.action_space = self.vector_env.action_space

    def load_maps(self, beatmaps, bpms):
        self.vector_env.load_maps([beatmaps[0]], [bpms[0]])

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        start_times = None
        if options is not None and "start_time" in options:
            start_times = [options["start_time"]]
        observations = self.vector_env.reset(start_times=start_times)
        return _tensor_to_cpu_numpy(observations[0]), {}

    def step(self, action):
        if hasattr(action, "ndim") and action.ndim == 1:
            action = action[None, :]
        observations, rewards, terminated, truncated, info = self.vector_env.step(action)
        single_info = {key: _info_value_at_index(value, 0) for key, value in info.items()}
        return (
            _tensor_to_cpu_numpy(observations[0]),
            float(rewards[0].detach().cpu().item()),
            bool(terminated[0].detach().cpu().item()),
            bool(truncated[0].detach().cpu().item()),
            single_info,
        )

    def __getattr__(self, item):
        return getattr(self.vector_env, item)


class _ConfiguredSimulatorHandle:
    """
    Backward-compatible simulator facade.

    High-level code should prefer BeatSaberVectorEnv directly, but legacy
    simulator callsites can use this handle without bypassing env-managed
    configuration on map loads.
    """

    def __init__(self, vector_env: BeatSaberVectorEnv):
        self._vector_env = vector_env

    @property
    def config(self):
        return self._vector_env.config

    @property
    def simulator(self):
        return self._vector_env.simulator

    @property
    def vector_env(self):
        return self._vector_env

    def configure(self, **kwargs):
        self._vector_env.configure(**kwargs)
        return self

    def load_maps(self, beatmaps, bpms, capacity=None):
        self._vector_env.load_maps(beatmaps, bpms, capacity=capacity)
        return self

    def reset(self, *args, **kwargs):
        return self.simulator.reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self.simulator.step(*args, **kwargs)

    def get_states(self):
        return self.simulator.get_states()

    def __getattr__(self, item):
        return getattr(self.simulator, item)


def make_simulator(num_envs, device="cuda", **config_overrides):
    config = BeatSaberEnvConfig(num_envs=int(num_envs), device=device, **config_overrides)
    return _ConfiguredSimulatorHandle(BeatSaberVectorEnv(config))


def make_vector_env(num_envs, device="cuda", **config_overrides):
    config = BeatSaberEnvConfig(num_envs=int(num_envs), device=device, **config_overrides)
    return BeatSaberVectorEnv(config)


def make_gym_env(device="cuda", **config_overrides):
    config = BeatSaberEnvConfig(num_envs=1, device=device, **config_overrides)
    return BeatSaberGymEnv(config)
