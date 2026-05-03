import contextlib
import types
import unittest

import numpy as np
import torch

import cybernoodles.envs.beat_saber_env as env_module
from cybernoodles.core.network import INPUT_DIM
from cybernoodles.envs import make_vector_env


EMPTY_BEATMAP = {
    "notes": [],
    "obstacles": [],
    "njs": 18.0,
    "offset": 0.0,
}


def _env_kwargs():
    return {
        "penalty_weights": (0.5, 0.0, 0.0, 0.0),
        "dense_reward_scale": 0.0,
        "training_wheels": 0.0,
        "rehab_assists": 0.0,
        "survival_assistance": 0.0,
        "stability_assistance": 0.0,
        "style_guidance_level": 0.0,
        "fail_enabled": False,
        "saber_inertia": 0.0,
        "rot_clamp": 0.12,
        "pos_clamp": 0.15,
    }


def _make_vector_env(num_envs=1):
    return make_vector_env(num_envs=num_envs, device="cpu", **_env_kwargs())


class _FakeBox:
    def __init__(self, low, high, shape, dtype):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype


@contextlib.contextmanager
def _gym_symbols_available():
    original_gym = env_module.gym
    original_spaces = env_module.spaces
    if env_module.gym is None:
        env_module.gym = types.SimpleNamespace(Env=object)
    if env_module.spaces is None:
        env_module.spaces = types.SimpleNamespace(Box=_FakeBox)
    try:
        yield
    finally:
        env_module.gym = original_gym
        env_module.spaces = original_spaces


class EnvContractTests(unittest.TestCase):
    def test_vector_env_keeps_tensor_contract_and_surfaces_timeout_truncation(self):
        env = _make_vector_env()
        env.load_maps([EMPTY_BEATMAP], [120.0])
        reset_obs = env.reset()

        self.assertIsInstance(reset_obs, torch.Tensor)
        self.assertEqual(tuple(reset_obs.shape), (1, INPUT_DIM))

        action = env.simulator.poses.clone()
        timeout_dt = float(env.simulator.map_durations[0].item()) + 0.25
        obs, reward, terminated, truncated, info = env.step(action, dt=timeout_dt)

        self.assertIsInstance(obs, torch.Tensor)
        self.assertIsInstance(reward, torch.Tensor)
        self.assertIsInstance(terminated, torch.Tensor)
        self.assertIsInstance(truncated, torch.Tensor)
        self.assertFalse(bool(terminated[0].item()))
        self.assertTrue(bool(truncated[0].item()))
        self.assertTrue(bool(info["done"][0].item()))
        self.assertEqual(int(info["terminal_reason"][0].item()), 3)

    def test_gym_wrapper_returns_numpy_and_python_contract_values(self):
        with _gym_symbols_available():
            env = env_module.make_gym_env(device="cpu", **_env_kwargs())
            env.load_maps([EMPTY_BEATMAP], [120.0])
            obs, reset_info = env.reset()

            self.assertIsInstance(obs, np.ndarray)
            self.assertEqual(obs.shape, (INPUT_DIM,))
            self.assertEqual(obs.dtype, np.float32)
            self.assertEqual(reset_info, {})

            action = env.vector_env.simulator.poses[0].detach().cpu().numpy().astype(np.float32)
            step_obs, reward, terminated, truncated, step_info = env.step(action)

            self.assertIsInstance(step_obs, np.ndarray)
            self.assertEqual(step_obs.shape, (INPUT_DIM,))
            self.assertIsInstance(reward, float)
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIsInstance(step_info, dict)
            self.assertIsInstance(step_info["done"], bool)
            self.assertIsInstance(step_info["terminal_reason"], int)

    def test_gym_wrapper_maps_simulator_timeout_to_truncated(self):
        with _gym_symbols_available():
            env = env_module.make_gym_env(device="cpu", **_env_kwargs())
            env.load_maps([EMPTY_BEATMAP], [120.0])
            env.reset()
            env.vector_env.simulator.map_durations.zero_()

            action = env.vector_env.simulator.poses[0].detach().cpu().numpy().astype(np.float32)
            _, _, terminated, truncated, info = env.step(action)

            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertTrue(info["done"])
            self.assertEqual(info["terminal_reason"], 3)

    def test_load_maps_rejects_cardinality_and_nonfinite_timing(self):
        env = _make_vector_env(num_envs=2)
        with self.assertRaisesRegex(ValueError, "Expected 2 beatmaps"):
            env.load_maps([EMPTY_BEATMAP], [120.0, 120.0])
        with self.assertRaisesRegex(ValueError, "Expected 2 BPM"):
            env.load_maps([EMPTY_BEATMAP, EMPTY_BEATMAP], [120.0])

        env = _make_vector_env()
        with self.assertRaisesRegex(ValueError, "bpm_list\\[0\\] must be finite"):
            env.load_maps([EMPTY_BEATMAP], [float("inf")])

        bad_timing = {
            "notes": [
                {
                    "time": float("nan"),
                    "lineIndex": 1,
                    "lineLayer": 1,
                    "type": 0,
                    "cutDirection": 1,
                }
            ],
            "obstacles": [],
            "njs": 18.0,
            "offset": 0.0,
        }
        with self.assertRaisesRegex(ValueError, "notes\\[0\\]\\.time"):
            env.load_maps([bad_timing], [120.0])

    def test_nonfinite_dt_and_start_times_are_rejected_before_state_mutation(self):
        env = _make_vector_env()
        env.load_maps([EMPTY_BEATMAP], [120.0])
        sim = env.simulator

        current_times = sim.current_times.clone()
        with self.assertRaisesRegex(ValueError, "dt must be finite"):
            sim.step(sim.poses.clone(), dt=float("nan"))
        self.assertTrue(torch.equal(sim.current_times, current_times))

        with self.assertRaisesRegex(ValueError, "start_times must be finite"):
            sim.reset(start_times=torch.tensor([float("nan")]))
        self.assertTrue(torch.equal(sim.current_times, current_times))

    def test_nonfinite_actions_are_sanitized(self):
        env = _make_vector_env()
        env.load_maps([EMPTY_BEATMAP], [120.0])
        sim = env.simulator
        action = sim.poses.clone()
        action[0, 0] = float("nan")
        action[0, 7] = float("inf")

        obs, reward, _, _, _ = env.step(action)

        self.assertTrue(torch.isfinite(sim.poses).all())
        self.assertTrue(torch.isfinite(obs).all())
        self.assertTrue(torch.isfinite(reward).all())


if __name__ == "__main__":
    unittest.main()
