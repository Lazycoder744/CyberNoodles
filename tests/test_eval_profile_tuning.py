import unittest

from cybernoodles.envs import make_simulator, make_vector_env
from cybernoodles.training.policy_eval import get_eval_profile


TEST_BEATMAP = {
    "notes": [
        {
            "time": 1.0,
            "lineIndex": 1,
            "lineLayer": 1,
            "type": 0,
            "cutDirection": 1,
        },
        {
            "time": 2.0,
            "lineIndex": 2,
            "lineLayer": 1,
            "type": 1,
            "cutDirection": 0,
        },
    ],
    "obstacles": [],
}
TEST_BPM = 120.0


def _scalar(value):
    if hasattr(value, "numel"):
        return float(value.reshape(-1)[0].item())
    return float(value)


def _build_eval_kwargs(profile_name):
    profile = get_eval_profile(profile_name)
    return {
        "penalty_weights": (0.5, 0.0, 0.0, 0.0),
        "dense_reward_scale": 0.0,
        "training_wheels": profile["training_wheels_level"],
        "rehab_assists": profile["assist_level"],
        "survival_assistance": profile["survival_assistance"],
        "stability_assistance": profile["stability_reward_level"],
        "style_guidance_level": profile["style_guidance_level"],
        "fail_enabled": profile["fail_enabled"],
        "saber_inertia": profile["saber_inertia"],
        "rot_clamp": profile["rot_clamp"],
        "pos_clamp": profile["pos_clamp"],
    }


def _capture_profile_state(sim):
    return {
        "fail_enabled": bool(sim._fail_enabled[0].item()),
        "good_hitbox_scale": _scalar(sim._good_hitbox_scale[0]),
        "contact_reward": _scalar(sim._w_contact[0]),
        "start_energy": _scalar(sim._start_energy[0]),
        "miss_energy": _scalar(sim._miss_energy[0]),
    }


class EvalProfileTuningRegressionTests(unittest.TestCase):
    def _load_with_vector_env(self, profile_name):
        env = make_vector_env(num_envs=1, device="cpu", **_build_eval_kwargs(profile_name))
        env.load_maps([TEST_BEATMAP], [TEST_BPM])
        return _capture_profile_state(env.simulator)

    def _load_with_simulator_factory(self, **config_overrides):
        sim = make_simulator(num_envs=1, device="cpu", **config_overrides)
        sim.load_maps([TEST_BEATMAP], [TEST_BPM])
        return sim

    def test_eval_profiles_apply_distinct_state_after_load_maps(self):
        strict = self._load_with_vector_env("strict")
        bc = self._load_with_vector_env("bc")
        rehab = self._load_with_vector_env("rehab")

        self.assertTrue(strict["fail_enabled"])
        self.assertFalse(bc["fail_enabled"])
        self.assertFalse(rehab["fail_enabled"])

        self.assertAlmostEqual(strict["good_hitbox_scale"], 1.0, places=5)
        self.assertGreater(bc["good_hitbox_scale"], strict["good_hitbox_scale"])
        self.assertGreater(rehab["good_hitbox_scale"], bc["good_hitbox_scale"])

        self.assertGreater(bc["contact_reward"], strict["contact_reward"])
        self.assertGreater(rehab["contact_reward"], bc["contact_reward"])

        self.assertGreater(bc["start_energy"], strict["start_energy"])
        self.assertGreater(rehab["start_energy"], bc["start_energy"])

        self.assertLess(bc["miss_energy"], strict["miss_energy"])
        self.assertLess(rehab["miss_energy"], bc["miss_energy"])

    def test_make_simulator_reapplies_config_on_load_maps(self):
        sim = self._load_with_simulator_factory(**_build_eval_kwargs("bc"))
        state = _capture_profile_state(sim)

        self.assertFalse(state["fail_enabled"])
        self.assertAlmostEqual(state["good_hitbox_scale"], 1.13, places=5)
        self.assertAlmostEqual(state["contact_reward"], 0.7025, places=5)
        self.assertAlmostEqual(state["start_energy"], 0.7275, places=5)
        self.assertAlmostEqual(state["miss_energy"], 0.05125, places=5)

    def test_make_simulator_preserves_replay_passthrough_flags(self):
        sim = self._load_with_simulator_factory(
            penalty_weights=(0.5, 0.0, 0.0, 0.0),
            dense_reward_scale=0.0,
            training_wheels=0.0,
            rehab_assists=0.0,
            survival_assistance=0.0,
            stability_assistance=0.0,
            style_guidance_level=0.0,
            fail_enabled=False,
            saber_inertia=0.0,
            rot_clamp=0.07,
            pos_clamp=0.12,
            score_only_mode=True,
            external_pose_passthrough=True,
        )

        self.assertTrue(sim._score_only_mode)
        self.assertTrue(sim._external_pose_passthrough)
        self.assertFalse(bool(sim._fail_enabled[0].item()))


if __name__ == "__main__":
    unittest.main()
