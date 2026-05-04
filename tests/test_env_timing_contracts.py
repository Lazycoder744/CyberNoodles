import unittest

from cybernoodles.core.gpu_simulator import (
    DEFAULT_HIT_WINDOW_BACK,
    DEFAULT_HIT_WINDOW_FRONT,
    DEFAULT_MISS_WINDOW_BACK,
    STRICT_HIT_WINDOW_BACK,
    STRICT_HIT_WINDOW_FRONT,
    STRICT_MISS_WINDOW_BACK,
    GPUBeatSaberSimulator,
)
from cybernoodles.envs import make_vector_env


TEST_BEATMAP = {
    "notes": [
        {
            "time": 1.0,
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
TEST_BPM = 120.0


def _windows(sim):
    return (
        float(sim._hit_window_front[0].item()),
        float(sim._hit_window_back[0].item()),
        float(sim._miss_window_back[0].item()),
    )


class EnvTimingContractTests(unittest.TestCase):
    def test_raw_simulator_zero_rehab_assist_preserves_default_timing(self):
        sim = GPUBeatSaberSimulator(num_envs=1, device="cpu")
        sim.load_maps([TEST_BEATMAP], [TEST_BPM])
        before = _windows(sim)

        sim.set_rehab_assists(0.0)

        self.assertEqual(before, _windows(sim))
        self.assertAlmostEqual(before[0], DEFAULT_HIT_WINDOW_FRONT, places=5)
        self.assertAlmostEqual(before[1], DEFAULT_HIT_WINDOW_BACK, places=5)
        self.assertAlmostEqual(before[2], DEFAULT_MISS_WINDOW_BACK, places=5)

    def test_default_env_load_preserves_raw_timing(self):
        env = make_vector_env(num_envs=1, device="cpu", rehab_assists=0.0)
        env.load_maps([TEST_BEATMAP], [TEST_BPM])

        front, back, miss_back = _windows(env.simulator)
        self.assertAlmostEqual(front, DEFAULT_HIT_WINDOW_FRONT, places=5)
        self.assertAlmostEqual(back, DEFAULT_HIT_WINDOW_BACK, places=5)
        self.assertAlmostEqual(miss_back, DEFAULT_MISS_WINDOW_BACK, places=5)

    def test_strict_profile_is_explicit(self):
        env = make_vector_env(
            num_envs=1,
            device="cpu",
            rehab_assists=0.0,
            hit_timing_profile="strict",
        )
        env.load_maps([TEST_BEATMAP], [TEST_BPM])

        front, back, miss_back = _windows(env.simulator)
        self.assertAlmostEqual(front, STRICT_HIT_WINDOW_FRONT, places=5)
        self.assertAlmostEqual(back, STRICT_HIT_WINDOW_BACK, places=5)
        self.assertAlmostEqual(miss_back, STRICT_MISS_WINDOW_BACK, places=5)

    def test_assisted_profile_widens_default_timing(self):
        env = make_vector_env(
            num_envs=1,
            device="cpu",
            rehab_assists=1.0,
            hit_timing_profile="assisted",
        )
        env.load_maps([TEST_BEATMAP], [TEST_BPM])

        front, back, miss_back = _windows(env.simulator)
        self.assertGreater(front, DEFAULT_HIT_WINDOW_FRONT)
        self.assertGreater(back, DEFAULT_HIT_WINDOW_BACK)
        self.assertLess(miss_back, DEFAULT_MISS_WINDOW_BACK)


if __name__ == "__main__":
    unittest.main()
