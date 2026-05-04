import unittest
from unittest import mock

import torch
import torch.nn as nn

from cybernoodles.training import policy_eval


def _empty_map_summary(map_hash="missing_hash", *, bpm=None, notes=None):
    notes = [] if notes is None else notes
    return {
        "map_hash": map_hash,
        "beatmap": {"notes": notes, "obstacles": []},
        "bpm": bpm,
        "notes": notes,
        "obstacles": [],
        "scorable_notes": len(notes),
        "note_count": len(notes),
        "obstacle_count": 0,
        "duration_sec": 0.0,
        "nps": 0.0,
        "obstacle_ratio": 0.0,
    }


def _playable_map_summary(map_hash="tiny_hash"):
    notes = [{"time": 0.1, "type": 0, "lineIndex": 1, "lineLayer": 1, "cutDirection": 0}]
    return {
        "map_hash": map_hash,
        "beatmap": {"notes": notes, "obstacles": []},
        "bpm": 120.0,
        "notes": notes,
        "obstacles": [],
        "scorable_notes": len(notes),
        "note_count": len(notes),
        "obstacle_count": 0,
        "duration_sec": 0.02,
        "nps": 1.0,
        "obstacle_ratio": 0.0,
    }


def _metric_summary(_sim, *, start_times=None, end_times=None):
    return {
        "accuracy": 0.0,
        "engaged_accuracy": 0.0,
        "resolved_accuracy": 0.0,
        "note_coverage": 0.0,
        "resolved_coverage": 0.0,
        "avg_cut": 0.0,
        "completion": 1.0,
        "clear_rate": 0.0,
        "fail_rate": 0.0,
        "timeout_rate": 1.0,
        "mean_target_notes": 1.0,
        "mean_hits": 0.0,
        "mean_misses": 1.0,
        "mean_engaged_notes": 0.0,
        "motion_efficiency": 0.0,
        "waste_motion": 0.0,
        "idle_motion": 0.0,
        "guard_error": 0.0,
        "oscillation": 0.0,
        "lateral_motion": 0.0,
        "style_violation": 0.0,
        "angular_violation": 0.0,
        "flail_index": 0.0,
    }


class _TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, state):
        batch = int(state.shape[0])
        mean = state.new_zeros((batch, policy_eval.ACTION_DIM)) + self.anchor
        std = state.new_ones((batch, policy_eval.ACTION_DIM))
        value = state.new_zeros((batch, 1))
        return mean, std, value


class _FakeSimulator:
    def __init__(self, num_envs):
        self.num_envs = int(num_envs)
        self.device = torch.device("cpu")
        self.episode_done = torch.zeros(self.num_envs, dtype=torch.bool)
        self.frames = 0

    def reset(self, start_times=None):
        self.frames = 0
        self.episode_done.zero_()

    def get_states(self):
        return torch.zeros((self.num_envs, 8), dtype=torch.float32)

    def step(self, actions, dt=1.0 / 60.0):
        self.frames += 1
        if self.frames >= 2:
            self.episode_done.fill_(True)
        return torch.zeros(self.num_envs), torch.zeros((self.num_envs, policy_eval.ACTION_DIM))


class _FakeVectorEnv:
    def __init__(self, num_envs):
        self.simulator = _FakeSimulator(num_envs)
        self.loaded = False

    def load_maps(self, beatmaps, bpms):
        self.loaded = True


class PolicyEvalEmptyContractTests(unittest.TestCase):
    def test_empty_requested_map_set_raises_by_default(self):
        with mock.patch.object(policy_eval, "make_vector_env") as make_vector_env:
            with self.assertRaisesRegex(RuntimeError, "No maps were evaluated"):
                policy_eval.evaluate_policy_model(None, "cpu", [])

        make_vector_env.assert_not_called()

    def test_allow_empty_returns_explicit_empty_summary(self):
        summary = policy_eval.evaluate_policy_model(None, "cpu", [], allow_empty=True)

        self.assertTrue(summary["empty_eval"])
        self.assertEqual(summary["requested_map_count"], 0)
        self.assertEqual(summary["evaluated_map_count"], 0)
        self.assertEqual(summary["maps"], [])
        self.assertIsNone(summary["mean_accuracy"])

    def test_missing_or_empty_maps_raise_by_default(self):
        with mock.patch.object(policy_eval, "summarize_eval_map", return_value=_empty_map_summary()):
            with self.assertRaisesRegex(RuntimeError, "No maps were evaluated"):
                policy_eval.evaluate_policy_model(None, "cpu", ["missing_hash"])

    def test_allow_empty_reports_skipped_maps(self):
        with mock.patch.object(policy_eval, "summarize_eval_map", return_value=_empty_map_summary()):
            summary = policy_eval.evaluate_policy_model(
                None,
                "cpu",
                ["missing_hash"],
                allow_empty=True,
            )

        self.assertTrue(summary["empty_eval"])
        self.assertEqual(summary["requested_map_count"], 1)
        self.assertEqual(summary["evaluated_map_count"], 0)
        self.assertEqual(summary["skipped_maps"], [{"map_hash": "missing_hash", "reason": "missing_map"}])

    def test_cuda_graph_request_falls_back_on_cpu(self):
        fake_env = _FakeVectorEnv(num_envs=2)
        with mock.patch.object(policy_eval, "summarize_eval_map", return_value=_playable_map_summary()):
            with mock.patch.object(policy_eval, "make_vector_env", return_value=fake_env):
                with mock.patch.object(policy_eval, "summarize_play_metrics", side_effect=_metric_summary):
                    summary = policy_eval.evaluate_policy_model(
                        _TinyPolicy(),
                        "cpu",
                        ["tiny_hash"],
                        num_envs=2,
                        use_cuda_graph=True,
                        action_repeat=2,
                    )

        self.assertTrue(summary["cuda_graph_requested"])
        self.assertFalse(summary["cuda_graph_used"])
        self.assertEqual(summary["evaluated_map_count"], 1)
        self.assertEqual(len(summary["maps"]), 1)
        self.assertEqual(summary["maps"][0]["frames"], 2)
        self.assertIn("eval device is not CUDA", summary["cuda_graph_fallback_reasons"][0]["reason"])

    def test_cuda_graph_require_raises_on_cpu(self):
        fake_env = _FakeVectorEnv(num_envs=1)
        with mock.patch.object(policy_eval, "summarize_eval_map", return_value=_playable_map_summary()):
            with mock.patch.object(policy_eval, "make_vector_env", return_value=fake_env):
                with self.assertRaisesRegex(RuntimeError, "CUDA graph eval requested but unsupported"):
                    policy_eval.evaluate_policy_model(
                        _TinyPolicy(),
                        "cpu",
                        ["tiny_hash"],
                        num_envs=1,
                        use_cuda_graph=True,
                        require_cuda_graph=True,
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_graph_matches_eager_on_tiny_deterministic_map(self):
        model = _TinyPolicy().to("cuda")
        map_summary = _playable_map_summary("tiny_cuda")
        map_summary["duration_sec"] = 0.05

        with mock.patch.object(policy_eval, "summarize_eval_map", return_value=map_summary):
            eager = policy_eval.evaluate_policy_model(
                model,
                torch.device("cuda"),
                ["tiny_cuda"],
                num_envs=1,
            )

        with mock.patch.object(policy_eval, "summarize_eval_map", return_value=map_summary):
            graph = policy_eval.evaluate_policy_model(
                model,
                torch.device("cuda"),
                ["tiny_cuda"],
                num_envs=1,
                use_cuda_graph=True,
                require_cuda_graph=True,
                cuda_graph_done_check_interval_frames=1,
            )

        self.assertTrue(graph["cuda_graph_used"])
        self.assertEqual(graph["cuda_graph_fallback_reasons"], [])
        self.assertEqual(eager["maps"][0]["frames"], graph["maps"][0]["frames"])
        for key in (
            "mean_accuracy",
            "mean_completion",
            "mean_note_coverage",
            "mean_clear_rate",
            "mean_fail_rate",
            "mean_timeout_rate",
        ):
            self.assertAlmostEqual(eager[key], graph[key], places=5)


if __name__ == "__main__":
    unittest.main()
