from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from cybernoodles.bsor_bridge import validate_bsor
from cybernoodles.replay.generate_replay import (
    FPS,
    _build_bsor_from_events,
    _format_bsor_validation_summary,
)


def _pose(head_x=0.0):
    return [
        head_x, 1.7, 0.0, 0.0, 0.0, 0.0, 1.0,
        -0.2, 1.0, 0.4, 0.0, 0.0, 0.0, 1.0,
        0.2, 1.0, 0.4, 0.0, 0.0, 0.0, 1.0,
    ]


def _beatmap(njs=16.0, offset=0.0):
    return {
        "song_name": "Timing Test",
        "level_author_name": "Unit Test",
        "difficulty": "Expert",
        "mode": "Standard",
        "environment_name": "DefaultEnvironment",
        "njs": njs,
        "offset": offset,
        "notes": [],
        "obstacles": [],
    }


def _parsed_replay():
    return SimpleNamespace(
        info=SimpleNamespace(songHash="abcd", difficulty="Expert", mode="Standard"),
        frames=[],
        notes=[],
        walls=[],
        pauses=[],
        user_data=[],
    )


class ReplayTimingRegressionTests(unittest.TestCase):
    def test_post_step_poses_are_timestamped_after_step(self):
        recorded_poses = np.asarray([_pose(head_x=1.0), _pose(head_x=2.0)], dtype=np.float32)

        bsor, _, _, _ = _build_bsor_from_events(
            [],
            recorded_poses,
            num_frames=2,
            map_hash="abcd1234",
            beatmap=_beatmap(),
            bpm=120.0,
        )

        self.assertAlmostEqual(bsor.frames[0].time, 1.0 / FPS)
        self.assertAlmostEqual(bsor.frames[1].time, 2.0 / FPS)
        self.assertAlmostEqual(bsor.frames[0].head.x, 1.0)
        self.assertAlmostEqual(bsor.frames[1].head.x, 2.0)

    def test_jump_distance_uses_beatmap_timing_when_available(self):
        recorded_poses = np.asarray([_pose()], dtype=np.float32)

        bsor, _, _, replay_stats = _build_bsor_from_events(
            [],
            recorded_poses,
            num_frames=1,
            map_hash="abcd1234",
            beatmap=_beatmap(njs=16.0, offset=0.0),
            bpm=120.0,
        )

        self.assertAlmostEqual(bsor.info.jumpDistance, 32.0)
        self.assertEqual(replay_stats["jump_distance_source"], "beatmap_timing")
        self.assertAlmostEqual(replay_stats["jump_distance_spawn_ahead_beats"], 2.0)

    def test_jump_distance_fallback_is_explicit_without_valid_timing(self):
        recorded_poses = np.asarray([_pose()], dtype=np.float32)

        bsor, _, _, replay_stats = _build_bsor_from_events(
            [],
            recorded_poses,
            num_frames=1,
            map_hash="abcd1234",
            beatmap=_beatmap(njs=17.5, offset=0.0),
            bpm=0.0,
        )

        self.assertAlmostEqual(bsor.info.jumpDistance, 17.5)
        self.assertEqual(replay_stats["jump_distance_source"], "fallback_njs")
        self.assertEqual(
            replay_stats["jump_distance_fallback_reason"],
            "missing_or_invalid_beatmap_timing",
        )

    def test_rust_validation_error_is_not_reported_as_ok(self):
        with (
            mock.patch("cybernoodles.bsor_bridge.bsor_tools_available", return_value=True),
            mock.patch(
                "cybernoodles.bsor_bridge._run_bsor_tools",
                side_effect=RuntimeError("rust validation exploded"),
            ),
            mock.patch("cybernoodles.bsor_bridge.load_bsor", return_value=_parsed_replay()),
        ):
            summary = validate_bsor("fake.bsor", backend="auto")

        self.assertEqual(summary["validation_backend"], "python")
        self.assertFalse(summary["rust_validation_ok"])
        self.assertIn("rust validation exploded", summary["rust_validation_error"])

        message = _format_bsor_validation_summary(summary)
        self.assertIn("Rust BSOR validation failed", message)
        self.assertNotIn("validation OK", message)


if __name__ == "__main__":
    unittest.main()
