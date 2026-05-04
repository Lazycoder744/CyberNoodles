import math
import json
import os
import tempfile
import unittest
import zipfile
from unittest import mock

import numpy as np
import torch

from cybernoodles.replay.generate_replay import (
    NOTE_EVENT_GOOD,
    NOTE_EVENT_MISS,
    _build_bsor_from_events,
    _choose_top_replay_candidates,
    _make_candidate_action_bias,
    _resolve_generation_candidate_counts,
    _select_primary_info_file,
    build_replay_generation_arg_parser,
    robust_get_notes,
)


class ReplayGenerationCpuPathTests(unittest.TestCase):
    def test_cuda_graph_generation_cli_exposes_candidate_controls(self):
        parser = build_replay_generation_arg_parser()
        args = parser.parse_args([
            "abc123",
            "model.pth",
            "out.bsor",
            "--cuda-graph",
            "--candidates",
            "512",
            "--top-k",
            "4",
            "--candidate-seed",
            "99",
        ])

        self.assertTrue(args.cuda_graph)
        self.assertEqual(args.candidates, 512)
        self.assertEqual(args.top_k, 4)
        self.assertEqual(args.candidate_seed, 99)

    def test_generation_candidate_count_resolution_caps_top_k(self):
        device = torch.device("cpu")

        recording_envs, candidate_envs, rerun_envs = _resolve_generation_candidate_counts(
            num_envs=8,
            candidates=64,
            top_k=128,
            device=device,
        )

        self.assertEqual(recording_envs, 8)
        self.assertEqual(candidate_envs, 64)
        self.assertEqual(rerun_envs, 64)

    def test_candidate_action_bias_is_zero_without_noise_and_seeded_with_noise(self):
        device = torch.device("cpu")

        no_noise = _make_candidate_action_bias(3, device, 0.0, 123)
        first = _make_candidate_action_bias(3, device, 0.5, 123)
        second = _make_candidate_action_bias(3, device, 0.5, 123)

        self.assertTrue(torch.equal(no_noise, torch.zeros_like(no_noise)))
        self.assertTrue(torch.equal(first, second))
        self.assertGreater(float(first.abs().sum().item()), 0.0)

    def test_top_replay_candidates_use_disciplined_rank_not_raw_score_only(self):
        class FakeSim:
            pass

        sim = FakeSim()
        sim.total_scores = torch.tensor([100.0, 98.0, 70.0])
        sim.total_hits = torch.tensor([10.0, 9.0, 5.0])
        sim.speed_samples = torch.ones(3)
        sim.speed_violation_sum = torch.zeros(3)
        sim.angular_violation_sum = torch.zeros(3)
        sim.waste_motion_sum = torch.zeros(3)
        sim.idle_motion_sum = torch.zeros(3)
        sim.guard_error_sum = torch.zeros(3)
        sim.oscillation_sum = torch.zeros(3)
        sim.lateral_motion_sum = torch.zeros(3)
        sim.useful_progress = torch.tensor([0.1, 0.9, 1.0])
        sim.motion_path = torch.ones(3)

        top_indices, strategy = _choose_top_replay_candidates(sim, 2)

        self.assertEqual(top_indices.tolist(), [1, 0])
        self.assertEqual(strategy, "disciplined-top-2")

    def test_build_bsor_from_events_normalizes_frames_and_preserves_event_counts(self):
        beatmap = {
            "song_name": "Unit Test Song",
            "level_author_name": "Unit Test Mapper",
            "difficulty": "Expert",
            "mode": "Standard",
            "environment_name": "DefaultEnvironment",
            "njs": 18.0,
            "offset": 0.0,
            "notes": [
                {
                    "time": 1.0,
                    "lineIndex": 1,
                    "lineLayer": 1,
                    "type": 0,
                    "cutDirection": 1,
                    "scoreCap": 115.0,
                },
                {
                    "time": 2.0,
                    "lineIndex": 2,
                    "lineLayer": 1,
                    "type": 1,
                    "cutDirection": 0,
                    "scoreCap": 115.0,
                },
            ],
            "obstacles": [
                {
                    "time": 1.5,
                    "lineIndex": 0,
                    "lineLayer": 0,
                    "width": 1,
                    "height": 1,
                    "duration": 0.5,
                }
            ],
        }
        recorded_poses = np.asarray(
            [
                [
                    0.0, 1.7, 0.0, 0.0, 0.0, 0.0, 2.0,
                    -0.2, 1.0, 0.4, 0.0, 0.0, 0.0, 3.0,
                    0.2, 1.0, 0.4, 0.0, 0.0, 0.0, 4.0,
                ],
                [
                    0.1, 1.7, 0.1, 0.0, 0.0, 0.0, 5.0,
                    -0.1, 1.1, 0.5, 0.0, 0.0, 0.0, 6.0,
                    0.1, 1.1, 0.5, 0.0, 0.0, 0.0, 7.0,
                ],
            ],
            dtype=np.float32,
        )
        events = [
            {
                "type": "hit",
                "time": 1.0,
                "note_index": 0,
                "pre_score": 70.0,
                "post_score": 30.0,
                "acc_score": 15.0,
                "saber_speed": 5.0,
                "saber_dir": [0.0, 0.0, 1.0],
                "cut_point": [0.0, 0.0, 0.0],
                "cut_normal": [0.0, 1.0, 0.0],
            },
            {
                "type": "wall",
                "time": 1.5,
                "obstacle_index": 0,
                "energy": 0.25,
            },
            {
                "type": "miss",
                "time": 2.0,
                "note_index": 1,
            },
        ]

        bsor, total_score, max_score, replay_stats = _build_bsor_from_events(
            events,
            recorded_poses,
            num_frames=2,
            map_hash="abcd1234",
            beatmap=beatmap,
            bpm=120.0,
        )

        self.assertEqual(len(bsor.frames), 2)
        self.assertEqual(len(bsor.notes), 2)
        self.assertEqual(len(bsor.walls), 1)
        self.assertEqual(bsor.notes[0].event_type, NOTE_EVENT_GOOD)
        self.assertEqual(bsor.notes[1].event_type, NOTE_EVENT_MISS)
        self.assertEqual(total_score, 115)
        self.assertGreater(max_score, total_score)
        self.assertEqual(replay_stats["hit_count"], 1)
        self.assertEqual(replay_stats["miss_count"], 1)
        self.assertEqual(replay_stats["wall_count"], 1)
        self.assertEqual(bsor.info.songHash, "ABCD1234")

        for frame in bsor.frames:
            for obj in (frame.head, frame.left_hand, frame.right_hand):
                quat_norm = math.sqrt(
                    obj.x_rot * obj.x_rot
                    + obj.y_rot * obj.y_rot
                    + obj.z_rot * obj.z_rot
                    + obj.w_rot * obj.w_rot
                )
                self.assertAlmostEqual(quat_norm, 1.0, places=6)

    def test_robust_get_notes_exposes_cache_controls(self):
        self.assertTrue(callable(robust_get_notes.cache_clear))
        self.assertTrue(callable(robust_get_notes.cache_info))

    def test_select_primary_info_file_prefers_real_info_over_bpminfo(self):
        selected = _select_primary_info_file([
            "BPMInfo.dat",
            "Info.dat",
        ])
        self.assertEqual(selected, "Info.dat")

    def test_robust_get_notes_ignores_bpminfo_when_info_dat_is_present(self):
        map_hash = "deadbeef"
        info_payload = {
            "_version": "2.1.0",
            "_songName": "Replay Test Song",
            "_levelAuthorName": "Replay Test Mapper",
            "_beatsPerMinute": 150.0,
            "_environmentName": "BigMirrorEnvironment",
            "_difficultyBeatmapSets": [
                {
                    "_beatmapCharacteristicName": "Standard",
                    "_difficultyBeatmaps": [
                        {
                            "_difficulty": "Hard",
                            "_beatmapFilename": "HardStandard.dat",
                            "_noteJumpMovementSpeed": 16.0,
                            "_noteJumpStartBeatOffset": -0.25,
                        }
                    ],
                }
            ],
        }
        bpm_info_payload = {
            "_version": "2.0.0",
            "_songSampleCount": 123,
            "_songFrequency": 44100,
            "_regions": [],
        }
        beatmap_payload = {
            "_version": "2.0.0",
            "_notes": [],
            "_obstacles": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, f"{map_hash}.zip")
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("BPMInfo.dat", json.dumps(bpm_info_payload))
                zf.writestr("Info.dat", json.dumps(info_payload))
                zf.writestr("HardStandard.dat", json.dumps(beatmap_payload))

            robust_get_notes.cache_clear()
            with mock.patch("cybernoodles.replay.generate_replay.MAPS_DIR", tmpdir):
                beatmap, bpm = robust_get_notes(map_hash)

            self.assertEqual(bpm, 150.0)
            self.assertEqual(beatmap["song_name"], "Replay Test Song")
            self.assertEqual(beatmap["level_author_name"], "Replay Test Mapper")
            self.assertEqual(beatmap["difficulty"], "Hard")
            self.assertEqual(beatmap["mode"], "Standard")
            self.assertEqual(beatmap["environment_name"], "BigMirrorEnvironment")


if __name__ == "__main__":
    unittest.main()
