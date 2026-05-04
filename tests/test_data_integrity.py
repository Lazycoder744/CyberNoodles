import json
import io
import math
import os
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import torch

from cybernoodles.core.network import (
    CURRENT_POSE_END,
    CURRENT_POSE_START,
    INPUT_DIM,
    NOTE_FEATURES,
    NOTES_DIM,
    OBSTACLE_FEATURES,
    OBSTACLES_DIM,
    POSE_DIM,
    STATE_FRAME_DIM,
    STATE_HISTORY_FRAMES,
)
from cybernoodles.data import dataset_builder
from cybernoodles.data.sim_calibration import DEFAULT_CALIBRATION, load_simulator_calibration
from cybernoodles.data.shard_io import load_shard_pair
from cybernoodles.data.style_calibration import DEFAULT_STYLE_CALIBRATION, load_style_calibration
from cybernoodles.training.train_bc import preflight_shard_records


class DataIntegrityTests(unittest.TestCase):
    def test_python_and_rust_manifest_versions_are_aligned_when_inspectable(self):
        repo_root = Path(__file__).resolve().parents[1]
        rust_builder = repo_root / "rust" / "bsor_tools" / "src" / "builder.rs"
        if not rust_builder.exists():
            self.skipTest("Rust dataset builder source is not present")

        source = rust_builder.read_text(encoding="utf-8")
        match = re.search(r"const\s+MANIFEST_VERSION\s*:\s*u32\s*=\s*(\d+)\s*;", source)
        if match is None:
            self.skipTest("Rust manifest version constant is not inspectable")

        self.assertEqual(int(match.group(1)), dataset_builder.MANIFEST_VERSION)
        self.assertIn('const TARGET_STORAGE_DTYPE: &str = "float32";', source)
        self.assertIn("Dtype::F32", source)

    def test_manifest_compatibility_accepts_current_legacy_top_level_semantics(self):
        base_manifest = {
            "version": dataset_builder.MANIFEST_VERSION,
            "feature_dim": INPUT_DIM,
            "target_dim": POSE_DIM,
            "target_pose_horizon_frames": dataset_builder.TARGET_POSE_HORIZON_FRAMES,
        }

        errors = dataset_builder.manifest_compatibility_errors(base_manifest)
        self.assertTrue(any("semantic_schema" in error for error in errors))

        compatible = dict(base_manifest)
        compatible["semantic_schema"] = dataset_builder.manifest_semantic_metadata()
        self.assertEqual(dataset_builder.manifest_compatibility_errors(compatible), [])

        legacy_compatible = dict(base_manifest)
        legacy_compatible.update({
            "sample_hz": dataset_builder.SIM_SAMPLE_HZ,
            "history_offsets": list(dataset_builder.STATE_HISTORY_OFFSETS),
            "note_feature_layout": dataset_builder.NOTE_FEATURE_LAYOUT,
            "note_lookahead_beats": dataset_builder.NOTE_LOOKAHEAD_BEATS,
            "followthrough_beats": dataset_builder.FOLLOWTHROUGH_BEATS,
            "background_stride": dataset_builder.BACKGROUND_FRAME_STRIDE,
            "shard_storage_dtype": dataset_builder.SHARD_STORAGE_DTYPE,
        })
        self.assertEqual(dataset_builder.manifest_compatibility_errors(legacy_compatible), [])

        stale_semantics = json.loads(json.dumps(compatible))
        stale_semantics["semantic_schema"]["note_feature_layout"] = "same_width_old_layout"
        errors = dataset_builder.manifest_compatibility_errors(stale_semantics)
        self.assertTrue(any("note_feature_layout" in error for error in errors))

        legacy_stale = dict(legacy_compatible)
        legacy_stale["note_feature_layout"] = "same_width_old_layout"
        errors = dataset_builder.manifest_compatibility_errors(legacy_stale)
        self.assertTrue(any("note_feature_layout" in error for error in errors))

    def test_python_builder_records_missing_selected_replays(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            replay_dir = os.path.join(tmpdir, "replays")
            output_dir = os.path.join(tmpdir, "processed")
            shard_root = os.path.join(output_dir, "bc_shards")
            selected_path = os.path.join(tmpdir, "selected_scores.json")
            manifest_path = os.path.join(shard_root, "manifest.json")
            os.makedirs(replay_dir, exist_ok=True)
            with open(selected_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "selected": [
                            {"id": "missing_a"},
                            {"id": "missing_b"},
                        ]
                    },
                    f,
                )

            with mock.patch.multiple(
                dataset_builder,
                REPLAYS_DIR=replay_dir,
                OUTPUT_DIR=output_dir,
                SHARD_ROOT=shard_root,
                TRAIN_DIR=os.path.join(shard_root, "train"),
                VAL_DIR=os.path.join(shard_root, "val"),
                MANIFEST_PATH=manifest_path,
                SELECTED_SCORES_PATH=selected_path,
            ):
                dataset_builder._process_data_python(
                    workers=1,
                    manifest_save_every=1,
                    max_pending_writes=1,
                    gc_every=0,
                    top_selected=2,
                    status_every=1,
                )

            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            warning = manifest["warnings"][0]
            self.assertEqual(warning["code"], "selected_replays_missing")
            self.assertEqual(warning["count"], 2)
            self.assertEqual(warning["replay_files"], ["missing_a.bsor", "missing_b.bsor"])

    def test_shard_preflight_rejects_bad_dtype_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_root = os.path.join(tmpdir, "bc_shards")
            train_dir = os.path.join(shard_root, "train")
            os.makedirs(train_dir, exist_ok=True)
            x_path = os.path.join(train_dir, "X_ok.pt")
            y_path = os.path.join(train_dir, "y_ok.pt")
            record = {
                "replay_file": "ok.bsor",
                "x_path": "train/X_ok.pt",
                "y_path": "train/y_ok.pt",
                "samples": 2,
            }

            torch.save(torch.ones((2, INPUT_DIM), dtype=torch.float16), x_path)
            torch.save(torch.ones((2, POSE_DIM), dtype=torch.float32), y_path)
            result = preflight_shard_records([record], "train", shard_root=shard_root)
            self.assertEqual(result["samples"], 2)

            torch.save(torch.ones((2, POSE_DIM), dtype=torch.float16), y_path)
            with self.assertRaisesRegex(RuntimeError, "dtype"):
                preflight_shard_records([record], "train", shard_root=shard_root)

            torch.save(torch.ones((2, POSE_DIM), dtype=torch.float32), y_path)
            torch.save(torch.ones((2, INPUT_DIM), dtype=torch.float32), x_path)
            with self.assertRaisesRegex(RuntimeError, "dtype"):
                preflight_shard_records([record], "train", shard_root=shard_root)

            bad_x = torch.ones((2, INPUT_DIM), dtype=torch.float16)
            bad_x[0, 0] = float("inf")
            torch.save(bad_x, x_path)
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                preflight_shard_records([record], "train", shard_root=shard_root)

    def test_fresh_python_written_shard_matches_writer_contract(self):
        frames = []
        for idx in range(48):
            pose = dataset_builder.DEFAULT_POSE.copy()
            pose[7] = 2.35 if idx % 7 == 0 else -0.35 + 0.01 * idx
            pose[8] = 1.05 + 0.004 * idx
            pose[14] = -2.25 if idx % 11 == 0 else 0.35 - 0.006 * idx
            pose[15] = 1.0 + 0.003 * idx
            left_angle = 0.01 * idx
            right_angle = -0.008 * idx
            pose[10:14] = [
                0.0,
                0.0,
                math.sin(left_angle * 0.5),
                math.cos(left_angle * 0.5),
            ]
            pose[17:21] = [
                0.0,
                math.sin(right_angle * 0.5),
                0.0,
                math.cos(right_angle * 0.5),
            ]
            frames.append({"time": idx / dataset_builder.SIM_SAMPLE_HZ, "pose": pose.tolist()})

        beatmap = {
            "notes": [
                {"time": 0.25, "lineIndex": 1, "lineLayer": 1, "type": 0, "cutDirection": 1},
                {"time": 4.50, "lineIndex": 2, "lineLayer": 0, "type": 1, "cutDirection": 0},
            ],
            "obstacles": [
                {"time": 0.5, "lineIndex": 0, "lineLayer": 0, "width": 1, "height": 3, "duration": 1.0},
            ],
            "njs": 18.0,
            "offset": 4.0,
        }
        X, y, stats = dataset_builder.extract_features(frames, beatmap, bpm=120.0)
        self.assertGreater(stats["samples_kept"], 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            shard_root = os.path.join(tmpdir, "bc_shards")
            with mock.patch.multiple(
                dataset_builder,
                SHARD_ROOT=shard_root,
                TRAIN_DIR=os.path.join(shard_root, "train"),
                VAL_DIR=os.path.join(shard_root, "val"),
            ):
                x_rel, y_rel, sample_count = dataset_builder._write_replay_shard_files(
                    "train",
                    "contract_probe.bsor",
                    X,
                    y,
                )

            record = {
                "replay_file": "contract_probe.bsor",
                "x_path": x_rel.replace("\\", "/"),
                "y_path": y_rel.replace("\\", "/"),
                "samples": sample_count,
            }
            shard_x, shard_y = load_shard_pair(record, shard_root, device="cpu")

        self.assertEqual(shard_x.dtype, torch.float16)
        self.assertEqual(shard_y.dtype, torch.float32)
        self.assertEqual(tuple(shard_x.shape), (sample_count, INPUT_DIM))
        self.assertEqual(tuple(shard_y.shape), (sample_count, POSE_DIM))

        note_times = shard_x[:, 0:NOTES_DIM:NOTE_FEATURES]
        self.assertGreaterEqual(float(note_times.min().item()), dataset_builder.NOTE_TIME_FEATURE_MIN_BEATS)
        self.assertLessEqual(float(note_times.max().item()), dataset_builder.NOTE_TIME_FEATURE_MAX_BEATS)

        obs_base = NOTES_DIM
        obs_times = shard_x[:, obs_base:obs_base + OBSTACLES_DIM:OBSTACLE_FEATURES]
        self.assertGreaterEqual(float(obs_times.min().item()), dataset_builder.SIM_OBSTACLE_TIME_MIN_BEATS)
        self.assertLessEqual(float(obs_times.max().item()), dataset_builder.SIM_OBSTACLE_TIME_MAX_BEATS)

        position_slices = ((0, 3), (7, 10), (14, 17))
        for hist_idx in range(STATE_HISTORY_FRAMES):
            pose_start = NOTES_DIM + OBSTACLES_DIM + hist_idx * STATE_FRAME_DIM
            pose = shard_x[:, pose_start:pose_start + POSE_DIM]
            for start, end in position_slices:
                self.assertLessEqual(
                    float(pose[:, start:end].abs().max().item()),
                    dataset_builder.STATE_POSE_POSITION_ABS_LIMIT + 1e-3,
                )
        for start, end in position_slices:
            self.assertLessEqual(
                float(shard_y[:, start:end].abs().max().item()),
                dataset_builder.ACTION_ABS_COMPONENT_LIMIT + 1e-6,
            )

        current_pose = shard_x[:, CURRENT_POSE_START:CURRENT_POSE_END].to(dtype=shard_y.dtype)
        delta_limit = torch.tensor(dataset_builder.TARGET_ACTION_DELTA_CLAMP, dtype=shard_y.dtype)
        self.assertTrue(torch.all((shard_y - current_pose).abs() <= (delta_limit + 2e-3)))

    def test_shard_preflight_prints_progress(self):
        records = [
            {
                "replay_file": f"progress_{idx}.bsor",
                "x_path": f"train/X_progress_{idx}.pt",
                "y_path": f"train/y_progress_{idx}.pt",
                "samples": 1,
            }
            for idx in range(3)
        ]
        x = torch.zeros((1, INPUT_DIM), dtype=torch.float16)
        y = torch.zeros((1, POSE_DIM), dtype=torch.float32)

        with mock.patch("cybernoodles.training.train_bc.validate_shard_record", return_value={"samples": 1}):
            output = io.StringIO()
            with redirect_stdout(output):
                preflight_shard_records(
                    records,
                    "train",
                    shard_root="unused",
                    progress_every=2,
                    progress_seconds=0.0,
                )

        text = output.getvalue()
        self.assertIn("BC shard preflight train: checking 3 shard(s)", text)
        self.assertIn("BC shard preflight train: 2/3 shard(s)", text)
        self.assertIn("BC shard preflight train: 3/3 shard(s)", text)

    def test_simulator_calibration_rejects_nonfinite_values(self):
        payload = DEFAULT_CALIBRATION.copy()
        payload["x_offset"] = float("nan")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(payload, f)
            path = f.name
        try:
            with self.assertRaisesRegex(ValueError, "finite"):
                load_simulator_calibration(path)
        finally:
            os.remove(path)

    def test_style_calibration_rejects_missing_version_and_invalid_ordering(self):
        missing_version = DEFAULT_STYLE_CALIBRATION.copy()
        missing_version.pop("version")
        invalid_ordering = DEFAULT_STYLE_CALIBRATION.copy()
        invalid_ordering["linear_speed_p99"] = invalid_ordering["linear_speed_p95"] - 0.1

        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = os.path.join(tmpdir, "missing_version.json")
            invalid_path = os.path.join(tmpdir, "invalid_ordering.json")
            with open(missing_path, "w", encoding="utf-8") as f:
                json.dump(missing_version, f)
            with open(invalid_path, "w", encoding="utf-8") as f:
                json.dump(invalid_ordering, f)

            with self.assertRaisesRegex(ValueError, "version"):
                load_style_calibration(missing_path)
            with self.assertRaisesRegex(ValueError, "p99"):
                load_style_calibration(invalid_path)


if __name__ == "__main__":
    unittest.main()
