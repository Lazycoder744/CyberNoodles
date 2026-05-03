import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from cybernoodles.core.network import INPUT_DIM, POSE_DIM
from cybernoodles.data import dataset_builder
from cybernoodles.data.sim_calibration import DEFAULT_CALIBRATION, load_simulator_calibration
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
            torch.save(torch.ones((2, POSE_DIM), dtype=torch.float16), y_path)
            result = preflight_shard_records([record], "train", shard_root=shard_root)
            self.assertEqual(result["samples"], 2)

            torch.save(torch.ones((2, INPUT_DIM), dtype=torch.float32), x_path)
            with self.assertRaisesRegex(RuntimeError, "dtype"):
                preflight_shard_records([record], "train", shard_root=shard_root)

            bad_x = torch.ones((2, INPUT_DIM), dtype=torch.float16)
            bad_x[0, 0] = float("inf")
            torch.save(bad_x, x_path)
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                preflight_shard_records([record], "train", shard_root=shard_root)

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
