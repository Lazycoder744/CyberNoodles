import json
import os
import tempfile
import unittest
import zipfile
from unittest import mock

from cybernoodles.data import dataset_builder


def _write_expertplus_only_map_zip(directory, map_hash):
    info_payload = {
        "_version": "2.1.0",
        "_songName": "Difficulty Contract Test",
        "_beatsPerMinute": 120.0,
        "_difficultyBeatmapSets": [
            {
                "_beatmapCharacteristicName": "Standard",
                "_difficultyBeatmaps": [
                    {
                        "_difficulty": "ExpertPlus",
                        "_beatmapFilename": "ExpertPlusStandard.dat",
                        "_noteJumpMovementSpeed": 18.0,
                        "_noteJumpStartBeatOffset": 0.0,
                    }
                ],
            }
        ],
    }
    beatmap_payload = {
        "_version": "2.0.0",
        "_notes": [],
        "_obstacles": [],
    }

    zip_path = os.path.join(directory, f"{map_hash}.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Info.dat", json.dumps(info_payload))
        zf.writestr("ExpertPlusStandard.dat", json.dumps(beatmap_payload))


class DatasetBuilderDifficultyContractTests(unittest.TestCase):
    def test_explicit_difficulty_requires_exact_chart(self):
        map_hash = "difficulty_contract"
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_expertplus_only_map_zip(tmpdir, map_hash)

            with mock.patch.object(dataset_builder, "MAPS_DIR", tmpdir):
                beatmap, bpm = dataset_builder.get_map_data(
                    map_hash,
                    preferred_mode="Standard",
                    preferred_difficulty="Hard",
                )

            self.assertIsNone(beatmap)
            self.assertIsNone(bpm)

    def test_no_explicit_difficulty_keeps_hardest_standard_fallback(self):
        map_hash = "difficulty_contract"
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_expertplus_only_map_zip(tmpdir, map_hash)

            with mock.patch.object(dataset_builder, "MAPS_DIR", tmpdir):
                beatmap, bpm = dataset_builder.get_map_data(
                    map_hash,
                    preferred_mode="Standard",
                )

            self.assertIsNotNone(beatmap)
            self.assertEqual(bpm, 120.0)
            self.assertEqual(beatmap["mode"], "Standard")
            self.assertEqual(beatmap["difficulty"], "ExpertPlus")

    def test_process_single_does_not_retry_generic_map_for_explicit_difficulty(self):
        map_hash = "difficulty_contract"
        frames = [
            {"time": 0.0, "pose": dataset_builder.DEFAULT_POSE.tolist()},
            {"time": 0.1, "pose": dataset_builder.DEFAULT_POSE.tolist()},
        ]
        replay_meta = {
            "song_hash": map_hash,
            "difficulty": "Hard",
            "mode": "Standard",
            "modifiers": "",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_expertplus_only_map_zip(tmpdir, map_hash)

            with mock.patch.object(dataset_builder, "MAPS_DIR", tmpdir), mock.patch.object(
                dataset_builder,
                "parse_bsor",
                return_value=(frames, dict(replay_meta)),
            ), mock.patch.object(dataset_builder, "extract_features") as extract_features:
                result = dataset_builder.process_single("explicit_hard.bsor")

            self.assertIsNone(result)
            extract_features.assert_not_called()

    def test_failed_replays_remain_pending_for_retry(self):
        replay_paths = [
            os.path.join("replays", "already_done.bsor"),
            os.path.join("replays", "failed_until_map_downloaded.bsor"),
        ]

        pending = dataset_builder._pending_replay_paths(
            replay_paths,
            done_set={"already_done.bsor"},
        )

        self.assertEqual(pending, [replay_paths[1]])


if __name__ == "__main__":
    unittest.main()
