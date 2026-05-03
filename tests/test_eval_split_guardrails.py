import unittest
from unittest import mock

from cybernoodles.training import policy_eval
from cybernoodles.training.eval_splits import (
    filter_curriculum_by_split,
    split_hashes_for,
    split_name_for_hash,
    stable_split_for_hash,
    validate_eval_splits,
)


def _beatmap(note_count=96):
    return {
        "notes": [
            {"time": float(index) * 1.2, "type": index % 2}
            for index in range(note_count)
        ],
        "obstacles": [],
    }


class EvalSplitGuardrailTests(unittest.TestCase):
    def test_explicit_split_overlap_is_rejected(self):
        splits = {
            "train": ["abc"],
            "dev_eval": ["def", "ABC"],
        }

        with self.assertRaisesRegex(ValueError, "appears in both"):
            validate_eval_splits(splits)

    def test_filter_curriculum_uses_explicit_split_membership(self):
        curriculum = [{"hash": "AAA"}, {"hash": "BBB"}, {"hash": "CCC"}]
        splits = {
            "train": ["aaa", "ccc"],
            "dev_eval": ["bbb"],
        }

        self.assertEqual(
            filter_curriculum_by_split(curriculum, "dev_eval", splits=splits),
            [{"hash": "BBB"}],
        )
        self.assertEqual(split_hashes_for(curriculum, "train", splits=splits), ["aaa", "ccc"])

    def test_stable_hash_split_is_deterministic(self):
        self.assertEqual(stable_split_for_hash("some-map-hash"), stable_split_for_hash("SOME-MAP-HASH"))
        self.assertEqual(split_name_for_hash("some-map-hash", splits={}), stable_split_for_hash("some-map-hash"))

    def test_choose_eval_hashes_honors_split_and_exclusions(self):
        curriculum = [
            {"hash": "train-a", "nps": 1.4},
            {"hash": "dev-a", "nps": 1.6},
            {"hash": "dev-b", "nps": 2.2},
        ]
        map_cache = {item["hash"]: (_beatmap(), 120.0) for item in curriculum}

        def fake_filter(records, split, allow_fallback=False):
            self.assertEqual(split, "dev_eval")
            return [item for item in records if item["hash"].startswith("dev-")]

        with mock.patch.object(policy_eval, "filter_curriculum_by_split", side_effect=fake_filter):
            hashes = policy_eval.choose_eval_hashes(
                curriculum,
                max_maps=1,
                map_cache=map_cache,
                suite="starter",
                split="dev_eval",
                exclude_hashes={"dev-a"},
            )

        self.assertEqual(hashes, ["dev-b"])


if __name__ == "__main__":
    unittest.main()
