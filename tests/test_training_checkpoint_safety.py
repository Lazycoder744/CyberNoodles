import importlib
import os
import sys
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn as nn

from cybernoodles.training import train_awac
from cybernoodles.training.policy_checkpoint import attach_policy_schema
from cybernoodles.training.policy_eval import remap_state_dict


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(2, 2)


def _load_train_rl_gpu():
    return importlib.import_module("cybernoodles.training.train_rl_gpu")


class TrainingCheckpointSafetyTests(unittest.TestCase):
    def test_policy_eval_remap_rejects_zero_usable_keys(self):
        model = TinyPolicy()
        bad_state = {"unrelated.weight": torch.ones(2, 2)}

        with self.assertRaisesRegex(RuntimeError, "usable model weights"):
            remap_state_dict(bad_state, model)

    def test_ppo_remap_rejects_zero_usable_keys(self):
        train_rl_gpu = _load_train_rl_gpu()
        model = TinyPolicy()
        bad_state = {"unrelated.weight": torch.ones(2, 2)}

        with self.assertRaisesRegex(RuntimeError, "usable model weights"):
            train_rl_gpu._remap_state_dict(bad_state, model)

    def test_awac_resume_requires_optimizer_state_dicts(self):
        payload = attach_policy_schema({
            "actor_state_dict": {},
            "critic_state_dict": {},
            "target_actor_state_dict": {},
            "target_critic_state_dict": {},
            "critic_optimizer_state_dict": {},
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "bsai_awac_checkpoint.pth")
            torch.save(payload, checkpoint_path)

            with mock.patch.object(train_awac, "AWAC_CHECKPOINT_PATH", checkpoint_path):
                with self.assertRaisesRegex(RuntimeError, "actor_optimizer_state_dict"):
                    train_awac.read_awac_resume("cpu")

    def test_ppo_periodic_checkpoint_does_not_promote_actor_model(self):
        train_rl_gpu = _load_train_rl_gpu()
        tribe = train_rl_gpu.Tribe(0, 1, torch.device("cpu"))

        with tempfile.TemporaryDirectory() as tmpdir:
            actor_path = os.path.join(tmpdir, "cybernoodles_rl_model.pth")
            checkpoint_path = os.path.join(tmpdir, "cybernoodles_rl_checkpoint.pth")
            state_path = os.path.join(tmpdir, "rl_state.json")

            with (
                mock.patch.object(train_rl_gpu, "RL_CHECKPOINT_PATH", checkpoint_path),
                mock.patch.object(train_rl_gpu, "TRAINER_STATE_PATH", state_path),
            ):
                train_rl_gpu.save_training_artifacts(
                    tribe,
                    [tribe],
                    actor_path,
                    epoch=7,
                    state={"epoch": 7},
                    promote_actor=False,
                )

            self.assertFalse(os.path.exists(actor_path))
            self.assertTrue(os.path.exists(checkpoint_path))
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertFalse(payload["promoted_actor"])
            self.assertEqual(payload["epoch"], 7)

    def test_train_rl_gpu_import_does_not_patch_inductor_write_atomic(self):
        try:
            import torch._inductor.codecache as codecache
        except Exception as exc:
            raise unittest.SkipTest(f"PyTorch Inductor codecache unavailable: {exc}") from exc

        original = codecache.write_atomic

        def sentinel_write_atomic(*args, **kwargs):
            raise AssertionError("sentinel should not be called")

        codecache.write_atomic = sentinel_write_atomic
        try:
            with mock.patch.dict(
                os.environ,
                {
                    "BSAI_ENABLE_TORCH_COMPILE": "0",
                    "BSAI_PATCH_INDUCTOR_WRITE_ATOMIC": "0",
                },
                clear=False,
            ):
                module = sys.modules.get("cybernoodles.training.train_rl_gpu")
                if module is None:
                    importlib.import_module("cybernoodles.training.train_rl_gpu")
                else:
                    importlib.reload(module)

            self.assertIs(codecache.write_atomic, sentinel_write_atomic)
        finally:
            codecache.write_atomic = original

    def test_inductor_write_atomic_patch_requires_explicit_flag(self):
        try:
            import torch._inductor.codecache as codecache
        except Exception as exc:
            raise unittest.SkipTest(f"PyTorch Inductor codecache unavailable: {exc}") from exc

        train_rl_gpu = _load_train_rl_gpu()
        original = codecache.write_atomic

        def sentinel_write_atomic(*args, **kwargs):
            raise AssertionError("sentinel should not be called")

        codecache.write_atomic = sentinel_write_atomic
        try:
            with mock.patch.dict(os.environ, {"BSAI_PATCH_INDUCTOR_WRITE_ATOMIC": "0"}, clear=False):
                self.assertFalse(train_rl_gpu.maybe_patch_inductor_write_atomic())
                self.assertIs(codecache.write_atomic, sentinel_write_atomic)

            with mock.patch.dict(os.environ, {"BSAI_PATCH_INDUCTOR_WRITE_ATOMIC": "1"}, clear=False):
                self.assertTrue(train_rl_gpu.maybe_patch_inductor_write_atomic())
                self.assertIsNot(codecache.write_atomic, sentinel_write_atomic)
                self.assertTrue(getattr(codecache.write_atomic, "_bsai_replace_atomic_patch", False))
        finally:
            codecache.write_atomic = original


if __name__ == "__main__":
    unittest.main()
