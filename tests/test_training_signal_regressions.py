import os
import tempfile
import unittest
from argparse import Namespace
from unittest import mock

import torch

from cybernoodles.core.network import ACTION_DIM, CURRENT_POSE_START, INPUT_DIM, NOTE_TIME_INDEX, NOTE_TYPE_INDEX
from cybernoodles.envs import make_vector_env
from cybernoodles.training.policy_eval import (
    compute_completion_ratios,
    compute_target_note_counts,
    get_eval_profile,
    policy_action_log_prob,
    sample_policy_action,
)
from cybernoodles.training.train_awac import (
    awac_eval_key_has_regressed,
    bootstrap_has_play_signal,
    choose_training_stage,
    eval_profiles_match,
    format_eval_signal,
    matched_bootstrap_has_play_signal,
    normalize_rollout_metrics,
    save_awac_artifacts,
    seed_awac_best_eval_key,
    update_trainer_state_from_eval,
)
from cybernoodles.training.train_rl_gpu import (
    awac_bootstrap_has_strict_signal,
    choose_default_bootstrap_actor_path,
    env_flag_enabled,
    get_recovery_profile,
    live_training_signal,
    reference_delta_direction_loss,
    select_map_pools,
    should_run_bc_baseline_probe,
    should_run_epoch_eval_probe,
    summarize_awac_bootstrap_signal,
    update_adaptive_state,
)
from cybernoodles.training.train_bc import (
    BC_LOSS_PRESETS,
    bc_probe_key_has_regressed,
    probe_sort_key,
    resolve_bc_loss_weights,
    sample_weights_from_state,
    saber_tip_direction_loss,
    should_save_bc_last_checkpoint,
)


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


class TrainingSignalRegressionTests(unittest.TestCase):
    def _make_sim(self):
        env = make_vector_env(
            num_envs=1,
            device="cpu",
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
        )
        env.load_maps([TEST_BEATMAP], [60.0])
        return env.simulator

    def test_target_counts_respect_partial_start_times(self):
        sim = self._make_sim()
        start_times = torch.tensor([1.5], dtype=torch.float32)
        sim.reset(start_times=start_times)

        counts = compute_target_note_counts(sim, start_times=start_times)
        self.assertEqual(float(counts[0].item()), 1.0)

    def test_completion_is_relative_to_rollout_start(self):
        sim = self._make_sim()
        start_times = torch.tensor([1.0], dtype=torch.float32)
        sim.reset(start_times=start_times)
        sim.current_times.fill_(3.0)

        completion = compute_completion_ratios(sim, start_times=start_times)
        expected = (3.0 - float(start_times[0].item())) / max(
            1e-6,
            float(sim.map_durations[0].item()) - float(start_times[0].item()),
        )
        self.assertAlmostEqual(float(completion[0].item()), expected, places=5)

    def test_bc_sample_weights_do_not_treat_missing_note_sentinel_as_imminent(self):
        state = torch.zeros((3, INPUT_DIM), dtype=torch.float32)
        state[:, NOTE_TIME_INDEX] = torch.tensor([0.0, 0.0, 0.45])
        state[:, NOTE_TYPE_INDEX] = torch.tensor([-1.0, 0.0, 1.0])

        weights = sample_weights_from_state(state)

        self.assertAlmostEqual(float(weights[0].item()), 1.0, places=6)
        self.assertGreater(float(weights[1].item()), float(weights[0].item()))
        self.assertGreater(float(weights[1].item()), float(weights[2].item()))

    def test_bc_cut_direction_loss_penalizes_wrong_swing_direction(self):
        state = torch.zeros((2, INPUT_DIM), dtype=torch.float32)
        state[:, NOTE_TIME_INDEX] = 0.0
        state[:, NOTE_TYPE_INDEX] = 0.0
        state[:, 4] = 1.0

        current_pose = state[:, CURRENT_POSE_START:CURRENT_POSE_START + 21]
        correct_pred = current_pose[0:1].clone()
        wrong_pred = current_pose[1:2].clone()
        correct_pred[:, 7] = 0.12
        wrong_pred[:, 7] = -0.12

        correct_loss, _ = saber_tip_direction_loss(correct_pred, state[0:1])
        wrong_loss, _ = saber_tip_direction_loss(wrong_pred, state[1:2])

        self.assertLess(float(correct_loss.item()), 0.05)
        self.assertGreater(float(wrong_loss.item()), 1.5)

    def test_bc_probe_key_keeps_baseline_above_collapsed_lower_val_loss(self):
        baseline_key = probe_sort_key(
            {
                "mean_note_coverage": 0.060,
                "mean_completion": 0.140,
                "mean_clear_rate": 0.0,
                "mean_accuracy": 3.20,
                "mean_resolved_accuracy": 24.0,
                "mean_engaged_accuracy": 48.0,
                "mean_resolved_coverage": 0.040,
                "mean_cut": 0.030,
            },
            {"loss": 0.50},
        )
        collapsed_key = probe_sort_key(
            {
                "mean_note_coverage": 0.002,
                "mean_completion": 0.020,
                "mean_clear_rate": 0.0,
                "mean_accuracy": 0.20,
                "mean_resolved_accuracy": 5.0,
                "mean_engaged_accuracy": 20.0,
                "mean_resolved_coverage": 0.001,
                "mean_cut": 0.001,
            },
            {"loss": 0.01},
        )

        self.assertGreater(baseline_key, collapsed_key)

    def test_bc_probe_regression_detects_collapse_but_tolerates_mild_drift(self):
        best_key = (
            1,
            1,
            0.060,
            0.140,
            0.0,
            3.20,
            24.0,
            48.0,
            0.040,
            0.030,
            -0.50,
        )
        collapsed_key = (
            0,
            0,
            0.002,
            0.020,
            0.0,
            0.20,
            5.0,
            20.0,
            0.001,
            0.001,
            -0.01,
        )
        mild_drift_key = (
            1,
            1,
            0.055,
            0.132,
            0.0,
            3.00,
            23.0,
            47.0,
            0.038,
            0.028,
            -0.01,
        )

        self.assertTrue(bc_probe_key_has_regressed(collapsed_key, best_key))
        self.assertFalse(bc_probe_key_has_regressed(mild_drift_key, best_key))

    def test_bc_last_checkpoint_save_skips_probe_collapse_only(self):
        best_key = (1, 1, 0.060, 0.140, 0.0, 3.20, 24.0, 48.0, 0.040, 0.030, -0.50)
        collapsed_key = (0, 0, 0.002, 0.020, 0.0, 0.20, 5.0, 20.0, 0.001, 0.001, -0.01)
        mild_drift_key = (1, 1, 0.055, 0.132, 0.0, 3.00, 23.0, 47.0, 0.038, 0.028, -0.01)

        self.assertFalse(should_save_bc_last_checkpoint(True, collapsed_key, best_key))
        self.assertTrue(should_save_bc_last_checkpoint(True, mild_drift_key, best_key))
        self.assertTrue(should_save_bc_last_checkpoint(False, collapsed_key, best_key))
        self.assertTrue(should_save_bc_last_checkpoint(True, None, best_key))
        self.assertTrue(should_save_bc_last_checkpoint(True, collapsed_key, None))

    def test_bc_cut_preset_keeps_pose_note_anchors_and_rejects_unknowns(self):
        balanced = BC_LOSS_PRESETS["balanced"]
        cut = resolve_bc_loss_weights("cut")

        self.assertGreater(cut["tip"], balanced["tip"])
        self.assertGreater(cut["swing"], balanced["swing"])
        self.assertGreaterEqual(cut["pos"], 0.75)
        self.assertGreaterEqual(cut["motion"], balanced["motion"])
        self.assertGreaterEqual(cut["note"], balanced["note"])
        self.assertLess(cut["direction"], cut["note"])
        self.assertLess(cut["direction"], cut["tip"])

        with self.assertRaisesRegex(ValueError, "Unknown BC loss preset"):
            resolve_bc_loss_weights("unknown")

    def test_policy_action_log_prob_uses_raw_action_not_sanitized_sim_action(self):
        mean = torch.zeros((1, ACTION_DIM), dtype=torch.float32)
        std = torch.full_like(mean, 0.5)
        noise = torch.zeros_like(mean)

        raw_action, sim_action, raw_log_prob, stats = sample_policy_action(
            mean,
            std,
            noise=noise,
            return_stats=True,
        )

        expected_raw_log_prob = policy_action_log_prob(mean, std, raw_action)
        sanitized_log_prob = policy_action_log_prob(mean, std, sim_action)

        self.assertGreater(float(stats["changed_fraction"].item()), 0.0)
        self.assertTrue(torch.allclose(raw_log_prob, expected_raw_log_prob))
        self.assertFalse(torch.allclose(raw_log_prob, sanitized_log_prob))

    def test_awac_handoff_requires_real_strict_signal(self):
        weak_signal = summarize_awac_bootstrap_signal({
            "last_strict_accuracy": 0.0,
            "best_strict_accuracy": 0.25,
            "last_strict_note_coverage": 0.0,
            "best_strict_coverage": 0.005,
        })
        self.assertFalse(awac_bootstrap_has_strict_signal(weak_signal))

        ready_signal = summarize_awac_bootstrap_signal({
            "last_strict_accuracy": 0.0,
            "best_strict_accuracy": 0.75,
            "last_strict_note_coverage": 0.0,
            "best_strict_coverage": 0.02,
        })
        self.assertTrue(awac_bootstrap_has_strict_signal(ready_signal))

        awac_path, awac_label = choose_default_bootstrap_actor_path({
            "actor_path": "bsai_awac_model.pth",
            "usable": True,
        })
        self.assertEqual(awac_path, "bsai_awac_model.pth")
        self.assertEqual(awac_label, "AWAC bootstrap")

    def test_default_awac_matched_profile_matches_strict_profile(self):
        strict_profile = get_eval_profile("strict")
        args = Namespace(
            training_wheels_level=0.0,
            assist_level=0.0,
            survival_assistance=0.0,
            stability_reward_level=0.0,
            style_guidance_level=0.0,
            fail_enabled=True,
            saber_inertia=0.0,
            rot_clamp=0.07,
            pos_clamp=0.12,
        )
        matched_profile = {
            "action_repeat": 1,
            "smoothing_alpha": 1.0,
            "training_wheels_level": float(args.training_wheels_level),
            "assist_level": float(args.assist_level),
            "survival_assistance": float(args.survival_assistance),
            "stability_reward_level": float(args.stability_reward_level),
            "style_guidance_level": float(args.style_guidance_level),
            "fail_enabled": bool(args.fail_enabled),
            "saber_inertia": float(args.saber_inertia),
            "rot_clamp": float(args.rot_clamp),
            "pos_clamp": float(args.pos_clamp),
        }
        self.assertTrue(eval_profiles_match(strict_profile, matched_profile))

        matched_profile["assist_level"] = 0.1
        self.assertFalse(eval_profiles_match(strict_profile, matched_profile))

    def test_identical_matched_profile_uses_strict_bootstrap_gate(self):
        summary = {
            "mean_accuracy": 2.48,
            "mean_note_coverage": 0.026,
            "mean_resolved_accuracy": 31.08,
            "mean_completion": 0.07,
            "mean_clear_rate": 0.0,
        }
        self.assertFalse(matched_bootstrap_has_play_signal(summary, profile_matches_strict=False))
        self.assertTrue(matched_bootstrap_has_play_signal(summary, profile_matches_strict=True))

    def test_strict_bootstrap_gate_tolerates_display_rounding(self):
        summary = {
            "mean_accuracy": 0.505,
            "mean_note_coverage": 0.0106,
            "mean_resolved_accuracy": 15.09,
            "mean_engaged_accuracy": 47.06,
            "mean_completion": 0.0496,
            "mean_clear_rate": 0.0,
        }
        self.assertTrue(bootstrap_has_play_signal(summary, strict=True))
        self.assertIn("comp 0.0496", format_eval_signal(summary))

        summary["mean_completion"] = 0.047
        self.assertFalse(bootstrap_has_play_signal(summary, strict=True))

    def test_rollout_metrics_are_normalized_to_task_accuracy(self):
        metrics = normalize_rollout_metrics({
            "accuracy": 2.5,
            "engaged_accuracy": 90.0,
            "resolved_coverage": 0.1,
        })
        self.assertEqual(metrics["task_accuracy"], 2.5)
        self.assertEqual(metrics["accuracy"], 2.5)

    def test_periodic_awac_save_does_not_overwrite_best_actor_model(self):
        class _DummyModule:
            def state_dict(self):
                return {"w": torch.tensor([1.0])}

        actor = _DummyModule()
        critics = _DummyModule()
        target_actor = _DummyModule()
        target_critics = _DummyModule()
        actor_optimizer = _DummyModule()
        critic_optimizer = _DummyModule()
        args = Namespace(storage_dtype=torch.float16, foo="bar")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "bsai_awac_model.pth")
            checkpoint_path = os.path.join(tmpdir, "bsai_awac_checkpoint.pth")
            state_path = os.path.join(tmpdir, "awac_state.json")
            saved_paths = []

            def _fake_torch_save(payload, path):
                saved_paths.append(path)

            with mock.patch("cybernoodles.training.train_awac.AWAC_MODEL_PATH", model_path), \
                mock.patch("cybernoodles.training.train_awac.AWAC_CHECKPOINT_PATH", checkpoint_path), \
                mock.patch("cybernoodles.training.train_awac.AWAC_STATE_PATH", state_path), \
                mock.patch("cybernoodles.training.train_awac.attach_policy_schema", side_effect=lambda payload: payload), \
                mock.patch("cybernoodles.training.train_awac.build_rl_bootstrap_state_dict", return_value={"boot": 1}), \
                mock.patch("cybernoodles.training.train_awac.save_awac_state"), \
                mock.patch("cybernoodles.training.train_awac.torch.save", side_effect=_fake_torch_save):
                save_awac_artifacts(
                    actor,
                    critics,
                    target_actor,
                    target_critics,
                    actor_optimizer,
                    critic_optimizer,
                    {"epoch": 1},
                    args,
                    save_actor_model=False,
                )

            self.assertEqual(saved_paths, [checkpoint_path])

    def test_awac_eval_regression_detects_strict_collapse(self):
        best_key = (
            1,
            0.034,
            3.05,
            0.19,
            0.0,
            23.0,
            48.0,
            1,
            0.034,
            3.05,
            0.19,
            0.0,
            23.0,
            48.0,
        )
        collapsed_key = (
            0,
            0.001,
            0.12,
            0.19,
            0.0,
            4.17,
            33.33,
            0,
            0.001,
            0.12,
            0.19,
            0.0,
            4.17,
            33.33,
        )
        mild_regression_key = (
            1,
            0.030,
            2.95,
            0.18,
            0.0,
            22.0,
            47.0,
            1,
            0.030,
            2.95,
            0.18,
            0.0,
            22.0,
            47.0,
        )

        self.assertTrue(awac_eval_key_has_regressed(collapsed_key, best_key))
        self.assertFalse(awac_eval_key_has_regressed(mild_regression_key, best_key))

    def test_awac_baseline_seeds_best_eval_key_before_first_epoch_eval(self):
        trainer_state = {}
        baseline_strict = {
            "mean_accuracy": 1.54,
            "mean_note_coverage": 0.027,
            "mean_resolved_accuracy": 33.04,
            "mean_engaged_accuracy": 57.81,
            "mean_completion": 0.06,
            "mean_clear_rate": 0.0,
        }
        collapsed_strict = {
            "mean_accuracy": 0.12,
            "mean_note_coverage": 0.001,
            "mean_resolved_accuracy": 4.76,
            "mean_engaged_accuracy": 33.33,
            "mean_completion": 0.19,
            "mean_clear_rate": 0.0,
        }

        best_key, seeded = seed_awac_best_eval_key(
            trainer_state,
            None,
            baseline_strict,
            baseline_strict,
            matched_profile_is_strict=True,
        )
        collapsed_key, collapsed_seeded = seed_awac_best_eval_key(
            trainer_state,
            best_key,
            collapsed_strict,
            collapsed_strict,
            matched_profile_is_strict=True,
        )

        self.assertTrue(seeded)
        self.assertFalse(collapsed_seeded)
        self.assertEqual(collapsed_key, best_key)
        self.assertEqual(trainer_state["best_eval_key"], list(best_key))

    def test_awac_stage_demotes_on_latest_strict_regression(self):
        args = Namespace(
            strict_expand_coverage=0.08,
            strict_expand_accuracy=8.0,
            strict_unlock_coverage=0.03,
            strict_unlock_accuracy=3.0,
        )
        trainer_state = {
            "last_strict_accuracy": 4.06,
            "last_strict_note_coverage": 0.041,
            "best_strict_accuracy": 13.03,
            "best_strict_coverage": 0.139,
        }
        stage, strict_accuracy, strict_coverage = choose_training_stage(trainer_state, args)
        self.assertEqual(stage, "bridge")
        self.assertAlmostEqual(strict_accuracy, 4.06, places=2)
        self.assertAlmostEqual(strict_coverage, 0.041, places=3)

    def test_awac_stage_does_not_unlock_on_coverage_alone(self):
        args = Namespace(
            strict_expand_coverage=0.08,
            strict_expand_accuracy=8.0,
            strict_unlock_coverage=0.03,
            strict_unlock_accuracy=3.0,
        )
        stage, strict_accuracy, strict_coverage = choose_training_stage(
            {
                "last_strict_accuracy": 2.35,
                "last_strict_note_coverage": 0.034,
            },
            args,
        )
        self.assertEqual(stage, "warmup")
        self.assertAlmostEqual(strict_accuracy, 2.35, places=2)
        self.assertAlmostEqual(strict_coverage, 0.034, places=3)

    def test_current_preflight_overrides_stale_last_strict_state(self):
        trainer_state, best_acc, best_cov = update_trainer_state_from_eval(
            {
                "last_strict_accuracy": 13.03,
                "last_strict_note_coverage": 0.139,
                "best_strict_accuracy": 13.03,
                "best_strict_coverage": 0.139,
            },
            {
                "mean_accuracy": 4.36,
                "mean_note_coverage": 0.044,
            },
            {
                "mean_accuracy": 4.36,
                "mean_note_coverage": 0.044,
            },
        )
        self.assertAlmostEqual(trainer_state["last_strict_accuracy"], 4.36, places=2)
        self.assertAlmostEqual(trainer_state["last_strict_note_coverage"], 0.044, places=3)
        self.assertAlmostEqual(best_acc, 13.03, places=2)
        self.assertAlmostEqual(best_cov, 0.139, places=3)

    def test_env_flag_enabled_accepts_common_truthy_spellings(self):
        with mock.patch.dict(os.environ, {"BSAI_TEST_FLAG": "yes"}, clear=False):
            self.assertTrue(env_flag_enabled("BSAI_TEST_FLAG"))
        with mock.patch.dict(os.environ, {"BSAI_TEST_FLAG": "0"}, clear=False):
            self.assertFalse(env_flag_enabled("BSAI_TEST_FLAG"))
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(env_flag_enabled("BSAI_TEST_FLAG", default=True))

    def test_skip_startup_probe_flags_only_disable_epoch_zero_diagnostics(self):
        self.assertTrue(should_run_bc_baseline_probe(object(), ["a"], skip_bc_probe=False))
        self.assertFalse(should_run_bc_baseline_probe(object(), ["a"], skip_bc_probe=True))
        self.assertFalse(should_run_bc_baseline_probe(None, ["a"], skip_bc_probe=False))

        self.assertTrue(should_run_epoch_eval_probe(0, 100, ["a"], skip_initial_eval=False))
        self.assertFalse(should_run_epoch_eval_probe(0, 100, ["a"], skip_initial_eval=True))
        self.assertFalse(should_run_epoch_eval_probe(1, 100, ["a"], skip_initial_eval=False))
        self.assertTrue(should_run_epoch_eval_probe(99, 100, ["a"], skip_initial_eval=True))
        self.assertFalse(should_run_epoch_eval_probe(99, 100, [], skip_initial_eval=False))

    def test_live_training_signal_prefers_current_progress_over_historical_best(self):
        adaptive_state = {
            "strict_eval_accuracy": 0.0,
            "matched_eval_accuracy": 0.0,
            "last_eval_accuracy": 0.0,
        }
        signal = live_training_signal(
            adaptive_state=adaptive_state,
            current_task_acc=4.1,
            fallback=66.94,
        )
        self.assertAlmostEqual(signal, 4.1, places=2)

    def test_recovery_profile_uses_live_signal_instead_of_historical_best(self):
        recovery = get_recovery_profile(
            {
                "rehab_level": 0,
                "stability_rehab_level": 4,
                "style_rehab_level": 4,
                "last_mean_stability": 0.21,
                "last_mean_note_coverage": 0.074,
                "last_mean_energy": 0.08,
                "last_mean_fail_rate": 0.80,
            },
            global_best_acc=66.94,
            current_task_acc=4.1,
        )
        self.assertGreaterEqual(recovery["training_wheels"], 0.75)
        self.assertGreaterEqual(recovery["assist_level"], 0.65)

    def test_adaptive_state_can_raise_rehab_after_live_collapse(self):
        adaptive_state = update_adaptive_state(
            {
                "rehab_level": 0,
                "stability_rehab_level": 0,
                "style_rehab_level": 0,
                "stagnation_epochs": 12,
                "stability_stagnation_epochs": 6,
                "best_signal_accuracy": 66.94,
                "run_best_signal_accuracy": 66.94,
                "last_progress_epoch": 20,
                "bc_probe_accuracy": 57.70,
                "strict_eval_accuracy": 0.0,
                "matched_eval_accuracy": 0.0,
                "last_eval_accuracy": 0.0,
                "best_eval_accuracy": 0.0,
                "best_matched_eval_accuracy": 0.0,
                "best_stability": 0.45,
                "run_best_stability": 0.45,
                "rehab_release_streak": 0,
                "escape_release_streak": 0,
                "escape_support_active": False,
            },
            epoch=38,
            global_best_acc=66.94,
            strict_eval_acc=None,
            matched_eval_acc=None,
            current_task_acc=4.1,
            mean_stability=0.21,
            mean_note_coverage=0.074,
            mean_energy=0.08,
            mean_completion=0.08,
            mean_fail_rate=0.80,
            mean_motion_efficiency=0.13,
            mean_idle_motion=0.048,
            mean_guard_error=3.448,
            mean_oscillation=0.011,
            mean_lateral_motion=0.045,
        )
        self.assertGreaterEqual(adaptive_state["rehab_level"], 1)

    def test_curriculum_demotes_when_live_signal_collapses(self):
        tribe = mock.Mock()
        tribe.id = 1
        tribe.last_task_accuracy = 4.1
        tribe.moving_acc = 4.1
        tribe.last_note_coverage = 0.074
        buckets = {
            "micro": ["m"],
            "bootstrap": ["b"],
            "easy": ["e"],
            "medium": ["md"],
            "hard": ["h"],
            "expert": ["x"],
            "all": ["all"],
            "micro_seen": ["m"],
            "bootstrap_seen": ["b"],
            "easy_seen": ["e"],
            "medium_seen": ["md"],
            "micro_clean": ["mc"],
            "bootstrap_clean": ["bc"],
            "easy_clean": ["ec"],
            "micro_clean_seen": ["mcs"],
            "bootstrap_clean_seen": ["bcs"],
            "easy_clean_seen": ["ecs"],
        }
        primary, fallback, tier = select_map_pools(
            tribe,
            buckets,
            global_best_acc=66.94,
            rehab_level=0,
            stagnation_epochs=14,
            current_task_acc=4.1,
        )
        self.assertEqual(tier, "EASY")
        self.assertEqual(primary, ["ecs"])

    def test_reference_delta_direction_loss_includes_right_hand(self):
        current_pose = torch.zeros(1, 21)
        mean = current_pose.clone()
        ref_mean = current_pose.clone()
        mean[:, 7:10] = torch.tensor([[1.0, 0.0, 0.0]])
        ref_mean[:, 7:10] = torch.tensor([[1.0, 0.0, 0.0]])
        mean[:, 14:17] = torch.tensor([[0.0, 1.0, 0.0]])
        ref_mean[:, 14:17] = torch.tensor([[0.0, -1.0, 0.0]])

        loss = reference_delta_direction_loss(mean, ref_mean, current_pose)

        self.assertGreater(float(loss.item()), 0.9)


if __name__ == "__main__":
    unittest.main()
