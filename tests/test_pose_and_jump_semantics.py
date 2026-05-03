import unittest

import numpy as np
import torch

from cybernoodles.core.gpu_simulator import BC_RESET_POSE
from cybernoodles.core.jump_timing import compute_spawn_ahead_beats
from cybernoodles.core.network import (
    ActorCritic,
    INPUT_DIM,
    NOTE_FEATURES,
    NOTE_TYPE_INDEX,
    normalize_pose_quaternions,
)
from cybernoodles.core.pose_defaults import DEFAULT_TRACKED_POSE
from cybernoodles.data.dataset_builder import DEFAULT_POSE, _build_note_feature_vector
from cybernoodles.envs import make_vector_env


class PoseAndJumpSemanticsTests(unittest.TestCase):
    def _make_direction_cut_sim(self, second_note=False):
        notes = [
            {
                "time": 1.0,
                "lineIndex": 1,
                "lineLayer": 1,
                "type": 0,
                "cutDirection": 0,
            }
        ]
        if second_note:
            notes.append({
                "time": 1.5,
                "lineIndex": 2,
                "lineLayer": 1,
                "type": 1,
                "cutDirection": 1,
            })
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
            external_pose_passthrough=True,
        )
        env.load_maps([{"notes": notes, "obstacles": [], "njs": 18.0, "offset": 0.0}], [60.0])
        return env.simulator

    def _swing_left_hand(self, sim, start_y, end_y):
        sim.reset(start_times=torch.tensor([0.97], dtype=torch.float32))
        start_pose = torch.tensor(DEFAULT_TRACKED_POSE, dtype=torch.float32).view(1, -1)
        end_pose = start_pose.clone()
        start_pose[:, 7:10] = torch.tensor([[-0.30, start_y, 0.35]], dtype=torch.float32)
        end_pose[:, 7:10] = torch.tensor([[-0.30, end_y, 0.35]], dtype=torch.float32)
        sim.poses.copy_(start_pose)
        sim.prev_poses.copy_(start_pose)
        sim.pose_history.copy_(start_pose.unsqueeze(1).expand_as(sim.pose_history))
        sim.step(end_pose, dt=1.0 / 60.0)

    def test_shared_default_pose_stays_aligned(self):
        self.assertTupleEqual(tuple(DEFAULT_TRACKED_POSE), tuple(BC_RESET_POSE))
        np.testing.assert_allclose(DEFAULT_POSE, np.asarray(DEFAULT_TRACKED_POSE, dtype=np.float32))

    def test_spawn_ahead_beats_matches_reference_example(self):
        self.assertAlmostEqual(compute_spawn_ahead_beats(120.0, 16.0, 0.0), 2.0, places=6)
        self.assertAlmostEqual(compute_spawn_ahead_beats(120.0, 16.0, -0.5), 1.5, places=6)
        self.assertAlmostEqual(compute_spawn_ahead_beats(60.0, 10.0, -0.75), 1.0, places=6)

    def test_dataset_builder_hides_notes_until_spawn_window(self):
        notes = [
            {
                "time": 2.0,
                "lineIndex": 1,
                "lineLayer": 1,
                "type": 0,
                "cutDirection": 1,
            }
        ]

        hidden = _build_note_feature_vector(
            notes,
            note_idx=0,
            t_beat=0.0,
            bps=2.0,
            note_jump_speed=16.0,
            head_z=0.0,
            spawn_ahead_beats=1.5,
        )
        visible = _build_note_feature_vector(
            notes,
            note_idx=0,
            t_beat=0.0,
            bps=2.0,
            note_jump_speed=16.0,
            head_z=0.0,
            spawn_ahead_beats=2.5,
        )

        self.assertEqual(hidden[NOTE_TYPE_INDEX], -1.0)
        self.assertEqual(visible[NOTE_TYPE_INDEX], 0.0)

    def test_actor_forward_returns_unit_quaternions(self):
        model = ActorCritic()
        batch = torch.zeros(8, INPUT_DIM)
        mean, _, _ = model(batch)

        for start, end in ((3, 7), (10, 14), (17, 21)):
            norms = torch.norm(mean[:, start:end], dim=-1)
            self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_actor_forward_supports_backward_through_quaternion_normalization(self):
        model = ActorCritic()
        batch = torch.randn(16, INPUT_DIM)
        mean, _, value = model(batch)
        loss = mean.square().mean() + value.square().mean()
        loss.backward()

        self.assertIsNotNone(model.actor_mean.weight.grad)
        self.assertTrue(torch.isfinite(model.actor_mean.weight.grad).all())

    def test_normalize_pose_quaternions_falls_back_to_identity_for_zero_quats(self):
        pose = torch.zeros(2, 21)
        pose[:, 0:3] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pose[:, 7:10] = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        pose[:, 14:17] = torch.tensor([[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]])

        normalized = normalize_pose_quaternions(pose)
        expected_identity = torch.tensor([0.0, 0.0, 0.0, 1.0])

        self.assertTrue(torch.equal(normalized[:, 0:3], pose[:, 0:3]))
        self.assertTrue(torch.equal(normalized[:, 7:10], pose[:, 7:10]))
        self.assertTrue(torch.equal(normalized[:, 14:17], pose[:, 14:17]))
        for start, end in ((3, 7), (10, 14), (17, 21)):
            self.assertTrue(torch.allclose(normalized[:, start:end], expected_identity.expand(2, -1)))

    def test_simulator_normalizes_action_quaternions_before_delta_clamp(self):
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
        env.load_maps([{"notes": [], "obstacles": [], "njs": 18.0, "offset": 0.0}], [120.0])
        sim = env.simulator

        action = sim.poses.clone()
        action[:, 3:7] *= 2.0
        action[:, 10:14] *= 3.0
        action[:, 17:21] *= 4.0

        sim.step(action, dt=1.0 / 60.0)

        self.assertLess(float(sim._delta_out.abs().max().item()), 1e-6)

    def test_wrong_direction_contact_is_bad_cut_not_hit(self):
        sim = self._make_direction_cut_sim()

        self._swing_left_hand(sim, start_y=1.45, end_y=0.95)

        self.assertEqual(float(sim.total_hits[0].item()), 0.0)
        self.assertEqual(float(sim.total_badcuts[0].item()), 1.0)
        self.assertFalse(bool(sim.note_active[0, 0].item()))

    def test_resolved_note_can_remain_as_non_scorable_followthrough_context(self):
        sim = self._make_direction_cut_sim(second_note=True)

        self._swing_left_hand(sim, start_y=0.95, end_y=1.45)
        state = sim.get_states()

        self.assertEqual(float(sim.total_hits[0].item()), 1.0)
        self.assertFalse(bool(sim.note_active[0, 0].item()))
        self.assertTrue(bool(sim.note_active[0, 1].item()))
        self.assertEqual(float(state[0, NOTE_TYPE_INDEX].item()), 0.0)
        self.assertEqual(float(state[0, NOTE_TYPE_INDEX + NOTE_FEATURES].item()), 1.0)

    def test_post_hit_followthrough_frame_cannot_score_resolved_note_again(self):
        sim = self._make_direction_cut_sim()

        self._swing_left_hand(sim, start_y=0.95, end_y=1.45)
        hits_after_first_cut = sim.total_hits.clone()
        resolved_after_first_cut = sim.total_resolved_scorable.clone()
        badcuts_after_first_cut = sim.total_badcuts.clone()

        followthrough_pose = sim.poses.clone()
        followthrough_pose[:, 7:10] = torch.tensor([[-0.30, 0.95, 0.35]], dtype=torch.float32)
        sim.step(followthrough_pose, dt=1.0 / 60.0)

        self.assertTrue(torch.equal(sim.total_hits, hits_after_first_cut))
        self.assertTrue(torch.equal(sim.total_resolved_scorable, resolved_after_first_cut))
        self.assertTrue(torch.equal(sim.total_badcuts, badcuts_after_first_cut))


if __name__ == "__main__":
    unittest.main()
