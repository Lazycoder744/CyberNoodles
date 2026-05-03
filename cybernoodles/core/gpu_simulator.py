import torch
import numpy as np
from cybernoodles.core.jump_timing import (
    INITIAL_HALF_JUMP_BEATS,
    MAX_HALF_JUMP_DISTANCE_METERS,
    MIN_HALF_JUMP_BEATS,
)
from cybernoodles.core.network import (
    CUT_DIR_VECTORS,
    INPUT_DIM,
    NOTE_FEATURES,
    NOTES_DIM,
    OBSTACLE_FEATURES,
    NUM_UPCOMING_OBSTACLES,
    NUM_UPCOMING_NOTES,
    OBSTACLES_DIM,
    POSE_DIM,
    STATE_FRAME_DIM,
    STATE_HISTORY_OFFSETS,
    STATIC_HEAD,
    VELOCITY_DIM,
    normalize_pose_quaternions,
)
from cybernoodles.core.pose_defaults import DEFAULT_TRACKED_POSE
from cybernoodles.core.sim_cuda_ext import get_sim_cuda_extension
from cybernoodles.data.sim_calibration import load_simulator_calibration
from cybernoodles.data.style_calibration import load_style_calibration

SABER_LENGTH = 1.0
SABER_CUT_START_FRACTION = 0.28
SABER_CONTACT_MARGIN = 0.05
ARROW_GOOD_HITBOX = (0.8, 0.5, 1.0)
DOT_GOOD_HITBOX = (0.8, 0.8, 1.0)
BAD_HITBOX = (0.4, 0.4, 0.4)
GOOD_HITBOX_Z_OFFSET = -0.25
BLADE_RIBBON_SCALE = 0.72
CUT_START_MARGIN = 0.35
BOMB_RADIUS = 0.18
HEAD_HITBOX_RADIUS = 0.18
HISTORY_LEN = 30
DEFAULT_DT = 1.0 / 60.0
TRACK_Z_BASE = 0.9
DENSE_APPROACH_SCALE = 2.0
CONTACT_DISCOVERY_SCALE = 0.15
SCORE_CAP_NORMAL = 115.0
SCORE_CAP_CHAIN_HEAD = 85.0
SCORE_CAP_CHAIN_LINK = 20.0
ACC_DISTANCE_SCALE = 0.35
BAD_CUT_ENERGY_SCALE = 0.65
WALL_DRAIN_PER_SEC = 0.40
FOLLOWTHROUGH_PREDICTION_FRAMES = 6.0
STATE_FOLLOWTHROUGH_BEATS = 0.35
ACTIVE_NOTE_WINDOW = 192
ACTIVE_NOTE_PAST = 32
ACTIVE_OBSTACLE_WINDOW = 128
BC_RESET_POSE = DEFAULT_TRACKED_POSE

REWARD_COMPONENT_NAMES = (
    "score_reward",
    "style_reward",
    "dense_approach_reward",
    "contact_reward",
    "energy_delta_reward",
    "survival_reward",
    "combo_bonus",
    "miss_penalty",
    "bomb_penalty",
    "wall_penalty",
    "motion_penalty",
    "style_penalty",
    "death_penalty",
    "total_reward",
)

ASSIST_STATE_NAMES = (
    "good_hitbox_scale",
    "bad_hitbox_scale",
    "controller_hit_mix",
    "speed_threshold",
    "direction_threshold",
    "hit_window_front",
    "hit_window_back",
    "miss_window_back",
    "contact_reward_weight",
    "dense_reward_weight",
    "start_energy",
    "miss_energy",
    "bomb_energy",
    "death_penalty",
    "fail_enabled",
)

class GPUBeatSaberSimulator:
    """
    CUDA-graph-compatible Beat Saber simulator.

    Rules enforced for CUDAGraph capture:
      - Every tensor read/written inside step()/get_states() lives at a FIXED
        memory address. No Python-level reassignment of tensors; all updates
        use in-place ops (.copy_(), .add_(), &=, index_put).
      - hist_ptr is a GPU tensor (not a Python int) so its increment is a
        captured CUDA kernel, not a Python side-effect frozen at capture time.
      - No dynamic tensor allocation inside the hot path (arange, full, etc.
        are pre-built in reset() and reused every step).
      - No .item() / CPU-GPU syncs inside step() or get_states().
    """

    def __init__(self, num_envs, device='cuda'):
        self.num_envs  = num_envs
        self.device    = device
        self.max_notes = 0
        self.max_obstacles = 0
        self._native_sim_ext = None
        if str(device).startswith('cuda') and torch.cuda.is_available():
            self._native_sim_ext = get_sim_cuda_extension()
        self._use_native_contact = False
        self._reserved_max_notes = 1
        self._reserved_max_obstacles = 1
        self._world_fwd = torch.tensor([0.0, 0.0, 1.0], device=device)
        self._world_up = torch.tensor([0.0, 1.0, 0.0], device=device)
        self._note_forward = torch.tensor([0.0, 0.0, -1.0], device=device)
        calibration = load_simulator_calibration()
        style_calibration = load_style_calibration()
        self._note_x_offset = float(calibration['x_offset'])
        self._note_x_spacing = float(calibration['x_spacing'])
        self._note_y_offset = float(calibration['y_offset'])
        self._note_y_spacing = float(calibration['y_spacing'])
        axis = torch.tensor(calibration.get('saber_axis', [0.0, 0.0, 1.0]), dtype=torch.float32, device=device)
        axis_norm = torch.norm(axis).clamp(min=1e-6)
        self._local_saber_axis = axis / axis_norm
        origin = torch.tensor(calibration.get('saber_origin', [0.0, 0.0, 0.0]), dtype=torch.float32, device=device)
        self._local_saber_origin = origin
        self._saber_length = float(calibration.get('saber_length', SABER_LENGTH))
        default_cut_start = self._saber_length * SABER_CUT_START_FRACTION
        self._saber_cut_start = float(
            np.clip(
                calibration.get('saber_cut_start', default_cut_start),
                0.0,
                max(0.0, self._saber_length - 0.05),
            )
        )
        self._default_style_speed_cap = float(style_calibration.get('linear_speed_cap', 3.35))
        self._default_style_angular_cap = float(style_calibration.get('angular_speed_cap', 12.5))
        self._score_only_mode = False
        self._external_pose_passthrough = False

    # ------------------------------------------------------------------
    # Map loading  (CPU-side, outside any graph)
    # ------------------------------------------------------------------

    def reserve_note_capacity(self, max_notes, max_obstacles=None):
        self._reserved_max_notes = max(1, int(max_notes))
        if max_obstacles is not None:
            self._reserved_max_obstacles = max(1, int(max_obstacles))

    def reserve_obstacle_capacity(self, max_obstacles):
        self._reserved_max_obstacles = max(1, int(max_obstacles))

    def clear_reserved_capacity(self):
        self._reserved_max_notes = 1
        self._reserved_max_obstacles = 1

    @staticmethod
    def _finite_float(value, label, *, positive=False, nonnegative=False):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be a finite number.") from exc
        if not np.isfinite(number):
            raise ValueError(f"{label} must be finite.")
        if positive and number <= 0.0:
            raise ValueError(f"{label} must be positive.")
        if nonnegative and number < 0.0:
            raise ValueError(f"{label} must be nonnegative.")
        return number

    @staticmethod
    def _sequence_or_single(value):
        if isinstance(value, dict):
            return [value]
        try:
            return list(value)
        except TypeError:
            return [value]

    def _coerce_start_times(self, start_times):
        if start_times is None:
            return None
        times = torch.as_tensor(start_times, dtype=torch.float32, device=self.device).flatten()
        if times.numel() != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} start_times entries; got {times.numel()}.")
        if not bool(torch.isfinite(times).all().item()):
            raise ValueError("start_times must be finite.")
        return times

    @staticmethod
    def _coerce_dt(dt):
        dt = GPUBeatSaberSimulator._finite_float(dt, "dt", positive=True)
        return dt

    def load_maps(self, all_notes_list, bpm_list, capacity=None):
        N = self.num_envs
        raw_maps = self._sequence_or_single(all_notes_list)
        raw_bpms = self._sequence_or_single(bpm_list)
        if len(raw_maps) != N:
            raise ValueError(f"Expected {N} beatmaps for {N} envs; got {len(raw_maps)}.")
        if len(raw_bpms) != N:
            raise ValueError(f"Expected {N} BPM values for {N} envs; got {len(raw_bpms)}.")
        bpm_values = [self._finite_float(bpm, f"bpm_list[{i}]", positive=True) for i, bpm in enumerate(raw_bpms)]

        def _coerce_map(entry):
            if isinstance(entry, dict):
                return entry
            return {
                'notes': entry,
                'obstacles': [],
                'njs': 18.0,
                'offset': 0.0,
            }

        maps = [_coerce_map(entry) for entry in raw_maps]

        actual_note_max = max((len(m.get('notes', [])) for m in maps), default=1)
        actual_obs_max = max((len(m.get('obstacles', [])) for m in maps), default=1)

        if capacity is None:
            reserved_notes = self._reserved_max_notes
            reserved_obstacles = self._reserved_max_obstacles
        elif isinstance(capacity, dict):
            reserved_notes = int(capacity.get('notes', self._reserved_max_notes) or 1)
            reserved_obstacles = int(capacity.get('obstacles', self._reserved_max_obstacles) or 1)
        else:
            reserved_notes, reserved_obstacles = capacity
            reserved_notes = int(reserved_notes or 1)
            reserved_obstacles = int(reserved_obstacles or 1)

        new_max = max(actual_note_max, reserved_notes, 1)
        new_obs_max = max(actual_obs_max, reserved_obstacles, 1)
        shape_changed = (
            (not hasattr(self, 'note_times'))
            or (new_max != self.max_notes)
            or (new_obs_max != self.max_obstacles)
        )
        self.max_notes = new_max
        self.max_obstacles = new_obs_max
        M = self.max_notes
        O = self.max_obstacles

        nt = torch.zeros(N, M)
        gx = torch.zeros(N, M)
        gy = torch.zeros(N, M)
        tp = torch.full((N, M), -1.0)
        dx = torch.zeros(N, M)
        dy = torch.zeros(N, M)
        score_class = torch.zeros(N, M)
        score_cap = torch.full((N, M), SCORE_CAP_NORMAL)
        pre_scale = torch.ones(N, M)
        post_scale = torch.ones(N, M)
        acc_scale = torch.ones(N, M)
        pre_auto = torch.zeros(N, M)
        post_auto = torch.zeros(N, M)
        fixed_score = torch.zeros(N, M)
        requires_speed = torch.ones(N, M)
        requires_direction = torch.ones(N, M)
        any_direction = torch.zeros(N, M)
        nc = torch.zeros(N, dtype=torch.long)

        ot = torch.zeros(N, O)
        ogx = torch.zeros(N, O)
        ogy = torch.zeros(N, O)
        owidth = torch.zeros(N, O)
        oheight = torch.zeros(N, O)
        odur = torch.zeros(N, O)
        oc = torch.zeros(N, dtype=torch.long)

        njs = torch.full((N,), 18.0)
        offsets = torch.zeros(N)
        map_end_beats = torch.zeros(N)

        for i, beatmap in enumerate(maps):
            notes = beatmap.get('notes', [])
            obstacles = beatmap.get('obstacles', [])
            num = len(notes)
            obs_num = len(obstacles)
            nc[i] = num
            oc[i] = obs_num
            njs[i] = self._finite_float(beatmap.get('njs', 18.0) or 18.0, f"map[{i}].njs", positive=True)
            offsets[i] = self._finite_float(beatmap.get('offset', 0.0) or 0.0, f"map[{i}].offset")

            if num > 0:
                t_arr = [
                    self._finite_float(n['time'], f"map[{i}].notes[{j}].time", nonnegative=True)
                    for j, n in enumerate(notes)
                ]
                x_arr = [n['lineIndex'] for n in notes]
                y_arr = [n['lineLayer'] for n in notes]
                p_arr = [n['type'] for n in notes]
                d_arr = [CUT_DIR_VECTORS.get(int(n['cutDirection']), (0.0, 0.0)) for n in notes]

                nt[i, :num] = torch.tensor(t_arr, dtype=torch.float32)
                gx[i, :num] = torch.tensor(x_arr, dtype=torch.float32)
                gy[i, :num] = torch.tensor(y_arr, dtype=torch.float32)
                tp[i, :num] = torch.tensor(p_arr, dtype=torch.float32)
                cdirs_t = torch.tensor(d_arr, dtype=torch.float32)
                dx[i, :num] = cdirs_t[:, 0]
                dy[i, :num] = cdirs_t[:, 1]
                score_class[i, :num] = torch.tensor([n.get('scoreClass', 0.0) for n in notes], dtype=torch.float32)
                score_cap[i, :num] = torch.tensor([n.get('scoreCap', SCORE_CAP_NORMAL) for n in notes], dtype=torch.float32)
                pre_scale[i, :num] = torch.tensor([n.get('preScale', 1.0) for n in notes], dtype=torch.float32)
                post_scale[i, :num] = torch.tensor([n.get('postScale', 1.0) for n in notes], dtype=torch.float32)
                acc_scale[i, :num] = torch.tensor([n.get('accScale', 1.0) for n in notes], dtype=torch.float32)
                pre_auto[i, :num] = torch.tensor([n.get('preAuto', 0.0) for n in notes], dtype=torch.float32)
                post_auto[i, :num] = torch.tensor([n.get('postAuto', 0.0) for n in notes], dtype=torch.float32)
                fixed_score[i, :num] = torch.tensor([n.get('fixedScore', 0.0) for n in notes], dtype=torch.float32)
                requires_speed[i, :num] = torch.tensor([1.0 if n.get('requiresSpeed', True) else 0.0 for n in notes], dtype=torch.float32)
                requires_direction[i, :num] = torch.tensor([1.0 if n.get('requiresDirection', True) else 0.0 for n in notes], dtype=torch.float32)
                any_direction[i, :num] = torch.tensor([1.0 if n.get('allowAnyDirection', False) else 0.0 for n in notes], dtype=torch.float32)
            else:
                t_arr = []

            if obs_num > 0:
                obs_t_arr = [
                    self._finite_float(o['time'], f"map[{i}].obstacles[{j}].time", nonnegative=True)
                    for j, o in enumerate(obstacles)
                ]
                obs_duration_arr = [
                    self._finite_float(o['duration'], f"map[{i}].obstacles[{j}].duration", nonnegative=True)
                    for j, o in enumerate(obstacles)
                ]
                ot[i, :obs_num] = torch.tensor(obs_t_arr, dtype=torch.float32)
                ogx[i, :obs_num] = torch.tensor([o['lineIndex'] for o in obstacles], dtype=torch.float32)
                ogy[i, :obs_num] = torch.tensor([o['lineLayer'] for o in obstacles], dtype=torch.float32)
                owidth[i, :obs_num] = torch.tensor([o['width'] for o in obstacles], dtype=torch.float32)
                oheight[i, :obs_num] = torch.tensor([o['height'] for o in obstacles], dtype=torch.float32)
                odur[i, :obs_num] = torch.tensor(obs_duration_arr, dtype=torch.float32)
            else:
                obs_t_arr = []
                obs_duration_arr = []

            note_end = max(t_arr, default=0.0)
            obstacle_end = max((t + d for t, d in zip(obs_t_arr, obs_duration_arr)), default=0.0)
            map_end_beats[i] = max(note_end, obstacle_end, 0.0)

        scorable_counts = (
            (torch.arange(M).unsqueeze(0) < nc.unsqueeze(1))
            & (tp != 3)
        ).sum(1, dtype=torch.long)

        dev_bpm = torch.tensor(bpm_values, dtype=torch.float32, device=self.device)
        dev_njs = njs.to(self.device)
        dev_offsets = offsets.to(self.device)
        dev_scorable_counts = scorable_counts.to(self.device)

        if shape_changed:
            self.note_times = nt.to(self.device)
            self.note_gx = gx.to(self.device)
            self.note_gy = gy.to(self.device)
            self.note_types = tp.to(self.device)
            self.note_dx = dx.to(self.device)
            self.note_dy = dy.to(self.device)
            self.note_score_class = score_class.to(self.device)
            self.note_score_cap = score_cap.to(self.device)
            self.note_pre_scale = pre_scale.to(self.device)
            self.note_post_scale = post_scale.to(self.device)
            self.note_acc_scale = acc_scale.to(self.device)
            self.note_pre_auto = pre_auto.to(self.device)
            self.note_post_auto = post_auto.to(self.device)
            self.note_fixed_score = fixed_score.to(self.device)
            self.note_requires_speed = requires_speed.to(self.device)
            self.note_requires_direction = requires_direction.to(self.device)
            self.note_any_direction = any_direction.to(self.device)
            self.note_counts = nc.to(self.device)
            self.scorable_note_counts = dev_scorable_counts

            self.obstacle_times = ot.to(self.device)
            self.obstacle_gx = ogx.to(self.device)
            self.obstacle_gy = ogy.to(self.device)
            self.obstacle_width = owidth.to(self.device)
            self.obstacle_height = oheight.to(self.device)
            self.obstacle_duration = odur.to(self.device)
            self.obstacle_counts = oc.to(self.device)

            self.bpms = dev_bpm
            self.note_jump_speed = dev_njs
            self.note_jump_offset = dev_offsets
        else:
            self.note_times.copy_(nt)
            self.note_gx.copy_(gx)
            self.note_gy.copy_(gy)
            self.note_types.copy_(tp)
            self.note_dx.copy_(dx)
            self.note_dy.copy_(dy)
            self.note_score_class.copy_(score_class)
            self.note_score_cap.copy_(score_cap)
            self.note_pre_scale.copy_(pre_scale)
            self.note_post_scale.copy_(post_scale)
            self.note_acc_scale.copy_(acc_scale)
            self.note_pre_auto.copy_(pre_auto)
            self.note_post_auto.copy_(post_auto)
            self.note_fixed_score.copy_(fixed_score)
            self.note_requires_speed.copy_(requires_speed)
            self.note_requires_direction.copy_(requires_direction)
            self.note_any_direction.copy_(any_direction)
            self.note_counts.copy_(nc)
            self.scorable_note_counts.copy_(dev_scorable_counts)

            self.obstacle_times.copy_(ot)
            self.obstacle_gx.copy_(ogx)
            self.obstacle_gy.copy_(ogy)
            self.obstacle_width.copy_(owidth)
            self.obstacle_height.copy_(oheight)
            self.obstacle_duration.copy_(odur)
            self.obstacle_counts.copy_(oc)

            self.bpms.copy_(dev_bpm)
            self.note_jump_speed.copy_(dev_njs)
            self.note_jump_offset.copy_(dev_offsets)

        bps = self.bpms / 60.0
        if shape_changed or not hasattr(self, 'bps'):
            self.bps = bps
        else:
            self.bps.copy_(bps)

        spawn_ahead_beats = self._compute_spawn_ahead_beats(self.bps, self.note_jump_speed, self.note_jump_offset)
        if shape_changed or not hasattr(self, 'note_spawn_ahead_beats'):
            self.note_spawn_ahead_beats = spawn_ahead_beats
        else:
            self.note_spawn_ahead_beats.copy_(spawn_ahead_beats)

        has_objects = ((nc > 0) | (oc > 0)).to(self.device)
        map_end_beats = map_end_beats.to(self.device)
        map_durations = torch.where(
            has_objects,
            ((map_end_beats / self.bps) + 2.0).clamp(min=10.0),
            torch.tensor(10.0, device=self.device),
        )
        if not hasattr(self, 'map_durations') or shape_changed:
            self.map_durations = map_durations
        else:
            self.map_durations.copy_(map_durations)

        max_note_idx = (nc - 1).clamp(min=0).to(self.device)
        max_obstacle_idx = (oc - 1).clamp(min=0).to(self.device)
        if not hasattr(self, '_max_note_idx') or shape_changed:
            self._max_note_idx = max_note_idx
            self._max_obstacle_idx = max_obstacle_idx
        else:
            self._max_note_idx.copy_(max_note_idx)
            self._max_obstacle_idx.copy_(max_obstacle_idx)

        self.reset()

    # ------------------------------------------------------------------
    # Reset  (outside graph — allocates all persistent tensors)
    # ------------------------------------------------------------------

    def reset(self, start_times=None):
        N   = self.num_envs
        M   = self.max_notes
        O   = self.max_obstacles
        dev = self.device
        start_times = self._coerce_start_times(start_times)

        # Match the BC training pose distribution so closed-loop rollout
        # starts from a state the policy actually knows how to recover from.
        cp = torch.tensor(BC_RESET_POSE, dtype=torch.float32, device=dev)
        init_poses = cp.unsqueeze(0).expand(N, -1).clone()

        first = not hasattr(self, 'poses')
        shape_key = (M, O)
        shape_changed = (not first) and (getattr(self, '_last_shape', None) != shape_key)
        self._last_shape = shape_key

        if first:
            self.poses           = init_poses.clone()
            self.prev_poses      = init_poses.clone()
            self.prev_head_delta = torch.zeros(N, 7, device=dev)
            self.prev_hand_delta = torch.zeros(N, 14, device=dev)
            self.current_times   = start_times.clone() if start_times is not None else torch.zeros(N, device=dev)
            self.note_idx        = torch.zeros(N, dtype=torch.long, device=dev)
            self.obstacle_idx    = torch.zeros(N, dtype=torch.long, device=dev)
            self.episode_done    = torch.zeros(N, dtype=torch.bool, device=dev)
            
            # ── Ghost Miss Fix ────────────────────────────────────────────────
            # Initialize note_active by masking out notes that have already passed.
            t_beat = self.current_times * self.bps
            self.note_active = (
                (torch.arange(M, device=dev).unsqueeze(0) < self.note_counts.unsqueeze(1)) &
                (self.note_times >= t_beat.unsqueeze(1))
            )
            self.obstacle_active = (
                (torch.arange(self.max_obstacles, device=dev).unsqueeze(0) < self.obstacle_counts.unsqueeze(1)) &
                ((self.obstacle_times + self.obstacle_duration) >= t_beat.unsqueeze(1))
            )
            # ──────────────────────────────────────────────────────────────────
            self.pose_history    = init_poses.unsqueeze(1).expand(N, HISTORY_LEN, -1).clone()
            self.hist_ptr        = torch.tensor([0], dtype=torch.long, device=dev)
            self.prev_accel      = torch.zeros(N, 14, device=dev)
            self.total_hits      = torch.zeros(N, device=dev)
            self.total_misses    = torch.zeros(N, device=dev)
            self.total_badcuts   = torch.zeros(N, device=dev)
            self.total_bombs     = torch.zeros(N, device=dev)
            self.total_note_misses = torch.zeros(N, device=dev)
            self.total_engaged_scorable = torch.zeros(N, device=dev)
            self.total_resolved_scorable = torch.zeros(N, device=dev)
            self.total_scores    = torch.zeros(N, device=dev)
            self.total_cut_scores = torch.zeros(N, device=dev)
            self.total_wall_hits = torch.zeros(N, device=dev)
            self.active_note_counts = self.note_active.sum(1, dtype=torch.long)
            self.remaining_obstacle_counts = self.obstacle_active.sum(1, dtype=torch.long)
            self.prev_wall_contact = torch.zeros(N, dtype=torch.bool, device=dev)
            # ── Health system (Beat Saber energy bar) ─────────────────────
            self.energy          = torch.full((N,), 0.5, device=dev)
            self.combo           = torch.zeros(N, device=dev)
            self.max_combo       = torch.zeros(N, device=dev)
            self.score_multiplier = torch.ones(N, device=dev)
            self.multiplier_progress = torch.zeros(N, device=dev)
            self._terminal_reason = torch.zeros(N, dtype=torch.long, device=dev)
            self._completion_ratio = torch.zeros(N, device=dev)
            self.motion_path = torch.zeros(N, device=dev)
            self.useful_progress = torch.zeros(N, device=dev)
            self.mean_speed_sum = torch.zeros(N, device=dev)
            self.speed_samples = torch.zeros(N, device=dev)
            self.speed_violation_sum = torch.zeros(N, device=dev)
            self.angular_violation_sum = torch.zeros(N, device=dev)
            self.waste_motion_sum = torch.zeros(N, device=dev)
            self.idle_motion_sum = torch.zeros(N, device=dev)
            self.guard_error_sum = torch.zeros(N, device=dev)
            self.oscillation_sum = torch.zeros(N, device=dev)
            self.lateral_motion_sum = torch.zeros(N, device=dev)
        else:
            self.poses.copy_(init_poses)
            self.prev_poses.copy_(init_poses)
            self.prev_head_delta.zero_()
            self.prev_hand_delta.zero_()
            if start_times is not None:
                self.current_times.copy_(start_times)
            else:
                self.current_times.zero_()
            self.note_idx.zero_()
            self.obstacle_idx.zero_()
            self.episode_done.zero_()
            
            # ── Ghost Miss Fix ────────────────────────────────────────────────
            t_beat = self.current_times * self.bps
            new_active = (
                (torch.arange(M, device=dev).unsqueeze(0) < self.note_counts.unsqueeze(1)) &
                (self.note_times >= t_beat.unsqueeze(1))
            )
            new_obstacles = (
                (torch.arange(self.max_obstacles, device=dev).unsqueeze(0) < self.obstacle_counts.unsqueeze(1)) &
                ((self.obstacle_times + self.obstacle_duration) >= t_beat.unsqueeze(1))
            )
            if shape_changed:
                self.note_active = new_active
                self.obstacle_active = new_obstacles
            else:
                self.note_active.copy_(new_active)
                self.obstacle_active.copy_(new_obstacles)
            # ──────────────────────────────────────────────────────────────────
            self.pose_history.copy_(init_poses.unsqueeze(1).expand(N, HISTORY_LEN, -1))
            self.hist_ptr.fill_(0)
            self.total_hits.zero_()
            self.total_misses.zero_()
            self.total_badcuts.zero_()
            self.total_bombs.zero_()
            self.total_note_misses.zero_()
            self.total_engaged_scorable.zero_()
            self.total_resolved_scorable.zero_()
            self.total_scores.zero_()
            self.total_cut_scores.zero_()
            self.total_wall_hits.zero_()
            self.active_note_counts.copy_(self.note_active.sum(1, dtype=torch.long))
            self.remaining_obstacle_counts.copy_(self.obstacle_active.sum(1, dtype=torch.long))
            self.prev_wall_contact.zero_()
            self.prev_accel.zero_()
            self.energy.copy_(self._start_energy)
            self.combo.zero_()
            self.max_combo.zero_()
            self.score_multiplier.fill_(1.0)
            self.multiplier_progress.zero_()
            self._terminal_reason.zero_()
            self._completion_ratio.zero_()
            self.motion_path.zero_()
            self.useful_progress.zero_()
            self.mean_speed_sum.zero_()
            self.speed_samples.zero_()
            self.speed_violation_sum.zero_()
            self.angular_violation_sum.zero_()
            self.waste_motion_sum.zero_()
            self.idle_motion_sum.zero_()
            self.guard_error_sum.zero_()
            self.oscillation_sum.zero_()
            self.lateral_motion_sum.zero_()
            self._done_out.zero_()
            self._last_dt.fill_(DEFAULT_DT)
            self._prev_target_note_idx.fill_(-1)
            self._prev_target_dist.zero_()

        if first or shape_changed:
            self._note_range   = torch.arange(M, device=dev)
            self._999_buf      = torch.full((N, M), 999.0, device=dev)
            self._note_pos_buf = torch.zeros(N, M, 3, device=dev)
            self._note_keep_buf = torch.ones(N, M, dtype=torch.int32, device=dev)
            self._obstacle_range = torch.arange(O, device=dev)
            self._obstacle_pos_buf = torch.zeros(N, O, 3, device=dev)
            self._obstacle_size_buf = torch.zeros(N, O, 3, device=dev)
            self._obstacle_keep_buf = torch.ones(N, O, dtype=torch.int32, device=dev)

        if first:
            self._static_head = (
                 torch.tensor(STATIC_HEAD, dtype=torch.float32, device=dev)
                      .unsqueeze(0).expand(N, -1).clone()
            )
            self._offs_range  = torch.arange(NUM_UPCOMING_NOTES, device=dev)
            self._obs_offs_range = torch.arange(NUM_UPCOMING_OBSTACLES, device=dev)
            self._active_note_offsets = torch.arange(ACTIVE_NOTE_WINDOW, device=dev)
            self._active_obstacle_offsets = torch.arange(ACTIVE_OBSTACLE_WINDOW, device=dev)
            self._pos_idx     = torch.tensor([0,1,2,7,8,9], device=dev)
            self._state_out   = torch.zeros(N, INPUT_DIM, device=dev)
            self._reward_out  = torch.zeros(N,      device=dev)
            self._success_out = torch.zeros(N,      device=dev)
            self._done_out    = torch.zeros(N, dtype=torch.bool, device=dev)
            self._delta_out   = torch.zeros(N, POSE_DIM,  device=dev)
            self._action_delta_requested = torch.zeros(N, POSE_DIM, device=dev)
            self._action_delta_clamped = torch.zeros(N, POSE_DIM, device=dev)
            self._action_delta_excess = torch.zeros(N, POSE_DIM, device=dev)
            self._reward_components = torch.zeros(N, len(REWARD_COMPONENT_NAMES), device=dev)
            self._assist_state = torch.zeros(N, len(ASSIST_STATE_NAMES), device=dev)
            self._state_note_visible = torch.zeros(N, NUM_UPCOMING_NOTES, dtype=torch.bool, device=dev)
            self._state_note_scorable = torch.zeros(N, NUM_UPCOMING_NOTES, dtype=torch.bool, device=dev)
            self._shoulder_l  = torch.tensor([-0.15, 1.5, 0.0], device=dev)
            self._shoulder_r  = torch.tensor([ 0.15, 1.5, 0.0], device=dev)
            self._guard_left_target = torch.tensor([-0.18, 1.42, 0.18], device=dev).view(1, 3).expand(N, -1).clone()
            self._guard_right_target = torch.tensor([0.18, 1.42, 0.18], device=dev).view(1, 3).expand(N, -1).clone()
            self._identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).view(1, 4)
            self._arrow_good_half_extents = (torch.tensor(ARROW_GOOD_HITBOX, device=dev) * 0.5).view(1, 1, 3)
            self._dot_good_half_extents = (torch.tensor(DOT_GOOD_HITBOX, device=dev) * 0.5).view(1, 1, 3)
            self._bad_half_extents = (torch.tensor(BAD_HITBOX, device=dev) * 0.5).view(1, 1, 3)
            self._good_hitbox_offset = torch.tensor([0.0, 0.0, GOOD_HITBOX_Z_OFFSET], device=dev).view(1, 1, 3)
            self._pre_angle_offsets = torch.arange(HISTORY_LEN - 1, dtype=torch.long, device=dev)
            self._last_dt       = torch.full((1,), DEFAULT_DT, device=dev)
            self._env_indices   = torch.arange(N, device=dev)
            self._no_target_idx = torch.full((N,), -1, dtype=torch.long, device=dev)
            self._prev_target_note_idx = torch.full((N,), -1, dtype=torch.long, device=dev)
            self._prev_target_dist = torch.zeros(N, device=dev)
            # ── Penalty & reward weights — fixed-address GPU scalars ──────────
            # Update via set_penalty_weights() / set_reward_weights() between
            # epochs. Never reassign — the CUDA graph captures the address.
            self._w_miss          = torch.full((N,), 2.0,   device=dev)
            self._w_jerk          = torch.full((N,), 0.001, device=dev)
            self._w_pos_jerk      = torch.full((N,), 0.002, device=dev)
            self._w_reach         = torch.full((N,), 0.01,  device=dev)
            self._w_approach      = torch.full((N,), DENSE_APPROACH_SCALE, device=dev)
            self._w_contact       = torch.full((N,), CONTACT_DISCOVERY_SCALE, device=dev)
            self._speed_threshold = torch.full((N,), 1.5, device=dev)
            self._direction_threshold = torch.full((N,), 0.35, device=dev)
            # Timing windows are tracked in seconds rather than world-space z.
            # A fixed z window makes faster maps artificially harder because the
            # allowed early-hit time shrinks as NJS rises.
            self._hit_window_front = torch.full((N,), 0.160, device=dev)
            self._hit_window_back = torch.full((N,), 0.080, device=dev)
            self._miss_window_back = torch.full((N,), -0.100, device=dev)
            self._controller_hit_mix = torch.zeros(N, device=dev)
            self._start_energy = torch.full((N,), 0.5, device=dev)
            self._miss_energy = torch.full((N,), 0.10, device=dev)
            self._bomb_energy = torch.full((N,), 0.15, device=dev)
            self._death_penalty = torch.full((N,), 50.0, device=dev)
            self._fail_enabled = torch.ones(N, dtype=torch.bool, device=dev)
            self._w_survival = torch.full((N,), 0.010, device=dev)
            self._w_energy = torch.full((N,), 6.0, device=dev)
            self._w_combo = torch.full((N,), 0.12, device=dev)
            self._w_low_energy = torch.full((N,), 0.75, device=dev)
            self._w_style_speed = torch.zeros(N, device=dev)
            self._w_style_waste = torch.zeros(N, device=dev)
            self._w_style_angular = torch.zeros(N, device=dev)
            self._w_style_lateral = torch.zeros(N, device=dev)
            self._w_style_oscillation = torch.zeros(N, device=dev)
            self._w_idle_guard = torch.zeros(N, device=dev)
            self._w_swing_arc = torch.zeros(N, device=dev)
            self._w_perfect_hit = torch.zeros(N, device=dev)
            self._style_speed_cap = torch.full((N,), self._default_style_speed_cap, device=dev)
            self._style_angular_cap = torch.full((N,), self._default_style_angular_cap, device=dev)
            self._style_step_allowance = torch.full((N,), self._default_style_speed_cap / 55.0, device=dev)
            self._style_support_scale = torch.full((N,), 0.35, device=dev)
            self._style_approach_far = torch.full((N,), 4.5, device=dev)
            self._good_hitbox_scale = torch.ones(N, device=dev)
            self._bad_hitbox_scale = torch.ones(N, device=dev)
            if self._native_sim_ext is not None:
                self._native_left_good_contact = torch.zeros(N, ACTIVE_NOTE_WINDOW, dtype=torch.bool, device=dev)
                self._native_right_good_contact = torch.zeros(N, ACTIVE_NOTE_WINDOW, dtype=torch.bool, device=dev)
                self._native_left_bad_contact = torch.zeros(N, ACTIVE_NOTE_WINDOW, dtype=torch.bool, device=dev)
                self._native_right_bad_contact = torch.zeros(N, ACTIVE_NOTE_WINDOW, dtype=torch.bool, device=dev)
                self._native_left_distance = torch.zeros(N, ACTIVE_NOTE_WINDOW, device=dev)
                self._native_right_distance = torch.zeros(N, ACTIVE_NOTE_WINDOW, device=dev)
            # ── Saber inertia ─ fixed-address tensors for CUDA graph ──────
            # Controls how "heavy" the sabers feel. Higher inertia = more
            # momentum, smoother arcs, harder to change direction quickly.
            # Formula: hand_delta = (1 - inertia) * target + inertia * prev
            self._saber_inertia   = torch.full((N,), 0.55,  device=dev)
            self._head_inertia    = torch.full((N,), 0.22,  device=dev)
            # Per-component delta clamp: position gets wider range than
            # rotation to prevent physically impossible angular velocity.
            # [hx,hy,hz, hqx,hqy,hqz,hqw, lx,ly,lz, lqx,lqy,lqz,lqw, rx,ry,rz, rqx,rqy,rqz,rqw]
            self._delta_clamp     = torch.tensor(
                [0.08, 0.08, 0.08, 0.045, 0.045, 0.045, 0.045,
                 0.12, 0.12, 0.12, 0.07, 0.07, 0.07, 0.07,
                 0.12, 0.12, 0.12, 0.07, 0.07, 0.07, 0.07],
                device=dev).unsqueeze(0).expand(N, -1).clone()

        self.energy.copy_(self._start_energy)
        self._zero_step_diagnostics()
        self._refresh_assist_state()

    def teleport_all(
        self,
        pose,
        history,
        current_time,
        note_active_mask,
        total_hits=0,
        total_misses=0,
        total_scores=0,
        total_cut_scores=0,
        total_badcuts=0,
        total_bombs=0,
        total_note_misses=0,
    ):
        """
        Force-syncs all environments to a single source state. 
        Used by the search replayer to branch from a successful note.
        """
        N = self.num_envs
        dev = self.device
        
        p_t = torch.as_tensor(pose, dtype=torch.float32, device=dev)
        h_t = torch.as_tensor(history, dtype=torch.float32, device=dev)
        
        self.poses.copy_(p_t.unsqueeze(0).expand(N, -1))
        self.prev_poses.copy_(p_t.unsqueeze(0).expand(N, -1))
        self.pose_history.copy_(h_t.unsqueeze(0).expand(N, -1, -1))
        
        self.current_times.fill_(float(current_time))
        self.note_active.copy_(torch.as_tensor(note_active_mask, dtype=torch.bool, device=dev).unsqueeze(0).expand(N, -1))
        self.active_note_counts.copy_(self.note_active.sum(1, dtype=torch.long))
        if hasattr(self, 'obstacle_active'):
            t_beat = self.current_times * self.bps
            self.obstacle_active.copy_(
                ((torch.arange(self.max_obstacles, device=dev).unsqueeze(0) < self.obstacle_counts.unsqueeze(1))
                 & ((self.obstacle_times + self.obstacle_duration) >= t_beat.unsqueeze(1)))
            )
            self.remaining_obstacle_counts.copy_(self.obstacle_active.sum(1, dtype=torch.long))
        
        self.total_hits.fill_(float(total_hits))
        self.total_misses.fill_(float(total_misses))
        self.total_badcuts.fill_(float(total_badcuts))
        self.total_bombs.fill_(float(total_bombs))
        self.total_note_misses.fill_(float(total_note_misses))
        self.total_engaged_scorable.zero_()
        self.total_resolved_scorable.zero_()
        self.total_scores.fill_(float(total_scores))
        self.total_cut_scores.fill_(float(total_cut_scores))
        
        self.prev_head_delta.zero_()
        self.prev_hand_delta.zero_()
        self.prev_accel.zero_()
        self.prev_wall_contact.zero_()
        self.hist_ptr.fill_(0)
        self.episode_done.zero_()
        self._done_out.zero_()
        self.score_multiplier.fill_(1.0)
        self.multiplier_progress.zero_()
        self._prev_target_note_idx.fill_(-1)
        self._prev_target_dist.zero_()
        self.motion_path.zero_()
        self.useful_progress.zero_()
        self.mean_speed_sum.zero_()
        self.speed_samples.zero_()
        self.speed_violation_sum.zero_()
        self.angular_violation_sum.zero_()
        self.waste_motion_sum.zero_()
        self.idle_motion_sum.zero_()
        self.guard_error_sum.zero_()
        self.oscillation_sum.zero_()
        self.lateral_motion_sum.zero_()
        self._zero_step_diagnostics()
        self._refresh_assist_state()

    # ------------------------------------------------------------------
    # Curriculum penalty / reward control  (call between epochs, outside graph)
    # ------------------------------------------------------------------

    def set_penalty_weights(self, w_miss, w_jerk, w_pos_jerk, w_reach, indices=None):
        """Update penalty weights in-place — safe while CUDA graph is active."""
        if indices is None:
            self._w_miss    .fill_(w_miss)
            self._w_pos_jerk.fill_(w_pos_jerk)
            self._w_jerk    .fill_(w_jerk)
            self._w_reach   .fill_(w_reach)
        else:
            self._w_miss[indices]     = w_miss
            self._w_pos_jerk[indices] = w_pos_jerk
            self._w_jerk[indices]     = w_jerk
            self._w_reach[indices]    = w_reach

    def set_saber_inertia(self, inertia, rot_clamp=None, pos_clamp=None, indices=None):
        """Set saber inertia (mass feel) in-place — CUDA-graph safe."""
        if indices is None:
            self._saber_inertia.fill_(inertia)
            if rot_clamp is not None:
                self._delta_clamp[:, 10:14].fill_(rot_clamp)
                self._delta_clamp[:, 17:21].fill_(rot_clamp)
            if pos_clamp is not None:
                self._delta_clamp[:, 7:10].fill_(pos_clamp)
                self._delta_clamp[:, 14:17].fill_(pos_clamp)
        else:
            self._saber_inertia[indices] = inertia
            if rot_clamp is not None:
                self._delta_clamp[indices, 10:14] = rot_clamp
                self._delta_clamp[indices, 17:21] = rot_clamp
            if pos_clamp is not None:
                self._delta_clamp[indices, 7:10] = pos_clamp
                self._delta_clamp[indices, 14:17] = pos_clamp

    def set_dense_reward_scale(self, scale, indices=None):
        """Scale the dense approach reward in-place — CUDA-graph safe."""
        if indices is None:
            self._w_approach.fill_(scale)
        else:
            self._w_approach[indices] = scale

    def set_score_only_mode(self, enabled=True):
        """Enable a lighter step path for replay-score validation."""
        self._score_only_mode = bool(enabled)

    def set_external_pose_passthrough(self, enabled=True):
        """Treat step() input as an exact pose instead of a policy target."""
        self._external_pose_passthrough = bool(enabled)

    def set_training_wheels(self, level=True):
        """Increase collision volume for easier discovery during rehabilitation.

        Accepts either a bool for legacy callers or a float in [0, 1] for
        adaptive control.
        """
        if isinstance(level, bool):
            scale = 1.18 if level else 1.0
            self._good_hitbox_scale.fill_(scale)
            self._bad_hitbox_scale.fill_(scale)
            return

        level = float(max(0.0, min(1.0, level)))
        self._good_hitbox_scale.fill_(1.0 + 0.20 * level)
        self._bad_hitbox_scale.fill_(1.0 + 0.18 * level)

    def set_rehab_assists(self, level, indices=None):
        """Relax timing, speed, and direction constraints adaptively."""
        if isinstance(level, bool):
            level = 1.0 if level else 0.0

        level = float(max(0.0, min(1.0, level)))
        speed_threshold = max(0.55, 1.5 - (0.9 * level))
        direction_threshold = max(-0.05, 0.35 - (0.45 * level))
        contact_reward = CONTACT_DISCOVERY_SCALE + (0.85 * level)
        # Widen the legal cut timing window in seconds rather than meters so
        # assistance behaves consistently across maps with different NJS.
        hit_window_front = 0.080 + (0.040 * level)
        hit_window_back = 0.080 + (0.040 * level)
        miss_window_back = -0.100 - (0.050 * level)
        controller_hit_mix = level

        if indices is None:
            self._speed_threshold.fill_(speed_threshold)
            self._direction_threshold.fill_(direction_threshold)
            self._w_contact.fill_(contact_reward)
            self._hit_window_front.fill_(hit_window_front)
            self._hit_window_back.fill_(hit_window_back)
            self._miss_window_back.fill_(miss_window_back)
            self._controller_hit_mix.fill_(controller_hit_mix)
        else:
            self._speed_threshold[indices] = speed_threshold
            self._direction_threshold[indices] = direction_threshold
            self._w_contact[indices] = contact_reward
            self._hit_window_front[indices] = hit_window_front
            self._hit_window_back[indices] = hit_window_back
            self._miss_window_back[indices] = miss_window_back
            self._controller_hit_mix[indices] = controller_hit_mix

    def set_survival_assistance(self, level, indices=None):
        """Loosen failure pressure so stuck policies can still see more notes."""
        if isinstance(level, bool):
            level = 1.0 if level else 0.0

        level = float(max(0.0, min(1.0, level)))
        start_energy = min(0.95, 0.5 + (0.35 * level))
        miss_energy = max(0.02, 0.10 * (1.0 - (0.75 * level)))
        bomb_energy = max(0.05, 0.15 * (1.0 - (0.55 * level)))
        death_penalty = max(10.0, 50.0 * (1.0 - (0.50 * level)))

        if indices is None:
            self._start_energy.fill_(start_energy)
            self._miss_energy.fill_(miss_energy)
            self._bomb_energy.fill_(bomb_energy)
            self._death_penalty.fill_(death_penalty)
        else:
            self._start_energy[indices] = start_energy
            self._miss_energy[indices] = miss_energy
            self._bomb_energy[indices] = bomb_energy
            self._death_penalty[indices] = death_penalty

    def set_fail_enabled(self, enabled, indices=None):
        if indices is None:
            self._fail_enabled.fill_(bool(enabled))
        else:
            self._fail_enabled[indices] = bool(enabled)

    def set_stability_assistance(self, level, indices=None):
        """Bias rewards toward sustained clean play when interception is learned."""
        if isinstance(level, bool):
            level = 1.0 if level else 0.0

        level = float(max(0.0, min(1.0, level)))
        # Keep the base survival reward modest so low-skill policies can't
        # plateau on "stay alive and drift" behavior. Higher rehab levels still
        # add the old safety net back when we explicitly ask for it.
        survival_reward = 0.002 + (0.038 * level)
        energy_reward = 1.50 + (9.00 * level)
        combo_reward = 0.05 + (0.35 * level)
        low_energy_penalty = 0.75 + (1.75 * level)

        if indices is None:
            self._w_survival.fill_(survival_reward)
            self._w_energy.fill_(energy_reward)
            self._w_combo.fill_(combo_reward)
            self._w_low_energy.fill_(low_energy_penalty)
        else:
            self._w_survival[indices] = survival_reward
            self._w_energy[indices] = energy_reward
            self._w_combo[indices] = combo_reward
            self._w_low_energy[indices] = low_energy_penalty

    def set_style_guidance(self, level, indices=None):
        """Encourage direct, human-like swings once the policy can already hit notes."""
        if isinstance(level, bool):
            level = 1.0 if level else 0.0

        level = float(max(0.0, min(1.0, level)))
        speed_weight = 0.30 * level
        waste_weight = 6.50 * level
        angular_weight = 0.050 * level
        lateral_weight = 2.80 * level
        oscillation_weight = 1.80 * level
        idle_guard_weight = 0.20 + (1.40 * level)
        swing_arc_weight = 0.35 + (3.65 * level)
        perfect_hit_weight = 0.20 + (0.80 * level)
        speed_cap = self._default_style_speed_cap * (1.08 - 0.28 * level)
        angular_cap = self._default_style_angular_cap * (1.08 - 0.30 * level)
        step_allowance = max(0.016, speed_cap / 82.0)
        support_scale = 0.34 - (0.18 * level)
        approach_far = max(2.3, 4.4 - (1.7 * level))

        if indices is None:
            self._w_style_speed.fill_(speed_weight)
            self._w_style_waste.fill_(waste_weight)
            self._w_style_angular.fill_(angular_weight)
            self._w_style_lateral.fill_(lateral_weight)
            self._w_style_oscillation.fill_(oscillation_weight)
            self._w_idle_guard.fill_(idle_guard_weight)
            self._w_swing_arc.fill_(swing_arc_weight)
            self._w_perfect_hit.fill_(perfect_hit_weight)
            self._style_speed_cap.fill_(speed_cap)
            self._style_angular_cap.fill_(angular_cap)
            self._style_step_allowance.fill_(step_allowance)
            self._style_support_scale.fill_(support_scale)
            self._style_approach_far.fill_(approach_far)
        else:
            self._w_style_speed[indices] = speed_weight
            self._w_style_waste[indices] = waste_weight
            self._w_style_angular[indices] = angular_weight
            self._w_style_lateral[indices] = lateral_weight
            self._w_style_oscillation[indices] = oscillation_weight
            self._w_idle_guard[indices] = idle_guard_weight
            self._w_swing_arc[indices] = swing_arc_weight
            self._w_perfect_hit[indices] = perfect_hit_weight
            self._style_speed_cap[indices] = speed_cap
            self._style_angular_cap[indices] = angular_cap
            self._style_step_allowance[indices] = step_allowance
            self._style_support_scale[indices] = support_scale
            self._style_approach_far[indices] = approach_far

    def _zero_step_diagnostics(self):
        if hasattr(self, "_action_delta_requested"):
            self._action_delta_requested.zero_()
            self._action_delta_clamped.zero_()
            self._action_delta_excess.zero_()
        if hasattr(self, "_reward_components"):
            self._reward_components.zero_()
        if hasattr(self, "_state_note_visible"):
            self._state_note_visible.zero_()
            self._state_note_scorable.zero_()

    def _refresh_assist_state(self):
        if not hasattr(self, "_assist_state"):
            return
        cols = self._assist_state
        cols[:, 0].copy_(self._good_hitbox_scale)
        cols[:, 1].copy_(self._bad_hitbox_scale)
        cols[:, 2].copy_(self._controller_hit_mix)
        cols[:, 3].copy_(self._speed_threshold)
        cols[:, 4].copy_(self._direction_threshold)
        cols[:, 5].copy_(self._hit_window_front)
        cols[:, 6].copy_(self._hit_window_back)
        cols[:, 7].copy_(self._miss_window_back)
        cols[:, 8].copy_(self._w_contact)
        cols[:, 9].copy_(self._w_approach)
        cols[:, 10].copy_(self._start_energy)
        cols[:, 11].copy_(self._miss_energy)
        cols[:, 12].copy_(self._bomb_energy)
        cols[:, 13].copy_(self._death_penalty)
        cols[:, 14].copy_(self._fail_enabled.float())

    def get_reward_components(self):
        return {
            name: self._reward_components[:, idx]
            for idx, name in enumerate(REWARD_COMPONENT_NAMES)
        }

    def get_assist_state(self):
        self._refresh_assist_state()
        return {
            name: self._assist_state[:, idx]
            for idx, name in enumerate(ASSIST_STATE_NAMES)
        }

    def get_action_envelope_diagnostics(self):
        return {
            "requested_delta": self._action_delta_requested,
            "clamped_target_delta": self._action_delta_clamped,
            "clamp_excess": self._action_delta_excess,
            "applied_delta": self._delta_out,
        }

    # ------------------------------------------------------------------
    # Event tracking  (replay generation only — OFF during training)
    # ------------------------------------------------------------------

    def enable_event_tracking(self):
        """Enable per-note event recording for BSOR generation.

        NOT CUDA-graph safe. Only use in eager-mode replay generation.
        When enabled, step() appends per-note hit/miss dicts to
        self.tracked_events[env_idx] for later BSOR construction.
        """
        self._tracking = True
        self.tracked_events = [[] for _ in range(self.num_envs)]

    def disable_event_tracking(self):
        """Disable event tracking and free memory."""
        self._tracking = False
        self.tracked_events = None

    # ------------------------------------------------------------------
    # Helpers (pure tensor ops)
    # ------------------------------------------------------------------

    def _quat_forward(self, q):
        q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        q_xyz = q[..., 0:3]
        q_w = q[..., 3:4]
        axis = self._local_saber_axis.view(*([1] * (q.dim() - 1)), 3).expand_as(q_xyz)
        t = 2.0 * torch.cross(q_xyz, axis, dim=-1)
        return axis + q_w * t + torch.cross(q_xyz, t, dim=-1)

    def _quat_rotate_local(self, q, local_vec):
        q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        q_xyz = q[..., 0:3]
        q_w = q[..., 3:4]
        vec = local_vec.view(*([1] * (q.dim() - 1)), 3).expand_as(q_xyz)
        t = 2.0 * torch.cross(q_xyz, vec, dim=-1)
        return vec + q_w * t + torch.cross(q_xyz, t, dim=-1)

    def _quat_angular_speed(self, prev_q, next_q, dt):
        prev_q = prev_q / prev_q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        next_q = next_q / next_q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        dot = torch.abs((prev_q * next_q).sum(dim=-1)).clamp(0.0, 1.0)
        angle = 2.0 * torch.acos(dot)
        return angle / max(dt, 1e-6)

    def _saber_geometry(self, poses):
        left_rot = poses[:, 10:14]
        right_rot = poses[:, 17:21]
        lf = self._quat_forward(left_rot)
        rf = self._quat_forward(right_rot)
        left_hand = poses[:, 7:10]
        right_hand = poses[:, 14:17]
        left_hilt = left_hand + self._quat_rotate_local(left_rot, self._local_saber_origin)
        right_hilt = right_hand + self._quat_rotate_local(right_rot, self._local_saber_origin)
        return (
            left_hilt,
            right_hilt,
            left_hilt + lf * self._saber_length,
            right_hilt + rf * self._saber_length,
            lf,
            rf,
        )

    def _point_to_segment_distance(self, points, seg_start, seg_end):
        """Distance from note centers [N, M, 3] to batched line segments [N, 3]."""
        seg = (seg_end - seg_start).unsqueeze(1)
        rel = points - seg_start.unsqueeze(1)
        seg_len_sq = seg.square().sum(-1, keepdim=True).clamp(min=1e-6)
        t = (rel * seg).sum(-1, keepdim=True) / seg_len_sq
        t = t.clamp(0.0, 1.0)
        closest = seg_start.unsqueeze(1) + t * seg
        return torch.norm(points - closest, dim=-1)

    def _note_positions(self):
        self._note_pos_buf[..., 0] = self._note_x_offset + self.note_gx * self._note_x_spacing
        self._note_pos_buf[..., 1] = self._note_y_offset + self.note_gy * self._note_y_spacing
        self._note_pos_buf[..., 2] = ((self.note_times / self.bps[:, None]) - self.current_times[:, None]) * self.note_jump_speed.unsqueeze(1)
        return self._note_pos_buf

    def _obstacle_geometry(self):
        self._obstacle_size_buf[..., 0] = self.obstacle_width.clamp(min=1.0) * self._note_x_spacing
        self._obstacle_size_buf[..., 1] = self.obstacle_height.clamp(min=1.0) * self._note_y_spacing
        self._obstacle_size_buf[..., 2] = (self.obstacle_duration / self.bps[:, None]) * self.note_jump_speed.unsqueeze(1)

        start_z = ((self.obstacle_times / self.bps[:, None]) - self.current_times[:, None]) * self.note_jump_speed.unsqueeze(1)
        self._obstacle_pos_buf[..., 0] = self._note_x_offset + (self.obstacle_gx + (self.obstacle_width.clamp(min=1.0) - 1.0) * 0.5) * self._note_x_spacing
        self._obstacle_pos_buf[..., 1] = self._note_y_offset + (self.obstacle_gy + (self.obstacle_height.clamp(min=1.0) - 1.0) * 0.5) * self._note_y_spacing
        self._obstacle_pos_buf[..., 2] = start_z + self._obstacle_size_buf[..., 2] * 0.5
        return self._obstacle_pos_buf, self._obstacle_size_buf

    def _segment_intersects_aabb(self, seg_start, seg_end, box_center, half_extents):
        p0 = seg_start.unsqueeze(1)
        p1 = seg_end.unsqueeze(1)
        d = p1 - p0
        bmin = box_center - half_extents
        bmax = box_center + half_extents

        inv_d = torch.where(d.abs() > 1e-6, 1.0 / d, torch.zeros_like(d))
        t0 = (bmin - p0) * inv_d
        t1 = (bmax - p0) * inv_d
        tmin = torch.minimum(t0, t1)
        tmax = torch.maximum(t0, t1)

        parallel = d.abs() <= 1e-6
        inside_parallel = (p0 >= bmin) & (p0 <= bmax)
        tmin = torch.where(parallel, torch.where(inside_parallel, torch.zeros_like(tmin), torch.ones_like(tmin) * 2.0), tmin)
        tmax = torch.where(parallel, torch.where(inside_parallel, torch.ones_like(tmax), torch.ones_like(tmax) * -1.0), tmax)

        entry = tmin.max(dim=-1).values
        exit = tmax.min(dim=-1).values
        return (exit >= entry) & (exit >= 0.0) & (entry <= 1.0)

    def _segment_point_distance(self, seg_start, seg_end, points):
        seg = seg_end - seg_start
        rel = points - seg_start.unsqueeze(1)
        seg_len_sq = seg.square().sum(-1, keepdim=True).unsqueeze(1).clamp(min=1e-6)
        t = (rel * seg.unsqueeze(1)).sum(-1, keepdim=True) / seg_len_sq
        t = t.clamp(0.0, 1.0)
        closest = seg_start.unsqueeze(1) + t * seg.unsqueeze(1)
        return torch.norm(points - closest, dim=-1)

    def _note_collision_metrics(
        self,
        left_hilt_prev,
        left_hilt_curr,
        left_tip_prev,
        left_tip_curr,
        left_ctrl_prev,
        left_ctrl_curr,
        right_hilt_prev,
        right_hilt_curr,
        right_tip_prev,
        right_tip_curr,
        right_ctrl_prev,
        right_ctrl_curr,
        good_centers,
        good_half,
        note_pos,
        bad_half,
        assist_enabled,
    ):
        ext = getattr(self, '_native_sim_ext', None)
        if self._use_native_contact and ext is not None and good_centers.is_cuda:
            def _contiguous(tensor):
                return tensor if tensor.is_contiguous() else tensor.contiguous()

            # The native kernel consumes the already-scaled half extents that the
            # Python path uses, so training wheels stay consistent across both paths.
            ext.note_collision_metrics_out(
                _contiguous(left_hilt_prev),
                _contiguous(left_hilt_curr),
                _contiguous(left_tip_prev),
                _contiguous(left_tip_curr),
                _contiguous(left_ctrl_prev),
                _contiguous(left_ctrl_curr),
                _contiguous(right_hilt_prev),
                _contiguous(right_hilt_curr),
                _contiguous(right_tip_prev),
                _contiguous(right_tip_curr),
                _contiguous(right_ctrl_prev),
                _contiguous(right_ctrl_curr),
                _contiguous(good_centers),
                _contiguous(good_half),
                _contiguous(note_pos),
                _contiguous(bad_half),
                _contiguous(assist_enabled),
                self._native_left_good_contact,
                self._native_right_good_contact,
                self._native_left_bad_contact,
                self._native_right_bad_contact,
                self._native_left_distance,
                self._native_right_distance,
            )
            return (
                self._native_left_good_contact,
                self._native_right_good_contact,
                self._native_left_bad_contact,
                self._native_right_bad_contact,
                self._native_left_distance,
                self._native_right_distance,
            )

        assist_mask = assist_enabled.unsqueeze(1)
        left_good_contact = (
            self._segment_intersects_aabb(left_tip_prev, left_tip_curr, good_centers, good_half)
            | self._segment_intersects_aabb(left_hilt_prev, left_hilt_curr, good_centers, good_half)
            | self._segment_intersects_aabb(left_hilt_curr, left_tip_curr, good_centers, good_half)
        )
        right_good_contact = (
            self._segment_intersects_aabb(right_tip_prev, right_tip_curr, good_centers, good_half)
            | self._segment_intersects_aabb(right_hilt_prev, right_hilt_curr, good_centers, good_half)
            | self._segment_intersects_aabb(right_hilt_curr, right_tip_curr, good_centers, good_half)
        )
        left_good_contact |= assist_mask & self._segment_intersects_aabb(left_ctrl_prev, left_ctrl_curr, good_centers, good_half)
        right_good_contact |= assist_mask & self._segment_intersects_aabb(right_ctrl_prev, right_ctrl_curr, good_centers, good_half)

        left_bad_contact = (
            self._segment_intersects_aabb(left_tip_prev, left_tip_curr, note_pos, bad_half)
            | self._segment_intersects_aabb(left_hilt_prev, left_hilt_curr, note_pos, bad_half)
            | self._segment_intersects_aabb(left_hilt_curr, left_tip_curr, note_pos, bad_half)
        )
        right_bad_contact = (
            self._segment_intersects_aabb(right_tip_prev, right_tip_curr, note_pos, bad_half)
            | self._segment_intersects_aabb(right_hilt_prev, right_hilt_curr, note_pos, bad_half)
            | self._segment_intersects_aabb(right_hilt_curr, right_tip_curr, note_pos, bad_half)
        )

        left_blade_dist = self._point_to_segment_distance(note_pos, left_hilt_curr, left_tip_curr)
        right_blade_dist = self._point_to_segment_distance(note_pos, right_hilt_curr, right_tip_curr)
        left_tip_sweep_dist = self._segment_point_distance(left_tip_prev, left_tip_curr, note_pos)
        right_tip_sweep_dist = self._segment_point_distance(right_tip_prev, right_tip_curr, note_pos)
        left_hilt_sweep_dist = self._segment_point_distance(left_hilt_prev, left_hilt_curr, note_pos)
        right_hilt_sweep_dist = self._segment_point_distance(right_hilt_prev, right_hilt_curr, note_pos)
        dl = torch.minimum(left_blade_dist, torch.minimum(left_tip_sweep_dist, left_hilt_sweep_dist))
        dr = torch.minimum(right_blade_dist, torch.minimum(right_tip_sweep_dist, right_hilt_sweep_dist))

        left_good_blade_dist = self._point_to_segment_distance(good_centers, left_hilt_curr, left_tip_curr)
        right_good_blade_dist = self._point_to_segment_distance(good_centers, right_hilt_curr, right_tip_curr)
        left_good_tip_sweep = self._segment_point_distance(left_tip_prev, left_tip_curr, good_centers)
        right_good_tip_sweep = self._segment_point_distance(right_tip_prev, right_tip_curr, good_centers)
        left_good_hilt_sweep = self._segment_point_distance(left_hilt_prev, left_hilt_curr, good_centers)
        right_good_hilt_sweep = self._segment_point_distance(right_hilt_prev, right_hilt_curr, good_centers)
        left_good_dist = torch.minimum(left_good_blade_dist, torch.minimum(left_good_tip_sweep, left_good_hilt_sweep))
        right_good_dist = torch.minimum(right_good_blade_dist, torch.minimum(right_good_tip_sweep, right_good_hilt_sweep))

        good_ribbon = torch.norm(good_half[..., 0:2], dim=-1) * BLADE_RIBBON_SCALE
        bad_ribbon = torch.norm(bad_half[..., 0:2], dim=-1) * BLADE_RIBBON_SCALE
        left_good_contact |= (left_good_dist <= good_ribbon) | (dl <= good_ribbon)
        right_good_contact |= (right_good_dist <= good_ribbon) | (dr <= good_ribbon)
        left_bad_contact |= dl <= bad_ribbon
        right_bad_contact |= dr <= bad_ribbon
        return left_good_contact, right_good_contact, left_bad_contact, right_bad_contact, dl, dr

    def _active_note_window_indices(self, center_idx):
        raw_idx = center_idx.unsqueeze(1) - ACTIVE_NOTE_PAST + self._active_note_offsets.unsqueeze(0)
        clamped_idx = raw_idx.clamp(0, max(0, self.max_notes - 1))
        valid = (raw_idx >= 0) & (raw_idx < self.note_counts.unsqueeze(1))
        return raw_idx, clamped_idx, valid

    def _active_obstacle_window_indices(self, center_idx):
        raw_idx = center_idx.unsqueeze(1) + self._active_obstacle_offsets.unsqueeze(0)
        clamped_idx = raw_idx.clamp(0, max(0, self.max_obstacles - 1))
        valid = raw_idx < self.obstacle_counts.unsqueeze(1)
        return raw_idx, clamped_idx, valid

    def _refresh_window_centers(self):
        t_beat = self.current_times * self.bps
        valid_notes = self._note_range.unsqueeze(0) < self.note_counts.unsqueeze(1)
        if hasattr(self, "note_active") and hasattr(self, "_miss_window_back"):
            contact_shift = (
                (TRACK_Z_BASE + self.poses[:, 2])
                / self.note_jump_speed.clamp(min=1e-6)
            ).unsqueeze(1)
            note_time_delta = (
                (self.note_times / self.bps[:, None].clamp(min=1e-6))
                - self.current_times[:, None]
                + contact_shift
            )
            followthrough_seconds = (
                STATE_FOLLOWTHROUGH_BEATS / self.bps.clamp(min=1e-6)
            ).unsqueeze(1)
            context_stale = note_time_delta < -followthrough_seconds
            passed = context_stale & valid_notes
        else:
            passed = (self.note_times < t_beat.unsqueeze(1)) & valid_notes
        raw_idx = passed.sum(1)
        self.note_idx.copy_(torch.min(raw_idx, self._max_note_idx))

        obs_passed = (
            ((self.obstacle_times + self.obstacle_duration) < t_beat.unsqueeze(1))
            & (self._obstacle_range.unsqueeze(0) < self.obstacle_counts.unsqueeze(1))
        )
        obs_raw_idx = obs_passed.sum(1)
        self.obstacle_idx.copy_(torch.min(obs_raw_idx, self._max_obstacle_idx))

    def _compute_spawn_ahead_beats(self, bps, note_jump_speed, note_jump_offset):
        safe_bps = bps.clamp(min=1e-6)
        safe_njs = note_jump_speed.clamp(min=0.0)
        seconds_per_beat = safe_bps.reciprocal()
        half_jump_beats = torch.full_like(safe_bps, INITIAL_HALF_JUMP_BEATS)
        for _ in range(3):
            too_far = (safe_njs * seconds_per_beat * half_jump_beats) > MAX_HALF_JUMP_DISTANCE_METERS
            half_jump_beats = torch.where(
                too_far & (half_jump_beats > MIN_HALF_JUMP_BEATS),
                half_jump_beats * 0.5,
                half_jump_beats,
            )
        return (half_jump_beats + note_jump_offset).clamp(min=MIN_HALF_JUMP_BEATS)

    # ------------------------------------------------------------------
    # get_states  — writes into self._state_out (fixed address)
    # ------------------------------------------------------------------

    def get_states(self):
        """Returns self._state_out [N, INPUT_DIM] — same tensor every call."""
        t_beat = self.current_times * self.bps
        self._refresh_window_centers()

        raw_offs = self._offs_range.unsqueeze(0) + self.note_idx.unsqueeze(1)
        offs = raw_offs.clamp(0, self.max_notes - 1)
        valid = raw_offs < self.note_counts.unsqueeze(1)

        safe_bps = self.bps[:, None].clamp(min=1e-6)
        head_z = self.poses[:, 2].unsqueeze(1)
        gathered_note_times = torch.gather(self.note_times, 1, offs)
        raw_note_time_beats = gathered_note_times - t_beat.unsqueeze(1)
        spawn_visible = raw_note_time_beats <= self.note_spawn_ahead_beats.unsqueeze(1)
        contact_shift_seconds = (TRACK_Z_BASE + head_z) / self.note_jump_speed.unsqueeze(1).clamp(min=1e-6)
        contact_time_seconds = (raw_note_time_beats / safe_bps) + contact_shift_seconds
        followthrough_seconds = (STATE_FOLLOWTHROUGH_BEATS / self.bps.clamp(min=1e-6)).unsqueeze(1)
        visible = valid & spawn_visible & (contact_time_seconds >= -followthrough_seconds)
        scorable = torch.gather(self.note_active, 1, offs) & visible
        self._state_note_visible.copy_(visible)
        self._state_note_scorable.copy_(scorable)
        raw_t_off_beats = raw_note_time_beats.clamp(-1, 4) * visible
        raw_t_off_seconds = (raw_t_off_beats / safe_bps) * visible
        t_off_seconds = (raw_t_off_seconds + contact_shift_seconds) * visible
        t_off_beats = (t_off_seconds * safe_bps) * visible
        t_off_z = (t_off_seconds * self.note_jump_speed.unsqueeze(1)) * visible
        ngx   = torch.gather(self.note_gx,    1, offs) * visible
        ngy   = torch.gather(self.note_gy,    1, offs) * visible
        ntp   = torch.where(visible, torch.gather(self.note_types, 1, offs), torch.full_like(torch.gather(self.note_gx, 1, offs), -1))
        nddx  = torch.gather(self.note_dx,    1, offs) * visible
        nddy  = torch.gather(self.note_dy,    1, offs) * visible
        nscl  = torch.gather(self.note_score_class, 1, offs) * visible
        ncap  = (torch.gather(self.note_score_cap, 1, offs) / SCORE_CAP_NORMAL) * visible

        self._state_out[:, 0:NOTES_DIM:NOTE_FEATURES] = t_off_beats
        self._state_out[:, 1:NOTES_DIM:NOTE_FEATURES] = ngx
        self._state_out[:, 2:NOTES_DIM:NOTE_FEATURES] = ngy
        self._state_out[:, 3:NOTES_DIM:NOTE_FEATURES] = ntp
        self._state_out[:, 4:NOTES_DIM:NOTE_FEATURES] = nddx
        self._state_out[:, 5:NOTES_DIM:NOTE_FEATURES] = nddy
        self._state_out[:, 6:NOTES_DIM:NOTE_FEATURES] = nscl
        self._state_out[:, 7:NOTES_DIM:NOTE_FEATURES] = ncap
        self._state_out[:, 8:NOTES_DIM:NOTE_FEATURES] = t_off_seconds
        self._state_out[:, 9:NOTES_DIM:NOTE_FEATURES] = t_off_z

        raw_obs_offs = self._obs_offs_range.unsqueeze(0) + self.obstacle_idx.unsqueeze(1)
        obs_offs = raw_obs_offs.clamp(0, max(0, self.max_obstacles - 1))
        obs_valid = raw_obs_offs < self.obstacle_counts.unsqueeze(1)
        obs_base = NOTES_DIM
        obs_visible = (
            torch.gather(self.obstacle_times, 1, obs_offs) - t_beat.unsqueeze(1)
        ) <= self.note_spawn_ahead_beats.unsqueeze(1)
        obs_feature_valid = obs_valid & obs_visible
        self._state_out[:, obs_base + 0:obs_base + OBSTACLES_DIM:OBSTACLE_FEATURES] = (torch.gather(self.obstacle_times, 1, obs_offs) - t_beat.unsqueeze(1)).clamp(-1, 6) * obs_feature_valid
        self._state_out[:, obs_base + 1:obs_base + OBSTACLES_DIM:OBSTACLE_FEATURES] = torch.gather(self.obstacle_gx, 1, obs_offs) * obs_feature_valid
        self._state_out[:, obs_base + 2:obs_base + OBSTACLES_DIM:OBSTACLE_FEATURES] = torch.gather(self.obstacle_gy, 1, obs_offs) * obs_feature_valid
        self._state_out[:, obs_base + 3:obs_base + OBSTACLES_DIM:OBSTACLE_FEATURES] = torch.gather(self.obstacle_width, 1, obs_offs) * obs_feature_valid
        self._state_out[:, obs_base + 4:obs_base + OBSTACLES_DIM:OBSTACLE_FEATURES] = torch.gather(self.obstacle_height, 1, obs_offs) * obs_feature_valid
        self._state_out[:, obs_base + 5:obs_base + OBSTACLES_DIM:OBSTACLE_FEATURES] = torch.gather(self.obstacle_duration, 1, obs_offs) * obs_feature_valid
        
        poses = self.poses
        vel   = (poses - self.prev_poses) / self._last_dt.clamp(min=1e-4)

        for hist_slot, offset in enumerate(STATE_HISTORY_OFFSETS):
            start = NOTES_DIM + OBSTACLES_DIM + hist_slot * STATE_FRAME_DIM
            pose_start = start
            vel_start = start + POSE_DIM

            if offset == 0:
                hist_poses = poses
                hist_vel = vel
            else:
                hist_idx_exp = ((self.hist_ptr - 1 - offset) % HISTORY_LEN).view(1, 1, 1).expand(
                    self.num_envs, 1, self.poses.shape[-1]
                )
                prev_idx_exp = ((self.hist_ptr - 2 - offset) % HISTORY_LEN).view(1, 1, 1).expand(
                    self.num_envs, 1, self.poses.shape[-1]
                )
                hist_pose = self.pose_history.gather(1, hist_idx_exp).squeeze(1)
                prev_hist_pose = self.pose_history.gather(1, prev_idx_exp).squeeze(1)
                hist_poses = hist_pose
                hist_vel = (hist_pose - prev_hist_pose) / self._last_dt.clamp(min=1e-4)

            self._state_out[:, pose_start:pose_start + POSE_DIM] = hist_poses
            self._state_out[:, vel_start:vel_start + VELOCITY_DIM] = hist_vel

        return self._state_out

    # ------------------------------------------------------------------
    # step  — all state updates in-place
    # ------------------------------------------------------------------

    def step(self, pose_actions, dt=1.0/60.0):
        """
        pose_actions : [N, 21]
        Returns (self._reward_out [N], self._delta_out [N, 21]) — fixed addresses.
        """
        dt = self._coerce_dt(dt)
        pose_actions = torch.as_tensor(pose_actions, dtype=torch.float32, device=self.device)
        if pose_actions.ndim == 1:
            if self.num_envs != 1:
                raise ValueError(
                    f"Expected pose_actions shape ({self.num_envs}, {POSE_DIM}); got {tuple(pose_actions.shape)}."
                )
            pose_actions = pose_actions.unsqueeze(0)
        expected_shape = (self.num_envs, POSE_DIM)
        if tuple(pose_actions.shape) != expected_shape:
            raise ValueError(f"Expected pose_actions shape {expected_shape}; got {tuple(pose_actions.shape)}.")

        def _advance_multiplier(multiplier, progress, hits):
            next_multiplier = multiplier.clone()
            next_progress = progress.clone()
            remaining_hits = hits.clone()

            for _ in range(3):
                stage_target = torch.where(
                    next_multiplier < 2.0,
                    torch.full_like(next_progress, 2.0),
                    torch.where(
                        next_multiplier < 4.0,
                        torch.full_like(next_progress, 4.0),
                        torch.where(
                            next_multiplier < 8.0,
                            torch.full_like(next_progress, 8.0),
                            torch.zeros_like(next_progress),
                        ),
                    ),
                )
                stage_room = torch.where(
                    next_multiplier < 8.0,
                    (stage_target - next_progress).clamp(min=0.0),
                    torch.zeros_like(next_progress),
                )
                gained = torch.minimum(remaining_hits, stage_room)
                next_progress = next_progress + gained
                remaining_hits = torch.relu(remaining_hits - gained)

                level_up = (next_multiplier < 8.0) & (next_progress >= stage_target - 1e-6)
                next_multiplier = torch.where(level_up, next_multiplier * 2.0, next_multiplier)
                next_progress = torch.where(level_up, torch.zeros_like(next_progress), next_progress)
                next_progress = torch.where(next_multiplier >= 8.0, torch.zeros_like(next_progress), next_progress)

            return next_multiplier, next_progress

        self._last_dt.fill_(dt)
        pose_actions = torch.nan_to_num(pose_actions, nan=0.0, posinf=0.0, neginf=0.0)
        pose_actions = normalize_pose_quaternions(pose_actions)

        active_env = (~self.episode_done).unsqueeze(1)
        # Capture the active mask before terminal flags are updated so the final
        # resolving hit still receives its reward on this step.
        active_env_f = (~self.episode_done).float()
        cur_pose = self.poses
        requested_delta = pose_actions - cur_pose
        self._action_delta_requested.copy_(requested_delta)
        self._action_delta_requested.mul_(active_env.float())
        if self._external_pose_passthrough:
            new_pose = pose_actions.clone()
            self._action_delta_clamped.copy_(self._action_delta_requested)
            self._action_delta_excess.zero_()
        else:
            pose_actions = torch.clamp(pose_actions, -2.0, 2.0)
            bounded_delta = pose_actions - cur_pose
            target_delta = bounded_delta.clamp(-self._delta_clamp, self._delta_clamp)
            self._action_delta_clamped.copy_(target_delta)
            self._action_delta_clamped.mul_(active_env.float())
            self._action_delta_excess.copy_(torch.relu(bounded_delta.abs() - self._delta_clamp))
            self._action_delta_excess.mul_(active_env.float())

            head_target_delta = target_delta[:, 0:7]
            hand_target_delta = target_delta[:, 7:21]

            desired_head_delta = (
                (1.0 - self._head_inertia).unsqueeze(1) * head_target_delta
                + self._head_inertia.unsqueeze(1) * self.prev_head_delta
            )
            desired_hand_delta = (
                (1.0 - self._saber_inertia).unsqueeze(1) * hand_target_delta
                + self._saber_inertia.unsqueeze(1) * self.prev_hand_delta
            )

            # Limit instantaneous acceleration so the policy cannot exploit near-teleport
            # motion even when it alternates extreme actions across frames.
            head_accel_limit = self._delta_clamp[:, 0:7] * (0.20 + 0.55 * (1.0 - self._head_inertia).unsqueeze(1))
            hand_accel_limit = self._delta_clamp[:, 7:21] * (0.16 + 0.46 * (1.0 - self._saber_inertia).unsqueeze(1))
            head_delta = self.prev_head_delta + (desired_head_delta - self.prev_head_delta).clamp(
                -head_accel_limit,
                head_accel_limit,
            )
            hand_delta = self.prev_hand_delta + (desired_hand_delta - self.prev_hand_delta).clamp(
                -hand_accel_limit,
                hand_accel_limit,
            )

            head_delta = torch.nan_to_num(head_delta, nan=0.0, posinf=0.0, neginf=0.0).clamp(-0.25, 0.25)
            hand_delta = torch.nan_to_num(hand_delta, nan=0.0, posinf=0.0, neginf=0.0).clamp(-0.5, 0.5)

            self._delta_out[:, 0:7].copy_(head_delta)
            self._delta_out[:, 7:21].copy_(hand_delta)
            self._delta_out.mul_(active_env.float())

            new_pose = (cur_pose + self._delta_out).clone()
        for q_start, q_end in ((3, 7), (10, 14), (17, 21)):
            q = torch.nan_to_num(new_pose[:, q_start:q_end], nan=0.0, posinf=0.0, neginf=0.0)
            q_norm = torch.norm(q, dim=-1, keepdim=True)
            q = torch.where(
                q_norm > 1e-8,
                q / q_norm.clamp(min=1e-8),
                self._identity_quat.expand_as(q),
            )
            new_pose[:, q_start:q_end] = q
        new_pose = torch.where(active_env, new_pose, cur_pose)
        self._delta_out.copy_(new_pose - cur_pose)
        self._delta_out.mul_(active_env.float())

        self.prev_poses.copy_(self.poses)
        self.poses.copy_(new_pose)
        self.current_times.add_(dt * active_env_f)
        self._refresh_window_centers()

        head_delta = self._delta_out[:, 0:7]
        hand_delta = self._delta_out[:, 7:21]
        new_hands = self.poses[:, 7:21]
        head_translation = self.poses[:, 0:3] - self._static_head[:, 0:3]

        left_step = torch.norm(self.poses[:, 7:10] - self.prev_poses[:, 7:10], dim=-1)
        right_step = torch.norm(self.poses[:, 14:17] - self.prev_poses[:, 14:17], dim=-1)
        left_speed_inst = left_step / max(dt, 1e-6)
        right_speed_inst = right_step / max(dt, 1e-6)
        mean_hand_speed = 0.5 * (left_speed_inst + right_speed_inst)
        peak_hand_speed = torch.maximum(left_speed_inst, right_speed_inst)
        left_ang_speed = self._quat_angular_speed(self.prev_poses[:, 10:14], self.poses[:, 10:14], dt)
        right_ang_speed = self._quat_angular_speed(self.prev_poses[:, 17:21], self.poses[:, 17:21], dt)
        peak_ang_speed = torch.maximum(left_ang_speed, right_ang_speed)

        idx_exp = (self.hist_ptr % HISTORY_LEN).view(1, 1, 1).expand(self.num_envs, 1, self.poses.shape[-1])
        self.pose_history.scatter_(1, idx_exp, self.poses.unsqueeze(1))
        self.hist_ptr.add_(1)

        pcl, pcr, plt, prt, plf, prf = self._saber_geometry(self.prev_poses)
        cl, cr, lt, rt, lf, rf = self._saber_geometry(self.poses)
        note_raw_idx, note_idx, note_valid = self._active_note_window_indices(self.note_idx)
        note_times = torch.gather(self.note_times, 1, note_idx)
        note_gx = torch.gather(self.note_gx, 1, note_idx)
        note_gy = torch.gather(self.note_gy, 1, note_idx)
        note_types = torch.gather(self.note_types, 1, note_idx)
        note_dx = torch.gather(self.note_dx, 1, note_idx)
        note_dy = torch.gather(self.note_dy, 1, note_idx)
        note_score_cap = torch.gather(self.note_score_cap, 1, note_idx)
        note_pre_scale = torch.gather(self.note_pre_scale, 1, note_idx)
        note_post_scale = torch.gather(self.note_post_scale, 1, note_idx)
        note_acc_scale = torch.gather(self.note_acc_scale, 1, note_idx)
        note_pre_auto = torch.gather(self.note_pre_auto, 1, note_idx)
        note_post_auto = torch.gather(self.note_post_auto, 1, note_idx)
        note_fixed_score = torch.gather(self.note_fixed_score, 1, note_idx)
        note_requires_speed = torch.gather(self.note_requires_speed, 1, note_idx)
        note_requires_direction = torch.gather(self.note_requires_direction, 1, note_idx)
        note_any_direction = torch.gather(self.note_any_direction, 1, note_idx)
        note_active_local = torch.gather(self.note_active, 1, note_idx) & note_valid

        note_world_z = (
            TRACK_Z_BASE
            + self.poses[:, 2].unsqueeze(1)
            + ((note_times / self.bps[:, None]) - self.current_times[:, None]) * self.note_jump_speed.unsqueeze(1)
        )
        npos = torch.stack((
            self._note_x_offset + note_gx * self._note_x_spacing,
            self._note_y_offset + note_gy * self._note_y_spacing,
            note_world_z,
        ), dim=-1)

        is_red = note_types == 0
        is_blue = note_types == 1
        is_bomb = note_types == 3
        is_dot = note_any_direction > 0.5
        note_forward = self._note_forward.view(1, 1, 3).expand(self.num_envs, ACTIVE_NOTE_WINDOW, 3)

        good_centers = npos + self._good_hitbox_offset
        good_half = torch.where(
            is_dot.unsqueeze(-1),
            self._dot_good_half_extents.expand(self.num_envs, ACTIVE_NOTE_WINDOW, 3),
            self._arrow_good_half_extents.expand(self.num_envs, ACTIVE_NOTE_WINDOW, 3),
        ) * self._good_hitbox_scale.view(self.num_envs, 1, 1)
        bad_half = self._bad_half_extents.expand(self.num_envs, ACTIVE_NOTE_WINDOW, 3) * self._bad_hitbox_scale.view(self.num_envs, 1, 1)

        assist_enabled = self._controller_hit_mix > 0.0
        left_good_contact, right_good_contact, left_bad_contact, right_bad_contact, dl, dr = self._note_collision_metrics(
            pcl,
            cl,
            plt,
            lt,
            self.prev_poses[:, 7:10],
            self.poses[:, 7:10],
            pcr,
            cr,
            prt,
            rt,
            self.prev_poses[:, 14:17],
            self.poses[:, 14:17],
            good_centers,
            good_half,
            npos,
            bad_half,
            assist_enabled,
        )

        left_cut_plane_normal = torch.cross(lt - cl, ((pcl + plt) * 0.5) - cl, dim=-1)
        right_cut_plane_normal = torch.cross(rt - cr, ((pcr + prt) * 0.5) - cr, dim=-1)
        left_fallback = torch.cross(lf, self._world_fwd.expand_as(lf), dim=-1)
        right_fallback = torch.cross(rf, self._world_fwd.expand_as(rf), dim=-1)
        left_cut_plane_normal = torch.where(
            torch.norm(left_cut_plane_normal, dim=-1, keepdim=True) > 1e-6,
            left_cut_plane_normal,
            left_fallback,
        )
        right_cut_plane_normal = torch.where(
            torch.norm(right_cut_plane_normal, dim=-1, keepdim=True) > 1e-6,
            right_cut_plane_normal,
            right_fallback,
        )
        left_cut_plane_normal = torch.where(
            torch.norm(left_cut_plane_normal, dim=-1, keepdim=True) > 1e-6,
            left_cut_plane_normal,
            torch.cross(lf, self._world_up.expand_as(lf), dim=-1),
        )
        right_cut_plane_normal = torch.where(
            torch.norm(right_cut_plane_normal, dim=-1, keepdim=True) > 1e-6,
            right_cut_plane_normal,
            torch.cross(rf, self._world_up.expand_as(rf), dim=-1),
        )
        left_cut_plane_normal = left_cut_plane_normal / torch.norm(left_cut_plane_normal, dim=-1, keepdim=True).clamp(min=1e-6)
        right_cut_plane_normal = right_cut_plane_normal / torch.norm(right_cut_plane_normal, dim=-1, keepdim=True).clamp(min=1e-6)

        left_note_plane_normal = torch.cross(note_forward, left_cut_plane_normal.unsqueeze(1).expand(-1, ACTIVE_NOTE_WINDOW, -1), dim=-1)
        right_note_plane_normal = torch.cross(note_forward, right_cut_plane_normal.unsqueeze(1).expand(-1, ACTIVE_NOTE_WINDOW, -1), dim=-1)
        left_note_plane_normal = left_note_plane_normal / torch.norm(left_note_plane_normal, dim=-1, keepdim=True).clamp(min=1e-6)
        right_note_plane_normal = right_note_plane_normal / torch.norm(right_note_plane_normal, dim=-1, keepdim=True).clamp(min=1e-6)

        left_prev_side = ((plt.unsqueeze(1) - npos) * left_note_plane_normal).sum(-1)
        left_curr_side = ((lt.unsqueeze(1) - npos) * left_note_plane_normal).sum(-1)
        right_prev_side = ((prt.unsqueeze(1) - npos) * right_note_plane_normal).sum(-1)
        right_curr_side = ((rt.unsqueeze(1) - npos) * right_note_plane_normal).sum(-1)
        left_plane_cross = (left_prev_side * left_curr_side) <= 0.0
        right_plane_cross = (right_prev_side * right_curr_side) <= 0.0

        left_denom = left_prev_side - left_curr_side
        right_denom = right_prev_side - right_curr_side
        left_interp_t = torch.where(left_denom.abs() > 1e-6, left_prev_side / left_denom, torch.full_like(left_prev_side, 0.5)).clamp(0.0, 1.0)
        right_interp_t = torch.where(right_denom.abs() > 1e-6, right_prev_side / right_denom, torch.full_like(right_prev_side, 0.5)).clamp(0.0, 1.0)

        left_interp_tip = plt.unsqueeze(1) + (lt - plt).unsqueeze(1) * left_interp_t.unsqueeze(-1)
        right_interp_tip = prt.unsqueeze(1) + (rt - prt).unsqueeze(1) * right_interp_t.unsqueeze(-1)
        left_interp_hilt = (0.5 * (pcl + cl)).unsqueeze(1).expand_as(left_interp_tip)
        right_interp_hilt = (0.5 * (pcr + cr)).unsqueeze(1).expand_as(right_interp_tip)
        left_interp_dir = left_interp_tip - left_interp_hilt
        right_interp_dir = right_interp_tip - right_interp_hilt
        left_interp_dir = left_interp_dir / torch.norm(left_interp_dir, dim=-1, keepdim=True).clamp(min=1e-6)
        right_interp_dir = right_interp_dir / torch.norm(right_interp_dir, dim=-1, keepdim=True).clamp(min=1e-6)

        cut_start_gate = -CUT_START_MARGIN
        cut_end_gate = self._saber_length + SABER_CONTACT_MARGIN
        left_along_blade = ((npos - left_interp_hilt) * left_interp_dir).sum(-1)
        right_along_blade = ((npos - right_interp_hilt) * right_interp_dir).sum(-1)
        left_cut_zone = (left_along_blade >= cut_start_gate) & (left_along_blade <= cut_end_gate)
        right_cut_zone = (right_along_blade >= cut_start_gate) & (right_along_blade <= cut_end_gate)

        left_cut_distance = torch.abs(((npos - left_interp_hilt) * left_cut_plane_normal.unsqueeze(1)).sum(-1))
        right_cut_distance = torch.abs(((npos - right_interp_hilt) * right_cut_plane_normal.unsqueeze(1)).sum(-1))
        left_cut_point = left_interp_hilt + left_interp_dir * left_along_blade.clamp(0.0, self._saber_length).unsqueeze(-1)
        right_cut_point = right_interp_hilt + right_interp_dir * right_along_blade.clamp(0.0, self._saber_length).unsqueeze(-1)
        left_prev_along = ((npos - pcl.unsqueeze(1)) * plf.unsqueeze(1)).sum(-1)
        right_prev_along = ((npos - pcr.unsqueeze(1)) * prf.unsqueeze(1)).sum(-1)
        left_prev_cut_point = pcl.unsqueeze(1) + plf.unsqueeze(1) * left_prev_along.clamp(0.0, self._saber_length).unsqueeze(-1)
        right_prev_cut_point = pcr.unsqueeze(1) + prf.unsqueeze(1) * right_prev_along.clamp(0.0, self._saber_length).unsqueeze(-1)

        history_frames = min(HISTORY_LEN - 1, max(2, int(round(0.4 / max(dt, 1e-6)))))
        hist_indices = ((self.hist_ptr - 2 - self._pre_angle_offsets[:history_frames]) % HISTORY_LEN).view(-1)
        hist_pose = self.pose_history.index_select(1, hist_indices)
        left_hist_dirs = self._quat_forward(hist_pose[:, :, 10:14])
        right_hist_dirs = self._quat_forward(hist_pose[:, :, 17:21])
        left_prev_dirs = torch.cat((lf.unsqueeze(1), left_hist_dirs[:, :-1]), dim=1)
        right_prev_dirs = torch.cat((rf.unsqueeze(1), right_hist_dirs[:, :-1]), dim=1)
        left_pre_angle = torch.acos((left_prev_dirs * left_hist_dirs).sum(-1).clamp(-1.0, 1.0)).sum(dim=1) * 57.2958
        right_pre_angle = torch.acos((right_prev_dirs * right_hist_dirs).sum(-1).clamp(-1.0, 1.0)).sum(dim=1) * 57.2958

        left_post_angle = (torch.acos((lf * plf).sum(-1).clamp(-1.0, 1.0)) * 57.2958 * FOLLOWTHROUGH_PREDICTION_FRAMES).clamp(0.0, 120.0)
        right_post_angle = (torch.acos((rf * prf).sum(-1).clamp(-1.0, 1.0)) * 57.2958 * FOLLOWTHROUGH_PREDICTION_FRAMES).clamp(0.0, 120.0)
        left_tip_speed = torch.norm(lt - plt, dim=-1) / max(dt, 1e-6)
        right_tip_speed = torch.norm(rt - prt, dim=-1) / max(dt, 1e-6)

        left_cut_vec = left_cut_point - left_prev_cut_point
        right_cut_vec = right_cut_point - right_prev_cut_point
        left_dir_dot = left_cut_vec[..., 0] * note_dx + left_cut_vec[..., 1] * note_dy
        right_dir_dot = right_cut_vec[..., 0] * note_dx + right_cut_vec[..., 1] * note_dy
        left_dir_mag = torch.norm(left_cut_vec[..., 0:2], dim=-1).clamp(min=1e-6)
        right_dir_mag = torch.norm(right_cut_vec[..., 0:2], dim=-1).clamp(min=1e-6)
        left_dir_ok = (left_dir_dot / left_dir_mag > self._direction_threshold.unsqueeze(1)) | (note_requires_direction <= 0.5) | is_dot
        right_dir_ok = (right_dir_dot / right_dir_mag > self._direction_threshold.unsqueeze(1)) | (note_requires_direction <= 0.5) | is_dot
        left_speed_ok = (note_requires_speed <= 0.5) | (left_tip_speed.unsqueeze(1) > self._speed_threshold.unsqueeze(1))
        right_speed_ok = (note_requires_speed <= 0.5) | (right_tip_speed.unsqueeze(1) > self._speed_threshold.unsqueeze(1))

        note_time_delta = ((note_times / self.bps[:, None]) - self.current_times[:, None]) + (
            (TRACK_Z_BASE + self.poses[:, 2].unsqueeze(1))
            / self.note_jump_speed.unsqueeze(1).clamp(min=1e-6)
        )
        in_zone = (
            (note_time_delta > -self._hit_window_back.unsqueeze(1))
            & (note_time_delta < self._hit_window_front.unsqueeze(1))
        )
        episode_active = (~self.episode_done).unsqueeze(1)
        can_score = in_zone & note_active_local & ~is_bomb & episode_active

        correct_contact = torch.where(
            is_red,
            left_good_contact,
            torch.where(is_blue, right_good_contact, torch.zeros_like(left_good_contact)),
        )
        cut_plane_cross = torch.where(
            is_red,
            left_plane_cross,
            torch.where(is_blue, right_plane_cross, torch.zeros_like(left_plane_cross)),
        )
        cut_zone = torch.where(
            is_red,
            left_cut_zone,
            torch.where(is_blue, right_cut_zone, torch.zeros_like(left_cut_zone)),
        )
        speed_ok = torch.where(
            is_red,
            left_speed_ok,
            torch.where(is_blue, right_speed_ok, torch.zeros_like(left_speed_ok)),
        )
        dir_ok = torch.where(
            is_red,
            left_dir_ok,
            torch.where(is_blue, right_dir_ok, torch.zeros_like(left_dir_ok)),
        )
        wrong_color_contact = torch.where(
            is_red,
            right_bad_contact,
            torch.where(is_blue, left_bad_contact, torch.zeros_like(left_bad_contact)),
        )

        # The swept contact test already checks whether the moving saber segment
        # intersected the note volume this frame. Requiring an extra sign flip on
        # a derived cut plane rejects many legitimate replay cuts simply because
        # the sampled controller frames missed that exact zero crossing.
        contact_hits = can_score & correct_contact & cut_zone
        real_hits = contact_hits & speed_ok & dir_ok & ~wrong_color_contact
        # Direction is part of the gameplay contract. Paying wrong-direction
        # contacts as hits teaches contact discovery instead of committed cuts.
        bad_cuts = (can_score & wrong_color_contact) | (
            contact_hits & ~wrong_color_contact & (~speed_ok | ~dir_ok)
        )
        bad_cuts &= ~real_hits

        pre_angle = torch.where(
            is_red,
            left_pre_angle.unsqueeze(1).expand_as(note_times),
            torch.where(is_blue, right_pre_angle.unsqueeze(1).expand_as(note_times), torch.zeros_like(note_times)),
        )
        post_angle = torch.where(
            is_red,
            left_post_angle.unsqueeze(1).expand_as(note_times),
            torch.where(is_blue, right_post_angle.unsqueeze(1).expand_as(note_times), torch.zeros_like(note_times)),
        )
        cut_distance = torch.where(
            is_red,
            left_cut_distance,
            torch.where(is_blue, right_cut_distance, torch.zeros_like(left_cut_distance)),
        )
        note_speed = torch.where(
            is_red,
            left_tip_speed.unsqueeze(1).expand_as(note_times),
            torch.where(is_blue, right_tip_speed.unsqueeze(1).expand_as(note_times), torch.zeros_like(note_times)),
        )
        pre_sc = note_pre_auto + note_pre_scale * (pre_angle / 100.0 * 70.0).clamp(0.0, 70.0)
        post_sc = note_post_auto + note_post_scale * (post_angle / 60.0 * 30.0).clamp(0.0, 30.0)
        acc_sc = note_acc_scale * ((1.0 - (cut_distance / ACC_DISTANCE_SCALE)).clamp(0.0, 1.0) * 15.0)
        note_cut_scores = pre_sc + post_sc + acc_sc
        note_cut_scores = torch.where(note_fixed_score > 0.0, note_fixed_score, note_cut_scores)
        note_cut_scores = torch.minimum(note_cut_scores, note_score_cap)
        note_cut_scores *= real_hits.float()

        hit_score = note_cut_scores.sum(1)
        num_contacts = contact_hits.sum(1).float()
        num_hits = real_hits.sum(1).float()

        missed = (
            (note_time_delta < self._miss_window_back.unsqueeze(1))
            & note_active_local
            & ~is_bomb
            & episode_active
            & ~real_hits
            & ~bad_cuts
        )
        num_badcuts = bad_cuts.sum(1).float()
        num_misses = missed.sum(1).float()
        is_bomb_hit = in_zone & note_active_local & is_bomb & (torch.minimum(dl, dr) < BOMB_RADIUS) & episode_active
        num_bombs = is_bomb_hit.sum(1).float()

        obstacle_raw_idx, obstacle_idx, obstacle_valid = self._active_obstacle_window_indices(self.obstacle_idx)
        obstacle_times = torch.gather(self.obstacle_times, 1, obstacle_idx)
        obstacle_gx = torch.gather(self.obstacle_gx, 1, obstacle_idx)
        obstacle_gy = torch.gather(self.obstacle_gy, 1, obstacle_idx)
        obstacle_width = torch.gather(self.obstacle_width, 1, obstacle_idx)
        obstacle_height = torch.gather(self.obstacle_height, 1, obstacle_idx)
        obstacle_duration = torch.gather(self.obstacle_duration, 1, obstacle_idx)
        obstacle_active_local = torch.gather(self.obstacle_active, 1, obstacle_idx) & obstacle_valid
        obstacle_still_live = obstacle_valid & ((obstacle_times + obstacle_duration) >= (self.current_times * self.bps).unsqueeze(1))
        updated_obstacle_active = obstacle_active_local & obstacle_still_live
        # Only valid obstacle slots are allowed to mutate the persistent mask.
        # Invalid/clamped window entries contribute `True`, which is neutral under
        # `amin` and therefore cannot resurrect cleared obstacles.
        obstacle_keep_mask = (updated_obstacle_active | ~obstacle_valid).to(torch.int32)
        self._obstacle_keep_buf.fill_(1)
        self._obstacle_keep_buf.scatter_reduce_(
            1,
            obstacle_idx,
            obstacle_keep_mask,
            reduce="amin",
            include_self=True,
        )
        self.obstacle_active.logical_and_(self._obstacle_keep_buf.bool())
        self.remaining_obstacle_counts.copy_(self.obstacle_active.sum(1, dtype=torch.long))

        obstacle_size = torch.stack((
            obstacle_width.clamp(min=1.0) * self._note_x_spacing,
            obstacle_height.clamp(min=1.0) * self._note_y_spacing,
            (obstacle_duration / self.bps[:, None]) * self.note_jump_speed.unsqueeze(1),
        ), dim=-1)
        start_z = (
            TRACK_Z_BASE
            + self.poses[:, 2].unsqueeze(1)
            + ((obstacle_times / self.bps[:, None]) - self.current_times[:, None]) * self.note_jump_speed.unsqueeze(1)
        )
        obstacle_pos = torch.stack((
            self._note_x_offset + (obstacle_gx + (obstacle_width.clamp(min=1.0) - 1.0) * 0.5) * self._note_x_spacing,
            self._note_y_offset + (obstacle_gy + (obstacle_height.clamp(min=1.0) - 1.0) * 0.5) * self._note_y_spacing,
            start_z + obstacle_size[..., 2] * 0.5,
        ), dim=-1)
        obstacle_half = obstacle_size * 0.5
        head_pos = self.poses[:, 0:3].unsqueeze(1)
        wall_overlap = (head_pos - obstacle_pos).abs() <= (obstacle_half + HEAD_HITBOX_RADIUS)
        wall_collision_mask = updated_obstacle_active & wall_overlap.all(dim=-1)
        wall_touching = wall_collision_mask.any(dim=1) & (~self.episode_done)
        wall_entered = wall_touching & ~self.prev_wall_contact & (~self.episode_done)
        wall_damage = wall_touching.float() * (WALL_DRAIN_PER_SEC * dt)
        self.total_wall_hits.add_(wall_entered.float())
        self.prev_wall_contact.copy_(wall_touching)

        prev_combo = self.combo.clone()
        prev_multiplier = self.score_multiplier.clone()
        prev_progress = self.multiplier_progress.clone()

        miss_damage = num_misses * self._miss_energy
        badcut_damage = num_badcuts * (self._miss_energy * BAD_CUT_ENERGY_SCALE)
        bomb_damage = num_bombs * self._bomb_energy

        self.energy.add_(num_hits * 0.01)
        self.energy.sub_(miss_damage + badcut_damage + bomb_damage + wall_damage)
        prev_energy = (
            self.energy
            - num_hits * 0.01
            + miss_damage
            + badcut_damage
            + bomb_damage
            + wall_damage
        )
        just_died = (self.energy <= 0) & (prev_energy > 0) & self._fail_enabled
        self.energy.clamp_(0.0, 1.0)

        hit_multiplier, hit_progress = _advance_multiplier(prev_multiplier, prev_progress, num_hits)
        weighted_hit_score = hit_score * hit_multiplier
        combo_broken = (num_misses + num_badcuts + num_bombs > 0.0) | wall_entered
        final_multiplier = torch.where(combo_broken, torch.clamp(hit_multiplier * 0.5, min=1.0), hit_multiplier)
        final_progress = torch.where(combo_broken, torch.zeros_like(hit_progress), hit_progress)
        new_combo = torch.where(
            combo_broken,
            torch.zeros_like(self.combo),
            torch.where(num_hits > 0.0, prev_combo + num_hits, prev_combo),
        )

        self.combo.copy_(torch.where(self.episode_done, prev_combo, new_combo))
        self.max_combo.copy_(torch.maximum(self.max_combo, self.combo))
        self.score_multiplier.copy_(torch.where(self.episode_done, prev_multiplier, final_multiplier))
        self.multiplier_progress.copy_(torch.where(self.episode_done, prev_progress, final_progress))
        self.total_engaged_scorable.add_(num_contacts)
        self.total_hits.add_(num_hits)
        self.total_misses.add_(num_misses + num_badcuts + num_bombs)
        self.total_badcuts.add_(num_badcuts)
        self.total_bombs.add_(num_bombs)
        self.total_note_misses.add_(num_misses)
        self.total_cut_scores.add_(hit_score)
        self.total_scores.add_(weighted_hit_score)

        resolved_scorable = real_hits | bad_cuts | missed
        self.total_resolved_scorable.add_(resolved_scorable.sum(1).float())
        resolved_notes = resolved_scorable | is_bomb_hit
        updated_note_active_local = note_active_local & ~resolved_notes
        # Only valid note slots are allowed to mutate the persistent mask.
        # Invalid/clamped window entries contribute `True`, which is neutral under
        # `amin` and therefore cannot resurrect already-resolved notes.
        note_keep_mask = (updated_note_active_local | ~note_valid).to(torch.int32)
        self._note_keep_buf.fill_(1)
        self._note_keep_buf.scatter_reduce_(
            1,
            note_idx,
            note_keep_mask,
            reduce="amin",
            include_self=True,
        )
        self.note_active.logical_and_(self._note_keep_buf.bool())
        self.active_note_counts.copy_(self.note_active.sum(1, dtype=torch.long))

        cut_normals = torch.where(
            is_red.unsqueeze(-1),
            left_cut_plane_normal.unsqueeze(1).expand(-1, ACTIVE_NOTE_WINDOW, -1),
            right_cut_plane_normal.unsqueeze(1).expand(-1, ACTIVE_NOTE_WINDOW, -1),
        )
        saber_dirs = torch.where(
            is_red.unsqueeze(-1),
            lf.unsqueeze(1).expand(-1, ACTIVE_NOTE_WINDOW, -1),
            rf.unsqueeze(1).expand(-1, ACTIVE_NOTE_WINDOW, -1),
        )

        if getattr(self, '_tracking', False):
            t_sec = self.current_times
            hit_envs, hit_notes = real_hits.nonzero(as_tuple=True)
            bad_envs, bad_notes = bad_cuts.nonzero(as_tuple=True)
            miss_envs, miss_notes = missed.nonzero(as_tuple=True)
            bomb_envs, bomb_notes = is_bomb_hit.nonzero(as_tuple=True)
            wall_envs = wall_entered.nonzero(as_tuple=True)[0]

            if hit_envs.numel() > 0:
                h_pre = pre_sc[hit_envs, hit_notes].cpu().tolist()
                h_post = post_sc[hit_envs, hit_notes].cpu().tolist()
                h_acc = acc_sc[hit_envs, hit_notes].cpu().tolist()
                h_speed = note_speed[hit_envs, hit_notes].cpu().tolist()
                h_type = note_types[hit_envs, hit_notes].cpu().tolist()
                h_time = t_sec[hit_envs].cpu().tolist()
                h_nidx = note_raw_idx[hit_envs, hit_notes].cpu().tolist()
                h_env = hit_envs.cpu().tolist()
                h_is_left = (note_types[hit_envs, hit_notes] == 0).cpu().tolist()
                h_cut_distance = cut_distance[hit_envs, hit_notes].cpu().tolist()
                h_dir_score = torch.where(
                    is_red,
                    left_dir_dot / left_dir_mag,
                    right_dir_dot / right_dir_mag,
                )[hit_envs, hit_notes].cpu().tolist()
                h_cut_points = torch.where(
                    is_red.unsqueeze(-1),
                    left_cut_point,
                    right_cut_point,
                )[hit_envs, hit_notes].cpu().tolist()
                h_cut_normals = cut_normals[hit_envs, hit_notes].cpu().tolist()
                h_saber_dirs = saber_dirs[hit_envs, hit_notes].cpu().tolist()

                for i in range(len(h_env)):
                    self.tracked_events[h_env[i]].append({
                        'type': 'hit',
                        'note_index': int(h_nidx[i]),
                        'time': h_time[i],
                        'pre_score': h_pre[i],
                        'post_score': h_post[i],
                        'acc_score': h_acc[i],
                        'cut_distance': float(h_cut_distance[i]),
                        'direction_score': float(h_dir_score[i]),
                        'saber_speed': float(h_speed[i]),
                        'saber_type': int(h_type[i]),
                        'used_saber_type': 0 if h_is_left[i] else 1,
                        'cut_point': [float(x) for x in h_cut_points[i]],
                        'saber_dir': [float(x) for x in h_saber_dirs[i]],
                        'cut_normal': [float(x) for x in h_cut_normals[i]],
                    })

            if bad_envs.numel() > 0:
                bad_used_left = (
                    (is_blue & wrong_color_contact)
                    | (is_red & contact_hits & (~speed_ok | ~dir_ok))
                )
                bad_speed_ok = torch.where(bad_used_left, left_speed_ok, right_speed_ok)
                bad_dir_ok = torch.where(bad_used_left, left_dir_ok, right_dir_ok)
                bad_saber_type_ok = ~wrong_color_contact
                bad_cut_distance = torch.where(bad_used_left, left_cut_distance, right_cut_distance)
                bad_note_speed = torch.where(
                    bad_used_left,
                    left_tip_speed.unsqueeze(1).expand_as(note_times),
                    right_tip_speed.unsqueeze(1).expand_as(note_times),
                )
                bad_dir_score = torch.where(
                    bad_used_left,
                    left_dir_dot / left_dir_mag,
                    right_dir_dot / right_dir_mag,
                )
                bad_left_contact = (left_good_contact | left_bad_contact)
                bad_right_contact = (right_good_contact | right_bad_contact)
                bad_cut_points = torch.where(bad_used_left.unsqueeze(-1), left_cut_point, right_cut_point)
                bad_cut_normals = torch.where(
                    bad_used_left.unsqueeze(-1),
                    left_cut_plane_normal.unsqueeze(1).expand(-1, ACTIVE_NOTE_WINDOW, -1),
                    right_cut_plane_normal.unsqueeze(1).expand(-1, ACTIVE_NOTE_WINDOW, -1),
                )
                bad_saber_dirs = torch.where(
                    bad_used_left.unsqueeze(-1),
                    lf.unsqueeze(1).expand(-1, ACTIVE_NOTE_WINDOW, -1),
                    rf.unsqueeze(1).expand(-1, ACTIVE_NOTE_WINDOW, -1),
                )
                b_time = t_sec[bad_envs].cpu().tolist()
                b_nidx = note_raw_idx[bad_envs, bad_notes].cpu().tolist()
                b_env = bad_envs.cpu().tolist()
                b_type = note_types[bad_envs, bad_notes].cpu().tolist()
                b_speed_ok = bad_speed_ok[bad_envs, bad_notes].cpu().tolist()
                b_dir_ok = bad_dir_ok[bad_envs, bad_notes].cpu().tolist()
                b_saber_ok = bad_saber_type_ok[bad_envs, bad_notes].cpu().tolist()
                b_cut_distance = bad_cut_distance[bad_envs, bad_notes].cpu().tolist()
                b_note_speed = bad_note_speed[bad_envs, bad_notes].cpu().tolist()
                b_dir_score = bad_dir_score[bad_envs, bad_notes].cpu().tolist()
                b_cut_points = bad_cut_points[bad_envs, bad_notes].cpu().tolist()
                b_cut_normals = bad_cut_normals[bad_envs, bad_notes].cpu().tolist()
                b_saber_dirs = bad_saber_dirs[bad_envs, bad_notes].cpu().tolist()
                b_used_saber = bad_used_left[bad_envs, bad_notes].cpu().tolist()
                b_left_contact = bad_left_contact[bad_envs, bad_notes].cpu().tolist()
                b_right_contact = bad_right_contact[bad_envs, bad_notes].cpu().tolist()

                for i in range(len(b_env)):
                    self.tracked_events[b_env[i]].append({
                        'type': 'bad',
                        'note_index': int(b_nidx[i]),
                        'time': b_time[i],
                        'speed_ok': bool(b_speed_ok[i]),
                        'direction_ok': bool(b_dir_ok[i]),
                        'direction_score': float(b_dir_score[i]),
                        'saber_type_ok': bool(b_saber_ok[i]),
                        'cut_distance': float(b_cut_distance[i]),
                        'saber_speed': float(b_note_speed[i]),
                        'saber_type': int(b_type[i]),
                        'used_saber_type': 0 if b_used_saber[i] else 1,
                        'left_contact': bool(b_left_contact[i]),
                        'right_contact': bool(b_right_contact[i]),
                        'cut_point': [float(x) for x in b_cut_points[i]],
                        'saber_dir': [float(x) for x in b_saber_dirs[i]],
                        'cut_normal': [float(x) for x in b_cut_normals[i]],
                    })

            if miss_envs.numel() > 0:
                miss_used_left = is_red[miss_envs, miss_notes]
                m_type = note_types[miss_envs, miss_notes].cpu().tolist()
                m_time = t_sec[miss_envs].cpu().tolist()
                m_nidx = note_raw_idx[miss_envs, miss_notes].cpu().tolist()
                m_env = miss_envs.cpu().tolist()
                m_used_saber = miss_used_left.cpu().tolist()
                m_speed_ok = torch.where(
                    miss_used_left,
                    left_speed_ok[miss_envs, miss_notes],
                    right_speed_ok[miss_envs, miss_notes],
                ).cpu().tolist()
                m_dir_ok = torch.where(
                    miss_used_left,
                    left_dir_ok[miss_envs, miss_notes],
                    right_dir_ok[miss_envs, miss_notes],
                ).cpu().tolist()
                m_cut_zone = torch.where(
                    miss_used_left,
                    left_cut_zone[miss_envs, miss_notes],
                    right_cut_zone[miss_envs, miss_notes],
                ).cpu().tolist()
                m_correct_contact = torch.where(
                    miss_used_left,
                    left_good_contact[miss_envs, miss_notes],
                    right_good_contact[miss_envs, miss_notes],
                ).cpu().tolist()
                m_wrong_contact = torch.where(
                    miss_used_left,
                    right_bad_contact[miss_envs, miss_notes],
                    left_bad_contact[miss_envs, miss_notes],
                ).cpu().tolist()
                m_dir_score = torch.where(
                    miss_used_left,
                    (left_dir_dot / left_dir_mag)[miss_envs, miss_notes],
                    (right_dir_dot / right_dir_mag)[miss_envs, miss_notes],
                ).cpu().tolist()
                m_cut_distance = torch.where(
                    miss_used_left,
                    left_cut_distance[miss_envs, miss_notes],
                    right_cut_distance[miss_envs, miss_notes],
                ).cpu().tolist()
                m_note_speed = torch.where(
                    miss_used_left,
                    left_tip_speed[miss_envs],
                    right_tip_speed[miss_envs],
                ).cpu().tolist()
                m_note_time_delta = note_time_delta[miss_envs, miss_notes].cpu().tolist()
                m_left_contact = (left_good_contact | left_bad_contact)[miss_envs, miss_notes].cpu().tolist()
                m_right_contact = (right_good_contact | right_bad_contact)[miss_envs, miss_notes].cpu().tolist()
                m_cut_points = torch.where(
                    miss_used_left.unsqueeze(-1),
                    left_cut_point[miss_envs, miss_notes],
                    right_cut_point[miss_envs, miss_notes],
                ).cpu().tolist()
                m_cut_normals = torch.where(
                    miss_used_left.unsqueeze(-1),
                    left_cut_plane_normal[miss_envs],
                    right_cut_plane_normal[miss_envs],
                ).cpu().tolist()
                m_saber_dirs = torch.where(
                    miss_used_left.unsqueeze(-1),
                    lf[miss_envs],
                    rf[miss_envs],
                ).cpu().tolist()
                for i in range(len(m_env)):
                    self.tracked_events[m_env[i]].append({
                        'type': 'miss',
                        'note_index': int(m_nidx[i]),
                        'time': m_time[i],
                        'saber_type': int(m_type[i]),
                        'used_saber_type': 0 if m_used_saber[i] else 1,
                        'speed_ok': bool(m_speed_ok[i]),
                        'direction_ok': bool(m_dir_ok[i]),
                        'cut_zone': bool(m_cut_zone[i]),
                        'correct_contact': bool(m_correct_contact[i]),
                        'wrong_color_contact': bool(m_wrong_contact[i]),
                        'direction_score': float(m_dir_score[i]),
                        'cut_distance': float(m_cut_distance[i]),
                        'saber_speed': float(m_note_speed[i]),
                        'note_time_delta': float(m_note_time_delta[i]),
                        'left_contact': bool(m_left_contact[i]),
                        'right_contact': bool(m_right_contact[i]),
                        'cut_point': [float(x) for x in m_cut_points[i]],
                        'saber_dir': [float(x) for x in m_saber_dirs[i]],
                        'cut_normal': [float(x) for x in m_cut_normals[i]],
                    })

            if bomb_envs.numel() > 0:
                b_time = t_sec[bomb_envs].cpu().tolist()
                b_nidx = note_raw_idx[bomb_envs, bomb_notes].cpu().tolist()
                b_env = bomb_envs.cpu().tolist()
                left_dist = dl[bomb_envs, bomb_notes].cpu().tolist()
                right_dist = dr[bomb_envs, bomb_notes].cpu().tolist()
                for i in range(len(b_env)):
                    saber_type = 0 if left_dist[i] <= right_dist[i] else 1
                    self.tracked_events[b_env[i]].append({
                        'type': 'bomb',
                        'note_index': int(b_nidx[i]),
                        'time': b_time[i],
                        'saber_type': saber_type,
                    })

            if wall_envs.numel() > 0:
                wall_idx = wall_collision_mask.float().argmax(dim=1)
                wall_times = t_sec[wall_envs].cpu().tolist()
                wall_obs = obstacle_raw_idx[self._env_indices, wall_idx][wall_envs].cpu().tolist()
                wall_energy = self.energy[wall_envs].cpu().tolist()
                for i, env_idx in enumerate(wall_envs.cpu().tolist()):
                    self.tracked_events[env_idx].append({
                        'type': 'wall',
                        'obstacle_index': int(wall_obs[i]),
                        'time': float(wall_times[i]),
                        'energy': float(wall_energy[i]),
                    })

        has_objects = (self.note_counts > 0) | (self.obstacle_counts > 0)
        live_note_counts = self.active_note_counts
        live_obstacle_counts = self.remaining_obstacle_counts
        resolved_fraction = self.total_resolved_scorable / self.scorable_note_counts.clamp(min=1).float()
        all_scorable_resolved = (self.scorable_note_counts == 0) | (resolved_fraction >= 0.98)
        song_cleared = (live_note_counts == 0) & (live_obstacle_counts == 0) & has_objects & all_scorable_resolved
        timed_out = self.current_times >= self.map_durations
        done_now = just_died | song_cleared | timed_out
        completion_ratio = (self.current_times / self.map_durations.clamp(min=1e-6)).clamp(0.0, 1.0)
        new_reason = torch.where(
            just_died,
            torch.ones_like(self._terminal_reason),
            torch.where(
                song_cleared,
                torch.full_like(self._terminal_reason, 2),
                torch.where(timed_out, torch.full_like(self._terminal_reason, 3), torch.zeros_like(self._terminal_reason)),
            ),
        )
        should_write_reason = done_now & (self._terminal_reason == 0)
        self._terminal_reason.copy_(torch.where(should_write_reason, new_reason, self._terminal_reason))
        self._completion_ratio.copy_(torch.where(should_write_reason, completion_ratio, self._completion_ratio))
        self.episode_done |= done_now
        self._done_out.copy_(self.episode_done)

        if self._score_only_mode:
            self._reward_out.zero_()
            self._reward_components.zero_()
            self._success_out.fill_(1.0)
            self.prev_accel.zero_()
            self.prev_head_delta.copy_(head_delta.detach())
            self.prev_hand_delta.copy_(hand_delta.detach())
            torch.nan_to_num_(self._reward_out, nan=0.0, posinf=0.0, neginf=0.0)
            torch.nan_to_num_(self.total_hits, nan=0.0, posinf=0.0, neginf=0.0)
            torch.nan_to_num_(self.total_misses, nan=0.0, posinf=0.0, neginf=0.0)
            torch.nan_to_num_(self.total_badcuts, nan=0.0, posinf=0.0, neginf=0.0)
            torch.nan_to_num_(self.total_bombs, nan=0.0, posinf=0.0, neginf=0.0)
            torch.nan_to_num_(self.total_note_misses, nan=0.0, posinf=0.0, neginf=0.0)
            torch.nan_to_num_(self.total_engaged_scorable, nan=0.0, posinf=0.0, neginf=0.0)
            torch.nan_to_num_(self.total_cut_scores, nan=0.0, posinf=0.0, neginf=0.0)
            torch.nan_to_num_(self.total_scores, nan=0.0, posinf=0.0, neginf=0.0)
            return self._reward_out, self._delta_out

        target_candidates = note_valid & updated_note_active_local & (note_types != 3)
        has_candidate = target_candidates.any(1)
        next_note_slot = target_candidates.float().argmax(dim=1)
        next_note_idx = note_raw_idx[self._env_indices, next_note_slot].clamp(min=0)
        target_type = note_types[self._env_indices, next_note_slot]
        has_target = has_candidate & (~self.episode_done)
        target_pos = npos[self._env_indices, next_note_slot]
        target_dist = torch.where(
            target_type == 0,
            torch.norm(lt - target_pos, dim=-1),
            torch.where(
                target_type == 1,
                torch.norm(rt - target_pos, dim=-1),
                torch.zeros_like(self._prev_target_dist),
            ),
        )
        target_changed = (next_note_idx != self._prev_target_note_idx) | (~has_target)
        prev_target_dist = torch.where(target_changed, target_dist, self._prev_target_dist)
        approach_delta = (prev_target_dist - target_dist).clamp(-0.5, 0.5)
        in_approach_window = (target_pos[:, 2] > -0.75) & (target_pos[:, 2] < self._style_approach_far) & has_target
        self._prev_target_note_idx.copy_(torch.where(has_target, next_note_idx, self._no_target_idx))
        self._prev_target_dist.copy_(torch.where(has_target, target_dist, torch.zeros_like(target_dist)))

        target_prev_tip = torch.where(
            (target_type == 0).unsqueeze(1),
            plt,
            torch.where((target_type == 1).unsqueeze(1), prt, torch.zeros_like(plt)),
        )
        target_curr_tip = torch.where(
            (target_type == 0).unsqueeze(1),
            lt,
            torch.where((target_type == 1).unsqueeze(1), rt, torch.zeros_like(lt)),
        )
        target_step_vec = target_curr_tip - target_prev_tip
        target_to_note = target_pos - target_prev_tip
        target_to_note_norm = torch.norm(target_to_note, dim=-1, keepdim=True).clamp(min=1e-6)
        target_dir_vec = target_to_note / target_to_note_norm
        projected_motion = (target_step_vec * target_dir_vec).sum(-1)
        direct_progress = torch.relu(projected_motion) * in_approach_window.float()
        lateral_motion = (
            torch.norm(target_step_vec - target_dir_vec * projected_motion.unsqueeze(-1), dim=-1)
            * in_approach_window.float()
        )
        target_step_mag = torch.norm(target_step_vec, dim=-1)
        motion_directness = (direct_progress / target_step_mag.clamp(min=1e-6)).clamp(0.0, 1.0)
        approach_gain = torch.relu(approach_delta)
        approach_backtrack = torch.relu(-approach_delta)
        dense_approach_reward = (
            approach_gain * (0.25 + 0.75 * motion_directness)
            - approach_backtrack * 0.25
        ) * in_approach_window.float() * self._w_approach
        target_hand_delta = torch.where(
            (target_type == 0).unsqueeze(1),
            hand_delta[:, 0:3],
            torch.where((target_type == 1).unsqueeze(1), hand_delta[:, 7:10], torch.zeros_like(hand_delta[:, 0:3])),
        )
        prev_target_hand_delta = torch.where(
            (target_type == 0).unsqueeze(1),
            self.prev_hand_delta[:, 0:3],
            torch.where(
                (target_type == 1).unsqueeze(1),
                self.prev_hand_delta[:, 7:10],
                torch.zeros_like(self.prev_hand_delta[:, 0:3]),
            ),
        )
        delta_cos = (target_hand_delta * prev_target_hand_delta).sum(-1) / (
            torch.norm(target_hand_delta, dim=-1).clamp(min=1e-6)
            * torch.norm(prev_target_hand_delta, dim=-1).clamp(min=1e-6)
        )
        oscillation = torch.relu(-delta_cos) * (torch.norm(target_hand_delta, dim=-1) > 1e-4).float() * in_approach_window.float()
        target_motion = torch.where(
            target_type == 0,
            left_step,
            torch.where(target_type == 1, right_step, torch.zeros_like(left_step)),
        )
        support_motion = torch.where(
            target_type == 0,
            right_step,
            torch.where(target_type == 1, left_step, torch.zeros_like(right_step)),
        )
        useful_progress = direct_progress
        allowed_motion = self._style_step_allowance + useful_progress * 1.15
        guided_motion = target_motion + support_motion * self._style_support_scale
        waste_motion = torch.relu(guided_motion - allowed_motion) * in_approach_window.float()
        idle_motion = torch.relu((left_step + right_step) - (self._style_step_allowance * 0.90)) * (~in_approach_window).float() * 1.10
        style_waste = waste_motion + idle_motion + lateral_motion * 0.90 + oscillation * 0.55
        speed_violation = torch.relu(peak_hand_speed - self._style_speed_cap)
        angular_violation = torch.relu(peak_ang_speed - self._style_angular_cap)
        self.motion_path.add_((left_step + right_step) * active_env_f)
        self.useful_progress.add_(useful_progress * active_env_f)
        self.mean_speed_sum.add_(mean_hand_speed * active_env_f)
        self.speed_samples.add_(active_env_f)
        self.speed_violation_sum.add_(speed_violation * active_env_f)
        self.angular_violation_sum.add_(angular_violation * active_env_f)
        self.waste_motion_sum.add_(style_waste * active_env_f)
        self.idle_motion_sum.add_(idle_motion * active_env_f)
        self.oscillation_sum.add_(oscillation * active_env_f)
        self.lateral_motion_sum.add_(lateral_motion * active_env_f)

        dir_score = torch.where(
            is_red,
            left_dir_dot / left_dir_mag,
            torch.where(is_blue, right_dir_dot / right_dir_mag, torch.zeros_like(left_dir_dot)),
        ).clamp(-1.0, 1.0)
        dir_quality = torch.where(is_dot, torch.ones_like(dir_score), ((dir_score + 1.0) * 0.5).clamp(0.0, 1.0))
        arc_prep = (pre_angle / 70.0).clamp(0.0, 1.0)
        arc_follow = (post_angle / 60.0).clamp(0.0, 1.0)
        arc_speed = (note_speed / self._style_speed_cap.unsqueeze(1).clamp(min=1e-6)).clamp(0.0, 1.0)
        hit_arc_quality = (0.35 * arc_prep + 0.25 * arc_follow + 0.20 * arc_speed + 0.20 * dir_quality).clamp(0.0, 1.0)
        swing_arc_reward = (real_hits.float() * hit_arc_quality).sum(1) * self._w_swing_arc
        perfect_hits = real_hits & (note_cut_scores >= (note_score_cap - 0.5))
        perfect_hit_bonus = perfect_hits.sum(1).float() * 6.0 * self._w_perfect_hit
        idle_zone = (~in_approach_window).float()
        guard_left_target = self._guard_left_target + head_translation
        guard_right_target = self._guard_right_target + head_translation
        guard_error = (
            torch.norm(new_hands[:, 0:3] - guard_left_target, dim=-1)
            + torch.norm(new_hands[:, 7:10] - guard_right_target, dim=-1)
        ) * idle_zone
        self.guard_error_sum.add_(guard_error * active_env_f)
        idle_guard_penalty = guard_error * self._w_idle_guard

        curr_accel = hand_delta - self.prev_hand_delta
        jerk = curr_accel - self.prev_accel
        jerk_penalty = jerk.pow(2).sum(1)
        pos_jerk = jerk.index_select(1, self._pos_idx)
        pos_jerk_penalty = pos_jerk.pow(2).sum(1)

        dynamic_shoulder_l = self._shoulder_l.view(1, 3) + head_translation
        dynamic_shoulder_r = self._shoulder_r.view(1, 3) + head_translation
        l_reach = torch.norm(new_hands[:, 0:3] - dynamic_shoulder_l, dim=-1)
        r_reach = torch.norm(new_hands[:, 7:10] - dynamic_shoulder_r, dim=-1)
        reach_penalty = torch.relu(l_reach - 0.85).pow(2) + torch.relu(r_reach - 0.85).pow(2)

        energy_delta_reward = (self.energy - prev_energy) * self._w_energy
        survival_reward = self._w_survival * active_env_f
        combo_bonus = torch.sqrt(self.combo.clamp(min=0.0)) * (num_hits > 0.0).float() * self._w_combo
        contact_reward = num_hits * self._w_contact
        low_energy_penalty = torch.relu(0.30 - self.energy) * self._w_low_energy
        style_speed_penalty = speed_violation * self._w_style_speed
        style_waste_penalty = style_waste * self._w_style_waste
        style_angular_penalty = angular_violation * self._w_style_angular
        style_lateral_penalty = lateral_motion * self._w_style_lateral
        style_oscillation_penalty = oscillation * self._w_style_oscillation
        wall_penalty = wall_touching.float() * 2.0

        self._reward_out.copy_(
            (
                (hit_score / SCORE_CAP_NORMAL) * 20.0
                + swing_arc_reward
                + perfect_hit_bonus
                + dense_approach_reward
                + contact_reward
                + energy_delta_reward
                + survival_reward
                + combo_bonus
                - (num_misses + num_badcuts) * self._w_miss
                - num_bombs * 5.0
                - wall_penalty
                - self._w_jerk * jerk_penalty
                - self._w_pos_jerk * pos_jerk_penalty
                - self._w_reach * reach_penalty
                - low_energy_penalty
                - style_speed_penalty
                - style_waste_penalty
                - style_angular_penalty
                - style_lateral_penalty
                - style_oscillation_penalty
                - idle_guard_penalty
            ) * active_env_f
            - just_died.float() * self._death_penalty
        )

        components = self._reward_components
        components[:, 0].copy_(((hit_score / SCORE_CAP_NORMAL) * 20.0) * active_env_f)
        components[:, 1].copy_((swing_arc_reward + perfect_hit_bonus) * active_env_f)
        components[:, 2].copy_(dense_approach_reward * active_env_f)
        components[:, 3].copy_(contact_reward * active_env_f)
        components[:, 4].copy_(energy_delta_reward * active_env_f)
        components[:, 5].copy_(survival_reward * active_env_f)
        components[:, 6].copy_(combo_bonus * active_env_f)
        components[:, 7].copy_(-((num_misses + num_badcuts) * self._w_miss) * active_env_f)
        components[:, 8].copy_(-(num_bombs * 5.0) * active_env_f)
        components[:, 9].copy_(-wall_penalty * active_env_f)
        components[:, 10].copy_(
            -(
                self._w_jerk * jerk_penalty
                + self._w_pos_jerk * pos_jerk_penalty
                + self._w_reach * reach_penalty
            ) * active_env_f
        )
        components[:, 11].copy_(
            -(
                low_energy_penalty
                + style_speed_penalty
                + style_waste_penalty
                + style_angular_penalty
                + style_lateral_penalty
                + style_oscillation_penalty
                + idle_guard_penalty
            ) * active_env_f
        )
        components[:, 12].copy_(-(just_died.float() * self._death_penalty))
        components[:, 13].copy_(self._reward_out)

        self._success_out.fill_(1.0)

        self.prev_accel.copy_(curr_accel.detach())
        self.prev_head_delta.copy_(head_delta.detach())
        self.prev_hand_delta.copy_(hand_delta.detach())
        torch.nan_to_num_(self._reward_out, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nan_to_num_(self._reward_components, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nan_to_num_(self.total_hits, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nan_to_num_(self.total_misses, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nan_to_num_(self.total_badcuts, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nan_to_num_(self.total_bombs, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nan_to_num_(self.total_note_misses, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nan_to_num_(self.total_engaged_scorable, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nan_to_num_(self.total_cut_scores, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nan_to_num_(self.total_scores, nan=0.0, posinf=0.0, neginf=0.0)

        return self._reward_out, self._delta_out
