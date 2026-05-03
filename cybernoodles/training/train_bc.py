import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
import math
import json
import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

from cybernoodles.core.network import (
    ActorCritic,
    CURRENT_POSE_END,
    CURRENT_POSE_START,
    INPUT_DIM,
    NOTE_LANE_INDEX,
    NOTES_DIM,
    NOTE_TIME_INDEX,
    NOTE_TYPE_INDEX,
    NOTE_LAYER_INDEX,
    OBSTACLES_DIM,
    POSE_DIM,
    STATE_FRAME_DIM,
    STATE_HISTORY_FRAMES,
    NOTE_FEATURES,
    OBSTACLE_FEATURES,
    build_rl_bootstrap_state_dict,
)
from cybernoodles.data.dataset_builder import (
    MANIFEST_PATH,
    OUTPUT_DIR,
    SHARD_ROOT,
    manifest_compatibility_errors,
)
from cybernoodles.data.shard_io import load_shard_pair, shard_record_label, validate_shard_record
from cybernoodles.data.sim_calibration import load_simulator_calibration
from cybernoodles.envs import DEFAULT_BC_PROBE_MAPS
from cybernoodles.training.policy_checkpoint import (
    attach_policy_schema,
    extract_policy_state_dict,
)
from cybernoodles.training.policy_eval import (
    choose_eval_hashes,
    evaluate_policy_model,
    get_eval_profile,
    load_curriculum,
)
from cybernoodles.training.bc_prefetch import (
    CudaBatchPrefetcher,
    LoaderProfile,
    ThreadedPrefetchIterator,
)

LEGACY_X_PATH = os.path.join(OUTPUT_DIR, "X_rl.pt")
LEGACY_Y_PATH = os.path.join(OUTPUT_DIR, "y_rl.pt")
BC_MODEL_PATH = "bsai_bc_model.pth"
BC_LAST_MODEL_PATH = "bsai_bc_last.pth"
SIM_NOTE_TIME_MIN_BEATS = -1.0
SIM_NOTE_TIME_MAX_BEATS = 4.0
SIM_OBSTACLE_TIME_MIN_BEATS = -1.0
SIM_OBSTACLE_TIME_MAX_BEATS = 6.0
DEFAULT_BC_SIM_PROBE_MAPS = DEFAULT_BC_PROBE_MAPS
DEFAULT_BC_SIM_PROBE_ENVS = 12
PROBE_RECOVERY_MIN_COVERAGE = 0.01
PROBE_RECOVERY_MIN_ACCURACY = 0.5
PROBE_RECOVERY_MIN_COMPLETION = 0.04
PROBE_PLAY_MIN_COVERAGE = 0.04
PROBE_PLAY_MIN_ACCURACY = 2.0
PROBE_PLAY_MIN_COMPLETION = 0.10
DEFAULT_LOADER_PREFETCH_BATCHES = 4
DEFAULT_LOADER_PREFETCH_SHARDS = 2
DIM = "\033[2m"
RST = "\033[0m"

CALIBRATION = load_simulator_calibration()
LOCAL_SABER_AXIS = torch.tensor(CALIBRATION.get("saber_axis", [0.0, 0.0, 1.0]), dtype=torch.float32)
LOCAL_SABER_AXIS = LOCAL_SABER_AXIS / LOCAL_SABER_AXIS.norm().clamp(min=1e-6)
LOCAL_SABER_ORIGIN = torch.tensor(CALIBRATION.get("saber_origin", [0.0, 0.0, 0.0]), dtype=torch.float32)
SABER_LENGTH = float(CALIBRATION.get("saber_length", 1.0))
NOTE_X_OFFSET = float(CALIBRATION.get("x_offset", -0.9))
NOTE_X_SPACING = float(CALIBRATION.get("x_spacing", 0.6))
NOTE_Y_OFFSET = float(CALIBRATION.get("y_offset", 0.85))
NOTE_Y_SPACING = float(CALIBRATION.get("y_spacing", 0.35))
IMMINENT_NOTE_BEAT_WINDOW = 0.55
NOTE_CUT_DX_INDEX = 4
NOTE_CUT_DY_INDEX = 5
BC_LOSS_PRESETS = {
    "balanced": {
        "pos": 1.00,
        "motion": 0.50,
        "rot": 0.18,
        "tip": 0.70,
        "swing": 0.95,
        "note": 0.65,
        "direction": 0.00,
    },
    "cut": {
        "pos": 0.55,
        "motion": 0.45,
        "rot": 0.12,
        "tip": 2.25,
        "swing": 3.00,
        "note": 0.35,
        "direction": 1.75,
    },
}
ACTIVE_BC_LOSS_WEIGHTS = dict(BC_LOSS_PRESETS["cut"])


def make_adam(params, lr):
    try:
        return optim.Adam(params, lr=lr, fused=True)
    except TypeError:
        return optim.Adam(params, lr=lr)


def resolve_bc_loss_weights(loss_preset):
    key = str(loss_preset or "cut").strip().lower()
    if key not in BC_LOSS_PRESETS:
        valid = ", ".join(sorted(BC_LOSS_PRESETS))
        raise ValueError(f"Unknown BC loss preset: {loss_preset!r}. Valid presets: {valid}")
    return dict(BC_LOSS_PRESETS[key])


def set_bc_loss_preset(loss_preset):
    ACTIVE_BC_LOSS_WEIGHTS.clear()
    ACTIVE_BC_LOSS_WEIGHTS.update(resolve_bc_loss_weights(loss_preset))
    return dict(ACTIVE_BC_LOSS_WEIGHTS)


def suggest_batch_size(device):
    if device.type != 'cuda' or not torch.cuda.is_available():
        return 8192

    device_index = device.index if device.index is not None else 0
    total_gb = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
    if total_gb >= 12.0:
        return 131072
    if total_gb >= 7.5:
        return 98304
    if total_gb >= 5.5:
        return 65536
    if total_gb >= 3.5:
        return 32768
    return 16384


def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        return None
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    compatibility_errors = manifest_compatibility_errors(manifest)
    if compatibility_errors:
        print("\033[93mBC shard manifest is out of date for the current training schema.\033[0m")
        for detail in compatibility_errors[:5]:
            print(f"\033[93m  - {detail}\033[0m")
        print("\033[93mRe-run `python -m cybernoodles.data.dataset_builder` before training BC again.\033[0m")
        print("\033[93m`main.py` option 1 will rebuild the shards for you before BC training.\033[0m")
        return None
    return manifest


def split_records(manifest, split):
    return [record for record in manifest.get("shards", []) if record.get("split") == split]


def take_record_subset(records, limit, seed=1337):
    if limit is None or limit <= 0 or limit >= len(records):
        return list(records)
    picker = random.Random(seed)
    subset = picker.sample(list(records), limit)
    subset.sort(key=lambda record: str(record.get("replay_file", "")))
    return subset


def preflight_shard_records(records, split_label, shard_root=SHARD_ROOT, max_errors=8):
    checked = 0
    samples = 0
    errors = []
    for record in records:
        try:
            info = validate_shard_record(
                record,
                shard_root,
                expected_feature_dim=INPUT_DIM,
                expected_target_dim=POSE_DIM,
                expected_dtype=torch.float16,
            )
            checked += 1
            samples += int(info["samples"])
        except Exception as exc:
            label = str(record.get("replay_file") or record.get("shard_path") or record.get("x_path") or "unknown")
            errors.append(f"{split_label}:{label}: {exc}")
            if len(errors) >= max_errors:
                break

    if errors:
        detail = "\n  - ".join(errors)
        raise RuntimeError(
            f"BC shard preflight failed for {split_label} split. "
            "Rebuild shards with `python -m cybernoodles.data.dataset_builder`.\n"
            f"  - {detail}"
        )

    return {
        "records": checked,
        "samples": samples,
    }


def normalize_quaternion_slice(tensor):
    norm = torch.norm(tensor, dim=-1, keepdim=True).clamp(min=1e-6)
    return tensor / norm


def sample_weights_from_state(batch_x):
    next_note_time = batch_x[:, NOTE_TIME_INDEX].float().clamp(min=0.0, max=4.0)
    note_type = batch_x[:, NOTE_TYPE_INDEX].long()
    scorable_note_mask = (note_type == 0) | (note_type == 1)
    imminent_note_weight = scorable_note_mask.float() * torch.exp(-next_note_time / 0.24)
    return 1.0 + 4.0 * imminent_note_weight


def quaternion_alignment_loss(pred_q, target_q):
    pred_q = normalize_quaternion_slice(pred_q)
    target_q = normalize_quaternion_slice(target_q)
    dot = torch.abs((pred_q * target_q).sum(dim=-1)).clamp(0.0, 1.0)
    return 1.0 - dot


def rotate_local_vector(quat, local_vec, device, dtype):
    quat = normalize_quaternion_slice(quat)
    q_xyz = quat[:, :3]
    q_w = quat[:, 3:4]
    vec = local_vec.to(device=device, dtype=dtype).view(1, 3).expand_as(q_xyz)
    t = 2.0 * torch.cross(q_xyz, vec, dim=-1)
    return vec + q_w * t + torch.cross(q_xyz, t, dim=-1)


def rotate_local_axis(quat, device, dtype):
    return rotate_local_vector(quat, LOCAL_SABER_AXIS, device, dtype)


def predicted_saber_tips(pred):
    left_hilt = pred[:, 7:10] + rotate_local_vector(pred[:, 10:14], LOCAL_SABER_ORIGIN, pred.device, pred.dtype)
    right_hilt = pred[:, 14:17] + rotate_local_vector(pred[:, 17:21], LOCAL_SABER_ORIGIN, pred.device, pred.dtype)
    left_dir = rotate_local_axis(pred[:, 10:14], pred.device, pred.dtype)
    right_dir = rotate_local_axis(pred[:, 17:21], pred.device, pred.dtype)
    left_tip = left_hilt + left_dir * SABER_LENGTH
    right_tip = right_hilt + right_dir * SABER_LENGTH
    return left_tip, right_tip


def resolve_warmstart_path(init_from):
    if not init_from:
        return None
    value = str(init_from).strip()
    if not value:
        return None
    lowered = value.lower()
    if lowered == "best":
        return BC_MODEL_PATH
    if lowered == "last":
        return BC_LAST_MODEL_PATH
    return value


def load_bc_warmstart(model, init_from):
    warmstart_path = resolve_warmstart_path(init_from)
    if not warmstart_path:
        return None
    if not os.path.exists(warmstart_path):
        raise FileNotFoundError(f"Warm-start checkpoint not found: {warmstart_path}")
    payload = torch.load(warmstart_path, weights_only=False)
    state = extract_policy_state_dict(
        payload,
        checkpoint_path=warmstart_path,
        accepted_keys=("model_state_dict", "actor_state_dict"),
        allow_legacy=True,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    loaded_count = len(state.keys()) - len(unexpected)
    print(f"Warm-started BC model from {warmstart_path} ({loaded_count} tensors loaded).")
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys ignored: {len(unexpected)}")
    return warmstart_path


def save_policy_actor_checkpoint(path, state_dict, checkpoint_kind):
    torch.save(
        attach_policy_schema({
            "checkpoint_kind": checkpoint_kind,
            "model_state_dict": state_dict,
        }),
        path,
    )


def run_bc_sim_probe(
    model,
    device,
    max_maps=DEFAULT_BC_PROBE_MAPS,
    num_envs=1,
    suite="starter",
    profile="strict",
    verbose=False,
):
    curriculum = load_curriculum()
    if not curriculum:
        return None
    map_hashes = choose_eval_hashes(
        curriculum,
        max_maps=max_maps,
        suite=suite,
        split="dev_eval",
    )
    if not map_hashes:
        return None
    probe_cfg = get_eval_profile(profile)
    return evaluate_policy_model(
        model,
        device,
        map_hashes,
        num_envs=max(1, int(num_envs)),
        noise_scale=0.0,
        label="bc-probe",
        verbose=verbose,
        **probe_cfg,
    )


def probe_has_real_signal(probe_stats):
    if not probe_stats:
        return False
    return (
        float(probe_stats.get("mean_note_coverage", 0.0)) >= PROBE_RECOVERY_MIN_COVERAGE
        or float(probe_stats.get("mean_accuracy", 0.0)) >= PROBE_RECOVERY_MIN_ACCURACY
        or float(probe_stats.get("mean_completion", 0.0)) >= PROBE_RECOVERY_MIN_COMPLETION
    )


def probe_has_play_signal(probe_stats):
    if not probe_stats:
        return False
    return (
        float(probe_stats.get("mean_note_coverage", 0.0)) >= PROBE_PLAY_MIN_COVERAGE
        and float(probe_stats.get("mean_accuracy", 0.0)) >= PROBE_PLAY_MIN_ACCURACY
        and float(probe_stats.get("mean_completion", 0.0)) >= PROBE_PLAY_MIN_COMPLETION
    )


def probe_sort_key(probe_stats, eval_stats=None):
    if not probe_stats:
        return None
    loss_tiebreak = -float(eval_stats["loss"]) if eval_stats is not None else float("-inf")
    return (
        int(probe_has_play_signal(probe_stats)),
        int(probe_has_real_signal(probe_stats)),
        float(probe_stats.get("mean_note_coverage", 0.0)),
        float(probe_stats.get("mean_completion", 0.0)),
        float(probe_stats.get("mean_clear_rate", 0.0)),
        float(probe_stats.get("mean_accuracy", 0.0)),
        float(probe_stats.get("mean_resolved_accuracy", probe_stats.get("mean_engaged_accuracy", 0.0))),
        float(probe_stats.get("mean_engaged_accuracy", 0.0)),
        float(probe_stats.get("mean_resolved_coverage", 0.0)),
        float(probe_stats.get("mean_cut", 0.0)),
        loss_tiebreak,
    )


def _relevant_saber_tip(left_tip, right_tip, note_type):
    return torch.where((note_type == 0).unsqueeze(1), left_tip, right_tip)


def _note_focus_weights(next_note_time, active_mask, decay):
    return active_mask.float() * torch.exp(-next_note_time / decay)


def _weighted_metric(per_sample, weights):
    denom = weights.sum().clamp(min=1.0)
    return float((per_sample * weights).sum().item() / denom.item())


def _weighted_loss(per_sample, weights):
    denom = weights.sum().clamp(min=1.0)
    return (per_sample * weights).sum() / denom


def note_guidance_loss(pred, state):
    next_note_time = state[:, NOTE_TIME_INDEX].float().clamp(min=0.0, max=4.0)
    note_lane = state[:, NOTE_LANE_INDEX].float()
    note_layer = state[:, NOTE_LAYER_INDEX].float()
    note_type = state[:, NOTE_TYPE_INDEX].long()

    active_mask = (next_note_time <= IMMINENT_NOTE_BEAT_WINDOW) & ((note_type == 0) | (note_type == 1))
    if not bool(active_mask.any()):
        zero = torch.zeros((), device=pred.device, dtype=pred.dtype)
        return zero, 0.0

    target_xy = torch.stack([
        NOTE_X_OFFSET + note_lane * NOTE_X_SPACING,
        NOTE_Y_OFFSET + note_layer * NOTE_Y_SPACING,
    ], dim=1).to(device=pred.device, dtype=pred.dtype)

    left_tip, right_tip = predicted_saber_tips(pred)
    pred_xy = _relevant_saber_tip(left_tip[:, 0:2], right_tip[:, 0:2], note_type)

    per_sample = nn.functional.smooth_l1_loss(pred_xy, target_xy, reduction="none").mean(dim=1)
    near_weight = _note_focus_weights(next_note_time, active_mask, decay=0.18)
    metric = _weighted_metric(per_sample, near_weight)
    return _weighted_loss(per_sample, near_weight), metric


def saber_tip_pose_loss(pred, target, state):
    next_note_time = state[:, NOTE_TIME_INDEX].float().clamp(min=0.0, max=4.0)
    note_type = state[:, NOTE_TYPE_INDEX].long()
    active_mask = (next_note_time <= IMMINENT_NOTE_BEAT_WINDOW) & ((note_type == 0) | (note_type == 1))
    if not bool(active_mask.any()):
        zero = torch.zeros((), device=pred.device, dtype=pred.dtype)
        return zero, 0.0

    pred_left_tip, pred_right_tip = predicted_saber_tips(pred)
    tgt_left_tip, tgt_right_tip = predicted_saber_tips(target)
    pred_tip = _relevant_saber_tip(pred_left_tip, pred_right_tip, note_type)
    tgt_tip = _relevant_saber_tip(tgt_left_tip, tgt_right_tip, note_type)

    per_sample = nn.functional.smooth_l1_loss(pred_tip, tgt_tip, reduction="none").mean(dim=1)
    near_weight = _note_focus_weights(next_note_time, active_mask, decay=0.20)
    metric = _weighted_metric(per_sample, near_weight)
    return _weighted_loss(per_sample, near_weight), metric


def saber_tip_motion_loss(pred, target, state):
    next_note_time = state[:, NOTE_TIME_INDEX].float().clamp(min=0.0, max=4.0)
    note_type = state[:, NOTE_TYPE_INDEX].long()
    active_mask = (next_note_time <= IMMINENT_NOTE_BEAT_WINDOW) & ((note_type == 0) | (note_type == 1))
    if not bool(active_mask.any()):
        zero = torch.zeros((), device=pred.device, dtype=pred.dtype)
        return zero, 0.0

    current_pose = state[:, CURRENT_POSE_START:CURRENT_POSE_END]
    cur_left_tip, cur_right_tip = predicted_saber_tips(current_pose)
    pred_left_tip, pred_right_tip = predicted_saber_tips(pred)
    tgt_left_tip, tgt_right_tip = predicted_saber_tips(target)

    cur_tip = _relevant_saber_tip(cur_left_tip, cur_right_tip, note_type)
    pred_tip = _relevant_saber_tip(pred_left_tip, pred_right_tip, note_type)
    tgt_tip = _relevant_saber_tip(tgt_left_tip, tgt_right_tip, note_type)

    pred_motion = pred_tip - cur_tip
    tgt_motion = tgt_tip - cur_tip
    per_sample = nn.functional.smooth_l1_loss(pred_motion, tgt_motion, reduction="none").mean(dim=1)
    near_weight = _note_focus_weights(next_note_time, active_mask, decay=0.18)
    metric = _weighted_metric(per_sample, near_weight)
    return _weighted_loss(per_sample, near_weight), metric


def saber_tip_direction_loss(pred, state):
    next_note_time = state[:, NOTE_TIME_INDEX].float().clamp(min=0.0, max=4.0)
    note_type = state[:, NOTE_TYPE_INDEX].long()
    cut_dir = state[:, NOTE_CUT_DX_INDEX:NOTE_CUT_DY_INDEX + 1].float()
    cut_dir_norm = torch.norm(cut_dir, dim=1, keepdim=True)
    active_mask = (
        (next_note_time <= IMMINENT_NOTE_BEAT_WINDOW)
        & ((note_type == 0) | (note_type == 1))
        & (cut_dir_norm.squeeze(1) > 0.10)
    )
    if not bool(active_mask.any()):
        zero = torch.zeros((), device=pred.device, dtype=pred.dtype)
        return zero, 0.0

    current_pose = state[:, CURRENT_POSE_START:CURRENT_POSE_END]
    cur_left_tip, cur_right_tip = predicted_saber_tips(current_pose)
    pred_left_tip, pred_right_tip = predicted_saber_tips(pred)

    cur_tip = _relevant_saber_tip(cur_left_tip, cur_right_tip, note_type)
    pred_tip = _relevant_saber_tip(pred_left_tip, pred_right_tip, note_type)
    pred_motion_xy = pred_tip[:, 0:2] - cur_tip[:, 0:2]
    pred_motion_norm = torch.norm(pred_motion_xy, dim=1, keepdim=True).clamp(min=1e-6)
    desired_dir = cut_dir.to(device=pred.device, dtype=pred.dtype) / cut_dir_norm.to(
        device=pred.device,
        dtype=pred.dtype,
    ).clamp(min=1e-6)

    direction_cos = (pred_motion_xy / pred_motion_norm * desired_dir).sum(dim=1).clamp(-1.0, 1.0)
    per_sample = (1.0 - direction_cos).clamp(min=0.0)
    near_weight = _note_focus_weights(next_note_time, active_mask, decay=0.18)
    metric = _weighted_metric(per_sample, near_weight)
    return _weighted_loss(per_sample, near_weight), metric


def augment_bc_inputs(batch_x):
    augmented = batch_x.clone()

    for hist_slot in range(STATE_HISTORY_FRAMES):
        if hist_slot == 0:
            start = CURRENT_POSE_START
        else:
            start = CURRENT_POSE_START + hist_slot * STATE_FRAME_DIM
        pose_start = start
        vel_start = start + POSE_DIM

        pos_scale = 0.012 + hist_slot * 0.004
        vel_scale = 0.18 + hist_slot * 0.05

        augmented[:, pose_start:pose_start + 3] += torch.randn_like(augmented[:, pose_start:pose_start + 3]) * (pos_scale * 0.45)
        augmented[:, pose_start + 7:pose_start + 10] += torch.randn_like(augmented[:, pose_start + 7:pose_start + 10]) * pos_scale
        augmented[:, pose_start + 14:pose_start + 17] += torch.randn_like(augmented[:, pose_start + 14:pose_start + 17]) * pos_scale
        augmented[:, vel_start:vel_start + 3] += torch.randn_like(augmented[:, vel_start:vel_start + 3]) * (vel_scale * 0.40)
        augmented[:, vel_start + 7:vel_start + 10] += torch.randn_like(augmented[:, vel_start + 7:vel_start + 10]) * vel_scale
        augmented[:, vel_start + 14:vel_start + 17] += torch.randn_like(augmented[:, vel_start + 14:vel_start + 17]) * vel_scale

    return augmented


def bc_pose_loss(pred, target, state):
    pos_idx = torch.tensor([0, 1, 2, 7, 8, 9, 14, 15, 16], device=pred.device)
    current_pose = state[:, CURRENT_POSE_START:CURRENT_POSE_END]

    pred_pos = pred.index_select(1, pos_idx)
    tgt_pos = target.index_select(1, pos_idx)
    cur_pos = current_pose.index_select(1, pos_idx)
    pos_loss = nn.functional.smooth_l1_loss(pred_pos, tgt_pos, reduction="none").mean(dim=1)
    motion_loss = nn.functional.smooth_l1_loss(
        pred_pos - cur_pos,
        tgt_pos - cur_pos,
        reduction="none",
    ).mean(dim=1)

    rot_loss_h = quaternion_alignment_loss(pred[:, 3:7], target[:, 3:7])
    rot_loss_l = quaternion_alignment_loss(pred[:, 10:14], target[:, 10:14])
    rot_loss_r = quaternion_alignment_loss(pred[:, 17:21], target[:, 17:21])
    rot_loss = 0.20 * rot_loss_h + 0.40 * rot_loss_l + 0.40 * rot_loss_r
    note_loss, note_metric = note_guidance_loss(pred, state)
    tip_loss, tip_metric = saber_tip_pose_loss(pred, target, state)
    swing_loss, swing_metric = saber_tip_motion_loss(pred, target, state)
    direction_loss, direction_metric = saber_tip_direction_loss(pred, state)

    weights = sample_weights_from_state(state)
    loss_weights = ACTIVE_BC_LOSS_WEIGHTS
    total = (
        (
            loss_weights["pos"] * pos_loss
            + loss_weights["motion"] * motion_loss
            + loss_weights["rot"] * rot_loss
        ) * weights
        + loss_weights["tip"] * tip_loss
        + loss_weights["swing"] * swing_loss
        + loss_weights["note"] * note_loss
        + loss_weights["direction"] * direction_loss
    ).mean()
    metrics = {
        "pos": pos_loss.mean().item(),
        "motion": motion_loss.mean().item(),
        "rot": rot_loss.mean().item(),
        "tip": tip_metric,
        "swing": swing_metric,
        "note": note_metric,
        "dir": direction_metric,
        "w": weights.mean().item(),
    }
    return total, metrics


def _load_tensor(path):
    return torch.load(path, weights_only=True)


def _pin_if_needed(tensor, pin_memory):
    return tensor.pin_memory() if pin_memory else tensor


def _clamp_state_time_features_(batch_x):
    if batch_x is None or batch_x.numel() == 0 or batch_x.shape[-1] < INPUT_DIM:
        return batch_x

    batch_x[:, 0:NOTES_DIM:NOTE_FEATURES].clamp_(SIM_NOTE_TIME_MIN_BEATS, SIM_NOTE_TIME_MAX_BEATS)
    obs_base = NOTES_DIM
    batch_x[:, obs_base:obs_base + OBSTACLES_DIM:OBSTACLE_FEATURES].clamp_(
        SIM_OBSTACLE_TIME_MIN_BEATS,
        SIM_OBSTACLE_TIME_MAX_BEATS,
    )
    return batch_x


def _take_packed_batch(pending, sample_count, pin_memory=False):
    if sample_count <= 0 or not pending:
        return None, None

    remaining = int(sample_count)
    first_x, first_y, first_offset = pending[0]
    first_available = first_x.shape[0] - first_offset
    if len(pending) == 1 and first_available >= remaining and not pin_memory:
        batch_slice = slice(first_offset, first_offset + remaining)
        batch_x = first_x[batch_slice]
        batch_y = first_y[batch_slice]
        new_offset = first_offset + remaining
        if new_offset >= first_x.shape[0]:
            pending.popleft()
        else:
            pending[0] = (first_x, first_y, new_offset)
        return batch_x, batch_y

    batch_x = torch.empty(
        (remaining, first_x.shape[1]),
        dtype=first_x.dtype,
        pin_memory=pin_memory,
    )
    batch_y = torch.empty(
        (remaining, first_y.shape[1]),
        dtype=first_y.dtype,
        pin_memory=pin_memory,
    )
    write_offset = 0

    while remaining > 0 and pending:
        shard_x, shard_y, offset = pending[0]
        available = shard_x.shape[0] - offset
        take = min(remaining, available)
        shard_slice = slice(offset, offset + take)
        batch_slice = slice(write_offset, write_offset + take)
        batch_x[batch_slice].copy_(shard_x[shard_slice])
        batch_y[batch_slice].copy_(shard_y[shard_slice])
        write_offset += take
        remaining -= take

        new_offset = offset + take
        if new_offset >= shard_x.shape[0]:
            pending.popleft()
        else:
            pending[0] = (shard_x, shard_y, new_offset)

    if write_offset <= 0:
        return None, None
    if write_offset < batch_x.shape[0]:
        batch_x = batch_x[:write_offset]
        batch_y = batch_y[:write_offset]
    return batch_x, batch_y


def _load_prepared_shard(record, shuffle):
    label = shard_record_label(record, SHARD_ROOT)
    load_started = time.perf_counter()
    shard_x, shard_y = load_shard_pair(record, SHARD_ROOT, device="cpu")
    load_duration = time.perf_counter() - load_started
    shuffle_duration = 0.0

    if shard_x.numel() == 0 or shard_y.numel() == 0:
        return label, shard_x, shard_y, load_duration, shuffle_duration
    if shard_x.ndim != 2 or shard_x.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"Shard feature width mismatch in {label}: expected {INPUT_DIM}, got {tuple(shard_x.shape)}. "
            "Rebuild BC shards with `python -m cybernoodles.data.dataset_builder`."
        )
    if shard_y.ndim != 2 or shard_y.shape[1] != POSE_DIM:
        raise RuntimeError(
            f"Shard target width mismatch in {label}: expected {POSE_DIM}, got {tuple(shard_y.shape)}. "
            "Rebuild BC shards with `python -m cybernoodles.data.dataset_builder`."
        )
    _clamp_state_time_features_(shard_x)

    if shuffle:
        shuffle_started = time.perf_counter()
        indices = torch.randperm(shard_x.shape[0])
        shard_x = shard_x.index_select(0, indices)
        shard_y = shard_y.index_select(0, indices)
        shuffle_duration = time.perf_counter() - shuffle_started

    return label, shard_x, shard_y, load_duration, shuffle_duration


def iter_shard_batches(
    records,
    batch_size,
    shuffle=True,
    pin_memory=False,
    prefetch_shards=1,
    profile=None,
):
    order = list(records)
    if shuffle:
        random.shuffle(order)

    pending = deque()
    pending_samples = 0
    max_workers = max(1, int(prefetch_shards))

    def _yield_ready_batches():
        nonlocal pending_samples
        while pending_samples >= batch_size:
            pack_started = time.perf_counter()
            batch_x, batch_y = _take_packed_batch(pending, batch_size, pin_memory=pin_memory)
            pack_duration = time.perf_counter() - pack_started
            if batch_x is None or batch_y is None:
                return
            pending_samples -= batch_x.shape[0]
            if profile is not None:
                profile.add_batch_pack(pack_duration)
            yield batch_x, batch_y

    def _consume_loaded_shard(result):
        nonlocal pending_samples
        _, shard_x, shard_y, load_duration, shuffle_duration = result
        if profile is not None:
            shard_samples = 0 if shard_x is None or shard_x.numel() == 0 else shard_x.shape[0]
            profile.add_shard(shard_samples, load_duration, shuffle_duration)
        if shard_x is None or shard_y is None or shard_x.numel() == 0 or shard_y.numel() == 0:
            return
        pending.append((shard_x, shard_y, 0))
        pending_samples += shard_x.shape[0]

    if max_workers == 1:
        for record in order:
            _consume_loaded_shard(_load_prepared_shard(record, shuffle))
            yield from _yield_ready_batches()
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            record_iter = iter(order)
            future_queue = deque()

            def _submit_next():
                try:
                    record = next(record_iter)
                except StopIteration:
                    return False
                future_queue.append(executor.submit(_load_prepared_shard, record, shuffle))
                return True

            while len(future_queue) < max_workers and _submit_next():
                pass

            while future_queue:
                future = future_queue.popleft()
                _consume_loaded_shard(future.result())
                _submit_next()
                yield from _yield_ready_batches()

    if pending_samples > 0:
        pack_started = time.perf_counter()
        batch_x, batch_y = _take_packed_batch(pending, pending_samples, pin_memory=pin_memory)
        pack_duration = time.perf_counter() - pack_started
        if batch_x is not None and batch_y is not None:
            if profile is not None:
                profile.add_batch_pack(pack_duration)
            yield batch_x, batch_y


def iter_legacy_batches(batch_size, shuffle=True, pin_memory=False, profile=None):
    batch_x = _load_tensor(LEGACY_X_PATH)
    batch_y = _load_tensor(LEGACY_Y_PATH)
    if batch_x.ndim != 2 or batch_x.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"Legacy BC feature tensor width mismatch: expected {INPUT_DIM}, got {tuple(batch_x.shape)}. "
            "Rebuild BC shards with `python -m cybernoodles.data.dataset_builder`."
        )
    if batch_y.ndim != 2 or batch_y.shape[1] != POSE_DIM:
        raise RuntimeError(
            f"Legacy BC target tensor width mismatch: expected {POSE_DIM}, got {tuple(batch_y.shape)}."
        )
    _clamp_state_time_features_(batch_x)
    indices = torch.randperm(batch_x.shape[0]) if shuffle else torch.arange(batch_x.shape[0])
    for start in range(0, batch_x.shape[0], batch_size):
        excerpt = indices[start:start + batch_size]
        legacy_batch_x = _pin_if_needed(batch_x[excerpt], pin_memory)
        legacy_batch_y = _pin_if_needed(batch_y[excerpt], pin_memory)
        if profile is not None:
            profile.add_batch_pack(0.0)
        yield legacy_batch_x, legacy_batch_y


def _iter_device_batches_sync(cpu_iterator, device, non_blocking=True, profile=None):
    for cpu_x, cpu_y in cpu_iterator:
        submit_started = time.perf_counter()
        if profile is not None and device.type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            batch_x = cpu_x.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
            batch_y = cpu_y.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
            end_event.record()
            profile.add_h2d_submit(time.perf_counter() - submit_started)
            profile.add_h2d_event(start_event, end_event)
        else:
            batch_x = cpu_x.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
            batch_y = cpu_y.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
        if profile is not None:
            profile.add_consumed_batch(batch_x.shape[0])
        yield batch_x, batch_y


@contextmanager
def iter_device_batches(
    records,
    batch_size,
    device,
    use_shards,
    shuffle=True,
    pin_memory=False,
    loader_prefetch_batches=DEFAULT_LOADER_PREFETCH_BATCHES,
    loader_prefetch_shards=DEFAULT_LOADER_PREFETCH_SHARDS,
    profile=None,
):
    async_enabled = int(loader_prefetch_batches) > 0

    if use_shards:
        source_factory = lambda: iter_shard_batches(
            records,
            batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            prefetch_shards=loader_prefetch_shards,
            profile=profile,
        )
    else:
        source_factory = lambda: iter_legacy_batches(
            batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            profile=profile,
        )

    cpu_context = (
        ThreadedPrefetchIterator(
            source_factory,
            max_prefetch=loader_prefetch_batches,
            profile=profile,
        )
        if async_enabled else
        closing(source_factory())
    )

    with cpu_context as cpu_iterator:
        if async_enabled and device.type == 'cuda':
            with CudaBatchPrefetcher(
                cpu_iterator,
                device=device,
                dtype=torch.float32,
                non_blocking=pin_memory,
                profile=profile,
            ) as gpu_iterator:
                yield gpu_iterator
        else:
            yield _iter_device_batches_sync(
                cpu_iterator,
                device=device,
                non_blocking=pin_memory and device.type == 'cuda',
                profile=profile,
            )


def print_loader_profile(label, profile, wall_time_s, device):
    if profile is None or not profile.enabled:
        return
    if device.type == 'cuda':
        profile.finalize_cuda(device)
    snap = profile.snapshot()
    wall = max(1e-9, float(wall_time_s))
    print(
        f"Loader profile [{label}] | wall {wall_time_s:.3f}s | "
        f"{snap.batch_count / wall:.2f} batches/s | {snap.sample_count / wall:,.0f} samples/s"
    )
    print(
        f"  shards {snap.shard_count} | batches {snap.batch_count} | "
        f"queue wait {snap.queue_wait_s:.3f}s | shard load {snap.shard_load_s:.3f}s | "
        f"shuffle {snap.shard_shuffle_s:.3f}s | pack {snap.batch_pack_s:.3f}s | "
        f"h2d copy {snap.h2d_copy_s:.3f}s (submit {snap.h2d_submit_s:.3f}s) | "
        f"step {snap.compute_s:.3f}s"
    )


def evaluate_model(
    model,
    records,
    device,
    batch_size,
    max_batches=None,
    use_shards=True,
    loader_prefetch_batches=DEFAULT_LOADER_PREFETCH_BATCHES,
    loader_prefetch_shards=DEFAULT_LOADER_PREFETCH_SHARDS,
    profile=None,
):
    model.eval()
    total_loss = 0.0
    total_pos = 0.0
    total_motion = 0.0
    total_rot = 0.0
    total_tip = 0.0
    total_swing = 0.0
    total_note = 0.0
    total_dir = 0.0
    total_batches = 0

    with iter_device_batches(
        records,
        batch_size,
        device,
        use_shards=use_shards,
        shuffle=False,
        pin_memory=(device.type == 'cuda'),
        loader_prefetch_batches=loader_prefetch_batches,
        loader_prefetch_shards=loader_prefetch_shards,
        profile=profile,
    ) as iterator:
        with torch.no_grad():
            for batch_x, batch_y in iterator:
                if profile is not None and device.type == 'cuda':
                    step_start = torch.cuda.Event(enable_timing=True)
                    step_end = torch.cuda.Event(enable_timing=True)
                    step_start.record()
                with torch.amp.autocast('cuda'):
                    mean, _, _ = model(batch_x)
                    loss, metrics = bc_pose_loss(mean, batch_y, batch_x)
                if profile is not None and device.type == 'cuda':
                    step_end.record()
                    profile.add_compute_event(step_start, step_end)

                total_loss += loss.item()
                total_pos += metrics["pos"]
                total_motion += metrics["motion"]
                total_rot += metrics["rot"]
                total_tip += metrics["tip"]
                total_swing += metrics["swing"]
                total_note += metrics["note"]
                total_dir += metrics["dir"]
                total_batches += 1

                if max_batches is not None and total_batches >= max_batches:
                    break

    if total_batches == 0:
        return None

    return {
        "loss": total_loss / total_batches,
        "pos": total_pos / total_batches,
        "motion": total_motion / total_batches,
        "rot": total_rot / total_batches,
        "tip": total_tip / total_batches,
        "swing": total_swing / total_batches,
        "note": total_note / total_batches,
        "dir": total_dir / total_batches,
    }


def train_bc(
    epochs=8,
    batch_size=None,
    lr=8e-4,
    patience=3,
    train_shards_limit=None,
    val_shards_limit=None,
    val_every=1,
    val_batches_limit=None,
    init_from=None,
    sim_probe=False,
    sim_probe_maps=1,
    sim_probe_num_envs=1,
    sim_probe_profile="strict",
    sim_probe_suite="starter",
    sim_probe_verbose=False,
    profile_loader=False,
    loader_prefetch_batches=DEFAULT_LOADER_PREFETCH_BATCHES,
    loader_prefetch_shards=DEFAULT_LOADER_PREFETCH_SHARDS,
    loss_preset="cut",
):
    print(f"\033[96m{'='*60}")
    print("  CyberNoodles 6.0: Behavioral Cloning (Sharded)")
    print(f"{'='*60}\033[0m")
    loss_weights = set_bc_loss_preset(loss_preset)
    loss_preset_key = str(loss_preset or "cut").strip().lower()
    print(
        f"BC loss preset: {loss_preset_key} "
        f"(pos {loss_weights['pos']:.2f}, motion {loss_weights['motion']:.2f}, "
        f"rot {loss_weights['rot']:.2f}, tip {loss_weights['tip']:.2f}, "
        f"swing {loss_weights['swing']:.2f}, note {loss_weights['note']:.2f}, "
        f"dir {loss_weights['direction']:.2f})"
    )

    manifest = load_manifest()
    if manifest is None and os.path.exists(MANIFEST_PATH):
        return
    use_shards = manifest is not None and len(manifest.get("shards", [])) > 0
    if not use_shards and (not os.path.exists(LEGACY_X_PATH) or not os.path.exists(LEGACY_Y_PATH)):
        print("\033[91mNo BC dataset found.\033[0m")
        print("\033[93mRun `python -m cybernoodles.data.dataset_builder` first.\033[0m")
        return

    if not torch.cuda.is_available():
        print("\033[91mCUDA not available — this script requires a CUDA GPU.\033[0m")
        return

    device = torch.device('cuda')
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    auto_batch = batch_size is None
    if batch_size is None:
        batch_size = suggest_batch_size(device)
    batch_size = max(1024, int(batch_size))
    loader_prefetch_batches = max(0, int(loader_prefetch_batches))
    loader_prefetch_shards = max(1, int(loader_prefetch_shards))

    if use_shards:
        train_records = split_records(manifest, "train")
        val_records = split_records(manifest, "val")
        full_train_shards = len(train_records)
        full_val_shards = len(val_records)
        train_records = take_record_subset(train_records, train_shards_limit)
        val_records = take_record_subset(val_records, val_shards_limit)
        if not train_records:
            print("\033[91mNo train BC shards found in manifest.\033[0m")
            print("\033[93mRun `python -m cybernoodles.data.dataset_builder` first.\033[0m")
            return
        train_preflight = preflight_shard_records(train_records, "train")
        val_preflight = (
            preflight_shard_records(val_records, "val")
            if val_records else
            {"records": 0, "samples": 0}
        )
        counts = manifest.get("counts", {})
        print(
            f"Dataset: {counts.get('train_samples', 0):,} train / "
            f"{counts.get('val_samples', 0):,} val samples across "
            f"{full_train_shards} / {full_val_shards} replay shards."
        )
        print(
            f"Preflight: {train_preflight['records']} train shard(s), "
            f"{val_preflight['records']} val shard(s), "
            f"{train_preflight['samples'] + val_preflight['samples']:,} verified samples."
        )
        packed_train_batches = math.ceil(max(1, counts.get('train_samples', 0)) / batch_size)
        packed_val_batches = math.ceil(max(1, counts.get('val_samples', 0)) / batch_size)
        batch_label = "auto" if auto_batch else "manual"
        print(
            f"Batch size: {batch_size:,} ({batch_label}) | "
            f"packed batches: ~{packed_train_batches} train / ~{packed_val_batches} val"
        )
        if len(train_records) != full_train_shards or len(val_records) != full_val_shards:
            print(
                f"Subset mode: using {len(train_records)} train / {len(val_records)} val shards "
                f"for faster BC iteration."
            )
        if val_batches_limit:
            print(f"Validation cap: at most {val_batches_limit} batch(es) per eval pass.")
        print(
            f"Loader: shard prefetch {loader_prefetch_shards} | "
            f"batch prefetch {loader_prefetch_batches} | "
            f"profile {'on' if profile_loader else 'off'}"
        )
    else:
        train_records = []
        val_records = []
        print("Using legacy monolithic dataset tensors. Rebuild with `python -m cybernoodles.data.dataset_builder` for the RAM-safe path.")
        print(
            f"Loader: batch prefetch {loader_prefetch_batches} | "
            f"profile {'on' if profile_loader else 'off'}"
        )

    model = ActorCritic().to(device)
    warmstart_path = load_bc_warmstart(model, init_from)
    actor_params = (
        list(model.actor_features.parameters()) +
        list(model.actor_mean.parameters()) +
        [model.actor_log_std]
    )
    optimizer = make_adam(actor_params, lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')

    best_score = float("inf")
    best_probe_key = None
    stale_epochs = 0
    baseline_eval = None
    baseline_suffix = ""
    if val_records:
        baseline_profile = LoaderProfile(enabled=profile_loader)
        baseline_started = time.perf_counter()
        baseline_eval = evaluate_model(
            model,
            val_records,
            device,
            batch_size,
            max_batches=val_batches_limit,
            use_shards=use_shards,
            loader_prefetch_batches=loader_prefetch_batches,
            loader_prefetch_shards=loader_prefetch_shards,
            profile=baseline_profile,
        )
        baseline_elapsed = time.perf_counter() - baseline_started
        if baseline_eval is not None:
            best_score = float(baseline_eval["loss"])
            if warmstart_path:
                raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                save_policy_actor_checkpoint(
                    BC_MODEL_PATH,
                    build_rl_bootstrap_state_dict(raw_model),
                    checkpoint_kind="bc_best",
                )
            print(f"Baseline val loss: {best_score:.5f}")
            if profile_loader:
                print_loader_profile("val-baseline", baseline_profile, baseline_elapsed, device)

    if sim_probe:
        baseline_probe_stats = run_bc_sim_probe(
            model,
            device,
            max_maps=sim_probe_maps,
            num_envs=sim_probe_num_envs,
            suite=sim_probe_suite,
            profile=sim_probe_profile,
            verbose=sim_probe_verbose,
        )
        if baseline_probe_stats is not None:
            best_probe_key = probe_sort_key(baseline_probe_stats, baseline_eval)
            if warmstart_path:
                raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                save_policy_actor_checkpoint(
                    BC_MODEL_PATH,
                    build_rl_bootstrap_state_dict(raw_model),
                    checkpoint_kind="bc_best",
                )
            baseline_suffix = (
                f"Baseline probe: cover {baseline_probe_stats['mean_note_coverage']:.2f} "
                f"task {baseline_probe_stats['mean_accuracy']:.2f}% "
                f"engaged {baseline_probe_stats['mean_engaged_accuracy']:.2f}% "
                f"comp {baseline_probe_stats['mean_completion']:.2f} "
                f"clear {baseline_probe_stats['mean_clear_rate']:.2f}"
            )
            print(baseline_suffix)

    for epoch in range(epochs):
        model.train()
        epoch_started = time.time()
        train_started = time.time()
        total_loss = 0.0
        total_pos = 0.0
        total_motion = 0.0
        total_rot = 0.0
        total_tip = 0.0
        total_swing = 0.0
        total_note = 0.0
        total_dir = 0.0
        total_batches = 0
        train_profile = LoaderProfile(enabled=profile_loader)

        with iter_device_batches(
            train_records,
            batch_size,
            device,
            use_shards=use_shards,
            shuffle=True,
            pin_memory=(device.type == 'cuda'),
            loader_prefetch_batches=loader_prefetch_batches,
            loader_prefetch_shards=loader_prefetch_shards,
            profile=train_profile,
        ) as iterator:
            for batch_x, batch_y in iterator:
                if profile_loader and device.type == 'cuda':
                    step_start = torch.cuda.Event(enable_timing=True)
                    step_end = torch.cuda.Event(enable_timing=True)
                    step_start.record()

                noisy_batch_x = augment_bc_inputs(batch_x)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    mean, _, _ = model(noisy_batch_x)
                    loss, metrics = bc_pose_loss(mean, batch_y, noisy_batch_x)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(actor_params, 1.0)
                scaler.step(optimizer)
                scaler.update()

                if profile_loader and device.type == 'cuda':
                    step_end.record()
                    train_profile.add_compute_event(step_start, step_end)

                total_loss += loss.item()
                total_pos += metrics["pos"]
                total_motion += metrics["motion"]
                total_rot += metrics["rot"]
                total_tip += metrics["tip"]
                total_swing += metrics["swing"]
                total_note += metrics["note"]
                total_dir += metrics["dir"]
                total_batches += 1

        scheduler.step()
        train_elapsed = time.time() - train_started
        train_loss = total_loss / max(1, total_batches)
        train_pos = total_pos / max(1, total_batches)
        train_motion = total_motion / max(1, total_batches)
        train_rot = total_rot / max(1, total_batches)
        train_tip = total_tip / max(1, total_batches)
        train_swing = total_swing / max(1, total_batches)
        train_note = total_note / max(1, total_batches)
        train_dir = total_dir / max(1, total_batches)

        should_eval = bool(val_records) and (val_every <= 1 or ((epoch + 1) % val_every == 0) or epoch == (epochs - 1))
        eval_profile = LoaderProfile(enabled=profile_loader)
        eval_started = time.time()
        eval_stats = (
            evaluate_model(
                model,
                val_records,
                device,
                batch_size,
                max_batches=val_batches_limit,
                use_shards=use_shards,
                loader_prefetch_batches=loader_prefetch_batches,
                loader_prefetch_shards=loader_prefetch_shards,
                profile=eval_profile,
            )
            if should_eval else
            None
        )
        eval_elapsed = (time.time() - eval_started) if should_eval else 0.0
        probe_stats = None
        if sim_probe and (should_eval or not val_records):
            probe_stats = run_bc_sim_probe(
                model,
                device,
                max_maps=sim_probe_maps,
                num_envs=sim_probe_num_envs,
                suite=sim_probe_suite,
                profile=sim_probe_profile,
                verbose=sim_probe_verbose,
            )

        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        rl_bootstrap_state = build_rl_bootstrap_state_dict(raw_model)
        save_policy_actor_checkpoint(
            BC_LAST_MODEL_PATH,
            rl_bootstrap_state,
            checkpoint_kind="bc_last",
        )

        probe_key = None
        if probe_stats is not None:
            probe_key = probe_sort_key(probe_stats, eval_stats)

        use_probe_for_selection = probe_key is not None and sim_probe
        if use_probe_for_selection:
            if best_probe_key is None or probe_key > best_probe_key:
                best_probe_key = probe_key
                stale_epochs = 0
                save_policy_actor_checkpoint(
                    BC_MODEL_PATH,
                    rl_bootstrap_state,
                    checkpoint_kind="bc_best",
                )
                best_tag = "\033[92mBEST\033[0m"
            else:
                stale_epochs += 1
                best_tag = "    "
        else:
            score = eval_stats["loss"] if eval_stats is not None else (train_loss if not val_records else None)
            if score is not None and score < best_score:
                best_score = score
                stale_epochs = 0
                save_policy_actor_checkpoint(
                    BC_MODEL_PATH,
                    rl_bootstrap_state,
                    checkpoint_kind="bc_best",
                )
                best_tag = "\033[92mBEST\033[0m"
            elif score is not None:
                stale_epochs += 1
                best_tag = "    "
            else:
                best_tag = f"{DIM}EVAL-SKIP{RST}"

        elapsed = time.time() - epoch_started
        if profile_loader:
            print_loader_profile(f"train epoch {epoch+1}", train_profile, train_elapsed, device)
            if should_eval and eval_stats is not None:
                print_loader_profile(f"val epoch {epoch+1}", eval_profile, eval_elapsed, device)
        probe_suffix = ""
        if probe_stats is not None:
            probe_suffix = (
                f" | probe cover {probe_stats['mean_note_coverage']:.2f} "
                f"task {probe_stats['mean_accuracy']:.2f}% "
                f"engaged {probe_stats['mean_engaged_accuracy']:.2f}% "
                f"comp {probe_stats['mean_completion']:.2f} "
                f"clear {probe_stats['mean_clear_rate']:.2f}"
            )
        if eval_stats is not None:
            print(
                f"Epoch {epoch+1:03d}/{epochs} | "
                f"train {train_loss:.5f} (pos {train_pos:.5f}, motion {train_motion:.5f}, rot {train_rot:.5f}, tip {train_tip:.5f}, swing {train_swing:.5f}, dir {train_dir:.5f}, note {train_note:.5f}) | "
                f"val {eval_stats['loss']:.5f} (pos {eval_stats['pos']:.5f}, motion {eval_stats['motion']:.5f}, rot {eval_stats['rot']:.5f}, tip {eval_stats['tip']:.5f}, swing {eval_stats['swing']:.5f}, dir {eval_stats['dir']:.5f}, note {eval_stats['note']:.5f}) | "
                f"{elapsed:.1f}s {best_tag}{probe_suffix}"
            )
        elif val_records:
            print(
                f"Epoch {epoch+1:03d}/{epochs} | "
                f"train {train_loss:.5f} (pos {train_pos:.5f}, motion {train_motion:.5f}, rot {train_rot:.5f}, tip {train_tip:.5f}, swing {train_swing:.5f}, dir {train_dir:.5f}, note {train_note:.5f}) | "
                f"{DIM}val skipped (runs every {val_every} epoch(s)){RST} | "
                f"{elapsed:.1f}s {best_tag}{probe_suffix}"
            )
        else:
            print(
                f"Epoch {epoch+1:03d}/{epochs} | "
                f"train {train_loss:.5f} (pos {train_pos:.5f}, motion {train_motion:.5f}, rot {train_rot:.5f}, tip {train_tip:.5f}, swing {train_swing:.5f}, dir {train_dir:.5f}, note {train_note:.5f}) | "
                f"{elapsed:.1f}s {best_tag}{probe_suffix}"
            )

        if stale_epochs >= patience:
            print(f"\033[93mEarly stopping after {patience} stale epochs.\033[0m")
            break

    print(f"\n\033[92mBC training complete.\033[0m Best model saved to {BC_MODEL_PATH}.")
    print(f"Latest checkpoint saved to {BC_LAST_MODEL_PATH}.")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Train the CyberNoodles BC warmstart.")
        parser.add_argument("legacy_epochs", nargs="?", type=int, help="Backward-compatible positional epochs override.")
        parser.add_argument("--epochs", type=int, default=None, help="Number of full passes over the selected BC shards.")
        parser.add_argument("--batch-size", type=int, default=None, help="BC minibatch size. Defaults to an automatic GPU-aware choice.")
        parser.add_argument("--lr", type=float, default=8e-4, help="Learning rate.")
        parser.add_argument("--patience", type=int, default=3, help="Early stopping patience in stale evals.")
        parser.add_argument("--train-shards", type=int, default=None, help="Use only this many training shards for faster iteration.")
        parser.add_argument("--val-shards", type=int, default=None, help="Use only this many validation shards for faster iteration.")
        parser.add_argument("--val-every", type=int, default=1, help="Run validation every N epochs.")
        parser.add_argument("--val-batches", type=int, default=None, help="Cap each validation pass to this many batches.")
        parser.add_argument("--init-from", type=str, default=None, help="Warm-start from an existing checkpoint path, or `best` / `last`.")
        parser.add_argument(
            "--loss-preset",
            choices=sorted(BC_LOSS_PRESETS),
            default="cut",
            help="BC objective weighting preset. `cut` prioritizes saber-tip swing geometry for strict play.",
        )
        parser.add_argument("--sim-probe", dest="sim_probe", action="store_true", help="Run a tiny simulator probe after each eval and select best checkpoints on probe metrics.")
        parser.add_argument("--no-sim-probe", dest="sim_probe", action="store_false", help="Disable simulator probes and fall back to shard-loss-only checkpoint selection.")
        parser.add_argument("--sim-probe-maps", type=int, default=DEFAULT_BC_SIM_PROBE_MAPS, help="How many eval maps to use for the simulator probe.")
        parser.add_argument("--sim-probe-num-envs", type=int, default=DEFAULT_BC_SIM_PROBE_ENVS, help="How many simulator envs to use for the probe.")
        parser.add_argument("--sim-probe-profile", type=str, default="strict", help="Simulator probe profile: strict, bc, or rehab.")
        parser.add_argument("--sim-probe-suite", type=str, default="starter", help="Curriculum suite for probe map selection.")
        parser.add_argument("--sim-probe-verbose", action="store_true", help="Print simulator probe progress.")
        parser.add_argument("--profile-loader", action="store_true", help="Print per-phase BC loader timing breakdowns.")
        parser.add_argument(
            "--loader-prefetch-batches",
            type=int,
            default=DEFAULT_LOADER_PREFETCH_BATCHES,
            help="How many ready CPU batches to keep queued ahead of the training loop. Set 0 to disable async batch prefetch.",
        )
        parser.add_argument(
            "--loader-prefetch-shards",
            type=int,
            default=DEFAULT_LOADER_PREFETCH_SHARDS,
            help="How many shard load/shuffle tasks to keep in flight inside the BC batch producer.",
        )
        parser.add_argument(
            "--quick",
            action="store_true",
            help="Fast sanity-check mode: fewer epochs, fewer shards, and cheaper validation.",
        )
        parser.set_defaults(sim_probe=True)
        args = parser.parse_args()

        epochs = args.epochs if args.epochs is not None else (args.legacy_epochs if args.legacy_epochs is not None else 8)
        train_shards = args.train_shards
        val_shards = args.val_shards
        val_every = max(1, int(args.val_every))
        val_batches = args.val_batches

        if args.quick:
            epochs = min(epochs, 3)
            train_shards = train_shards or 96
            val_shards = val_shards or 16
            val_every = max(val_every, 1)
            val_batches = val_batches or 24
            print("Quick BC mode enabled: reduced shard set and validation budget for faster iteration.")

        train_bc(
            epochs=epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            train_shards_limit=train_shards,
            val_shards_limit=val_shards,
            val_every=val_every,
            val_batches_limit=val_batches,
            init_from=args.init_from,
            sim_probe=args.sim_probe,
            sim_probe_maps=args.sim_probe_maps,
            sim_probe_num_envs=args.sim_probe_num_envs,
            sim_probe_profile=args.sim_probe_profile,
            sim_probe_suite=args.sim_probe_suite,
            sim_probe_verbose=args.sim_probe_verbose,
            profile_loader=args.profile_loader,
            loader_prefetch_batches=args.loader_prefetch_batches,
            loader_prefetch_shards=args.loader_prefetch_shards,
            loss_preset=args.loss_preset,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
