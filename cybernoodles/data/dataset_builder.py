import argparse
import concurrent.futures
import gc
import glob
import hashlib
import json
import os
import time
import zipfile
from collections import deque

import numpy as np
import torch
from bsor.Bsor import make_bsor

from cybernoodles.bsor_bridge import build_bc_dataset_via_rust, parse_dataset_view_via_rust
from cybernoodles.core.jump_timing import compute_spawn_ahead_beats
from cybernoodles.core.network import (
    INPUT_DIM,
    NOTE_FEATURES,
    NOTES_DIM,
    NUM_UPCOMING_OBSTACLES,
    NUM_UPCOMING_NOTES,
    OBSTACLE_FEATURES,
    POSE_DIM,
    STATE_HISTORY_OFFSETS,
    VELOCITY_DIM,
    encode_cut_direction,
)
from cybernoodles.core.pose_defaults import DEFAULT_TRACKED_POSE

DATA_DIR = "data"
REPLAYS_DIR = os.path.join(DATA_DIR, "replays")
MAPS_DIR = os.path.join(DATA_DIR, "maps")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")
SHARD_ROOT = os.path.join(OUTPUT_DIR, "bc_shards")
TRAIN_DIR = os.path.join(SHARD_ROOT, "train")
VAL_DIR = os.path.join(SHARD_ROOT, "val")
MANIFEST_PATH = os.path.join(SHARD_ROOT, "manifest.json")
SELECTED_SCORES_PATH = os.path.join(DATA_DIR, "selected_scores.json")

MANIFEST_VERSION = 16
MANIFEST_SEMANTIC_SCHEMA_ID = "bc-shard-semantics-v1"
NOTE_LOOKAHEAD_BEATS = 1.25
FOLLOWTHROUGH_BEATS = 0.35
BACKGROUND_FRAME_STRIDE = 6
VAL_FRACTION = 0.10
MIN_REPLAY_FRAMES = 16
MIN_SAMPLE_FRAMES = 8
# Keep the imitation target within the simulator's per-step clamp so BC
# learns motions the runtime controller can actually execute.
TARGET_POSE_HORIZON_FRAMES = 2
SIM_SAMPLE_HZ = 60.0
SIM_SAMPLE_DT = 1.0 / SIM_SAMPLE_HZ
# Mirror the simulator state ranges so BC trains on the same note/obstacle
# timing distribution it sees during closed-loop rollout.
SIM_NOTE_TIME_MIN_BEATS = -1.0
SIM_NOTE_TIME_MAX_BEATS = 4.0
NOTE_TIME_FEATURE_MIN_BEATS = -1.0
NOTE_TIME_FEATURE_MAX_BEATS = 8.0
SIM_OBSTACLE_TIME_MIN_BEATS = -1.0
SIM_OBSTACLE_TIME_MAX_BEATS = 6.0
NOTE_FEATURE_LAYOUT = "spawn_visible_contact_shifted_beat_time+physical_time+physical_z"
TRACK_Z_BASE = 0.9
FLOAT16_MAX = float(np.finfo(np.float16).max)
POSITION_ABS_MAX = 8.0
QUAT_COMPONENT_ABS_MAX = 2.0

SCORE_CLASS_NORMAL = 0.0
SCORE_CLASS_ARC_HEAD = 1.0
SCORE_CLASS_ARC_TAIL = 2.0
SCORE_CLASS_CHAIN_HEAD = 3.0
SCORE_CLASS_CHAIN_LINK = 4.0
ACTION_ABS_COMPONENT_LIMIT = 2.0
STATE_POSE_POSITION_ABS_LIMIT = ACTION_ABS_COMPONENT_LIMIT
FEATURE_STORAGE_DTYPE = "float16"
TARGET_STORAGE_DTYPE = "float32"
SHARD_STORAGE_DTYPE = f"features={FEATURE_STORAGE_DTYPE},targets={TARGET_STORAGE_DTYPE}"
TARGET_ACTION_DELTA_CLAMP = (
    0.08, 0.08, 0.08, 0.045, 0.045, 0.045, 0.045,
    0.12, 0.12, 0.12, 0.07, 0.07, 0.07, 0.07,
    0.12, 0.12, 0.12, 0.07, 0.07, 0.07, 0.07,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEFAULT_BUILD_WORKERS = max(1, min(8, (os.cpu_count() or 4) - 1))
DEFAULT_MANIFEST_SAVE_EVERY = 32
DEFAULT_MAX_PENDING_WRITES = 16
DEFAULT_GC_EVERY = 16
DEFAULT_STATUS_EVERY = 25

DEFAULT_POSE = np.asarray(DEFAULT_TRACKED_POSE, dtype=np.float32)
POSE_POSITION_SLICES = ((0, 3), (7, 10), (14, 17))
POSE_QUATERNION_SLICES = ((3, 7), (10, 14), (17, 21))
TARGET_ACTION_DELTA_CLAMP_ARRAY = np.asarray(TARGET_ACTION_DELTA_CLAMP, dtype=np.float32)


def normalize_token(value):
    return ''.join(ch.lower() for ch in str(value or '') if ch.isalnum())


def normalize_mode_name(mode_name):
    token = normalize_token(mode_name)
    if token in {"", "standard"}:
        return "standard"
    if token in {"360degree", "360"}:
        return "360degree"
    if token in {"90degree", "90"}:
        return "90degree"
    if token in {"onesaber", "onesaberstandard"}:
        return "onesaber"
    if token in {"noarrows"}:
        return "noarrows"
    return token


def normalize_difficulty_name(difficulty_name):
    token = normalize_token(difficulty_name)
    aliases = {
        "expertplus": "expertplus",
        "expert": "expert",
        "hard": "hard",
        "normal": "normal",
        "easy": "easy",
    }
    return aliases.get(token, token)


def manifest_semantic_metadata():
    return {
        "schema_id": MANIFEST_SEMANTIC_SCHEMA_ID,
        "sample_hz": SIM_SAMPLE_HZ,
        "state_history_offsets": list(STATE_HISTORY_OFFSETS),
        "note_feature_layout": NOTE_FEATURE_LAYOUT,
        "note_lookahead_beats": NOTE_LOOKAHEAD_BEATS,
        "followthrough_beats": FOLLOWTHROUGH_BEATS,
        "background_frame_stride": BACKGROUND_FRAME_STRIDE,
        "target_pose_horizon_frames": TARGET_POSE_HORIZON_FRAMES,
        "sim_note_time_range_beats": [SIM_NOTE_TIME_MIN_BEATS, SIM_NOTE_TIME_MAX_BEATS],
        "note_time_feature_range_beats": [NOTE_TIME_FEATURE_MIN_BEATS, NOTE_TIME_FEATURE_MAX_BEATS],
        "sim_obstacle_time_range_beats": [SIM_OBSTACLE_TIME_MIN_BEATS, SIM_OBSTACLE_TIME_MAX_BEATS],
        "num_upcoming_notes": NUM_UPCOMING_NOTES,
        "note_features": NOTE_FEATURES,
        "num_upcoming_obstacles": NUM_UPCOMING_OBSTACLES,
        "obstacle_features": OBSTACLE_FEATURES,
        "pose_dim": POSE_DIM,
        "velocity_dim": VELOCITY_DIM,
        "track_z_base": TRACK_Z_BASE,
        "shard_storage_dtype": SHARD_STORAGE_DTYPE,
        "feature_storage_dtype": FEATURE_STORAGE_DTYPE,
        "action_contract": {
            "action_dim": POSE_DIM,
            "action_representation": "absolute_tracked_pose_target",
            "policy_mean_contract": "current_pose_plus_residual_delta",
            "simulator_consumption": "GpuBeatSaberSimulator.step(pose_actions)",
            "absolute_component_limit": ACTION_ABS_COMPONENT_LIMIT,
            "per_step_delta_clamp": list(TARGET_ACTION_DELTA_CLAMP),
            "quaternion_slices": [list(bounds) for bounds in POSE_QUATERNION_SLICES],
        },
        "target_contract": {
            "target_representation": "absolute_tracked_pose_action",
            "target_generation": (
                "current_pose + clamp((future_pose - current_pose) / "
                "target_pose_horizon_frames, -per_step_delta_clamp, per_step_delta_clamp)"
            ),
            "future_pose_horizon_frames": TARGET_POSE_HORIZON_FRAMES,
            "normalizes_quaternions": True,
            "stored_dtype": TARGET_STORAGE_DTYPE,
        },
        "sentinel_contract": {
            "missing_note": {
                "time": 0.0,
                "line_index": 0.0,
                "line_layer": 0.0,
                "note_type": -1.0,
                "cut_dx": 0.0,
                "cut_dy": 0.0,
                "score_class": 0.0,
                "score_cap": 0.0,
                "time_seconds": 0.0,
                "z_distance": 0.0,
            },
            "hidden_future_note": "uses_missing_note_sentinel_until_spawn_visible",
            "missing_obstacle": [0.0] * OBSTACLE_FEATURES,
        },
        "followthrough_contract": {
            "keep_next_note_window_beats": NOTE_LOOKAHEAD_BEATS,
            "keep_previous_note_window_beats": FOLLOWTHROUGH_BEATS,
            "keep_background_frame_stride": BACKGROUND_FRAME_STRIDE,
            "timing_reference": "contact_shifted_note_time",
        },
        "score_class_values": {
            "normal": SCORE_CLASS_NORMAL,
            "arc_head": SCORE_CLASS_ARC_HEAD,
            "arc_tail": SCORE_CLASS_ARC_TAIL,
            "chain_head": SCORE_CLASS_CHAIN_HEAD,
            "chain_link": SCORE_CLASS_CHAIN_LINK,
        },
    }


def _manifest_value_matches(expected, actual):
    if isinstance(expected, float):
        try:
            actual_float = float(actual)
        except (TypeError, ValueError):
            return False
        return np.isfinite(actual_float) and abs(actual_float - expected) <= 1e-6
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_manifest_value_matches(exp, got) for exp, got in zip(expected, actual))
        )
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(
            key in actual and _manifest_value_matches(value, actual[key])
            for key, value in expected.items()
        )
    return actual == expected


def _manifest_semantic_mismatches(expected, actual, prefix="semantic_schema"):
    if not isinstance(actual, dict):
        return [f"{prefix} missing or not an object"]

    mismatches = []
    for key, expected_value in expected.items():
        path = f"{prefix}.{key}"
        if key not in actual:
            mismatches.append(f"{path} missing")
            continue
        actual_value = actual[key]
        if isinstance(expected_value, dict) and isinstance(actual_value, dict):
            mismatches.extend(_manifest_semantic_mismatches(expected_value, actual_value, path))
        elif not _manifest_value_matches(expected_value, actual_value):
            mismatches.append(f"{path} mismatch")
    return mismatches


def _manifest_semantic_for_compatibility(manifest):
    semantic_schema = manifest.get("semantic_schema")
    if isinstance(semantic_schema, dict):
        return semantic_schema

    # Current-version manifests may still carry layout fields at top level if
    # they were written by a compatibility builder without nested semantics.
    legacy_required = (
        "sample_hz",
        "history_offsets",
        "note_feature_layout",
        "note_lookahead_beats",
        "followthrough_beats",
        "background_stride",
        "target_pose_horizon_frames",
    )
    if manifest.get("version") != MANIFEST_VERSION or any(key not in manifest for key in legacy_required):
        return semantic_schema

    legacy_schema = manifest_semantic_metadata()
    legacy_schema.update({
        "sample_hz": manifest.get("sample_hz"),
        "state_history_offsets": manifest.get("history_offsets"),
        "note_feature_layout": manifest.get("note_feature_layout"),
        "note_lookahead_beats": manifest.get("note_lookahead_beats"),
        "followthrough_beats": manifest.get("followthrough_beats"),
        "background_frame_stride": manifest.get("background_stride"),
        "target_pose_horizon_frames": manifest.get("target_pose_horizon_frames"),
    })
    if "shard_storage_dtype" in manifest:
        legacy_schema["shard_storage_dtype"] = manifest.get("shard_storage_dtype")
    return legacy_schema


def manifest_provenance_schema():
    return {
        "source": "bsor.info",
        "shard_fields": [
            "player_id",
            "player_name",
            "replay_timestamp",
            "game_version",
            "platform",
            "tracking_system",
            "hmd",
            "controller",
            "score",
            "left_handed",
            "player_height",
            "replay_start_time",
            "replay_fail_time",
        ],
    }


def manifest_compatibility_errors(manifest):
    expected_fields = {
        "version": MANIFEST_VERSION,
        "feature_dim": INPUT_DIM,
        "target_dim": POSE_DIM,
        "target_pose_horizon_frames": TARGET_POSE_HORIZON_FRAMES,
    }
    errors = [
        f"{key} expected {expected!r}, got {manifest.get(key)!r}"
        for key, expected in expected_fields.items()
        if manifest.get(key) != expected
    ]
    errors.extend(
        _manifest_semantic_mismatches(
            manifest_semantic_metadata(),
            _manifest_semantic_for_compatibility(manifest),
        )
    )
    return errors


DIFFICULTY_RANK = {
    "easy": 0,
    "normal": 1,
    "hard": 2,
    "expert": 3,
    "expertplus": 4,
}


def _sanitize_frame_time(raw_time, prev_time):
    sanitized = False
    try:
        time_value = float(raw_time)
    except (TypeError, ValueError):
        time_value = 0.0 if prev_time is None else prev_time + (1.0 / 120.0)
        sanitized = True

    if not np.isfinite(time_value):
        time_value = 0.0 if prev_time is None else prev_time + (1.0 / 120.0)
        sanitized = True

    if prev_time is not None and time_value < prev_time:
        time_value = prev_time
        sanitized = True

    return time_value, sanitized


def _sanitize_pose(raw_pose, prev_pose):
    pose = np.asarray(raw_pose, dtype=np.float32).copy()
    fallback = DEFAULT_POSE if prev_pose is None else prev_pose
    replaced_segments = 0

    for start, end in POSE_POSITION_SLICES:
        segment = pose[start:end]
        if (not np.isfinite(segment).all()) or float(np.max(np.abs(segment))) > POSITION_ABS_MAX:
            pose[start:end] = fallback[start:end]
            replaced_segments += 1

    for start, end in POSE_QUATERNION_SLICES:
        segment = pose[start:end]
        if (not np.isfinite(segment).all()) or float(np.max(np.abs(segment))) > QUAT_COMPONENT_ABS_MAX:
            pose[start:end] = fallback[start:end]
            replaced_segments += 1
            continue

        norm = float(np.linalg.norm(segment))
        if (not np.isfinite(norm)) or norm < 1e-6:
            pose[start:end] = fallback[start:end]
            replaced_segments += 1
            continue

        pose[start:end] = segment / norm

    return pose, replaced_segments


def _normalize_quaternion(quat):
    norm = max(float(np.linalg.norm(quat)), 1e-6)
    return quat / norm


def _interpolate_pose(pose_a, pose_b, alpha):
    pose_a = np.asarray(pose_a, dtype=np.float32)
    pose_b = np.asarray(pose_b, dtype=np.float32)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = pose_a.copy()

    for start, end in POSE_POSITION_SLICES:
        blended[start:end] = pose_a[start:end] + (pose_b[start:end] - pose_a[start:end]) * alpha

    for start, end in POSE_QUATERNION_SLICES:
        qa = _normalize_quaternion(pose_a[start:end])
        qb = _normalize_quaternion(pose_b[start:end])
        if float(np.dot(qa, qb)) < 0.0:
            qb = -qb
        blended[start:end] = _normalize_quaternion(qa + (qb - qa) * alpha)

    return blended


def _normalize_pose_quaternions(pose):
    pose = np.asarray(pose, dtype=np.float32).copy()
    for start, end in POSE_QUATERNION_SLICES:
        pose[start:end] = _normalize_quaternion(pose[start:end])
    return pose


def _apply_state_pose_contract(poses):
    poses = np.asarray(poses, dtype=np.float32).copy()
    for start, end in POSE_POSITION_SLICES:
        poses[..., start:end] = np.clip(
            poses[..., start:end],
            -STATE_POSE_POSITION_ABS_LIMIT,
            STATE_POSE_POSITION_ABS_LIMIT,
        )
    for start, end in POSE_QUATERNION_SLICES:
        segment = poses[..., start:end]
        norm = np.linalg.norm(segment, axis=-1, keepdims=True)
        identity = np.zeros_like(segment)
        identity[..., 3] = 1.0
        poses[..., start:end] = np.where(
            norm > 1e-6,
            segment / np.clip(norm, 1e-6, None),
            identity,
        )
    return poses


def _sim_executable_pose_target(current_pose, future_pose):
    current_pose = np.asarray(current_pose, dtype=np.float32)
    future_pose = np.asarray(future_pose, dtype=np.float32)
    horizon = max(1, int(TARGET_POSE_HORIZON_FRAMES))
    target_delta = (future_pose - current_pose) / float(horizon)
    target_delta = np.clip(
        target_delta,
        -TARGET_ACTION_DELTA_CLAMP_ARRAY,
        TARGET_ACTION_DELTA_CLAMP_ARRAY,
    )
    target_pose = current_pose + target_delta
    target_pose = np.clip(
        target_pose,
        -ACTION_ABS_COMPONENT_LIMIT,
        ACTION_ABS_COMPONENT_LIMIT,
    )
    return _normalize_pose_quaternions(target_pose)


def resample_frames_to_sim_rate(frames, target_dt=SIM_SAMPLE_DT):
    if len(frames) <= 1:
        return list(frames)

    source_times = np.asarray([frame["time"] for frame in frames], dtype=np.float32)
    if source_times[-1] <= source_times[0]:
        return [dict(frames[0])]

    sample_times = np.arange(
        float(source_times[0]),
        float(source_times[-1]) + (0.25 * float(target_dt)),
        float(target_dt),
        dtype=np.float32,
    )
    if sample_times.size == 0:
        sample_times = np.asarray([source_times[0]], dtype=np.float32)

    resampled = []
    src_idx = 0
    max_src_idx = len(frames) - 1

    for sample_time in sample_times:
        while src_idx + 1 < len(frames) and source_times[src_idx + 1] < sample_time:
            src_idx += 1

        next_idx = min(src_idx + 1, max_src_idx)
        left = frames[src_idx]
        right = frames[next_idx]
        left_time = float(source_times[src_idx])
        right_time = float(source_times[next_idx])
        if next_idx == src_idx or right_time <= left_time:
            pose = np.asarray(left["pose"], dtype=np.float32)
        else:
            alpha = (float(sample_time) - left_time) / max(right_time - left_time, 1e-6)
            pose = _interpolate_pose(left["pose"], right["pose"], alpha)

        resampled.append({
            "time": float(sample_time),
            "pose": pose.tolist(),
        })

    return resampled


def _arrays_fit_float16(*arrays):
    for array in arrays:
        arr = np.asarray(array, dtype=np.float32)
        if arr.size == 0:
            continue
        if not np.isfinite(arr).all():
            return False
        if float(np.max(np.abs(arr))) > FLOAT16_MAX:
            return False
    return True


def _make_note(index, time_beat, line_index, line_layer, note_type, cut_direction, **extra):
    note = {
        'index': int(index),
        'time': float(time_beat),
        'lineIndex': float(line_index),
        'lineLayer': float(line_layer),
        'type': int(note_type),
        'cutDirection': int(cut_direction),
        'arcHead': False,
        'arcTail': False,
        'chainHead': False,
        'chainLink': False,
        'scoreClass': SCORE_CLASS_NORMAL,
        'scoreCap': 115.0,
        'preScale': 1.0,
        'postScale': 1.0,
        'accScale': 1.0,
        'preAuto': 0.0,
        'postAuto': 0.0,
        'fixedScore': 0.0,
        'requiresSpeed': True,
        'requiresDirection': True,
        'allowAnyDirection': False,
    }
    note.update(extra)
    return note


def _make_obstacle(time_beat, line_index, line_layer, width, height, duration):
    return {
        'time': float(time_beat),
        'lineIndex': float(line_index),
        'lineLayer': float(line_layer),
        'width': float(width),
        'height': float(height),
        'duration': float(duration),
    }


def _match_note_key(time_beat, line_index, line_layer, note_type):
    return (
        round(float(time_beat), 6),
        round(float(line_index), 4),
        round(float(line_layer), 4),
        int(note_type),
    )


def _finalize_note_scoring(note):
    if int(note['type']) == 3:
        note['scoreClass'] = -1.0
        note['scoreCap'] = 0.0
        note['preScale'] = 0.0
        note['postScale'] = 0.0
        note['accScale'] = 0.0
        note['requiresSpeed'] = False
        note['requiresDirection'] = False
        note['allowAnyDirection'] = True
        return note

    if note.get('chainLink', False):
        note['scoreClass'] = SCORE_CLASS_CHAIN_LINK
        note['scoreCap'] = 20.0
        note['preScale'] = 0.0
        note['postScale'] = 0.0
        note['accScale'] = 0.0
        note['fixedScore'] = 20.0
        note['requiresSpeed'] = False
        note['requiresDirection'] = False
        note['allowAnyDirection'] = True
        note['cutDirection'] = 8
        return note

    note['allowAnyDirection'] = int(note.get('cutDirection', 8)) == 8
    note['requiresDirection'] = not note['allowAnyDirection']
    note['requiresSpeed'] = True
    note['preScale'] = 0.0 if note.get('arcTail', False) else 1.0
    note['postScale'] = 0.0 if (note.get('arcHead', False) or note.get('chainHead', False)) else 1.0
    note['preAuto'] = 70.0 if note.get('arcTail', False) else 0.0
    note['postAuto'] = 30.0 if (note.get('arcHead', False) and not note.get('chainHead', False)) else 0.0

    if note.get('chainHead', False):
        note['scoreClass'] = SCORE_CLASS_CHAIN_HEAD
        note['scoreCap'] = 85.0
    elif note.get('arcHead', False):
        note['scoreClass'] = SCORE_CLASS_ARC_HEAD
    elif note.get('arcTail', False):
        note['scoreClass'] = SCORE_CLASS_ARC_TAIL

    return note


def parse_beatmap_dat(dat_content):
    data = json.loads(dat_content.decode('utf-8'))
    beatmap = {
        'notes': [],
        'obstacles': [],
        'arcs': [],
        'chains': [],
    }

    if '_notes' in data:
        for i, note in enumerate(data.get('_notes', [])):
            if note.get('_type') in [0, 1, 3]:
                beatmap['notes'].append(_make_note(
                    i,
                    note.get('_time', 0.0),
                    note.get('_lineIndex', 0),
                    note.get('_lineLayer', 0),
                    note.get('_type', 0),
                    note.get('_cutDirection', 8 if note.get('_type') == 3 else 0),
                ))

        for obstacle in data.get('_obstacles', []):
            legacy_type = int(obstacle.get('_type', 0))
            if legacy_type == 1:
                line_layer, height = 2, 3
            else:
                line_layer, height = 0, 5
            beatmap['obstacles'].append(_make_obstacle(
                obstacle.get('_time', 0.0),
                obstacle.get('_lineIndex', 0),
                line_layer,
                obstacle.get('_width', 1),
                height,
                obstacle.get('_duration', 0.0),
            ))

        for slider in data.get('_sliders', []):
            beatmap['arcs'].append({
                'type': 'arc',
                'color': int(slider.get('_colorType', 0)),
                'headTime': float(slider.get('_headTime', 0.0)),
                'headLineIndex': float(slider.get('_headLineIndex', 0)),
                'headLineLayer': float(slider.get('_headLineLayer', 0)),
                'tailTime': float(slider.get('_tailTime', 0.0)),
                'tailLineIndex': float(slider.get('_tailLineIndex', 0)),
                'tailLineLayer': float(slider.get('_tailLineLayer', 0)),
            })

    elif 'colorNotes' in data or 'bombNotes' in data:
        all_notes = []
        for n in data.get('colorNotes', []):
            all_notes.append(_make_note(
                len(all_notes),
                n.get('b', 0.0),
                n.get('x', 0),
                n.get('y', 0),
                n.get('c', 0),
                n.get('d', 0),
            ))
        for n in data.get('bombNotes', []):
            all_notes.append(_make_note(
                len(all_notes),
                n.get('b', 0.0),
                n.get('x', 0),
                n.get('y', 0),
                3,
                8,
            ))
        beatmap['notes'].extend(all_notes)

        for obstacle in data.get('obstacles', []):
            beatmap['obstacles'].append(_make_obstacle(
                obstacle.get('b', 0.0),
                obstacle.get('x', 0),
                obstacle.get('y', 0),
                obstacle.get('w', 1),
                obstacle.get('h', 5),
                obstacle.get('d', 0.0),
            ))

        for slider in data.get('sliders', []):
            beatmap['arcs'].append({
                'type': 'arc',
                'color': int(slider.get('c', 0)),
                'headTime': float(slider.get('b', 0.0)),
                'headLineIndex': float(slider.get('x', 0)),
                'headLineLayer': float(slider.get('y', 0)),
                'tailTime': float(slider.get('tb', 0.0)),
                'tailLineIndex': float(slider.get('tx', 0)),
                'tailLineLayer': float(slider.get('ty', 0)),
            })

        for chain in data.get('burstSliders', []):
            beatmap['chains'].append({
                'type': 'chain',
                'color': int(chain.get('c', 0)),
                'headTime': float(chain.get('b', 0.0)),
                'headLineIndex': float(chain.get('x', 0)),
                'headLineLayer': float(chain.get('y', 0)),
                'headCutDirection': int(chain.get('d', 0)),
                'tailTime': float(chain.get('tb', 0.0)),
                'tailLineIndex': float(chain.get('tx', 0)),
                'tailLineLayer': float(chain.get('ty', 0)),
                'sliceCount': max(1, int(chain.get('sc', 1))),
                'squish': float(chain.get('s', 1.0)),
            })

    beatmap['notes'].sort(key=lambda x: x['time'])
    beatmap['obstacles'].sort(key=lambda x: x['time'])

    note_lookup = {}
    for idx, note in enumerate(beatmap['notes']):
        note['index'] = idx
        if int(note['type']) in (0, 1):
            key = _match_note_key(note['time'], note['lineIndex'], note['lineLayer'], note['type'])
            note_lookup.setdefault(key, []).append(note)

    for arc in beatmap['arcs']:
        head_key = _match_note_key(arc['headTime'], arc['headLineIndex'], arc['headLineLayer'], arc['color'])
        tail_key = _match_note_key(arc['tailTime'], arc['tailLineIndex'], arc['tailLineLayer'], arc['color'])
        for note in note_lookup.get(head_key, []):
            note['arcHead'] = True
        for note in note_lookup.get(tail_key, []):
            note['arcTail'] = True

    next_index = len(beatmap['notes'])
    for chain in beatmap['chains']:
        head_key = _match_note_key(
            chain['headTime'],
            chain['headLineIndex'],
            chain['headLineLayer'],
            chain['color'],
        )
        for note in note_lookup.get(head_key, []):
            note['chainHead'] = True

        slice_count = max(1, int(chain.get('sliceCount', 1)))
        squish = max(1e-3, float(chain.get('squish', 1.0)))
        for slice_idx in range(1, slice_count):
            frac = min(1.0, (slice_idx / max(1, slice_count - 1)) * squish)
            link = _make_note(
                next_index,
                chain['headTime'] + (chain['tailTime'] - chain['headTime']) * frac,
                chain['headLineIndex'] + (chain['tailLineIndex'] - chain['headLineIndex']) * frac,
                chain['headLineLayer'] + (chain['tailLineLayer'] - chain['headLineLayer']) * frac,
                chain['color'],
                8,
                chainLink=True,
            )
            beatmap['notes'].append(link)
            next_index += 1

    beatmap['notes'].sort(key=lambda x: (x['time'], x['type'], x['lineIndex'], x['lineLayer']))
    for idx, note in enumerate(beatmap['notes']):
        note['index'] = idx
        _finalize_note_scoring(note)

    return beatmap


def parse_map_dat(dat_content):
    return parse_beatmap_dat(dat_content)['notes']


def _collect_difficulty_entries(info_data):
    entries = []

    if '_difficultyBeatmapSets' in info_data:
        beatmap_sets = info_data['_difficultyBeatmapSets']
        for set_data in beatmap_sets:
            mode = set_data.get('_beatmapCharacteristicName', 'Standard')
            for beatmap in set_data.get('_difficultyBeatmaps', []):
                entries.append({
                    'mode': mode,
                    'difficulty': beatmap.get('_difficulty', ''),
                    'filename': beatmap.get('_beatmapFilename'),
                    'rank': DIFFICULTY_RANK.get(normalize_difficulty_name(beatmap.get('_difficulty', '')), -1),
                    'note_jump_movement_speed': beatmap.get('_noteJumpMovementSpeed'),
                    'note_jump_start_beat_offset': beatmap.get('_noteJumpStartBeatOffset'),
                })

    if 'difficultyBeatmapSets' in info_data:
        beatmap_sets = info_data['difficultyBeatmapSets']
        for set_data in beatmap_sets:
            mode = set_data.get('beatmapCharacteristicName', 'Standard')
            for beatmap in set_data.get('difficultyBeatmaps', []):
                difficulty = (
                    beatmap.get('difficulty')
                    or beatmap.get('difficultyName')
                    or beatmap.get('customDifficultyName')
                    or ''
                )
                entries.append({
                    'mode': mode,
                    'difficulty': difficulty,
                    'filename': beatmap.get('beatmapFilename'),
                    'rank': DIFFICULTY_RANK.get(normalize_difficulty_name(difficulty), -1),
                    'note_jump_movement_speed': beatmap.get('noteJumpMovementSpeed'),
                    'note_jump_start_beat_offset': beatmap.get('noteJumpStartBeatOffset'),
                })

    return [entry for entry in entries if entry.get('filename')]


def _select_dat_file(info_data, preferred_mode=None, preferred_difficulty=None, strict_mode=False):
    entries = _collect_difficulty_entries(info_data)
    if not entries:
        return None

    pref_mode = normalize_mode_name(preferred_mode or "Standard")
    pref_diff = normalize_difficulty_name(preferred_difficulty or "")
    has_preferred_difficulty = pref_diff != ""

    exact_match = [
        entry for entry in entries
        if normalize_mode_name(entry['mode']) == pref_mode
        and normalize_difficulty_name(entry['difficulty']) == pref_diff
    ]
    if exact_match:
        return exact_match[0]['filename']

    if has_preferred_difficulty:
        return None

    mode_match = [entry for entry in entries if normalize_mode_name(entry['mode']) == pref_mode]
    if mode_match:
        return max(mode_match, key=lambda entry: entry['rank'])['filename']

    if strict_mode and preferred_mode is not None:
        return None

    standard_match = [entry for entry in entries if normalize_mode_name(entry['mode']) == "standard"]
    if standard_match:
        return max(standard_match, key=lambda entry: entry['rank'])['filename']

    return max(entries, key=lambda entry: entry['rank'])['filename']


def get_map_data(map_hash, preferred_difficulty=None, preferred_mode="Standard"):
    zip_path = os.path.join(MAPS_DIR, f"{map_hash}.zip")
    if not os.path.exists(zip_path):
        zip_path = os.path.join(MAPS_DIR, f"{str(map_hash).upper()}.zip")
        if not os.path.exists(zip_path):
            zip_path = os.path.join(MAPS_DIR, f"{str(map_hash).lower()}.zip")
            if not os.path.exists(zip_path):
                return None, None

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            info_files = [f for f in z.namelist() if os.path.basename(f).lower() == 'info.dat']
            if not info_files:
                return None, None

            info_data = json.loads(z.read(info_files[0]).decode('utf-8'))
            bpm = info_data.get('_beatsPerMinute', info_data.get('beatsPerMinute', 120.0))

            selected_entry = None
            entries = _collect_difficulty_entries(info_data)
            dat_file = _select_dat_file(
                info_data,
                preferred_mode=preferred_mode,
                preferred_difficulty=preferred_difficulty,
                strict_mode=preferred_mode is not None,
            )
            if dat_file:
                for entry in entries:
                    if entry.get('filename') == dat_file:
                        selected_entry = entry
                        break

            if not dat_file:
                if preferred_mode is not None:
                    return None, None
                dats = [
                    f for f in z.namelist()
                    if f.lower().endswith('.dat') and os.path.basename(f).lower() != 'info.dat'
                ]
                dat_file = dats[0] if dats else None

            if dat_file:
                dat_content = z.read(dat_file)
                beatmap = parse_beatmap_dat(dat_content)
                beatmap['mode'] = (selected_entry or {}).get('mode', preferred_mode or 'Standard')
                beatmap['difficulty'] = (selected_entry or {}).get('difficulty', preferred_difficulty)
                beatmap['njs'] = float((selected_entry or {}).get('note_jump_movement_speed', info_data.get('_noteJumpMovementSpeed', info_data.get('noteJumpMovementSpeed', 18.0))))
                beatmap['offset'] = float((selected_entry or {}).get('note_jump_start_beat_offset', info_data.get('_noteJumpStartBeatOffset', info_data.get('noteJumpStartBeatOffset', 0.0))))
                beatmap['song_name'] = (
                    info_data.get('_songName')
                    or info_data.get('songName')
                    or info_data.get('_songSubName')
                    or info_data.get('songSubName')
                    or map_hash
                )
                beatmap['song_author_name'] = info_data.get('_songAuthorName') or info_data.get('songAuthorName') or ''
                beatmap['level_author_name'] = info_data.get('_levelAuthorName') or info_data.get('levelAuthorName') or ''
                beatmap['environment_name'] = info_data.get('_environmentName') or info_data.get('environmentName') or 'DefaultEnvironment'
                return beatmap, bpm
    except Exception as e:
        print(f"Failed to parse map data {map_hash}: {e}")

    return None, None


def parse_bsor(file_path):
    backend = str(os.environ.get("CYBERNOODLES_BSOR_BACKEND", "auto")).strip().lower()
    python_error = None

    if backend != "rust":
        try:
            with open(file_path, "rb") as f:
                replay = make_bsor(f)

            info = getattr(replay, "info", None)
            frames = replay.frames
            parsed_frames = []
            prev_time = None
            prev_pose = None
            sanitized_time_frames = 0
            sanitized_pose_segments = 0
            for frame in frames:
                frame_time, time_sanitized = _sanitize_frame_time(getattr(frame, 'time', 0.0), prev_time)
                pose, replaced_segments = _sanitize_pose([
                    *frame.head.position, *frame.head.rotation,
                    *frame.left_hand.position, *frame.left_hand.rotation,
                    *frame.right_hand.position, *frame.right_hand.rotation
                ], prev_pose)
                parsed_frames.append({
                    'time': frame_time,
                    'pose': pose.tolist(),
                })
                prev_time = frame_time
                prev_pose = pose
                sanitized_time_frames += int(time_sanitized)
                sanitized_pose_segments += replaced_segments

            replay_meta = {
                'song_hash': getattr(info, 'songHash', None),
                'difficulty': getattr(info, 'difficulty', None),
                'mode': getattr(info, 'mode', 'Standard') or 'Standard',
                'modifiers': getattr(info, 'modifiers', '') or '',
                'player_id': getattr(info, 'playerId', None),
                'player_name': getattr(info, 'playerName', None),
                'replay_timestamp': getattr(info, 'timestamp', None),
                'game_version': getattr(info, 'gameVersion', None),
                'platform': getattr(info, 'platform', None),
                'tracking_system': getattr(info, 'trackingSystem', None),
                'hmd': getattr(info, 'hmd', None),
                'controller': getattr(info, 'controller', None),
                'score': int(getattr(info, 'score', 0) or 0),
                'left_handed': bool(getattr(info, 'leftHanded', False)),
                'player_height': float(getattr(info, 'height', 0.0) or 0.0),
                'replay_start_time': float(getattr(info, 'startTime', 0.0) or 0.0),
                'replay_fail_time': float(getattr(info, 'failTime', 0.0) or 0.0),
                'sanitized_time_frames': sanitized_time_frames,
                'sanitized_pose_segments': sanitized_pose_segments,
            }
            return parsed_frames, replay_meta
        except BaseException as exc:
            python_error = exc
            if backend == "python":
                print(f"Error parsing BSOR {file_path}: {exc}")
                return None, None

    try:
        auto_build = backend == "rust" or python_error is not None
        return parse_dataset_view_via_rust(file_path, auto_build=auto_build)
    except Exception as rust_exc:
        if python_error is not None:
            print(
                f"Error parsing BSOR {file_path}: "
                f"python={python_error} | rust={rust_exc}"
            )
        else:
            print(f"Error parsing BSOR {file_path}: {rust_exc}")
        return None, None


def get_map_notes(map_hash, preferred_difficulty=None, preferred_mode="Standard"):
    beatmap, bpm = get_map_data(
        map_hash,
        preferred_difficulty=preferred_difficulty,
        preferred_mode=preferred_mode,
    )
    if not beatmap:
        return None, None
    return beatmap['notes'], bpm


def _frame_velocity(hand_poses, times, frame_idx):
    if frame_idx <= 0:
        return np.zeros(VELOCITY_DIM, dtype=np.float32)

    prev_idx = frame_idx - 1
    dt = max(times[frame_idx] - times[prev_idx], 1.0 / 120.0)
    return (hand_poses[frame_idx] - hand_poses[prev_idx]) / dt


def _coerce_beatmap(beatmap_or_notes):
    if isinstance(beatmap_or_notes, dict):
        return beatmap_or_notes
    return {
        'notes': beatmap_or_notes,
        'obstacles': [],
    }


def _build_note_feature_vector(notes, note_idx, t_beat, bps, note_jump_speed, head_z, spawn_ahead_beats):
    feature_vec = []
    safe_bps = max(float(bps), 1e-6)
    safe_njs = max(float(note_jump_speed), 0.0)
    safe_head_z = float(head_z)
    visible_beats = float(max(spawn_ahead_beats, 0.0))
    for offset in range(NUM_UPCOMING_NOTES):
        if note_idx + offset < len(notes):
            note = notes[note_idx + offset]
            raw_time_offset = float(note['time'] - t_beat)
            if raw_time_offset > visible_beats:
                feature_vec.extend([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                continue
            time_offset = float(np.clip(
                raw_time_offset,
                SIM_NOTE_TIME_MIN_BEATS,
                SIM_NOTE_TIME_MAX_BEATS,
            ))
            # Mirror the live simulator exactly: clamp the raw beat offset first,
            # but only after the note would have spawned for this map's jump
            # settings. BC must see the same note visibility and state timing
            # that it will get during closed-loop rollout.
            time_seconds = (time_offset / safe_bps) + ((TRACK_Z_BASE + safe_head_z) / max(safe_njs, 1e-6))
            contact_time_beats = float(np.clip(
                time_seconds * safe_bps,
                NOTE_TIME_FEATURE_MIN_BEATS,
                NOTE_TIME_FEATURE_MAX_BEATS,
            ))
            time_seconds = contact_time_beats / safe_bps
            z_distance = time_seconds * safe_njs
            dx, dy = encode_cut_direction(note['cutDirection'])
            feature_vec.extend([
                contact_time_beats,
                note['lineIndex'],
                note['lineLayer'],
                note['type'],
                dx,
                dy,
                note.get('scoreClass', SCORE_CLASS_NORMAL),
                float(note.get('scoreCap', 115.0)) / 115.0,
                time_seconds,
                z_distance,
            ])
        else:
            feature_vec.extend([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return feature_vec


def _build_obstacle_feature_vector(obstacles, obstacle_idx, t_beat, spawn_ahead_beats):
    feature_vec = []
    visible_beats = float(max(spawn_ahead_beats, 0.0))
    for offset in range(NUM_UPCOMING_OBSTACLES):
        if obstacle_idx + offset < len(obstacles):
            obstacle = obstacles[obstacle_idx + offset]
            raw_time_offset = float(obstacle['time'] - t_beat)
            if raw_time_offset > visible_beats:
                feature_vec.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                continue
            time_offset = float(np.clip(
                raw_time_offset,
                SIM_OBSTACLE_TIME_MIN_BEATS,
                SIM_OBSTACLE_TIME_MAX_BEATS,
            ))
            feature_vec.extend([
                time_offset,
                obstacle['lineIndex'],
                obstacle['lineLayer'],
                obstacle['width'],
                obstacle['height'],
                obstacle['duration'],
            ])
        else:
            feature_vec.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return feature_vec


def _should_keep_frame(frame_idx, next_delta, prev_delta):
    return (
        next_delta <= NOTE_LOOKAHEAD_BEATS
        or prev_delta <= FOLLOWTHROUGH_BEATS
        or frame_idx % BACKGROUND_FRAME_STRIDE == 0
    )


def extract_features(frames, beatmap_or_notes, bpm):
    beatmap = _coerce_beatmap(beatmap_or_notes)
    notes = beatmap.get('notes', [])
    obstacles = beatmap.get('obstacles', [])

    if len(frames) < (MIN_REPLAY_FRAMES + TARGET_POSE_HORIZON_FRAMES) or len(notes) < 2:
        return [], [], {}

    poses = _apply_state_pose_contract(
        np.asarray([frame['pose'][0:21] for frame in frames], dtype=np.float32)
    )
    times = np.asarray([frame['time'] for frame in frames], dtype=np.float32)
    bps = max(float(bpm) / 60.0, 1e-6)
    note_jump_speed = float(beatmap.get('njs', 18.0) or 18.0)
    note_jump_offset = float(beatmap.get('offset', 0.0) or 0.0)
    spawn_ahead_beats = compute_spawn_ahead_beats(float(bpm), note_jump_speed, note_jump_offset)
    note_times = np.asarray([note['time'] for note in notes], dtype=np.float32)
    obstacle_times = np.asarray([obstacle['time'] for obstacle in obstacles], dtype=np.float32) if obstacles else np.asarray([], dtype=np.float32)

    X = []
    y = []
    note_idx = 0
    obstacle_idx = 0
    kept_note_window = 0
    kept_background = 0
    invalid_samples = 0

    total_candidates = max(0, len(frames) - TARGET_POSE_HORIZON_FRAMES)

    for current_idx in range(total_candidates):
        t_beat = float(times[current_idx] * bps)
        contact_shift_beats = (
            (TRACK_Z_BASE + float(poses[current_idx][2]))
            / max(note_jump_speed, 1e-6)
        ) * bps
        while (
            note_idx < len(notes)
            and (note_times[note_idx] + contact_shift_beats + FOLLOWTHROUGH_BEATS) < t_beat
        ):
            note_idx += 1
        while obstacle_idx < len(obstacles) and (obstacle_times[obstacle_idx] + obstacles[obstacle_idx]['duration']) < t_beat:
            obstacle_idx += 1

        next_delta = float((note_times[note_idx] + contact_shift_beats) - t_beat) if note_idx < len(notes) else 99.0
        prev_delta = float(t_beat - (note_times[note_idx - 1] + contact_shift_beats)) if note_idx > 0 else 99.0
        if not _should_keep_frame(current_idx, next_delta, prev_delta):
            continue

        feature_vec = _build_note_feature_vector(
            notes,
            note_idx,
            t_beat,
            bps,
            note_jump_speed,
            poses[current_idx][2],
            spawn_ahead_beats,
        )
        feature_vec.extend(_build_obstacle_feature_vector(obstacles, obstacle_idx, t_beat, spawn_ahead_beats))
        for offset in STATE_HISTORY_OFFSETS:
            hist_idx = max(0, current_idx - offset)
            feature_vec.extend(poses[hist_idx].tolist())
            feature_vec.extend(_frame_velocity(poses, times, hist_idx).tolist())

        if len(feature_vec) != INPUT_DIM:
            raise RuntimeError(f"Feature width mismatch: expected {INPUT_DIM}, got {len(feature_vec)}")

        target_idx = current_idx + TARGET_POSE_HORIZON_FRAMES
        target_pose = _sim_executable_pose_target(poses[current_idx], poses[target_idx])
        if not _arrays_fit_float16(feature_vec, target_pose):
            invalid_samples += 1
            continue
        X.append(feature_vec)
        y.append(target_pose.tolist())

        if next_delta <= NOTE_LOOKAHEAD_BEATS or prev_delta <= FOLLOWTHROUGH_BEATS:
            kept_note_window += 1
        else:
            kept_background += 1

    stats = {
        'frames_total': len(frames),
        'samples_kept': len(X),
        'samples_dropped': max(0, total_candidates - len(X)),
        'note_window_samples': kept_note_window,
        'background_samples': kept_background,
        'invalid_samples': invalid_samples,
    }
    return X, y, stats


def process_single(bsor_file):
    frames, replay_meta = parse_bsor(bsor_file)
    if frames is None or not replay_meta or not replay_meta.get('song_hash'):
        return None

    replay_meta["frames_total_raw"] = len(frames)
    frames = resample_frames_to_sim_rate(frames)
    replay_meta["frames_total_resampled"] = len(frames)
    replay_meta["sample_hz"] = float(SIM_SAMPLE_HZ)

    if normalize_mode_name(replay_meta.get('mode')) != "standard":
        print(f"  Skipping non-standard replay {os.path.basename(bsor_file)}")
        return None

    if str(replay_meta.get('modifiers', '')).strip():
        print(f"  Skipping modified replay {os.path.basename(bsor_file)}")
        return None

    beatmap, bpm = get_map_data(
        replay_meta['song_hash'],
        preferred_difficulty=replay_meta.get('difficulty'),
        preferred_mode=replay_meta.get('mode'),
    )
    if not beatmap:
        if normalize_difficulty_name(replay_meta.get('difficulty')) != "":
            print(f"  Missing map/difficulty for {replay_meta['song_hash']}")
            return None
        fallback_beatmap, fallback_bpm = get_map_data(replay_meta['song_hash'])
        if not fallback_beatmap:
            print(f"  Missing map/difficulty for {replay_meta['song_hash']}")
            return None
        beatmap, bpm = fallback_beatmap, fallback_bpm

    X, y, stats = extract_features(frames, beatmap, bpm)
    if len(X) < MIN_SAMPLE_FRAMES:
        print(f"  Too few usable samples in {os.path.basename(bsor_file)}")
        return None
    if not _arrays_fit_float16(X, y):
        print(f"  Non-finite or overflowed samples in {os.path.basename(bsor_file)}")
        return None

    replay_meta.update(stats)
    replay_meta["mode"] = beatmap.get("mode", replay_meta.get("mode"))
    replay_meta["difficulty"] = beatmap.get("difficulty", replay_meta.get("difficulty"))
    replay_meta["bpm"] = float(bpm)
    replay_meta["njs"] = float(beatmap.get('njs', 18.0) or 18.0)
    replay_meta["jump_offset"] = float(beatmap.get('offset', 0.0) or 0.0)
    return replay_meta, X, y


def assign_split(song_hash):
    digest = hashlib.md5(str(song_hash).upper().encode('utf-8')).digest()[0]
    threshold = int(256 * VAL_FRACTION)
    return "val" if digest < threshold else "train"


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        compatibility_errors = manifest_compatibility_errors(manifest)
        if not compatibility_errors:
            manifest.setdefault("warnings", [])
            manifest.setdefault("provenance_schema", manifest_provenance_schema())
            return manifest

        print("Existing BC shard manifest is incompatible. Rebuilding shards with the current state semantics.")
        for detail in compatibility_errors[:5]:
            print(f"  - {detail}")

    return {
        "version": MANIFEST_VERSION,
        "feature_dim": INPUT_DIM,
        "target_dim": POSE_DIM,
        "target_pose_horizon_frames": TARGET_POSE_HORIZON_FRAMES,
        "semantic_schema": manifest_semantic_metadata(),
        "provenance_schema": manifest_provenance_schema(),
        "sample_hz": SIM_SAMPLE_HZ,
        "history_offsets": list(STATE_HISTORY_OFFSETS),
        "note_feature_layout": NOTE_FEATURE_LAYOUT,
        "note_lookahead_beats": NOTE_LOOKAHEAD_BEATS,
        "followthrough_beats": FOLLOWTHROUGH_BEATS,
        "background_stride": BACKGROUND_FRAME_STRIDE,
        "done": [],
        "failed": [],
        "warnings": [],
        "shards": [],
        "counts": {
            "train_samples": 0,
            "val_samples": 0,
            "train_replays": 0,
            "val_replays": 0,
        },
    }


def save_manifest(manifest):
    os.makedirs(SHARD_ROOT, exist_ok=True)
    temp_path = f"{MANIFEST_PATH}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(temp_path, MANIFEST_PATH)


def _safe_torch_save(tensor, path):
    temp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(tensor, temp_path)
    os.replace(temp_path, path)


def _prepare_shard_paths(split, replay_name):
    split_dir = TRAIN_DIR if split == "train" else VAL_DIR
    os.makedirs(split_dir, exist_ok=True)
    shard_id = os.path.splitext(os.path.basename(replay_name))[0]
    x_rel = os.path.join(split, f"X_{shard_id}.pt")
    y_rel = os.path.join(split, f"y_{shard_id}.pt")
    x_path = os.path.join(SHARD_ROOT, x_rel)
    y_path = os.path.join(SHARD_ROOT, y_rel)
    return x_rel, y_rel, x_path, y_path


def _write_replay_shard_files(split, replay_name, X, y):
    x_rel, y_rel, x_path, y_path = _prepare_shard_paths(split, replay_name)

    tx = torch.tensor(X, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.float32)
    if not torch.isfinite(tx).all() or not torch.isfinite(ty).all():
        raise ValueError(f"{replay_name} contains non-finite shard tensors")

    tx = tx.to(dtype=torch.float16)
    if not torch.isfinite(tx).all() or not torch.isfinite(ty).all():
        raise ValueError(f"{replay_name} overflows shard storage")

    _safe_torch_save(tx, x_path)
    _safe_torch_save(ty, y_path)
    return x_rel, y_rel, len(X)


def _build_manifest_shard_entry(split, replay_name, replay_meta, x_rel, y_rel, sample_count):
    return {
        "split": split,
        "song_hash": str(replay_meta.get("song_hash", "")).upper(),
        "difficulty": replay_meta.get("difficulty"),
        "mode": replay_meta.get("mode"),
        "bpm": replay_meta.get("bpm"),
        "njs": replay_meta.get("njs"),
        "jump_offset": replay_meta.get("jump_offset"),
        "replay_file": os.path.basename(replay_name),
        "x_path": x_rel.replace("\\", "/"),
        "y_path": y_rel.replace("\\", "/"),
        "samples": sample_count,
        "frames_total": replay_meta.get("frames_total", 0),
        "frames_total_raw": replay_meta.get("frames_total_raw", replay_meta.get("frames_total", 0)),
        "frames_total_resampled": replay_meta.get("frames_total_resampled", replay_meta.get("frames_total", 0)),
        "sample_hz": replay_meta.get("sample_hz", SIM_SAMPLE_HZ),
        "samples_dropped": replay_meta.get("samples_dropped", 0),
        "note_window_samples": replay_meta.get("note_window_samples", 0),
        "background_samples": replay_meta.get("background_samples", 0),
        "player_id": replay_meta.get("player_id"),
        "player_name": replay_meta.get("player_name"),
        "replay_timestamp": replay_meta.get("replay_timestamp"),
        "game_version": replay_meta.get("game_version"),
        "platform": replay_meta.get("platform"),
        "tracking_system": replay_meta.get("tracking_system"),
        "hmd": replay_meta.get("hmd"),
        "controller": replay_meta.get("controller"),
        "score": replay_meta.get("score"),
        "left_handed": replay_meta.get("left_handed"),
        "player_height": replay_meta.get("player_height"),
        "replay_start_time": replay_meta.get("replay_start_time"),
        "replay_fail_time": replay_meta.get("replay_fail_time"),
    }


def _record_manifest_success(manifest, split, replay_name, replay_meta, x_rel, y_rel, sample_count):
    manifest["done"].append(os.path.basename(replay_name))
    manifest["shards"].append(_build_manifest_shard_entry(split, replay_name, replay_meta, x_rel, y_rel, sample_count))
    manifest["counts"][f"{split}_samples"] += sample_count
    manifest["counts"][f"{split}_replays"] += 1


def _record_manifest_failure(manifest, replay_name):
    if os.path.basename(replay_name) not in manifest["failed"]:
        manifest["failed"].append(os.path.basename(replay_name))


def _set_manifest_warning(manifest, code, message, **details):
    warning = {
        "code": str(code),
        "message": str(message),
        **details,
    }
    current = [
        item for item in manifest.get("warnings", [])
        if isinstance(item, dict)
    ]
    updated = [item for item in current if item.get("code") != code]
    updated.append(warning)
    if current == updated:
        return False
    manifest["warnings"] = updated
    return True


def _clear_manifest_warning(manifest, code):
    current = [
        item for item in manifest.get("warnings", [])
        if isinstance(item, dict)
    ]
    updated = [item for item in current if item.get("code") != code]
    if current == updated:
        return False
    manifest["warnings"] = updated
    return True


def _format_replay_preview(replay_names, limit=5):
    preview = list(replay_names[:limit])
    if len(replay_names) > limit:
        preview.append("...")
    return ", ".join(preview)


def _pending_replay_paths(bsor_files, done_set):
    return [
        replay for replay in bsor_files
        if os.path.basename(replay) not in done_set
    ]


def _process_and_save_single_for_builder(bsor_file):
    name = os.path.basename(bsor_file)
    result = process_single(bsor_file)
    if result is None:
        return {
            "status": "failed",
            "name": name,
            "reason": "process_single returned no data",
        }

    replay_meta, X, y = result
    split = assign_split(replay_meta.get("song_hash"))
    try:
        x_rel, y_rel, sample_count = _write_replay_shard_files(split, name, X, y)
    except ValueError as exc:
        return {
            "status": "failed",
            "name": name,
            "reason": str(exc),
        }

    return {
        "status": "success",
        "name": name,
        "split": split,
        "replay_meta": replay_meta,
        "x_rel": x_rel,
        "y_rel": y_rel,
        "sample_count": sample_count,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build BC shard dataset from downloaded replays.")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_BUILD_WORKERS,
        help="Parallel worker processes for replay parsing, feature extraction, and shard writes.",
    )
    parser.add_argument(
        "--manifest-save-every",
        type=int,
        default=DEFAULT_MANIFEST_SAVE_EVERY,
        help="How many committed replays to buffer between manifest saves.",
    )
    parser.add_argument(
        "--max-pending-writes",
        type=int,
        default=DEFAULT_MAX_PENDING_WRITES,
        help="Maximum in-flight replay jobs to keep queued across worker processes.",
    )
    parser.add_argument(
        "--gc-every",
        type=int,
        default=DEFAULT_GC_EVERY,
        help="How many committed replays to process between manual gc.collect() calls.",
    )
    parser.add_argument(
        "--top-selected",
        type=int,
        default=None,
        help="Only build shards for the first N replays from data/selected_scores.json.",
    )
    parser.add_argument(
        "--status-every",
        type=int,
        default=DEFAULT_STATUS_EVERY,
        help="Print one progress summary every N completed replays.",
    )
    return parser.parse_args()


def _load_selected_replay_subset(limit):
    if limit is None:
        return None
    limit = max(0, int(limit))
    if limit <= 0:
        return []
    if not os.path.exists(SELECTED_SCORES_PATH):
        raise FileNotFoundError(f"Missing selection manifest: {SELECTED_SCORES_PATH}")
    with open(SELECTED_SCORES_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    selected = payload.get("selected", [])
    if not isinstance(selected, list):
        raise ValueError(f"Invalid selection manifest: {SELECTED_SCORES_PATH}")

    replay_names = []
    seen = set()
    for item in selected:
        replay_id = str(item.get("id") or "").strip()
        if not replay_id:
            continue
        replay_name = f"{replay_id}.bsor"
        if replay_name in seen:
            continue
        seen.add(replay_name)
        replay_names.append(replay_name)
        if len(replay_names) >= int(limit):
            break
    return replay_names


def _normalize_dataset_builder_backend(value, default="rust"):
    backend = str(value or default).strip().lower()
    if backend not in {"auto", "python", "rust"}:
        return default
    return backend


def _process_data_via_rust(
    workers=DEFAULT_BUILD_WORKERS,
    manifest_save_every=DEFAULT_MANIFEST_SAVE_EVERY,
    max_pending_writes=DEFAULT_MAX_PENDING_WRITES,
    gc_every=DEFAULT_GC_EVERY,
    top_selected=None,
    status_every=DEFAULT_STATUS_EVERY,
):
    output = build_bc_dataset_via_rust(
        REPLAYS_DIR,
        MAPS_DIR,
        OUTPUT_DIR,
        SELECTED_SCORES_PATH,
        workers=workers,
        top_selected=top_selected,
        manifest_save_every=manifest_save_every,
        max_pending_writes=max_pending_writes,
        gc_every=gc_every,
        status_every=status_every,
    )
    if output.strip():
        print(output.rstrip())


def process_data(
    workers=DEFAULT_BUILD_WORKERS,
    manifest_save_every=DEFAULT_MANIFEST_SAVE_EVERY,
    max_pending_writes=DEFAULT_MAX_PENDING_WRITES,
    gc_every=DEFAULT_GC_EVERY,
    top_selected=None,
    status_every=DEFAULT_STATUS_EVERY,
):
    backend = _normalize_dataset_builder_backend(
        os.environ.get("CYBERNOODLES_DATASET_BUILDER_BACKEND", "rust")
    )
    if backend != "python":
        try:
            return _process_data_via_rust(
                workers=workers,
                manifest_save_every=manifest_save_every,
                max_pending_writes=max_pending_writes,
                gc_every=gc_every,
                top_selected=top_selected,
                status_every=status_every,
            )
        except Exception:
            if backend == "rust":
                raise
            print("Rust dataset builder failed; falling back to the Python compatibility path.")

    return _process_data_python(
        workers=workers,
        manifest_save_every=manifest_save_every,
        max_pending_writes=max_pending_writes,
        gc_every=gc_every,
        top_selected=top_selected,
        status_every=status_every,
    )


def _process_data_python(
    workers=DEFAULT_BUILD_WORKERS,
    manifest_save_every=DEFAULT_MANIFEST_SAVE_EVERY,
    max_pending_writes=DEFAULT_MAX_PENDING_WRITES,
    gc_every=DEFAULT_GC_EVERY,
    top_selected=None,
    status_every=DEFAULT_STATUS_EVERY,
):
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    workers = max(1, int(workers))
    manifest_save_every = max(1, int(manifest_save_every))
    max_pending_writes = max(1, int(max_pending_writes))
    gc_every = max(0, int(gc_every))
    status_every = max(1, int(status_every))

    bsor_files = sorted(glob.glob(os.path.join(REPLAYS_DIR, "*.bsor")))
    selected_subset = _load_selected_replay_subset(top_selected)
    missing_selected = []
    if selected_subset is not None:
        selected_set = set(selected_subset)
        local_replay_names = {os.path.basename(replay) for replay in bsor_files}
        missing_selected = [
            replay_name for replay_name in selected_subset
            if replay_name not in local_replay_names
        ]
        bsor_files = [
            replay for replay in bsor_files
            if os.path.basename(replay) in selected_set
        ]
    manifest = load_manifest()
    selected_warning_dirty = False
    if selected_subset is not None and missing_selected:
        message = (
            f"{len(missing_selected)} selected replay(s) are missing locally and will not be built."
        )
        print(f"Warning: {message} {_format_replay_preview(missing_selected)}")
        selected_warning_dirty = _set_manifest_warning(
            manifest,
            "selected_replays_missing",
            message,
            count=len(missing_selected),
            replay_files=missing_selected,
        )
    elif selected_subset is not None:
        selected_warning_dirty = _clear_manifest_warning(
            manifest,
            "selected_replays_missing",
        )
    if selected_warning_dirty:
        save_manifest(manifest)

    done_set = set(manifest.get("done", []))
    remaining = _pending_replay_paths(bsor_files, done_set)

    failed_count = len(set(manifest.get("failed", [])))
    print(
        f"Found {len(bsor_files)} replays total. "
        f"{len(done_set)} already processed, {failed_count} failed history, "
        f"{len(remaining)} remaining."
    )
    if selected_subset is not None:
        print(f"Subset mode: top {len(selected_subset)} replay(s) from {SELECTED_SCORES_PATH}.")
    if not remaining:
        counts = manifest["counts"]
        print("\nDataset build complete.")
        print(f"  Train: {counts['train_replays']} replays, {counts['train_samples']:,} samples")
        print(f"  Val:   {counts['val_replays']} replays, {counts['val_samples']:,} samples")
        print(f"  Shards: {len(manifest['shards'])}")
        return

    print(f"Using {workers} worker process(es) for replay processing.")
    print(
        f"Inflight queue: {max_pending_writes} pending | "
        f"manifest save every {manifest_save_every} replay(s)"
    )

    total_remaining = len(remaining)
    completed = 0
    inflight_limit = max(workers * 2, max_pending_writes)
    submitted = 0
    future_to_path = {}
    manifest_dirty = False
    unsaved_commits = 0
    commits_since_gc = 0
    succeeded = 0
    failed = 0
    started_at = time.perf_counter()

    def maybe_save_manifest(force=False):
        nonlocal manifest_dirty, unsaved_commits
        if not manifest_dirty:
            return
        if force or unsaved_commits >= manifest_save_every:
            save_manifest(manifest)
            manifest_dirty = False
            unsaved_commits = 0

    def mark_manifest_dirty():
        nonlocal manifest_dirty, unsaved_commits
        manifest_dirty = True
        unsaved_commits += 1

    def maybe_collect_gc(force=False):
        nonlocal commits_since_gc
        if gc_every <= 0:
            return
        if force or commits_since_gc >= gc_every:
            gc.collect()
            commits_since_gc = 0

    def print_status(force=False):
        elapsed = max(1e-6, time.perf_counter() - started_at)
        rate_per_min = completed / elapsed * 60.0
        remaining_count = max(0, total_remaining - completed)
        eta_minutes = (remaining_count / rate_per_min) if rate_per_min > 0 else float("inf")
        if force or completed == total_remaining or (completed % status_every) == 0:
            eta_text = f"{eta_minutes / 60.0:.1f}h" if eta_minutes != float("inf") else "unknown"
            print(
                f"Progress {completed}/{total_remaining} | ok {succeeded} | failed {failed} | "
                f"{rate_per_min:.2f} replays/min | ETA {eta_text}"
            )

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        while submitted < total_remaining and len(future_to_path) < inflight_limit:
            bsor_file = remaining[submitted]
            future = executor.submit(_process_and_save_single_for_builder, bsor_file)
            future_to_path[future] = bsor_file
            submitted += 1

        while future_to_path:
            future = next(concurrent.futures.as_completed(tuple(future_to_path.keys())))
            bsor_file = future_to_path.pop(future)
            name = os.path.basename(bsor_file)
            completed += 1

            try:
                payload = future.result()
            except Exception as exc:
                payload = {
                    "status": "failed",
                    "name": name,
                    "reason": str(exc),
                }

            if payload.get("status") == "success":
                succeeded += 1
                _record_manifest_success(
                    manifest,
                    payload["split"],
                    payload["name"],
                    payload["replay_meta"],
                    payload["x_rel"],
                    payload["y_rel"],
                    payload["sample_count"],
                )
            else:
                failed += 1
                _record_manifest_failure(manifest, payload.get("name", name))
                reason = payload.get("reason")
                if reason:
                    print(f"  Skipping {payload.get('name', name)}: {reason}")

            mark_manifest_dirty()
            maybe_save_manifest()
            commits_since_gc += 1
            maybe_collect_gc()
            print_status()

            while submitted < total_remaining and len(future_to_path) < inflight_limit:
                next_bsor = remaining[submitted]
                next_future = executor.submit(_process_and_save_single_for_builder, next_bsor)
                future_to_path[next_future] = next_bsor
                submitted += 1

        maybe_save_manifest(force=True)
        maybe_collect_gc(force=True)
        print_status(force=True)

    counts = manifest["counts"]
    print("\nDataset build complete.")
    print(
        f"  Train: {counts['train_replays']} replays, {counts['train_samples']:,} samples"
    )
    print(
        f"  Val:   {counts['val_replays']} replays, {counts['val_samples']:,} samples"
    )
    print(f"  Shards: {len(manifest['shards'])}")

    legacy_x = os.path.join(OUTPUT_DIR, "X_rl.pt")
    legacy_y = os.path.join(OUTPUT_DIR, "y_rl.pt")
    if os.path.exists(legacy_x) or os.path.exists(legacy_y):
        print("  Legacy monolithic tensors still exist in data/processed.")
        print("  You can delete X_rl.pt / y_rl.pt later if you want the disk space back.")


if __name__ == "__main__":
    args = parse_args()
    process_data(
        workers=args.workers,
        manifest_save_every=args.manifest_save_every,
        max_pending_writes=args.max_pending_writes,
        gc_every=args.gc_every,
        top_selected=args.top_selected,
        status_every=args.status_every,
    )
