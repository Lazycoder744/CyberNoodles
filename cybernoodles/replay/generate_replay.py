import argparse
from functools import lru_cache
import json
import os
import random
import sys
import time

import numpy as np
import requests
import torch
from bsor.Bsor import (
    Bsor,
    ControllerOffsets,
    Frame,
    Height,
    Info,
    Note,
    Cut,
    NOTE_EVENT_BAD,
    NOTE_EVENT_BOMB,
    NOTE_EVENT_GOOD,
    NOTE_EVENT_MISS,
    NOTE_SCORE_TYPE_BURSTSLIDERELEMENT,
    NOTE_SCORE_TYPE_BURSTSLIDERHEAD,
    NOTE_SCORE_TYPE_NOSCORE,
    NOTE_SCORE_TYPE_NORMAL_2,
    NOTE_SCORE_TYPE_SLIDERHEAD,
    NOTE_SCORE_TYPE_SLIDERTAIL,
    SABER_LEFT,
    SABER_RIGHT,
    UserData,
    VRObject,
    Wall,
)
from cybernoodles.bsor_bridge import load_bsor, validate_bsor, write_bsor
from cybernoodles.core.jump_timing import compute_spawn_ahead_beats
from cybernoodles.core.map_storage import slim_map_archive
from cybernoodles.core.network import ActorCritic, encode_cut_direction, STATIC_HEAD, NOTES_DIM
from cybernoodles.paths import existing_or_preferred_model_path
from cybernoodles.training.policy_checkpoint import extract_policy_state_dict
from cybernoodles.data.dataset_builder import MAPS_DIR, get_map_data, parse_beatmap_dat
from cybernoodles.envs import make_vector_env
from cybernoodles.training.policy_eval import sanitize_policy_actions

NUM_UPCOMING_NOTES = 20
NOTE_FEATURES = 8
FPS = 60.0


def sanitize_tensor(tensor, name="tensor"):
    """Detect and fix NaN/Inf values in tensors"""
    if torch.isnan(tensor).any():
        print(f"  WARNING: NaN detected in {name}, replacing with zeros")
        tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    if torch.isinf(tensor).any():
        print(f"  WARNING: Inf detected in {name}, replacing with zeros")
        tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor


def _blend_actions(new_actions, last_actions, smoothing_alpha):
    """Use smoothing_alpha as retention of the previous action."""
    smoothing_alpha = float(max(0.0, min(1.0, smoothing_alpha)))
    if last_actions is None or smoothing_alpha >= 0.999:
        return new_actions
    return (smoothing_alpha * last_actions) + ((1.0 - smoothing_alpha) * new_actions)


def auto_replay_envs(device):
    if not torch.cuda.is_available():
        return 32
    total_vram_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    if total_vram_gb >= 20:
        return 256
    if total_vram_gb >= 12:
        return 192
    if total_vram_gb >= 8:
        return 128
    return 96


def _create_vr_object_from_pose(pose_data, start=0, *, normalize=True):
    obj = VRObject()
    obj.x = float(pose_data[start + 0])
    obj.y = float(pose_data[start + 1])
    obj.z = float(pose_data[start + 2])

    rx = float(pose_data[start + 3])
    ry = float(pose_data[start + 4])
    rz = float(pose_data[start + 5])
    rw = float(pose_data[start + 6])
    if normalize:
        norm = np.sqrt(rx**2 + ry**2 + rz**2 + rw**2) + 1e-8
        rx /= norm
        ry /= norm
        rz /= norm
        rw /= norm

    obj.x_rot = float(rx)
    obj.y_rot = float(ry)
    obj.z_rot = float(rz)
    obj.w_rot = float(rw)
    return obj


def create_vr_object(pose_slice, *, normalize=True):
    return _create_vr_object_from_pose(pose_slice, 0, normalize=normalize)


def _normalize_recorded_pose_quaternions(recorded_poses):
    pose_buffer = np.array(recorded_poses, dtype=np.float32, copy=True)
    for quat_start in (3, 10, 17):
        quat = pose_buffer[:, quat_start:quat_start + 4]
        norms = np.sqrt(np.sum(quat * quat, axis=1, keepdims=True)) + 1e-8
        quat /= norms
    return pose_buffer


def _quat_from_axis_angle(axis, angle_radians):
    axis = np.asarray(axis, dtype=np.float32)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = axis / axis_norm
    half_angle = angle_radians * 0.5
    sin_half = float(np.sin(half_angle))
    return np.asarray([
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half,
        float(np.cos(half_angle)),
    ], dtype=np.float32)


def _append_ai_watermark_tail(bsor, start_time, duration_seconds=0.45):
    """Append an unmistakably synthetic post-song saber flourish."""
    if not bsor.frames:
        return

    tail_frames = max(12, int(round(duration_seconds * FPS)))
    base_head = bsor.frames[-1].head
    left_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    right_axis = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)

    for idx in range(tail_frames):
        t_sec = start_time + ((idx + 1) / FPS)
        angle = 14.0 * np.pi * ((idx + 1) / max(1, tail_frames))
        left_pose = np.concatenate((np.zeros(3, dtype=np.float32), _quat_from_axis_angle(left_axis, angle)))
        right_pose = np.concatenate((np.zeros(3, dtype=np.float32), _quat_from_axis_angle(right_axis, -angle * 1.4)))

        frame = Frame()
        frame.time = float(t_sec)
        frame.fps = int(FPS)
        frame.head = base_head
        frame.left_hand = create_vr_object(left_pose)
        frame.right_hand = create_vr_object(right_pose)
        bsor.frames.append(frame)


def _make_user_data_entry(key, payload):
    user_data = UserData()
    user_data.key = str(key)
    user_data.bytes = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return user_data


def _normalize_difficulty_name(value):
    return str(value or "").strip().lower().replace(" ", "")


def _select_primary_info_file(zip_names):
    candidates = [name for name in zip_names if name.lower().endswith("info.dat")]
    if not candidates:
        return None

    exact_matches = [
        name for name in candidates
        if os.path.basename(name).lower() == "info.dat"
    ]
    if exact_matches:
        return min(exact_matches, key=lambda name: (name.count("/"), len(name), name.lower()))

    non_bpm_matches = [
        name for name in candidates
        if os.path.basename(name).lower() != "bpminfo.dat"
    ]
    if non_bpm_matches:
        return min(non_bpm_matches, key=lambda name: (name.count("/"), len(name), name.lower()))

    return min(candidates, key=lambda name: (name.count("/"), len(name), name.lower()))


@lru_cache(maxsize=32)
def _robust_get_notes_cached(map_hash, diff_index, preferred_difficulty):
    """Fallback map parser for generate_replay.py to handle non-standard zip structures."""
    import zipfile, json

    # Try standard directory first
    zip_path = os.path.join(MAPS_DIR, f"{map_hash}.zip")
    print(f"Looking for zip at: {zip_path}")
    if not os.path.exists(zip_path):
        print(f"Direct path failed. Scanning maps folder for matches...")
        matches = [f for f in os.listdir(MAPS_DIR) if map_hash.lower() in f.lower()]
        zip_path = os.path.join(MAPS_DIR, matches[0]) if matches else None
        print(f"Scan found: {zip_path}")

    if not zip_path:
        print("Zip not found even after scan.")
        beatmap, bpm = get_map_data(map_hash)
        return beatmap, bpm

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Case-insensitive Info.dat
            info_files = [f for f in z.namelist() if f.lower().endswith('info.dat')]
            print(f"Info files found: {info_files}")
            if not info_files: return None, 120.0

            info_file = _select_primary_info_file(info_files)
            print(f"Selected info file: {info_file}")
            info = json.loads(z.read(info_file).decode('utf-8'))
            bpm = info.get('_beatsPerMinute', info.get('beatsPerMinute', 120.0))
            print(f"Map BPM: {bpm}")
            song_name = (
                info.get('_songName')
                or info.get('songName')
                or info.get('_songSubName')
                or info.get('songSubName')
                or "Generated Map"
            )
            level_author = (
                info.get('_levelAuthorName')
                or info.get('levelAuthorName')
                or info.get('_songAuthorName')
                or info.get('songAuthorName')
                or "Unknown Mapper"
            )
            environment_name = (
                info.get('_environmentName')
                or info.get('environmentName')
                or "DefaultEnvironment"
            )
            
            # Difficulty mapping
            dat_file = None
            dat_entry = None
            char_name = "Standard"
            diff_sets = info.get('_difficultyBeatmapSets', info.get('difficultyBeatmapSets', []))
            print(f"Found {len(diff_sets)} difficulty sets.")
            for s in diff_sets:
                char_name = s.get('_beatmapCharacteristicName', s.get('beatmapCharacteristicName', ''))
                print(f"  Checking characteristic: {char_name}")
                if char_name == 'Standard':
                    maps = s.get('_difficultyBeatmaps', s.get('difficultyBeatmaps', []))
                    if maps:
                        print(f"  Standard set has {len(maps)} difficulties.")
                        # Log available diffs
                        print("Available Difficulties:")
                        for idx, m in enumerate(maps):
                            d_name = m.get('_difficulty', m.get('difficulty', 'Unknown'))
                            print(f"  [{idx}] {d_name}")
                        
                        # Selection logic
                        if diff_index is None and preferred_difficulty:
                            wanted = _normalize_difficulty_name(preferred_difficulty)
                            target_idx = None
                            for idx, m in enumerate(maps):
                                d_name = m.get('_difficulty', m.get('difficulty', 'Unknown'))
                                if _normalize_difficulty_name(d_name) == wanted:
                                    target_idx = idx
                                    break
                            if target_idx is None:
                                target_idx = len(maps) - 1
                        elif diff_index is None:
                            target_idx = len(maps) - 1
                        else:
                            target_idx = diff_index if (0 <= diff_index < len(maps)) else (len(maps) - 1)
                            
                        dat_entry = maps[target_idx]
                        dat_file = dat_entry.get('_beatmapFilename', dat_entry.get('beatmapFilename'))
                        print(f"Selected Difficulty Index: {target_idx} -> {dat_file}")
            
            if not dat_file:
                print("Standard characteristic not found. Trying fallback to ANY .dat...")
                dats = [f for f in z.namelist() if f.endswith('.dat') and f.lower() != 'info.dat']
                if dats: 
                    dat_file = sorted(dats, key=lambda x: len(z.read(x)), reverse=True)[0]
                    print(f"Fallback selected biggest .dat: {dat_file}")

            if dat_file:
                beatmap = parse_beatmap_dat(z.read(dat_file))
                beatmap['mode'] = char_name or 'Standard'
                beatmap['difficulty'] = (
                    dat_entry.get('_difficulty', dat_entry.get('difficulty', 'ExpertPlus'))
                    if dat_entry else 'ExpertPlus'
                )
                beatmap['njs'] = float(
                    dat_entry.get('_noteJumpMovementSpeed', dat_entry.get('noteJumpMovementSpeed', 18.0))
                    if dat_entry else 18.0
                )
                beatmap['offset'] = float(
                    dat_entry.get('_noteJumpStartBeatOffset', dat_entry.get('noteJumpStartBeatOffset', 0.0))
                    if dat_entry else 0.0
                )
                beatmap['song_name'] = song_name
                beatmap['level_author_name'] = level_author
                beatmap['environment_name'] = environment_name
                return beatmap, bpm
            else:
                print("No playable .dat files found in zip.")
    except Exception as e:
        print(f"Error during zip extraction: {e}")
    return None, 120.0


def robust_get_notes(map_hash, diff_index=None, preferred_difficulty=None):
    normalized_hash = str(map_hash or "").strip().lower()
    normalized_diff = None if diff_index is None else int(diff_index)
    normalized_preferred = None if preferred_difficulty is None else str(preferred_difficulty)
    return _robust_get_notes_cached(normalized_hash, normalized_diff, normalized_preferred)


robust_get_notes.cache_clear = _robust_get_notes_cached.cache_clear
robust_get_notes.cache_info = _robust_get_notes_cached.cache_info


def resolve_generation_input(input_id):
    """Resolve a CLI input into (map_hash, preferred_difficulty)."""
    raw = str(input_id or "").strip()
    if not raw:
        raise ValueError("Empty replay generation input.")

    if os.path.exists(raw) and raw.lower().endswith(".bsor"):
        replay = load_bsor(raw)
        info = getattr(replay, "info", None)
        song_hash = str(getattr(info, "songHash", "") or "").strip()
        difficulty = str(getattr(info, "difficulty", "") or "").strip() or None
        if not song_hash:
            raise RuntimeError(f"Replay does not contain a song hash: {raw}")
        print(f"Resolved BSOR source -> hash {song_hash} | difficulty {difficulty or 'auto'}")
        return song_hash, difficulty

    if len(raw) < 10:
        resolved_hash = fetch_map_by_bsr(raw)
        if not resolved_hash:
            raise RuntimeError("BSR lookup failed.")
        return resolved_hash, None

    return raw, None


# ─────────────────────────────────────────────────────────────────────────────
# BSOR multiplier logic (standard Beat Saber combo multiplier)
# ─────────────────────────────────────────────────────────────────────────────

def _advance_multiplier(multiplier, progress):
    target = None
    if multiplier < 2:
        target = 2
    elif multiplier < 4:
        target = 4
    elif multiplier < 8:
        target = 8

    if target is None:
        return 8, 0

    progress += 1
    if progress >= target:
        return min(8, multiplier * 2), 0
    return multiplier, progress


def _break_multiplier(multiplier):
    return max(1, multiplier // 2)


def _map_saber_type(note_type, fallback=SABER_RIGHT):
    if note_type == 0:
        return SABER_LEFT
    if note_type == 1:
        return SABER_RIGHT
    return fallback


def _encode_note_id(note_info):
    return int(_note_identity(note_info)["note_id"])


def _note_id_from_components(scoring_type, line_index, line_layer, color_type, cut_direction):
    return (
        int(scoring_type) * 10000
        + int(line_index) * 1000
        + int(line_layer) * 100
        + int(color_type) * 10
        + int(cut_direction)
    )


def _note_identity(note_info):
    note_type = int(note_info.get('type', 0))
    if note_type == 3:
        scoring_type = NOTE_SCORE_TYPE_NOSCORE
    elif note_info.get('chainLink', False):
        scoring_type = NOTE_SCORE_TYPE_BURSTSLIDERELEMENT
    elif note_info.get('chainHead', False):
        scoring_type = NOTE_SCORE_TYPE_BURSTSLIDERHEAD
    elif note_info.get('arcHead', False):
        scoring_type = NOTE_SCORE_TYPE_SLIDERHEAD
    elif note_info.get('arcTail', False):
        scoring_type = NOTE_SCORE_TYPE_SLIDERTAIL
    else:
        scoring_type = NOTE_SCORE_TYPE_NORMAL_2

    line_index = int(note_info.get('lineIndex', 0))
    line_layer = int(note_info.get('lineLayer', 0))
    color_type = 3 if note_type == 3 else max(0, note_type)
    cut_direction = 9 if note_type == 3 else int(note_info.get('cutDirection', 8))
    return {
        "note_id": _note_id_from_components(scoring_type, line_index, line_layer, color_type, cut_direction),
        "scoringType": int(scoring_type),
        "lineIndex": int(line_index),
        "noteLineLayer": int(line_layer),
        "colorType": int(color_type),
        "cutDirection": int(cut_direction),
    }


def _apply_note_identity(note_obj, identity):
    note_obj.note_id = int(identity["note_id"])
    note_obj.scoringType = int(identity["scoringType"])
    note_obj.lineIndex = int(identity["lineIndex"])
    note_obj.noteLineLayer = int(identity["noteLineLayer"])
    note_obj.colorType = int(identity["colorType"])
    note_obj.cutDirection = int(identity["cutDirection"])
    return note_obj


def _default_note_info(note_index, event_time):
    return {
        'time': float(event_time * 2.0),
        'lineIndex': int(note_index % 4),
        'lineLayer': 1,
        'type': 0,
        'cutDirection': 8,
    }


def _score_cap_for_note(note_info):
    if int(note_info.get('type', 0)) == 3:
        return 0.0
    if note_info.get('chainLink', False):
        return 20.0
    if note_info.get('chainHead', False):
        return 85.0
    return float(note_info.get('scoreCap', 115.0))


def _make_cut_from_event(event, note_type):
    cut = Cut()
    cut.speedOK = bool(event.get('speed_ok', True))
    cut.directionOk = bool(event.get('direction_ok', True))
    cut.saberTypeOk = bool(event.get('saber_type_ok', True))
    cut.wasCutTooSoon = bool(event.get('was_cut_too_soon', False))
    cut.saberSpeed = float(event.get('saber_speed', 0.0))
    cut.saberDirection = [float(x) for x in event.get('saber_dir', [0.0, 0.0, 1.0])]
    used_saber_type = event.get('used_saber_type')
    if used_saber_type is None:
        used_saber_type = int(event.get('saber_type', 1))
    cut.saberType = _map_saber_type(int(used_saber_type), fallback=int(used_saber_type))
    cut.timeDeviation = float(event.get('time_deviation', 0.0))
    cut.cutDeviation = float(event.get('cut_deviation', 0.0))
    cut.cutPoint = [float(x) for x in event.get('cut_point', [0.0, 0.0, 0.0])]
    cut.cutNormal = [float(x) for x in event.get('cut_normal', [0.0, 1.0, 0.0])]
    cut.cutDistanceToCenter = float(event.get('cut_distance', 0.3))
    cut.cutAngle = float(event.get('cut_angle', 0.0))
    cut.beforeCutRating = float(event.get('before_cut_rating', 0.0))
    cut.afterCutRating = float(event.get('after_cut_rating', 0.0))
    return cut


def _events_are_time_sorted(events):
    last_time = float("-inf")
    for event in events:
        event_time = float(event.get('time', 0.0))
        if event_time < last_time:
            return False
        last_time = event_time
    return True


def _finite_float(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(result):
        return default
    return result


def _resolve_jump_distance_metadata(beatmap, bpm):
    default_njs = 18.0
    beatmap_njs = (
        _finite_float(beatmap.get('njs'), None)
        if isinstance(beatmap, dict)
        else None
    )
    fallback_njs = beatmap_njs if beatmap_njs and beatmap_njs > 0.0 else default_njs
    bpm_value = _finite_float(bpm, None)

    fallback = {
        "jump_distance": float(fallback_njs),
        "jump_distance_source": "fallback_njs",
        "jump_distance_fallback_reason": "missing_or_invalid_beatmap_timing",
    }
    if beatmap_njs is None or beatmap_njs <= 0.0 or bpm_value is None or bpm_value <= 0.0:
        return fallback

    jump_offset = _finite_float(beatmap.get('offset', 0.0), 0.0)
    spawn_ahead_beats = compute_spawn_ahead_beats(bpm_value, beatmap_njs, jump_offset)
    jump_duration = 2.0 * (spawn_ahead_beats / max(bpm_value / 60.0, 1e-6))
    jump_distance = beatmap_njs * jump_duration
    if not np.isfinite(jump_distance) or jump_distance <= 0.0:
        return fallback

    return {
        "jump_distance": float(jump_distance),
        "jump_distance_source": "beatmap_timing",
        "jump_distance_njs": float(beatmap_njs),
        "jump_distance_bpm": float(bpm_value),
        "jump_distance_offset": float(jump_offset),
        "jump_distance_spawn_ahead_beats": float(spawn_ahead_beats),
    }


def _format_bsor_validation_summary(validation):
    frame_summary = (
        f"frames {int(validation.get('frame_count', 0))} | "
        f"notes {int(validation.get('note_count', 0))} | "
        f"left span {float(validation.get('left_span', 0.0)):.3f} m | "
        f"right span {float(validation.get('right_span', 0.0)):.3f} m"
    )
    rust_error = validation.get("rust_validation_error")
    if rust_error:
        return (
            "WARNING: Rust BSOR validation failed; Python fallback parsed replay | "
            f"{frame_summary} | rust error: {rust_error}"
        )

    backend = validation.get("validation_backend")
    if backend == "rust":
        label = "BSOR Rust validation OK"
    elif backend == "python":
        label = (
            "BSOR Python validation OK (Rust unavailable)"
            if validation.get("rust_validation_skipped")
            else "BSOR Python validation OK"
        )
    else:
        label = "BSOR validation OK"
    return f"{label} | {frame_summary}"


# ─────────────────────────────────────────────────────────────────────────────
# GPU-only BSOR construction from tracked events
# ─────────────────────────────────────────────────────────────────────────────

def _build_bsor_from_events(
    events,
    recorded_poses,
    num_frames,
    map_hash,
    beatmap,
    bpm,
    *,
    fail_time=0.0,
    final_score=None,
    append_ai_watermark=False,
):
    """Build a BSOR object from GPU-tracked events and recorded poses.
    
    Args:
        events: List of per-note event dicts from GPU simulator's event tracker
        recorded_poses: numpy array [num_frames, 21] of champion poses
        num_frames: total frame count
        map_hash: map hash for BSOR metadata
    
    Returns:
        Bsor object ready to write
    """
    bsor = Bsor()
    bsor.magic_number = 0x442d3d69
    bsor.file_version = 1

    bsor.info = Info()
    bsor.info.version = "1.0.0"
    bsor.info.gameVersion = "1.40.8"
    bsor.info.timestamp = str(int(time.time()))
    bsor.info.playerId = "Cyber_Noodles"
    bsor.info.playerName = "CyberNoodles AI"
    bsor.info.platform = "AI"
    bsor.info.trackingSystem = "SyntheticReplay"
    bsor.info.hmd = "CyberNoodles"
    bsor.info.controller = "SyntheticSabers"
    bsor.info.songHash = map_hash.upper()
    bsor.info.songName = beatmap.get('song_name', 'Generated Map') if isinstance(beatmap, dict) else "Generated Map"
    bsor.info.mapper = beatmap.get('level_author_name', 'Unknown Mapper') if isinstance(beatmap, dict) else "Unknown Mapper"
    map_notes = beatmap.get('notes', []) if isinstance(beatmap, dict) else beatmap
    map_obstacles = beatmap.get('obstacles', []) if isinstance(beatmap, dict) else []
    bsor.info.difficulty = beatmap.get('difficulty', 'ExpertPlus') if isinstance(beatmap, dict) else "ExpertPlus"
    bsor.info.mode = beatmap.get('mode', 'Standard') if isinstance(beatmap, dict) else "Standard"
    bsor.info.environment = beatmap.get('environment_name', 'DefaultEnvironment') if isinstance(beatmap, dict) else "DefaultEnvironment"
    bsor.info.modifiers = ""
    jump_distance_meta = _resolve_jump_distance_metadata(beatmap, bpm)
    bsor.info.jumpDistance = float(jump_distance_meta["jump_distance"])
    bsor.info.leftHanded = False
    bsor.info.height = 1.7
    bsor.info.startTime = 0.0
    bsor.info.failTime = float(max(0.0, fail_time))
    bsor.info.speed = 1.0

    bsor.frames = []
    bsor.notes = []
    bsor.walls = []
    height = Height()
    height.height = float(bsor.info.height)
    height.time = 0.0
    bsor.heights = [height]
    bsor.pauses = []
    neutral_offset = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    bsor.controller_offsets = ControllerOffsets()
    bsor.controller_offsets.left = create_vr_object(neutral_offset)
    bsor.controller_offsets.right = create_vr_object(neutral_offset)
    bsor.user_data = []

    # Build frames from recorded poses
    fps_value = int(FPS)
    normalized_poses = _normalize_recorded_pose_quaternions(recorded_poses[:num_frames])
    frames = [None] * num_frames
    frame_cls = Frame
    create_vr_from_pose = _create_vr_object_from_pose
    for f, pose in enumerate(normalized_poses):
        frame = frame_cls()
        frame.time = (f + 1) / FPS
        frame.fps = fps_value
        frame.head = create_vr_from_pose(pose, 0, normalize=False)
        frame.left_hand = create_vr_from_pose(pose, 7, normalize=False)
        frame.right_hand = create_vr_from_pose(pose, 14, normalize=False)
        frames[f] = frame
    bsor.frames = frames

    total_score = 0 if final_score is None else int(final_score)
    max_score = 0
    multiplier = 1
    progress = 0
    combo = 0
    max_combo = 0
    hit_count = 0
    bad_cut_count = 0
    miss_count = 0
    bomb_count = 0
    wall_count = 0

    for note_info in map_notes:
        if int(note_info.get('type', 0)) == 3:
            continue
        multiplier, progress = _advance_multiplier(multiplier, progress)
        max_score += int(round(_score_cap_for_note(note_info) * multiplier))

    multiplier = 1
    progress = 0

    # GPU tracking currently emits events in order; avoid a redundant sort on the hot path.
    events_sorted = events if _events_are_time_sorted(events) else sorted(events, key=lambda e: e['time'])
    note_time_scale = max(bpm / 60.0, 1e-6)

    for event in events_sorted:
        note_index = int(event.get('note_index', 0))
        note_info = map_notes[note_index] if 0 <= note_index < len(map_notes) else _default_note_info(note_index, event['time'])
        note_identity = _note_identity(note_info)
        note_time_sec = float(note_info.get('time', event['time'] * note_time_scale)) / note_time_scale
        spawn_time = max(0.0, note_time_sec - 1.0)
        note_type = int(note_info.get('type', 0))

        if event['type'] == 'hit':
            multiplier, progress = _advance_multiplier(multiplier, progress)
            combo += 1
            max_combo = max(max_combo, combo)
            hit_count += 1

            n_obj = Note()
            _apply_note_identity(n_obj, note_identity)
            n_obj.event_time = event['time']
            n_obj.spawn_time = spawn_time
            n_obj.event_type = NOTE_EVENT_GOOD

            event_data = dict(event)
            event_data['before_cut_rating'] = float(event_data.get('pre_score', 0.0)) / 70.0
            event_data['after_cut_rating'] = float(event_data.get('post_score', 0.0)) / 30.0
            event_data['speed_ok'] = True
            event_data['direction_ok'] = True
            event_data['saber_type_ok'] = bool(event_data.get('saber_type_ok', True))
            cut = _make_cut_from_event(event_data, note_type)
            note_score = int(round(
                float(event_data.get('pre_score', 0.0))
                + float(event_data.get('post_score', 0.0))
                + float(event_data.get('acc_score', 0.0))
            ))
            if final_score is None:
                total_score += note_score * multiplier
            n_obj.cut = cut
            n_obj.pre_score = int(round(float(event_data.get('pre_score', 0.0))))
            n_obj.post_score = int(round(float(event_data.get('post_score', 0.0))))
            n_obj.acc_score = int(round(float(event_data.get('acc_score', 0.0))))
            n_obj.score = int(note_score)
            bsor.notes.append(n_obj)

        elif event['type'] == 'bad':
            combo = 0
            bad_cut_count += 1
            n_obj = Note()
            _apply_note_identity(n_obj, note_identity)
            n_obj.event_time = event['time']
            n_obj.spawn_time = spawn_time
            n_obj.event_type = NOTE_EVENT_BAD
            n_obj.cut = _make_cut_from_event(event, note_type)
            bsor.notes.append(n_obj)
            multiplier = _break_multiplier(multiplier)
            progress = 0

        elif event['type'] == 'miss':
            combo = 0
            miss_count += 1
            n_obj = Note()
            _apply_note_identity(n_obj, note_identity)
            n_obj.event_time = event['time']
            n_obj.spawn_time = spawn_time
            n_obj.event_type = NOTE_EVENT_MISS
            n_obj.cut = None
            bsor.notes.append(n_obj)
            multiplier = _break_multiplier(multiplier)
            progress = 0

        elif event['type'] == 'bomb':
            combo = 0
            bomb_count += 1
            n_obj = Note()
            _apply_note_identity(n_obj, note_identity)
            n_obj.event_time = event['time']
            n_obj.spawn_time = spawn_time
            n_obj.event_type = NOTE_EVENT_BOMB
            n_obj.cut = None
            bsor.notes.append(n_obj)
            multiplier = _break_multiplier(multiplier)
            progress = 0

        elif event['type'] == 'wall':
            combo = 0
            wall_count += 1
            obs_idx = int(event.get('obstacle_index', 0))
            obstacle_info = map_obstacles[obs_idx] if 0 <= obs_idx < len(map_obstacles) else {}
            wall = Wall()
            wall.id = obs_idx
            wall.energy = float(event.get('energy', 0.0))
            wall.time = float(event.get('time', 0.0))
            wall_note_time = float(obstacle_info.get('time', wall.time * note_time_scale)) / note_time_scale
            wall.spawnTime = max(0.0, wall_note_time - 1.0)
            bsor.walls.append(wall)
            multiplier = _break_multiplier(multiplier)
            progress = 0

    if append_ai_watermark:
        final_frame_time = bsor.frames[-1].time if bsor.frames else max(0.0, float(num_frames - 1) / FPS)
        _append_ai_watermark_tail(bsor, final_frame_time)

    accuracy_percent = (total_score / max_score) * 100.0 if max_score > 0 else 0.0
    replay_stats = {
        "score": int(total_score),
        "max_score": int(max_score),
        "accuracy_percent": float(accuracy_percent),
        "max_combo": int(max_combo),
        "hit_count": int(hit_count),
        "bad_cut_count": int(bad_cut_count),
        "miss_count": int(miss_count),
        "bomb_count": int(bomb_count),
        "wall_count": int(wall_count),
        "watermarked": bool(append_ai_watermark),
        "frame_count": int(len(bsor.frames)),
        "note_event_count": int(len(bsor.notes)),
    }
    replay_stats.update(jump_distance_meta)
    bsor.user_data.append(_make_user_data_entry("cybernoodles:metadata", replay_stats))
    bsor.info.score = int(total_score)
    return bsor, total_score, max_score, replay_stats


def _choose_replay_champion(sim_gpu):
    scores = np.nan_to_num(sim_gpu.total_scores.detach().cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    hits = np.nan_to_num(sim_gpu.total_hits.detach().cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    speed_samples = np.nan_to_num(sim_gpu.speed_samples.clamp(min=1.0).detach().cpu().numpy(), nan=1.0, posinf=1.0, neginf=1.0)
    style_violation = np.nan_to_num((sim_gpu.speed_violation_sum / sim_gpu.speed_samples.clamp(min=1.0)).detach().cpu().numpy(), nan=1.0, posinf=1.0, neginf=1.0)
    angular_violation = np.nan_to_num((sim_gpu.angular_violation_sum / sim_gpu.speed_samples.clamp(min=1.0)).detach().cpu().numpy(), nan=1.0, posinf=1.0, neginf=1.0)
    waste_motion = np.nan_to_num((sim_gpu.waste_motion_sum / sim_gpu.speed_samples.clamp(min=1.0)).detach().cpu().numpy(), nan=1.0, posinf=1.0, neginf=1.0)
    idle_motion = np.nan_to_num((sim_gpu.idle_motion_sum / sim_gpu.speed_samples.clamp(min=1.0)).detach().cpu().numpy(), nan=1.0, posinf=1.0, neginf=1.0)
    guard_error = np.nan_to_num((sim_gpu.guard_error_sum / sim_gpu.speed_samples.clamp(min=1.0)).detach().cpu().numpy(), nan=1.0, posinf=1.0, neginf=1.0)
    oscillation = np.nan_to_num((sim_gpu.oscillation_sum / sim_gpu.speed_samples.clamp(min=1.0)).detach().cpu().numpy(), nan=1.0, posinf=1.0, neginf=1.0)
    lateral_motion = np.nan_to_num((sim_gpu.lateral_motion_sum / sim_gpu.speed_samples.clamp(min=1.0)).detach().cpu().numpy(), nan=1.0, posinf=1.0, neginf=1.0)
    motion_efficiency = np.nan_to_num((
        sim_gpu.useful_progress / sim_gpu.motion_path.clamp(min=1e-6)
    ).clamp(0.0, 1.0).detach().cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)

    top_score = float(scores.max()) if scores.size else 0.0
    top_hits = float(hits.max()) if hits.size else 0.0
    if top_score <= 0.0:
        return int(np.argmax(scores)), "score-only fallback"

    candidate_mask = scores >= (top_score * 0.92)
    if top_hits > 0.0:
        candidate_mask &= hits >= max(1.0, top_hits * 0.80)
    candidate_indices = np.where(candidate_mask)[0]
    if candidate_indices.size == 0:
        candidate_indices = np.array([int(np.argmax(scores))], dtype=np.int64)

    normalized_score = scores / max(top_score, 1e-6)
    normalized_hits = hits / max(top_hits, 1e-6) if top_hits > 0.0 else np.zeros_like(hits)
    replay_rank = (
        1.00 * normalized_score
        + 0.08 * normalized_hits
        + 0.18 * motion_efficiency
        - 0.12 * style_violation
        - 0.08 * angular_violation
        - 0.22 * waste_motion
        - 0.18 * idle_motion
        - 0.12 * guard_error
        - 0.16 * oscillation
        - 0.12 * lateral_motion
    )
    best_local = int(candidate_indices[np.argmax(replay_rank[candidate_indices])])
    return best_local, f"disciplined-top-{candidate_indices.size}"


# ─────────────────────────────────────────────────────────────────────────────
# Main replay generation  (GPU-only — no CPU simulator)
# ─────────────────────────────────────────────────────────────────────────────

def generate_bsor(map_hash, output_file="CyberNoodles_Replay.bsor", diff_index=None,
                  model_path=None, model=None, device=None,
                  num_envs=None, noise_scale=0.0, action_repeat=1, smoothing_alpha=1.0,
                  fail_enabled=False, survival_level=0.0, assist_level=0.0,
                  training_wheels=0.0, max_duration_seconds=None, validate_output=True,
                  preferred_difficulty=None):
    """Generate a .bsor replay file for a given map.
    
    Args:
        map_hash: Map hash string
        output_file: Output .bsor file path
        diff_index: Difficulty index (None = interactive or auto-highest)
        model_path: Path to model weights (used if model is None)
        model: Pre-loaded ActorCritic model (skips disk load when provided)
        device: torch device (inferred if None)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not model_path:
        model_path = existing_or_preferred_model_path("rl_model")

    if model is None:
        print(f"Loading model on {device}...")
        model = ActorCritic().to(device)
        if os.path.exists(model_path):
            payload = torch.load(model_path, map_location=device, weights_only=False)
            state_dict = extract_policy_state_dict(
                payload,
                checkpoint_path=model_path,
                accepted_keys=("model_state_dict", "actor_state_dict"),
                allow_legacy=True,
            )
            model.load_state_dict(state_dict, strict=False)
        else:
            print("Warning: No trained model found! Using random untrained weights.")
    model.eval()

    print(f"Parsing Map {map_hash}...")
    beatmap, bpm = robust_get_notes(map_hash, diff_index, preferred_difficulty=preferred_difficulty)
    notes = beatmap.get('notes', []) if beatmap else []
    obstacles = beatmap.get('obstacles', []) if beatmap else []
    if not beatmap:
        print(f"Error: Could not parse map notes for {map_hash}. Check your data/maps folder.")
        return
    if len(notes) == 0:
        print("No notes found in the map.")
        return

    bps = bpm / 60.0
    last_note = max((note['time'] for note in notes), default=0.0)
    last_obstacle = max((obs['time'] + obs.get('duration', 0.0) for obs in obstacles), default=0.0)
    duration_sec = (max(last_note, last_obstacle) / bps) + 3.0
    if max_duration_seconds is not None:
        clipped_duration = min(duration_sec, float(max_duration_seconds))
        if clipped_duration < duration_sec - 1e-6:
            print(f"Replay clip mode: limiting simulation to {clipped_duration:.1f}s (full map {duration_sec:.1f}s).")
        duration_sec = clipped_duration
    num_frames = int(duration_sec * FPS) + 90  # Buffer for follow-through

    # ── GPU Parallel Championship with Event Tracking ─────────────────────────
    NUM_ENVS = num_envs or auto_replay_envs(device)
    print(f"Tournament mode: Simulating {NUM_ENVS} parallel replays (GPU-only)...")
    vector_env = make_vector_env(
        num_envs=NUM_ENVS,
        device=device,
        penalty_weights=(0.5, 0.0, 0.0, 0.0),
        dense_reward_scale=0.0,
        training_wheels=training_wheels,
        rehab_assists=assist_level,
        survival_assistance=survival_level,
        stability_assistance=0.0,
        style_guidance_level=0.0,
        fail_enabled=fail_enabled,
        saber_inertia=0.0,
        rot_clamp=0.07,
        pos_clamp=0.12,
    )
    vector_env.load_maps([beatmap] * NUM_ENVS, [bpm] * NUM_ENVS)
    sim_gpu = vector_env.simulator
    sim_gpu.reset()

    # Enable per-note event tracking for BSOR construction
    sim_gpu.enable_event_tracking()

    # Record poses on the simulator device, then copy only the winning path to
    # CPU at the end. This avoids stale host-side frames during long runs.
    recorded_poses = torch.zeros((num_frames, NUM_ENVS, 21), device=device)

    t_start = time.time()
    actual_frames = num_frames
    with torch.no_grad():
        actions = None
        last_actions = None
        for f in range(num_frames):
            if f % action_repeat == 0 or actions is None:
                state = sim_gpu.get_states()
                
                # ADD: State sanitization
                state = sanitize_tensor(state, "state input")
                
                mean, std, _ = model(state)
                
                # ADD: Model output sanitization
                mean = sanitize_tensor(mean, "mean")
                std = sanitize_tensor(std, "std")
                
                # ADD: Clamp extreme values
                mean = torch.clamp(mean, -10.0, 10.0)
                std = torch.clamp(std, 1e-6, 5.0)  # Ensure positive std
                
                # Default to deterministic mean actions for showcase replays.
                if noise_scale <= 0.0:
                    new_actions = mean
                else:
                    new_actions = mean + torch.randn_like(mean) * std * noise_scale

                # Apply a low-pass filter while keeping 1.0 as an explicit "off" value.
                actions = sanitize_policy_actions(_blend_actions(new_actions, last_actions, smoothing_alpha))
                last_actions = actions.clone()
                
                # ADD: Action sanitization
                actions = sanitize_tensor(actions, "actions")
                actions = sanitize_policy_actions(actions)

            sim_gpu.step(actions, dt=1.0 / FPS)
            recorded_poses[f].copy_(sim_gpu.poses)

            if bool(sim_gpu.episode_done.all().item()):
                actual_frames = f + 1
                break

            # ADD: Score validation during simulation
            if f % int(FPS * 5) == 0:
                score_snapshot = sim_gpu.total_scores
                bad_mask = (~torch.isfinite(score_snapshot)) | (score_snapshot < 0.0) | (score_snapshot >= 1e10)
                if bool(bad_mask.any().item()):
                    print(f"  WARNING: Invalid score detected in {int(bad_mask.sum().item())} env(s); zeroing only those candidates.")
                    sim_gpu.total_scores[bad_mask] = 0.0
                    sim_gpu.total_hits[bad_mask] = 0.0
                max_score = float(sim_gpu.total_scores.max().item())
                 
                print(f"  Simulating... {f / num_frames * 100:.1f}% | "
                      f"Highest Score: {max_score:.0f}")

    if device.type == 'cuda':
        torch.cuda.synchronize()
    print(f"Tournament complete in {time.time() - t_start:.2f}s.")

    # Identify the Champion
    champion_idx, champion_strategy = _choose_replay_champion(sim_gpu)
    champion_score = sim_gpu.total_scores[champion_idx].item()
    champion_hits = sim_gpu.total_hits[champion_idx].item()
    print(f"\033[92mChampion found! Env #{champion_idx} scored "
          f"{int(champion_score)} with {int(champion_hits)} hits "
          f"({champion_strategy}).\033[0m")

    # Extract the winner's path and events
    champion_end_time = float(sim_gpu.current_times[champion_idx].item())
    champion_end_frame = max(1, min(actual_frames, int(np.ceil(champion_end_time * FPS)) + 2))
    best_path = recorded_poses[:champion_end_frame, champion_idx].detach().float().cpu().numpy()
    champion_events = sim_gpu.tracked_events[champion_idx]
    champion_reason = int(sim_gpu._terminal_reason[champion_idx].item()) if hasattr(sim_gpu, '_terminal_reason') else 0
    champion_completion = float(sim_gpu._completion_ratio[champion_idx].item()) if hasattr(sim_gpu, '_completion_ratio') else 0.0
    fail_time = champion_end_time if champion_reason == 1 else 0.0
    if champion_end_frame > 1:
        left_span = np.ptp(best_path[:, 7:10], axis=0)
        right_span = np.ptp(best_path[:, 14:17], axis=0)
        print(f"Replay motion span | left {float(np.linalg.norm(left_span)):.3f} m | "
              f"right {float(np.linalg.norm(right_span)):.3f} m")

    # Clean up simulator
    sim_gpu.disable_event_tracking()

    # ── Build BSOR from GPU-tracked events ────────────────────────────────────
    print(f"Building BSOR from {len(champion_events)} tracked events...")
    watermark_tail = champion_reason == 2 or champion_completion >= 0.995
    bsor, total_score, max_score, replay_stats = _build_bsor_from_events(
        champion_events,
        best_path,
        champion_end_frame,
        map_hash,
        beatmap,
        bpm,
        fail_time=fail_time,
        final_score=champion_score,
        append_ai_watermark=watermark_tail,
    )

    final_acc = (total_score / max_score) if max_score > 0 else 0.0
    print(f"Generated {len(bsor.notes)} note events. "
          f"Total score: {int(total_score)} | Accuracy: {final_acc:.2%}")
    reason_text = {
        0: "unknown",
        1: "failed",
        2: "song cleared",
        3: "timed out",
    }.get(champion_reason, f"reason-{champion_reason}")
    print(f"Replay end: {reason_text} | completion: {champion_completion:.1%} | "
          f"frames: {champion_end_frame}")
    print(
        f"Replay metadata: max combo {replay_stats['max_combo']} | "
        f"hits {replay_stats['hit_count']} | bad {replay_stats['bad_cut_count']} | "
        f"miss {replay_stats['miss_count']}"
    )
    if watermark_tail:
        print("  AI watermark tail appended after the final playable frame.")
    if champion_reason == 1 and fail_enabled:
        print("  Tip: This replay ended because the selected env failed. "
              "Use --no-fail for a full-map showcase replay.")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"Saving to {output_file}...")
    if output_file.lower().endswith(".pth") or "model" in output_file.lower():
        print(f"❌ ERROR: Safety system blocked attempt to write replay over "
              f"a model file ({output_file}).")
        return

    try:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        write_bsor(bsor, output_file)
        print("Success!")
        if validate_output:
            try:
                validation = validate_bsor(output_file)
                print(_format_bsor_validation_summary(validation))
            except Exception as parse_exc:
                print(f"WARNING: Replay file was written but failed validation parse: {parse_exc}")
    except Exception as e:
        print(f"Failed to save BSOR file: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# BeatSaver BSR utilities
# ─────────────────────────────────────────────────────────────────────────────

def fetch_map_by_bsr(bsr_code):
    """Fetches map metadata and downloads zip from BeatSaver using BSR key."""
    print(f"Fetching metadata for BSR: {bsr_code}...")
    try:
        response = requests.get(f"https://api.beatsaver.com/maps/id/{bsr_code}", timeout=10)
        if response.status_code != 200:
            print(f"Error: BeatSaver returned status {response.status_code}")
            return None
        
        data = response.json()
        latest = data['versions'][0]
        map_hash = latest['hash']
        download_url = latest['downloadURL']
        
        zip_path = os.path.join(MAPS_DIR, f"{map_hash}.zip")
        if os.path.exists(zip_path):
            slim_map_archive(zip_path)
            print(f"Map {map_hash} already exists locally.")
            return map_hash

        print(f"Downloading map {map_hash}...")
        r = requests.get(download_url, stream=True)
        if r.status_code == 200:
            os.makedirs(MAPS_DIR, exist_ok=True)
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            slim_map_archive(zip_path)
            print("Download complete.")
            return map_hash
        else:
            print(f"Failed to download zip: {r.status_code}")
    except Exception as e:
        print(f"Error fetching from BeatSaver: {e}")
    return None


def fetch_random_ranked_bsr():
    """Fetch a random BeatLeader-ranked map from BeatSaver.
    
    Returns:
        (bsr_code, map_hash) tuple, or (None, None) on failure.
        Downloads the map zip if not already cached.
    """
    page = random.randint(0, 50)
    try:
        url = (f"https://api.beatsaver.com/search/text/{page}"
               f"?ranked=true&sortOrder=Relevance&pageSize=20")
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            print(f"  BeatSaver API returned {resp.status_code}")
            return None, None

        docs = resp.json().get('docs', [])
        random.shuffle(docs)

        for m in docs:
            versions = m.get('versions', [])
            if not versions:
                continue

            bsr_code = m.get('id', '')
            map_hash = versions[0].get('hash', '')
            download_url = versions[0].get('downloadURL', '')

            if not bsr_code or not map_hash or not download_url:
                continue

            # Download if not cached
            zip_path = os.path.join(MAPS_DIR, f"{map_hash}.zip")
            if not os.path.exists(zip_path):
                try:
                    r = requests.get(download_url, timeout=30)
                    if r.status_code == 200:
                        os.makedirs(MAPS_DIR, exist_ok=True)
                        with open(zip_path, 'wb') as f:
                            f.write(r.content)
                        slim_map_archive(zip_path)
                    else:
                        continue
                except Exception:
                    continue
            else:
                slim_map_archive(zip_path)

            song_name = m.get('metadata', {}).get('songName', 'Unknown')
            print(f"  Selected map: {song_name} (BSR: {bsr_code})")
            return bsr_code, map_hash

    except Exception as e:
        print(f"  Error fetching ranked map: {e}")
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Progress replay generation  (called from training loop)
# ─────────────────────────────────────────────────────────────────────────────

PROGRESS_DIR = "Progress Replays"

def generate_progress_replay(model, device, epoch, map_hash=None, tag=None, num_envs=None, fetch_remote=True):
    """Generate a progress replay on a fixed benchmark map or remote sample.
    
    Called automatically every N epochs during training.
    Uses a provided local benchmark map when available; otherwise it can
    optionally fetch a remote ranked map, generate a replay with the current
    model state, and save it to the Progress Replays folder.
    
    This function is fail-safe — any exception is caught and logged,
    never crashing the training loop.
    
    Args:
        model: The live ActorCritic model (in-memory, no disk I/O)
        device: torch device
        epoch: Current epoch number (for filename)
    """
    try:
        print(f"\033[94m  [Progress Replay] Generating replay for epoch {epoch}...\033[0m")
        os.makedirs(PROGRESS_DIR, exist_ok=True)

        replay_tag = tag
        if map_hash is None:
            if not fetch_remote:
                print(f"\033[93m  [Progress Replay] No benchmark map provided. Skipping.\033[0m")
                return
            bsr_code, map_hash = fetch_random_ranked_bsr()
            if not bsr_code or not map_hash:
                print(f"\033[93m  [Progress Replay] Could not fetch a ranked map. Skipping.\033[0m")
                return
            replay_tag = replay_tag or bsr_code
        else:
            replay_tag = replay_tag or map_hash[:8]

        output_file = os.path.join(PROGRESS_DIR, f"Cybernoodles_{epoch}_{replay_tag}.bsor")

        # Skip if this exact replay already exists (e.g. resumed training)
        if os.path.exists(output_file):
            print(f"\033[90m  [Progress Replay] {output_file} already exists. Skipping.\033[0m")
            return

        env_count = int(num_envs) if num_envs else auto_replay_envs(device)
        print(
            f"\033[90m  [Progress Replay] Lightweight mode: {env_count} envs | "
            f"clip 75s | validation off\033[0m"
        )
        generate_bsor(
            map_hash,
            output_file=output_file,
            diff_index=-1,  # Auto-select hardest difficulty
            model=model,
            device=device,
            num_envs=num_envs,
            noise_scale=0.0,
            action_repeat=1,
            smoothing_alpha=1.0,
            fail_enabled=False,
            survival_level=0.0,
            assist_level=0.0,
            training_wheels=0.0,
            max_duration_seconds=75.0,
            validate_output=False,
        )
        print(f"\033[92m  [Progress Replay] Saved: {output_file}\033[0m")

    except Exception as e:
        print(f"\033[91m  [Progress Replay] Failed (non-fatal): {e}\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a BSOR replay from a model using a BSR code, map hash, or source BSOR."
    )
    parser.add_argument("input", help="BSR code, map hash, or a source .bsor path to copy hash/difficulty from.")
    parser.add_argument("model", nargs="?", default=existing_or_preferred_model_path("rl_model"), help="Model checkpoint path.")
    parser.add_argument("output", nargs="?", default="CyberNoodles_Replay.bsor", help="Output .bsor path.")
    parser.add_argument("diff_idx", nargs="?", type=int, default=None, help="Optional explicit difficulty index.")
    parser.add_argument("--num-envs", type=int, default=None, help="How many parallel envs to simulate.")
    parser.add_argument("--fail", dest="fail_enabled", action="store_true", help="Allow the replay to fail naturally.")
    parser.add_argument("--no-fail", dest="fail_enabled", action="store_false", help="Force the replay to run the full map.")
    parser.add_argument("--noise-scale", type=float, default=0.0, help="Exploration noise scale.")
    parser.add_argument("--smoothing-alpha", type=float, default=1.0, help="Action smoothing retention.")
    parser.set_defaults(fail_enabled=False)
    args = parser.parse_args()

    try:
        hash_id, preferred_difficulty = resolve_generation_input(args.input)
    except Exception as exc:
        print(f"Aborting: {exc}")
        sys.exit(1)

    print(
        f"\033[94mRunning generation:\033[0m "
        f"Map={hash_id}, Model={args.model}, SaveTo={args.output}, Envs={args.num_envs or 'auto'}"
    )
    generate_bsor(
        hash_id,
        args.output,
        diff_index=args.diff_idx,
        model_path=args.model,
        noise_scale=float(args.noise_scale),
        smoothing_alpha=float(args.smoothing_alpha),
        fail_enabled=bool(args.fail_enabled),
        num_envs=args.num_envs,
        preferred_difficulty=preferred_difficulty,
    )
