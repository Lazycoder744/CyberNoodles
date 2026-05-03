import argparse
import json
import os

import numpy as np
from bsor.Bsor import (
    NOTE_SCORE_TYPE_BURSTSLIDERELEMENT,
    NOTE_SCORE_TYPE_BURSTSLIDERHEAD,
    NOTE_SCORE_TYPE_NOSCORE,
    NOTE_SCORE_TYPE_NORMAL_2,
    NOTE_SCORE_TYPE_SLIDERHEAD,
    NOTE_SCORE_TYPE_SLIDERTAIL,
)

from cybernoodles.bsor_bridge import load_bsor
from cybernoodles.data.dataset_builder import MAPS_DIR, REPLAYS_DIR, get_map_notes, parse_bsor

CALIBRATION_PATH = "sim_calibration.json"
CALIBRATION_VERSION = 3
DEFAULT_CALIBRATION = {
    "version": CALIBRATION_VERSION,
    "x_offset": -0.90,
    "x_spacing": 0.60,
    "y_offset": 0.85,
    "y_spacing": 0.35,
    "saber_axis": [0.0, 0.0, 1.0],
    "saber_axis_label": "+z",
    "saber_origin": [0.0, 0.05, -0.5],
    "saber_length": 1.0,
    "source": "defaults",
}


def _require_json_object(value, path):
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _require_finite_float(payload, key, *, min_value=None, max_value=None):
    if key not in payload:
        raise ValueError(f"missing required calibration field: {key}")
    try:
        value = float(payload[key])
    except (TypeError, ValueError):
        raise ValueError(f"calibration field {key} must be numeric")
    if not np.isfinite(value):
        raise ValueError(f"calibration field {key} must be finite")
    if min_value is not None and value < min_value:
        raise ValueError(f"calibration field {key} below minimum {min_value}: {value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"calibration field {key} above maximum {max_value}: {value}")
    return value


def _require_finite_vector(payload, key, length, *, max_abs=None, min_norm=None, max_norm=None):
    if key not in payload:
        raise ValueError(f"missing required calibration field: {key}")
    value = payload[key]
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"calibration field {key} must be a {length}-element list")
    vector = []
    for item in value:
        try:
            item_value = float(item)
        except (TypeError, ValueError):
            raise ValueError(f"calibration field {key} must contain numeric values")
        if not np.isfinite(item_value):
            raise ValueError(f"calibration field {key} must contain finite values")
        if max_abs is not None and abs(item_value) > max_abs:
            raise ValueError(f"calibration field {key} component out of range: {item_value}")
        vector.append(item_value)

    norm = float(np.linalg.norm(np.asarray(vector, dtype=np.float32)))
    if min_norm is not None and norm < min_norm:
        raise ValueError(f"calibration field {key} norm below minimum {min_norm}: {norm}")
    if max_norm is not None and norm > max_norm:
        raise ValueError(f"calibration field {key} norm above maximum {max_norm}: {norm}")
    return vector


def _validate_simulator_calibration_payload(loaded, path):
    loaded = _require_json_object(loaded, path)
    if "version" not in loaded:
        raise ValueError("missing required calibration field: version")
    try:
        version = int(loaded["version"])
    except (TypeError, ValueError):
        raise ValueError("calibration field version must be an integer")
    if version != CALIBRATION_VERSION:
        raise ValueError(f"unsupported simulator calibration version {version}; expected {CALIBRATION_VERSION}")

    axis = _require_finite_vector(
        loaded,
        "saber_axis",
        3,
        max_abs=1.5,
        min_norm=0.5,
        max_norm=1.5,
    )
    origin = _require_finite_vector(
        loaded,
        "saber_origin",
        3,
        max_abs=2.0,
    )

    source = loaded.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("calibration field source must be a non-empty string")

    calibration = DEFAULT_CALIBRATION.copy()
    calibration.update({
        "version": version,
        "x_offset": _require_finite_float(loaded, "x_offset", min_value=-2.0, max_value=2.0),
        "x_spacing": _require_finite_float(loaded, "x_spacing", min_value=0.40, max_value=0.80),
        "y_offset": _require_finite_float(loaded, "y_offset", min_value=0.0, max_value=2.0),
        "y_spacing": _require_finite_float(loaded, "y_spacing", min_value=0.22, max_value=0.60),
        "saber_axis": axis,
        "saber_axis_label": str(loaded.get("saber_axis_label", calibration["saber_axis_label"])),
        "saber_origin": origin,
        "saber_length": _require_finite_float(loaded, "saber_length", min_value=0.55, max_value=1.30),
        "source": source,
    })

    for key in ("fit_error", "grid_fit_error", "local_fit_error", "axis_fit_error"):
        if key in loaded:
            calibration[key] = _require_finite_float(loaded, key, min_value=0.0)
    for key in ("replays_used", "notes_used"):
        if key in loaded:
            try:
                count = int(loaded[key])
            except (TypeError, ValueError):
                raise ValueError(f"calibration field {key} must be an integer")
            if count < 0:
                raise ValueError(f"calibration field {key} must be non-negative")
            calibration[key] = count
    for key in ("lane_counts", "layer_counts", "maps_dir"):
        if key in loaded:
            calibration[key] = loaded[key]

    return calibration



def load_simulator_calibration(path=CALIBRATION_PATH):
    calibration = DEFAULT_CALIBRATION.copy()
    if not os.path.exists(path):
        return calibration

    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        return _validate_simulator_calibration_payload(loaded, path)
    except Exception as exc:
        raise ValueError(f"Invalid simulator calibration {path}: {exc}") from exc


def save_simulator_calibration(calibration, path=CALIBRATION_PATH):
    payload = DEFAULT_CALIBRATION.copy()
    payload.update(calibration)
    payload["version"] = CALIBRATION_VERSION
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _normalize_quaternions(quat):
    quat = np.asarray(quat, dtype=np.float32)
    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    return quat / norms


def _rotate_axis(quat, axis):
    quat = _normalize_quaternions(quat)
    q_xyz = quat[:, :3]
    q_w = quat[:, 3:4]
    axis_vec = np.broadcast_to(axis.reshape(1, 3), q_xyz.shape)
    t = 2.0 * np.cross(q_xyz, axis_vec)
    return axis_vec + q_w * t + np.cross(q_xyz, t)


def _rotate_world_to_local(quat, vec):
    quat = _normalize_quaternions(quat)
    q_xyz = quat[:, :3]
    q_w = quat[:, 3:4]
    vec = np.asarray(vec, dtype=np.float32)
    t = 2.0 * np.cross(-q_xyz, vec)
    return vec + q_w * t + np.cross(-q_xyz, t)


def _fit_linear_axis(indices, positions, default_offset, default_spacing, spacing_min, spacing_max):
    if len(indices) < 2 or len(positions) < 2:
        return default_offset, default_spacing, {}, float("inf")

    idx = np.asarray(indices, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)

    try:
        spacing, offset = np.polyfit(idx, pos, deg=1)
    except Exception:
        spacing = default_spacing
        offset = default_offset

    spacing = float(np.clip(spacing, spacing_min, spacing_max))
    offset = float(np.median(pos - spacing * idx))
    residual = float(np.mean(np.abs((offset + spacing * idx) - pos)))
    counts = {int(i): int(np.sum(idx == i)) for i in sorted(set(int(v) for v in idx.tolist()))}
    return offset, spacing, counts, residual


def _extract_event_attr(note_event, *names):
    for name in names:
        value = getattr(note_event, name, None)
        if value is not None:
            return value
    return None


def _encode_note_id(note_info):
    note_type = int(note_info.get("type", 0))
    if note_type == 3:
        scoring_type = NOTE_SCORE_TYPE_NOSCORE
    elif note_info.get("chainLink", False):
        scoring_type = NOTE_SCORE_TYPE_BURSTSLIDERELEMENT
    elif note_info.get("chainHead", False):
        scoring_type = NOTE_SCORE_TYPE_BURSTSLIDERHEAD
    elif note_info.get("arcHead", False):
        scoring_type = NOTE_SCORE_TYPE_SLIDERHEAD
    elif note_info.get("arcTail", False):
        scoring_type = NOTE_SCORE_TYPE_SLIDERTAIL
    else:
        scoring_type = NOTE_SCORE_TYPE_NORMAL_2

    line_index = int(note_info.get("lineIndex", 0))
    line_layer = int(note_info.get("lineLayer", 0))
    color_type = 3 if note_type == 3 else max(0, note_type)
    cut_direction = 9 if note_type == 3 else int(note_info.get("cutDirection", 8))
    return (
        scoring_type * 10000
        + line_index * 1000
        + line_layer * 100
        + color_type * 10
        + cut_direction
    )


def _read_hit_samples(replay_path, max_hits_per_replay=250, time_window=0.080):
    frames, replay_meta = parse_bsor(replay_path)
    if not frames or not replay_meta or not replay_meta.get("song_hash"):
        return []

    notes, bpm = get_map_notes(
        replay_meta["song_hash"],
        preferred_difficulty=replay_meta.get("difficulty"),
        preferred_mode=replay_meta.get("mode"),
    )
    if not notes or bpm is None:
        return []

    try:
        replay = load_bsor(replay_path)
    except Exception:
        return []

    replay_notes = getattr(replay, "notes", None)
    if not replay_notes:
        return []

    times = np.asarray([frame["time"] for frame in frames], dtype=np.float32)
    poses = np.asarray([frame["pose"] for frame in frames], dtype=np.float32)
    if times.size == 0 or poses.shape[0] == 0:
        return []

    beat_notes = [note for note in notes if note.get("type") in (0, 1)]
    beat_by_note_id = {}
    for note in beat_notes:
        beat_by_note_id.setdefault(_encode_note_id(note), []).append(note)
    used_note_ids = set()
    samples = []

    for replay_note in replay_notes:
        event_type = _extract_event_attr(replay_note, "event_type", "eventType")
        try:
            event_type = int(event_type)
        except (TypeError, ValueError):
            continue
        if event_type != 0:
            continue

        event_time = _extract_event_attr(replay_note, "event_time", "eventTime")
        if event_time is None:
            continue
        try:
            event_time = float(event_time)
        except Exception:
            continue

        note_id = _extract_event_attr(replay_note, "note_id", "noteID")
        beat_note = None
        if note_id is not None:
            try:
                note_id = int(note_id)
                candidates = beat_by_note_id.get(note_id, [])
                if candidates:
                    remaining = [note for note in candidates if note["index"] not in used_note_ids]
                    if not remaining:
                        remaining = candidates
                    beat_note = min(
                        remaining,
                        key=lambda note: abs(float(note["time"]) / max(bpm / 60.0, 1e-6) - event_time),
                    )
            except Exception:
                beat_note = None

        if beat_note is None:
            remaining = [note for note in beat_notes if note["index"] not in used_note_ids]
            if not remaining:
                continue
            beat_note = min(
                remaining,
                key=lambda note: abs(float(note["time"]) / max(bpm / 60.0, 1e-6) - event_time),
            )

        used_note_ids.add(int(beat_note["index"]))

        frame_idx = int(np.searchsorted(times, event_time))
        candidates = []
        if frame_idx < len(times):
            candidates.append(frame_idx)
        if frame_idx > 0:
            candidates.append(frame_idx - 1)
        if not candidates:
            continue

        best_idx = min(candidates, key=lambda idx: abs(float(times[idx]) - event_time))
        if abs(float(times[best_idx]) - event_time) > time_window:
            continue

        if beat_note["type"] == 0:
            hand_pos = poses[best_idx, 7:10]
            hand_rot = poses[best_idx, 10:14]
        else:
            hand_pos = poses[best_idx, 14:17]
            hand_rot = poses[best_idx, 17:21]

        samples.append({
            "lane": int(beat_note["lineIndex"]),
            "layer": int(beat_note["lineLayer"]),
            "type": int(beat_note["type"]),
            "hand_pos": np.asarray(hand_pos, dtype=np.float32),
            "hand_rot": np.asarray(hand_rot, dtype=np.float32),
            "cut_point": np.asarray(getattr(replay_note.cut, "cutPoint", hand_pos), dtype=np.float32),
            "saber_direction": np.asarray(getattr(replay_note.cut, "saberDirection", [0.0, 0.0, 1.0]), dtype=np.float32),
            "cut_normal": np.asarray(getattr(replay_note.cut, "cutNormal", [0.0, 0.0, -1.0]), dtype=np.float32),
        })

    if len(samples) > max_hits_per_replay:
        step = max(1, len(samples) // max_hits_per_replay)
        samples = samples[::step][:max_hits_per_replay]

    return samples


def _fit_local_saber_axis(hand_rot, saber_direction, cut_normal):
    blade_world = np.cross(cut_normal, saber_direction)
    norms = np.linalg.norm(blade_world, axis=1)
    valid = norms > 1e-6
    if not np.any(valid):
        return np.asarray(DEFAULT_CALIBRATION["saber_axis"], dtype=np.float32), float("inf")

    blade_world = blade_world[valid] / norms[valid, None]
    blade_local = _rotate_world_to_local(hand_rot[valid], blade_world)
    blade_local_norm = np.linalg.norm(blade_local, axis=1, keepdims=True)
    blade_local = blade_local / np.maximum(blade_local_norm, 1e-6)
    blade_local[blade_local[:, 2] < 0.0] *= -1.0
    axis = blade_local.mean(axis=0)
    axis_norm = max(float(np.linalg.norm(axis)), 1e-6)
    axis = axis / axis_norm
    fit_error = float(np.mean(np.linalg.norm(blade_local - axis[None, :], axis=1)))
    return axis.astype(np.float32), fit_error


def _fit_saber_segment_geometry(
    lane_idx,
    layer_idx,
    hand_pos,
    hand_rot,
    cut_point,
    fitted_axis,
    axis_fit_error,
    replay_count,
    note_count,
):
    world_dir = _rotate_axis(hand_rot, fitted_axis)
    best = None

    origin_x_candidates = (-0.05, 0.0, 0.05)
    origin_y_candidates = (-0.05, 0.0, 0.05, 0.10, 0.15)
    origin_z_candidates = (-0.80, -0.70, -0.60, -0.50, -0.40, -0.30, -0.20)
    length_candidates = (0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20)

    for origin_x in origin_x_candidates:
        for origin_y in origin_y_candidates:
            for origin_z in origin_z_candidates:
                origin_local = np.asarray([origin_x, origin_y, origin_z], dtype=np.float32)
                hilt_pos = hand_pos + _rotate_axis(hand_rot, origin_local)
                raw_along = np.sum((cut_point - hilt_pos) * world_dir, axis=1)
                for saber_length in length_candidates:
                    closest_along = np.clip(raw_along, 0.0, float(saber_length))
                    contact_pos = hilt_pos + world_dir * closest_along[:, None]
                    local_fit_error = float(np.mean(np.linalg.norm(contact_pos - cut_point, axis=1)))
                    x_offset, x_spacing, lane_counts, x_res = _fit_linear_axis(
                        lane_idx,
                        contact_pos[:, 0],
                        DEFAULT_CALIBRATION["x_offset"],
                        DEFAULT_CALIBRATION["x_spacing"],
                        spacing_min=0.40,
                        spacing_max=0.80,
                    )
                    y_offset, y_spacing, layer_counts, y_res = _fit_linear_axis(
                        layer_idx,
                        contact_pos[:, 1],
                        DEFAULT_CALIBRATION["y_offset"],
                        DEFAULT_CALIBRATION["y_spacing"],
                        spacing_min=0.22,
                        spacing_max=0.60,
                    )
                    total_error = x_res + y_res + axis_fit_error + local_fit_error + (abs(origin_x) * 0.05)
                    candidate = {
                        "version": CALIBRATION_VERSION,
                        "x_offset": x_offset,
                        "x_spacing": x_spacing,
                        "y_offset": y_offset,
                        "y_spacing": y_spacing,
                        "saber_axis": fitted_axis.tolist(),
                        "saber_axis_label": f"[{fitted_axis[0]:.3f}, {fitted_axis[1]:.3f}, {fitted_axis[2]:.3f}]",
                        "saber_origin": [float(origin_x), float(origin_y), float(origin_z)],
                        "saber_length": float(saber_length),
                        "source": "hit-event-calibration+origin-fit",
                        "fit_error": float(total_error),
                        "grid_fit_error": float(x_res + y_res),
                        "local_fit_error": float(local_fit_error),
                        "axis_fit_error": float(axis_fit_error),
                        "replays_used": replay_count,
                        "notes_used": note_count,
                        "lane_counts": lane_counts,
                        "layer_counts": layer_counts,
                        "maps_dir": MAPS_DIR,
                    }
                    if best is None or candidate["fit_error"] < best["fit_error"]:
                        best = candidate
    return best


def _fallback_controller_fit(
    replay_files,
    max_replays,
    notes_per_replay,
    time_window,
):
    lane_points = {idx: [] for idx in range(4)}
    layer_points = {idx: [] for idx in range(3)}
    replay_count = 0
    used_notes = 0

    for replay_path in replay_files[:max_replays]:
        frames, replay_meta = parse_bsor(replay_path)
        if not frames or not replay_meta or not replay_meta.get("song_hash"):
            continue

        notes, bpm = get_map_notes(
            replay_meta["song_hash"],
            preferred_difficulty=replay_meta.get("difficulty"),
            preferred_mode=replay_meta.get("mode"),
        )
        if not notes or bpm is None:
            continue

        times = np.asarray([frame["time"] for frame in frames], dtype=np.float32)
        hand_poses = np.asarray([frame["pose"][7:21] for frame in frames], dtype=np.float32)
        if times.size == 0 or hand_poses.shape[0] == 0:
            continue

        usable_notes = [note for note in notes if note.get("type") in (0, 1)]
        if not usable_notes:
            continue

        if len(usable_notes) > notes_per_replay:
            step = max(1, len(usable_notes) // notes_per_replay)
            usable_notes = usable_notes[::step][:notes_per_replay]

        bps = bpm / 60.0
        replay_used_any = False
        for note in usable_notes:
            note_time_sec = float(note["time"]) / max(bps, 1e-6)
            frame_idx = int(np.searchsorted(times, note_time_sec))
            candidates = []
            if frame_idx < len(times):
                candidates.append(frame_idx)
            if frame_idx > 0:
                candidates.append(frame_idx - 1)
            if not candidates:
                continue

            best_idx = min(candidates, key=lambda idx: abs(float(times[idx]) - note_time_sec))
            if abs(float(times[best_idx]) - note_time_sec) > time_window:
                continue

            hand_pos = hand_poses[best_idx, 0:3] if note["type"] == 0 else hand_poses[best_idx, 7:10]
            lane = int(note.get("lineIndex", -1))
            layer = int(note.get("lineLayer", -1))
            if lane not in lane_points or layer not in layer_points:
                continue

            lane_points[lane].append(float(hand_pos[0]))
            layer_points[layer].append(float(hand_pos[1]))
            used_notes += 1
            replay_used_any = True

        if replay_used_any:
            replay_count += 1

    lane_idx = []
    lane_pos = []
    for lane, values in lane_points.items():
        lane_idx.extend([lane] * len(values))
        lane_pos.extend(values)

    layer_idx = []
    layer_pos = []
    for layer, values in layer_points.items():
        layer_idx.extend([layer] * len(values))
        layer_pos.extend(values)

    x_offset, x_spacing, lane_counts, x_res = _fit_linear_axis(
        lane_idx,
        lane_pos,
        DEFAULT_CALIBRATION["x_offset"],
        DEFAULT_CALIBRATION["x_spacing"],
        spacing_min=0.40,
        spacing_max=0.80,
    )
    y_offset, y_spacing, layer_counts, y_res = _fit_linear_axis(
        layer_idx,
        layer_pos,
        DEFAULT_CALIBRATION["y_offset"],
        DEFAULT_CALIBRATION["y_spacing"],
        spacing_min=0.22,
        spacing_max=0.60,
    )

    return {
        "version": CALIBRATION_VERSION,
        "x_offset": x_offset,
        "x_spacing": x_spacing,
        "y_offset": y_offset,
        "y_spacing": y_spacing,
        "saber_axis": DEFAULT_CALIBRATION["saber_axis"],
        "saber_axis_label": DEFAULT_CALIBRATION["saber_axis_label"],
        "saber_origin": DEFAULT_CALIBRATION["saber_origin"],
        "saber_length": DEFAULT_CALIBRATION["saber_length"],
        "source": "fallback-controller-fit",
        "fit_error": x_res + y_res,
        "replays_used": replay_count,
        "notes_used": used_notes,
        "lane_counts": lane_counts,
        "layer_counts": layer_counts,
        "maps_dir": MAPS_DIR,
    }


def calibrate_from_replays(
    max_replays=80,
    notes_per_replay=300,
    time_window=0.080,
    out_path=CALIBRATION_PATH,
):
    replay_files = sorted(
        os.path.join(REPLAYS_DIR, name)
        for name in os.listdir(REPLAYS_DIR)
        if name.lower().endswith(".bsor")
    )
    if not replay_files:
        raise RuntimeError(f"No .bsor files found in {REPLAYS_DIR}")

    hit_samples = []
    replay_count = 0
    for replay_path in replay_files[:max_replays]:
        samples = _read_hit_samples(
            replay_path,
            max_hits_per_replay=max(32, notes_per_replay),
            time_window=time_window,
        )
        if samples:
            hit_samples.extend(samples)
            replay_count += 1

    if len(hit_samples) < 200:
        calibration = _fallback_controller_fit(
            replay_files=replay_files,
            max_replays=max_replays,
            notes_per_replay=notes_per_replay,
            time_window=time_window,
        )
        save_simulator_calibration(calibration, out_path)
        return calibration

    lane_idx = np.asarray([sample["lane"] for sample in hit_samples], dtype=np.int32)
    layer_idx = np.asarray([sample["layer"] for sample in hit_samples], dtype=np.int32)
    hand_pos = np.asarray([sample["hand_pos"] for sample in hit_samples], dtype=np.float32)
    hand_rot = np.asarray([sample["hand_rot"] for sample in hit_samples], dtype=np.float32)
    cut_point = np.asarray([sample["cut_point"] for sample in hit_samples], dtype=np.float32)
    saber_direction = np.asarray([sample["saber_direction"] for sample in hit_samples], dtype=np.float32)
    cut_normal = np.asarray([sample["cut_normal"] for sample in hit_samples], dtype=np.float32)
    fitted_axis, axis_fit_error = _fit_local_saber_axis(hand_rot, saber_direction, cut_normal)

    best = _fit_saber_segment_geometry(
        lane_idx=lane_idx,
        layer_idx=layer_idx,
        hand_pos=hand_pos,
        hand_rot=hand_rot,
        cut_point=cut_point,
        fitted_axis=fitted_axis,
        axis_fit_error=axis_fit_error,
        replay_count=replay_count,
        note_count=len(hit_samples),
    )

    save_simulator_calibration(best, out_path)
    return best


def main():
    parser = argparse.ArgumentParser(description="Estimate Beat Saber simulator geometry from local replays.")
    parser.add_argument("--max-replays", type=int, default=80, help="Maximum number of replays to scan.")
    parser.add_argument("--notes-per-replay", type=int, default=300, help="Cap sampled hits per replay.")
    parser.add_argument("--time-window", type=float, default=0.080, help="Max seconds between hit event time and sampled frame.")
    parser.add_argument("--out", default=CALIBRATION_PATH, help="Output calibration JSON path.")
    args = parser.parse_args()

    calibration = calibrate_from_replays(
        max_replays=max(1, args.max_replays),
        notes_per_replay=max(16, args.notes_per_replay),
        time_window=max(0.01, args.time_window),
        out_path=args.out,
    )

    print("Simulator calibration saved.")
    print(f"  Path:        {args.out}")
    print(f"  Source:      {calibration['source']}")
    print(f"  Replays:     {calibration['replays_used']}")
    print(f"  Notes:       {calibration['notes_used']}")
    print(f"  X offset:    {calibration['x_offset']:.4f}")
    print(f"  X spacing:   {calibration['x_spacing']:.4f}")
    print(f"  Y offset:    {calibration['y_offset']:.4f}")
    print(f"  Y spacing:   {calibration['y_spacing']:.4f}")
    print(f"  Saber axis:  {calibration['saber_axis_label']} {calibration['saber_axis']}")
    print(f"  Saber origin:{calibration.get('saber_origin', [0.0, 0.0, 0.0])}")
    print(f"  Saber len:   {calibration['saber_length']:.2f}")
    print(f"  Fit error:   {calibration.get('fit_error', 0.0):.4f}")


if __name__ == "__main__":
    main()
