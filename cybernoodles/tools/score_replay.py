import argparse
import json
import os
import time

import numpy as np
import torch

from cybernoodles.bsor_bridge import load_bsor
from cybernoodles.core.gpu_simulator import HISTORY_LEN
from cybernoodles.data.dataset_builder import (
    MAPS_DIR,
    REPLAYS_DIR,
    get_map_data,
    normalize_difficulty_name,
    parse_bsor,
)
from cybernoodles.data.fetch_data import download_file, get_beatsaver_map_url, slim_map_postprocess
from cybernoodles.envs import make_vector_env
from cybernoodles.oracle import score_loaded_replay_with_oracle

FPS = 60.0


def _advance_multiplier(multiplier, progress):
    target = 0
    if multiplier < 2:
        target = 2
    elif multiplier < 4:
        target = 4
    elif multiplier < 8:
        target = 8

    if target == 0:
        return 8, 0

    progress += 1
    if progress >= target:
        return min(8, multiplier * 2), 0
    return multiplier, progress


def compute_standard_max_score(map_notes):
    max_score = 0
    multiplier = 1
    progress = 0
    for note_info in map_notes:
        if int(note_info.get("type", 0)) == 3:
            continue
        multiplier, progress = _advance_multiplier(multiplier, progress)
        max_score += int(round(float(note_info.get("scoreCap", 115.0)) * multiplier))
    return max_score


def compute_oracle_reference_max_score(map_notes):
    max_score = 0
    successful_cuts = 0
    for note_info in map_notes:
        if int(note_info.get("type", 0)) == 3:
            continue
        successful_cuts += 1
        if successful_cuts < 5:
            multiplier = 2
        elif successful_cuts < 13:
            multiplier = 4
        else:
            multiplier = 8
        max_score += int(round(float(note_info.get("scoreCap", 115.0)) * multiplier))
    return max_score


def compute_max_score(map_notes, scoring_model="oracle"):
    model = str(scoring_model).strip().lower()
    if model == "standard":
        return compute_standard_max_score(map_notes)
    if model not in {"oracle", "oracle_reference", "oracle_reference_diagnostic"}:
        raise ValueError(
            "Unsupported scoring_model "
            f"{scoring_model!r}; expected 'standard', 'oracle', or 'oracle_reference'."
        )
    return compute_oracle_reference_max_score(map_notes)


def compute_score_maxima(map_notes):
    return {
        "standard_max_score": int(compute_standard_max_score(map_notes)),
        "oracle_reference_max_score": int(compute_oracle_reference_max_score(map_notes)),
    }


def resolve_replay_path(replay_arg):
    if os.path.exists(replay_arg):
        return replay_arg

    candidate = os.path.join(REPLAYS_DIR, replay_arg)
    if os.path.exists(candidate):
        return candidate

    if not replay_arg.lower().endswith(".bsor"):
        candidate = os.path.join(REPLAYS_DIR, f"{replay_arg}.bsor")
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"Replay not found: {replay_arg}")


def ensure_map_cached(map_hash):
    map_hash = str(map_hash or "").strip()
    if not map_hash:
        return False

    for candidate in (
        os.path.join(MAPS_DIR, f"{map_hash}.zip"),
        os.path.join(MAPS_DIR, f"{map_hash.upper()}.zip"),
        os.path.join(MAPS_DIR, f"{map_hash.lower()}.zip"),
    ):
        if os.path.exists(candidate):
            return True

    os.makedirs(MAPS_DIR, exist_ok=True)
    print(f"Map {map_hash} not cached. Resolving BeatSaver download URL...")
    download_url = get_beatsaver_map_url(None, map_hash)
    if not download_url:
        return False

    target_path = os.path.join(MAPS_DIR, f"{map_hash.lower()}.zip")
    result = download_file(download_url, target_path, postprocess=slim_map_postprocess)
    if result in {"downloaded", "skipped"} and os.path.exists(target_path):
        print(f"Cached map {map_hash} to {target_path}")
        return True

    print(f"Map download failed for {map_hash}: {result}")
    return False


def load_replay_score(replay_path):
    try:
        replay = load_bsor(replay_path)
        info = getattr(replay, "info", None)
        return int(getattr(info, "score", 0) or 0)
    except Exception:
        return None


def load_replay_bundle(replay_path):
    replay = load_bsor(replay_path)

    info = getattr(replay, "info", None)
    frames = []
    prev_time = None
    prev_pose = None
    sanitized_time_frames = 0
    sanitized_pose_segments = 0

    from cybernoodles.data.dataset_builder import _sanitize_frame_time, _sanitize_pose

    for frame in replay.frames:
        frame_time, time_sanitized = _sanitize_frame_time(getattr(frame, "time", 0.0), prev_time)
        pose, replaced_segments = _sanitize_pose(
            [
                *frame.head.position,
                *frame.head.rotation,
                *frame.left_hand.position,
                *frame.left_hand.rotation,
                *frame.right_hand.position,
                *frame.right_hand.rotation,
            ],
            prev_pose,
        )
        frames.append({"time": frame_time, "pose": pose.tolist()})
        prev_time = frame_time
        prev_pose = pose
        sanitized_time_frames += int(time_sanitized)
        sanitized_pose_segments += replaced_segments

    replay_meta = {
        "song_hash": getattr(info, "songHash", None),
        "difficulty": getattr(info, "difficulty", None),
        "mode": getattr(info, "mode", "Standard") or "Standard",
        "modifiers": getattr(info, "modifiers", "") or "",
        "sanitized_time_frames": sanitized_time_frames,
        "sanitized_pose_segments": sanitized_pose_segments,
    }
    return replay, frames, replay_meta


def build_initial_note_mask(sim, current_time):
    t_beat = current_time * sim.bps[0].item()
    note_times = sim.note_times[0, : sim.max_notes].detach().cpu().numpy()
    return note_times >= float(t_beat)


def score_replay(replay_path, device, fail_enabled=True, track_events=False, frame_stride=1, max_seconds=None):
    replay, frames, replay_meta = load_replay_bundle(replay_path)
    if not frames or not replay_meta or not replay_meta.get("song_hash"):
        raise RuntimeError(f"Unable to parse replay: {replay_path}")
    frame_stride = max(1, int(frame_stride))
    time_budget = None if max_seconds is None else max(0.0, float(max_seconds))

    beatmap, bpm = get_map_data(
        replay_meta["song_hash"],
        preferred_difficulty=replay_meta.get("difficulty"),
        preferred_mode=replay_meta.get("mode"),
    )
    if not beatmap:
        ensure_map_cached(replay_meta["song_hash"])
        beatmap, bpm = get_map_data(
            replay_meta["song_hash"],
            preferred_difficulty=replay_meta.get("difficulty"),
            preferred_mode=replay_meta.get("mode"),
        )
    if not beatmap and normalize_difficulty_name(replay_meta.get("difficulty")) == "":
        beatmap, bpm = get_map_data(replay_meta["song_hash"])
    if not beatmap or bpm is None:
        raise RuntimeError(f"Missing beatmap for replay song hash {replay_meta['song_hash']}")

    vector_env = make_vector_env(
        num_envs=1,
        device=device,
        penalty_weights=(0.5, 0.0, 0.0, 0.0),
        dense_reward_scale=0.0,
        training_wheels=0.0,
        rehab_assists=0.0,
        survival_assistance=0.0,
        stability_assistance=0.0,
        style_guidance_level=0.0,
        fail_enabled=bool(fail_enabled),
        saber_inertia=0.0,
        rot_clamp=0.07,
        pos_clamp=0.12,
        score_only_mode=True,
        external_pose_passthrough=True,
    )
    vector_env.load_maps([beatmap], [float(bpm)])
    sim = vector_env.simulator
    if track_events:
        sim.enable_event_tracking()

    first_pose = np.asarray(frames[0]["pose"][:21], dtype=np.float32)
    first_time = float(frames[0]["time"])
    history = np.repeat(first_pose.reshape(1, -1), HISTORY_LEN, axis=0)
    note_active_mask = build_initial_note_mask(sim, first_time)
    sim.teleport_all(first_pose, history, first_time, note_active_mask)

    last_pose = torch.as_tensor(first_pose, dtype=torch.float32, device=device).unsqueeze(0)
    last_time = first_time
    truncated = False
    duplicate_time_frames = 0
    with torch.no_grad():
        for frame in frames[1::frame_stride]:
            frame_pose = np.asarray(frame["pose"][:21], dtype=np.float32)
            frame_time = float(frame["time"])
            if time_budget is not None and (frame_time - first_time) > time_budget:
                truncated = True
                break
            if frame_time <= last_time + 1e-9:
                last_pose = torch.as_tensor(frame_pose, dtype=torch.float32, device=device).unsqueeze(0)
                duplicate_time_frames += 1
                continue
            dt = frame_time - last_time
            last_pose = torch.as_tensor(frame_pose, dtype=torch.float32, device=device).unsqueeze(0)
            sim.step(last_pose, dt=dt)
            last_time = frame_time
            if bool(sim.episode_done[0].item()):
                break

        map_duration = float(sim.map_durations[0].item())
        if not bool(sim.episode_done[0].item()) and not truncated:
            tail_steps = max(0, int(np.ceil((map_duration - last_time) * FPS))) + 2
            for _ in range(tail_steps):
                resolved = float(sim.total_resolved_scorable[0].item())
                scorable = float(sim.scorable_note_counts[0].item())
                if last_time >= map_duration and resolved >= max(0.0, scorable - 0.5):
                    break
                sim.step(last_pose, dt=1.0 / FPS)
                last_time += 1.0 / FPS
                if bool(sim.episode_done[0].item()):
                    break

    tracked_events = None
    if track_events:
        tracked_events = list(sim.tracked_events[0])
        sim.disable_event_tracking()

    sim_score = float(sim.total_scores[0].item())
    sim_hits = int(round(float(sim.total_hits[0].item())))
    sim_total_misses = int(round(float(sim.total_misses[0].item())))
    sim_badcuts = int(round(float(sim.total_badcuts[0].item())))
    sim_bombs = int(round(float(sim.total_bombs[0].item())))
    sim_note_misses = int(round(float(sim.total_note_misses[0].item())))
    sim_cut_total = float(sim.total_cut_scores[0].item())
    sim_max_combo = int(round(float(sim.max_combo[0].item())))
    sim_wall_hits = int(round(float(sim.total_wall_hits[0].item())))
    scorable_notes = int(round(float(sim.scorable_note_counts[0].item())))
    resolved_notes = int(round(float(sim.total_resolved_scorable[0].item())))
    engaged_notes = int(round(float(sim.total_engaged_scorable[0].item())))
    score_maxima = compute_score_maxima(beatmap.get("notes", []))
    standard_max_score = float(score_maxima["standard_max_score"])
    oracle_reference_max_score = float(score_maxima["oracle_reference_max_score"])
    final_energy = float(sim.energy[0].item())

    hit_count = sim_hits
    bad_count = sim_badcuts
    miss_count = sim_note_misses
    bomb_count = sim_bombs
    wall_count = sim_wall_hits

    original_score = int(getattr(getattr(replay, "info", None), "score", 0) or 0)
    oracle = score_loaded_replay_with_oracle(replay, beatmap, bpm)
    original_accuracy = (100.0 * original_score / max(1.0, standard_max_score)) if original_score is not None else None
    simulator_accuracy = 100.0 * sim_score / max(1.0, standard_max_score)
    original_oracle_reference_accuracy = (
        100.0 * original_score / max(1.0, oracle_reference_max_score)
    ) if original_score is not None else None
    simulator_oracle_reference_accuracy = 100.0 * sim_score / max(1.0, oracle_reference_max_score)
    oracle_diagnostic_score = int(oracle["score"]) if oracle and oracle.get("score") is not None else None
    oracle_diagnostic_accuracy = None
    oracle_diagnostic_standard_accuracy = None
    if oracle and oracle.get("score") is not None:
        oracle_diagnostic_accuracy = 100.0 * oracle["score"] / max(1.0, oracle_reference_max_score)
        oracle_diagnostic_standard_accuracy = 100.0 * oracle["score"] / max(1.0, standard_max_score)
    task_accuracy = 100.0 * sim_hits / max(1, scorable_notes)
    engaged_accuracy = 100.0 * sim_hits / max(1, hit_count + bad_count + miss_count)
    average_cut = sim_cut_total / max(1, sim_hits)
    full_combo = (bad_count == 0 and miss_count == 0 and bomb_count == 0 and wall_count == 0)

    return {
        "replay_path": os.path.abspath(replay_path),
        "song_hash": str(replay_meta.get("song_hash", "")).upper(),
        "difficulty": replay_meta.get("difficulty"),
        "mode": replay_meta.get("mode"),
        "modifiers": replay_meta.get("modifiers", ""),
        "frames": len(frames),
        "bpm": float(bpm),
        "njs": float(beatmap.get("njs", 18.0) or 18.0),
        "max_score": int(round(standard_max_score)),
        "max_score_model": "standard",
        "score_normalizer_model": "standard",
        "standard_max_score": int(round(standard_max_score)),
        "oracle_reference_max_score": int(round(oracle_reference_max_score)),
        "original_score": int(original_score) if original_score is not None else None,
        "original_accuracy_percent": original_accuracy,
        "original_accuracy_percent_oracle_reference": original_oracle_reference_accuracy,
        "simulator_score": int(round(sim_score)),
        "simulator_accuracy_percent": simulator_accuracy,
        "simulator_accuracy_percent_oracle_reference": simulator_oracle_reference_accuracy,
        "oracle_diagnostic_model": "diagnostic_reference",
        "oracle_diagnostic_reference_max_score": int(round(oracle_reference_max_score)),
        "oracle_diagnostic_score": oracle_diagnostic_score,
        "oracle_diagnostic_accuracy_percent": oracle_diagnostic_accuracy,
        "oracle_diagnostic_accuracy_percent_oracle_reference": oracle_diagnostic_accuracy,
        "oracle_diagnostic_accuracy_percent_standard": oracle_diagnostic_standard_accuracy,
        "oracle_diagnostic_error": (oracle.get("error") if oracle and oracle.get("error") else None),
        "oracle_diagnostic_raw_breakdown": (oracle.get("raw_breakdown") if oracle else None),
        "oracle_score": oracle_diagnostic_score,
        "oracle_accuracy_percent": oracle_diagnostic_accuracy,
        "oracle_accuracy_percent_model": "oracle_reference",
        "oracle_accuracy_percent_standard": oracle_diagnostic_standard_accuracy,
        "oracle_error": (oracle.get("error") if oracle and oracle.get("error") else None),
        "oracle_raw_breakdown": (oracle.get("raw_breakdown") if oracle else None),
        "simsaber_score": oracle_diagnostic_score,
        "simsaber_accuracy_percent": oracle_diagnostic_accuracy,
        "simsaber_accuracy_percent_standard": oracle_diagnostic_standard_accuracy,
        "simsaber_error": (oracle.get("error") if oracle and oracle.get("error") else None),
        "task_accuracy_percent": task_accuracy,
        "engaged_accuracy_percent": engaged_accuracy,
        "hits": hit_count,
        "bad_cuts": bad_count,
        "misses": miss_count,
        "bombs": bomb_count,
        "wall_hits": wall_count,
        "scorable_notes": scorable_notes,
        "resolved_notes": resolved_notes,
        "engaged_notes": engaged_notes,
        "avg_cut": average_cut,
        "max_combo": sim_max_combo,
        "full_combo": full_combo,
        "final_energy": final_energy,
        "fail_enabled": bool(fail_enabled),
        "event_tracking": bool(track_events),
        "tracked_events": tracked_events,
        "frame_stride": int(frame_stride),
        "max_seconds": (None if time_budget is None else float(time_budget)),
        "truncated": bool(truncated),
        "skipped_duplicate_time_frames": int(duplicate_time_frames),
    }


def main():
    parser = argparse.ArgumentParser(description="Score a replay through the CyberNoodles simulator.")
    parser.add_argument("replay", help="Replay path, replay basename, or replay ID inside data/replays.")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), help="Device to run the simulator on.")
    parser.set_defaults(fail=True)
    parser.add_argument("--fail", dest="fail", action="store_true", help="Enable strict fail state during scoring.")
    parser.add_argument("--no-fail", dest="fail", action="store_false", help="Disable fail state and force a full-map score pass.")
    parser.add_argument("--track-events", action="store_true", help="Enable detailed per-event tracking. Slower; only needed for debugging.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Only feed every Nth replay frame to the simulator. Faster but approximate.")
    parser.add_argument("--max-seconds", type=float, default=None, help="Optional cap on analyzed replay seconds for quick debugging.")
    parser.add_argument("--json-out", default=None, help="Optional path to write the score summary JSON.")
    args = parser.parse_args()

    replay_path = resolve_replay_path(args.replay)
    t0 = time.perf_counter()
    summary = score_replay(
        replay_path,
        args.device,
        fail_enabled=args.fail,
        track_events=args.track_events,
        frame_stride=args.frame_stride,
        max_seconds=args.max_seconds,
    )
    elapsed = time.perf_counter() - t0

    print(f"Replay: {summary['replay_path']}")
    print(f"Song:   {summary['song_hash']} | {summary['difficulty']} | {summary['mode']}")
    print(f"Map:    BPM {summary['bpm']:.2f} | NJS {summary['njs']:.2f}")
    print(f"Frames: {summary['frames']} | Fail enabled: {summary['fail_enabled']} | Event tracking: {summary['event_tracking']}")
    print(
        f"Max:    standard {summary['standard_max_score']} | "
        f"oracle reference {summary['oracle_reference_max_score']}"
    )
    if summary["frame_stride"] != 1 or summary["max_seconds"] is not None:
        print(f"Debug mode: frame stride {summary['frame_stride']} | max seconds {summary['max_seconds']} | truncated {summary['truncated']}")
    if summary.get("skipped_duplicate_time_frames", 0):
        print(f"Skipped duplicate-time frames: {summary['skipped_duplicate_time_frames']}")
    if summary["original_score"] is not None:
        print(
            f"Replay score:    {summary['original_score']} / {summary['standard_max_score']} "
            f"standard ({summary['original_accuracy_percent']:.2f}%)"
        )
    else:
        print(f"Replay score:    unavailable / {summary['standard_max_score']} standard")
    if summary["oracle_score"] is not None:
        print(
            f"Oracle diagnostic: {summary['oracle_score']} / {summary['oracle_reference_max_score']} "
            f"oracle reference ({summary['oracle_accuracy_percent']:.2f}%)"
        )
    elif summary.get("oracle_error"):
        print(f"Oracle diagnostic: error ({summary['oracle_error']})")
    print(
        f"Simulator score: {summary['simulator_score']} / {summary['standard_max_score']} "
        f"standard ({summary['simulator_accuracy_percent']:.2f}%)"
    )
    print(
        f"Hits {summary['hits']} | bad {summary['bad_cuts']} | misses {summary['misses']} | "
        f"bombs {summary['bombs']} | walls {summary['wall_hits']}"
    )
    print(
        f"Task acc {summary['task_accuracy_percent']:.2f}% | engaged {summary['engaged_accuracy_percent']:.2f}% | "
        f"avg cut {summary['avg_cut']:.1f} | max combo {summary['max_combo']} | FC {summary['full_combo']}"
    )
    print(f"Elapsed: {elapsed:.1f}s")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved JSON summary to {args.json_out}")


if __name__ == "__main__":
    main()
