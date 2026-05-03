"""Adapter from CyberNoodles beatmap dicts to the internal replay oracle."""

from __future__ import annotations

from typing import Any, Dict, Optional

from bsor.Bsor import Bsor

from cybernoodles.bsor_bridge import load_bsor
from .core import (
    OracleBeatMap,
    OracleMap,
    OracleNote,
    OracleObstacle,
    calculate_score_assuming_valid_times,
)


def build_oracle_map(beatmap: Dict[str, Any], bpm: float, difficulty: str, mode: str) -> OracleMap:
    oracle_beatmap = OracleBeatMap(
        difficulty=str(difficulty),
        note_jump_movement_speed=float(beatmap.get("njs", 18.0) or 18.0),
        note_jump_start_beat_offset=float(beatmap.get("offset", 0.0) or 0.0),
        notes=[
            OracleNote(
                time=float(note_info.get("time", 0.0)),
                note_type=int(note_info.get("type", 0)),
                line_index=int(note_info.get("lineIndex", 0)),
                line_layer=int(note_info.get("lineLayer", 0)),
                cut_direction=int(note_info.get("cutDirection", 8)),
            )
            for note_info in beatmap.get("notes", [])
        ],
        obstacles=[
            OracleObstacle(
                width=int(obstacle_info.get("width", 1)),
                line_index=int(obstacle_info.get("lineIndex", 0)),
                time=float(obstacle_info.get("time", 0.0)),
                obstacle_type=int(obstacle_info.get("type", 0)),
                duration=float(obstacle_info.get("duration", 0.0)),
            )
            for obstacle_info in beatmap.get("obstacles", [])
        ],
    )
    return OracleMap(
        beats_per_minute=float(bpm),
        beatmaps={str(mode): {str(difficulty): oracle_beatmap}},
    )


def score_loaded_replay_with_oracle(replay: Bsor, beatmap: Dict[str, Any], bpm: float) -> Dict[str, Any]:
    oracle_map = build_oracle_map(
        beatmap=beatmap,
        bpm=float(bpm),
        difficulty=str(replay.info.difficulty),
        mode=str(replay.info.mode),
    )
    try:
        result = calculate_score_assuming_valid_times(oracle_map, replay)
    except Exception as exc:
        return {"error": str(exc)}

    return {
        "score": int(result.score),
        "raw_breakdown": list(result.raw_breakdown),
        "cut_breakdowns": [list(x) for x in result.cut_breakdowns],
    }


def score_replay_with_oracle(replay_path: str, beatmap: Dict[str, Any], bpm: float) -> Dict[str, Any]:
    replay = load_bsor(replay_path)
    return score_loaded_replay_with_oracle(replay, beatmap, bpm)
