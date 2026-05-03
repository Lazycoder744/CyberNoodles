import json
import os
from typing import Dict, List

import numpy as np
import torch

from cybernoodles.core.network import CURRENT_POSE_END, CURRENT_POSE_START, VELOCITY_DIM
from cybernoodles.data.dataset_builder import MANIFEST_PATH, SHARD_ROOT, TARGET_POSE_HORIZON_FRAMES
from cybernoodles.data.shard_io import load_shard_pair, shard_files_exist, shard_record_sort_key

STYLE_CALIBRATION_PATH = os.path.join("data", "processed", "style_calibration.json")
STYLE_CALIBRATION_VERSION = 1

DEFAULT_STYLE_CALIBRATION = {
    "version": STYLE_CALIBRATION_VERSION,
    "source": "default",
    "linear_speed_p95": 3.0,
    "linear_speed_p99": 4.4,
    "angular_speed_p95": 11.0,
    "angular_speed_p99": 17.0,
    "linear_speed_cap": 3.35,
    "angular_speed_cap": 12.5,
}

STYLE_FLOAT_RANGES = {
    "linear_speed_p95": (0.1, 20.0),
    "linear_speed_p99": (0.1, 20.0),
    "angular_speed_p95": (0.1, 80.0),
    "angular_speed_p99": (0.1, 80.0),
    "linear_speed_cap": (0.1, 20.0),
    "angular_speed_cap": (0.1, 80.0),
}


def _require_style_float(payload, key):
    if key not in payload:
        raise ValueError(f"missing required style calibration field: {key}")
    try:
        value = float(payload[key])
    except (TypeError, ValueError):
        raise ValueError(f"style calibration field {key} must be numeric")
    if not np.isfinite(value):
        raise ValueError(f"style calibration field {key} must be finite")
    min_value, max_value = STYLE_FLOAT_RANGES[key]
    if value < min_value or value > max_value:
        raise ValueError(
            f"style calibration field {key} out of range [{min_value}, {max_value}]: {value}"
        )
    return value


def _validate_style_calibration_payload(data, path):
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    if "version" not in data:
        raise ValueError("missing required style calibration field: version")
    try:
        version = int(data["version"])
    except (TypeError, ValueError):
        raise ValueError("style calibration field version must be an integer")
    if version != STYLE_CALIBRATION_VERSION:
        raise ValueError(f"unsupported style calibration version {version}; expected {STYLE_CALIBRATION_VERSION}")

    source = data.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("style calibration field source must be a non-empty string")

    merged = DEFAULT_STYLE_CALIBRATION.copy()
    merged["version"] = version
    merged["source"] = source
    for key in STYLE_FLOAT_RANGES:
        merged[key] = _require_style_float(data, key)

    if merged["linear_speed_p99"] < merged["linear_speed_p95"]:
        raise ValueError("style calibration linear_speed_p99 must be >= linear_speed_p95")
    if merged["angular_speed_p99"] < merged["angular_speed_p95"]:
        raise ValueError("style calibration angular_speed_p99 must be >= angular_speed_p95")
    if "records_used" in data:
        try:
            records_used = int(data["records_used"])
        except (TypeError, ValueError):
            raise ValueError("style calibration field records_used must be an integer")
        if records_used < 0:
            raise ValueError("style calibration field records_used must be non-negative")
        merged["records_used"] = records_used
    return merged


def _safe_quantile(values: List[float], q: float, fallback: float) -> float:
    if not values:
        return float(fallback)
    return float(np.quantile(np.asarray(values, dtype=np.float32), q))


def _quat_angle_speed(curr_q: torch.Tensor, tgt_q: torch.Tensor, horizon_seconds: float) -> torch.Tensor:
    curr_q = curr_q / curr_q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    tgt_q = tgt_q / tgt_q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    dot = torch.abs((curr_q * tgt_q).sum(dim=-1)).clamp(0.0, 1.0)
    angle = 2.0 * torch.acos(dot)
    return angle / max(horizon_seconds, 1e-6)


def load_style_calibration(path=None) -> Dict[str, float]:
    if path is None:
        path = STYLE_CALIBRATION_PATH
    if not os.path.exists(path):
        return DEFAULT_STYLE_CALIBRATION.copy()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _validate_style_calibration_payload(data, path)
    except Exception as exc:
        raise ValueError(f"Invalid style calibration {path}: {exc}") from exc


def ensure_style_calibration(force: bool = False, verbose: bool = False) -> Dict[str, float]:
    if not force and os.path.exists(STYLE_CALIBRATION_PATH):
        return load_style_calibration()

    if not os.path.exists(MANIFEST_PATH):
        if verbose:
            print("Style calibration fallback: BC shard manifest not found, using defaults.")
        return DEFAULT_STYLE_CALIBRATION.copy()

    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        if verbose:
            print("Style calibration fallback: manifest unreadable, using defaults.")
        return DEFAULT_STYLE_CALIBRATION.copy()

    records = manifest.get("shards", [])
    if not records:
        return DEFAULT_STYLE_CALIBRATION.copy()

    linear_speeds: List[float] = []
    angular_speeds: List[float] = []
    horizon_seconds = TARGET_POSE_HORIZON_FRAMES / 60.0

    selected = sorted(records, key=shard_record_sort_key)[:64]
    for record in selected:
        if not shard_files_exist(record, SHARD_ROOT):
            continue

        try:
            batch_x, batch_y = load_shard_pair(record, SHARD_ROOT)
            batch_x = batch_x.float()
            batch_y = batch_y.float()
        except Exception:
            continue

        if batch_x.numel() == 0 or batch_y.numel() == 0:
            continue

        sample_count = min(batch_x.shape[0], 4096)
        if sample_count <= 0:
            continue
        indices = torch.linspace(0, batch_x.shape[0] - 1, steps=sample_count).long()
        batch_x = batch_x.index_select(0, indices)
        batch_y = batch_y.index_select(0, indices)

        vel_start = CURRENT_POSE_END
        vel = batch_x[:, vel_start:vel_start + VELOCITY_DIM]
        left_linear = torch.norm(vel[:, 7:10], dim=-1)
        right_linear = torch.norm(vel[:, 14:17], dim=-1)
        max_linear = torch.maximum(left_linear, right_linear)
        linear_speeds.extend(max_linear.cpu().tolist())

        current_pose = batch_x[:, CURRENT_POSE_START:CURRENT_POSE_END]
        left_ang = _quat_angle_speed(current_pose[:, 10:14], batch_y[:, 10:14], horizon_seconds)
        right_ang = _quat_angle_speed(current_pose[:, 17:21], batch_y[:, 17:21], horizon_seconds)
        max_ang = torch.maximum(left_ang, right_ang)
        angular_speeds.extend(max_ang.cpu().tolist())

    if not linear_speeds:
        return DEFAULT_STYLE_CALIBRATION.copy()

    linear_p95 = _safe_quantile(linear_speeds, 0.95, DEFAULT_STYLE_CALIBRATION["linear_speed_p95"])
    linear_p99 = _safe_quantile(linear_speeds, 0.99, DEFAULT_STYLE_CALIBRATION["linear_speed_p99"])
    angular_p95 = _safe_quantile(angular_speeds, 0.95, DEFAULT_STYLE_CALIBRATION["angular_speed_p95"])
    angular_p99 = _safe_quantile(angular_speeds, 0.99, DEFAULT_STYLE_CALIBRATION["angular_speed_p99"])

    calibration = {
        "version": STYLE_CALIBRATION_VERSION,
        "source": "bc-shards",
        "records_used": len(selected),
        "linear_speed_p95": linear_p95,
        "linear_speed_p99": linear_p99,
        "angular_speed_p95": angular_p95,
        "angular_speed_p99": angular_p99,
        "linear_speed_cap": max(2.6, min(5.2, linear_p95 * 1.18)),
        "angular_speed_cap": max(8.5, min(22.0, angular_p95 * 1.18)),
    }

    os.makedirs(os.path.dirname(STYLE_CALIBRATION_PATH), exist_ok=True)
    with open(STYLE_CALIBRATION_PATH, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)

    if verbose:
        print(
            "Style calibration saved: "
            f"linear cap {calibration['linear_speed_cap']:.2f} m/s | "
            f"angular cap {calibration['angular_speed_cap']:.2f} rad/s"
        )

    merged = DEFAULT_STYLE_CALIBRATION.copy()
    merged.update(calibration)
    return merged


if __name__ == "__main__":
    calibration = ensure_style_calibration(force=True, verbose=True)
    print(json.dumps(calibration, indent=2))
