import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors_file


PROVENANCE_FIELDS = (
    "provenance",
    "player_id",
    "player_name",
    "score_id",
    "rank",
    "score_rank",
    "accuracy",
    "pp",
    "timepost",
    "map_hash",
    "song_hash",
    "song_name",
    "song_author",
    "mapper",
    "difficulty",
    "mode",
    "replay_file",
    "replay_id",
    "leaderboard_id",
    "selected_index",
    "source",
)


def shard_record_sort_key(record):
    return str(
        record.get("shard_path")
        or record.get("x_path")
        or record.get("replay_file")
        or ""
    )


def shard_record_paths(record, shard_root):
    shard_path = record.get("shard_path")
    if shard_path:
        return [os.path.join(shard_root, shard_path.replace("/", os.sep))]

    x_path = record.get("x_path")
    y_path = record.get("y_path")
    if x_path and y_path:
        return [
            os.path.join(shard_root, x_path.replace("/", os.sep)),
            os.path.join(shard_root, y_path.replace("/", os.sep)),
        ]
    raise KeyError("Shard record is missing shard_path and x_path/y_path fields.")


def shard_record_label(record, shard_root):
    return " | ".join(shard_record_paths(record, shard_root))


def shard_files_exist(record, shard_root):
    try:
        return all(os.path.exists(path) for path in shard_record_paths(record, shard_root))
    except KeyError:
        return False


def _merge_provenance(provenance, payload):
    if not isinstance(payload, dict):
        return provenance
    for key, value in payload.items():
        if key == "provenance":
            continue
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        provenance[str(key)] = value
    return provenance


def _parse_provenance_blob(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        payload = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def shard_record_provenance(record, metadata=None):
    provenance = {}
    if isinstance(metadata, dict):
        _merge_provenance(provenance, _parse_provenance_blob(metadata.get("provenance")))
        _merge_provenance(
            provenance,
            {key: metadata.get(key) for key in PROVENANCE_FIELDS if key in metadata},
        )

    if isinstance(record, dict):
        _merge_provenance(provenance, _parse_provenance_blob(record.get("provenance")))
        _merge_provenance(
            provenance,
            {key: record.get(key) for key in PROVENANCE_FIELDS if key in record},
        )

    if "map_hash" not in provenance and provenance.get("song_hash"):
        provenance["map_hash"] = str(provenance["song_hash"]).lower()
    return provenance


def _expected_record_samples(record, label):
    if "samples" not in record:
        raise RuntimeError(f"Shard record is missing samples: {label}")
    try:
        samples = int(record.get("samples"))
    except (TypeError, ValueError):
        raise RuntimeError(f"Shard record has invalid samples value in {label}: {record.get('samples')!r}")
    if samples <= 0:
        raise RuntimeError(f"Shard record has non-positive sample count in {label}: {samples}")
    return samples


def validate_shard_record(
    record,
    shard_root,
    expected_feature_dim,
    expected_target_dim,
    expected_dtype=torch.float16,
):
    label = shard_record_label(record, shard_root)
    paths = shard_record_paths(record, shard_root)
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Shard file missing for {label}: {', '.join(missing)}")

    payload = load_shard_payload(record, shard_root, device="cpu")
    shard_x = payload["x"]
    shard_y = payload["y"]
    if not torch.is_tensor(shard_x) or not torch.is_tensor(shard_y):
        raise RuntimeError(f"Shard loader did not return tensor pairs for {label}")
    if shard_x.ndim != 2 or shard_x.shape[1] != expected_feature_dim:
        raise RuntimeError(
            f"Shard feature shape mismatch in {label}: expected (*, {expected_feature_dim}), "
            f"got {tuple(shard_x.shape)}"
        )
    if shard_y.ndim != 2 or shard_y.shape[1] != expected_target_dim:
        raise RuntimeError(
            f"Shard target shape mismatch in {label}: expected (*, {expected_target_dim}), "
            f"got {tuple(shard_y.shape)}"
        )
    if shard_x.shape[0] != shard_y.shape[0]:
        raise RuntimeError(
            f"Shard row count mismatch in {label}: x has {shard_x.shape[0]}, y has {shard_y.shape[0]}"
        )

    expected_samples = _expected_record_samples(record, label)
    if shard_x.shape[0] != expected_samples:
        raise RuntimeError(
            f"Shard manifest row count mismatch in {label}: manifest has {expected_samples}, "
            f"file has {shard_x.shape[0]}"
        )

    for name, tensor in (("x", shard_x), ("y", shard_y)):
        if tensor.dtype != expected_dtype:
            raise RuntimeError(
                f"Shard dtype mismatch in {label}: {name} expected {expected_dtype}, got {tensor.dtype}"
            )
        if not torch.isfinite(tensor).all().item():
            raise RuntimeError(f"Shard contains non-finite values in {label}: tensor {name}")

    result = {
        "label": label,
        "samples": int(shard_x.shape[0]),
        "dtype": str(shard_x.dtype),
    }
    if payload.get("provenance"):
        result["provenance"] = payload["provenance"]
    return result


def _normalize_load_device(device):
    if device is None:
        return "cpu"
    if isinstance(device, torch.device):
        return str(device)
    return device


def _load_safetensors_payload(path, device):
    with safe_open(path, framework="pt", device="cpu") as shard_file:
        metadata = shard_file.metadata() or {}
    tensors = load_safetensors_file(path, device=device)
    return tensors, metadata


def load_shard_payload(record, shard_root, device="cpu"):
    load_device = _normalize_load_device(device)
    shard_path = record.get("shard_path")
    if shard_path:
        path = os.path.join(shard_root, shard_path.replace("/", os.sep))
        tensors, metadata = _load_safetensors_payload(path, load_device)
        if "x" not in tensors or "y" not in tensors:
            raise RuntimeError(f"Safetensors shard file is missing x/y tensors: {path}")
        payload = {
            "x": tensors["x"],
            "y": tensors["y"],
            "provenance": shard_record_provenance(record, metadata=metadata),
        }
        extra_tensors = {key: value for key, value in tensors.items() if key not in ("x", "y")}
        if extra_tensors:
            payload["extra_tensors"] = extra_tensors
        return payload

    x_path, y_path = shard_record_paths(record, shard_root)
    return {
        "x": torch.load(x_path, weights_only=True, map_location=load_device),
        "y": torch.load(y_path, weights_only=True, map_location=load_device),
        "provenance": shard_record_provenance(record),
    }


def load_shard_pair(record, shard_root, device="cpu"):
    payload = load_shard_payload(record, shard_root, device=device)
    return payload["x"], payload["y"]
