import hashlib
import json
from pathlib import Path

from cybernoodles.paths import DATA_DIR


SPLIT_TRAIN = "train"
SPLIT_DEV_EVAL = "dev_eval"
SPLIT_SEALED_EVAL = "sealed_eval"
SPLIT_PROGRESS_REPLAY = "progress_replay"
SPLIT_REPLAY_REGRESSION = "replay_regression"

SPLIT_NAMES = (
    SPLIT_TRAIN,
    SPLIT_DEV_EVAL,
    SPLIT_SEALED_EVAL,
    SPLIT_PROGRESS_REPLAY,
    SPLIT_REPLAY_REGRESSION,
)
DEFAULT_SPLITS_PATH = DATA_DIR / "eval_splits.json"

_STABLE_BUCKETS = (
    (80, SPLIT_TRAIN),
    (90, SPLIT_DEV_EVAL),
    (96, SPLIT_SEALED_EVAL),
    (98, SPLIT_PROGRESS_REPLAY),
    (100, SPLIT_REPLAY_REGRESSION),
)


HASH_FIELDS = ("hash", "map_hash", "song_hash")


def normalize_hash(map_hash):
    return str(map_hash or "").strip().lower()


def map_hash_from_record(record):
    if isinstance(record, dict):
        sources = [record]
        provenance = record.get("provenance")
        if isinstance(provenance, dict):
            sources.append(provenance)
        for source in sources:
            for field in HASH_FIELDS:
                normalized = normalize_hash(source.get(field))
                if normalized:
                    return normalized
        return ""
    return normalize_hash(record)


def normalize_split_name(split):
    split_name = str(split or "").strip().lower().replace("-", "_")
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"Unknown evaluation split: {split}")
    return split_name


def stable_split_for_hash(map_hash):
    normalized = map_hash_from_record(map_hash)
    if not normalized:
        return None
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    for upper_bound, split_name in _STABLE_BUCKETS:
        if bucket < upper_bound:
            return split_name
    return SPLIT_REPLAY_REGRESSION


def _split_payload(raw_payload):
    if not isinstance(raw_payload, dict):
        raise ValueError("eval_splits.json must contain a JSON object.")
    payload = raw_payload.get("splits", raw_payload)
    if not isinstance(payload, dict):
        raise ValueError("eval_splits.json 'splits' field must be an object.")
    return payload


def _coerce_split_sets(raw_splits):
    splits = {name: set() for name in SPLIT_NAMES}
    for raw_name, values in raw_splits.items():
        split_name = normalize_split_name(raw_name)
        if values is None:
            continue
        if not isinstance(values, (list, tuple, set)):
            raise ValueError(f"Split {split_name!r} must be a list of map hashes.")
        for value in values:
            normalized = map_hash_from_record(value)
            if normalized:
                splits[split_name].add(normalized)
    return splits


def validate_eval_splits(splits):
    seen = {}
    for raw_name, hashes in (splits or {}).items():
        split_name = normalize_split_name(raw_name)
        for map_hash in hashes or set():
            normalized = map_hash_from_record(map_hash)
            if not normalized:
                continue
            previous = seen.get(normalized)
            if previous is not None and previous != split_name:
                raise ValueError(
                    f"Map hash {normalized} appears in both {previous!r} and {split_name!r} splits."
                )
            seen[normalized] = split_name
    return True


def load_eval_splits(path=None):
    split_path = Path(path) if path is not None else DEFAULT_SPLITS_PATH
    if not split_path.exists():
        return {name: set() for name in SPLIT_NAMES}
    with split_path.open("r", encoding="utf-8") as split_file:
        raw_payload = json.load(split_file)
    splits = _coerce_split_sets(_split_payload(raw_payload))
    validate_eval_splits(splits)
    return splits


def split_name_for_hash(map_hash, splits=None):
    normalized = map_hash_from_record(map_hash)
    if not normalized:
        return None
    split_sets = load_eval_splits() if splits is None else _coerce_split_sets(splits)
    validate_eval_splits(split_sets)
    for split_name, hashes in split_sets.items():
        if normalized in hashes:
            return split_name
    return stable_split_for_hash(normalized)


def split_hashes_for(curriculum, split, splits=None):
    split_name = normalize_split_name(split)
    split_sets = load_eval_splits() if splits is None else _coerce_split_sets(splits)
    validate_eval_splits(split_sets)
    hashes = []
    for item in curriculum or []:
        map_hash = map_hash_from_record(item)
        if map_hash and split_name_for_hash(map_hash, splits=split_sets) == split_name:
            hashes.append(map_hash)
    return hashes


def filter_curriculum_by_split(curriculum, split, *, splits=None, allow_fallback=False):
    records = list(curriculum or [])
    split_name = normalize_split_name(split)
    split_sets = load_eval_splits() if splits is None else _coerce_split_sets(splits)
    validate_eval_splits(split_sets)
    filtered = [
        item for item in records
        if split_name_for_hash(item, splits=split_sets) == split_name
    ]
    if not filtered and allow_fallback:
        return records
    return filtered
