from cybernoodles.core.network import (
    ACTION_DIM,
    CURRENT_POSE_END,
    CURRENT_POSE_START,
    INPUT_DIM,
    NOTE_FEATURES,
    OBSTACLE_FEATURES,
    POSE_DIM,
    STATE_FRAME_DIM,
    STATE_HISTORY_OFFSETS,
)
from cybernoodles.data.dataset_builder import (
    MANIFEST_VERSION,
    NOTE_FEATURE_LAYOUT,
    NOTE_TIME_FEATURE_MAX_BEATS,
    NOTE_TIME_FEATURE_MIN_BEATS,
    SIM_NOTE_TIME_MAX_BEATS,
    SIM_NOTE_TIME_MIN_BEATS,
    SIM_OBSTACLE_TIME_MAX_BEATS,
    SIM_OBSTACLE_TIME_MIN_BEATS,
    SIM_SAMPLE_HZ,
    TARGET_POSE_HORIZON_FRAMES,
)


POLICY_SCHEMA_VERSION = 2


def current_policy_schema():
    return {
        "policy_schema_version": POLICY_SCHEMA_VERSION,
        "input_dim": INPUT_DIM,
        "action_dim": ACTION_DIM,
        "pose_dim": POSE_DIM,
        "note_features": NOTE_FEATURES,
        "obstacle_features": OBSTACLE_FEATURES,
        "state_frame_dim": STATE_FRAME_DIM,
        "state_history_offsets": tuple(int(v) for v in STATE_HISTORY_OFFSETS),
        "current_pose_start": CURRENT_POSE_START,
        "current_pose_end": CURRENT_POSE_END,
        "bc_manifest_version": MANIFEST_VERSION,
        "bc_target_pose_horizon_frames": TARGET_POSE_HORIZON_FRAMES,
        "bc_sample_hz": float(SIM_SAMPLE_HZ),
        "note_feature_layout": NOTE_FEATURE_LAYOUT,
        "sim_note_time_range_beats": (
            float(SIM_NOTE_TIME_MIN_BEATS),
            float(SIM_NOTE_TIME_MAX_BEATS),
        ),
        "note_time_feature_range_beats": (
            float(NOTE_TIME_FEATURE_MIN_BEATS),
            float(NOTE_TIME_FEATURE_MAX_BEATS),
        ),
        "sim_obstacle_time_range_beats": (
            float(SIM_OBSTACLE_TIME_MIN_BEATS),
            float(SIM_OBSTACLE_TIME_MAX_BEATS),
        ),
    }


def attach_policy_schema(payload):
    enriched = dict(payload)
    enriched["policy_schema"] = current_policy_schema()
    return enriched


def _normalize_schema_value(expected_value, actual_value):
    if isinstance(expected_value, tuple):
        if isinstance(actual_value, (list, tuple)):
            return tuple(actual_value)
    return actual_value


def policy_schema_mismatches(schema):
    if not isinstance(schema, dict):
        return ["missing policy_schema metadata"]

    mismatches = []
    for key, expected_value in current_policy_schema().items():
        actual_value = _normalize_schema_value(expected_value, schema.get(key))
        if actual_value != expected_value:
            mismatches.append(
                f"{key}: expected {expected_value!r}, got {actual_value!r}"
            )
    return mismatches


def validate_policy_checkpoint_payload(payload, checkpoint_path=None, required_keys=(), allow_legacy=False):
    if not isinstance(payload, dict):
        raise RuntimeError("Policy checkpoint payload must be a dictionary.")

    schema = payload.get("policy_schema")
    if schema is None:
        if allow_legacy:
            schema = None
        else:
            location = checkpoint_path or "checkpoint"
            raise RuntimeError(
                f"{location} predates the current policy schema metadata. "
                "Rebuild BC shards and retrain BC/RL before reusing this checkpoint."
            )
    if schema is not None:
        mismatches = policy_schema_mismatches(schema)
        if mismatches:
            location = checkpoint_path or "checkpoint"
            detail = "; ".join(mismatches[:4])
            raise RuntimeError(
                f"{location} was built for an incompatible policy state schema. "
                f"{detail}. Rebuild BC shards and retrain BC/RL before reusing it."
            )

    for key in required_keys:
        if key not in payload:
            location = checkpoint_path or "checkpoint"
            raise RuntimeError(f"{location} is missing required key: {key}")

    return payload


def extract_policy_state_dict(payload, checkpoint_path=None, accepted_keys=("model_state_dict",), allow_legacy=False):
    validated = validate_policy_checkpoint_payload(
        payload,
        checkpoint_path=checkpoint_path,
        allow_legacy=allow_legacy,
    )

    for key in accepted_keys:
        state_dict = validated.get(key)
        if isinstance(state_dict, dict):
            return state_dict

    if allow_legacy and validated and all(hasattr(value, "shape") for value in validated.values()):
        return validated

    location = checkpoint_path or "checkpoint"
    accepted = ", ".join(accepted_keys)
    raise RuntimeError(f"{location} did not contain any of: {accepted}")
