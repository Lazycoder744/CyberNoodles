import json
import os

import torch

from cybernoodles.core.network import ACTION_DIM, ActorCritic, normalize_pose_quaternions
from cybernoodles.data.dataset_builder import MANIFEST_PATH
from cybernoodles.data.dataset_builder import get_map_data
from cybernoodles.envs import get_eval_profile as resolve_eval_profile
from cybernoodles.envs import make_vector_env
from cybernoodles.training.policy_checkpoint import extract_policy_state_dict
from cybernoodles.training.eval_splits import filter_curriculum_by_split, normalize_hash

FPS = 60.0
DEFAULT_EVAL_SUITE = "starter"
POLICY_ACTION_ABS_CLAMP = 2.0
DEFAULT_HEAD_POS_CLAMP = 0.08
DEFAULT_HEAD_ROT_CLAMP = 0.045
DEFAULT_HAND_POS_CLAMP = 0.12
DEFAULT_HAND_ROT_CLAMP = 0.07


def _blend_actions(new_actions, last_actions, smoothing_alpha):
    """Use smoothing_alpha as retention of the previous action."""
    smoothing_alpha = float(max(0.0, min(1.0, smoothing_alpha)))
    if last_actions is None or smoothing_alpha >= 0.999:
        return new_actions
    return (smoothing_alpha * last_actions) + ((1.0 - smoothing_alpha) * new_actions)


def _coerce_time_vector(times, reference, *, default_zero=False):
    if times is None:
        return torch.zeros_like(reference, dtype=torch.float32) if default_zero else None
    vector = torch.as_tensor(times, dtype=torch.float32, device=reference.device).flatten()
    if vector.numel() != reference.numel():
        raise RuntimeError(
            f"Expected {reference.numel()} start/end times, received {vector.numel()}."
        )
    return vector


def sanitize_policy_actions(actions):
    """Return a simulator-safe pose action.

    This is the boundary transform for simulator inputs only. Policy-gradient
    code should keep the raw stochastic sample for Normal log-prob evaluation.
    """
    actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
    actions = normalize_pose_quaternions(actions)
    return torch.clamp(actions, -POLICY_ACTION_ABS_CLAMP, POLICY_ACTION_ABS_CLAMP)


def _validate_policy_action_tensor(name, tensor):
    if tensor.shape[-1] != ACTION_DIM:
        raise ValueError(f"{name} must end with ACTION_DIM={ACTION_DIM}; got shape {tuple(tensor.shape)}.")


def _effective_policy_std(std, exploration_scale):
    if isinstance(exploration_scale, torch.Tensor):
        scale = exploration_scale.to(device=std.device, dtype=std.dtype)
        if scale.dim() == 1 and std.dim() > 1 and scale.shape[0] == std.shape[0]:
            scale = scale.unsqueeze(-1)
        return std * scale

    scale = float(exploration_scale)
    if scale <= 0.0:
        raise ValueError("exploration_scale must be positive for stochastic policy sampling.")
    return std if scale == 1.0 else std * scale


def _normal_log_prob(mean, std, raw_action):
    reduced_precision = (torch.float16, torch.bfloat16)
    if mean.dtype in reduced_precision or std.dtype in reduced_precision or raw_action.dtype in reduced_precision:
        mean = mean.float()
        std = std.float()
        raw_action = raw_action.float()
    dist = torch.distributions.Normal(mean, std, validate_args=False)
    return dist.log_prob(raw_action).sum(dim=-1)


def policy_action_log_prob(mean, std, raw_action, exploration_scale=1.0):
    """Log-probability of an unsanitized action under the raw policy Normal."""
    _validate_policy_action_tensor("mean", mean)
    _validate_policy_action_tensor("std", std)
    _validate_policy_action_tensor("raw_action", raw_action)
    return _normal_log_prob(mean, _effective_policy_std(std, exploration_scale), raw_action)


def policy_action_transform_stats(raw_action, sim_action, *, tolerance=1e-6):
    raw_action = raw_action.detach()
    sim_action = sim_action.detach()
    delta = (sim_action - raw_action).abs()
    if delta.numel() == 0:
        zero = delta.new_zeros(())
        return {
            "mean_abs_delta": zero,
            "max_abs_delta": zero,
            "changed_fraction": zero,
        }
    return {
        "mean_abs_delta": delta.mean(),
        "max_abs_delta": delta.max(),
        "changed_fraction": (delta > float(tolerance)).to(delta.dtype).mean(),
    }


def sample_policy_action(mean, std, noise=None, exploration_scale=1.0, return_stats=False):
    """Sample a raw policy action and its simulator-safe counterpart.

    Returns (raw_action, sim_action, raw_log_prob, stats). Store raw_action for
    policy log-probs and pass sim_action to the simulator. noise is interpreted
    as a standard-Normal sample before exploration_scale is applied.
    """
    _validate_policy_action_tensor("mean", mean)
    _validate_policy_action_tensor("std", std)
    if noise is None:
        noise = torch.randn_like(mean)
    else:
        noise = torch.as_tensor(noise, dtype=mean.dtype, device=mean.device)

    effective_std = _effective_policy_std(std, exploration_scale)
    raw_action = mean + effective_std * noise
    sim_action = sanitize_policy_actions(raw_action)
    raw_log_prob = _normal_log_prob(mean, effective_std, raw_action)
    stats = policy_action_transform_stats(raw_action, sim_action) if return_stats else {}
    return raw_action, sim_action, raw_log_prob, stats


def _policy_action_delta_clamp(
    reference,
    *,
    pos_clamp=DEFAULT_HAND_POS_CLAMP,
    rot_clamp=DEFAULT_HAND_ROT_CLAMP,
    head_pos_clamp=DEFAULT_HEAD_POS_CLAMP,
    head_rot_clamp=DEFAULT_HEAD_ROT_CLAMP,
):
    clamp = torch.empty_like(reference)
    clamp[..., 0:3] = float(head_pos_clamp)
    clamp[..., 3:7] = float(head_rot_clamp)
    clamp[..., 7:10] = float(pos_clamp)
    clamp[..., 10:14] = float(rot_clamp)
    clamp[..., 14:17] = float(pos_clamp)
    clamp[..., 17:21] = float(rot_clamp)
    return clamp


def project_policy_action_to_simulator_envelope(
    actions,
    current_pose,
    *,
    pos_clamp=DEFAULT_HAND_POS_CLAMP,
    rot_clamp=DEFAULT_HAND_ROT_CLAMP,
    head_pos_clamp=DEFAULT_HEAD_POS_CLAMP,
    head_rot_clamp=DEFAULT_HEAD_ROT_CLAMP,
    return_stats=False,
):
    """Project pose-action targets into the simulator's per-step command clamp."""
    _validate_policy_action_tensor("actions", actions)
    _validate_policy_action_tensor("current_pose", current_pose)
    sim_actions = sanitize_policy_actions(actions)
    current_pose = normalize_pose_quaternions(
        torch.nan_to_num(current_pose, nan=0.0, posinf=0.0, neginf=0.0)
    )
    delta_clamp = _policy_action_delta_clamp(
        sim_actions,
        pos_clamp=pos_clamp,
        rot_clamp=rot_clamp,
        head_pos_clamp=head_pos_clamp,
        head_rot_clamp=head_rot_clamp,
    )
    projected = current_pose + (sim_actions - current_pose).clamp(-delta_clamp, delta_clamp)
    projected = sanitize_policy_actions(projected)
    stats = policy_action_transform_stats(sim_actions, projected) if return_stats else {}
    return projected, stats


def compute_target_note_counts(sim, start_times=None, end_times=None):
    note_mask = (
        (sim._note_range.unsqueeze(0) < sim.note_counts.unsqueeze(1))
        & (sim.note_types != 3)
    )

    start_vector = _coerce_time_vector(start_times, sim.current_times)
    if start_vector is not None:
        note_mask &= sim.note_times >= (start_vector * sim.bps).unsqueeze(1)

    end_vector = _coerce_time_vector(end_times, sim.current_times)
    if end_vector is not None:
        note_mask &= sim.note_times < (end_vector * sim.bps).unsqueeze(1)

    return note_mask.sum(1).float()


def compute_completion_ratios(sim, start_times=None):
    start_vector = _coerce_time_vector(start_times, sim.current_times, default_zero=True)
    remaining = (sim.map_durations - start_vector).clamp(min=1e-6)
    progressed = (sim.current_times - start_vector).clamp(min=0.0)
    return (progressed / remaining).clamp(0.0, 1.0)


def summarize_play_metrics(sim, *, start_times=None, end_times=None):
    target_counts = compute_target_note_counts(sim, start_times=start_times, end_times=end_times)
    hits = sim.total_hits.float()
    misses = sim.total_misses.float()
    engaged = sim.total_engaged_scorable.float()
    resolved = sim.total_resolved_scorable.float()
    total_scores = sim.total_cut_scores.float()
    style_samples = sim.speed_samples.float().clamp(min=1.0)
    motion_efficiency = (sim.useful_progress.float() / sim.motion_path.float().clamp(min=1e-6)).clamp(0.0, 1.0)
    mean_waste_motion = float((sim.waste_motion_sum.float() / style_samples).mean().item())
    mean_idle_motion = float((sim.idle_motion_sum.float() / style_samples).mean().item())
    mean_guard_error = float((sim.guard_error_sum.float() / style_samples).mean().item())
    mean_oscillation = float((sim.oscillation_sum.float() / style_samples).mean().item())
    mean_lateral_motion = float((sim.lateral_motion_sum.float() / style_samples).mean().item())
    mean_style_violation = float((sim.speed_violation_sum.float() / style_samples).mean().item())
    mean_angular_violation = float((sim.angular_violation_sum.float() / style_samples).mean().item())
    mean_motion_efficiency = float(motion_efficiency.mean().item())
    flail_index = (
        mean_waste_motion
        + 0.5 * mean_idle_motion
        + 0.5 * mean_oscillation
        + 0.5 * mean_lateral_motion
    )

    total_note_targets = float(max(1.0, target_counts.sum().item()))
    total_hits = float(hits.sum().item())
    total_misses = float(misses.sum().item())
    total_engaged = float(engaged.sum().item())
    total_resolved = float(resolved.sum().item())

    return {
        "accuracy": (total_hits / total_note_targets) * 100.0,
        "engaged_accuracy": (total_hits / max(1.0, total_engaged)) * 100.0,
        "resolved_accuracy": (total_hits / max(1.0, total_resolved)) * 100.0,
        "note_coverage": min(1.0, total_engaged / total_note_targets),
        "resolved_coverage": min(1.0, total_resolved / total_note_targets),
        "avg_cut": float(total_scores.sum().item()) / max(1.0, total_hits),
        "completion": float(compute_completion_ratios(sim, start_times=start_times).mean().item()),
        "clear_rate": float((sim._terminal_reason == 2).float().mean().item()) if hasattr(sim, "_terminal_reason") else 0.0,
        "fail_rate": float((sim._terminal_reason == 1).float().mean().item()) if hasattr(sim, "_terminal_reason") else 0.0,
        "timeout_rate": float((sim._terminal_reason == 3).float().mean().item()) if hasattr(sim, "_terminal_reason") else 0.0,
        "mean_target_notes": float(target_counts.mean().item()) if target_counts.numel() > 0 else 0.0,
        "mean_hits": float(hits.mean().item()) if hits.numel() > 0 else 0.0,
        "mean_misses": float(misses.mean().item()) if misses.numel() > 0 else 0.0,
        "mean_engaged_notes": float(engaged.mean().item()) if engaged.numel() > 0 else 0.0,
        "motion_efficiency": mean_motion_efficiency,
        "waste_motion": mean_waste_motion,
        "idle_motion": mean_idle_motion,
        "guard_error": mean_guard_error,
        "oscillation": mean_oscillation,
        "lateral_motion": mean_lateral_motion,
        "style_violation": mean_style_violation,
        "angular_violation": mean_angular_violation,
        "flail_index": flail_index,
    }


def remap_state_dict(state_dict, model):
    """Load weights with backward compatibility for old model layouts."""
    model_sd = model.state_dict()
    sd_keys = set(state_dict.keys())
    if sd_keys == set(model_sd.keys()):
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint shape mismatch for the current policy state layout. "
                "Rebuild BC shards and retrain BC so the checkpoint matches the updated INPUT_DIM."
            ) from exc
        return

    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("features."):
            for prefix in ("actor_features.", "critic_features."):
                new_key = key.replace("features.", prefix, 1)
                if new_key in model_sd and model_sd[new_key].shape == value.shape:
                    remapped[new_key] = value
        elif key == "critic":
            if "critic_head" in model_sd and model_sd["critic_head"].shape == value.shape:
                remapped["critic_head"] = value
        elif key in model_sd and model_sd[key].shape == value.shape:
            remapped[key] = value

    if not remapped:
        raise RuntimeError(
            "Checkpoint did not contain any usable model weights for the current policy state layout. "
            "Rebuild BC shards and retrain BC so the checkpoint matches the updated INPUT_DIM."
        )

    try:
        model.load_state_dict(remapped, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint shape mismatch for the current policy state layout. "
            "Rebuild BC shards and retrain BC so the checkpoint matches the updated INPUT_DIM."
        ) from exc


def load_actor_critic(model_path, device):
    model = ActorCritic().to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    remap_state_dict(
        extract_policy_state_dict(
            ckpt,
            checkpoint_path=model_path,
            accepted_keys=("model_state_dict", "actor_state_dict"),
            allow_legacy=True,
        ),
        model,
    )
    model.eval()
    return model


def load_replay_backed_hashes(manifest_path=MANIFEST_PATH):
    if not os.path.exists(manifest_path):
        return set()

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return {
            str(record.get("song_hash", "")).strip().lower()
            for record in manifest.get("shards", [])
            if record.get("song_hash")
        }
    except Exception:
        return set()


def _coerce_beatmap(beatmap):
    if beatmap and not isinstance(beatmap, dict):
        return {'notes': beatmap, 'obstacles': []}
    return beatmap or {'notes': [], 'obstacles': []}


def load_map_for_eval(map_hash, map_cache=None):
    if map_cache is not None and map_hash in map_cache:
        beatmap, bpm = map_cache[map_hash]
    else:
        beatmap, bpm = get_map_data(map_hash)
    return _coerce_beatmap(beatmap), bpm


def summarize_eval_map(map_hash, curriculum_entry=None, map_cache=None):
    beatmap, bpm = load_map_for_eval(map_hash, map_cache=map_cache)
    notes = beatmap.get('notes', [])
    obstacles = beatmap.get('obstacles', [])
    scorable_notes = sum(1 for note in notes if int(note.get('type', -1)) != 3)
    duration_sec = estimate_map_duration_seconds(beatmap, bpm)
    entry_nps = float((curriculum_entry or {}).get('nps', 0.0) or 0.0)
    derived_nps = scorable_notes / max(duration_sec, 1e-6) if duration_sec > 0.0 else 0.0
    return {
        "map_hash": map_hash,
        "beatmap": beatmap,
        "bpm": bpm,
        "notes": notes,
        "obstacles": obstacles,
        "scorable_notes": scorable_notes,
        "note_count": len(notes),
        "obstacle_count": len(obstacles),
        "duration_sec": duration_sec,
        "nps": entry_nps if entry_nps > 0.0 else derived_nps,
        "obstacle_ratio": len(obstacles) / max(1, scorable_notes),
    }


EVAL_SUITE_CONFIGS = {
    "starter": {
        "targets": [1.0, 1.6, 2.2, 2.8],
        "target_duration": 55.0,
        "target_notes": 180,
        "min_notes": 72,
        "min_duration": 22.0,
        "max_duration": 135.0,
        "max_obstacle_ratio": 0.45,
        "band_radius": 0.75,
    },
    "standard": {
        "targets": [1.4, 2.2, 3.0, 3.8],
        "target_duration": 65.0,
        "target_notes": 240,
        "min_notes": 96,
        "min_duration": 24.0,
        "max_duration": 150.0,
        "max_obstacle_ratio": 0.60,
        "band_radius": 1.00,
    },
    "mixed": {
        "targets": [1.2, 2.0, 2.8, 3.6, 4.4],
        "target_duration": 70.0,
        "target_notes": 260,
        "min_notes": 96,
        "min_duration": 24.0,
        "max_duration": 165.0,
        "max_obstacle_ratio": 0.70,
        "band_radius": 1.20,
    },
}


def choose_eval_hashes(
    curriculum,
    max_maps=3,
    preferred_hashes=None,
    map_cache=None,
    suite=DEFAULT_EVAL_SUITE,
    split=None,
    exclude_hashes=None,
    allow_split_fallback=True,
):
    curriculum = list(curriculum or [])
    if split is not None:
        split_curriculum = filter_curriculum_by_split(
            curriculum,
            split,
            allow_fallback=bool(allow_split_fallback),
        )
        curriculum = split_curriculum

    if not curriculum:
        return []

    config = EVAL_SUITE_CONFIGS.get(str(suite or DEFAULT_EVAL_SUITE).strip().lower())
    if config is None:
        raise ValueError(f"Unknown evaluation suite: {suite}")

    preferred = {normalize_hash(h) for h in (preferred_hashes or set())}
    excluded = {normalize_hash(h) for h in (exclude_hashes or set())}
    catalog = []
    for item in curriculum:
        map_hash = str(item.get('hash', '')).strip()
        normalized_hash = normalize_hash(map_hash)
        if not normalized_hash or normalized_hash in excluded:
            continue
        summary = summarize_eval_map(map_hash, curriculum_entry=item, map_cache=map_cache)
        if summary["scorable_notes"] <= 0 or summary["duration_sec"] <= 0.0:
            continue
        summary["preferred"] = normalize_hash(summary["map_hash"]) in preferred
        catalog.append(summary)

    if not catalog:
        return []

    def eligible(meta, relax_level):
        return (
            meta["scorable_notes"] >= max(32, config["min_notes"] - (24 * relax_level))
            and meta["duration_sec"] >= max(12.0, config["min_duration"] - (4.0 * relax_level))
            and meta["duration_sec"] <= config["max_duration"] + (20.0 * relax_level)
            and meta["obstacle_ratio"] <= config["max_obstacle_ratio"] + (0.20 * relax_level)
        )

    eligible_catalog = []
    relax_used = 0
    for relax_level in range(4):
        eligible_catalog = [meta for meta in catalog if eligible(meta, relax_level)]
        if eligible_catalog:
            relax_used = relax_level
            break
    if not eligible_catalog:
        eligible_catalog = list(catalog)
        relax_used = 4

    if max_maps == 1:
        middle_target = config["targets"][min(len(config["targets"]) - 1, max(0, len(config["targets"]) // 2 - 1))]
        best = min(
            eligible_catalog,
            key=lambda meta: (
                abs(meta["duration_sec"] - config["target_duration"]),
                meta["obstacle_ratio"],
                abs(meta["nps"] - middle_target),
                abs(meta["scorable_notes"] - config["target_notes"]),
                0 if meta["preferred"] else 1,
                meta["map_hash"],
            ),
        )
        return [best["map_hash"]]

    def candidate_key(meta, target_nps):
        return (
            abs(meta["nps"] - target_nps),
            abs(meta["duration_sec"] - config["target_duration"]),
            meta["obstacle_ratio"],
            abs(meta["scorable_notes"] - config["target_notes"]),
            0 if meta["preferred"] else 1,
            meta["map_hash"],
        )

    picks = []
    for target_nps in config["targets"]:
        if len(picks) >= max_maps:
            break
        pool = [meta for meta in eligible_catalog if meta["map_hash"] not in picks]
        if not pool:
            break
        radius = config["band_radius"] + (0.25 * relax_used)
        band_pool = [meta for meta in pool if abs(meta["nps"] - target_nps) <= radius]
        if not band_pool:
            band_pool = pool
        best = min(band_pool, key=lambda meta: candidate_key(meta, target_nps))
        picks.append(best["map_hash"])

    if len(picks) < max_maps:
        fallback = sorted(
            [meta for meta in eligible_catalog if meta["map_hash"] not in picks],
            key=lambda meta: candidate_key(meta, config["targets"][min(len(config["targets"]) - 1, len(picks))]),
        )
        for meta in fallback:
            picks.append(meta["map_hash"])
            if len(picks) >= max_maps:
                break

    return picks[:max_maps]


def estimate_map_duration_seconds(beatmap, bpm):
    if not beatmap or bpm is None:
        return 0.0
    notes = beatmap.get('notes', []) if isinstance(beatmap, dict) else []
    obstacles = beatmap.get('obstacles', []) if isinstance(beatmap, dict) else []
    if not notes and not obstacles:
        return 0.0
    bps = bpm / 60.0
    last_note = max((note['time'] for note in notes), default=0.0)
    last_obstacle = max((obs['time'] + obs.get('duration', 0.0) for obs in obstacles), default=0.0)
    return (max(last_note, last_obstacle) / max(1e-6, bps)) + 3.0


def pick_short_eval_hashes(map_hashes, map_cache=None, max_maps=3):
    ranked = []
    for map_hash in map_hashes:
        summary = summarize_eval_map(map_hash, map_cache=map_cache)
        duration = summary["duration_sec"] if summary["duration_sec"] > 0.0 else float('inf')
        ranked.append(
            (
                summary["scorable_notes"] < 72,
                summary["obstacle_ratio"] > 0.60,
                abs(duration - 55.0),
                summary["obstacle_ratio"],
                duration,
                map_hash,
            )
        )
    ranked.sort(key=lambda item: item)
    return [item[-1] for item in ranked[:max(1, min(max_maps, len(ranked)))]]


def get_eval_profile(profile):
    resolved = resolve_eval_profile(profile)
    tuning = resolved["sim_tuning"]
    return {
        "action_repeat": int(resolved["action_repeat"]),
        "smoothing_alpha": float(resolved["smoothing_alpha"]),
        "training_wheels_level": float(tuning.training_wheels),
        "assist_level": float(tuning.rehab_assists),
        "survival_assistance": float(tuning.survival_assistance),
        "stability_reward_level": float(tuning.stability_assistance),
        "style_guidance_level": float(tuning.style_guidance_level),
        "fail_enabled": bool(tuning.fail_enabled),
        "saber_inertia": float(tuning.saber_inertia),
        "rot_clamp": float(tuning.rot_clamp),
        "pos_clamp": float(tuning.pos_clamp),
    }


def load_curriculum(path="curriculum.json"):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_policy_model(
    model,
    device,
    map_hashes,
    map_cache=None,
    num_envs=64,
    noise_scale=0.0,
    action_repeat=1,
    smoothing_alpha=1.0,
    training_wheels_level=0.0,
    assist_level=0.0,
    survival_assistance=0.0,
    stability_reward_level=0.0,
    style_guidance_level=0.0,
    fail_enabled=True,
    saber_inertia=0.0,
    rot_clamp=DEFAULT_HAND_ROT_CLAMP,
    pos_clamp=DEFAULT_HAND_POS_CLAMP,
    label=None,
    verbose=False,
    start_time_sampler=None,
):
    results = []
    action_repeat = max(1, int(action_repeat))
    progress_interval_frames = max(1, int(15.0 * FPS))
    vector_env = make_vector_env(
        num_envs=num_envs,
        device=device,
        penalty_weights=(0.5, 0.0, 0.0, 0.0),
        dense_reward_scale=0.0,
        training_wheels=training_wheels_level,
        rehab_assists=assist_level,
        survival_assistance=survival_assistance,
        stability_assistance=stability_reward_level,
        style_guidance_level=style_guidance_level,
        fail_enabled=fail_enabled,
        saber_inertia=float(saber_inertia),
        rot_clamp=rot_clamp,
        pos_clamp=pos_clamp,
    )
    sim = vector_env.simulator

    for map_hash in map_hashes:
        summary = summarize_eval_map(map_hash, map_cache=map_cache)
        beatmap = summary["beatmap"]
        bpm = summary["bpm"]
        notes = summary["notes"]
        obstacles = summary["obstacles"]
        scorable_notes = summary["scorable_notes"]

        if not notes or bpm is None:
            continue

        duration_sec = summary["duration_sec"]
        num_frames = int(duration_sec * FPS) + 90

        if verbose:
            tag = f"[{label}] " if label else ""
            print(
                f"  {tag}map {map_hash[:8]}... "
                f"({duration_sec:.1f}s, {scorable_notes} scorable notes, {len(obstacles)} obstacles, {num_envs} envs)"
            )

        vector_env.load_maps([beatmap] * num_envs, [bpm] * num_envs)
        start_times = None
        if start_time_sampler is not None:
            sampled_times = start_time_sampler(summary, num_envs, device)
            if sampled_times is not None:
                start_times = torch.as_tensor(sampled_times, dtype=torch.float32, device=device).flatten()
                if start_times.numel() != int(num_envs):
                    raise RuntimeError(
                        f"start_time_sampler returned {start_times.numel()} entries for {num_envs} envs."
                    )
        sim.reset(start_times=start_times)

        actions = None
        last_actions = None
        frames_simulated = 0
        with torch.no_grad():
            for frame_idx in range(num_frames):
                if frame_idx % action_repeat == 0 or actions is None:
                    state = sim.get_states()
                    mean, std, _ = model(state)

                    if noise_scale > 0.0:
                        new_actions = mean + torch.randn_like(mean) * std * noise_scale
                    else:
                        new_actions = mean

                    actions = sanitize_policy_actions(_blend_actions(new_actions, last_actions, smoothing_alpha))
                    last_actions = actions

                sim.step(actions, dt=1.0 / FPS)
                frames_simulated = frame_idx + 1
                if verbose and frames_simulated % progress_interval_frames == 0:
                    tag = f"[{label}] " if label else ""
                    mean_completion = float(compute_completion_ratios(sim, start_times=start_times).mean().item())
                    done_ratio = sim.episode_done.float().mean().item()
                    print(
                        f"  {tag}progress {map_hash[:8]}... "
                        f"{frames_simulated / FPS:.1f}s/{duration_sec:.1f}s | "
                        f"done {done_ratio:.2f} | comp {mean_completion:.2f}"
                    )
                if bool(sim.episode_done.all()):
                    break

        metrics = summarize_play_metrics(sim, start_times=start_times)

        results.append({
            "map_hash": map_hash,
            "hits": metrics["mean_hits"],
            "misses": metrics["mean_misses"],
            "engaged_notes": metrics["mean_engaged_notes"],
            "total_notes": metrics["mean_target_notes"],
            "duration_sec": duration_sec,
            "nps": summary["nps"],
            "obstacle_count": len(obstacles),
            "obstacle_ratio": summary["obstacle_ratio"],
            "accuracy": metrics["accuracy"],
            "engaged_accuracy": metrics["engaged_accuracy"],
            "resolved_accuracy": metrics["resolved_accuracy"],
            "avg_cut": metrics["avg_cut"],
            "completion": metrics["completion"],
            "clear_rate": metrics["clear_rate"],
            "fail_rate": metrics["fail_rate"],
            "timeout_rate": metrics["timeout_rate"],
            "note_coverage": metrics["note_coverage"],
            "resolved_coverage": metrics["resolved_coverage"],
            "motion_efficiency": metrics["motion_efficiency"],
            "waste_motion": metrics["waste_motion"],
            "idle_motion": metrics["idle_motion"],
            "guard_error": metrics["guard_error"],
            "oscillation": metrics["oscillation"],
            "lateral_motion": metrics["lateral_motion"],
            "style_violation": metrics["style_violation"],
            "angular_violation": metrics["angular_violation"],
            "flail_index": metrics["flail_index"],
            "frames": frames_simulated,
        })

        if verbose:
            tag = f"[{label}] " if label else ""
            print(
                f"  {tag}done {map_hash[:8]}... "
                f"task {metrics['accuracy']:.2f}% | engaged {metrics['engaged_accuracy']:.2f}% | "
                f"resolved {metrics['resolved_accuracy']:.2f}% | clear {metrics['clear_rate']:.2f} | "
                f"comp {metrics['completion']:.2f} | cover {metrics['note_coverage']:.2f} | "
                f"resolved-cover {metrics['resolved_coverage']:.2f} | cut {metrics['avg_cut']:.2f} | "
                f"frames {frames_simulated}"
            )

    if not results:
        return {
            "maps": [],
            "mean_accuracy": 0.0,
            "mean_resolved_accuracy": 0.0,
            "mean_cut": 0.0,
            "mean_completion": 0.0,
            "mean_clear_rate": 0.0,
            "mean_fail_rate": 0.0,
            "mean_timeout_rate": 0.0,
            "mean_note_coverage": 0.0,
            "mean_resolved_coverage": 0.0,
            "mean_engaged_accuracy": 0.0,
            "mean_obstacle_ratio": 0.0,
            "mean_motion_efficiency": 0.0,
            "mean_waste_motion": 0.0,
            "mean_idle_motion": 0.0,
            "mean_guard_error": 0.0,
            "mean_oscillation": 0.0,
            "mean_lateral_motion": 0.0,
            "mean_style_violation": 0.0,
            "mean_angular_violation": 0.0,
            "mean_flail_index": 0.0,
        }

    return {
        "maps": results,
        "mean_accuracy": sum(r["accuracy"] for r in results) / len(results),
        "mean_resolved_accuracy": sum(r["resolved_accuracy"] for r in results) / len(results),
        "mean_engaged_accuracy": sum(r["engaged_accuracy"] for r in results) / len(results),
        "mean_cut": sum(r["avg_cut"] for r in results) / len(results),
        "mean_completion": sum(r["completion"] for r in results) / len(results),
        "mean_clear_rate": sum(r["clear_rate"] for r in results) / len(results),
        "mean_fail_rate": sum(r["fail_rate"] for r in results) / len(results),
        "mean_timeout_rate": sum(r["timeout_rate"] for r in results) / len(results),
        "mean_note_coverage": sum(r["note_coverage"] for r in results) / len(results),
        "mean_resolved_coverage": sum(r["resolved_coverage"] for r in results) / len(results),
        "mean_obstacle_ratio": sum(r["obstacle_ratio"] for r in results) / len(results),
        "mean_motion_efficiency": sum(r["motion_efficiency"] for r in results) / len(results),
        "mean_waste_motion": sum(r["waste_motion"] for r in results) / len(results),
        "mean_idle_motion": sum(r["idle_motion"] for r in results) / len(results),
        "mean_guard_error": sum(r["guard_error"] for r in results) / len(results),
        "mean_oscillation": sum(r["oscillation"] for r in results) / len(results),
        "mean_lateral_motion": sum(r["lateral_motion"] for r in results) / len(results),
        "mean_style_violation": sum(r["style_violation"] for r in results) / len(results),
        "mean_angular_violation": sum(r["angular_violation"] for r in results) / len(results),
        "mean_flail_index": sum(r["flail_index"] for r in results) / len(results),
    }
