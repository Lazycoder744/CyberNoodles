import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from cybernoodles.core.network import ACTION_DIM, INPUT_DIM, ActorCritic, build_rl_bootstrap_state_dict, normalize_state
from cybernoodles.data.dataset_builder import get_map_data
from cybernoodles.envs import apply_simulator_tuning, build_awac_training_tuning, make_simulator
from cybernoodles.training.policy_eval import (
    choose_eval_hashes,
    evaluate_policy_model,
    get_eval_profile,
    load_replay_backed_hashes,
    policy_action_log_prob,
    sample_policy_action,
    remap_state_dict,
    sanitize_policy_actions,
    summarize_play_metrics,
    summarize_eval_map,
)
from cybernoodles.training.eval_splits import filter_curriculum_by_split
from cybernoodles.training.policy_checkpoint import (
    attach_policy_schema,
    extract_policy_state_dict,
    validate_policy_checkpoint_payload,
)
from cybernoodles.training.watchdog import (
    assert_finite_gradients,
    assert_finite_module,
    ensure_finite_scalar,
    ensure_optimizer_advanced,
    ensure_parameter_moved,
    optimizer_step_total,
    parameter_delta_l2,
    parameter_snapshot,
)
from cybernoodles.training.train_bc import (
    bc_pose_loss,
    iter_shard_batches,
    load_manifest,
    split_records,
    suggest_batch_size,
    take_record_subset,
)

BC_MODEL_PATH = "bsai_bc_model.pth"
AWAC_MODEL_PATH = "bsai_awac_model.pth"
AWAC_CHECKPOINT_PATH = "bsai_awac_checkpoint.pth"
AWAC_STATE_PATH = "awac_state.json"
CURRICULUM_PATH = "curriculum.json"
FPS = 60.0
BOOTSTRAP_MIN_COVERAGE = 0.05
BOOTSTRAP_MIN_ACCURACY = 3.0
BOOTSTRAP_MIN_RESOLVED_ACCURACY = 25.0
BOOTSTRAP_MIN_CLEAR_RATE = 0.02
BOOTSTRAP_MIN_COMPLETION = 0.10
STRICT_BOOTSTRAP_MIN_COVERAGE = 0.01
STRICT_BOOTSTRAP_MIN_ACCURACY = 0.5
STRICT_BOOTSTRAP_MIN_RESOLVED_ACCURACY = 12.0
STRICT_BOOTSTRAP_MIN_COMPLETION = 0.05
BOOTSTRAP_GATE_EPS = 1e-3
AWAC_RESUME_REQUIRED_KEYS = (
    "actor_state_dict",
    "critic_state_dict",
    "target_actor_state_dict",
    "target_critic_state_dict",
    "actor_optimizer_state_dict",
    "critic_optimizer_state_dict",
)


def make_adam(params, lr):
    try:
        return optim.Adam(params, lr=lr, fused=True)
    except TypeError:
        return optim.Adam(params, lr=lr)


class TwinQCritic(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = INPUT_DIM + ACTION_DIM
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, states, actions):
        sa = torch.cat((normalize_state(states), actions), dim=-1)
        return self.q1(sa), self.q2(sa)

    def min_q(self, states, actions):
        q1, q2 = self.forward(states, actions)
        return torch.minimum(q1, q2)


class ReplayBuffer:
    def __init__(self, capacity, device, storage_dtype=torch.float16):
        self.capacity = int(max(1, capacity))
        self.device = device
        self.storage_dtype = storage_dtype
        self.states = torch.empty((self.capacity, INPUT_DIM), dtype=storage_dtype)
        self.raw_actions = torch.empty((self.capacity, ACTION_DIM), dtype=storage_dtype)
        self.sim_actions = torch.empty((self.capacity, ACTION_DIM), dtype=storage_dtype)
        self.behavior_log_probs = torch.empty((self.capacity, 1), dtype=torch.float32)
        self.next_states = torch.empty((self.capacity, INPUT_DIM), dtype=storage_dtype)
        self.rewards = torch.empty((self.capacity, 1), dtype=torch.float32)
        self.dones = torch.empty((self.capacity, 1), dtype=torch.float32)
        self.ptr = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add_batch(self, states, raw_actions, sim_actions, behavior_log_probs, rewards, next_states, dones):
        count = int(states.shape[0])
        if count <= 0:
            return
        if count >= self.capacity:
            states = states[-self.capacity:]
            raw_actions = raw_actions[-self.capacity:]
            sim_actions = sim_actions[-self.capacity:]
            behavior_log_probs = behavior_log_probs[-self.capacity:]
            rewards = rewards[-self.capacity:]
            next_states = next_states[-self.capacity:]
            dones = dones[-self.capacity:]
            count = self.capacity

        end = self.ptr + count
        if end <= self.capacity:
            sl = slice(self.ptr, end)
            self.states[sl].copy_(states)
            self.raw_actions[sl].copy_(raw_actions)
            self.sim_actions[sl].copy_(sim_actions)
            self.behavior_log_probs[sl].copy_(behavior_log_probs)
            self.rewards[sl].copy_(rewards)
            self.next_states[sl].copy_(next_states)
            self.dones[sl].copy_(dones)
        else:
            first = self.capacity - self.ptr
            second = count - first
            self.states[self.ptr:].copy_(states[:first])
            self.raw_actions[self.ptr:].copy_(raw_actions[:first])
            self.sim_actions[self.ptr:].copy_(sim_actions[:first])
            self.behavior_log_probs[self.ptr:].copy_(behavior_log_probs[:first])
            self.rewards[self.ptr:].copy_(rewards[:first])
            self.next_states[self.ptr:].copy_(next_states[:first])
            self.dones[self.ptr:].copy_(dones[:first])
            self.states[:second].copy_(states[first:])
            self.raw_actions[:second].copy_(raw_actions[first:])
            self.sim_actions[:second].copy_(sim_actions[first:])
            self.behavior_log_probs[:second].copy_(behavior_log_probs[first:])
            self.rewards[:second].copy_(rewards[first:])
            self.next_states[:second].copy_(next_states[first:])
            self.dones[:second].copy_(dones[first:])

        self.ptr = (self.ptr + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, batch_size):
        if self.size <= 0:
            raise RuntimeError("Replay buffer is empty.")
        indices = torch.randint(0, self.size, (int(batch_size),))
        return {
            "states": self.states[indices].to(device=self.device, dtype=torch.float32),
            "raw_actions": self.raw_actions[indices].to(device=self.device, dtype=torch.float32),
            "sim_actions": self.sim_actions[indices].to(device=self.device, dtype=torch.float32),
            "behavior_log_probs": self.behavior_log_probs[indices].to(device=self.device, dtype=torch.float32),
            "rewards": self.rewards[indices].to(device=self.device, dtype=torch.float32),
            "next_states": self.next_states[indices].to(device=self.device, dtype=torch.float32),
            "dones": self.dones[indices].to(device=self.device, dtype=torch.float32),
        }


class DemoBatchStream:
    def __init__(self, records, batch_size, device):
        self.records = list(records)
        self.batch_size = int(batch_size)
        self.device = device
        self._iterator = None
        self._prefetched = None

    def _reset(self):
        self._iterator = iter_shard_batches(
            self.records,
            self.batch_size,
            shuffle=True,
            pin_memory=(self.device.type == "cuda"),
        )

    def next(self):
        if self._prefetched is not None:
            batch = self._prefetched
            self._prefetched = None
            return batch
        if not self.records or self.batch_size <= 0:
            return None
        if self._iterator is None:
            self._reset()
        try:
            batch_x, batch_y = next(self._iterator)
        except StopIteration:
            self._reset()
            batch_x, batch_y = next(self._iterator)
        return (
            batch_x.to(device=self.device, dtype=torch.float32, non_blocking=True),
            batch_y.to(device=self.device, dtype=torch.float32, non_blocking=True),
        )

    def prefetch(self):
        batch = self.next()
        self._prefetched = batch
        return batch


def build_required_demo_stream(args, device):
    if bool(getattr(args, "disable_demo_regularizer", False)):
        print("Demo regularizer explicitly disabled via --disable-demo-regularizer.")
        return None, 0

    if float(args.demo_bc_weight) <= 0.0:
        raise RuntimeError(
            "Demo regularizer is enabled by default, but --demo-bc-weight is <= 0. "
            "Use --disable-demo-regularizer to opt out explicitly."
        )
    if int(args.demo_batch_size) <= 0:
        raise RuntimeError(
            "Demo regularizer is enabled by default, but --demo-batch-size is <= 0. "
            "Use --disable-demo-regularizer to opt out explicitly."
        )

    manifest = load_manifest()
    if manifest is None:
        raise RuntimeError(
            "Demo regularizer requires a compatible BC shard manifest. "
            "Build BC shards or use --disable-demo-regularizer to opt out explicitly."
        )

    train_records = split_records(manifest, "train")
    if int(args.demo_shards_limit) > 0:
        train_records = take_record_subset(train_records, args.demo_shards_limit, seed=int(args.seed))
    if not train_records:
        raise RuntimeError(
            "Demo regularizer found no train split shard records. "
            "Regenerate the manifest or use --disable-demo-regularizer to opt out explicitly."
        )

    demo_stream = DemoBatchStream(train_records, args.demo_batch_size, device=device)
    try:
        sample = demo_stream.prefetch()
    except Exception as exc:
        raise RuntimeError(
            "Demo regularizer could not read a compatible BC shard batch. "
            "Regenerate BC shards or use --disable-demo-regularizer to opt out explicitly."
        ) from exc
    if sample is None:
        raise RuntimeError(
            "Demo regularizer produced no BC shard batch. "
            "Regenerate BC shards or use --disable-demo-regularizer to opt out explicitly."
        )

    demo_x, demo_y = sample
    if demo_x.ndim != 2 or demo_x.shape[-1] != INPUT_DIM or demo_y.ndim != 2 or demo_y.shape[-1] != ACTION_DIM:
        raise RuntimeError(
            "Demo regularizer shard batch has incompatible shapes: "
            f"x {tuple(demo_x.shape)}, y {tuple(demo_y.shape)}; expected (*, {INPUT_DIM}) and (*, {ACTION_DIM})."
        )
    if not torch.isfinite(demo_x).all() or not torch.isfinite(demo_y).all():
        raise RuntimeError("Demo regularizer shard batch contains non-finite values.")

    print(
        f"Demo regularizer: {len(train_records)} shard records | batch {args.demo_batch_size} | "
        f"weight {float(args.demo_bc_weight):.3f}"
    )
    return demo_stream, len(train_records)


def soft_update(target, source, tau):
    tau = float(tau)
    one_minus_tau = 1.0 - tau
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.mul_(one_minus_tau).add_(source_param, alpha=tau)


def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def pick_default_num_envs(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return 32
    device_index = device.index if device.index is not None else 0
    total_gb = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
    if total_gb >= 20.0:
        return 384
    if total_gb >= 12.0:
        return 256
    if total_gb >= 8.0:
        return 192
    return 96


def choose_update_count(collected_steps, batch_size):
    batch_size = max(1, int(batch_size))
    samples_per_epoch = max(1, int(collected_steps))
    return max(16, min(160, (samples_per_epoch // batch_size) * 3))


def load_actor_weights(model, path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    remap_state_dict(
        extract_policy_state_dict(
            payload,
            checkpoint_path=path,
            accepted_keys=("model_state_dict", "actor_state_dict"),
            allow_legacy=True,
        ),
        model,
    )


def read_awac_resume(device):
    if not os.path.exists(AWAC_CHECKPOINT_PATH):
        return None
    try:
        payload = torch.load(AWAC_CHECKPOINT_PATH, map_location=device, weights_only=False)
    except Exception:
        return None
    validate_policy_checkpoint_payload(
        payload,
        checkpoint_path=AWAC_CHECKPOINT_PATH,
        required_keys=AWAC_RESUME_REQUIRED_KEYS,
    )
    return payload


def load_curriculum_records():
    curriculum = load_json(CURRICULUM_PATH, [])
    if not isinstance(curriculum, list):
        return []
    return curriculum


def build_training_pool(curriculum, replay_backed_hashes, pool_limit=None):
    pool = []
    for item in curriculum:
        map_hash = str(item.get("hash", "")).strip()
        if not map_hash:
            continue
        beatmap, bpm = get_map_data(map_hash)
        notes = beatmap.get("notes", []) if beatmap else []
        if not beatmap or not notes or bpm is None:
            continue
        summary = summarize_eval_map(map_hash, curriculum_entry=item, map_cache={map_hash: (beatmap, bpm)})
        scorable_seconds = [
            float(note["time"]) / max(1e-6, bpm / 60.0)
            for note in notes
            if int(note.get("type", -1)) != 3
        ]
        if not scorable_seconds:
            continue
        nps = float(item.get("nps", summary["nps"]) or summary["nps"] or 0.0)
        preferred = map_hash.lower() in replay_backed_hashes
        weight = 1.0
        weight *= 1.45 if preferred else 1.0
        weight *= 1.0 / max(0.45, abs(nps - 2.1) + 0.55)
        if summary["obstacle_ratio"] <= 0.35:
            weight *= 1.15
        if summary["duration_sec"] > 110.0:
            weight *= 0.8
        if summary["scorable_notes"] < 56:
            weight *= 0.75
        pool.append({
            "hash": map_hash,
            "beatmap": beatmap,
            "bpm": bpm,
            "nps": nps,
            "preferred": preferred,
            "duration_sec": summary["duration_sec"],
            "obstacle_ratio": summary["obstacle_ratio"],
            "scorable_notes": summary["scorable_notes"],
            "scorable_seconds": scorable_seconds,
            "weight": max(0.05, weight),
            "note_count": len(beatmap.get("notes", [])),
            "obstacle_count": len(beatmap.get("obstacles", [])),
        })

    pool.sort(
        key=lambda meta: (
            0 if meta["preferred"] else 1,
            meta["nps"] > 3.8,
            abs(meta["nps"] - 2.1),
            meta["obstacle_ratio"],
            abs(meta["duration_sec"] - 60.0),
            meta["hash"],
        )
    )
    if pool_limit is not None and pool_limit > 0:
        pool = pool[: int(pool_limit)]
    return pool


def current_strict_progress(trainer_state):
    trainer_state = trainer_state or {}
    last_accuracy = float(trainer_state.get("last_strict_accuracy", 0.0) or 0.0)
    last_coverage = float(trainer_state.get("last_strict_note_coverage", 0.0) or 0.0)
    if last_accuracy > 0.0 or last_coverage > 0.0:
        return last_accuracy, last_coverage

    best_accuracy = float(trainer_state.get("best_strict_accuracy", 0.0) or 0.0)
    best_coverage = float(trainer_state.get("best_strict_coverage", trainer_state.get("best_strict_note_coverage", 0.0)) or 0.0)
    return best_accuracy, best_coverage


def choose_training_stage(trainer_state, args):
    strict_accuracy, strict_coverage = current_strict_progress(trainer_state)
    if (
        strict_coverage >= float(args.strict_expand_coverage)
        and strict_accuracy >= float(args.strict_expand_accuracy)
    ):
        return "full", strict_accuracy, strict_coverage
    if (
        strict_coverage >= float(args.strict_unlock_coverage)
        and strict_accuracy >= float(args.strict_unlock_accuracy)
    ):
        return "bridge", strict_accuracy, strict_coverage
    return "warmup", strict_accuracy, strict_coverage


def build_active_training_pool(full_pool, trainer_state, args):
    if not full_pool:
        return [], "empty", 0.0, 0.0

    stage, strict_accuracy, strict_coverage = choose_training_stage(trainer_state, args)
    preferred = [meta for meta in full_pool if meta["preferred"]]
    fallback = list(full_pool)

    if stage == "warmup":
        pool = [
            meta for meta in preferred
            if meta["nps"] <= 2.8 and meta["obstacle_ratio"] <= 0.25 and meta["duration_sec"] <= 95.0
        ]
        if len(pool) < max(2, min(len(preferred), int(args.warmup_pool_size))):
            pool = [
                meta for meta in preferred
                if meta["nps"] <= 3.2 and meta["obstacle_ratio"] <= 0.35 and meta["duration_sec"] <= 120.0
            ]
        if not pool:
            pool = preferred or fallback
        pool = pool[: max(1, int(args.warmup_pool_size))]
        return pool, stage, strict_accuracy, strict_coverage

    if stage == "bridge":
        bridge_target = max(int(args.warmup_pool_size), min(int(args.train_pool_size), int(args.warmup_pool_size) * 2))
        pool = list(preferred[:bridge_target])
        if len(pool) < bridge_target:
            seen = {meta["hash"] for meta in pool}
            for meta in fallback:
                if meta["hash"] in seen:
                    continue
                pool.append(meta)
                seen.add(meta["hash"])
                if len(pool) >= bridge_target:
                    break
        return pool, stage, strict_accuracy, strict_coverage

    return fallback[: max(1, int(args.train_pool_size))], stage, strict_accuracy, strict_coverage


def select_training_batch(pool, num_envs):
    weights = [meta["weight"] for meta in pool]
    return random.choices(pool, weights=weights, k=int(num_envs))


def sample_start_times(map_batch, device):
    starts = []
    for meta in map_batch:
        scorable_seconds = meta["scorable_seconds"]
        if not scorable_seconds:
            starts.append(0.0)
            continue
        focus_count = max(1, min(len(scorable_seconds), int(math.ceil(len(scorable_seconds) * 0.40))))
        anchor = scorable_seconds[random.randrange(focus_count)]
        lead = random.uniform(0.35, 0.90)
        latest_start = max(0.0, float(meta["duration_sec"]) - 6.0)
        starts.append(min(latest_start, max(0.0, anchor - lead)))
    return torch.tensor(starts, dtype=torch.float32, device=device)


def configure_training_sim(sim, args):
    tuning = build_awac_training_tuning(args)
    apply_simulator_tuning(sim, tuning)


def normalize_rollout_metrics(metrics):
    metrics = dict(metrics or {})
    if "task_accuracy" not in metrics:
        metrics["task_accuracy"] = float(metrics.get("accuracy", 0.0))
    if "accuracy" not in metrics:
        metrics["accuracy"] = float(metrics.get("task_accuracy", 0.0))
    return metrics


def update_trainer_state_from_eval(
    trainer_state,
    strict_summary,
    matched_summary,
    *,
    best_strict_accuracy=0.0,
    best_strict_coverage=0.0,
):
    trainer_state = dict(trainer_state or {})
    strict_summary = strict_summary or {}
    matched_summary = matched_summary or {}

    strict_accuracy = float(strict_summary.get("mean_accuracy", 0.0))
    strict_coverage = float(strict_summary.get("mean_note_coverage", 0.0))
    matched_accuracy = float(matched_summary.get("mean_accuracy", 0.0))
    matched_coverage = float(matched_summary.get("mean_note_coverage", 0.0))

    trainer_state.update({
        "last_strict_accuracy": strict_accuracy,
        "last_strict_note_coverage": strict_coverage,
        "last_matched_accuracy": matched_accuracy,
        "last_matched_note_coverage": matched_coverage,
    })

    best_strict_accuracy = max(
        float(best_strict_accuracy or 0.0),
        float(trainer_state.get("best_strict_accuracy", 0.0) or 0.0),
    )
    best_strict_coverage = max(
        float(best_strict_coverage or 0.0),
        float(trainer_state.get("best_strict_coverage", trainer_state.get("best_strict_note_coverage", 0.0)) or 0.0),
    )
    if strict_accuracy > best_strict_accuracy:
        best_strict_accuracy = strict_accuracy
    if strict_coverage > best_strict_coverage:
        best_strict_coverage = strict_coverage

    trainer_state["best_strict_accuracy"] = best_strict_accuracy
    trainer_state["best_strict_coverage"] = best_strict_coverage
    return trainer_state, best_strict_accuracy, best_strict_coverage


def collect_rollout(actor, sim, args, map_batch):
    beatmaps = [meta["beatmap"] for meta in map_batch]
    bpms = [meta["bpm"] for meta in map_batch]
    start_times = sample_start_times(map_batch, sim.device)
    sim.load_maps(beatmaps, bpms)
    configure_training_sim(sim, args)
    sim.reset(start_times=start_times)

    states = []
    raw_actions = []
    sim_actions = []
    behavior_log_probs = []
    rewards = []
    next_states = []
    dones = []
    active_masks = []

    prev_done = sim.episode_done.clone()
    frames = 0
    with torch.no_grad():
        for step in range(int(args.rollout_steps)):
            state = sim.get_states()
            mean, std, _ = actor(state)
            exploration_scale = float(args.exploration_scale)
            if exploration_scale > 0.0:
                raw_action, sim_action, behavior_log_prob_flat, _ = sample_policy_action(
                    mean,
                    std,
                    exploration_scale=exploration_scale,
                )
                behavior_log_prob = behavior_log_prob_flat.unsqueeze(-1)
            else:
                raw_action = mean
                sim_action = sanitize_policy_actions(raw_action)
                behavior_log_prob = policy_action_log_prob(mean, std, raw_action).unsqueeze(-1)

            states.append(state.detach().clone())
            raw_actions.append(raw_action.detach().clone())
            sim_actions.append(sim_action.detach().clone())
            behavior_log_probs.append(behavior_log_prob.detach().clone())
            active_masks.append((~prev_done).detach().clone())

            reward, _ = sim.step(sim_action, dt=1.0 / FPS)
            next_state = sim.get_states()
            prev_done = sim.episode_done.clone()

            reward_snapshot = reward.detach().clone()
            if float(args.reward_clip) > 0.0:
                reward_snapshot = reward_snapshot.clamp(-float(args.reward_clip), float(args.reward_clip))
            rewards.append(reward_snapshot)
            next_states.append(next_state.detach().clone())
            dones.append(prev_done.detach().clone())
            frames = step + 1

            if bool(prev_done.all()):
                break

    transition_count = 0
    state_tensor = torch.stack(states, dim=0).reshape(-1, INPUT_DIM)
    raw_action_tensor = torch.stack(raw_actions, dim=0).reshape(-1, ACTION_DIM)
    sim_action_tensor = torch.stack(sim_actions, dim=0).reshape(-1, ACTION_DIM)
    behavior_log_prob_tensor = torch.stack(behavior_log_probs, dim=0).reshape(-1, 1)
    reward_tensor = torch.stack(rewards, dim=0).reshape(-1, 1)
    next_state_tensor = torch.stack(next_states, dim=0).reshape(-1, INPUT_DIM)
    done_tensor = torch.stack(dones, dim=0).reshape(-1, 1).float()
    active_tensor = torch.stack(active_masks, dim=0).reshape(-1)
    transition_count = int(active_tensor.sum().item())
    reward_values = reward_tensor[active_tensor]

    batch = {
        "states": state_tensor[active_tensor].to(device="cpu", dtype=args.storage_dtype),
        "raw_actions": raw_action_tensor[active_tensor].to(device="cpu", dtype=args.storage_dtype),
        "sim_actions": sim_action_tensor[active_tensor].to(device="cpu", dtype=args.storage_dtype),
        "behavior_log_probs": behavior_log_prob_tensor[active_tensor].to(device="cpu", dtype=torch.float32),
        "rewards": reward_values.to(device="cpu", dtype=torch.float32),
        "next_states": next_state_tensor[active_tensor].to(device="cpu", dtype=args.storage_dtype),
        "dones": done_tensor[active_tensor].to(device="cpu", dtype=torch.float32),
    }
    active_raw_actions = raw_action_tensor[active_tensor]
    active_sim_actions = sim_action_tensor[active_tensor]
    active_behavior_log_probs = behavior_log_prob_tensor[active_tensor]
    metrics = normalize_rollout_metrics(summarize_play_metrics(sim, start_times=start_times))
    metrics.update({
        "frames": frames,
        "transitions": transition_count,
        "reward_mean": float(reward_values.mean().item()) if transition_count > 0 else 0.0,
        "reward_min": float(reward_values.min().item()) if transition_count > 0 else 0.0,
        "reward_max": float(reward_values.max().item()) if transition_count > 0 else 0.0,
        "raw_action_std": float(active_raw_actions.std(dim=0).mean().item()) if transition_count > 1 else 0.0,
        "sim_action_std": float(active_sim_actions.std(dim=0).mean().item()) if transition_count > 1 else 0.0,
        "sanitize_delta": float((active_sim_actions - active_raw_actions).abs().mean().item()) if transition_count > 0 else 0.0,
        "behavior_log_prob_mean": float(active_behavior_log_probs.mean().item()) if transition_count > 0 else 0.0,
    })
    return batch, metrics


def run_eval(actor, device, eval_hashes, map_cache, args, label, profile, start_time_sampler=None):
    summary = evaluate_policy_model(
        actor,
        device,
        eval_hashes,
        map_cache=map_cache,
        num_envs=min(int(args.eval_envs), int(args.num_envs)),
        noise_scale=0.0,
        verbose=True,
        label=label,
        start_time_sampler=start_time_sampler,
        **profile,
    )
    print(
        f"  [{label}] task {summary['mean_accuracy']:.2f}% | "
        f"cover {summary['mean_note_coverage']:.3f} | "
        f"resolved {summary.get('mean_resolved_accuracy', summary['mean_engaged_accuracy']):.2f}% | "
        f"engaged {summary['mean_engaged_accuracy']:.2f}% | "
        f"comp {summary['mean_completion']:.2f} | "
        f"clear {summary['mean_clear_rate']:.2f}"
    )
    return summary


def build_rollout_start_time_sampler(map_meta_by_hash):
    def _sample(summary, num_envs, device):
        meta = map_meta_by_hash.get(summary["map_hash"])
        if meta is None:
            scorable_seconds = [
                float(note["time"]) / max(1e-6, summary["bpm"] / 60.0)
                for note in summary.get("notes", [])
                if int(note.get("type", -1)) != 3
            ]
            meta = {
                "duration_sec": float(summary["duration_sec"]),
                "scorable_seconds": scorable_seconds,
            }
        return sample_start_times([meta] * int(num_envs), device)

    return _sample


def save_awac_state(path, state):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _meets_bootstrap_threshold(value, threshold):
    return float(value) + BOOTSTRAP_GATE_EPS >= float(threshold)


def bootstrap_has_play_signal(summary, *, strict=False):
    if not summary:
        return False
    min_coverage = STRICT_BOOTSTRAP_MIN_COVERAGE if strict else BOOTSTRAP_MIN_COVERAGE
    min_accuracy = STRICT_BOOTSTRAP_MIN_ACCURACY if strict else BOOTSTRAP_MIN_ACCURACY
    min_resolved_accuracy = STRICT_BOOTSTRAP_MIN_RESOLVED_ACCURACY if strict else BOOTSTRAP_MIN_RESOLVED_ACCURACY
    min_completion = STRICT_BOOTSTRAP_MIN_COMPLETION if strict else BOOTSTRAP_MIN_COMPLETION
    min_clear_rate = 0.0 if strict else BOOTSTRAP_MIN_CLEAR_RATE
    coverage = float(summary.get("mean_note_coverage", 0.0))
    accuracy = float(summary.get("mean_accuracy", 0.0))
    resolved_accuracy = float(summary.get("mean_resolved_accuracy", summary.get("mean_engaged_accuracy", 0.0)))
    completion = float(summary.get("mean_completion", 0.0))
    clear_rate = float(summary.get("mean_clear_rate", 0.0))
    signal_ok = (
        _meets_bootstrap_threshold(accuracy, min_accuracy)
        or _meets_bootstrap_threshold(resolved_accuracy, min_resolved_accuracy)
    )
    if not strict:
        signal_ok = signal_ok or _meets_bootstrap_threshold(clear_rate, min_clear_rate)
    return (
        _meets_bootstrap_threshold(coverage, min_coverage)
        and _meets_bootstrap_threshold(completion, min_completion)
        and signal_ok
    )


def format_eval_signal(summary):
    summary = summary or {}
    return (
        f"task {float(summary.get('mean_accuracy', 0.0)):.3f}% | "
        f"cover {float(summary.get('mean_note_coverage', 0.0)):.4f} | "
        f"resolved {float(summary.get('mean_resolved_accuracy', summary.get('mean_engaged_accuracy', 0.0))):.2f}% | "
        f"engaged {float(summary.get('mean_engaged_accuracy', 0.0)):.2f}% | "
        f"comp {float(summary.get('mean_completion', 0.0)):.4f} | "
        f"clear {float(summary.get('mean_clear_rate', 0.0)):.4f}"
    )


def eval_profiles_match(left, right, *, atol=1e-6):
    if not left or not right:
        return False
    if int(left.get("action_repeat", 1)) != int(right.get("action_repeat", 1)):
        return False
    if bool(left.get("fail_enabled", True)) != bool(right.get("fail_enabled", True)):
        return False
    numeric_keys = (
        "smoothing_alpha",
        "training_wheels_level",
        "assist_level",
        "survival_assistance",
        "stability_reward_level",
        "style_guidance_level",
        "saber_inertia",
        "rot_clamp",
        "pos_clamp",
    )
    for key in numeric_keys:
        if abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) > float(atol):
            return False
    return True


def matched_bootstrap_has_play_signal(summary, *, profile_matches_strict=False):
    return bootstrap_has_play_signal(summary, strict=bool(profile_matches_strict))


def awac_checkpoint_key(strict_summary, matched_summary, *, matched_profile_is_strict=False):
    strict_summary = strict_summary or {}
    matched_summary = matched_summary or {}
    return (
        int(bootstrap_has_play_signal(strict_summary, strict=True)),
        round(float(strict_summary.get("mean_note_coverage", 0.0)), 4),
        round(float(strict_summary.get("mean_accuracy", 0.0)), 2),
        round(float(strict_summary.get("mean_completion", 0.0)), 4),
        round(float(strict_summary.get("mean_clear_rate", 0.0)), 4),
        round(float(strict_summary.get("mean_resolved_accuracy", strict_summary.get("mean_engaged_accuracy", 0.0))), 2),
        round(float(strict_summary.get("mean_engaged_accuracy", 0.0)), 2),
        int(matched_bootstrap_has_play_signal(matched_summary, profile_matches_strict=matched_profile_is_strict)),
        round(float(matched_summary.get("mean_note_coverage", 0.0)), 4),
        round(float(matched_summary.get("mean_accuracy", 0.0)), 2),
        round(float(matched_summary.get("mean_completion", 0.0)), 4),
        round(float(matched_summary.get("mean_clear_rate", 0.0)), 4),
        round(float(matched_summary.get("mean_resolved_accuracy", matched_summary.get("mean_engaged_accuracy", 0.0))), 2),
        round(float(matched_summary.get("mean_engaged_accuracy", 0.0)), 2),
    )


def awac_eval_key_has_regressed(
    candidate_eval_key,
    best_eval_key,
    *,
    accuracy_fraction=0.5,
    coverage_fraction=0.5,
):
    if candidate_eval_key is None or best_eval_key is None:
        return False
    candidate_eval_key = tuple(candidate_eval_key)
    best_eval_key = tuple(best_eval_key)
    if candidate_eval_key >= best_eval_key:
        return False
    if len(candidate_eval_key) < 3 or len(best_eval_key) < 3:
        return False

    best_ready = int(best_eval_key[0]) > 0
    candidate_ready = int(candidate_eval_key[0]) > 0
    if best_ready and not candidate_ready:
        return True

    best_coverage = float(best_eval_key[1])
    candidate_coverage = float(candidate_eval_key[1])
    if best_coverage > 0.0 and candidate_coverage < best_coverage * float(coverage_fraction):
        return True

    best_accuracy = float(best_eval_key[2])
    candidate_accuracy = float(candidate_eval_key[2])
    if best_accuracy > 0.0 and candidate_accuracy < best_accuracy * float(accuracy_fraction):
        return True

    return False


def seed_awac_best_eval_key(
    trainer_state,
    best_eval_key,
    strict_summary,
    matched_summary,
    *,
    matched_profile_is_strict=False,
):
    candidate_eval_key = awac_checkpoint_key(
        strict_summary,
        matched_summary,
        matched_profile_is_strict=matched_profile_is_strict,
    )
    if best_eval_key is None or candidate_eval_key > tuple(best_eval_key):
        trainer_state["best_eval_key"] = list(candidate_eval_key)
        return candidate_eval_key, True
    return tuple(best_eval_key), False


def save_awac_actor_model(actor):
    torch.save(
        attach_policy_schema({
            "checkpoint_kind": "awac_actor",
            "model_state_dict": build_rl_bootstrap_state_dict(actor),
        }),
        AWAC_MODEL_PATH,
    )


def save_awac_artifacts(
    actor,
    critics,
    target_actor,
    target_critics,
    actor_optimizer,
    critic_optimizer,
    state,
    args,
    *,
    save_actor_model=True,
):
    if save_actor_model:
        save_awac_actor_model(actor)
    checkpoint = attach_policy_schema({
        "checkpoint_kind": "awac_bootstrap",
        "epoch": int(state.get("epoch", 0)),
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critics.state_dict(),
        "target_actor_state_dict": target_actor.state_dict(),
        "target_critic_state_dict": target_critics.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
        "trainer_state": dict(state),
        "config": {k: v for k, v in vars(args).items() if k != "storage_dtype"},
    })
    torch.save(checkpoint, AWAC_CHECKPOINT_PATH)
    save_awac_state(AWAC_STATE_PATH, state)


def parse_args():
    parser = argparse.ArgumentParser(description="AWAC bootstrap trainer for CyberNoodles.")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--num-envs", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=192)
    parser.add_argument("--buffer-capacity", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--updates-per-epoch", type=int, default=0)
    parser.add_argument("--min-replay-size", type=int, default=8_192)
    parser.add_argument("--actor-lr", type=float, default=1.0e-4)
    parser.add_argument("--critic-lr", type=float, default=3.0e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--awac-lambda", type=float, default=1.0)
    parser.add_argument("--max-awac-weight", type=float, default=20.0)
    parser.add_argument("--exploration-scale", type=float, default=0.35)
    parser.add_argument("--reward-clip", type=float, default=0.0, help="Clip rollout rewards before storing them. <= 0 disables clipping.")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--demo-bc-weight", type=float, default=0.35)
    parser.add_argument("--demo-batch-size", type=int, default=1024)
    parser.add_argument("--demo-shards-limit", type=int, default=256)
    parser.add_argument("--disable-demo-regularizer", action="store_true", help="Explicitly opt out of BC demo regularization.")
    parser.add_argument("--assist-level", type=float, default=0.0)
    parser.add_argument("--training-wheels-level", type=float, default=0.0)
    parser.add_argument("--survival-assistance", type=float, default=0.0)
    parser.add_argument("--stability-reward-level", type=float, default=0.0)
    parser.add_argument("--style-guidance-level", type=float, default=0.0)
    parser.set_defaults(fail_enabled=True)
    parser.add_argument("--fail-enabled", dest="fail_enabled", action="store_true", help="Train with strict fail states enabled.")
    parser.add_argument("--no-fail-enabled", dest="fail_enabled", action="store_false", help="Disable fail states during AWAC rollouts.")
    parser.add_argument("--dense-reward-scale", type=float, default=0.0)
    parser.add_argument("--saber-inertia", type=float, default=0.0)
    parser.add_argument("--rot-clamp", type=float, default=0.12)
    parser.add_argument("--pos-clamp", type=float, default=0.15)
    parser.add_argument("--w-miss", type=float, default=0.5)
    parser.add_argument("--w-jerk", type=float, default=0.0005)
    parser.add_argument("--w-pos-jerk", type=float, default=0.0005)
    parser.add_argument("--w-reach", type=float, default=0.0)
    parser.add_argument("--train-pool-size", type=int, default=24)
    parser.add_argument("--warmup-pool-size", type=int, default=8)
    parser.add_argument("--strict-unlock-coverage", type=float, default=0.03)
    parser.add_argument("--strict-unlock-accuracy", type=float, default=3.0)
    parser.add_argument("--strict-expand-coverage", type=float, default=0.08)
    parser.add_argument("--strict-expand-accuracy", type=float, default=8.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-envs", type=int, default=24)
    parser.add_argument("--bootstrap-map-count", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--strict-rollback-frac", type=float, default=0.5)
    parser.add_argument("--no-strict-rollback", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--base-model", type=str, default=BC_MODEL_PATH)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--allow-cold-start", action="store_true", help="Skip the BC bootstrap preflight check and allow AWAC to start from a dead warmstart.")
    return parser.parse_args()


def train_awac(args=None):
    args = parse_args() if args is None else args
    args.storage_dtype = torch.float16

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if int(args.num_envs) <= 0:
        args.num_envs = pick_default_num_envs(device)
    if int(args.batch_size) <= 0:
        args.batch_size = min(2048, suggest_batch_size(device))

    print(f"\033[96m{'=' * 68}\033[0m")
    print("CyberNoodles AWAC Bootstrap")
    print(f"\033[96m{'=' * 68}\033[0m")
    print(f"Device: {device} | Envs: {args.num_envs} | Rollout steps: {args.rollout_steps}")
    print(f"Replay: {args.buffer_capacity:,} | Batch: {args.batch_size:,}")

    curriculum = load_curriculum_records()
    replay_backed_hashes = load_replay_backed_hashes()
    if not curriculum:
        raise RuntimeError("curriculum.json is missing or empty. Build the curriculum before AWAC.")

    training_curriculum = filter_curriculum_by_split(curriculum, "train", allow_fallback=True)
    training_pool = build_training_pool(
        training_curriculum,
        replay_backed_hashes,
        pool_limit=max(int(args.train_pool_size), int(args.warmup_pool_size) * 4),
    )
    if not training_pool:
        raise RuntimeError("No playable maps were available for AWAC training.")

    map_cache = {meta["hash"]: (meta["beatmap"], meta["bpm"]) for meta in training_pool}
    eval_hashes = choose_eval_hashes(
        curriculum,
        max_maps=min(3, len(curriculum)),
        map_cache=map_cache,
        suite="starter",
        split="dev_eval",
    )
    if not eval_hashes:
        eval_hashes = [training_pool[0]["hash"]]

    max_notes = max(meta["note_count"] for meta in training_pool)
    max_obstacles = max(meta["obstacle_count"] for meta in training_pool)

    actor = ActorCritic().to(device)
    critics = TwinQCritic().to(device)
    target_actor = ActorCritic().to(device)
    target_critics = TwinQCritic().to(device)

    actor_optimizer = make_adam(
        list(actor.actor_features.parameters()) + list(actor.actor_mean.parameters()) + [actor.actor_log_std],
        lr=float(args.actor_lr),
    )
    critic_optimizer = make_adam(critics.parameters(), lr=float(args.critic_lr))
    replay = ReplayBuffer(args.buffer_capacity, device=device, storage_dtype=args.storage_dtype)

    start_epoch = 0
    trainer_state = {} if args.fresh else load_json(AWAC_STATE_PATH, {})
    resume_payload = None if args.fresh else read_awac_resume(device)
    if resume_payload is not None:
        remap_state_dict(resume_payload["actor_state_dict"], actor)
        critics.load_state_dict(resume_payload["critic_state_dict"])
        remap_state_dict(resume_payload["target_actor_state_dict"], target_actor)
        target_critics.load_state_dict(resume_payload["target_critic_state_dict"])
        actor_optimizer.load_state_dict(resume_payload["actor_optimizer_state_dict"])
        critic_optimizer.load_state_dict(resume_payload["critic_optimizer_state_dict"])
        start_epoch = int(resume_payload.get("epoch", 0) or 0)
        trainer_state = dict(resume_payload.get("trainer_state", trainer_state))
        print(f"Resuming AWAC from epoch {start_epoch}. Replay buffer will be rebuilt from fresh rollouts.")
    else:
        if not os.path.exists(args.base_model):
            raise RuntimeError(f"Base model not found: {args.base_model}")
        load_actor_weights(actor, args.base_model, device)
        target_actor.load_state_dict(actor.state_dict())
        target_critics.load_state_dict(critics.state_dict())
        print(f"Bootstrapping AWAC actor from: {args.base_model}")

    target_actor.eval()
    writer = SummaryWriter(log_dir="runs/cybernoodles_awac")
    sim = make_simulator(num_envs=int(args.num_envs), device=device)
    sim.reserve_note_capacity(max_notes, max_obstacles)

    demo_stream, demo_record_count = build_required_demo_stream(args, device)

    best_strict_accuracy = float(trainer_state.get("best_strict_accuracy", 0.0) or 0.0)
    best_strict_coverage = float(trainer_state.get("best_strict_coverage", 0.0) or 0.0)
    stored_best_eval_key = trainer_state.get("best_eval_key")
    best_eval_key = tuple(stored_best_eval_key) if isinstance(stored_best_eval_key, (list, tuple)) else None

    strict_profile = get_eval_profile("strict")
    matched_tuning = build_awac_training_tuning(args)
    matched_profile = {
        "action_repeat": 1,
        "smoothing_alpha": 1.0,
        "training_wheels_level": float(matched_tuning.training_wheels),
        "assist_level": float(matched_tuning.rehab_assists),
        "survival_assistance": float(matched_tuning.survival_assistance),
        "stability_reward_level": float(matched_tuning.stability_assistance),
        "style_guidance_level": float(matched_tuning.style_guidance_level),
        "fail_enabled": bool(matched_tuning.fail_enabled),
        "saber_inertia": float(matched_tuning.saber_inertia),
        "rot_clamp": float(matched_tuning.rot_clamp),
        "pos_clamp": float(matched_tuning.pos_clamp),
    }
    matched_profile_is_strict = eval_profiles_match(matched_profile, strict_profile)

    preflight_hashes = eval_hashes[: max(1, min(len(eval_hashes), int(args.bootstrap_map_count)))]
    if preflight_hashes:
        checkpoint_label = "resume" if resume_payload is not None else "bootstrap"
        print(f"Running AWAC {checkpoint_label} preflight...")
        preflight_meta_by_hash = {
            meta["hash"]: meta for meta in training_pool if meta["hash"] in preflight_hashes
        }
        preflight_start_sampler = build_rollout_start_time_sampler(preflight_meta_by_hash)
        strict_bootstrap = run_eval(
            actor,
            device,
            preflight_hashes,
            map_cache,
            args,
            "bootstrap-strict",
            strict_profile,
            start_time_sampler=preflight_start_sampler,
        )
        if matched_profile_is_strict:
            print("  [bootstrap-matched] profile matches strict; reusing strict preflight result.")
            matched_bootstrap = dict(strict_bootstrap)
        else:
            matched_bootstrap = run_eval(
                actor,
                device,
                preflight_hashes,
                map_cache,
                args,
                "bootstrap-matched",
                matched_profile,
                start_time_sampler=preflight_start_sampler,
            )
        trainer_state, best_strict_accuracy, best_strict_coverage = update_trainer_state_from_eval(
            trainer_state,
            strict_bootstrap,
            matched_bootstrap,
            best_strict_accuracy=best_strict_accuracy,
            best_strict_coverage=best_strict_coverage,
        )
        strict_ready = bootstrap_has_play_signal(strict_bootstrap, strict=True)
        matched_ready = matched_bootstrap_has_play_signal(
            matched_bootstrap,
            profile_matches_strict=matched_profile_is_strict,
        )
        if not args.allow_cold_start and not (strict_ready and matched_ready):
            raise RuntimeError(
                "AWAC preflight failed.\n"
                f"  strict : {format_eval_signal(strict_bootstrap)}\n"
                f"  matched: {format_eval_signal(matched_bootstrap)}\n"
                "Rebuild the BC shards, retrain BC, and verify strict closed-loop play moves off the floor before AWAC. "
                "Use --allow-cold-start only if you explicitly want to bypass that gate."
            )

    if best_eval_key is None:
        print("Running AWAC baseline strict eval for checkpoint selection...")
        baseline_strict = run_eval(actor, device, eval_hashes, map_cache, args, "baseline-strict", strict_profile)
        if matched_profile_is_strict:
            print("  [baseline-matched] profile matches strict; reusing strict baseline result.")
            baseline_matched = dict(baseline_strict)
        else:
            baseline_matched = run_eval(actor, device, eval_hashes, map_cache, args, "baseline-matched", matched_profile)
        trainer_state, best_strict_accuracy, best_strict_coverage = update_trainer_state_from_eval(
            trainer_state,
            baseline_strict,
            baseline_matched,
            best_strict_accuracy=best_strict_accuracy,
            best_strict_coverage=best_strict_coverage,
        )
        best_eval_key, _ = seed_awac_best_eval_key(
            trainer_state,
            best_eval_key,
            baseline_strict,
            baseline_matched,
            matched_profile_is_strict=matched_profile_is_strict,
        )
        print("  Seeding best AWAC actor from baseline strict eval.")
        save_awac_actor_model(actor)
        save_awac_state(AWAC_STATE_PATH, trainer_state)

    last_pool_stage = None
    for epoch in range(start_epoch, int(args.epochs)):
        epoch_start = time.perf_counter()
        actor.eval()

        active_training_pool, pool_stage, strict_progress_accuracy, strict_progress_coverage = build_active_training_pool(
            training_pool,
            trainer_state,
            args,
        )
        if not active_training_pool:
            raise RuntimeError("No active maps are available for the current AWAC curriculum stage.")
        if pool_stage != last_pool_stage:
            print(
                f"Curriculum stage: {pool_stage} | strict task {strict_progress_accuracy:.2f}% | "
                f"strict cover {strict_progress_coverage:.3f} | maps {len(active_training_pool)}"
            )
            last_pool_stage = pool_stage
        trainer_state["curriculum_stage"] = pool_stage
        map_batch = select_training_batch(active_training_pool, args.num_envs)
        rollout_batch, rollout_metrics = collect_rollout(actor, sim, args, map_batch)
        if int(rollout_metrics.get("transitions", 0) or 0) <= 0:
            raise RuntimeError("AWAC rollout produced zero active transitions; stopping instead of silently stalling.")
        for metric_name in (
            "reward_mean",
            "reward_min",
            "reward_max",
            "raw_action_std",
            "sim_action_std",
            "sanitize_delta",
            "behavior_log_prob_mean",
        ):
            ensure_finite_scalar(f"rollout/{metric_name}", rollout_metrics.get(metric_name, 0.0))
        replay.add_batch(
            rollout_batch["states"],
            rollout_batch["raw_actions"],
            rollout_batch["sim_actions"],
            rollout_batch["behavior_log_probs"],
            rollout_batch["rewards"],
            rollout_batch["next_states"],
            rollout_batch["dones"],
        )

        actor.train()
        critics.train()

        actor_loss_value = 0.0
        policy_loss_value = 0.0
        critic_loss_value = 0.0
        demo_loss_value = 0.0
        weighted_demo_loss_value = 0.0
        behavior_log_prob_delta_value = 0.0
        updates_ran = 0

        if len(replay) >= int(args.min_replay_size):
            actor_step_before = optimizer_step_total(actor_optimizer)
            critic_step_before = optimizer_step_total(critic_optimizer)
            actor_snapshot = parameter_snapshot(actor)
            critic_snapshot = parameter_snapshot(critics)
            updates_per_epoch = int(args.updates_per_epoch) if int(args.updates_per_epoch) > 0 else choose_update_count(
                rollout_metrics["transitions"],
                args.batch_size,
            )
            if updates_per_epoch <= 0:
                raise RuntimeError("AWAC update budget resolved to zero despite a ready replay buffer.")
            for _ in range(updates_per_epoch):
                batch = replay.sample(args.batch_size)

                with torch.no_grad():
                    next_mean, _, _ = target_actor(batch["next_states"])
                    target_q = target_critics.min_q(batch["next_states"], sanitize_policy_actions(next_mean))
                    q_backup = batch["rewards"] + (1.0 - batch["dones"]) * float(args.gamma) * target_q

                q1, q2 = critics(batch["states"], batch["sim_actions"])
                critic_loss = nn.functional.mse_loss(q1, q_backup) + nn.functional.mse_loss(q2, q_backup)
                ensure_finite_scalar("awac/critic_loss", critic_loss)
                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                assert_finite_gradients("awac/critic_grad", critics)
                critic_grad_norm = nn.utils.clip_grad_norm_(critics.parameters(), float(args.grad_clip))
                ensure_finite_scalar("awac/critic_grad_norm", critic_grad_norm)
                critic_optimizer.step()

                mean, std, _ = actor(batch["states"])
                log_prob = policy_action_log_prob(mean, std, batch["raw_actions"]).unsqueeze(-1)
                with torch.no_grad():
                    pi_mean, _, _ = actor(batch["states"])
                    v_pi = target_critics.min_q(batch["states"], sanitize_policy_actions(pi_mean))
                    q_data = target_critics.min_q(batch["states"], batch["sim_actions"])
                    advantages = q_data - v_pi
                    weights = torch.exp(advantages / max(1e-6, float(args.awac_lambda)))
                    weights = torch.clamp(weights, max=float(args.max_awac_weight))
                policy_loss = -(weights * log_prob).mean()
                actor_loss = policy_loss
                behavior_log_prob_delta = (log_prob.detach() - batch["behavior_log_probs"]).mean()

                demo_loss = torch.tensor(0.0, device=device)
                weighted_demo_loss = torch.tensor(0.0, device=device)
                if demo_stream is not None:
                    demo_x, demo_y = demo_stream.next()
                    demo_mean, _, _ = actor(demo_x)
                    demo_loss, _ = bc_pose_loss(demo_mean, demo_y, demo_x)
                    weighted_demo_loss = float(args.demo_bc_weight) * demo_loss
                    actor_loss = actor_loss + weighted_demo_loss

                ensure_finite_scalar("awac/actor_loss", actor_loss)
                ensure_finite_scalar("awac/policy_loss", policy_loss)
                ensure_finite_scalar("awac/demo_loss", demo_loss)
                ensure_finite_scalar("awac/weighted_demo_loss", weighted_demo_loss)
                ensure_finite_scalar("awac/behavior_log_prob_delta", behavior_log_prob_delta)
                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                assert_finite_gradients("awac/actor_grad", actor)
                actor_grad_norm = nn.utils.clip_grad_norm_(
                    list(actor.actor_features.parameters()) + list(actor.actor_mean.parameters()) + [actor.actor_log_std],
                    float(args.grad_clip),
                )
                ensure_finite_scalar("awac/actor_grad_norm", actor_grad_norm)
                actor_optimizer.step()

                soft_update(target_actor, actor, float(args.tau))
                soft_update(target_critics, critics, float(args.tau))

                critic_loss_value += float(critic_loss.item())
                actor_loss_value += float(actor_loss.item())
                policy_loss_value += float(policy_loss.item())
                demo_loss_value += float(demo_loss.item())
                weighted_demo_loss_value += float(weighted_demo_loss.item())
                behavior_log_prob_delta_value += float(behavior_log_prob_delta.item())
                updates_ran += 1
            if updates_ran <= 0:
                raise RuntimeError("AWAC replay buffer is ready, but no optimizer updates ran.")
            ensure_optimizer_advanced("AWAC actor", actor_step_before, optimizer_step_total(actor_optimizer))
            ensure_optimizer_advanced("AWAC critic", critic_step_before, optimizer_step_total(critic_optimizer))
            ensure_parameter_moved("AWAC actor", parameter_delta_l2(actor, actor_snapshot))
            ensure_parameter_moved("AWAC critic", parameter_delta_l2(critics, critic_snapshot))
            assert_finite_module("awac/actor_param", actor)
            assert_finite_module("awac/critic_param", critics)

        epoch_seconds = time.perf_counter() - epoch_start
        if updates_ran > 0:
            critic_loss_value /= updates_ran
            actor_loss_value /= updates_ran
            policy_loss_value /= updates_ran
            demo_loss_value /= updates_ran
            weighted_demo_loss_value /= updates_ran
            behavior_log_prob_delta_value /= updates_ran
        demo_contribution_fraction = (
            abs(weighted_demo_loss_value) / max(1e-6, abs(actor_loss_value))
            if updates_ran > 0
            else 0.0
        )
        demo_samples_used = updates_ran * int(args.demo_batch_size) if demo_stream is not None else 0
        self_to_demo_sample_ratio = (
            float(rollout_metrics["transitions"]) / float(max(1, demo_samples_used))
            if demo_samples_used > 0
            else 0.0
        )

        trainer_state.update({
            "epoch": epoch + 1,
            "buffer_size": len(replay),
            "last_rollout_task_accuracy": rollout_metrics["task_accuracy"],
            "last_rollout_engaged_accuracy": rollout_metrics["engaged_accuracy"],
            "last_rollout_note_coverage": rollout_metrics["note_coverage"],
            "last_rollout_resolved_coverage": rollout_metrics["resolved_coverage"],
            "last_rollout_completion": rollout_metrics["completion"],
            "last_rollout_clear_rate": rollout_metrics["clear_rate"],
            "last_rollout_fail_rate": rollout_metrics["fail_rate"],
            "last_actor_loss": actor_loss_value,
            "last_policy_loss": policy_loss_value,
            "last_critic_loss": critic_loss_value,
            "last_demo_loss": demo_loss_value,
            "last_weighted_demo_loss": weighted_demo_loss_value,
            "last_demo_contribution_fraction": demo_contribution_fraction,
            "last_behavior_log_prob_delta": behavior_log_prob_delta_value,
            "last_raw_action_std": rollout_metrics["raw_action_std"],
            "last_sim_action_std": rollout_metrics["sim_action_std"],
            "last_sanitize_delta": rollout_metrics["sanitize_delta"],
            "demo_record_count": int(demo_record_count),
        })

        writer.add_scalar("awac/replay_size", len(replay), epoch + 1)
        writer.add_scalar("awac/actor_loss", actor_loss_value, epoch + 1)
        writer.add_scalar("awac/policy_loss", policy_loss_value, epoch + 1)
        writer.add_scalar("awac/critic_loss", critic_loss_value, epoch + 1)
        writer.add_scalar("awac/demo_loss", demo_loss_value, epoch + 1)
        writer.add_scalar("awac/weighted_demo_loss", weighted_demo_loss_value, epoch + 1)
        writer.add_scalar("awac/demo_contribution_fraction", demo_contribution_fraction, epoch + 1)
        writer.add_scalar("awac/behavior_log_prob_delta", behavior_log_prob_delta_value, epoch + 1)
        writer.add_scalar("awac/self_replay_transitions", rollout_metrics["transitions"], epoch + 1)
        writer.add_scalar("awac/self_to_demo_sample_ratio", self_to_demo_sample_ratio, epoch + 1)
        writer.add_scalar("rollout/task_accuracy", rollout_metrics["task_accuracy"], epoch + 1)
        writer.add_scalar("rollout/note_coverage", rollout_metrics["note_coverage"], epoch + 1)
        writer.add_scalar("rollout/resolved_coverage", rollout_metrics["resolved_coverage"], epoch + 1)
        writer.add_scalar("rollout/completion", rollout_metrics["completion"], epoch + 1)
        writer.add_scalar("rollout/raw_action_std", rollout_metrics["raw_action_std"], epoch + 1)
        writer.add_scalar("rollout/sim_action_std", rollout_metrics["sim_action_std"], epoch + 1)
        writer.add_scalar("rollout/sanitize_delta", rollout_metrics["sanitize_delta"], epoch + 1)
        writer.add_scalar("rollout/behavior_log_prob_mean", rollout_metrics["behavior_log_prob_mean"], epoch + 1)

        print(
            f"Epoch {epoch + 1} | envs {args.num_envs} | transitions {rollout_metrics['transitions']:,} | "
            f"buffer {len(replay):,} | task {rollout_metrics['task_accuracy']:.2f}% | "
            f"cover {rollout_metrics['note_coverage']:.3f} | comp {rollout_metrics['completion']:.2f} | "
            f"clear {rollout_metrics['clear_rate']:.2f} | actor {actor_loss_value:.4f} | "
            f"policy {policy_loss_value:.4f} | critic {critic_loss_value:.4f} | "
            f"demo {demo_loss_value:.4f} -> {weighted_demo_loss_value:.4f} ({demo_contribution_fraction:.2f}) | "
            f"act_std raw/sim {rollout_metrics['raw_action_std']:.3f}/{rollout_metrics['sim_action_std']:.3f} | "
            f"sanitize {rollout_metrics['sanitize_delta']:.3f} | self/demo {self_to_demo_sample_ratio:.2f} | "
            f"{epoch_seconds:.2f}s"
        )

        should_eval = int(args.eval_every) > 0 and (((epoch + 1) % int(args.eval_every) == 0) or epoch == 0)
        if should_eval:
            actor.eval()
            strict_summary = run_eval(actor, device, eval_hashes, map_cache, args, "strict", strict_profile)
            if matched_profile_is_strict:
                print("  [matched] profile matches strict; reusing strict evaluation result.")
                matched_summary = dict(strict_summary)
            else:
                matched_summary = run_eval(actor, device, eval_hashes, map_cache, args, "matched", matched_profile)
            writer.add_scalar("eval/strict_task_accuracy", strict_summary["mean_accuracy"], epoch + 1)
            writer.add_scalar("eval/strict_note_coverage", strict_summary["mean_note_coverage"], epoch + 1)
            writer.add_scalar("eval/matched_task_accuracy", matched_summary["mean_accuracy"], epoch + 1)
            writer.add_scalar("eval/matched_note_coverage", matched_summary["mean_note_coverage"], epoch + 1)

            trainer_state, best_strict_accuracy, best_strict_coverage = update_trainer_state_from_eval(
                trainer_state,
                strict_summary,
                matched_summary,
                best_strict_accuracy=best_strict_accuracy,
                best_strict_coverage=best_strict_coverage,
            )
            candidate_eval_key = awac_checkpoint_key(
                strict_summary,
                matched_summary,
                matched_profile_is_strict=matched_profile_is_strict,
            )
            improved = best_eval_key is None or candidate_eval_key > best_eval_key
            if improved:
                best_eval_key = candidate_eval_key
                trainer_state["best_eval_key"] = list(best_eval_key)
            if improved:
                print("  Saving improved AWAC checkpoint.")
                save_awac_artifacts(
                    actor,
                    critics,
                    target_actor,
                    target_critics,
                    actor_optimizer,
                    critic_optimizer,
                    trainer_state,
                    args,
                    save_actor_model=True,
                )
            elif (
                not bool(args.no_strict_rollback)
                and awac_eval_key_has_regressed(
                    candidate_eval_key,
                    best_eval_key,
                    accuracy_fraction=float(args.strict_rollback_frac),
                    coverage_fraction=float(args.strict_rollback_frac),
                )
            ):
                if os.path.exists(AWAC_MODEL_PATH):
                    print("  Strict eval regressed; restoring best strict AWAC actor.")
                    load_actor_weights(actor, AWAC_MODEL_PATH, device)
                    target_actor.load_state_dict(actor.state_dict())
                    actor_optimizer = make_adam(
                        list(actor.actor_features.parameters()) + list(actor.actor_mean.parameters()) + [actor.actor_log_std],
                        lr=float(args.actor_lr),
                    )
                    trainer_state["last_strict_rollback_epoch"] = epoch + 1
                    trainer_state["last_strict_rollback_from_key"] = list(candidate_eval_key)
                    trainer_state["last_strict_rollback_to_key"] = list(best_eval_key)
                else:
                    print("  Strict eval regressed, but no saved best AWAC actor exists to restore.")

        if ((epoch + 1) % int(args.checkpoint_every) == 0) or (epoch + 1 == int(args.epochs)):
            save_awac_artifacts(
                actor,
                critics,
                target_actor,
                target_critics,
                actor_optimizer,
                critic_optimizer,
                trainer_state,
                args,
                save_actor_model=False,
            )

    writer.close()


def main():
    train_awac()


if __name__ == "__main__":
    main()
