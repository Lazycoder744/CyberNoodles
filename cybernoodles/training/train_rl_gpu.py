import os, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, kl_divergence
import numpy as np
import time, json
from torch.utils.tensorboard import SummaryWriter
from cybernoodles.core.gpu_simulator import DEFAULT_DT
from cybernoodles.core.network import ACTION_DIM, CURRENT_POSE_END, CURRENT_POSE_START, INPUT_DIM, ActorCritic
from cybernoodles.data.dataset_builder import get_map_data
from cybernoodles.data.style_calibration import ensure_style_calibration
from cybernoodles.envs import SimulatorTuning, apply_simulator_tuning, make_simulator
from cybernoodles.replay.generate_replay import generate_progress_replay
from cybernoodles.training.eval_splits import filter_curriculum_by_split
from cybernoodles.training.policy_checkpoint import (
    attach_policy_schema,
    extract_policy_state_dict,
    validate_policy_checkpoint_payload,
)
from cybernoodles.training.policy_eval import (
    choose_eval_hashes,
    compute_completion_ratios,
    compute_target_note_counts,
    evaluate_policy_model,
    get_eval_profile,
    load_replay_backed_hashes,
    policy_action_log_prob,
    sanitize_policy_actions,
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

BC_MODEL_PATH = "bsai_bc_model.pth"
AWAC_MODEL_PATH = "bsai_awac_model.pth"
AWAC_CHECKPOINT_PATH = "bsai_awac_checkpoint.pth"
AWAC_STATE_PATH = "awac_state.json"
RL_CHECKPOINT_PATH = "bsai_rl_checkpoint.pth"
TRAINER_STATE_PATH = "rl_state.json"
RESUME_SNAPSHOT_VERSION = 1
PPO_BASE_MODEL_ENV = "BSAI_PPO_BASE_MODEL"
PPO_SKIP_BC_PROBE_ENV = "BSAI_SKIP_PPO_BC_PROBE"
PPO_SKIP_INITIAL_EVAL_ENV = "BSAI_SKIP_PPO_INITIAL_EVAL"
INDUCTOR_WRITE_ATOMIC_PATCH_ENV = "BSAI_PATCH_INDUCTOR_WRITE_ATOMIC"
AWAC_STRICT_READY_ACCURACY = 0.5
AWAC_STRICT_READY_COVERAGE = 0.01

# ─────────────────────────────────────────────────────────────────────────────
# Note: AutoCurriculum is now replaced by PBT Genetics
# ─────────────────────────────────────────────────────────────────────────────


def make_adam(param_groups):
    """Prefer fused Adam on CUDA when the installed PyTorch supports it."""
    try:
        return optim.Adam(param_groups, fused=True)
    except TypeError:
        return optim.Adam(param_groups)


def env_flag_enabled(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def patch_inductor_write_atomic_for_windows():
    """Opt-in Windows workaround for old PyTorch Inductor atomic writes."""
    import pathlib
    import torch._inductor.codecache as codecache

    current = getattr(codecache, "write_atomic", None)
    if getattr(current, "_bsai_replace_atomic_patch", False):
        return False

    def _patched_write_atomic(path, content, make_dirs=False, mode="w"):
        if make_dirs:
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = pathlib.Path(str(path) + ".tmp2")
        tmp.write_bytes(content if isinstance(content, bytes) else content.encode())
        os.replace(tmp, path)

    _patched_write_atomic._bsai_replace_atomic_patch = True
    _patched_write_atomic._bsai_original_write_atomic = current
    codecache.write_atomic = _patched_write_atomic
    return True


def maybe_patch_inductor_write_atomic():
    if not env_flag_enabled(INDUCTOR_WRITE_ATOMIC_PATCH_ENV, default=False):
        return False
    return patch_inductor_write_atomic_for_windows()


def max_nonnegative_signal(*values):
    best = 0.0
    for value in values:
        try:
            candidate = float(value or 0.0)
        except (TypeError, ValueError):
            continue
        if np.isfinite(candidate) and candidate > best:
            best = candidate
    return best


def live_training_signal(
    adaptive_state=None,
    current_task_acc=None,
    strict_eval_acc=None,
    matched_eval_acc=None,
    fallback=0.0,
):
    signal = max_nonnegative_signal(current_task_acc, strict_eval_acc, matched_eval_acc)
    if isinstance(adaptive_state, dict):
        signal = max(
            signal,
            max_nonnegative_signal(
                adaptive_state.get('strict_eval_accuracy', 0.0),
                adaptive_state.get('matched_eval_accuracy', 0.0),
                adaptive_state.get('last_eval_accuracy', 0.0),
            ),
        )
    if signal > 0.0:
        return signal
    return max_nonnegative_signal(fallback)


def should_run_bc_baseline_probe(bc_reference_model, eval_hashes, skip_bc_probe=False):
    return bc_reference_model is not None and bool(eval_hashes) and not bool(skip_bc_probe)


def should_run_epoch_eval_probe(epoch, eval_interval, eval_hashes, skip_initial_eval=False):
    if not eval_hashes:
        return False
    if epoch == 0 and skip_initial_eval:
        return False
    return epoch == 0 or ((epoch + 1) % int(eval_interval) == 0)


def pick_total_envs(device, num_tribes):
    vram_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    if vram_gb >= 20:
        total_envs = 2048
    elif vram_gb >= 14:
        total_envs = 1536
    elif vram_gb >= 10:
        total_envs = 1024
    elif vram_gb >= 8:
        total_envs = 768
    else:
        total_envs = 512
    return max(num_tribes, (total_envs // num_tribes) * num_tribes)


def build_recovery_sim_tuning(recovery):
    assist_level = float(recovery['assist_level'])
    return SimulatorTuning(
        penalty_weights=(0.0, 0.0, 0.0, 0.0),
        dense_reward_scale=0.0,
        training_wheels=float(recovery['training_wheels']),
        rehab_assists=assist_level,
        survival_assistance=float(recovery['survival_level']),
        stability_assistance=float(recovery['stability_reward_level']),
        style_guidance_level=float(recovery['style_guidance_level']),
        hit_timing_profile="assisted" if assist_level > 0.0 else "default",
        fail_enabled=bool(recovery['fail_enabled']),
        saber_inertia=0.0,
        rot_clamp=0.07,
        pos_clamp=0.12,
    )


def allocate_rollout_buffers(num_tribes, steps, envs_per_tribe, device):
    rollouts = []
    for _ in range(num_tribes):
        rollouts.append({
            'states': torch.zeros(steps, envs_per_tribe, INPUT_DIM, device=device),
            'actions': torch.zeros(steps, envs_per_tribe, ACTION_DIM, device=device),
            'logprobs': torch.zeros(steps, envs_per_tribe, device=device),
            'values': torch.zeros(steps, envs_per_tribe, device=device),
            'rewards': torch.zeros(steps, envs_per_tribe, device=device),
            'dones': torch.zeros(steps, envs_per_tribe, dtype=torch.bool, device=device),
            'valid': torch.zeros(steps, envs_per_tribe, dtype=torch.bool, device=device),
            'spans': torch.zeros(steps, device=device),
        })
    return rollouts


def load_trainer_state():
    if not os.path.exists(TRAINER_STATE_PATH):
        return {}
    try:
        with open(TRAINER_STATE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def load_rl_checkpoint_payload():
    if not os.path.exists(RL_CHECKPOINT_PATH):
        return {}
    try:
        ckpt = torch.load(RL_CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    except Exception:
        return {}
    if not isinstance(ckpt, dict):
        return {}
    try:
        validate_policy_checkpoint_payload(
            ckpt,
            checkpoint_path=RL_CHECKPOINT_PATH,
            required_keys=('model_state_dict',),
        )
    except RuntimeError as exc:
        print(f"  [Load] Ignoring incompatible RL checkpoint: {exc}")
        return {}
    return ckpt


def _load_json_state(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def summarize_awac_bootstrap_signal(state):
    state = state or {}
    best_strict_coverage = float(
        state.get('best_strict_coverage', state.get('best_strict_note_coverage', 0.0)) or 0.0
    )
    return {
        'epoch': int(state.get('epoch', 0) or 0),
        'strict_accuracy': max(
            float(state.get('last_strict_accuracy', 0.0) or 0.0),
            float(state.get('best_strict_accuracy', 0.0) or 0.0),
        ),
        'strict_coverage': max(
            float(state.get('last_strict_note_coverage', 0.0) or 0.0),
            best_strict_coverage,
        ),
        'matched_accuracy': float(state.get('last_matched_accuracy', 0.0) or 0.0),
        'matched_coverage': float(state.get('last_matched_note_coverage', 0.0) or 0.0),
    }


def awac_bootstrap_has_strict_signal(signal):
    signal = signal or {}
    return (
        float(signal.get('strict_accuracy', 0.0) or 0.0) >= AWAC_STRICT_READY_ACCURACY
        or float(signal.get('strict_coverage', 0.0) or 0.0) >= AWAC_STRICT_READY_COVERAGE
    )


def load_awac_bootstrap_signal():
    actor_path = ""
    if os.path.exists(AWAC_MODEL_PATH):
        actor_path = AWAC_MODEL_PATH
    elif os.path.exists(AWAC_CHECKPOINT_PATH):
        actor_path = AWAC_CHECKPOINT_PATH

    signal = {
        'actor_path': actor_path,
        'epoch': 0,
        'strict_accuracy': 0.0,
        'strict_coverage': 0.0,
        'matched_accuracy': 0.0,
        'matched_coverage': 0.0,
        'usable': False,
    }

    candidate_states = []
    json_state = _load_json_state(AWAC_STATE_PATH)
    if json_state:
        candidate_states.append(json_state)

    if os.path.exists(AWAC_CHECKPOINT_PATH):
        try:
            payload = torch.load(AWAC_CHECKPOINT_PATH, map_location='cpu', weights_only=False)
            trainer_state = payload.get('trainer_state') if isinstance(payload, dict) else None
            if isinstance(trainer_state, dict):
                candidate_states.append(trainer_state)
        except Exception:
            pass

    for state in candidate_states:
        candidate = summarize_awac_bootstrap_signal(state)
        signal['epoch'] = max(signal['epoch'], candidate['epoch'])
        signal['strict_accuracy'] = max(signal['strict_accuracy'], candidate['strict_accuracy'])
        signal['strict_coverage'] = max(signal['strict_coverage'], candidate['strict_coverage'])
        signal['matched_accuracy'] = max(signal['matched_accuracy'], candidate['matched_accuracy'])
        signal['matched_coverage'] = max(signal['matched_coverage'], candidate['matched_coverage'])

    signal['usable'] = bool(signal['actor_path']) and awac_bootstrap_has_strict_signal(signal)
    return signal


def choose_default_bootstrap_actor_path(awac_signal):
    awac_signal = awac_signal or {}
    actor_path = str(awac_signal.get('actor_path', '') or '').strip()
    if actor_path and bool(awac_signal.get('usable', False)):
        return actor_path, "AWAC bootstrap"
    return BC_MODEL_PATH, "BC baseline"


def checkpoint_has_resume_snapshot(checkpoint_payload, expected_num_tribes=None):
    if not isinstance(checkpoint_payload, dict):
        return False
    if int(checkpoint_payload.get('resume_snapshot_version', 0) or 0) != RESUME_SNAPSHOT_VERSION:
        return False
    population = checkpoint_payload.get('population')
    if not isinstance(population, dict):
        return False
    tribes = population.get('tribes')
    if not isinstance(tribes, list) or not tribes:
        return False
    if expected_num_tribes is not None and len(tribes) != expected_num_tribes:
        return False
    return True


def serialize_rng_state():
    return {
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        'numpy': np.random.get_state(),
        'python': random.getstate(),
    }


def restore_rng_state(rng_state):
    if not isinstance(rng_state, dict):
        return False

    restored = False
    torch_cpu = rng_state.get('torch_cpu')
    if isinstance(torch_cpu, torch.Tensor):
        torch.set_rng_state(torch_cpu)
        restored = True

    torch_cuda = rng_state.get('torch_cuda')
    if torch.cuda.is_available() and isinstance(torch_cuda, (list, tuple)) and torch_cuda:
        try:
            torch.cuda.set_rng_state_all(list(torch_cuda))
            restored = True
        except Exception:
            pass

    numpy_state = rng_state.get('numpy')
    if isinstance(numpy_state, tuple) and len(numpy_state) == 5:
        try:
            np.random.set_state(numpy_state)
            restored = True
        except Exception:
            pass

    python_state = rng_state.get('python')
    if python_state is not None:
        try:
            random.setstate(python_state)
            restored = True
        except Exception:
            pass

    return restored


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def serialize_tribe_state(tribe):
    return {
        'id': tribe.id,
        'num_envs': tribe.num_envs,
        'model_state_dict': tribe.model.state_dict(),
        'optimizer_state_dict': tribe.optimizer.state_dict(),
        'scaler_state_dict': tribe.scaler.state_dict(),
        'hparams': tribe.hparams.copy(),
        'current_policy_std': float(getattr(tribe, 'current_policy_std', tribe.hparams.get('policy_std', 0.10))),
        'current_action_repeat': int(getattr(tribe, 'current_action_repeat', tribe.hparams.get('action_repeat', 2))),
        'current_saber_inertia': float(getattr(tribe, 'current_saber_inertia', tribe.hparams.get('saber_inertia', 0.35))),
        'current_rot_clamp': float(getattr(tribe, 'current_rot_clamp', tribe.hparams.get('rot_clamp', 0.10))),
        'current_pos_clamp': float(getattr(tribe, 'current_pos_clamp', 0.15)),
        'fitness': float(tribe.fitness),
        'selection_score': float(getattr(tribe, 'selection_score', 0.0)),
        'moving_acc': float(tribe.moving_acc),
        'last_acc': float(tribe.last_acc),
        'last_task_accuracy': float(getattr(tribe, 'last_task_accuracy', 0.0)),
        'last_note_coverage': float(getattr(tribe, 'last_note_coverage', 0.0)),
        'last_rollout_task_accuracy': float(getattr(tribe, 'last_rollout_task_accuracy', 0.0)),
        'last_rollout_note_coverage': float(getattr(tribe, 'last_rollout_note_coverage', 0.0)),
        'last_rollout_notes_seen': float(getattr(tribe, 'last_rollout_notes_seen', 0.0)),
        'generation': int(tribe.generation),
        'death_counter': int(tribe.death_counter),
        'best_fitness': float(tribe.best_fitness),
        'best_selection_score': float(getattr(tribe, 'best_selection_score', 0.0)),
        'stagnation_counter': int(tribe.stagnation_counter),
        'diversity_score': float(tribe.diversity_score),
        'performance_log': list(tribe.performance_log),
        'stability_score': float(getattr(tribe, 'stability_score', 1.0)),
        'exploration_score': float(getattr(tribe, 'exploration_score', 0.0)),
        'last_energy': float(tribe.last_energy),
        'last_completion': float(tribe.last_completion),
        'last_fail_rate': float(tribe.last_fail_rate),
        'last_clear_rate': float(tribe.last_clear_rate),
        'last_timeout_rate': float(tribe.last_timeout_rate),
        'last_combo_ratio': float(tribe.last_combo_ratio),
        'last_mean_speed': float(tribe.last_mean_speed),
        'last_style_violation': float(tribe.last_style_violation),
        'last_angular_violation': float(tribe.last_angular_violation),
        'last_motion_efficiency': float(tribe.last_motion_efficiency),
        'last_waste_motion': float(tribe.last_waste_motion),
        'last_idle_motion': float(tribe.last_idle_motion),
        'last_guard_error': float(tribe.last_guard_error),
        'last_oscillation': float(tribe.last_oscillation),
        'last_lateral_motion': float(tribe.last_lateral_motion),
        'leader_cooldown_epochs': int(getattr(tribe, 'leader_cooldown_epochs', 0)),
    }


def restore_tribe_snapshot(tribe, snapshot):
    if not isinstance(snapshot, dict):
        return False

    model_state = snapshot.get('model_state_dict')
    if isinstance(model_state, dict):
        tribe.model.load_state_dict(model_state)

    hparams = snapshot.get('hparams')
    if isinstance(hparams, dict):
        tribe.hparams = hparams.copy()

    optimizer_state = snapshot.get('optimizer_state_dict')
    if isinstance(optimizer_state, dict):
        tribe.optimizer.load_state_dict(optimizer_state)
        move_optimizer_state_to_device(tribe.optimizer, tribe.device)

    scaler_state = snapshot.get('scaler_state_dict')
    if isinstance(scaler_state, dict):
        tribe.scaler.load_state_dict(scaler_state)

    tribe.optimizer.param_groups[0]['lr'] = tribe.hparams['lr_actor']
    tribe.optimizer.param_groups[1]['lr'] = tribe.hparams['lr_critic']
    tribe.current_policy_std = float(snapshot.get('current_policy_std', tribe.hparams.get('policy_std', 0.10)))
    tribe.current_action_repeat = int(snapshot.get('current_action_repeat', tribe.hparams.get('action_repeat', 2)))
    tribe.current_saber_inertia = float(snapshot.get('current_saber_inertia', tribe.hparams.get('saber_inertia', 0.35)))
    tribe.current_rot_clamp = float(snapshot.get('current_rot_clamp', tribe.hparams.get('rot_clamp', 0.10)))
    tribe.current_pos_clamp = float(snapshot.get('current_pos_clamp', 0.15))
    tribe.fitness = float(snapshot.get('fitness', 0.0))
    tribe.selection_score = float(snapshot.get('selection_score', snapshot.get('fitness', 0.0)))
    tribe.moving_acc = float(snapshot.get('moving_acc', 0.0))
    tribe.last_acc = float(snapshot.get('last_acc', 0.0))
    tribe.last_task_accuracy = float(snapshot.get('last_task_accuracy', snapshot.get('last_acc', 0.0)))
    tribe.last_note_coverage = float(snapshot.get('last_note_coverage', 0.0))
    tribe.last_rollout_task_accuracy = float(snapshot.get('last_rollout_task_accuracy', tribe.last_task_accuracy))
    tribe.last_rollout_note_coverage = float(snapshot.get('last_rollout_note_coverage', tribe.last_note_coverage))
    tribe.last_rollout_notes_seen = float(snapshot.get('last_rollout_notes_seen', 0.0))
    tribe.generation = int(snapshot.get('generation', 0))
    tribe.death_counter = int(snapshot.get('death_counter', 0))
    tribe.best_fitness = float(snapshot.get('best_fitness', 0.0))
    tribe.best_selection_score = float(snapshot.get('best_selection_score', snapshot.get('selection_score', 0.0)))
    tribe.stagnation_counter = int(snapshot.get('stagnation_counter', 0))
    tribe.diversity_score = float(snapshot.get('diversity_score', 1.0))
    tribe.performance_log = list(snapshot.get('performance_log', []))
    tribe.stability_score = float(snapshot.get('stability_score', 1.0))
    tribe.exploration_score = float(snapshot.get('exploration_score', 0.0))
    tribe.last_energy = float(snapshot.get('last_energy', 0.5))
    tribe.last_completion = float(snapshot.get('last_completion', 0.0))
    tribe.last_fail_rate = float(snapshot.get('last_fail_rate', 0.0))
    tribe.last_clear_rate = float(snapshot.get('last_clear_rate', 0.0))
    tribe.last_timeout_rate = float(snapshot.get('last_timeout_rate', 0.0))
    tribe.last_combo_ratio = float(snapshot.get('last_combo_ratio', 0.0))
    tribe.last_mean_speed = float(snapshot.get('last_mean_speed', 0.0))
    tribe.last_style_violation = float(snapshot.get('last_style_violation', 0.0))
    tribe.last_angular_violation = float(snapshot.get('last_angular_violation', 0.0))
    tribe.last_motion_efficiency = float(snapshot.get('last_motion_efficiency', 0.0))
    tribe.last_waste_motion = float(snapshot.get('last_waste_motion', 0.0))
    tribe.last_idle_motion = float(snapshot.get('last_idle_motion', 0.0))
    tribe.last_guard_error = float(snapshot.get('last_guard_error', 0.0))
    tribe.last_oscillation = float(snapshot.get('last_oscillation', 0.0))
    tribe.last_lateral_motion = float(snapshot.get('last_lateral_motion', 0.0))
    tribe.leader_cooldown_epochs = int(snapshot.get('leader_cooldown_epochs', 0))
    return True


def load_population_from_checkpoint(checkpoint_payload, num_tribes, envs_per_tribe, device):
    population = checkpoint_payload.get('population', {}) if isinstance(checkpoint_payload, dict) else {}
    tribe_snapshots = population.get('tribes', []) if isinstance(population, dict) else []
    if len(tribe_snapshots) != num_tribes:
        return None

    ordered_snapshots = sorted(
        tribe_snapshots,
        key=lambda snapshot: int(snapshot.get('id', 0)) if isinstance(snapshot, dict) else 0,
    )
    tribes = []
    for tribe_id, snapshot in enumerate(ordered_snapshots):
        tribe = Tribe(tribe_id, envs_per_tribe, device, base_model_path=None)
        restore_tribe_snapshot(tribe, snapshot)
        tribes.append(tribe)
    return tribes


def recover_tribe_from_checkpoint_snapshot(tribe, checkpoint_payload):
    population = checkpoint_payload.get('population', {}) if isinstance(checkpoint_payload, dict) else {}
    tribe_snapshots = population.get('tribes', []) if isinstance(population, dict) else []
    for snapshot in tribe_snapshots:
        if int(snapshot.get('id', -1)) != tribe.id:
            continue
        return restore_tribe_snapshot(tribe, snapshot)
    return False


def tribe_nonfinite_tensors(tribe):
    bad_names = []
    for name, param in tribe.model.named_parameters():
        if not torch.isfinite(param).all():
            bad_names.append(name)

    for group_idx, group in enumerate(tribe.optimizer.param_groups):
        for param_idx, param in enumerate(group.get('params', [])):
            state = tribe.optimizer.state.get(param, {})
            for state_name, state_value in state.items():
                if torch.is_tensor(state_value) and not torch.isfinite(state_value).all():
                    bad_names.append(f"optimizer[{group_idx}].{param_idx}.{state_name}")
    return bad_names


def strict_task_signal(adaptive_state=None, trainer_state=None, checkpoint_payload=None):
    candidates = []
    if isinstance(adaptive_state, dict):
        candidates.extend([
            adaptive_state.get('strict_eval_accuracy'),
            adaptive_state.get('last_eval_accuracy'),
            adaptive_state.get('best_eval_accuracy'),
        ])
    if isinstance(trainer_state, dict):
        candidates.append(trainer_state.get('global_best_eval_accuracy'))
        candidates.append(trainer_state.get('global_best_task_accuracy'))
    if isinstance(checkpoint_payload, dict):
        embedded_state = checkpoint_payload.get('trainer_state')
        if isinstance(embedded_state, dict):
            candidates.append(embedded_state.get('global_best_eval_accuracy'))
            candidates.append(embedded_state.get('global_best_task_accuracy'))
        candidates.append(checkpoint_payload.get('strict_eval_accuracy'))
        candidates.append(checkpoint_payload.get('task_accuracy'))

    finite = []
    for value in candidates:
        try:
            finite.append(float(value))
        except (TypeError, ValueError):
            continue
    return max([0.0] + finite)


def compute_selection_score(
    *,
    task_accuracy,
    note_coverage,
    completion,
    clear_rate,
    fail_rate,
    timeout_rate,
    cut_quality,
    combo_ratio,
    recovery=None,
):
    task_frac = float(np.clip(task_accuracy / 100.0, 0.0, 1.0))
    coverage = float(np.clip(note_coverage, 0.0, 1.0))
    completion = float(np.clip(completion, 0.0, 1.0))
    clear_rate = float(np.clip(clear_rate, 0.0, 1.0))
    fail_rate = float(np.clip(fail_rate, 0.0, 1.0))
    timeout_rate = float(np.clip(timeout_rate, 0.0, 1.0))
    cut_quality = float(np.clip(cut_quality, 0.0, 1.0))
    combo_ratio = float(np.clip(combo_ratio, 0.0, 1.0))

    real_progress = coverage * task_frac
    coverage_gate = float(np.clip(coverage * 3.0, 0.0, 1.0))
    hit_phase = float(np.clip((task_accuracy - 3.0) / 12.0, 0.0, 1.0))
    quality_phase = float(np.clip((task_accuracy - 10.0) / 20.0, 0.0, 1.0))
    task_gate = float(np.clip(task_frac * 5.0, 0.0, 1.0))
    low_coverage_penalty = max(0.0, 0.30 - coverage) * 45.0
    early_engagement_bonus = coverage * (26.0 + 18.0 * (1.0 - hit_phase))
    completion_bonus = completion * (4.0 + 10.0 * task_gate) * coverage_gate

    strictness_penalty = 0.0
    if isinstance(recovery, dict):
        strictness_penalty += float(recovery.get('assist_level', 0.0)) * 8.0
        strictness_penalty += float(recovery.get('training_wheels', 0.0)) * 4.0
        strictness_penalty += max(0.0, float(recovery.get('survival_level', 0.0)) - 0.15) * 5.0
        if not recovery.get('fail_enabled', True):
            strictness_penalty += 10.0

    return float(
        real_progress * 160.0
        + early_engagement_bonus
        + clear_rate * 48.0 * task_gate
        + completion_bonus
        + combo_ratio * 10.0 * hit_phase
        + cut_quality * 8.0 * quality_phase
        - fail_rate * 12.0
        - timeout_rate * 7.0
        - low_coverage_penalty
        - strictness_penalty
    )


def sync_trainer_state_with_checkpoint(trainer_state, checkpoint_payload):
    state = dict(trainer_state or {})
    if not checkpoint_payload:
        return state, False

    state_epoch = int(state.get('epoch', 0) or 0)
    ckpt_epoch = int(checkpoint_payload.get('epoch', 0) or 0)
    embedded_state = checkpoint_payload.get('trainer_state')
    used_checkpoint_state = False
    prefer_embedded_state = checkpoint_has_resume_snapshot(checkpoint_payload)

    if isinstance(embedded_state, dict) and (prefer_embedded_state or ckpt_epoch > state_epoch):
        state = dict(embedded_state)
        used_checkpoint_state = True
    elif ckpt_epoch > state_epoch:
        state['epoch'] = ckpt_epoch
        state['moving_acc'] = max(
            float(state.get('moving_acc', 0.0) or 0.0),
            float(checkpoint_payload.get('moving_acc', 0.0) or 0.0),
        )
        state['global_best_accuracy'] = max(
            float(state.get('global_best_accuracy', 0.0) or 0.0),
            float(checkpoint_payload.get('moving_acc', 0.0) or 0.0),
        )
        state['global_best_fitness'] = max(
            float(state.get('global_best_fitness', 0.0) or 0.0),
            float(checkpoint_payload.get('fitness', 0.0) or 0.0),
        )
        state['global_best_selection_score'] = max(
            float(state.get('global_best_selection_score', -1e9) or -1e9),
            float(checkpoint_payload.get('selection_score', checkpoint_payload.get('fitness', 0.0)) or 0.0),
        )
        state['global_best_task_accuracy'] = max(
            float(state.get('global_best_task_accuracy', 0.0) or 0.0),
            float(checkpoint_payload.get('task_accuracy', checkpoint_payload.get('moving_acc', 0.0)) or 0.0),
        )

    return state, used_checkpoint_state


def should_prefer_bc_bootstrap(trainer_state, checkpoint_payload):
    if not checkpoint_payload:
        return True

    adaptive = trainer_state.get('adaptive', {}) if isinstance(trainer_state, dict) else {}
    bc_probe = float(adaptive.get('bc_probe_accuracy', 0.0) or 0.0)
    strict_eval = strict_task_signal(adaptive_state=adaptive, trainer_state=trainer_state)
    ckpt_state = checkpoint_payload.get('trainer_state') if isinstance(checkpoint_payload, dict) else {}
    ckpt_strict_eval = strict_task_signal(
        adaptive_state=ckpt_state.get('adaptive', {}) if isinstance(ckpt_state, dict) else None,
        trainer_state=ckpt_state if isinstance(ckpt_state, dict) else None,
        checkpoint_payload=checkpoint_payload,
    )
    ckpt_task_acc = float(checkpoint_payload.get('task_accuracy', 0.0) or 0.0)
    ckpt_proxy_acc = float(checkpoint_payload.get('moving_acc', 0.0) or 0.0)

    # Compare BC against the checkpoint's strict-task signal first. Proxy
    # training accuracy is only a last-resort fallback for legacy checkpoints.
    effective_ckpt_skill = ckpt_strict_eval if ckpt_strict_eval > 0.0 else max(ckpt_task_acc, ckpt_proxy_acc)
    if bc_probe >= 15.0 and effective_ckpt_skill < max(3.0, bc_probe * 0.15):
        return True
    if strict_eval >= 10.0 and effective_ckpt_skill < 2.5:
        return True
    return False


def save_training_artifacts(best_tribe, tribes, model_path, epoch, state, *, promote_actor=True):
    if promote_actor:
        torch.save(
            attach_policy_schema({
                'checkpoint_kind': 'ppo_actor',
                'model_state_dict': best_tribe.model.state_dict(),
            }),
            model_path,
        )
    checkpoint_payload = attach_policy_schema({
        'checkpoint_kind': 'ppo_population_resume',
        'resume_snapshot_version': RESUME_SNAPSHOT_VERSION,
        'promoted_actor': bool(promote_actor),
        'model_state_dict': best_tribe.model.state_dict(),
        'moving_acc': best_tribe.moving_acc,
        'task_accuracy': float(getattr(best_tribe, 'last_task_accuracy', 0.0)),
        'selection_score': float(getattr(best_tribe, 'selection_score', best_tribe.fitness)),
        'epoch': epoch,
        'tribe_id': best_tribe.id,
        'fitness': best_tribe.fitness,
        'hparams': best_tribe.hparams.copy(),
        'trainer_state': dict(state),
        'population': {
            'num_tribes': len(tribes),
            'tribes': [serialize_tribe_state(tribe) for tribe in tribes],
        },
        'rng_state': serialize_rng_state(),
    })
    torch.save(checkpoint_payload, RL_CHECKPOINT_PATH)
    with open(TRAINER_STATE_PATH, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)


def default_adaptive_state():
    return {
        'rehab_level': 0,
        'stability_rehab_level': 0,
        'style_rehab_level': 0,
        'stagnation_epochs': 0,
        'stability_stagnation_epochs': 0,
        'best_signal_accuracy': 0.0,
        'best_eval_accuracy': 0.0,
        'best_stability': 0.0,
        'last_eval_accuracy': 0.0,
        'last_mean_stability': 0.0,
        'last_mean_note_coverage': 0.0,
        'last_mean_energy': 0.0,
        'last_mean_completion': 0.0,
        'last_mean_fail_rate': 0.0,
        'last_mean_speed': 0.0,
        'last_mean_style_violation': 0.0,
        'last_mean_angular_violation': 0.0,
        'last_mean_motion_efficiency': 0.0,
        'last_mean_waste_motion': 0.0,
        'last_mean_idle_motion': 0.0,
        'last_mean_guard_error': 0.0,
        'last_mean_oscillation': 0.0,
        'last_mean_lateral_motion': 0.0,
        'last_progress_epoch': 0,
        'bc_probe_accuracy': 0.0,
        'strict_eval_accuracy': 0.0,
        'matched_eval_accuracy': 0.0,
        'best_matched_eval_accuracy': 0.0,
        'run_best_signal_accuracy': 0.0,
        'run_best_stability': 0.0,
        'rehab_release_streak': 0,
        'escape_release_streak': 0,
        'escape_support_active': False,
    }


def reset_active_adaptation_for_new_run(state, trainer_state):
    state['rehab_level'] = 0
    state['stability_rehab_level'] = 0
    state['style_rehab_level'] = 0
    state['stagnation_epochs'] = 0
    state['stability_stagnation_epochs'] = 0
    state['last_progress_epoch'] = int(trainer_state.get('epoch', 0))
    state['last_eval_accuracy'] = 0.0
    state['strict_eval_accuracy'] = 0.0
    state['matched_eval_accuracy'] = 0.0
    state['run_best_signal_accuracy'] = 0.0
    state['run_best_stability'] = 0.0
    state['last_mean_stability'] = 0.0
    state['last_mean_note_coverage'] = 0.0
    state['last_mean_energy'] = 0.0
    state['last_mean_completion'] = 0.0
    state['last_mean_fail_rate'] = 0.0
    state['last_mean_speed'] = 0.0
    state['last_mean_style_violation'] = 0.0
    state['last_mean_angular_violation'] = 0.0
    state['last_mean_motion_efficiency'] = 0.0
    state['last_mean_waste_motion'] = 0.0
    state['last_mean_idle_motion'] = 0.0
    state['last_mean_guard_error'] = 0.0
    state['last_mean_oscillation'] = 0.0
    state['last_mean_lateral_motion'] = 0.0
    state['rehab_release_streak'] = 0
    state['escape_release_streak'] = 0
    state['escape_support_active'] = False
    return state


def load_adaptive_state(trainer_state, resume_in_place=False):
    state = default_adaptive_state()
    state.update(trainer_state.get('adaptive', {}))
    state['best_signal_accuracy'] = max(
        float(state.get('best_signal_accuracy', 0.0)),
        float(trainer_state.get('global_best_accuracy', trainer_state.get('moving_acc', 0.0))),
    )
    state['best_eval_accuracy'] = max(
        float(state.get('best_eval_accuracy', 0.0)),
        float(trainer_state.get('global_best_eval_accuracy', 0.0)),
    )
    state['best_matched_eval_accuracy'] = max(
        float(state.get('best_matched_eval_accuracy', 0.0)),
        float(trainer_state.get('global_best_matched_eval_accuracy', 0.0)),
    )
    if resume_in_place:
        return state
    return reset_active_adaptation_for_new_run(state, trainer_state)


def mature_rehab_caps(
    signal,
    strict_eval_acc,
    mean_stability,
    mean_energy,
    mean_completion,
    mean_fail_rate,
    mean_motion_efficiency,
    mean_idle_motion,
    mean_guard_error,
):
    stability_cap = 4
    style_cap = 4

    if (
        signal >= 45.0
        and (mean_energy is None or mean_energy >= 0.45)
        and (mean_stability is None or mean_stability >= 0.42)
        and (mean_fail_rate is None or mean_fail_rate <= 0.55)
    ):
        stability_cap = min(stability_cap, 3)
        style_cap = min(style_cap, 3)

    if (
        signal >= 60.0
        and (mean_energy is None or mean_energy >= 0.52)
        and (mean_stability is None or mean_stability >= 0.48)
        and (mean_completion is None or mean_completion >= 0.55)
        and (mean_fail_rate is None or mean_fail_rate <= 0.45)
    ):
        stability_cap = min(stability_cap, 2)
        style_cap = min(style_cap, 2)

    if (
        signal >= 68.0
        and (strict_eval_acc is not None and strict_eval_acc >= 12.0)
        and (mean_energy is None or mean_energy >= 0.55)
        and (mean_stability is None or mean_stability >= 0.52)
        and (mean_motion_efficiency is None or mean_motion_efficiency >= 0.08)
        and (mean_idle_motion is None or mean_idle_motion <= 0.09)
        and (mean_guard_error is None or mean_guard_error <= 0.55)
    ):
        stability_cap = min(stability_cap, 1)
        style_cap = min(style_cap, 1)

    return stability_cap, style_cap


def summarize_tribe_population(tribes):
    if not tribes:
        return {
            'ranked': [],
            'leaderboard': [],
            'core': [],
            'mean_acc': 0.0,
            'mean_proxy_acc': 0.0,
            'mean_selection_score': 0.0,
            'mean_note_coverage': 0.0,
            'mean_stability': 0.0,
            'mean_energy': 0.0,
            'mean_completion': 0.0,
            'mean_fail_rate': 0.0,
            'mean_speed': 0.0,
            'mean_style_violation': 0.0,
            'mean_angular_violation': 0.0,
            'mean_motion_efficiency': 0.0,
            'mean_waste_motion': 0.0,
            'mean_idle_motion': 0.0,
            'mean_guard_error': 0.0,
            'mean_oscillation': 0.0,
            'mean_lateral_motion': 0.0,
            'core_mean_acc': 0.0,
            'core_mean_proxy_acc': 0.0,
            'core_mean_selection_score': 0.0,
            'core_mean_note_coverage': 0.0,
            'core_mean_stability': 0.0,
            'core_mean_energy': 0.0,
            'core_mean_completion': 0.0,
            'core_mean_fail_rate': 0.0,
            'core_mean_speed': 0.0,
            'core_mean_style_violation': 0.0,
            'core_mean_angular_violation': 0.0,
            'core_mean_motion_efficiency': 0.0,
            'core_mean_waste_motion': 0.0,
            'core_mean_idle_motion': 0.0,
            'core_mean_guard_error': 0.0,
            'core_mean_oscillation': 0.0,
            'core_mean_lateral_motion': 0.0,
        }

    stability_by_id = {tribe.id: tribe.calculate_stability() for tribe in tribes}
    ranked = sorted(
        tribes,
        key=lambda tribe: (
            getattr(tribe, 'selection_score', tribe.fitness),
            getattr(tribe, 'last_task_accuracy', tribe.moving_acc),
            tribe.last_note_coverage,
            stability_by_id[tribe.id],
            tribe.last_energy,
        ),
        reverse=True,
    )
    leaderboard = [
        tribe for tribe in ranked
        if int(getattr(tribe, 'leader_cooldown_epochs', 0) or 0) <= 0
    ]
    if not leaderboard:
        leaderboard = ranked
    core_count = min(len(leaderboard), max(2, (len(leaderboard) + 1) // 2))
    core = leaderboard[:core_count]

    def avg(items, getter):
        return sum(getter(item) for item in items) / len(items) if items else 0.0

    return {
        'ranked': ranked,
        'leaderboard': leaderboard,
        'core': core,
        'mean_acc': avg(ranked, lambda tribe: tribe.last_task_accuracy),
        'mean_proxy_acc': avg(ranked, lambda tribe: tribe.moving_acc),
        'mean_selection_score': avg(ranked, lambda tribe: tribe.selection_score),
        'mean_note_coverage': avg(ranked, lambda tribe: tribe.last_note_coverage),
        'mean_stability': avg(ranked, lambda tribe: stability_by_id[tribe.id]),
        'mean_energy': avg(ranked, lambda tribe: tribe.last_energy),
        'mean_completion': avg(ranked, lambda tribe: tribe.last_completion),
        'mean_fail_rate': avg(ranked, lambda tribe: tribe.last_fail_rate),
        'mean_speed': avg(ranked, lambda tribe: tribe.last_mean_speed),
        'mean_style_violation': avg(ranked, lambda tribe: tribe.last_style_violation),
        'mean_angular_violation': avg(ranked, lambda tribe: tribe.last_angular_violation),
        'mean_motion_efficiency': avg(ranked, lambda tribe: tribe.last_motion_efficiency),
        'mean_waste_motion': avg(ranked, lambda tribe: tribe.last_waste_motion),
        'mean_idle_motion': avg(ranked, lambda tribe: tribe.last_idle_motion),
        'mean_guard_error': avg(ranked, lambda tribe: tribe.last_guard_error),
        'mean_oscillation': avg(ranked, lambda tribe: tribe.last_oscillation),
        'mean_lateral_motion': avg(ranked, lambda tribe: tribe.last_lateral_motion),
        'core_mean_acc': avg(core, lambda tribe: tribe.last_task_accuracy),
        'core_mean_proxy_acc': avg(core, lambda tribe: tribe.moving_acc),
        'core_mean_selection_score': avg(core, lambda tribe: tribe.selection_score),
        'core_mean_note_coverage': avg(core, lambda tribe: tribe.last_note_coverage),
        'core_mean_stability': avg(core, lambda tribe: stability_by_id[tribe.id]),
        'core_mean_energy': avg(core, lambda tribe: tribe.last_energy),
        'core_mean_completion': avg(core, lambda tribe: tribe.last_completion),
        'core_mean_fail_rate': avg(core, lambda tribe: tribe.last_fail_rate),
        'core_mean_speed': avg(core, lambda tribe: tribe.last_mean_speed),
        'core_mean_style_violation': avg(core, lambda tribe: tribe.last_style_violation),
        'core_mean_angular_violation': avg(core, lambda tribe: tribe.last_angular_violation),
        'core_mean_motion_efficiency': avg(core, lambda tribe: tribe.last_motion_efficiency),
        'core_mean_waste_motion': avg(core, lambda tribe: tribe.last_waste_motion),
        'core_mean_idle_motion': avg(core, lambda tribe: tribe.last_idle_motion),
        'core_mean_guard_error': avg(core, lambda tribe: tribe.last_guard_error),
        'core_mean_oscillation': avg(core, lambda tribe: tribe.last_oscillation),
        'core_mean_lateral_motion': avg(core, lambda tribe: tribe.last_lateral_motion),
    }


def choose_tribe_replacements(population_summary, signal):
    ranked = population_summary.get('leaderboard') or population_summary['ranked']
    if len(ranked) < 2:
        return []

    replacements = [(ranked[0], ranked[-1])]
    if len(ranked) < 4:
        return replacements

    donor = ranked[1]
    target = ranked[-2]
    selection_gap = donor.selection_score - target.selection_score
    acc_gap = donor.last_task_accuracy - target.last_task_accuracy
    coverage_gap = donor.last_note_coverage - target.last_note_coverage
    stability_gap = donor.calculate_stability() - target.calculate_stability()
    target_is_dragging = (
        target.last_fail_rate > 0.50
        or target.last_energy < 0.32
        or target.calculate_stability() < 0.44
        or target.last_note_coverage < 0.18
    )

    if signal >= 18.0 and (
        selection_gap >= 12.0
        or acc_gap >= 4.0
        or coverage_gap >= 0.15
        or stability_gap >= 0.08
        or target_is_dragging
    ):
        replacements.append((donor, target))

    return replacements


def update_adaptive_state(
    adaptive_state,
    epoch,
    global_best_acc,
    strict_eval_acc=None,
    matched_eval_acc=None,
    current_task_acc=None,
    mean_stability=None,
    mean_note_coverage=None,
    mean_energy=None,
    mean_completion=None,
    mean_fail_rate=None,
    mean_speed=None,
    mean_style_violation=None,
    mean_angular_violation=None,
    mean_motion_efficiency=None,
    mean_waste_motion=None,
    mean_idle_motion=None,
    mean_guard_error=None,
    mean_oscillation=None,
    mean_lateral_motion=None,
):
    signal = live_training_signal(
        adaptive_state=adaptive_state,
        current_task_acc=current_task_acc,
        strict_eval_acc=strict_eval_acc,
        matched_eval_acc=matched_eval_acc,
        fallback=global_best_acc,
    )
    progress_threshold = 0.25 if signal < 5.0 else 0.75
    if strict_eval_acc is not None:
        adaptive_state['last_eval_accuracy'] = strict_eval_acc
        adaptive_state['strict_eval_accuracy'] = strict_eval_acc
        adaptive_state['best_eval_accuracy'] = max(adaptive_state['best_eval_accuracy'], strict_eval_acc)
    if matched_eval_acc is not None:
        adaptive_state['matched_eval_accuracy'] = matched_eval_acc
        adaptive_state['best_matched_eval_accuracy'] = max(adaptive_state['best_matched_eval_accuracy'], matched_eval_acc)

    if signal > adaptive_state['best_signal_accuracy']:
        adaptive_state['best_signal_accuracy'] = signal

    run_best_signal = float(adaptive_state.get('run_best_signal_accuracy', 0.0))
    if signal > run_best_signal + progress_threshold:
        adaptive_state['run_best_signal_accuracy'] = signal
        adaptive_state['stagnation_epochs'] = 0
        adaptive_state['last_progress_epoch'] = epoch
    else:
        adaptive_state['stagnation_epochs'] += 1

    if mean_stability is not None:
        stability_threshold = 0.015 if mean_stability < 0.4 else 0.025
        run_best_stability = float(adaptive_state.get('run_best_stability', 0.0))
        if (
            mean_stability > run_best_stability + stability_threshold
            or (mean_fail_rate is not None and mean_fail_rate < adaptive_state.get('last_mean_fail_rate', 1.0) - 0.04)
            or (mean_completion is not None and mean_completion > adaptive_state.get('last_mean_completion', 0.0) + 0.04)
        ):
            adaptive_state['run_best_stability'] = max(run_best_stability, mean_stability)
            adaptive_state['best_stability'] = max(adaptive_state['best_stability'], mean_stability)
            adaptive_state['stability_stagnation_epochs'] = 0
        else:
            adaptive_state['stability_stagnation_epochs'] += 1
        adaptive_state['last_mean_stability'] = mean_stability
    if mean_note_coverage is not None:
        adaptive_state['last_mean_note_coverage'] = mean_note_coverage
    if mean_energy is not None:
        adaptive_state['last_mean_energy'] = mean_energy
    if mean_completion is not None:
        adaptive_state['last_mean_completion'] = mean_completion
    if mean_fail_rate is not None:
        adaptive_state['last_mean_fail_rate'] = mean_fail_rate
    if mean_speed is not None:
        adaptive_state['last_mean_speed'] = mean_speed
    if mean_style_violation is not None:
        adaptive_state['last_mean_style_violation'] = mean_style_violation
    if mean_angular_violation is not None:
        adaptive_state['last_mean_angular_violation'] = mean_angular_violation
    if mean_motion_efficiency is not None:
        adaptive_state['last_mean_motion_efficiency'] = mean_motion_efficiency
    if mean_waste_motion is not None:
        adaptive_state['last_mean_waste_motion'] = mean_waste_motion
    if mean_idle_motion is not None:
        adaptive_state['last_mean_idle_motion'] = mean_idle_motion
    if mean_guard_error is not None:
        adaptive_state['last_mean_guard_error'] = mean_guard_error
    if mean_oscillation is not None:
        adaptive_state['last_mean_oscillation'] = mean_oscillation
    if mean_lateral_motion is not None:
        adaptive_state['last_mean_lateral_motion'] = mean_lateral_motion

    recovery_signal = live_training_signal(
        adaptive_state=adaptive_state,
        current_task_acc=current_task_acc,
        strict_eval_acc=strict_eval_acc,
        matched_eval_acc=matched_eval_acc,
        fallback=signal,
    )
    stability_value = float(adaptive_state.get('last_mean_stability', 0.0) or 0.0)
    coverage_value = float(adaptive_state.get('last_mean_note_coverage', 0.0) or 0.0)
    energy_value = float(adaptive_state.get('last_mean_energy', 0.0) or 0.0)
    fail_rate_value = float(adaptive_state.get('last_mean_fail_rate', 1.0) or 0.0)

    collapse_risk = (
        coverage_value < 0.08
        or energy_value < 0.34
        or stability_value < 0.42
        or fail_rate_value > 0.45
        or adaptive_state['stagnation_epochs'] >= 12
    )
    healthy_recovery = (
        recovery_signal >= 5.0
        and coverage_value >= 0.10
        and energy_value >= 0.40
        and stability_value >= 0.44
        and fail_rate_value <= 0.35
    )

    stability_cap, style_cap = mature_rehab_caps(
        signal,
        strict_eval_acc,
        mean_stability,
        mean_energy,
        mean_completion,
        mean_fail_rate,
        mean_motion_efficiency,
        mean_idle_motion,
        mean_guard_error,
    )

    target = 0
    if signal < 15.0:
        target = 1
    if signal < 5.0:
        target = 2
    if signal < 2.0:
        target = 3
    if signal < 1.0:
        target = 4

    if adaptive_state['stagnation_epochs'] >= 8 and signal < 15.0:
        target += 1
    if adaptive_state['stagnation_epochs'] >= 16 and signal < 8.0:
        target += 1
    if adaptive_state['bc_probe_accuracy'] >= 5.0 and signal < 2.0 and epoch >= 8:
        target += 1

    target = max(0, min(4, target))
    if target > adaptive_state['rehab_level']:
        adaptive_state['rehab_level'] += 1
        adaptive_state['rehab_release_streak'] = 0
    elif target < adaptive_state['rehab_level']:
        if healthy_recovery:
            adaptive_state['rehab_release_streak'] += 1
            if adaptive_state['rehab_release_streak'] >= 3:
                adaptive_state['rehab_level'] -= 1
                adaptive_state['rehab_release_streak'] = 0
        else:
            adaptive_state['rehab_release_streak'] = 0
    else:
        adaptive_state['rehab_release_streak'] = 0

    if adaptive_state['rehab_level'] >= 3:
        if recovery_signal < 5.0 or collapse_risk:
            adaptive_state['escape_support_active'] = True
            adaptive_state['escape_release_streak'] = 0
        elif adaptive_state.get('escape_support_active', False):
            if healthy_recovery:
                adaptive_state['escape_release_streak'] += 1
                if adaptive_state['escape_release_streak'] >= 3:
                    adaptive_state['escape_support_active'] = False
                    adaptive_state['escape_release_streak'] = 0
            else:
                adaptive_state['escape_release_streak'] = 0
        else:
            adaptive_state['escape_release_streak'] = 0
    else:
        adaptive_state['escape_support_active'] = False
        adaptive_state['escape_release_streak'] = 0

    stability_rehab = 0
    if signal >= 8.0 and mean_stability is not None and mean_stability < 0.42:
        stability_rehab = 1
    if signal >= 15.0 and (
        (mean_stability is not None and mean_stability < 0.34)
        or (mean_fail_rate is not None and mean_fail_rate > 0.45)
        or (mean_energy is not None and mean_energy < 0.38)
    ):
        stability_rehab = 2
    if signal >= 20.0 and (
        (mean_stability is not None and mean_stability < 0.28)
        or (mean_fail_rate is not None and mean_fail_rate > 0.58)
        or adaptive_state['stability_stagnation_epochs'] >= 6
    ):
        stability_rehab = 3
    if signal >= 25.0 and (
        (mean_stability is not None and mean_stability < 0.24)
        or (mean_completion is not None and mean_completion < 0.65)
        or adaptive_state['stability_stagnation_epochs'] >= 10
    ):
        stability_rehab = 4

    stability_rehab = max(0, min(stability_cap, stability_rehab))
    if stability_rehab > adaptive_state['stability_rehab_level']:
        adaptive_state['stability_rehab_level'] += 1
    elif stability_rehab < adaptive_state['stability_rehab_level']:
        adaptive_state['stability_rehab_level'] -= 1

    style_rehab = 0
    if signal >= 18.0 and (
        (mean_style_violation is not None and mean_style_violation > 0.18)
        or (mean_motion_efficiency is not None and mean_motion_efficiency < 0.22)
        or (mean_waste_motion is not None and mean_waste_motion > 0.030)
        or (mean_idle_motion is not None and mean_idle_motion > 0.018)
        or (mean_guard_error is not None and mean_guard_error > 0.16)
    ):
        style_rehab = 1
    if signal >= 28.0 and (
        (mean_style_violation is not None and mean_style_violation > 0.28)
        or (mean_motion_efficiency is not None and mean_motion_efficiency < 0.18)
        or (mean_waste_motion is not None and mean_waste_motion > 0.045)
        or (mean_idle_motion is not None and mean_idle_motion > 0.028)
        or (mean_guard_error is not None and mean_guard_error > 0.24)
        or (mean_oscillation is not None and mean_oscillation > 0.08)
    ):
        style_rehab = 2
    if signal >= 40.0 and (
        (mean_style_violation is not None and mean_style_violation > 0.38)
        or (mean_motion_efficiency is not None and mean_motion_efficiency < 0.15)
        or (mean_waste_motion is not None and mean_waste_motion > 0.060)
        or (mean_idle_motion is not None and mean_idle_motion > 0.036)
        or (mean_guard_error is not None and mean_guard_error > 0.30)
        or (mean_oscillation is not None and mean_oscillation > 0.12)
        or (mean_lateral_motion is not None and mean_lateral_motion > 0.10)
    ):
        style_rehab = 3
    if signal >= 55.0 and (
        (mean_style_violation is not None and mean_style_violation > 0.48)
        or (mean_motion_efficiency is not None and mean_motion_efficiency < 0.12)
        or (mean_waste_motion is not None and mean_waste_motion > 0.075)
        or (mean_idle_motion is not None and mean_idle_motion > 0.045)
        or (mean_guard_error is not None and mean_guard_error > 0.36)
        or (mean_oscillation is not None and mean_oscillation > 0.15)
        or (mean_lateral_motion is not None and mean_lateral_motion > 0.13)
    ):
        style_rehab = 4

    style_rehab = max(0, min(style_cap, style_rehab))
    if style_rehab > adaptive_state['style_rehab_level']:
        adaptive_state['style_rehab_level'] += 1
    elif style_rehab < adaptive_state['style_rehab_level']:
        adaptive_state['style_rehab_level'] -= 1

    return adaptive_state


def get_recovery_profile(adaptive_state, global_best_acc, current_task_acc=None):
    rehab_level = int(adaptive_state.get('rehab_level', 0))
    stability_rehab_level = int(adaptive_state.get('stability_rehab_level', 0))
    style_rehab_level = int(adaptive_state.get('style_rehab_level', 0))
    signal = live_training_signal(
        adaptive_state=adaptive_state,
        current_task_acc=current_task_acc,
        fallback=global_best_acc,
    )
    stability_signal = adaptive_state.get('last_mean_stability', 0.0)
    mean_note_coverage = adaptive_state.get('last_mean_note_coverage', 0.0)
    mean_energy = adaptive_state.get('last_mean_energy', 0.0)
    mean_fail_rate = adaptive_state.get('last_mean_fail_rate', 1.0)
    escape_support_active = bool(adaptive_state.get('escape_support_active', False))

    training_wheels = 0.0
    if signal < 25.0:
        training_wheels = 0.2
    if signal < 12.0:
        training_wheels = 0.45
    if signal < 5.0:
        training_wheels = 0.75
    if signal < 2.0:
        training_wheels = 1.0
    training_wheels = min(1.0, training_wheels + rehab_level * 0.05)
    if stability_rehab_level > 0:
        training_wheels = max(training_wheels, min(0.55, 0.10 + stability_rehab_level * 0.08))

    if signal < 2.0:
        miss_scale = 0.35
        motion_scale = 0.30
    elif signal < 5.0:
        miss_scale = 0.45
        motion_scale = 0.45
    elif signal < 12.0:
        miss_scale = 0.65
        motion_scale = 0.65
    elif signal < 25.0:
        miss_scale = 0.85
        motion_scale = 0.85
    else:
        miss_scale = 1.0
        motion_scale = 1.0
    if stability_rehab_level > 0:
        motion_scale = min(1.35, motion_scale + 0.08 * stability_rehab_level)

    dense_scale = 2.0 + (rehab_level * 0.75)
    if signal < 5.0:
        dense_scale += 1.0
    if stability_rehab_level > 0:
        dense_scale += 0.20 * stability_rehab_level
    if signal < 5.0 and mean_note_coverage < 0.15:
        dense_scale += 1.25
    if signal < 2.0 and mean_note_coverage < 0.10:
        dense_scale += 0.75

    assist_level = 0.0
    if signal < 25.0:
        assist_level = 0.15
    if signal < 12.0:
        assist_level = 0.35
    if signal < 5.0:
        assist_level = 0.65
    if signal < 2.0:
        assist_level = 0.95
    assist_level = min(1.0, assist_level + rehab_level * 0.05)
    if stability_rehab_level > 0:
        assist_floor = 0.10 + 0.08 * stability_rehab_level
        if stability_signal < 0.30 or mean_fail_rate > 0.50:
            assist_floor += 0.05
        assist_level = max(assist_level, min(0.55, assist_floor))

    survival_level = max(0.0, min(1.0, assist_level + 0.10))
    if stability_rehab_level > 0:
        survival_floor = 0.20 + 0.14 * stability_rehab_level
        if mean_energy < 0.35:
            survival_floor += 0.05
        survival_level = max(survival_level, min(0.80, survival_floor))
    collapse_rehab = rehab_level >= 3 and (escape_support_active or mean_note_coverage < 0.05)
    if collapse_rehab:
        training_wheels = max(training_wheels, 0.90)
        assist_level = max(assist_level, 0.90)
        survival_level = max(survival_level, 0.95)
    fail_enabled = not (signal < 2.0 and rehab_level >= 2)
    if rehab_level >= 3 and (signal < 5.0 or collapse_rehab):
        fail_enabled = False
    if signal >= 10.0 and stability_rehab_level >= 2 and mean_fail_rate > 0.70:
        fail_enabled = False
    stability_reward_level = min(1.0, 0.20 + 0.20 * stability_rehab_level) if stability_rehab_level > 0 else 0.0
    style_guidance_level = min(1.0, 0.25 + 0.20 * style_rehab_level) if style_rehab_level > 0 else 0.0
    if signal < 5.0 and mean_note_coverage < 0.15:
        style_guidance_level = max(style_guidance_level, 0.12)
    if signal < 2.0 and mean_note_coverage < 0.10:
        style_guidance_level = max(style_guidance_level, 0.18)

    return {
        'rehab_level': rehab_level,
        'stability_rehab_level': stability_rehab_level,
        'style_rehab_level': style_rehab_level,
        'training_wheels': training_wheels,
        'miss_scale': miss_scale,
        'motion_scale': motion_scale,
        'dense_scale': dense_scale,
        'assist_level': assist_level,
        'survival_level': survival_level,
        'stability_reward_level': stability_reward_level,
        'style_guidance_level': style_guidance_level,
        'fail_enabled': fail_enabled,
    }


def build_training_matched_eval_profile(tribe, recovery):
    assist_level = float(recovery.get('assist_level', 0.0))
    return {
        'action_repeat': int(max(1, getattr(tribe, 'current_action_repeat', tribe.hparams.get('action_repeat', 2)))),
        'smoothing_alpha': 1.0,
        'training_wheels_level': float(recovery.get('training_wheels', 0.0)),
        'assist_level': assist_level,
        'survival_assistance': float(recovery.get('survival_level', 0.0)),
        'stability_reward_level': float(recovery.get('stability_reward_level', 0.0)),
        'style_guidance_level': float(recovery.get('style_guidance_level', 0.0)),
        'hit_timing_profile': "assisted" if assist_level > 0.0 else "default",
        'fail_enabled': bool(recovery.get('fail_enabled', True)),
        'saber_inertia': float(getattr(tribe, 'current_saber_inertia', tribe.hparams.get('saber_inertia', 0.35))),
        'rot_clamp': float(getattr(tribe, 'current_rot_clamp', tribe.hparams.get('rot_clamp', 0.10))),
        'pos_clamp': float(getattr(tribe, 'current_pos_clamp', 0.15)),
    }


def policy_std_cap(signal_acc, rehab_level, style_rehab_level=0):
    if signal_acc < 2.0:
        cap = 0.110
    elif signal_acc < 5.0:
        cap = 0.100
    elif signal_acc < 12.0:
        cap = 0.090
    elif signal_acc < 25.0:
        cap = 0.110
    elif signal_acc < 40.0:
        cap = 0.140
    else:
        cap = 0.180
    if signal_acc >= 5.0:
        cap *= max(0.85, 1.0 - rehab_level * 0.03)
    else:
        cap *= max(0.95, 1.0 - rehab_level * 0.01)
    if style_rehab_level >= 2 and signal_acc >= 30.0:
        cap = min(cap, 0.14)
    if style_rehab_level >= 3 and signal_acc >= 45.0:
        cap = min(cap, 0.12)
    if style_rehab_level >= 4 and signal_acc >= 60.0:
        cap = min(cap, 0.10)
    return float(max(0.035, min(0.22, cap)))


def apply_control_profile(tribe, global_best_acc, rehab_level, stability_rehab_level=0, style_rehab_level=0, current_task_acc=None):
    live_signal = max_nonnegative_signal(current_task_acc)
    signal = max(
        getattr(tribe, 'last_task_accuracy', tribe.moving_acc),
        live_signal if live_signal > 0.0 else global_best_acc,
    )
    action_repeat = int(tribe.hparams.get('action_repeat', 2))
    inertia = float(tribe.hparams.get('saber_inertia', 0.35))
    rot_clamp = float(tribe.hparams.get('rot_clamp', 0.10))
    pos_clamp = float(getattr(tribe, 'current_pos_clamp', 0.15))
    tribe_stability = tribe.calculate_stability()

    if signal < 2.0 or rehab_level >= 3:
        action_repeat = 1
        inertia = min(inertia, 0.12)
        rot_clamp = max(rot_clamp, 0.120)
        pos_clamp = max(pos_clamp, 0.150)
    elif signal < 5.0 or rehab_level >= 2:
        action_repeat = 1
        inertia = min(inertia, 0.18)
        rot_clamp = max(rot_clamp, 0.112)
        pos_clamp = max(pos_clamp, 0.148)
    elif signal < 12.0 or rehab_level >= 1:
        action_repeat = min(action_repeat, 2)
        inertia = min(inertia, 0.28)
        rot_clamp = max(rot_clamp, 0.104)
        pos_clamp = max(pos_clamp, 0.142)

    if stability_rehab_level >= 1 and signal >= 10.0 and tribe_stability < 0.42:
        action_repeat = min(action_repeat, 1)
        inertia = max(inertia, 0.52)
        rot_clamp = min(rot_clamp, 0.078)
        pos_clamp = min(pos_clamp, 0.100)
    if stability_rehab_level >= 2 and signal >= 15.0 and tribe_stability < 0.34:
        action_repeat = 1
        inertia = max(inertia, 0.58)
        rot_clamp = min(rot_clamp, 0.072)
        pos_clamp = min(pos_clamp, 0.095)
    if stability_rehab_level >= 3 and tribe_stability < 0.28:
        action_repeat = 1
        inertia = max(inertia, 0.64)
        rot_clamp = min(rot_clamp, 0.068)
        pos_clamp = min(pos_clamp, 0.090)

    if style_rehab_level >= 1 and signal >= 20.0:
        inertia = max(inertia, 0.42)
        rot_clamp = min(rot_clamp, 0.095)
        pos_clamp = min(pos_clamp, 0.11)
    if style_rehab_level >= 2 and signal >= 32.0:
        inertia = max(inertia, 0.48)
        rot_clamp = min(rot_clamp, 0.088)
        pos_clamp = min(pos_clamp, 0.10)
    if style_rehab_level >= 3 and signal >= 45.0:
        inertia = max(inertia, 0.56)
        rot_clamp = min(rot_clamp, 0.074)
        pos_clamp = min(pos_clamp, 0.09)
    if style_rehab_level >= 4 and signal >= 60.0:
        action_repeat = min(action_repeat, 2)
        inertia = max(inertia, 0.62)
        rot_clamp = min(rot_clamp, 0.070)
        pos_clamp = min(pos_clamp, 0.088)

    tribe.current_action_repeat = int(max(1, min(3, action_repeat)))
    tribe.current_saber_inertia = float(max(0.0, min(0.8, inertia)))
    tribe.current_rot_clamp = float(max(0.05, min(0.12, rot_clamp)))
    tribe.current_pos_clamp = float(max(0.06, min(0.16, pos_clamp)))
    return tribe.current_action_repeat, tribe.current_saber_inertia, tribe.current_rot_clamp, tribe.current_pos_clamp


def effective_reference_coeffs(tribe, strict_anchor_signal, rehab_level, style_rehab_level=0):
    progress = max(0.0, float(strict_anchor_signal))
    fade = min(1.0, progress / 24.0)
    coverage = float(getattr(tribe, 'last_note_coverage', 0.0) or 0.0)
    rehab_boost = 1.0 + (0.12 * min(2, rehab_level))
    if progress < 5.0:
        rehab_boost = min(rehab_boost, 1.10)
    elif progress < 12.0:
        rehab_boost = min(rehab_boost, 1.20)
    style_boost = 1.0 + (0.12 * style_rehab_level)
    bc_mean_coeff = tribe.hparams['bc_mean_coeff'] * rehab_boost * style_boost * (1.0 - 0.88 * fade)
    bc_kl_coeff = tribe.hparams['bc_kl_coeff'] * (1.0 - 0.82 * fade)
    if progress < 5.0 and coverage < 0.12:
        bc_mean_coeff *= 0.35
        bc_kl_coeff *= 0.40
    if progress < 3.0 and coverage < 0.15:
        bc_mean_coeff *= 0.40
        bc_kl_coeff *= 0.45
    if tribe.last_task_accuracy < 2.0 and coverage < 0.10 and tribe.stagnation_counter >= 8:
        bc_mean_coeff *= 0.35
        bc_kl_coeff *= 0.40
    if progress >= 8.0:
        bc_mean_coeff *= 0.85
    if progress < 5.0:
        if coverage < 0.08:
            mean_floor = 0.0
            kl_floor = 0.0
        else:
            mean_floor = 0.04 if coverage < 0.12 else 0.10
            kl_floor = 0.0005 if coverage < 0.12 else 0.0015
        bc_mean_coeff = max(bc_mean_coeff, mean_floor)
        bc_kl_coeff = max(bc_kl_coeff, kl_floor)
    if tribe.last_task_accuracy < 4.0 and tribe.stagnation_counter >= 6:
        if coverage < 0.08:
            mean_floor = 0.0
            kl_floor = 0.0
        else:
            mean_floor = 0.03 if coverage < 0.12 else 0.08
            kl_floor = 0.0005 if coverage < 0.12 else 0.001
        bc_mean_coeff = max(mean_floor, bc_mean_coeff * 0.70)
        bc_kl_coeff = max(kl_floor, bc_kl_coeff * 0.70)
    if tribe.last_task_accuracy < 2.0 and coverage < 0.08 and tribe.stagnation_counter >= 10:
        bc_mean_coeff = min(bc_mean_coeff, 0.02)
        bc_kl_coeff = min(bc_kl_coeff, 0.0005)
    bc_delta_coeff = max(0.0, bc_mean_coeff * (0.20 + 0.08 * style_rehab_level))
    if progress < 8.0 and coverage < 0.15:
        bc_delta_coeff = 0.0
    return max(0.0, bc_kl_coeff), max(0.0, bc_mean_coeff), bc_delta_coeff


def build_curriculum_buckets(curriculum, map_cache, replay_backed_hashes=None):
    filtered = [c for c in curriculum if c.get('hash') in map_cache]
    if not filtered:
        return {}

    enriched = []
    for item in filtered:
        map_hash = item.get('hash')
        beatmap, _ = map_cache.get(map_hash, ({}, 0.0))
        notes = beatmap.get('notes', []) if isinstance(beatmap, dict) else []
        scorable_notes = [note for note in notes if int(note.get('type', -1)) != 3]
        obstacle_count = len(beatmap.get('obstacles', [])) if isinstance(beatmap, dict) else 0
        scorable_count = max(1, len(scorable_notes))
        meta_item = dict(item)
        meta_item['note_count'] = max(meta_item.get('note_count', 0), len(scorable_notes))
        meta_item['obstacle_count'] = obstacle_count
        meta_item['obstacle_ratio'] = obstacle_count / scorable_count
        enriched.append(meta_item)

    filtered = enriched
    filtered.sort(key=lambda c: (c.get('nps', 0.0), c.get('obstacle_ratio', 0.0), c.get('obstacle_count', 0), c.get('hash', '')))
    replay_backed = {str(h).strip().lower() for h in (replay_backed_hashes or set())}

    def take(predicate):
        hashes = [c['hash'] for c in filtered if predicate(c)]
        return hashes or [c['hash'] for c in filtered]

    def take_seen(predicate):
        hashes = [
            c['hash'] for c in filtered
            if predicate(c) and str(c.get('hash', '')).strip().lower() in replay_backed
        ]
        return hashes

    def clean_micro(candidate):
        return (
            candidate.get('nps', 0.0) < 1.5
            and candidate.get('obstacle_ratio', 0.0) <= 0.18
            and candidate.get('obstacle_count', 0) <= 24
        )

    def clean_bootstrap(candidate):
        return (
            candidate.get('nps', 0.0) < 2.2
            and candidate.get('obstacle_ratio', 0.0) <= 0.22
            and candidate.get('obstacle_count', 0) <= 36
        )

    def clean_easy(candidate):
        return (
            candidate.get('nps', 0.0) < 3.0
            and candidate.get('obstacle_ratio', 0.0) <= 0.26
            and candidate.get('obstacle_count', 0) <= 48
        )

    return {
        'all': [c['hash'] for c in filtered],
        'micro': take(lambda c: c.get('nps', 0.0) < 1.5),
        'bootstrap': take(lambda c: c.get('nps', 0.0) < 2.2),
        'easy': take(lambda c: c.get('nps', 0.0) < 3.0),
        'medium': take(lambda c: c.get('nps', 0.0) < 4.5),
        'hard': take(lambda c: c.get('nps', 0.0) < 6.0),
        'micro_clean': take(clean_micro),
        'bootstrap_clean': take(clean_bootstrap),
        'easy_clean': take(clean_easy),
        'micro_seen': take_seen(lambda c: c.get('nps', 0.0) < 1.5),
        'bootstrap_seen': take_seen(lambda c: c.get('nps', 0.0) < 2.2),
        'easy_seen': take_seen(lambda c: c.get('nps', 0.0) < 3.0),
        'medium_seen': take_seen(lambda c: c.get('nps', 0.0) < 4.5),
        'micro_clean_seen': take_seen(clean_micro),
        'bootstrap_clean_seen': take_seen(clean_bootstrap),
        'easy_clean_seen': take_seen(clean_easy),
        'expert': take(lambda c: c.get('nps', 0.0) >= 4.0),
        'meta': {c['hash']: c for c in filtered},
    }


def _quantize_capacity(value, steps):
    value = max(1, int(value))
    for limit, step in steps:
        if value <= limit:
            return int(((value + step - 1) // step) * step)
    step = steps[-1][1]
    return int(((value + step - 1) // step) * step)


def quantize_note_capacity(value):
    return _quantize_capacity(value, (
        (256, 32),
        (1024, 64),
        (2048, 128),
        (4096, 256),
        (8192, 512),
        (1 << 30, 1024),
    ))


def quantize_obstacle_capacity(value):
    return _quantize_capacity(value, (
        (128, 16),
        (512, 32),
        (2048, 64),
        (8192, 256),
        (1 << 30, 512),
    ))


def choose_epoch_sim_capacity(sampled_maps):
    max_notes = max((len(beatmap.get('notes', [])) for beatmap in sampled_maps), default=1)
    max_obstacles = max((len(beatmap.get('obstacles', [])) for beatmap in sampled_maps), default=1)
    return {
        'notes_actual': max_notes,
        'obstacles_actual': max_obstacles,
        'notes_capacity': quantize_note_capacity(max_notes),
        'obstacles_capacity': quantize_obstacle_capacity(max_obstacles),
    }


def sample_hash_pool(pool, size):
    unique_pool = list(dict.fromkeys(pool or []))
    if size <= 0 or not unique_pool:
        return []
    if len(unique_pool) >= size:
        return list(np.random.choice(unique_pool, size=size, replace=False))

    shuffled = unique_pool.copy()
    np.random.shuffle(shuffled)
    extra = list(np.random.choice(unique_pool, size=size - len(shuffled), replace=True))
    return shuffled + extra


def pick_benchmark_hash(curriculum_buckets):
    hashes = curriculum_buckets.get('all', [])
    meta = curriculum_buckets.get('meta', {})
    if not hashes:
        return None
    ranked = sorted(hashes, key=lambda h: (meta.get(h, {}).get('nps', 0.0), h))
    idx = min(len(ranked) - 1, max(0, int((len(ranked) - 1) * 0.65)))
    return ranked[idx]


def select_map_pools(tribe, curriculum_buckets, global_best_acc, rehab_level, stagnation_epochs=0, current_task_acc=None):
    micro = curriculum_buckets['micro']
    bootstrap = curriculum_buckets['bootstrap']
    easy = curriculum_buckets['easy']
    medium = curriculum_buckets['medium']
    hard = curriculum_buckets['hard']
    expert = curriculum_buckets['expert']
    all_maps = curriculum_buckets['all']
    micro_seen = curriculum_buckets.get('micro_seen') or micro
    bootstrap_seen = curriculum_buckets.get('bootstrap_seen') or bootstrap
    easy_seen = curriculum_buckets.get('easy_seen') or easy
    medium_seen = curriculum_buckets.get('medium_seen') or medium
    micro_clean = curriculum_buckets.get('micro_clean') or micro
    bootstrap_clean = curriculum_buckets.get('bootstrap_clean') or bootstrap
    easy_clean = curriculum_buckets.get('easy_clean') or easy
    micro_clean_seen = curriculum_buckets.get('micro_clean_seen') or micro_seen
    bootstrap_clean_seen = curriculum_buckets.get('bootstrap_clean_seen') or bootstrap_seen
    easy_clean_seen = curriculum_buckets.get('easy_clean_seen') or easy_seen

    tribe_skill = float(getattr(tribe, 'last_task_accuracy', tribe.moving_acc) or 0.0)
    live_signal = max_nonnegative_signal(current_task_acc)
    if live_signal > 0.0:
        skill = max(tribe_skill, live_signal * 0.8)
    else:
        skill = max(tribe_skill, global_best_acc * 0.8)
    low_skill_cover = float(getattr(tribe, 'last_note_coverage', 0.0) or 0.0)
    needs_micro_escape = skill < 3.0 or low_skill_cover < 0.12
    if rehab_level >= 4:
        if needs_micro_escape:
            return micro_clean_seen, bootstrap_clean_seen, "REHAB-4-ESCAPE"
        if stagnation_epochs >= 24:
            return bootstrap_clean_seen, easy_clean_seen, "REHAB-4-ESCAPE"
        if stagnation_epochs >= 12:
            return bootstrap_clean_seen, easy_clean_seen, "REHAB-4-WIDE"
        return micro_clean_seen, bootstrap_clean_seen, "REHAB-4"
    if rehab_level >= 3:
        if needs_micro_escape:
            return micro_clean_seen, bootstrap_clean_seen, "REHAB-3-ESCAPE"
        if stagnation_epochs >= 12:
            return bootstrap_clean_seen, easy_clean_seen, "REHAB-3-WIDE"
        if tribe.id == 0:
            return micro_clean_seen, bootstrap_clean_seen, "REHAB-3"
        return bootstrap_clean_seen, easy_clean_seen, "REHAB-3"
    if rehab_level >= 2:
        if tribe.id == 0:
            return bootstrap_clean_seen, easy_clean_seen, "REHAB-2"
        return easy_clean_seen, bootstrap_clean_seen, "REHAB-2"
    if rehab_level >= 1 and skill < 25.0:
        return bootstrap_clean_seen, easy_clean_seen, "REHAB-1"
    if tribe.id == 0 and skill < 45.0:
        return bootstrap_clean_seen, easy_clean_seen, "BOOTSTRAP"
    if skill < 30.0:
        return easy_clean_seen, medium_seen, "EASY"
    if skill < 55.0:
        return medium, easy_clean, "MEDIUM"
    if skill < 78.0:
        return hard, medium, "HARD"
    return expert or all_maps, hard, "EXPERT"


def sample_maps_for_epoch(tribes, curriculum_buckets, map_cache, envs_per_tribe, global_best_acc, rehab_level, stagnation_epochs=0, current_task_acc=None):
    all_notes, all_bpms, tribe_tiers = [], [], []
    for tribe in tribes:
        primary, fallback, tier = select_map_pools(
            tribe,
            curriculum_buckets,
            global_best_acc,
            rehab_level,
            stagnation_epochs=stagnation_epochs,
            current_task_acc=current_task_acc,
        )
        tribe_tiers.append(tier)
        primary_ratio = 0.8
        if tier.endswith("ESCAPE"):
            primary_ratio = 1.0
        if rehab_level >= 4 and stagnation_epochs >= 24:
            primary_ratio = 0.55
        elif rehab_level >= 4 and stagnation_epochs >= 12:
            primary_ratio = 0.65
        elif rehab_level >= 3 and stagnation_epochs >= 12:
            primary_ratio = 0.70
        if tier.endswith("ESCAPE"):
            primary_ratio = 1.0
        count_primary = max(1, int(round(envs_per_tribe * primary_ratio)))
        count_primary = min(envs_per_tribe, count_primary)
        count_fallback = envs_per_tribe - count_primary
        chosen_hashes = sample_hash_pool(primary, count_primary)
        if count_fallback > 0:
            chosen_hashes.extend(sample_hash_pool(fallback, count_fallback))
        for map_hash in chosen_hashes:
            notes, bpm = map_cache[map_hash]
            all_notes.append(notes)
            all_bpms.append(bpm)
    return all_notes, all_bpms, tribe_tiers


def effective_bc_kl(base_coeff, tribe_acc, global_best_acc):
    anchor_acc = max(tribe_acc, global_best_acc)
    progress = min(1.0, anchor_acc / 90.0)
    return base_coeff * (1.0 - 0.85 * progress)


def choose_ppo_epochs(epoch, tribe):
    if tribe.last_task_accuracy >= 75.0:
        return 3
    if epoch >= 250 and tribe.last_task_accuracy >= 50.0:
        return 3
    return 2


def _remap_state_dict(state_dict, model):
    """Load weights with backward compatibility for old model layouts."""
    model_sd = model.state_dict()
    sd_keys  = set(state_dict.keys())
    
    if sd_keys == set(model_sd.keys()):
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint shape mismatch for the current policy state layout. "
                "Rebuild BC shards and retrain BC before reusing this checkpoint."
            ) from exc
        print(f"  [Load] Exact key match: loaded {len(sd_keys)} keys.")
        return

    # Try to match old shared-backbone keys to the new split architecture.
    # Old: features.0.weight → New: actor_features.0.weight + critic_features.0.weight
    # Only transfer weights where shapes match — the old 512-wide layers can't
    # fit into 256-wide layers, so those are silently skipped.
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
            "Rebuild BC shards and retrain BC before reusing this checkpoint."
        )

    print(f"  [Load] Remapped {len(remapped)}/{len(model_sd)} keys from checkpoint.")

    try:
        model.load_state_dict(remapped, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint shape mismatch for the current policy state layout. "
            "Rebuild BC shards and retrain BC before reusing this checkpoint."
        ) from exc


def load_frozen_reference_model(model_path, device):
    """Frozen bootstrap prior used to keep PPO close to the chosen handoff policy."""
    if not model_path or not os.path.exists(model_path):
        return None

    model = ActorCritic().to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    _remap_state_dict(
        extract_policy_state_dict(
            ckpt,
            checkpoint_path=model_path,
            accepted_keys=("model_state_dict", "actor_state_dict"),
            allow_legacy=True,
        ),
        model,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def reference_delta_direction_loss(mean, ref_mean, current_pose):
    pred_delta = torch.stack(
        (mean[:, 7:10] - current_pose[:, 7:10], mean[:, 14:17] - current_pose[:, 14:17]),
        dim=1,
    )
    ref_delta = torch.stack(
        (ref_mean[:, 7:10] - current_pose[:, 7:10], ref_mean[:, 14:17] - current_pose[:, 14:17]),
        dim=1,
    )
    pred_dir = pred_delta / torch.norm(pred_delta, dim=-1, keepdim=True).clamp(min=1e-6)
    ref_dir = ref_delta / torch.norm(ref_delta, dim=-1, keepdim=True).clamp(min=1e-6)
    direction_error = 1.0 - (pred_dir * ref_dir).sum(dim=-1).clamp(-1.0, 1.0)
    active = (torch.norm(ref_delta, dim=-1) > 1e-4).float()
    return (direction_error * active).sum() / active.sum().clamp(min=1.0)


def compute_ppo_loss(states, actions, log_probs_old, returns, advantages, model,
                     clip_param=0.2, entropy_coeff=0.01,
                     reference_model=None, bc_kl_coeff=0.0, bc_mean_coeff=0.0,
                     bc_delta_coeff=0.0):
    mean, std, value = model(states)
    dist = Normal(mean, std, validate_args=False)
    log_probs = policy_action_log_prob(mean, std, actions)
    ratio = torch.exp((log_probs - log_probs_old).clamp(-20.0, 20.0))

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages

    actor_loss = -torch.min(surr1, surr2).mean()
    critic_loss = 0.5 * nn.MSELoss()(value.squeeze(-1), returns)
    entropy_loss = -entropy_coeff * dist.entropy().mean()

    behavior_loss = torch.zeros((), device=states.device)
    behavior_mean_loss = torch.zeros((), device=states.device)
    behavior_delta_loss = torch.zeros((), device=states.device)
    if reference_model is not None and (bc_kl_coeff > 0.0 or bc_mean_coeff > 0.0 or bc_delta_coeff > 0.0):
        with torch.no_grad():
            ref_mean, ref_std, _ = reference_model(states)
        ref_dist = Normal(ref_mean, ref_std, validate_args=False)
        behavior_loss = kl_divergence(dist, ref_dist).sum(dim=-1).mean()
        if bc_mean_coeff > 0.0:
            behavior_mean_loss = nn.SmoothL1Loss()(mean, ref_mean)
        if bc_delta_coeff > 0.0:
            current_pose = states[:, CURRENT_POSE_START:CURRENT_POSE_END]
            behavior_delta_loss = reference_delta_direction_loss(mean, ref_mean, current_pose)

    total_loss = (
        actor_loss
        + critic_loss
        + entropy_loss
        + bc_kl_coeff * behavior_loss
        + bc_mean_coeff * behavior_mean_loss
        + bc_delta_coeff * behavior_delta_loss
    )
    return total_loss


# ─────────────────────────────────────────────────────────────────────────────
# CUDA graph rollout capture (safe — no optimizer state involved)
# ─────────────────────────────────────────────────────────────────────────────

def build_cuda_graph_tribe(sim, model, device, start_idx, end_idx):
    N_tribe = end_idx - start_idx

    static_noise  = torch.zeros(N_tribe, ACTION_DIM,  device=device)
    static_state  = sim._state_out[start_idx:end_idx]

    print(f"  [CUDA graph Tribe] Warming up ({start_idx}-{end_idx})...", flush=True)
    sim.get_states()
    for _ in range(3):
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                mean, std, value = model(static_state)
                raw_action = mean + std * static_noise
                sim_action = sanitize_policy_actions(raw_action)
            _ = raw_action.float(), sim_action.float(), value.float()
    torch.cuda.synchronize()

    static_mean   = torch.zeros(N_tribe, ACTION_DIM, device=device)
    static_std    = torch.zeros(N_tribe, ACTION_DIM, device=device)
    static_value  = torch.zeros(N_tribe,  1, device=device)
    static_raw_action = torch.zeros(N_tribe, ACTION_DIM, device=device)
    static_sim_action = torch.zeros(N_tribe, ACTION_DIM, device=device)
    static_logp   = torch.zeros(N_tribe,     device=device)

    print(f"  [CUDA graph Tribe] Capturing...", flush=True)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                _mean, _std, _value = model(static_state)
        # Upcast to float32: FP16 log-probs can overflow; keeping outputs in
        # float32 here costs nothing — the FP16 speedup is in the Linear ops above.
        static_mean.copy_(_mean.float())
        static_std.copy_(_std.float())
        static_value.copy_(_value.float())
        static_raw_action.copy_(static_mean + static_std * static_noise)
        static_sim_action.copy_(sanitize_policy_actions(static_raw_action))
        static_logp.copy_(policy_action_log_prob(static_mean, static_std, static_raw_action))
    
    torch.cuda.synchronize()
    return g, static_mean, static_std, static_value, static_raw_action, static_sim_action, static_logp, static_noise


def build_cuda_graph_sim_step(sim, full_action):
    print("  [CUDA graph Sim] Warming up step graph...", flush=True)
    for _ in range(3):
        sim.step(full_action)
        sim.get_states()
    torch.cuda.synchronize()

    print("  [CUDA graph Sim] Capturing step+state...", flush=True)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        sim.step(full_action)
        sim.get_states()
    torch.cuda.synchronize()
    return g


# Note: Removed @torch.compile here because Triton can hang on Windows.
# The overhead of the Python loop is negligible compared to the GPU sim.
def warm_start_rollout_state(sim, tribes, full_action, start_times, warmup_steps=14, dt=DEFAULT_DT):
    if warmup_steps <= 0:
        sim.reset(start_times)
        return

    warmup_seconds = warmup_steps * dt
    sim.reset((start_times - warmup_seconds).clamp(min=0.0))
    full_action.zero_()

    with torch.no_grad():
        for tribe in tribes:
            tribe.model.eval()

        for _ in range(warmup_steps):
            states = sim.get_states()
            for i, tribe in enumerate(tribes):
                start = i * tribe.num_envs
                end = start + tribe.num_envs
                with torch.amp.autocast('cuda'):
                    mean, _, _ = tribe.model(states[start:end])
                full_action[start:end].copy_(sanitize_policy_actions(mean.float()))
            sim.step(full_action, dt=dt)

    full_action.zero_()


if os.name != "nt" and os.environ.get("BSAI_ENABLE_TORCH_COMPILE", "0") == "1" and hasattr(torch, "compile"):
    warm_start_rollout_state = torch.compile(warm_start_rollout_state, dynamic=False)


def sample_rollout_start_times(
    sim,
    adaptive_state,
    global_best_acc,
    total_envs,
    device,
    start_margin_seconds,
    warmup_steps=14,
    current_task_acc=None,
):
    signal = live_training_signal(
        adaptive_state=adaptive_state,
        current_task_acc=current_task_acc,
        fallback=global_best_acc,
    )
    if signal < 2.0:
        note_fraction = 0.12
    elif signal < 5.0:
        note_fraction = 0.20
    elif signal < 12.0:
        note_fraction = 0.32
    elif signal < 25.0:
        note_fraction = 0.48
    elif signal < 40.0:
        note_fraction = 0.65
    else:
        note_fraction = 0.82

    warmup_seconds = warmup_steps * DEFAULT_DT
    max_start = (sim.map_durations - start_margin_seconds).clamp(min=0.0)
    last_note_beat = torch.gather(sim.note_times, 1, sim._max_note_idx.unsqueeze(1)).squeeze(1)
    last_note_sec = torch.where(
        sim.note_counts > 0,
        last_note_beat / sim.bps.clamp(min=1e-6),
        max_start,
    )
    note_cap = last_note_sec * note_fraction
    start_cap = torch.minimum(
        max_start,
        torch.maximum(note_cap, torch.full_like(max_start, warmup_seconds)),
    )
    start_cap = torch.where(sim.note_counts > 0, start_cap, max_start)

    fallback_bias = torch.rand(total_envs, device=device)
    fallback_bias = fallback_bias * fallback_bias
    fallback_start = fallback_bias * start_cap.clamp(min=0.0)

    note_valid = (
        (sim._note_range.unsqueeze(0) < sim.note_counts.unsqueeze(1))
        & (sim.note_types != 3)
    )
    if not bool(note_valid.any()):
        return fallback_start

    note_seconds = sim.note_times / sim.bps.clamp(min=1e-6).unsqueeze(1)
    early_note_mask = note_valid & (note_seconds <= note_cap.unsqueeze(1) + 1e-6)
    early_note_count = early_note_mask.sum(1)
    target_pool_mask = torch.where(early_note_count.unsqueeze(1) > 0, early_note_mask, note_valid)
    target_pool_count = target_pool_mask.sum(1)
    has_targets = target_pool_count > 0
    if not bool(has_targets.any()):
        return fallback_start

    target_rank = torch.cumsum(target_pool_mask.to(torch.int32), dim=1)
    random_pick = (
        torch.floor(torch.rand(total_envs, device=device) * target_pool_count.clamp(min=1).float()).to(torch.long) + 1
    )
    candidate_mask = target_pool_mask & (target_rank >= random_pick.unsqueeze(1))
    target_idx = torch.argmax(candidate_mask.to(torch.int32), dim=1)
    env_idx = torch.arange(total_envs, device=device)
    target_note_sec = note_seconds[env_idx, target_idx]

    if signal < 2.0:
        lead_seconds = 0.58
    elif signal < 5.0:
        lead_seconds = 0.50
    elif signal < 12.0:
        lead_seconds = 0.42
    elif signal < 25.0:
        lead_seconds = 0.34
    else:
        lead_seconds = 0.26

    lead_jitter = torch.rand(total_envs, device=device) * 0.10
    anchored_start = target_note_sec - (warmup_seconds + lead_seconds + lead_jitter)
    anchored_start = anchored_start.clamp(min=0.0)
    anchored_start = torch.minimum(anchored_start, start_cap.clamp(min=0.0))
    return torch.where(has_targets, anchored_start, fallback_start)


def compute_gae_fast(rewards, values, dones, valid_mask, spans, gamma, lmbda, last_value=None):
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(N, device=rewards.device)
    next_value = last_value if last_value is not None else torch.zeros(N, device=rewards.device)
    gamma_pows = torch.pow(torch.full_like(spans, gamma), spans)
    gae_pows = torch.pow(torch.full_like(spans, gamma * lmbda), spans)
    for t in reversed(range(T)):
        valid_t = valid_mask[t].float()
        next_non_terminal = 1.0 - dones[t].float()
        delta = (
            rewards[t] + gamma_pows[t] * next_value * next_non_terminal - values[t]
        ) * valid_t
        gae = delta + gae_pows[t] * next_non_terminal * gae
        gae = gae * valid_t
        advantages[t] = gae
        next_value = torch.where(valid_mask[t], values[t], next_value)
    return advantages


# ─────────────────────────────────────────────────────────────────────────────
# PBT Tribe Class
# ─────────────────────────────────────────────────────────────────────────────

class Tribe:
    def __init__(self, tribe_id, num_envs, device, base_model_path=None):
        self.id = tribe_id
        self.num_envs = num_envs
        self.device = device
        
        self.model = ActorCritic().to(device)
        if base_model_path and os.path.exists(base_model_path):
            ckpt = torch.load(base_model_path, map_location=device, weights_only=False)
            _remap_state_dict(
                extract_policy_state_dict(
                    ckpt,
                    checkpoint_path=base_model_path,
                    accepted_keys=("model_state_dict", "actor_state_dict"),
                    allow_legacy=True,
                ),
                self.model,
            )
        
        # Separate param groups with proper learning rates
        actor_params  = list(self.model.actor_features.parameters()) + [self.model.actor_log_std] + list(self.model.actor_mean.parameters())
        critic_params = list(self.model.critic_features.parameters()) + list(self.model.critic_head.parameters())
        
        self.optimizer = make_adam([
            {'params': actor_params,  'lr': 3e-4},
            {'params': critic_params, 'lr': 1e-3},
        ])
        self.scaler = torch.amp.GradScaler('cuda')
        
        # Diverse Tribal DNA
        # Tribe 0 starts as the "Bootstrap" tribe with low penalties to encourage learning.
        if tribe_id == 0:
            self.hparams = {
                'lr_actor': 1e-4, 'lr_critic': 3e-4, 'entropy_coeff': 0.02,
                'w_miss': 1.0, 'w_jerk': 0.1, 'w_pos_jerk': 0.1, 'w_reach': 0.1,
                'saber_inertia': 0.55, 'rot_clamp': 0.075, 'action_repeat': 2,
                'bc_kl_coeff': 0.003, 'bc_mean_coeff': 0.16, 'policy_std': 0.08
            }
        else:
            self.hparams = {
                'lr_actor': 1e-4, 'lr_critic': 3e-4, 'entropy_coeff': 0.012,
                'w_miss': 3.0, 'w_jerk': 0.35, 'w_pos_jerk': 0.45, 'w_reach': 0.15,
                'saber_inertia': 0.50, 'rot_clamp': 0.082, 'action_repeat': 2,
                'bc_kl_coeff': 0.002, 'bc_mean_coeff': 0.10, 'policy_std': 0.10
            }
        self.optimizer.param_groups[0]['lr'] = self.hparams['lr_actor']
        self.optimizer.param_groups[1]['lr'] = self.hparams['lr_critic']
        self.current_policy_std = self.hparams['policy_std']
        self.current_action_repeat = int(self.hparams['action_repeat'])
        self.current_saber_inertia = float(self.hparams['saber_inertia'])
        self.current_rot_clamp = float(self.hparams['rot_clamp'])
        self.current_pos_clamp = 0.15
        
        # Performance tracking
        self.fitness = 0.0
        self.selection_score = 0.0
        self.moving_acc = 0.0
        self.last_acc = 0.0
        self.last_task_accuracy = 0.0
        self.last_note_coverage = 0.0
        self.last_rollout_task_accuracy = 0.0
        self.last_rollout_note_coverage = 0.0
        self.last_rollout_notes_seen = 0.0
        self.generation = 0
        self.death_counter = 0      # For instant failure detection
        self.best_fitness = 0.0
        self.best_selection_score = 0.0
        self.stagnation_counter = 0 # Epochs without improvement
        self.diversity_score = 1.0
        
        # Performance logging for stability tracking
        self.performance_log = []  # Track fitness history
        self.stability_score = 1.0  # Measure consistency
        self.exploration_score = 0.0  # Track how much we're exploring
        self.last_energy = 0.5
        self.last_completion = 0.0
        self.last_fail_rate = 0.0
        self.last_clear_rate = 0.0
        self.last_timeout_rate = 0.0
        self.last_combo_ratio = 0.0
        self.last_mean_speed = 0.0
        self.last_style_violation = 0.0
        self.last_angular_violation = 0.0
        self.last_motion_efficiency = 0.0
        self.last_waste_motion = 0.0
        self.last_idle_motion = 0.0
        self.last_guard_error = 0.0
        self.last_oscillation = 0.0
        self.last_lateral_motion = 0.0
        self.leader_cooldown_epochs = 0
        
    def mutate(self):
        """Randomly perturb hyperparameters by +/- 20%, boosted by stagnation."""
        # Adaptive mutation: higher rate if the tribe is stuck
        mutation_factor = 0.2 + (min(6, self.stagnation_counter) * 0.05)
        
        for k in self.hparams:
            if isinstance(self.hparams[k], float):
                # Apply adaptive mutation
                factor = 1.0 + (np.random.rand() * (mutation_factor * 2) - mutation_factor)
                self.hparams[k] *= factor
                
        # Clamp to reasonable ranges
        self.hparams['lr_actor']    = max(1e-5, min(1e-3, self.hparams['lr_actor']))
        self.hparams['lr_critic']   = max(5e-5, min(5e-3, self.hparams['lr_critic']))
        self.hparams['entropy_coeff'] = max(0.0001, min(0.05, self.hparams['entropy_coeff']))
        self.hparams['saber_inertia'] = max(0.35, min(0.85, self.hparams['saber_inertia']))
        self.hparams['rot_clamp']     = max(0.05, min(0.12, self.hparams['rot_clamp']))
        self.hparams['bc_kl_coeff']   = max(0.0002, min(0.02, self.hparams['bc_kl_coeff']))
        self.hparams['bc_mean_coeff'] = max(0.0, min(0.60, self.hparams['bc_mean_coeff']))
        self.hparams['policy_std']    = max(0.035, min(0.20, self.hparams['policy_std']))
        
        # Action repeat mutation: 90% stay, 5% up, 5% down
        r = np.random.rand()
        if r < 0.05: self.hparams['action_repeat'] = max(1, self.hparams['action_repeat'] - 1)
        elif r < 0.10: self.hparams['action_repeat'] = min(3, self.hparams['action_repeat'] + 1)

        # Update optimizer
        self.optimizer.param_groups[0]['lr'] = self.hparams['lr_actor']
        self.optimizer.param_groups[1]['lr'] = self.hparams['lr_critic']
        self.generation += 1

    def copy_from(self, source):
        """Clone weights and hyperparameters from another tribe."""
        self.model.load_state_dict(source.model.state_dict())
        self.optimizer.load_state_dict(source.optimizer.state_dict())
        move_optimizer_state_to_device(self.optimizer, self.device)
        self.hparams = source.hparams.copy()
        self.generation = source.generation
        self.stagnation_counter = 0 # Reset on cloning
        self.best_fitness = source.best_fitness
        self.best_selection_score = source.best_selection_score
        self.death_counter = 0
        self.last_acc = source.last_acc
        self.last_rollout_task_accuracy = source.last_rollout_task_accuracy
        self.last_rollout_note_coverage = source.last_rollout_note_coverage
        self.last_rollout_notes_seen = source.last_rollout_notes_seen
        self.current_policy_std = source.hparams.get('policy_std', 0.10)
        self.current_action_repeat = int(source.hparams.get('action_repeat', 2))
        self.current_saber_inertia = float(source.hparams.get('saber_inertia', 0.35))
        self.current_rot_clamp = float(source.hparams.get('rot_clamp', 0.10))
        self.current_pos_clamp = float(getattr(source, 'current_pos_clamp', 0.15))
        self.performance_log = source.performance_log[-3:]
        self.last_energy = source.last_energy
        self.selection_score = source.selection_score
        self.last_completion = source.last_completion
        self.last_fail_rate = source.last_fail_rate
        self.last_clear_rate = source.last_clear_rate
        self.last_timeout_rate = source.last_timeout_rate
        self.last_combo_ratio = source.last_combo_ratio
        self.last_task_accuracy = source.last_task_accuracy
        self.last_note_coverage = source.last_note_coverage
        self.last_mean_speed = source.last_mean_speed
        self.last_style_violation = source.last_style_violation
        self.last_angular_violation = source.last_angular_violation
        self.last_motion_efficiency = source.last_motion_efficiency
        self.last_waste_motion = source.last_waste_motion
        self.last_idle_motion = source.last_idle_motion
        self.last_guard_error = source.last_guard_error
        self.last_oscillation = source.last_oscillation
        self.last_lateral_motion = source.last_lateral_motion
        self.leader_cooldown_epochs = 0
        self.scaler.load_state_dict(source.scaler.state_dict())
        self.optimizer.param_groups[0]['lr'] = self.hparams['lr_actor']
        self.optimizer.param_groups[1]['lr'] = self.hparams['lr_critic']

    def update_performance(self, current_score):
        """Track history and detect stagnation."""
        self.best_fitness = max(self.best_fitness, self.fitness)
        if not self.performance_log:
            self.best_selection_score = current_score
            self.stagnation_counter = 0
            return
        improvement_margin = max(0.5, abs(self.best_selection_score) * 0.05)
        if current_score > self.best_selection_score + improvement_margin:
            self.best_selection_score = current_score
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

    def calculate_diversity(self, others):
        """Calculate how unique this tribe is based on hyperparameter distance."""
        dists = []
        for o in others:
            if o.id == self.id: continue
            # Simple Euclidean distance in normalized hparam space
            d = 0.0
            for k in ['w_miss', 'w_jerk', 'saber_inertia', 'rot_clamp', 'entropy_coeff']:
                # Roughly normalized ranges
                val_self = self.hparams[k] / 5.0
                val_other = o.hparams[k] / 5.0
                d += (val_self - val_other) ** 2
            dists.append(np.sqrt(d))
        self.diversity_score = sum(dists) / len(dists) if dists else 1.0
        return self.diversity_score

    def log_performance(
        self,
        task_accuracy_signal,
        fitness,
        epoch,
        *,
        proxy_accuracy=0.0,
        task_accuracy=0.0,
        note_coverage=0.0,
        rollout_task_accuracy=None,
        rollout_note_coverage=None,
        rollout_notes_seen=None,
        selection_score=0.0,
        mean_energy=0.0,
        completion=0.0,
        fail_rate=0.0,
        clear_rate=0.0,
        timeout_rate=0.0,
        combo_ratio=0.0,
        mean_speed=0.0,
        style_violation=0.0,
        angular_violation=0.0,
        motion_efficiency=0.0,
        waste_motion=0.0,
        idle_motion=0.0,
        guard_error=0.0,
        oscillation=0.0,
        lateral_motion=0.0,
    ):
        self.last_acc = proxy_accuracy
        self.last_task_accuracy = task_accuracy
        self.last_note_coverage = note_coverage
        self.last_rollout_task_accuracy = float(task_accuracy if rollout_task_accuracy is None else rollout_task_accuracy)
        self.last_rollout_note_coverage = float(note_coverage if rollout_note_coverage is None else rollout_note_coverage)
        self.last_rollout_notes_seen = float(0.0 if rollout_notes_seen is None else rollout_notes_seen)
        self.selection_score = selection_score
        self.performance_log.append({
            'epoch': epoch,
            'accuracy': task_accuracy_signal,
            'task_accuracy_signal': task_accuracy_signal,
            'fitness': fitness,
            'proxy_accuracy': proxy_accuracy,
            'selection_score': selection_score,
            'task_accuracy': task_accuracy,
            'note_coverage': note_coverage,
            'rollout_task_accuracy': self.last_rollout_task_accuracy,
            'rollout_note_coverage': self.last_rollout_note_coverage,
            'rollout_notes_seen': self.last_rollout_notes_seen,
            'energy': mean_energy,
            'completion': completion,
            'fail_rate': fail_rate,
            'clear_rate': clear_rate,
            'timeout_rate': timeout_rate,
            'combo_ratio': combo_ratio,
            'mean_speed': mean_speed,
            'style_violation': style_violation,
            'angular_violation': angular_violation,
            'motion_efficiency': motion_efficiency,
            'waste_motion': waste_motion,
            'idle_motion': idle_motion,
            'guard_error': guard_error,
            'oscillation': oscillation,
            'lateral_motion': lateral_motion,
            'timestamp': time.time()
        })
        self.last_energy = mean_energy
        self.last_completion = completion
        self.last_fail_rate = fail_rate
        self.last_clear_rate = clear_rate
        self.last_timeout_rate = timeout_rate
        self.last_combo_ratio = combo_ratio
        self.last_mean_speed = mean_speed
        self.last_style_violation = style_violation
        self.last_angular_violation = angular_violation
        self.last_motion_efficiency = motion_efficiency
        self.last_waste_motion = waste_motion
        self.last_idle_motion = idle_motion
        self.last_guard_error = guard_error
        self.last_oscillation = oscillation
        self.last_lateral_motion = lateral_motion
        
        # Keep only last 50 entries
        if len(self.performance_log) > 50:
            self.performance_log.pop(0)

    def calculate_stability(self):
        """Estimate stable, recoverable play instead of raw fitness variance."""
        if len(self.performance_log) < 3:
            return 0.45

        recent = self.performance_log[-10:]
        fitnesses = np.array([log['fitness'] for log in recent], dtype=np.float32)
        accuracies = np.array([
            log.get('task_accuracy_signal', log.get('task_accuracy', log.get('accuracy', 0.0)))
            for log in recent
        ], dtype=np.float32)
        energies = np.array([log.get('energy', 0.0) for log in recent], dtype=np.float32)
        completions = np.array([log.get('completion', 0.0) for log in recent], dtype=np.float32)
        fail_rates = np.array([log.get('fail_rate', 0.0) for log in recent], dtype=np.float32)
        clear_rates = np.array([log.get('clear_rate', 0.0) for log in recent], dtype=np.float32)
        combo_ratios = np.array([log.get('combo_ratio', 0.0) for log in recent], dtype=np.float32)
        style_violations = np.array([log.get('style_violation', 0.0) for log in recent], dtype=np.float32)
        motion_efficiencies = np.array([log.get('motion_efficiency', 0.0) for log in recent], dtype=np.float32)
        waste_motion = np.array([log.get('waste_motion', 0.0) for log in recent], dtype=np.float32)
        idle_motion = np.array([log.get('idle_motion', 0.0) for log in recent], dtype=np.float32)
        guard_error = np.array([log.get('guard_error', 0.0) for log in recent], dtype=np.float32)
        oscillation = np.array([log.get('oscillation', 0.0) for log in recent], dtype=np.float32)
        lateral_motion = np.array([log.get('lateral_motion', 0.0) for log in recent], dtype=np.float32)

        fit_mean = float(np.mean(np.abs(fitnesses))) if fitnesses.size else 0.0
        acc_mean = float(np.mean(np.abs(accuracies))) if accuracies.size else 0.0
        fit_cv = float(np.std(fitnesses) / max(25.0, fit_mean))
        acc_cv = float(np.std(accuracies) / max(5.0, acc_mean))
        volatility = max(0.0, 1.0 - min(1.0, 0.70 * fit_cv + 0.55 * acc_cv))

        energy_score = float(np.clip(np.mean(energies), 0.0, 1.0))
        completion_score = float(np.clip(np.mean(completions), 0.0, 1.0))
        fail_resistance = float(np.clip(1.0 - np.mean(fail_rates), 0.0, 1.0))
        clear_score = float(np.clip(np.mean(clear_rates), 0.0, 1.0))
        combo_score = float(np.clip(np.mean(combo_ratios), 0.0, 1.0))
        style_score = float(np.clip(1.0 - np.mean(style_violations), 0.0, 1.0))
        motion_score = float(np.clip(np.mean(motion_efficiencies), 0.0, 1.0))
        waste_score = float(np.clip(1.0 - np.mean(waste_motion) * 8.0, 0.0, 1.0))
        idle_score = float(np.clip(1.0 - np.mean(idle_motion) * 12.0, 0.0, 1.0))
        guard_score = float(np.clip(1.0 - np.mean(guard_error) * 1.8, 0.0, 1.0))
        oscillation_score = float(np.clip(1.0 - np.mean(oscillation) * 4.0, 0.0, 1.0))
        lateral_score = float(np.clip(1.0 - np.mean(lateral_motion) * 3.5, 0.0, 1.0))

        stability = (
            0.22 * volatility
            + 0.20 * energy_score
            + 0.16 * completion_score
            + 0.12 * fail_resistance
            + 0.08 * combo_score
            + 0.05 * clear_score
            + 0.04 * style_score
            + 0.03 * motion_score
            + 0.03 * waste_score
            + 0.03 * idle_score
            + 0.02 * guard_score
            + 0.01 * oscillation_score
            + 0.01 * lateral_score
        )
        return float(min(1.0, max(0.05, stability)))


def apply_policy_std(tribe, global_best_acc, rehab_level, style_rehab_level=0, current_task_acc=None):
    signal = max_nonnegative_signal(current_task_acc)
    std_cap = policy_std_cap(signal if signal > 0.0 else global_best_acc, rehab_level, style_rehab_level)
    target_std = max(0.035, min(std_cap, float(tribe.hparams.get('policy_std', 0.10))))
    rollout_skill = float(max(getattr(tribe, 'last_rollout_task_accuracy', 0.0), tribe.last_task_accuracy))
    if rehab_level >= 3 and rollout_skill < 6.0 and tribe.stagnation_counter >= 6:
        target_std = max(target_std, min(0.11, std_cap + 0.02))
    tribe.model.actor_log_std.data.fill_(float(np.log(target_std)))
    tribe.current_policy_std = target_std
    return target_std

def adaptive_learning_rate(tribe):
    """Adjust learning rates based on performance stability"""
    stability = tribe.calculate_stability()
    rollout_skill = float(max(getattr(tribe, 'last_rollout_task_accuracy', 0.0), tribe.last_task_accuracy))
    rollout_cover = float(max(getattr(tribe, 'last_rollout_note_coverage', 0.0), tribe.last_note_coverage))
    rollout_notes_seen = float(getattr(tribe, 'last_rollout_notes_seen', 0.0))
    
    # Don't decay bootstrap learning rates just because the early policy is bad.
    if stability < 0.32 and tribe.stagnation_counter > 6 and tribe.last_task_accuracy > 10.0:
        tribe.hparams['lr_actor'] *= 0.96
        tribe.hparams['lr_critic'] *= 0.96
        tribe.hparams['policy_std'] = max(0.04, tribe.hparams['policy_std'] * 0.97)
        tribe.hparams['bc_mean_coeff'] = min(1.5, tribe.hparams['bc_mean_coeff'] * 1.03)
        print(f"  Tribe {tribe.id}: Reduced LR after sustained instability ({stability:.2f})")

    # If performance is very stable but still poor, increase exploration gently.
    elif stability > 0.8 and tribe.fitness < 8.0 and tribe.stagnation_counter > 4:
        tribe.hparams['entropy_coeff'] = min(0.1, tribe.hparams['entropy_coeff'] * 1.1)
        print(f"  Tribe {tribe.id}: Increased exploration due to stagnation ({stability:.2f})")
    elif rollout_skill < 6.0 and tribe.stagnation_counter > 8 and (tribe.last_energy < 0.40 or rollout_notes_seen < 8.0):
        # Low-skill runs benefit more from preserving the BC prior than from
        # cranking exploration until the actor forgets how to swing at all.
        tribe.hparams['entropy_coeff'] = min(0.035, max(0.012, tribe.hparams['entropy_coeff'] * 1.04))
        tribe.hparams['policy_std'] = min(0.095, max(0.065, tribe.hparams['policy_std'] * 1.04))
        tribe.hparams['bc_mean_coeff'] = max(0.28, tribe.hparams['bc_mean_coeff'] * 1.02)
        tribe.hparams['bc_kl_coeff'] = max(0.006, tribe.hparams['bc_kl_coeff'] * 1.01)
        tribe.hparams['lr_actor'] = min(1.6e-4, max(5e-5, tribe.hparams['lr_actor']))
        if rollout_cover < 0.65:
            tribe.hparams['entropy_coeff'] = min(0.045, max(0.014, tribe.hparams['entropy_coeff'] * 1.08))
            tribe.hparams['policy_std'] = min(0.105, max(0.072, tribe.hparams['policy_std'] * 1.08))
            tribe.hparams['bc_mean_coeff'] = max(0.24, tribe.hparams['bc_mean_coeff'] * 0.98)
        print(f"  Tribe {tribe.id}: Low-skill re-anchor triggered.")
    elif stability > 0.62 and tribe.last_task_accuracy > 18.0:
        if (
            tribe.last_motion_efficiency >= 0.20
            and tribe.last_idle_motion <= 0.035
            and tribe.last_guard_error <= 0.28
            and tribe.last_oscillation <= 0.05
        ):
            tribe.hparams['policy_std'] = min(0.12, tribe.hparams['policy_std'] * 1.003)
        else:
            tribe.hparams['policy_std'] = max(0.06, tribe.hparams['policy_std'] * 0.97)
    
    # Update optimizer with new rates
    tribe.optimizer.param_groups[0]['lr'] = tribe.hparams['lr_actor']
    tribe.optimizer.param_groups[1]['lr'] = tribe.hparams['lr_critic']

def print_detailed_progress(tribes, epoch):
    """Print more detailed training progress"""
    print(f"\n\033[1m[Epoch {epoch+1} Detailed Report]\033[0m")
    print("+-----+--------+--------+--------+--------+-----------+--------+-------------+")
    print("|Tribe| TaskAcc| Select |Stability| Energy | PolicyStd | Repeat |    Status   |")
    print("+-----+--------+--------+--------+--------+-----------+--------+-------------+")
    
    for tribe in tribes:
        stability = tribe.calculate_stability()
        policy_std = getattr(tribe, 'current_policy_std', tribe.hparams.get('policy_std', 0.10))
        action_repeat = int(getattr(tribe, 'current_action_repeat', tribe.hparams.get('action_repeat', 2)))
        rollout_task_accuracy = getattr(tribe, 'last_rollout_task_accuracy', tribe.last_task_accuracy)
        rollout_note_coverage = getattr(tribe, 'last_rollout_note_coverage', getattr(tribe, 'last_note_coverage', 0.0))
        
        # Status indicator
        status_text = "Stable"
        if tribe.last_fail_rate > 0.70 or tribe.death_counter > 2:
            status_text = "Near Death"
            status = "\033[91mNear Death\033[0m"
        elif stability < 0.25 or tribe.last_energy < 0.20:
            status_text = "Fragile"
            status = "\033[91mFragile\033[0m"
        elif stability < 0.45 or tribe.last_fail_rate > 0.45:
            status_text = "Unstable"
            status = "\033[93mUnstable\033[0m"
        elif (
            tribe.last_task_accuracy >= 35.0
            or (
                tribe.last_task_accuracy >= 20.0
                and getattr(tribe, 'last_note_coverage', 0.0) >= 0.50
                and tribe.last_energy > 0.55
            )
        ):
            status_text = "Excellent"
            status = "\033[92mExcellent\033[0m"
        elif (
            (
                tribe.selection_score > 35.0
                and (rollout_task_accuracy >= 5.0 or rollout_note_coverage >= 0.15)
            )
            or rollout_task_accuracy >= 12.0
            or rollout_note_coverage >= 0.30
        ):
            status_text = "Promising"
            status = "\033[96mPromising\033[0m"
        elif tribe.last_completion > 0.80 and tribe.last_energy > 0.45:
            status_text = "Recovering"
            status = "\033[96mRecovering\033[0m"
        else:
            status = "Stable"
            
        # Manually pad status to avoid ANSI code length issues with string formatting
        padding = " " * (11 - len(status_text))
        print(f"|  {tribe.id}  | {tribe.last_task_accuracy:6.1f}% | {tribe.selection_score:6.1f} | {stability:6.2f} | {tribe.last_energy:6.2f} |   {policy_std:6.3f}  |   x{action_repeat:<2d}  | {padding}{status} |")
    
    print("+-----+--------+--------+--------+--------+-----------+--------+-------------+")


def restore_tribe_from_base(tribe, base_model_path, reason="rehab"):
    """Restore a tribe from the selected bootstrap/base checkpoint instead of random reset."""
    print(f"  REHAB: Restoring Tribe {tribe.id} from base model ({reason})")

    if base_model_path and os.path.exists(base_model_path):
        ckpt = torch.load(base_model_path, map_location=tribe.device, weights_only=False)
        _remap_state_dict(
            extract_policy_state_dict(
                ckpt,
                checkpoint_path=base_model_path,
                accepted_keys=("model_state_dict", "actor_state_dict"),
                allow_legacy=True,
            ),
            tribe.model,
        )
    else:
        print(f"  WARNING: Base model {base_model_path} missing, falling back to fresh reset")
        def reset_weights(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        tribe.model.apply(reset_weights)

    tribe.hparams['entropy_coeff'] = min(0.03, max(0.008, tribe.hparams['entropy_coeff']))
    tribe.hparams['policy_std'] = min(0.10, max(0.06, tribe.hparams['policy_std']))
    tribe.hparams['bc_mean_coeff'] = min(0.14, max(0.03, tribe.hparams['bc_mean_coeff']))
    tribe.hparams['bc_kl_coeff'] = min(0.003, max(0.0005, tribe.hparams['bc_kl_coeff']))
    tribe.hparams['w_miss'] = min(2.5, max(0.8, tribe.hparams.get('w_miss', 2.0)))
    tribe.hparams['w_jerk'] = min(0.30, max(0.08, tribe.hparams.get('w_jerk', 0.20)))
    tribe.hparams['w_pos_jerk'] = min(0.40, max(0.10, tribe.hparams.get('w_pos_jerk', 0.25)))
    tribe.hparams['w_reach'] = min(0.12, max(0.04, tribe.hparams.get('w_reach', 0.08)))
    tribe.hparams['action_repeat'] = 1
    tribe.hparams['saber_inertia'] = min(max(tribe.hparams['saber_inertia'], 0.46), 0.56)
    tribe.hparams['rot_clamp'] = min(tribe.hparams['rot_clamp'], 0.082)

    # Reset optimizer
    actor_params = list(tribe.model.actor_features.parameters()) + [tribe.model.actor_log_std] + list(tribe.model.actor_mean.parameters())
    critic_params = list(tribe.model.critic_features.parameters()) + list(tribe.model.critic_head.parameters())
    
    tribe.optimizer = make_adam([
        {'params': actor_params, 'lr': tribe.hparams['lr_actor']},
        {'params': critic_params, 'lr': tribe.hparams['lr_critic']},
    ])
    tribe.scaler = torch.amp.GradScaler('cuda')  # stale scale state would corrupt the first recovery step
    
    # Reset performance tracking
    tribe.fitness = 0.0
    tribe.selection_score = 0.0
    tribe.moving_acc = 0.0
    tribe.last_acc = 0.0
    tribe.last_task_accuracy = 0.0
    tribe.last_note_coverage = 0.0
    tribe.last_rollout_task_accuracy = 0.0
    tribe.last_rollout_note_coverage = 0.0
    tribe.last_rollout_notes_seen = 0.0
    tribe.best_fitness = 0.0
    tribe.best_selection_score = 0.0
    tribe.death_counter = 0
    tribe.performance_log = []
    tribe.last_energy = 0.5
    tribe.last_completion = 0.0
    tribe.last_fail_rate = 0.0
    tribe.last_clear_rate = 0.0
    tribe.last_timeout_rate = 0.0
    tribe.last_combo_ratio = 0.0
    tribe.last_mean_speed = 0.0
    tribe.last_style_violation = 0.0
    tribe.last_angular_violation = 0.0
    tribe.last_motion_efficiency = 0.0
    tribe.last_waste_motion = 0.0
    tribe.last_idle_motion = 0.0
    tribe.last_guard_error = 0.0
    tribe.last_oscillation = 0.0
    tribe.last_lateral_motion = 0.0
    tribe.leader_cooldown_epochs = 2
    
    tribe.stagnation_counter = 0
    tribe.current_policy_std = tribe.hparams['policy_std']
    tribe.current_action_repeat = int(tribe.hparams['action_repeat'])
    tribe.current_saber_inertia = float(tribe.hparams['saber_inertia'])
    tribe.current_rot_clamp = float(tribe.hparams['rot_clamp'])
    tribe.current_pos_clamp = 0.15
    tribe.optimizer.param_groups[0]['lr'] = tribe.hparams['lr_actor']
    tribe.optimizer.param_groups[1]['lr'] = tribe.hparams['lr_critic']

    print(f"  Tribe {tribe.id} restored from base model.")


def recover_nonfinite_tribe(tribe, base_model_path):
    checkpoint_payload = load_rl_checkpoint_payload()
    if recover_tribe_from_checkpoint_snapshot(tribe, checkpoint_payload) and not tribe_nonfinite_tensors(tribe):
        print(f"  Tribe {tribe.id} recovered from last good checkpoint snapshot.")
        return "checkpoint snapshot"

    restore_tribe_from_base(tribe, base_model_path, reason="non-finite-recovery")
    return "base model"


def train_ppo_gpu(epochs=5000, model_path="bsai_rl_model.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("WARNING: GPU not available!")
        return

    if maybe_patch_inductor_write_atomic():
        print(f"  Enabled PyTorch Inductor write_atomic workaround via {INDUCTOR_WRITE_ATOMIC_PATCH_ENV}=1.")

    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print(f"\033[96m{'='*60}")
    print(f"  CyberNoodles 6.0: PBT Evolutionary Arena")
    print(f"  Device: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}\033[0m")

    # ── PBT Configuration ──────────────────────────────────────────────
    NUM_TRIBES = 4
    TOTAL_ENVS = pick_total_envs(device, NUM_TRIBES)
    ENVS_PER_TRIBE = TOTAL_ENVS // NUM_TRIBES
    STEPS = 512  # Shorter steps for faster genetic turnaround
    BATCH = max(8192, ENVS_PER_TRIBE * 64)
    GEN_INTERVAL = 8
    MIGRATION_INTERVAL = 5
    CHECKPOINT_INTERVAL = 5
    REPLAY_INTERVAL = 40
    REPLAY_MIN_GAP = 15
    MIN_REPLAY_IMPROVEMENT = 7.5
    EVAL_INTERVAL = 100
    skip_bc_baseline_probe = env_flag_enabled(PPO_SKIP_BC_PROBE_ENV, default=False)
    skip_initial_eval = env_flag_enabled(PPO_SKIP_INITIAL_EVAL_ENV, default=False)
    WARMUP_STEPS = 14
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    ROLLOUT_SECONDS = STEPS * DEFAULT_DT
    START_MARGIN_SECONDS = ROLLOUT_SECONDS + 2.0
    REPLAY_ENVS = 64 if TOTAL_ENVS >= 1024 else 32

    trainer_state = load_trainer_state()
    rl_checkpoint = load_rl_checkpoint_payload()
    trainer_state, used_checkpoint_state = sync_trainer_state_with_checkpoint(trainer_state, rl_checkpoint)
    ckpt_epoch = int(rl_checkpoint.get('epoch', 0) or 0)
    ckpt_acc = float(rl_checkpoint.get('moving_acc', 0.0) or 0.0)
    ckpt_trainer_state = rl_checkpoint.get('trainer_state') if isinstance(rl_checkpoint, dict) else {}
    ckpt_strict_eval = strict_task_signal(
        adaptive_state=ckpt_trainer_state.get('adaptive', {}) if isinstance(ckpt_trainer_state, dict) else None,
        trainer_state=ckpt_trainer_state if isinstance(ckpt_trainer_state, dict) else None,
        checkpoint_payload=rl_checkpoint,
    )
    force_bc_bootstrap = os.getenv("BSAI_FORCE_BC_BOOTSTRAP", "0") == "1"
    force_rl_resume = os.getenv("BSAI_FORCE_RL_RESUME", "0") == "1"
    override_base_model = str(os.getenv(PPO_BASE_MODEL_ENV, "") or "").strip()
    has_override_base_model = bool(override_base_model) and os.path.exists(override_base_model)
    awac_signal = load_awac_bootstrap_signal()
    default_bootstrap_path, default_bootstrap_label = choose_default_bootstrap_actor_path(awac_signal)
    if force_bc_bootstrap and not has_override_base_model:
        default_bootstrap_path = BC_MODEL_PATH
        default_bootstrap_label = "forced BC baseline"
    bootstrap_actor_path = override_base_model if has_override_base_model else default_bootstrap_path
    bootstrap_actor_label = "custom base model" if has_override_base_model else default_bootstrap_label
    bc_bootstrap_preferred = should_prefer_bc_bootstrap(trainer_state, rl_checkpoint)
    has_resume_snapshot = checkpoint_has_resume_snapshot(rl_checkpoint, expected_num_tribes=NUM_TRIBES)
    rl_resume_ok = (
        has_resume_snapshot
        and not force_bc_bootstrap
        and (force_rl_resume or not bc_bootstrap_preferred)
    )
    rl_warm_start_ok = (
        bool(rl_checkpoint)
        and not rl_resume_ok
        and not force_bc_bootstrap
        and (force_rl_resume or not bc_bootstrap_preferred)
        and (
            force_rl_resume
            or ckpt_epoch >= 16
            or ckpt_acc >= 2.0
            or ckpt_strict_eval >= 2.0
        )
    )
    base_path = RL_CHECKPOINT_PATH if rl_warm_start_ok else bootstrap_actor_path
    base_label = (
        "RL resume snapshot"
        if rl_resume_ok
        else ("RL checkpoint warm start" if rl_warm_start_ok else bootstrap_actor_label)
    )
    if rl_resume_ok:
        print(f"Loading training state from: {RL_CHECKPOINT_PATH} ({base_label})")
    else:
        print(f"Loading base model from: {base_path} ({base_label})")
    if used_checkpoint_state:
        print(f"  Trainer state synced from newer RL checkpoint at epoch {ckpt_epoch}.")
    elif rl_checkpoint and int(trainer_state.get('epoch', 0) or 0) < ckpt_epoch:
        print(f"  Trainer state epoch was behind the RL checkpoint; synced to epoch {ckpt_epoch}.")
    if rl_checkpoint:
        print(
            f"  RL checkpoint metadata: epoch {ckpt_epoch}, moving_acc {ckpt_acc:.2f}%, "
            f"strict_eval {ckpt_strict_eval:.2f}%."
        )
    if awac_signal.get('actor_path'):
        if awac_signal.get('usable', False):
            print(
                f"  AWAC bootstrap ready: strict {awac_signal['strict_accuracy']:.2f}% | "
                f"cover {awac_signal['strict_coverage']:.3f}."
            )
        else:
            print(
                f"  AWAC bootstrap skipped: strict {awac_signal['strict_accuracy']:.2f}% | "
                f"cover {awac_signal['strict_coverage']:.3f} is still below the PPO handoff floor."
            )
    if bc_bootstrap_preferred and rl_checkpoint and not force_rl_resume:
        print("  Offline bootstrap is preferred because the RL checkpoint is far weaker than the BC probe.")
    if os.path.exists(RL_CHECKPOINT_PATH) and not (rl_resume_ok or rl_warm_start_ok):
        print(
            f"  RL checkpoint exists but bootstrap is starting from {bootstrap_actor_label}. "
            "Set BSAI_FORCE_RL_RESUME=1 to override."
        )
    if force_bc_bootstrap and os.path.exists(RL_CHECKPOINT_PATH):
        print("  Forced BC bootstrap is enabled via BSAI_FORCE_BC_BOOTSTRAP=1.")
    if has_override_base_model and not rl_resume_ok:
        print(f"  PPO base override via {PPO_BASE_MODEL_ENV}: {override_base_model}")

    if rl_resume_ok:
        tribes = load_population_from_checkpoint(rl_checkpoint, NUM_TRIBES, ENVS_PER_TRIBE, device)
        if tribes is None:
            print("  Resume snapshot was incomplete; falling back to RL checkpoint warm start.")
            rl_resume_ok = False
            rl_warm_start_ok = True
            base_path = RL_CHECKPOINT_PATH
            base_label = "RL checkpoint warm start"
    if not rl_resume_ok:
        tribes = [Tribe(i, ENVS_PER_TRIBE, device, base_model_path=base_path) for i in range(NUM_TRIBES)]
    elif restore_rng_state(rl_checkpoint.get('rng_state')):
        print("  Restored RNG state from resume snapshot.")
    else:
        print("  WARNING: Resume snapshot loaded without RNG state; rollout sequence may drift.")

    writer = SummaryWriter(log_dir="runs/cybernoodles_pbt")
    reference_model = load_frozen_reference_model(bootstrap_actor_path, device)
    bc_reference_model = load_frozen_reference_model(BC_MODEL_PATH, device)
    adaptive_state = load_adaptive_state(trainer_state, resume_in_place=rl_resume_ok)
    global_best_fitness = float(trainer_state.get('global_best_fitness', 0.0))
    global_best_selection_score = float(trainer_state.get('global_best_selection_score', -1e9))
    global_best_eval_accuracy = float(trainer_state.get('global_best_eval_accuracy', 0.0))
    global_best_matched_eval_accuracy = float(trainer_state.get('global_best_matched_eval_accuracy', 0.0))
    last_replay_epoch = int(trainer_state.get('last_replay_epoch', 0))
    best_replay_fitness = float(trainer_state.get('best_replay_fitness', 0.0))
    
    # Curriculum maps pool
    curriculum = []
    if os.path.exists('curriculum.json'):
        with open('curriculum.json', 'r', encoding='utf-8') as f:
            curriculum = json.load(f)
    training_curriculum = filter_curriculum_by_split(curriculum, "train", allow_fallback=True)
    if len(training_curriculum) < len(curriculum):
        print(
            f"\033[90mEval split guard: training on {len(training_curriculum)} of "
            f"{len(curriculum)} curriculum maps; held-out maps remain for eval/replay checks.\033[0m"
        )
    curriculum_hashes = [c['hash'] for c in training_curriculum if c.get('hash')]
    replay_backed_hashes = load_replay_backed_hashes()
    ensure_style_calibration(verbose=True)

    sim = make_simulator(num_envs=TOTAL_ENVS, device=device)
    
    print(f"\033[94mCaching maps into RAM...\033[0m")
    map_cache = {}
    for h in curriculum_hashes:
        beatmap, bpm = get_map_data(h)
        notes = beatmap.get('notes', []) if beatmap else []
        if notes and len(notes) > 0:
            map_cache[h] = (beatmap, bpm)
    curriculum_hashes = list(map_cache.keys())
    if not curriculum_hashes:
        raise RuntimeError("No playable maps were loaded into the curriculum cache.")
    curriculum_buckets = build_curriculum_buckets(training_curriculum, map_cache, replay_backed_hashes=replay_backed_hashes)
    eval_curriculum = filter_curriculum_by_split(curriculum, "dev_eval", allow_fallback=True)
    eval_hashes = choose_eval_hashes(
        eval_curriculum,
        max_maps=4,
        map_cache=map_cache,
        suite="starter",
        split="dev_eval",
        exclude_hashes=curriculum_hashes,
    )
    benchmark_hash = trainer_state.get('benchmark_hash')
    preferred_benchmark_hash = eval_hashes[0] if eval_hashes else pick_benchmark_hash(curriculum_buckets)
    if benchmark_hash not in map_cache or benchmark_hash not in set(eval_hashes or [benchmark_hash]):
        benchmark_hash = preferred_benchmark_hash
    benchmark_meta = curriculum_buckets.get('meta', {}).get(benchmark_hash, {})
    benchmark_tag = f"{benchmark_hash[:6]}_nps{benchmark_meta.get('nps', 0.0):.1f}" if benchmark_hash else "benchmark"

    tribe_graphs = [None] * NUM_TRIBES
    sim_step_graph = None
    sim_graph_shape = None
    full_action = torch.zeros(TOTAL_ENVS, ACTION_DIM, device=device)
    all_noise = torch.zeros(STEPS, TOTAL_ENVS, ACTION_DIM, device=device)
    tribe_indices = [
        torch.arange(i * ENVS_PER_TRIBE, (i + 1) * ENVS_PER_TRIBE, device=device)
        for i in range(NUM_TRIBES)
    ]
    current_actions = [torch.zeros(ENVS_PER_TRIBE, ACTION_DIM, device=device) for _ in range(NUM_TRIBES)]
    tribe_rollouts = allocate_rollout_buffers(NUM_TRIBES, STEPS, ENVS_PER_TRIBE, device)
    global_best_acc = float(
        trainer_state.get(
            'global_best_task_accuracy',
            trainer_state.get('global_best_accuracy', max((tribe.last_task_accuracy for tribe in tribes), default=0.0)),
        )
    )

    if should_run_bc_baseline_probe(
        bc_reference_model,
        eval_hashes,
        skip_bc_probe=skip_bc_baseline_probe,
    ):
        print("\033[94mProbing BC baseline on starter eval suite...\033[0m")
        bc_probe_profile = get_eval_profile("bc")
        bc_probe_max_maps = min(1, len(eval_curriculum))
        bc_probe_hashes = choose_eval_hashes(
            eval_curriculum,
            max_maps=bc_probe_max_maps,
            map_cache=map_cache,
            suite="starter",
            split="dev_eval",
            exclude_hashes=curriculum_hashes,
        )
        probe_summary = evaluate_policy_model(
            bc_reference_model,
            device,
            bc_probe_hashes,
            map_cache=map_cache,
            num_envs=min(8, ENVS_PER_TRIBE),
            noise_scale=0.0,
            verbose=True,
            label="bc-probe",
            **bc_probe_profile,
        )
        adaptive_state['bc_probe_accuracy'] = float(probe_summary.get('mean_accuracy', 0.0))
        adaptive_state['bc_probe_engaged_accuracy'] = float(probe_summary.get('mean_engaged_accuracy', 0.0))
        adaptive_state['bc_probe_note_coverage'] = float(probe_summary.get('mean_note_coverage', 0.0))
        adaptive_state['bc_probe_completion'] = float(probe_summary.get('mean_completion', 0.0))
        adaptive_state['bc_probe_clear_rate'] = float(probe_summary.get('mean_clear_rate', 0.0))
        print(
            f"  BC baseline probe: task {adaptive_state['bc_probe_accuracy']:.2f}% | "
            f"engaged {adaptive_state['bc_probe_engaged_accuracy']:.2f}% | "
            f"clear {adaptive_state['bc_probe_clear_rate']:.2f} | "
            f"completion {adaptive_state['bc_probe_completion']:.2f} | "
            f"coverage {adaptive_state['bc_probe_note_coverage']:.2f} "
            f"on {len(probe_summary.get('maps', []))} maps"
        )
    elif bc_reference_model is not None and eval_hashes and skip_bc_baseline_probe:
        print(
            f"\033[90mSkipping BC baseline probe via {PPO_SKIP_BC_PROBE_ENV}=1. "
            f"Stored BC probe metrics will be reused if present.\033[0m"
        )

    if skip_initial_eval and eval_hashes:
        print(
            f"\033[90mSkipping epoch-1 strict/matched eval via {PPO_SKIP_INITIAL_EVAL_ENV}=1. "
            f"Next scheduled eval is epoch {EVAL_INTERVAL}.\033[0m"
        )

    for epoch in range(epochs):
        t0 = time.time()
        live_epoch_signal = live_training_signal(
            adaptive_state=adaptive_state,
            current_task_acc=max((tribe.last_task_accuracy for tribe in tribes), default=0.0),
            fallback=global_best_acc,
        )
        recovery = get_recovery_profile(adaptive_state, global_best_acc, current_task_acc=live_epoch_signal)
        rehab_level = recovery['rehab_level']
        
        if epoch > 0 and epoch % 5 == 0:
            for tribe_idx, tribe in enumerate(tribes):
                bad_tensors = tribe_nonfinite_tensors(tribe)
                if not bad_tensors:
                    continue
                preview = ", ".join(bad_tensors[:4])
                if len(bad_tensors) > 4:
                    preview += ", ..."
                print(f"  CRITICAL: Non-finite state detected in Tribe {tribe.id} at epoch {epoch}: {preview}")
                recovery_source = recover_nonfinite_tribe(tribe, base_path)
                tribe_graphs[tribe_idx] = None
                print(f"  Tribe {tribe.id} recovered using {recovery_source}.")
        
        all_notes, all_bpms, tribe_tiers = sample_maps_for_epoch(
            tribes,
            curriculum_buckets,
            map_cache,
            ENVS_PER_TRIBE,
            global_best_acc,
            rehab_level,
            stagnation_epochs=adaptive_state.get('stagnation_epochs', 0),
            current_task_acc=live_epoch_signal,
        )

        epoch_capacity = choose_epoch_sim_capacity(all_notes)
        sim.load_maps(
            all_notes,
            all_bpms,
            capacity=(
                epoch_capacity['notes_capacity'],
                epoch_capacity['obstacles_capacity'],
            ),
        )
        apply_simulator_tuning(sim, build_recovery_sim_tuning(recovery))
        
        # ── Update Tribe Hyperparams in Simulator ──────────────────
        for i, tribe in enumerate(tribes):
            indices = tribe_indices[i]
            current_std = apply_policy_std(
                tribe,
                global_best_acc,
                rehab_level,
                recovery['style_rehab_level'],
                current_task_acc=live_epoch_signal,
            )
            current_repeat, current_inertia, current_rot_clamp, current_pos_clamp = apply_control_profile(
                tribe,
                global_best_acc,
                rehab_level,
                recovery['stability_rehab_level'],
                recovery['style_rehab_level'],
                current_task_acc=live_epoch_signal,
            )
            sim.set_penalty_weights(
                tribe.hparams['w_miss'] * recovery['miss_scale'],
                tribe.hparams['w_jerk'] * recovery['motion_scale'],
                tribe.hparams['w_pos_jerk'] * recovery['motion_scale'],
                tribe.hparams['w_reach'] * recovery['motion_scale'],
                indices=indices
            )
            sim.set_dense_reward_scale(tribe.hparams.get('approach_scale', recovery['dense_scale']), indices=indices)
            sim.set_saber_inertia(current_inertia, current_rot_clamp, current_pos_clamp, indices=indices)
            writer.add_scalar(f"tribe_{i}/policy_std", current_std, epoch + 1)
            writer.add_scalar(f"tribe_{i}/action_repeat", current_repeat, epoch + 1)
            writer.add_scalar(f"tribe_{i}/saber_inertia", current_inertia, epoch + 1)
            writer.add_scalar(f"tribe_{i}/pos_clamp", current_pos_clamp, epoch + 1)

        start_times = sample_rollout_start_times(
            sim,
            adaptive_state,
            global_best_acc,
            TOTAL_ENVS,
            device,
            START_MARGIN_SECONDS,
            warmup_steps=WARMUP_STEPS,
            current_task_acc=live_epoch_signal,
        )

        current_sim_shape = (sim.max_notes, sim.max_obstacles, TOTAL_ENVS)
        if sim_step_graph is None or sim_graph_shape != current_sim_shape:
            full_action.zero_()
            sim_step_graph = build_cuda_graph_sim_step(sim, full_action)
            sim_graph_shape = current_sim_shape

        if any(g is None for g in tribe_graphs):
            for i in range(NUM_TRIBES):
                if tribe_graphs[i] is not None:
                    continue
                tribes[i].model.eval()
                start, end = i * ENVS_PER_TRIBE, (i + 1) * ENVS_PER_TRIBE
                tribe_graphs[i] = build_cuda_graph_tribe(sim, tribes[i].model, device, start, end)
        
        warm_start_rollout_state(sim, tribes, full_action, start_times, warmup_steps=WARMUP_STEPS)
        sim.get_states()
        all_noise.normal_()
        slot_counts = [0] * NUM_TRIBES
        active_slots = [0] * NUM_TRIBES
        for rollout in tribe_rollouts:
            rollout['rewards'].zero_()
            rollout['dones'].zero_()
            rollout['valid'].zero_()
            rollout['spans'].zero_()
        
        # ── Rollout Loop ──────────────────────────────────────────
        for step in range(STEPS):
            # 1. Action Inference (per tribe graph)
            for i in range(NUM_TRIBES):
                tribe = tribes[i]
                g, g_mean, g_std, g_value, g_raw_action, g_sim_action, g_logp, g_noise = tribe_graphs[i]
                repeat = int(getattr(tribe, 'current_action_repeat', tribe.hparams['action_repeat']))
                start, end = i*ENVS_PER_TRIBE, (i+1)*ENVS_PER_TRIBE
                
                if step % repeat == 0:
                    slot = slot_counts[i]
                    slot_counts[i] += 1
                    active_slots[i] = slot
                    g_noise.copy_(all_noise[step, start:end])
                    g.replay()
                    current_actions[i].copy_(g_sim_action)
                    rollout = tribe_rollouts[i]
                    rollout['states'][slot].copy_(sim._state_out[start:end])
                    rollout['actions'][slot].copy_(g_raw_action)
                    rollout['logprobs'][slot].copy_(g_logp)
                    rollout['values'][slot].copy_(g_value.squeeze(-1))
                    rollout['rewards'][slot].zero_()
                    rollout['dones'][slot].zero_()
                    rollout['valid'][slot].copy_(~sim.episode_done[start:end])
                    rollout['spans'][slot] = float(min(repeat, STEPS - step))

                full_action[start:end].copy_(current_actions[i])
            
            # 2. Step Environment
            sim_step_graph.replay()
            for i in range(NUM_TRIBES):
                repeat = int(getattr(tribes[i], 'current_action_repeat', tribes[i].hparams['action_repeat']))
                slot = active_slots[i]
                start, end = i*ENVS_PER_TRIBE, (i+1)*ENVS_PER_TRIBE
                rollout = tribe_rollouts[i]
                rollout['rewards'][slot].add_(sim._reward_out[start:end] * (GAMMA ** (step % repeat)))
                rollout['dones'][slot] |= sim.episode_done[start:end]

        torch.cuda.synchronize()
        sim_time = time.time() - t0
        decision_transitions = sum(
            int(tribe_rollouts[i]['valid'][:slot_counts[i]].sum().item())
            for i in range(NUM_TRIBES)
        )
        if decision_transitions <= TOTAL_ENVS * 2:
            fail_done = int((sim._terminal_reason == 1).sum().item())
            clear_done = int((sim._terminal_reason == 2).sum().item())
            timeout_done = int((sim._terminal_reason == 3).sum().item())
            print(
                f"  \033[93m[Early termination spike] fail {fail_done} | clear {clear_done} | "
                f"timeout {timeout_done}\033[0m"
            )

        # ── GAE & Update Phase (Per Tribe) ──────────────────────
        print(f"\n\033[1mEpoch {epoch+1} | {TOTAL_ENVS} envs | {decision_transitions:,} decision transitions\033[0m")
        print(f"  Curriculum mix: {' | '.join(f'T{i}:{tier}' for i, tier in enumerate(tribe_tiers))}")
        print(
            f"  Sim capacity: notes {epoch_capacity['notes_capacity']} "
            f"(actual {epoch_capacity['notes_actual']}) | obstacles {epoch_capacity['obstacles_capacity']} "
            f"(actual {epoch_capacity['obstacles_actual']})"
        )
        print(
            f"  Recovery: level {rehab_level} | wheels {recovery['training_wheels']:.2f} | "
            f"assist {recovery['assist_level']:.2f} | dense {recovery['dense_scale']:.2f} | "
            f"stability {recovery['stability_rehab_level']} | style {recovery['style_rehab_level']} | "
            f"fail {recovery['fail_enabled']} | "
            f"stagnation {adaptive_state['stagnation_epochs']}"
        )
        
        if epoch == 0:
            print(f"  \033[90m[One-time cost: Initializing advantage kernels...]\033[0m") # Happens once per run, not like a one time compile that stays

        rollout_target_counts = compute_target_note_counts(
            sim,
            start_times=start_times,
            end_times=sim.current_times,
        ).clamp(min=1.0)
        rollout_completion = compute_completion_ratios(sim, start_times=start_times)

        for i, tribe in enumerate(tribes):
            start, end = i*ENVS_PER_TRIBE, (i+1)*ENVS_PER_TRIBE
            slots_used = slot_counts[i]
            if slots_used == 0:
                continue
            
            with torch.no_grad():
                tribe.model.eval()
                _, _, last_val = tribe.model(sim._state_out[start:end])
                last_value = last_val.squeeze(-1).detach()
                tribe.model.train()

            # Advantages slice
            rollout = tribe_rollouts[i]
            rew = rollout['rewards'][:slots_used]
            val = rollout['values'][:slots_used]
            done = rollout['dones'][:slots_used]
            valid = rollout['valid'][:slots_used]
            spans = rollout['spans'][:slots_used]
            adv = compute_gae_fast(rew, val, done, valid, spans, GAMMA, GAE_LAMBDA, last_value)
            ret = adv + val.detach()
            
            valid_flat = valid.reshape(-1)
            st_t = rollout['states'][:slots_used].reshape(-1, INPUT_DIM)[valid_flat]
            ac_t = rollout['actions'][:slots_used].reshape(-1, ACTION_DIM)[valid_flat]
            lp_t = rollout['logprobs'][:slots_used].reshape(-1).detach()[valid_flat]
            re_t = ret.reshape(-1)[valid_flat]
            ad_t = adv.reshape(-1)[valid_flat]
            if st_t.size(0) == 0:
                continue
            st_t = torch.nan_to_num(st_t, nan=0.0, posinf=0.0, neginf=0.0)
            ac_t = torch.nan_to_num(ac_t, nan=0.0, posinf=0.0, neginf=0.0)
            lp_t = torch.nan_to_num(lp_t, nan=0.0, posinf=0.0, neginf=0.0)
            re_t = torch.nan_to_num(re_t, nan=0.0, posinf=0.0, neginf=0.0)
            ad_t = torch.nan_to_num(ad_t, nan=0.0, posinf=0.0, neginf=0.0)
            ad_t = (ad_t - ad_t.mean()) / (ad_t.std() + 1e-8)
            ad_t = ad_t.clamp(-4.0, 4.0)
            
            # PPO Update
            total_loss = torch.zeros((), device=device)
            ds = st_t.size(0)
            batch_size = min(BATCH, ds)
            n_batches = 0
            strict_anchor_acc = float(
                adaptive_state.get('strict_eval_accuracy', adaptive_state.get('last_eval_accuracy', 0.0)) or 0.0
            )
            bc_kl_coeff, bc_mean_coeff, bc_delta_coeff = effective_reference_coeffs(
                tribe,
                strict_anchor_acc,
                rehab_level,
                recovery['style_rehab_level'],
            )
            optimizer_step_before = optimizer_step_total(tribe.optimizer)
            model_snapshot = parameter_snapshot(tribe.model)
            for ppo_epoch in range(choose_ppo_epochs(epoch, tribe)):
                idx = torch.randperm(ds, device=device)
                for j in range(0, ds, batch_size):
                    b = idx[j:j+batch_size]
                    if b.numel() == 0:
                        continue

                    tribe.optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast('cuda'):
                        loss = compute_ppo_loss(
                            st_t[b], ac_t[b], lp_t[b], re_t[b], ad_t[b], tribe.model,
                            clip_param=0.2,
                            entropy_coeff=tribe.hparams['entropy_coeff'],
                            reference_model=reference_model,
                            bc_kl_coeff=bc_kl_coeff,
                            bc_mean_coeff=bc_mean_coeff,
                            bc_delta_coeff=bc_delta_coeff,
                        )

                    ensure_finite_scalar(f"ppo/tribe_{tribe.id}_loss", loss)
                    tribe.scaler.scale(loss).backward()
                    tribe.scaler.unscale_(tribe.optimizer)
                    assert_finite_gradients(f"ppo/tribe_{tribe.id}_grad", tribe.model)
                    grad_norm = torch.nn.utils.clip_grad_norm_(tribe.model.parameters(), 0.5)
                    ensure_finite_scalar(f"ppo/tribe_{tribe.id}_grad_norm", grad_norm)
                    tribe.scaler.step(tribe.optimizer)
                    tribe.scaler.update()
                    total_loss.add_(loss.detach())
                    n_batches += 1
            if n_batches <= 0:
                raise RuntimeError(f"PPO Tribe {tribe.id} had rollout samples but ran zero optimizer batches.")
            ensure_optimizer_advanced(
                f"PPO Tribe {tribe.id}",
                optimizer_step_before,
                optimizer_step_total(tribe.optimizer),
            )
            ensure_parameter_moved(
                f"PPO Tribe {tribe.id}",
                parameter_delta_l2(tribe.model, model_snapshot),
            )
            assert_finite_module(f"ppo/tribe_{tribe.id}_param", tribe.model)
            
            # ── Stats ──────────────────────────────────────────────
            t_hits = sim.total_hits[start:end].mean().item()
            t_miss = sim.total_misses[start:end].mean().item()
            acc = (t_hits / max(1, t_hits + t_miss)) * 100
            scorable_notes = rollout_target_counts[start:end]
            task_acc = (
                (sim.total_hits[start:end] / scorable_notes)
                .clamp(0.0, 1.0)
                .mean()
                .item()
                * 100.0
            )
            cut = sim.total_cut_scores[start:end].sum().item() / max(1, sim.total_hits[start:end].sum().item())
            mean_energy = sim.energy[start:end].mean().item()
            completion = rollout_completion[start:end].mean().item()
            fail_like = (sim._terminal_reason[start:end] == 1) | (sim.energy[start:end] <= 0.05)
            fail_rate = fail_like.float().mean().item()
            clear_rate = (sim._terminal_reason[start:end] == 2).float().mean().item()
            timeout_rate = (sim._terminal_reason[start:end] == 3).float().mean().item()
            engaged_notes = sim.total_engaged_scorable[start:end]
            resolved_notes = sim.total_resolved_scorable[start:end]
            resolved_denom = resolved_notes.clamp(min=1.0)
            mean_engaged_notes = engaged_notes.mean().item()
            note_coverage = (engaged_notes / scorable_notes).clamp(0.0, 1.0).mean().item()
            resolved_note_coverage = (resolved_notes / scorable_notes).clamp(0.0, 1.0).mean().item()
            rollout_task_acc = task_acc
            rollout_note_coverage = note_coverage
            rollout_resolved_note_coverage = resolved_note_coverage
            combo_ratio = (sim.max_combo[start:end] / resolved_denom).clamp(0.0, 1.0).mean().item()
            speed_denom = sim.speed_samples[start:end].clamp(min=1.0)
            mean_speed = (sim.mean_speed_sum[start:end] / speed_denom).mean().item()
            style_violation = (sim.speed_violation_sum[start:end] / speed_denom).mean().item()
            angular_violation = (sim.angular_violation_sum[start:end] / speed_denom).mean().item()
            waste_motion = (sim.waste_motion_sum[start:end] / speed_denom).mean().item()
            motion_efficiency = (
                sim.useful_progress[start:end] / sim.motion_path[start:end].clamp(min=1e-6)
            ).clamp(0.0, 1.0).mean().item()
            idle_motion = (sim.idle_motion_sum[start:end] / speed_denom).mean().item()
            guard_error = (sim.guard_error_sum[start:end] / speed_denom).mean().item()
            oscillation = (sim.oscillation_sum[start:end] / speed_denom).mean().item()
            lateral_motion = (sim.lateral_motion_sum[start:end] / speed_denom).mean().item()
            tribe.last_acc = acc
            tribe.moving_acc = 0.7 * tribe.moving_acc + 0.3 * task_acc
             
            # ── Adaptive fitness: interception + survival + composure ─────
            engagement = float(np.clip((mean_engaged_notes - 4.0) / 12.0, 0.0, 1.0))
            hit_phase = float(np.clip((task_acc - 1.5) / 8.5, 0.0, 1.0))
            fitness_acc = task_acc * 0.95 + tribe.moving_acc * 0.45 + acc * 0.15
            cut_quality = min(1.0, cut / 115.0) if np.isfinite(cut) else 0.0
            quality_bonus = cut_quality * (2.0 + 6.0 * hit_phase)
            survival_scale = min(1.0, max(engagement, note_coverage)) * (0.20 + 0.80 * hit_phase)
            survival_bonus = survival_scale * (
                completion * 6.0
                + mean_energy * 4.0
                + clear_rate * 8.0
                + combo_ratio * 4.0
            )
            fail_penalty = fail_rate * 12.0 + timeout_rate * 4.0
            early_contact_bonus = note_coverage * (42.0 + 18.0 * (1.0 - hit_phase))
            early_engagement_bonus = engagement * (12.0 + 8.0 * (1.0 - hit_phase))
            low_engagement_penalty = (
                max(0.0, 0.30 - engagement) * 14.0
                + max(0.0, 0.12 - note_coverage) * 55.0
            )
            style_phase = float(np.clip((max(task_acc, tribe.moving_acc) - 10.0) / 20.0, 0.0, 1.0))
            anti_drift_phase = 1.0 - style_phase
            anti_drift_penalty = anti_drift_phase * (
                idle_motion * 24.0
                + guard_error * 2.4
                + oscillation * 10.0
                + lateral_motion * 10.0
            )
            economy_bonus = style_phase * (
                motion_efficiency * 18.0
                + max(0.0, 1.0 - style_violation * 1.25) * 8.0
                + max(0.0, 1.0 - waste_motion * 14.0) * 6.0
                + max(0.0, 1.0 - idle_motion * 20.0) * 5.0
                + max(0.0, 1.0 - guard_error * 2.8) * 4.0
            )
            economy_penalty = style_phase * (
                style_violation * 16.0
                + angular_violation * 1.1
                + waste_motion * 85.0
                + idle_motion * 95.0
                + guard_error * 9.0
                + oscillation * 16.0
                + lateral_motion * 18.0
            )
            tribe.fitness = (
                fitness_acc * 8.0
                + quality_bonus * 6.0
                + survival_bonus
                + early_contact_bonus
                + early_engagement_bonus
                + economy_bonus
                - fail_penalty
                - anti_drift_penalty
                - economy_penalty
                - low_engagement_penalty
            )
            tribe.selection_score = compute_selection_score(
                task_accuracy=rollout_task_acc,
                note_coverage=rollout_note_coverage,
                completion=completion,
                clear_rate=clear_rate,
                fail_rate=fail_rate,
                timeout_rate=timeout_rate,
                cut_quality=cut_quality,
                combo_ratio=combo_ratio,
                recovery=recovery,
            )
            tribe.update_performance(tribe.selection_score) # Track task-grounded stagnation
             
            # ── Death Watch ──────────────────────────────────────────
            if rollout_task_acc < 4.0 and (mean_energy < 0.35 or fail_rate > 0.35 or mean_engaged_notes < 8.0):
                tribe.death_counter += 1
            else:
                tribe.death_counter = 0

            # ── Adaptive DNA & Logging ──────────────────
            tribe.log_performance(
                rollout_task_acc,
                tribe.selection_score,
                epoch,
                proxy_accuracy=acc,
                task_accuracy=task_acc,
                note_coverage=note_coverage,
                rollout_task_accuracy=rollout_task_acc,
                rollout_note_coverage=rollout_note_coverage,
                rollout_notes_seen=mean_engaged_notes,
                selection_score=tribe.selection_score,
                mean_energy=mean_energy,
                completion=completion,
                fail_rate=fail_rate,
                clear_rate=clear_rate,
                timeout_rate=timeout_rate,
                combo_ratio=combo_ratio,
                mean_speed=mean_speed,
                style_violation=style_violation,
                angular_violation=angular_violation,
                motion_efficiency=motion_efficiency,
                waste_motion=waste_motion,
                idle_motion=idle_motion,
                guard_error=guard_error,
                oscillation=oscillation,
                lateral_motion=lateral_motion,
            )
            adaptive_learning_rate(tribe)
             
            # ── Emergency Reset Check ──────────────────
            if tribe.last_rollout_task_accuracy < 1.0 and tribe.death_counter > 5:
                restore_tribe_from_base(tribe, base_path, reason="death-watch")
             
            step_num = epoch + 1
            writer.add_scalar(f"tribe_{i}/task_accuracy", task_acc, step_num)
            writer.add_scalar(f"tribe_{i}/proxy_accuracy", acc, step_num)
            writer.add_scalar(f"tribe_{i}/selection_score", tribe.selection_score, step_num)
            writer.add_scalar(f"tribe_{i}/fitness", tribe.fitness, step_num)
            writer.add_scalar(f"tribe_{i}/lr", tribe.optimizer.param_groups[0]['lr'], step_num)
            writer.add_scalar(f"tribe_{i}/entropy", tribe.hparams['entropy_coeff'], step_num)
            writer.add_scalar(f"tribe_{i}/stability", tribe.calculate_stability(), step_num)
            writer.add_scalar(f"tribe_{i}/energy", mean_energy, step_num)
            writer.add_scalar(f"tribe_{i}/completion", completion, step_num)
            writer.add_scalar(f"tribe_{i}/fail_rate", fail_rate, step_num)
            writer.add_scalar(f"tribe_{i}/clear_rate", clear_rate, step_num)
            writer.add_scalar(f"tribe_{i}/combo_ratio", combo_ratio, step_num)
            writer.add_scalar(f"tribe_{i}/notes_seen", mean_engaged_notes, step_num)
            writer.add_scalar(f"tribe_{i}/note_coverage", note_coverage, step_num)
            writer.add_scalar(f"tribe_{i}/resolved_note_coverage", resolved_note_coverage, step_num)
            writer.add_scalar(f"tribe_{i}/rollout_task_accuracy", rollout_task_acc, step_num)
            writer.add_scalar(f"tribe_{i}/rollout_note_coverage", rollout_note_coverage, step_num)
            writer.add_scalar(f"tribe_{i}/rollout_resolved_note_coverage", rollout_resolved_note_coverage, step_num)
            writer.add_scalar(f"tribe_{i}/engagement", engagement, step_num)
            writer.add_scalar(f"tribe_{i}/mean_speed", mean_speed, step_num)
            writer.add_scalar(f"tribe_{i}/style_violation", style_violation, step_num)
            writer.add_scalar(f"tribe_{i}/angular_violation", angular_violation, step_num)
            writer.add_scalar(f"tribe_{i}/motion_efficiency", motion_efficiency, step_num)
            writer.add_scalar(f"tribe_{i}/waste_motion", waste_motion, step_num)
            writer.add_scalar(f"tribe_{i}/idle_motion", idle_motion, step_num)
            writer.add_scalar(f"tribe_{i}/guard_error", guard_error, step_num)
            writer.add_scalar(f"tribe_{i}/oscillation", oscillation, step_num)
            writer.add_scalar(f"tribe_{i}/lateral_motion", lateral_motion, step_num)
            writer.add_scalar(f"tribe_{i}/bc_kl", bc_kl_coeff, step_num)
            writer.add_scalar(f"tribe_{i}/bc_mean", bc_mean_coeff, step_num)
            writer.add_scalar(f"tribe_{i}/bc_delta", bc_delta_coeff, step_num)
            writer.add_scalar(f"tribe_{i}/loss", (total_loss / max(1, n_batches)).item(), step_num)
            writer.add_scalar(f"tribe_{i}/samples", ds, step_num)

        # ── Detailed Progress Dashboard ──────────────────────────────
        print_detailed_progress(tribes, epoch)
        if (epoch + 1) % GEN_INTERVAL == 0:
            print(f"\033[93m\n[Genetics] Generation {epoch // GEN_INTERVAL + 1} Evolution...\033[0m")
            
            # IMMEDIATE FIX 1: Implement Tribe Resurrection
            for i, tribe in enumerate(tribes):
                if tribe.selection_score < 1.0 and tribe.death_counter > 3:
                    restore_tribe_from_base(tribe, base_path, reason="tribe-resurrection")
                    tribe_graphs[i] = None # Reset graphs for mutant
            
            population_summary = summarize_tribe_population(tribes)
            ranked_tribes = population_summary.get('leaderboard') or population_summary['ranked']
            genetics_signal = max(
                ranked_tribes[0].last_task_accuracy if ranked_tribes else 0.0,
                population_summary['core_mean_acc'],
            )

            if ranked_tribes:
                print(
                    f"  \033[92mLeader: Tribe {ranked_tribes[0].id} "
                    f"(Sel {ranked_tribes[0].selection_score:.1f}, Task {ranked_tribes[0].last_task_accuracy:.1f}%)\033[0m"
                )

            replacement_plans = choose_tribe_replacements(population_summary, genetics_signal)
            low_skill_lock = rehab_level >= 2 and ranked_tribes and ranked_tribes[0].last_task_accuracy < 5.0
            if low_skill_lock:
                replacement_plans = []
                print("  Genetics paused: leader is still below 5% task accuracy, preserving live RL progress.")
                anchor_refreshed_id = None
                if (
                    len(ranked_tribes) >= 1
                    and adaptive_state.get('stagnation_epochs', 0) >= 16
                    and adaptive_state.get('bc_probe_accuracy', 0.0) >= 15.0
                ):
                    anchor = ranked_tribes[-1]
                    restore_tribe_from_base(anchor, bootstrap_actor_path, reason="bootstrap-reanchor")
                    anchor.hparams['entropy_coeff'] = min(0.015, max(0.008, anchor.hparams['entropy_coeff']))
                    anchor.hparams['policy_std'] = min(0.070, max(0.050, anchor.hparams['policy_std']))
                    anchor.hparams['bc_mean_coeff'] = max(anchor.hparams['bc_mean_coeff'], 0.75)
                    anchor.hparams['bc_kl_coeff'] = max(anchor.hparams['bc_kl_coeff'], 0.012)
                    anchor.current_policy_std = anchor.hparams['policy_std']
                    tribe_graphs[anchor.id] = None
                    anchor_refreshed_id = anchor.id
                    print(
                        f"  BC anchor refresh for Tribe {anchor.id}: "
                        f"std={anchor.hparams['policy_std']:.3f}, "
                        f"bc_mean={anchor.hparams['bc_mean_coeff']:.3f}"
                    )
                scout_candidates = [t for t in ranked_tribes if t.id != anchor_refreshed_id]
                if len(scout_candidates) >= 1:
                    scout = scout_candidates[-1]
                    scout.stagnation_counter = max(scout.stagnation_counter, 10)
                    scout.mutate()
                    scout.hparams['entropy_coeff'] = min(0.030, max(0.012, scout.hparams['entropy_coeff'] * 1.10))
                    scout.hparams['policy_std'] = min(0.095, max(0.075, scout.hparams['policy_std']))
                    scout.hparams['bc_mean_coeff'] = max(0.25, scout.hparams['bc_mean_coeff'] * 0.92)
                    scout.hparams['bc_kl_coeff'] = max(0.006, scout.hparams['bc_kl_coeff'])
                    scout.hparams['action_repeat'] = 1
                    scout.current_action_repeat = 1
                    scout.current_policy_std = scout.hparams['policy_std']
                    tribe_graphs[scout.id] = None
                    print(
                        f"  Scout mutation for Tribe {scout.id}: "
                        f"ent={scout.hparams['entropy_coeff']:.4f}, "
                        f"std={scout.hparams['policy_std']:.3f}, "
                        f"bc_mean={scout.hparams['bc_mean_coeff']:.3f}"
                    )

            for donor, target in replacement_plans:
                print(
                    f"  \033[91mTribe {target.id} (Sel {target.selection_score:.1f}, Task {target.last_task_accuracy:.1f}%) -> "
                    f"reseeded from Tribe {donor.id}\033[0m"
                )

                target.copy_from(donor)
                target.mutate()
                print(
                    f"  \033[94mNew DNA for Tribe {target.id}: "
                    f"ent={target.hparams['entropy_coeff']:.4f}, "
                    f"lr={target.hparams['lr_actor']:.1e}\033[0m"
                )

                tribe_graphs[target.id] = None

        population_summary = summarize_tribe_population(tribes)
        best_candidates = population_summary.get('leaderboard') or population_summary['ranked']
        best_tribe = best_candidates[0]
        current_best_acc = max(t.last_task_accuracy for t in tribes)
        if (epoch + 1) % MIGRATION_INTERVAL == 0 and np.random.random() < 0.3:
            genetics_signal = max(current_best_acc, population_summary['core_mean_acc'])
            if genetics_signal >= 8.0:
                i, j = np.random.choice(len(tribes), 2, replace=False)
                tribes[i].hparams, tribes[j].hparams = tribes[j].hparams.copy(), tribes[i].hparams.copy()
                tribes[i].optimizer.param_groups[0]['lr'] = tribes[i].hparams['lr_actor']
                tribes[i].optimizer.param_groups[1]['lr'] = tribes[i].hparams['lr_critic']
                tribes[j].optimizer.param_groups[0]['lr'] = tribes[j].hparams['lr_actor']
                tribes[j].optimizer.param_groups[1]['lr'] = tribes[j].hparams['lr_critic']
                tribes[i].current_policy_std = tribes[i].hparams.get('policy_std', tribes[i].current_policy_std)
                tribes[j].current_policy_std = tribes[j].hparams.get('policy_std', tribes[j].current_policy_std)
                tribes[i].current_action_repeat = int(tribes[i].hparams.get('action_repeat', tribes[i].current_action_repeat))
                tribes[j].current_action_repeat = int(tribes[j].hparams.get('action_repeat', tribes[j].current_action_repeat))
                tribes[i].current_saber_inertia = float(tribes[i].hparams.get('saber_inertia', tribes[i].current_saber_inertia))
                tribes[j].current_saber_inertia = float(tribes[j].hparams.get('saber_inertia', tribes[j].current_saber_inertia))
                tribes[i].current_rot_clamp = float(tribes[i].hparams.get('rot_clamp', tribes[i].current_rot_clamp))
                tribes[j].current_rot_clamp = float(tribes[j].hparams.get('rot_clamp', tribes[j].current_rot_clamp))
                print(f"  Migrated DNA between tribes {i} and {j}")
                tribe_graphs[i] = None
                tribe_graphs[j] = None

        global_best_acc = max(global_best_acc, current_best_acc)
        mean_acc = population_summary['mean_acc']
        mean_proxy_acc = population_summary['mean_proxy_acc']
        mean_selection_score = population_summary['mean_selection_score']
        mean_note_coverage = population_summary['mean_note_coverage']
        mean_stability = population_summary['mean_stability']
        mean_energy = population_summary['mean_energy']
        mean_completion = population_summary['mean_completion']
        mean_fail_rate = population_summary['mean_fail_rate']
        mean_speed = population_summary['mean_speed']
        mean_style_violation = population_summary['mean_style_violation']
        mean_angular_violation = population_summary['mean_angular_violation']
        mean_motion_efficiency = population_summary['mean_motion_efficiency']
        mean_waste_motion = population_summary['mean_waste_motion']
        mean_idle_motion = population_summary['mean_idle_motion']
        mean_guard_error = population_summary['mean_guard_error']
        mean_oscillation = population_summary['mean_oscillation']
        mean_lateral_motion = population_summary['mean_lateral_motion']
        core_mean_acc = population_summary['core_mean_acc']
        core_mean_proxy_acc = population_summary['core_mean_proxy_acc']
        core_mean_selection_score = population_summary['core_mean_selection_score']
        core_mean_note_coverage = population_summary['core_mean_note_coverage']
        core_mean_stability = population_summary['core_mean_stability']
        core_mean_energy = population_summary['core_mean_energy']
        core_mean_completion = population_summary['core_mean_completion']
        core_mean_fail_rate = population_summary['core_mean_fail_rate']
        core_mean_speed = population_summary['core_mean_speed']
        core_mean_style_violation = population_summary['core_mean_style_violation']
        core_mean_angular_violation = population_summary['core_mean_angular_violation']
        core_mean_motion_efficiency = population_summary['core_mean_motion_efficiency']
        core_mean_waste_motion = population_summary['core_mean_waste_motion']
        core_mean_idle_motion = population_summary['core_mean_idle_motion']
        core_mean_guard_error = population_summary['core_mean_guard_error']
        core_mean_oscillation = population_summary['core_mean_oscillation']
        core_mean_lateral_motion = population_summary['core_mean_lateral_motion']
        epoch_time = time.time() - t0
        strict_eval_acc = None
        matched_eval_acc = None
        strict_eval_completion = None
        strict_eval_clear_rate = None
        strict_eval_note_coverage = None
        strict_eval_resolved_acc = None
        matched_eval_completion = None
        matched_eval_clear_rate = None
        matched_eval_note_coverage = None
        matched_eval_resolved_acc = None

        if should_run_epoch_eval_probe(
            epoch,
            EVAL_INTERVAL,
            eval_hashes,
            skip_initial_eval=skip_initial_eval,
        ):
            was_training = best_tribe.model.training
            best_tribe.model.eval()
            eval_signal = live_training_signal(
                adaptive_state=adaptive_state,
                current_task_acc=current_best_acc,
                fallback=global_best_acc,
            )
            if eval_signal < 5.0:
                eval_num_envs = min(24, ENVS_PER_TRIBE)
                eval_map_count = 2
                eval_suite = "starter"
            elif eval_signal < 12.0:
                eval_num_envs = min(32, ENVS_PER_TRIBE)
                eval_map_count = 3
                eval_suite = "starter"
            else:
                eval_num_envs = min(64, ENVS_PER_TRIBE)
                eval_map_count = 4
                eval_suite = "standard"
            runtime_eval_hashes = choose_eval_hashes(
                eval_curriculum,
                max_maps=eval_map_count,
                map_cache=map_cache,
                suite=eval_suite,
                split="dev_eval",
                exclude_hashes=curriculum_hashes,
            )
            print(
                f"  Eval probe running on {len(runtime_eval_hashes)} {eval_suite} map(s) "
                f"with {eval_num_envs} envs..."
            )
            strict_eval_profile = get_eval_profile("strict")
            strict_eval_summary = evaluate_policy_model(
                best_tribe.model,
                device,
                runtime_eval_hashes,
                map_cache=map_cache,
                num_envs=eval_num_envs,
                noise_scale=0.0,
                label="strict",
                verbose=True,
                **strict_eval_profile,
            )
            strict_eval_acc = float(strict_eval_summary.get('mean_accuracy', 0.0))
            strict_eval_resolved_acc = float(strict_eval_summary.get('mean_resolved_accuracy', 0.0))
            strict_eval_completion = float(strict_eval_summary.get('mean_completion', 0.0))
            strict_eval_clear_rate = float(strict_eval_summary.get('mean_clear_rate', 0.0))
            strict_eval_note_coverage = float(strict_eval_summary.get('mean_note_coverage', 0.0))
            matched_eval_profile = build_training_matched_eval_profile(best_tribe, recovery)
            matched_eval_summary = evaluate_policy_model(
                best_tribe.model,
                device,
                runtime_eval_hashes,
                map_cache=map_cache,
                num_envs=eval_num_envs,
                noise_scale=0.0,
                label="matched",
                verbose=True,
                **matched_eval_profile,
            )
            matched_eval_acc = float(matched_eval_summary.get('mean_accuracy', 0.0))
            matched_eval_resolved_acc = float(matched_eval_summary.get('mean_resolved_accuracy', 0.0))
            matched_eval_completion = float(matched_eval_summary.get('mean_completion', 0.0))
            matched_eval_clear_rate = float(matched_eval_summary.get('mean_clear_rate', 0.0))
            matched_eval_note_coverage = float(matched_eval_summary.get('mean_note_coverage', 0.0))
            writer.add_scalar("global/eval_accuracy", strict_eval_acc, epoch + 1)
            writer.add_scalar("global/eval_avg_cut", strict_eval_summary.get('mean_cut', 0.0), epoch + 1)
            writer.add_scalar("global/eval_accuracy_strict", strict_eval_acc, epoch + 1)
            writer.add_scalar("global/eval_accuracy_resolved_strict", strict_eval_resolved_acc, epoch + 1)
            writer.add_scalar("global/eval_completion_strict", strict_eval_completion, epoch + 1)
            writer.add_scalar("global/eval_clear_rate_strict", strict_eval_clear_rate, epoch + 1)
            writer.add_scalar("global/eval_note_coverage_strict", strict_eval_note_coverage, epoch + 1)
            writer.add_scalar("global/eval_accuracy_matched", matched_eval_acc, epoch + 1)
            writer.add_scalar("global/eval_accuracy_resolved_matched", matched_eval_resolved_acc, epoch + 1)
            writer.add_scalar("global/eval_avg_cut_matched", matched_eval_summary.get('mean_cut', 0.0), epoch + 1)
            writer.add_scalar("global/eval_completion_matched", matched_eval_completion, epoch + 1)
            writer.add_scalar("global/eval_clear_rate_matched", matched_eval_clear_rate, epoch + 1)
            writer.add_scalar("global/eval_note_coverage_matched", matched_eval_note_coverage, epoch + 1)
            writer.add_scalar("global/eval_accuracy_gap", matched_eval_acc - strict_eval_acc, epoch + 1)
            print(
                f"  \033[96mEval probe: strict {strict_eval_acc:.2f}% "
                f"(clear {strict_eval_clear_rate:.2f}, comp {strict_eval_completion:.2f}, "
                f"cover {strict_eval_note_coverage:.2f}, engaged {strict_eval_resolved_acc:.2f}%) | "
                f"matched {matched_eval_acc:.2f}% "
                f"(clear {matched_eval_clear_rate:.2f}, comp {matched_eval_completion:.2f}, "
                f"cover {matched_eval_note_coverage:.2f}, engaged {matched_eval_resolved_acc:.2f}%) on "
                f"{len(strict_eval_summary.get('maps', []))} map(s).\033[0m"
            )
            if was_training:
                best_tribe.model.train()

        adaptive_state = update_adaptive_state(
            adaptive_state,
            epoch + 1,
            global_best_acc,
            strict_eval_acc,
            matched_eval_acc,
            current_task_acc=current_best_acc,
            mean_stability=core_mean_stability,
            mean_note_coverage=core_mean_note_coverage,
            mean_energy=core_mean_energy,
            mean_completion=core_mean_completion,
            mean_fail_rate=core_mean_fail_rate,
            mean_speed=core_mean_speed,
            mean_style_violation=core_mean_style_violation,
            mean_angular_violation=core_mean_angular_violation,
            mean_motion_efficiency=core_mean_motion_efficiency,
            mean_waste_motion=core_mean_waste_motion,
            mean_idle_motion=core_mean_idle_motion,
            mean_guard_error=core_mean_guard_error,
            mean_oscillation=core_mean_oscillation,
            mean_lateral_motion=core_mean_lateral_motion,
        )
        writer.add_scalar("global/best_fitness", best_tribe.fitness, epoch + 1)
        writer.add_scalar("global/best_selection_score", best_tribe.selection_score, epoch + 1)
        writer.add_scalar("global/best_accuracy", global_best_acc, epoch + 1)
        writer.add_scalar("global/best_proxy_accuracy", max(t.moving_acc for t in tribes), epoch + 1)
        writer.add_scalar("global/mean_accuracy", mean_acc, epoch + 1)
        writer.add_scalar("global/mean_proxy_accuracy", mean_proxy_acc, epoch + 1)
        writer.add_scalar("global/mean_selection_score", mean_selection_score, epoch + 1)
        writer.add_scalar("global/mean_note_coverage", mean_note_coverage, epoch + 1)
        writer.add_scalar("global/mean_stability", mean_stability, epoch + 1)
        writer.add_scalar("global/mean_energy", mean_energy, epoch + 1)
        writer.add_scalar("global/mean_completion", mean_completion, epoch + 1)
        writer.add_scalar("global/mean_fail_rate", mean_fail_rate, epoch + 1)
        writer.add_scalar("global/mean_speed", mean_speed, epoch + 1)
        writer.add_scalar("global/mean_style_violation", mean_style_violation, epoch + 1)
        writer.add_scalar("global/mean_angular_violation", mean_angular_violation, epoch + 1)
        writer.add_scalar("global/mean_motion_efficiency", mean_motion_efficiency, epoch + 1)
        writer.add_scalar("global/mean_waste_motion", mean_waste_motion, epoch + 1)
        writer.add_scalar("global/mean_idle_motion", mean_idle_motion, epoch + 1)
        writer.add_scalar("global/mean_guard_error", mean_guard_error, epoch + 1)
        writer.add_scalar("global/mean_oscillation", mean_oscillation, epoch + 1)
        writer.add_scalar("global/mean_lateral_motion", mean_lateral_motion, epoch + 1)
        writer.add_scalar("global/core_mean_accuracy", core_mean_acc, epoch + 1)
        writer.add_scalar("global/core_mean_proxy_accuracy", core_mean_proxy_acc, epoch + 1)
        writer.add_scalar("global/core_mean_selection_score", core_mean_selection_score, epoch + 1)
        writer.add_scalar("global/core_mean_note_coverage", core_mean_note_coverage, epoch + 1)
        writer.add_scalar("global/core_mean_stability", core_mean_stability, epoch + 1)
        writer.add_scalar("global/core_mean_energy", core_mean_energy, epoch + 1)
        writer.add_scalar("global/core_mean_completion", core_mean_completion, epoch + 1)
        writer.add_scalar("global/core_mean_fail_rate", core_mean_fail_rate, epoch + 1)
        writer.add_scalar("global/core_mean_motion_efficiency", core_mean_motion_efficiency, epoch + 1)
        writer.add_scalar("global/core_mean_idle_motion", core_mean_idle_motion, epoch + 1)
        writer.add_scalar("global/core_mean_guard_error", core_mean_guard_error, epoch + 1)
        writer.add_scalar("global/sim_time", sim_time, epoch + 1)
        writer.add_scalar("global/epoch_time", epoch_time, epoch + 1)
        writer.add_scalar("global/env_steps_per_sec", (STEPS * TOTAL_ENVS) / max(sim_time, 1e-6), epoch + 1)
        writer.add_scalar("global/decision_transitions", decision_transitions, epoch + 1)
        writer.add_scalar("global/rehab_level", adaptive_state['rehab_level'], epoch + 1)
        writer.add_scalar("global/stability_rehab_level", adaptive_state['stability_rehab_level'], epoch + 1)
        writer.add_scalar("global/style_rehab_level", adaptive_state['style_rehab_level'], epoch + 1)
        writer.add_scalar("global/stagnation_epochs", adaptive_state['stagnation_epochs'], epoch + 1)
        writer.add_scalar("global/stability_stagnation_epochs", adaptive_state['stability_stagnation_epochs'], epoch + 1)
        writer.add_scalar("global/rehab_release_streak", adaptive_state.get('rehab_release_streak', 0), epoch + 1)
        writer.add_scalar("global/escape_support_active", int(bool(adaptive_state.get('escape_support_active', False))), epoch + 1)

        trainer_state = {
            'epoch': epoch + 1,
            'moving_acc': best_tribe.moving_acc,
            'task_accuracy': best_tribe.last_task_accuracy,
            'selection_score': best_tribe.selection_score,
            'global_best_accuracy': global_best_acc,
            'global_best_task_accuracy': global_best_acc,
            'global_best_proxy_accuracy': max(t.moving_acc for t in tribes),
            'global_best_selection_score': global_best_selection_score,
            'global_mean_accuracy': mean_acc,
            'global_mean_proxy_accuracy': mean_proxy_acc,
            'global_mean_selection_score': mean_selection_score,
            'global_mean_note_coverage': mean_note_coverage,
            'global_mean_stability': mean_stability,
            'global_mean_energy': mean_energy,
            'global_mean_completion': mean_completion,
            'global_mean_fail_rate': mean_fail_rate,
            'global_mean_speed': mean_speed,
            'global_mean_style_violation': mean_style_violation,
            'global_mean_angular_violation': mean_angular_violation,
            'global_mean_motion_efficiency': mean_motion_efficiency,
            'global_mean_waste_motion': mean_waste_motion,
            'global_mean_idle_motion': mean_idle_motion,
            'global_mean_guard_error': mean_guard_error,
            'global_mean_oscillation': mean_oscillation,
            'global_mean_lateral_motion': mean_lateral_motion,
            'global_core_mean_accuracy': core_mean_acc,
            'global_core_mean_proxy_accuracy': core_mean_proxy_acc,
            'global_core_mean_selection_score': core_mean_selection_score,
            'global_core_mean_note_coverage': core_mean_note_coverage,
            'global_core_mean_stability': core_mean_stability,
            'global_core_mean_energy': core_mean_energy,
            'global_core_mean_completion': core_mean_completion,
            'global_core_mean_fail_rate': core_mean_fail_rate,
            'global_core_mean_motion_efficiency': core_mean_motion_efficiency,
            'global_core_mean_idle_motion': core_mean_idle_motion,
            'global_core_mean_guard_error': core_mean_guard_error,
            'global_best_fitness': global_best_fitness,
            'global_best_eval_accuracy': global_best_eval_accuracy,
            'global_best_matched_eval_accuracy': global_best_matched_eval_accuracy,
            'last_strict_eval_accuracy': strict_eval_acc,
            'last_strict_eval_resolved_accuracy': strict_eval_resolved_acc,
            'last_strict_eval_completion': strict_eval_completion,
            'last_strict_eval_clear_rate': strict_eval_clear_rate,
            'last_strict_eval_note_coverage': strict_eval_note_coverage,
            'last_matched_eval_accuracy': matched_eval_acc,
            'last_matched_eval_resolved_accuracy': matched_eval_resolved_acc,
            'last_matched_eval_completion': matched_eval_completion,
            'last_matched_eval_clear_rate': matched_eval_clear_rate,
            'last_matched_eval_note_coverage': matched_eval_note_coverage,
            'last_replay_epoch': last_replay_epoch,
            'best_replay_fitness': best_replay_fitness,
            'benchmark_hash': benchmark_hash,
            'benchmark_tag': benchmark_tag,
            'adaptive': adaptive_state,
        }

        eval_improved = strict_eval_acc is not None and strict_eval_acc > global_best_eval_accuracy + 0.25
        matched_eval_improved = matched_eval_acc is not None and matched_eval_acc > global_best_matched_eval_accuracy + 0.25
        if eval_improved:
            global_best_eval_accuracy = strict_eval_acc
            trainer_state['global_best_eval_accuracy'] = global_best_eval_accuracy
        if matched_eval_improved:
            global_best_matched_eval_accuracy = matched_eval_acc
            trainer_state['global_best_matched_eval_accuracy'] = global_best_matched_eval_accuracy

        selection_candidate_is_real = (
            (
                best_tribe.last_note_coverage >= 0.08
                and best_tribe.last_task_accuracy >= 1.0
            )
            or (
                best_tribe.last_note_coverage >= 0.05
                and best_tribe.last_completion >= 0.15
            )
            or best_tribe.last_clear_rate >= 0.02
            or (strict_eval_acc is not None and strict_eval_acc >= 1.0)
        )
        raw_selection_improved = best_tribe.selection_score > global_best_selection_score + 0.25
        selection_improved = raw_selection_improved and selection_candidate_is_real
        if best_tribe.fitness > global_best_fitness:
            global_best_fitness = best_tribe.fitness
            trainer_state['global_best_fitness'] = global_best_fitness
        if selection_improved:
            global_best_selection_score = best_tribe.selection_score
            trainer_state['global_best_selection_score'] = global_best_selection_score
        elif raw_selection_improved and not selection_candidate_is_real:
            print("  \033[90mSelection checkpoint skipped: top tribe has no real play signal yet.\033[0m")

        if selection_improved or eval_improved:
            global_best_fitness = best_tribe.fitness
            trainer_state['global_best_fitness'] = global_best_fitness
            save_training_artifacts(best_tribe, tribes, model_path, epoch + 1, trainer_state, promote_actor=True)
            if eval_improved:
                print(f"  \033[92mNew strict eval best saved from Tribe {best_tribe.id} ({strict_eval_acc:.2f}% eval acc).\033[0m")
            else:
                print(
                    f"  \033[92mNew selection best saved from Tribe {best_tribe.id} "
                    f"(score {best_tribe.selection_score:.2f}, task {best_tribe.last_task_accuracy:.2f}%).\033[0m"
                )
        elif (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            save_training_artifacts(best_tribe, tribes, model_path, epoch + 1, trainer_state, promote_actor=False)
            print(f"  \033[90mPeriodic resume checkpoint saved from Tribe {best_tribe.id}; actor model not promoted.\033[0m")

        writer.flush()

        replay_gap = (epoch + 1) - last_replay_epoch
        interval_due = replay_gap >= REPLAY_INTERVAL
        improvement_due = replay_gap >= REPLAY_MIN_GAP and best_tribe.selection_score >= best_replay_fitness + MIN_REPLAY_IMPROVEMENT
        if benchmark_hash and (interval_due or improvement_due):
            was_training = best_tribe.model.training
            best_tribe.model.eval()
            try:
                generate_progress_replay(
                    best_tribe.model,
                    device,
                    epoch + 1,
                    map_hash=benchmark_hash,
                    tag=benchmark_tag,
                    num_envs=REPLAY_ENVS,
                    fetch_remote=False,
                )
                last_replay_epoch = epoch + 1
                best_replay_fitness = max(best_replay_fitness, best_tribe.selection_score)
            except Exception as exc:
                print(f"  \033[91m[Progress Replay] Failed (non-fatal): {exc}\033[0m")
            finally:
                if was_training:
                    best_tribe.model.train()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        trainer_state['last_replay_epoch'] = last_replay_epoch
        trainer_state['best_replay_fitness'] = best_replay_fitness
        trainer_state['global_best_fitness'] = global_best_fitness
        with open(TRAINER_STATE_PATH, 'w', encoding='utf-8') as f:
            json.dump(trainer_state, f, indent=2)

        print(
            f"  \033[90mEpoch time: {epoch_time:.2f}s | Sim: {sim_time:.2f}s | "
            f"Best tribe: {best_tribe.id} | Best sel: {best_tribe.selection_score:.2f} | "
            f"Best task: {global_best_acc:.2f}% | Current top task: {current_best_acc:.2f}% | "
            f"Proxy: {max(t.moving_acc for t in tribes):.2f}% | "
            f"Stability: {mean_stability:.2f} | "
            f"StyleEff: {mean_motion_efficiency:.2f} | "
            f"Rehab: {adaptive_state['rehab_level']}/{adaptive_state['stability_rehab_level']}/{adaptive_state['style_rehab_level']}\033[0m"
        )
        print(
            f"  \033[90mStyle diag: idle {mean_idle_motion:.3f} | guard {mean_guard_error:.3f} | "
            f"osc {mean_oscillation:.3f} | lateral {mean_lateral_motion:.3f} | "
            f"cover {mean_note_coverage:.3f}\033[0m"
        )
        for tribe in tribes:
            tribe.leader_cooldown_epochs = max(0, int(getattr(tribe, 'leader_cooldown_epochs', 0)) - 1)

if __name__ == "__main__":
    try:
        train_ppo_gpu()
    except Exception as e:
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...") # You really should keep training the model, it yearns for more...
