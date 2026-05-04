from dataclasses import asdict, dataclass
from typing import Dict, Tuple


DEFAULT_BC_PROBE_MAPS = 3


@dataclass
class SimulatorTuning:
    penalty_weights: Tuple[float, float, float, float] = (0.5, 0.0, 0.0, 0.0)
    dense_reward_scale: float = 0.0
    training_wheels: float = 0.0
    rehab_assists: float = 0.0
    survival_assistance: float = 0.0
    stability_assistance: float = 0.0
    style_guidance_level: float = 0.0
    hit_timing_profile: str = "default"
    fail_enabled: bool = True
    saber_inertia: float = 0.0
    rot_clamp: float = 0.07
    pos_clamp: float = 0.12
    score_only_mode: bool = False
    external_pose_passthrough: bool = False

    def to_env_kwargs(self) -> Dict[str, object]:
        return asdict(self)


def apply_simulator_tuning(sim, tuning: SimulatorTuning, indices=None):
    sim.set_penalty_weights(*tuple(tuning.penalty_weights), indices=indices)
    sim.set_dense_reward_scale(float(tuning.dense_reward_scale), indices=indices)
    sim.set_training_wheels(float(tuning.training_wheels))
    sim.set_rehab_assists(float(tuning.rehab_assists), indices=indices)
    sim.set_hit_timing_profile(
        tuning.hit_timing_profile,
        assist_level=float(tuning.rehab_assists),
        indices=indices,
    )
    sim.set_survival_assistance(float(tuning.survival_assistance), indices=indices)
    sim.set_stability_assistance(float(tuning.stability_assistance), indices=indices)
    sim.set_style_guidance(float(tuning.style_guidance_level), indices=indices)
    sim.set_fail_enabled(bool(tuning.fail_enabled), indices=indices)
    sim.set_saber_inertia(float(tuning.saber_inertia), float(tuning.rot_clamp), float(tuning.pos_clamp), indices=indices)
    sim.set_score_only_mode(bool(tuning.score_only_mode))
    sim.set_external_pose_passthrough(bool(tuning.external_pose_passthrough))
    return sim


def get_eval_profile(profile):
    profile_key = str(profile or "strict").strip().lower()
    profiles = {
        "strict": {
            "action_repeat": 1,
            "smoothing_alpha": 1.0,
            "sim_tuning": SimulatorTuning(
                training_wheels=0.0,
                rehab_assists=0.0,
                survival_assistance=0.0,
                stability_assistance=0.0,
                style_guidance_level=0.0,
                hit_timing_profile="strict",
                fail_enabled=True,
                saber_inertia=0.0,
                rot_clamp=0.07,
                pos_clamp=0.12,
            ),
        },
        "bc": {
            "action_repeat": 1,
            "smoothing_alpha": 1.0,
            "sim_tuning": SimulatorTuning(
                training_wheels=0.65,
                rehab_assists=0.65,
                survival_assistance=0.65,
                stability_assistance=0.0,
                style_guidance_level=0.0,
                hit_timing_profile="assisted",
                fail_enabled=False,
                saber_inertia=0.0,
                rot_clamp=0.07,
                pos_clamp=0.12,
            ),
        },
        "rehab": {
            "action_repeat": 1,
            "smoothing_alpha": 1.0,
            "sim_tuning": SimulatorTuning(
                training_wheels=1.0,
                rehab_assists=1.0,
                survival_assistance=1.0,
                stability_assistance=0.0,
                style_guidance_level=0.0,
                hit_timing_profile="assisted",
                fail_enabled=False,
                saber_inertia=0.0,
                rot_clamp=0.07,
                pos_clamp=0.12,
            ),
        },
    }
    if profile_key not in profiles:
        raise ValueError(f"Unknown evaluation profile: {profile}")
    selected = profiles[profile_key]
    return {
        "action_repeat": int(selected["action_repeat"]),
        "smoothing_alpha": float(selected["smoothing_alpha"]),
        "sim_tuning": SimulatorTuning(**selected["sim_tuning"].to_env_kwargs()),
    }


def build_awac_training_tuning(args) -> SimulatorTuning:
    assist_level = float(args.assist_level)
    return SimulatorTuning(
        penalty_weights=(
            float(args.w_miss),
            float(args.w_jerk),
            float(args.w_pos_jerk),
            float(args.w_reach),
        ),
        dense_reward_scale=float(args.dense_reward_scale),
        training_wheels=float(args.training_wheels_level),
        rehab_assists=assist_level,
        survival_assistance=float(args.survival_assistance),
        stability_assistance=float(args.stability_reward_level),
        style_guidance_level=float(args.style_guidance_level),
        hit_timing_profile="assisted" if assist_level > 0.0 else "default",
        fail_enabled=bool(args.fail_enabled),
        saber_inertia=float(args.saber_inertia),
        rot_clamp=float(args.rot_clamp),
        pos_clamp=float(args.pos_clamp),
    )
