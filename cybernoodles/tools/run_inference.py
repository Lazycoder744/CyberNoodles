import argparse

from cybernoodles.paths import existing_or_preferred_model_path
from cybernoodles.replay.generate_replay import generate_bsor


def main():
    parser = argparse.ArgumentParser(description="Run standalone CyberNoodles model inference and export a BSOR replay.")
    parser.add_argument("map_id", help="BeatSaver BSR code or full map hash.")
    parser.add_argument("--model", default=existing_or_preferred_model_path("rl_model"), help="Path to the released model or checkpoint.")
    parser.add_argument("--output", default="generated_AI_play.bsor", help="Output BSOR filename.")
    parser.add_argument("--diff-index", type=int, default=-1, help="Difficulty index to use. Defaults to the hardest map difficulty.")
    parser.add_argument("--num-envs", type=int, default=None, help="Parallel replay candidates to simulate.")
    parser.add_argument("--noise-scale", type=float, default=0.0, help="Extra stochasticity multiplier on policy std.")
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=1.0,
        help="Retention factor for replay action smoothing. 1.0 disables smoothing; lower values retain more of the previous action.",
    )
    parser.add_argument("--fail", action="store_true", help="Enable fail conditions instead of generating a full showcase replay.")
    parser.add_argument("--assist-level", type=float, default=0.0, help="Replay assist level.")
    parser.add_argument("--survival-level", type=float, default=0.0, help="Replay survival assistance level.")
    parser.add_argument("--training-wheels", type=float, default=0.0, help="Replay hitbox assistance level.")
    args = parser.parse_args()

    generate_bsor(
        args.map_id,
        output_file=args.output,
        diff_index=args.diff_index,
        model_path=args.model,
        num_envs=args.num_envs,
        noise_scale=args.noise_scale,
        smoothing_alpha=args.smoothing_alpha,
        fail_enabled=args.fail,
        assist_level=args.assist_level,
        survival_level=args.survival_level,
        training_wheels=args.training_wheels,
    )


if __name__ == "__main__":
    main()
