import argparse
import json

import torch

from cybernoodles.paths import existing_or_preferred_model_path
from cybernoodles.training.eval_splits import SPLIT_NAMES
from cybernoodles.training.policy_eval import (
    choose_eval_hashes,
    evaluate_policy_model,
    get_eval_profile,
    load_actor_critic,
    load_curriculum,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a BC/RL policy directly in the simulator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Performance sanity example:\n"
            "  python -m cybernoodles.tools.diagnose_policy --profile bc --max-maps 1 --num-envs 1\n"
            "  python -m cybernoodles.tools.diagnose_policy --profile bc --max-maps 1 --num-envs 1 --cuda-graph\n"
        ),
    )
    parser.add_argument("--model", default=existing_or_preferred_model_path("bc_model"), help="Path to model or checkpoint file.")
    parser.add_argument("--num-envs", type=int, default=64, help="Parallel envs per evaluated map.")
    parser.add_argument("--noise-scale", type=float, default=0.0, help="Extra stochasticity multiplier on policy std.")
    parser.add_argument("--profile", default="all", choices=["all", "strict", "bc", "rehab"], help="Evaluation profile.")
    parser.add_argument("--suite", default="starter", choices=["starter", "standard", "mixed"], help="Curriculum suite to benchmark when --maps is omitted.")
    parser.add_argument("--split", default="dev_eval", choices=SPLIT_NAMES, help="Evaluation split to use when --maps is omitted.")
    parser.add_argument("--maps", type=str, default="", help="Comma-separated map hashes to test.")
    parser.add_argument("--max-maps", type=int, default=4, help="How many curriculum maps to test if --maps is omitted.")
    parser.add_argument("--json-out", default="", help="Optional JSON file for the evaluation summary.")
    parser.add_argument("--verbose", action="store_true", help="Print per-map simulator progress while evaluating.")
    parser.add_argument("--cuda-graph", action="store_true", help="Opt into CUDA graph capture for supported deterministic eval rollouts.")
    parser.add_argument("--require-cuda-graph", action="store_true", help="Fail instead of falling back if CUDA graph eval is unsupported.")
    parser.add_argument(
        "--cuda-graph-done-check-interval-frames",
        type=int,
        default=0,
        help="Optional graph-mode done check interval. 0 runs each map for its full frame budget.",
    )
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_actor_critic(args.model, device)

    if args.maps.strip():
        map_hashes = [part.strip() for part in args.maps.split(",") if part.strip()]
    else:
        curriculum = load_curriculum()
        map_hashes = choose_eval_hashes(
            curriculum,
            max_maps=args.max_maps,
            suite=args.suite,
            split=args.split,
        )

    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Maps: {', '.join(map_hashes)}")
    if not args.maps.strip():
        print(f"Map source: {args.split} / {args.suite} benchmark suite")
    print(f"Noise scale: {args.noise_scale}")
    print(f"CUDA graph: {'required' if args.require_cuda_graph else ('requested' if args.cuda_graph else 'off')}")
    print()

    profiles = ["strict", "bc", "rehab"] if args.profile == "all" else [args.profile]
    json_summary = {
        "model": args.model,
        "device": str(device),
        "maps": map_hashes,
        "suite": args.suite,
        "split": args.split if not args.maps.strip() else None,
        "profiles": {},
    }

    for profile_name in profiles:
        profile = get_eval_profile(profile_name)
        summary = evaluate_policy_model(
            model,
            device,
            map_hashes,
            num_envs=args.num_envs,
            noise_scale=args.noise_scale,
            verbose=args.verbose,
            label=profile_name,
            use_cuda_graph=args.cuda_graph or args.require_cuda_graph,
            require_cuda_graph=args.require_cuda_graph,
            cuda_graph_done_check_interval_frames=args.cuda_graph_done_check_interval_frames,
            **profile,
        )
        json_summary["profiles"][profile_name] = summary

        print(f"[{profile_name.upper()}]")
        print(
            f"  action_repeat={profile['action_repeat']} "
            f"smoothing={profile['smoothing_alpha']:.2f} "
            f"wheels={profile['training_wheels_level']:.2f} "
            f"assist={profile['assist_level']:.2f} "
            f"survival={profile['survival_assistance']:.2f} "
            f"fail={profile['fail_enabled']} "
            f"inertia={profile['saber_inertia']:.2f}"
        )
        if summary.get("cuda_graph_requested"):
            graph_status = "used" if summary.get("cuda_graph_used") else "fallback"
            print(f"  cuda_graph={graph_status}")
        for item in summary["maps"]:
            print(
                f"  {item['map_hash'][:8]} | "
                f"nps {item['nps']:.2f} | "
                f"notes {item['total_notes']} | "
                f"obstacles {item['obstacle_count']} | "
                f"task {item['accuracy']:.2f}% | "
                f"engaged {item['engaged_accuracy']:.2f}% | "
                f"clear {item['clear_rate']:.2f} | comp {item['completion']:.2f} | "
                f"cover {item['note_coverage']:.2f} | "
                f"resolved {item.get('resolved_coverage', item['note_coverage']):.2f} | "
                f"hits {item['hits']:.1f} | misses {item['misses']:.1f} | "
                f"avg_cut {item['avg_cut']:.1f}"
            )
        print(f"  Mean task accuracy:     {summary['mean_accuracy']:.2f}%")
        print(f"  Mean engaged accuracy:  {summary['mean_engaged_accuracy']:.2f}%")
        print(f"  Mean clear rate:        {summary['mean_clear_rate']:.2f}")
        print(f"  Mean completion:        {summary['mean_completion']:.2f}")
        print(f"  Mean note coverage:     {summary['mean_note_coverage']:.2f}")
        print(f"  Mean resolved coverage: {summary.get('mean_resolved_coverage', summary['mean_note_coverage']):.2f}")
        print(f"  Mean obstacle ratio:    {summary['mean_obstacle_ratio']:.2f}")
        print(f"  Mean avg cut:           {summary['mean_cut']:.1f}")
        print(f"  Mean motion efficiency: {summary.get('mean_motion_efficiency', 0.0):.3f}")
        print(f"  Mean flail index:       {summary.get('mean_flail_index', 0.0):.3f}")
        print()

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(json_summary, f, indent=2)
        print(f"Saved JSON summary to {args.json_out}")


if __name__ == "__main__":
    main()
