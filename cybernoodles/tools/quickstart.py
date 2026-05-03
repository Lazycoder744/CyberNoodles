import argparse
import subprocess
import sys

from cybernoodles.paths import existing_or_preferred_model_path


def _run_step(label, command):
    print(f"\n=== {label} ===")
    print(" ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="One-click bootstrap pipeline for new CyberNoodles users.")
    parser.add_argument("--player-id", action="append", default=[], help="BeatLeader or Steam player ID. Repeat for multiple players.")
    parser.add_argument("--top-n", type=int, default=5000, help="Global top replay cap for the fetch stage.")
    parser.add_argument("--min-accuracy", type=float, default=0.85, help="Minimum replay accuracy to keep during fetch.")
    parser.add_argument("--per-player-limit", type=int, default=None, help="Optional per-player cap before global selection.")
    parser.add_argument("--max-pages-per-player", type=int, default=None, help="Optional BeatLeader safety cap for paging.")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetching maps and replays.")
    parser.add_argument("--skip-sanity", action="store_true", help="Skip replay parser audit and simulator calibration.")
    parser.add_argument("--skip-bc", action="store_true", help="Skip BC pretraining.")
    parser.add_argument("--skip-awac", action="store_true", help="Skip AWAC fine-tuning.")
    args = parser.parse_args()

    python = sys.executable

    if not args.skip_fetch:
        if not args.player_id:
            raise RuntimeError("At least one --player-id is required unless --skip-fetch is used.")
        fetch_cmd = [python, "-m", "cybernoodles.data.fetch_data"]
        for player_id in args.player_id:
            fetch_cmd.extend(["--player-id", player_id])
        fetch_cmd.extend(["--top-n", str(args.top_n), "--min-accuracy", str(args.min_accuracy)])
        if args.per_player_limit is not None:
            fetch_cmd.extend(["--per-player-limit", str(args.per_player_limit)])
        if args.max_pages_per_player is not None:
            fetch_cmd.extend(["--max-pages-per-player", str(args.max_pages_per_player)])
        _run_step("Fetch replay corpus", fetch_cmd)
        _run_step("Analyze map curriculum", [python, "-m", "cybernoodles.data.map_analyzer"])

    if not args.skip_sanity:
        _run_step("Audit replay parser coverage", [python, "-m", "cybernoodles.tools.audit_replays"])
        _run_step("Calibrate simulator geometry", [python, "-m", "cybernoodles.data.sim_calibration"])

    _run_step("Build training dataset", [python, "-m", "cybernoodles.data.dataset_builder"])

    if not args.skip_bc:
        _run_step("Train BC baseline", [python, "-m", "cybernoodles.training.train_bc"])
        _run_step(
            "Evaluate BC strict closed-loop play",
            [
                python,
                "-m",
                "cybernoodles.tools.diagnose_policy",
                "--model",
                existing_or_preferred_model_path("bc_model"),
                "--profile",
                "strict",
                "--max-maps",
                "3",
                "--json-out",
                "data/bc_strict_eval_quickstart.json",
            ],
        )

    if not args.skip_awac:
        _run_step("Start AWAC bootstrap training", [python, "-m", "cybernoodles.training.train_awac"])


if __name__ == "__main__":
    main()
