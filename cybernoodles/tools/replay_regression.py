import argparse
import json
import os
import time

import torch

from cybernoodles.tools.score_replay import resolve_replay_path, score_replay

SELECTED_SCORES_PATH = os.path.join("data", "selected_scores.json")


def _fmt_percent(value):
    return "n/a" if value is None else f"{float(value):.2f}%"


def load_selected_manifest():
    if not os.path.exists(SELECTED_SCORES_PATH):
        raise FileNotFoundError(f"Missing selection manifest: {SELECTED_SCORES_PATH}")
    with open(SELECTED_SCORES_PATH, "r", encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def choose_replays(manifest, top_n, player_name=None):
    selected = list(manifest.get("selected", []))
    if player_name:
        wanted = str(player_name).strip().lower()
        selected = [item for item in selected if str(item.get("player_name", "")).strip().lower() == wanted]
    return selected[: max(0, int(top_n))]


def main():
    parser = argparse.ArgumentParser(description="Run a small replay regression suite against the simulator.")
    parser.add_argument("--top-n", type=int, default=5, help="Use the first N replays from data/selected_scores.json.")
    parser.add_argument("--player-name", default=None, help="Optional player-name filter inside data/selected_scores.json.")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--fail", action="store_true", help="Use fail-fast sim evaluation.")
    parser.add_argument("--frame-stride", type=int, default=4, help="Only feed every Nth replay frame to the simulator.")
    parser.add_argument("--max-seconds", type=float, default=20.0, help="Cap analyzed replay seconds for quick regression.")
    parser.add_argument("--max-abs-oracle-sim-gap", type=float, default=None, help="Fail when |oracle accuracy - simulator accuracy| exceeds this percentage.")
    parser.add_argument("--min-simulator-accuracy", type=float, default=None, help="Fail when simulator accuracy falls below this percentage.")
    parser.add_argument("--min-oracle-accuracy", type=float, default=None, help="Fail when oracle accuracy falls below this percentage.")
    parser.add_argument("--json-out", default=os.path.join("data", "replay_regression.json"))
    args = parser.parse_args()

    manifest = load_selected_manifest()
    selected = choose_replays(manifest, top_n=args.top_n, player_name=args.player_name)
    if not selected:
        raise RuntimeError("No selected replays matched the requested filters.")

    results = []
    violations = []
    started = time.perf_counter()
    print(
        f"Running replay regression on {len(selected)} replay(s) "
        f"| fail={args.fail} stride={args.frame_stride} max_seconds={args.max_seconds}"
    )

    for index, item in enumerate(selected, start=1):
        replay_id = str(item.get("id"))
        replay_path = resolve_replay_path(replay_id)
        print(f"[{index}/{len(selected)}] {item.get('player_name')} | {item.get('song_name')} | replay {replay_id}")
        summary = score_replay(
            replay_path,
            args.device,
            fail_enabled=args.fail,
            track_events=False,
            frame_stride=args.frame_stride,
            max_seconds=args.max_seconds,
        )
        summary["selected_manifest_entry"] = {
            "id": replay_id,
            "player_name": item.get("player_name"),
            "song_name": item.get("song_name"),
            "difficulty": item.get("difficulty"),
            "stars": item.get("stars"),
            "accuracy": item.get("accuracy"),
        }
        oracle_acc = summary.get("oracle_accuracy_percent")
        sim_acc = summary.get("simulator_accuracy_percent")
        gap = None
        if oracle_acc is not None and sim_acc is not None:
            gap = oracle_acc - sim_acc
        summary["oracle_sim_accuracy_gap"] = gap
        item_violations = []
        if args.max_abs_oracle_sim_gap is not None and gap is not None:
            if abs(float(gap)) > float(args.max_abs_oracle_sim_gap):
                item_violations.append(
                    f"oracle/simulator gap {float(gap):.2f}% exceeds {float(args.max_abs_oracle_sim_gap):.2f}%"
                )
        if args.min_simulator_accuracy is not None and sim_acc is not None:
            if float(sim_acc) < float(args.min_simulator_accuracy):
                item_violations.append(
                    f"simulator accuracy {float(sim_acc):.2f}% below {float(args.min_simulator_accuracy):.2f}%"
                )
        if args.min_oracle_accuracy is not None and oracle_acc is not None:
            if float(oracle_acc) < float(args.min_oracle_accuracy):
                item_violations.append(
                    f"oracle accuracy {float(oracle_acc):.2f}% below {float(args.min_oracle_accuracy):.2f}%"
                )
        summary["violations"] = item_violations
        violations.extend({"replay_id": replay_id, "message": message} for message in item_violations)
        results.append(summary)
        print(
            f"  oracle {_fmt_percent(oracle_acc)} | sim {_fmt_percent(sim_acc)} | gap {_fmt_percent(gap)} | "
            f"hits {summary['hits']} | bad {summary['bad_cuts']} | misses {summary['misses']}"
        )

    elapsed = time.perf_counter() - started
    payload = {
        "count": len(results),
        "elapsed_seconds": elapsed,
        "config": {
            "top_n": args.top_n,
            "player_name": args.player_name,
            "device": args.device,
            "fail": args.fail,
            "frame_stride": args.frame_stride,
            "max_seconds": args.max_seconds,
            "max_abs_oracle_sim_gap": args.max_abs_oracle_sim_gap,
            "min_simulator_accuracy": args.min_simulator_accuracy,
            "min_oracle_accuracy": args.min_oracle_accuracy,
        },
        "results": results,
        "violations": violations,
    }

    with open(args.json_out, "w", encoding="utf-8") as out_file:
        json.dump(payload, out_file, indent=2)

    print(f"Saved regression summary to {args.json_out}")
    print(f"Elapsed: {elapsed:.1f}s")
    if violations:
        print(f"Replay regression failed with {len(violations)} violation(s):")
        for violation in violations[:10]:
            print(f"  {violation['replay_id']}: {violation['message']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
