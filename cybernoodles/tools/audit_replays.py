import argparse
import json
import os
from collections import Counter

from cybernoodles.bsor_bridge import audit_replays_via_rust, bsor_tools_available
from cybernoodles.data.dataset_builder import REPLAYS_DIR, parse_bsor


def audit_replay(path, check_mode):
    result = {
        "path": os.path.abspath(path),
        "raw_bsor_ok": False,
        "dataset_parse_ok": False,
        "error": None,
    }
    try:
        from cybernoodles.bsor_bridge import load_bsor

        load_bsor(path)
        result["raw_bsor_ok"] = True
    except Exception as exc:
        result["error"] = f"bsor:{type(exc).__name__}: {exc}"
        if check_mode == "raw":
            return result

    if check_mode in {"dataset", "both"}:
        try:
            frames, meta = parse_bsor(path)
            result["dataset_parse_ok"] = bool(frames) and bool(meta)
            if not result["dataset_parse_ok"] and result["error"] is None:
                result["error"] = "dataset: parse_bsor returned no frames/meta"
        except Exception as exc:
            if result["error"] is None:
                result["error"] = f"dataset:{type(exc).__name__}: {exc}"
    return result


def summarize_results(results):
    raw_ok = sum(1 for item in results if item["raw_bsor_ok"])
    dataset_ok = sum(1 for item in results if item["dataset_parse_ok"])
    failures = [item for item in results if item["error"]]
    by_error = Counter(item["error"] for item in failures if item["error"])
    return {
        "count": len(results),
        "raw_bsor_ok": raw_ok,
        "dataset_parse_ok": dataset_ok,
        "failure_count": len(failures),
        "top_errors": [
            {"error": error, "count": count}
            for error, count in by_error.most_common(10)
        ],
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit local BSOR replay parser coverage before calibration or training.")
    parser.add_argument("--replay-dir", default=REPLAYS_DIR, help="Directory containing .bsor files.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on how many replays to scan.")
    parser.add_argument("--check", choices=["raw", "dataset", "both"], default="both", help="Which parser path to validate.")
    parser.add_argument("--backend", choices=["auto", "python", "rust"], default="auto", help="Audit with the Python path, the Rust bsor_tools backend, or auto-select.")
    parser.add_argument("--json-out", default="", help="Optional path for a JSON summary.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any replay fails the selected checks.")
    args = parser.parse_args()

    replay_dir = os.path.abspath(args.replay_dir)
    if not os.path.isdir(replay_dir):
        raise RuntimeError(f"Replay directory not found: {replay_dir}")

    replay_files = sorted(
        os.path.join(replay_dir, name)
        for name in os.listdir(replay_dir)
        if name.lower().endswith(".bsor")
    )
    if int(args.limit) > 0:
        replay_files = replay_files[: int(args.limit)]
    if not replay_files:
        raise RuntimeError(f"No .bsor files found in {replay_dir}")

    use_rust = args.backend == "rust" or (
        args.backend == "auto" and bsor_tools_available(auto_build=False)
    )
    if use_rust:
        summary = audit_replays_via_rust(
            replay_dir,
            limit=args.limit,
            check=args.check,
            strict=args.strict,
            auto_build=(args.backend == "rust"),
        )
    else:
        results = [audit_replay(path, args.check) for path in replay_files]
        summary = summarize_results(results)
        summary["replay_dir"] = replay_dir
        summary["check"] = args.check

    print(f"Scanned {summary['count']} replays from {replay_dir}")
    print(f"  Raw BSOR parse ok: {summary['raw_bsor_ok']}/{summary['count']}")
    if args.check in {"dataset", "both"}:
        print(f"  Dataset parse ok:  {summary['dataset_parse_ok']}/{summary['count']}")
    print(f"  Failures:          {summary['failure_count']}")
    if summary["top_errors"]:
        print("  Top errors:")
        for item in summary["top_errors"]:
            print(f"    {item['count']:4d}  {item['error']}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved JSON summary to {args.json_out}")

    if args.strict and summary["failure_count"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
