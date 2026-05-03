import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from cybernoodles.data.fetch_data import load_player_config, save_player_config
from cybernoodles.paths import BRAND_NAME, PROJECT_ROOT, SETUP_COMPLETE_PATH, first_existing_model_path, model_candidate_paths


if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


C = "\033[96m"
M = "\033[95m"
G = "\033[92m"
Y = "\033[93m"
R = "\033[91m"
B = "\033[94m"
DIM = "\033[90m"
RST = "\033[0m"
BOLD = "\033[1m"


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def appdata_setup_dir():
    return SETUP_COMPLETE_PATH.parent


def ensure_app_state():
    appdata_setup_dir().mkdir(parents=True, exist_ok=True)


def setup_complete():
    return SETUP_COMPLETE_PATH.exists()


def mark_setup_complete():
    ensure_app_state()
    SETUP_COMPLETE_PATH.write_text("ok\n", encoding="utf-8")


def pause(message="Press Enter to return to the menu..."):
    input(f"\n{DIM}{message}{RST}")


def prompt_yes_no(message, default=True):
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"{message} {suffix} ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print(f"{Y}Please answer with y or n.{RST}")


def prompt_int(message, default):
    while True:
        raw = input(f"{message} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print(f"{Y}That needs to be a whole number.{RST}")


def prompt_float(message, default):
    while True:
        raw = input(f"{message} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print(f"{Y}That needs to be a number.{RST}")


def resolve_player_ids(raw_text):
    player_refs = []
    seen = set()
    for raw_part in str(raw_text or "").split(","):
        raw_part = raw_part.strip()
        if not raw_part:
            continue
        if raw_part in seen:
            continue
        seen.add(raw_part)
        player_refs.append(raw_part)
    return player_refs


def prompt_player_ids():
    existing = load_player_config()
    default_text = ", ".join(existing) if existing else ""
    prompt = "BeatLeader player IDs, usernames, or profile URLs"
    if default_text:
        raw = input(f"{prompt} [{default_text}]: ").strip()
        player_ids = resolve_player_ids(raw or default_text)
    else:
        while True:
            raw = input(f"{prompt}: ").strip()
            player_ids = resolve_player_ids(raw)
            if player_ids:
                break
            print(f"{Y}At least one player reference is needed to pull fresh replay data.{RST}")
    if player_ids:
        save_player_config(player_ids)
    return player_ids


def run_module(module_name, *args):
    command = [sys.executable, "-m", module_name, *args]
    print(f"\n{B}[RUN]{RST} {' '.join(command)}\n")
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    return result.returncode == 0


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def best_available_model():
    return first_existing_model_path(
        "rl_model",
        "awac_model",
        "rl_checkpoint",
        "awac_checkpoint",
        "bc_model",
        "bc_last",
    )


def print_banner():
    clear()
    print(
        f"""{M}
+====================================================================+
| {C}{BRAND_NAME.upper()} 6.0{M} :: {Y}AI TRAINING TERMINAL{M}                           |
| {DIM}CyberNoodles training, evaluation, and replay forging{M}              |
+====================================================================+{RST}
"""
    )


def print_status():
    state = read_json(PROJECT_ROOT / "rl_state.json", {})
    curriculum = read_json(PROJECT_ROOT / "curriculum.json", [])
    manifest = read_json(PROJECT_ROOT / "data" / "processed" / "bc_shards" / "manifest.json", {})
    counts = manifest.get("counts", {}) if isinstance(manifest, dict) else {}
    players = load_player_config()

    print(f"{C}SYSTEM STATUS{RST}")
    print(f"{DIM}{'-' * 68}{RST}")
    print(f"  Setup marker : {G}READY{RST}" if setup_complete() else f"  Setup marker : {Y}FIRST RUN PENDING{RST}")
    print(f"  AppData file : {DIM}{SETUP_COMPLETE_PATH}{RST}")
    print(f"  Players      : {G}{', '.join(players)}{RST}" if players else f"  Players      : {DIM}none saved yet{RST}")

    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"  GPU          : {G}{torch.cuda.get_device_name(0)}{RST}  {DIM}({props.total_memory / 1e9:.1f} GB){RST}")
        else:
            print(f"  GPU          : {R}No CUDA device detected{RST}")
    except Exception as exc:
        print(f"  GPU          : {Y}Probe failed: {exc}{RST}")

    epoch = int(state.get("epoch", 0) or 0)
    moving_acc = float(state.get("moving_acc", 0.0) or 0.0)
    best_task = float(state.get("global_best_task_accuracy", state.get("global_best_accuracy", 0.0)) or 0.0)
    best_sel = float(state.get("global_best_selection_score", 0.0) or 0.0)
    awac_state = read_json(PROJECT_ROOT / "awac_state.json", {})
    awac_epoch = int(awac_state.get("epoch", 0) or 0)
    awac_best = float(awac_state.get("best_strict_accuracy", 0.0) or 0.0)
    print(f"  RL epoch     : {Y}{epoch}{RST}")
    print(f"  RL acc       : {Y}{moving_acc:.2f}%{RST}")
    print(f"  Best task    : {G}{best_task:.2f}%{RST}")
    print(f"  Best score   : {G}{best_sel:.4f}{RST}")
    if awac_epoch > 0 or awac_best > 0.0:
        print(f"  AWAC epoch   : {Y}{awac_epoch}{RST}")
        print(f"  AWAC strict  : {G}{awac_best:.2f}%{RST}")

    if isinstance(curriculum, list):
        easy = sum(1 for item in curriculum if float(item.get("nps", 0.0) or 0.0) < 2.0)
        medium = sum(1 for item in curriculum if 2.0 <= float(item.get("nps", 0.0) or 0.0) < 4.0)
        hard = sum(1 for item in curriculum if float(item.get("nps", 0.0) or 0.0) >= 4.0)
        print(f"  Curriculum   : {G}{len(curriculum)} maps{RST}  {DIM}({easy} easy / {medium} medium / {hard} hard){RST}")
    else:
        print(f"  Curriculum   : {DIM}not built yet{RST}")

    train_samples = int(counts.get("train_samples", 0) or 0)
    val_samples = int(counts.get("val_samples", 0) or 0)
    if train_samples or val_samples:
        print(f"  BC shards    : {G}{train_samples:,}{RST} train / {G}{val_samples:,}{RST} val")
    else:
        print(f"  BC shards    : {DIM}not built yet{RST}")

    for label, kind in (
        ("BC model", "bc_model"),
        ("AWAC model", "awac_model"),
        ("RL model", "rl_model"),
        ("Checkpoint", "rl_checkpoint"),
    ):
        path = first_existing_model_path(kind)
        if path is not None and path.exists():
            legacy = " legacy" if path.name.startswith("bsai_") else ""
            print(f"  {label:<12}: {G}{path.stat().st_size / 1e6:.1f} MB{RST}{DIM}{legacy}{RST}")
        else:
            print(f"  {label:<12}: {DIM}missing{RST}")
    print(f"{DIM}{'-' * 68}{RST}")


def guided_data_and_bc(initial=False):
    print_banner()
    print(f"{C}GUIDED BC PIPELINE{RST}")
    print(f"{DIM}{'-' * 68}{RST}")
    print("  This pipeline can pull fresh BeatLeader replays, rebuild BC shards,")
    print("  and retrain the behavior-cloning warmstart before PPO takes over.\n")

    pull_data = True if initial else prompt_yes_no("Pull fresh BeatLeader replay data first?", default=True)
    player_ids = load_player_config()

    if pull_data:
        player_ids = prompt_player_ids()
        top_n = prompt_int("Replay cap to keep after filtering", 5000)
        min_accuracy = prompt_float("Minimum replay accuracy", 0.85)

        fetch_args = []
        for player_id in player_ids:
            fetch_args.extend(["--player-id", player_id])
        fetch_args.extend(["--top-n", str(top_n), "--min-accuracy", str(min_accuracy)])

        if not run_module("cybernoodles.data.fetch_data", *fetch_args):
            print(f"{R}Replay pulling failed. Stopping the pipeline.{RST}")
            return False
        if not run_module("cybernoodles.data.map_analyzer"):
            print(f"{R}Map analysis failed. Stopping the pipeline.{RST}")
            return False
    elif not player_ids:
            print(f"{Y}No saved player references were found. You can still rebuild shards from existing local data.{RST}")

    if not run_module("cybernoodles.data.dataset_builder"):
        print(f"{R}Dataset building failed. Stopping the pipeline.{RST}")
        return False

    if not run_module("cybernoodles.data.sim_calibration"):
        print(f"{R}Simulator calibration failed. Stopping the pipeline.{RST}")
        return False

    if not run_module("cybernoodles.data.style_calibration"):
        print(f"{R}Style calibration failed. Stopping the pipeline.{RST}")
        return False

    should_train_bc = True if initial else prompt_yes_no("Train BC after rebuilding the dataset?", default=True)
    if should_train_bc:
        bc_args = []
        if prompt_yes_no("Use fast BC sanity-check mode instead of a full BC retrain?", default=False):
            bc_args.append("--quick")
        if not run_module("cybernoodles.training.train_bc", *bc_args):
            print(f"{R}BC training failed.{RST}")
            return False
    else:
        print(f"{Y}Skipped BC training. Your shards are still ready for later.{RST}")

    return True


def first_run_setup():
    if setup_complete():
        return

    while True:
        print_banner()
        print(f"{Y}FIRST RUN SETUP{RST}")
        print(f"{DIM}{'-' * 68}{RST}")
        print("  No setup marker was found in AppData yet.")
        print("  We can do the full training bootstrap now, or skip it if you only")
        print("  want to generate replays / run inference today.\n")
        print(f"    {C}[1]{RST} Full setup: pull replays, build shards, train BC")
        print(f"    {C}[2]{RST} Skip setup for now and go straight to the terminal")
        print(f"    {C}[3]{RST} Exit\n")

        choice = input(f"  {M}>{RST} ").strip()
        if choice == "1":
            success = guided_data_and_bc(initial=True)
            if success:
                mark_setup_complete()
                print(f"\n{G}Setup complete. The AppData marker has been written.{RST}")
                time.sleep(1.5)
                return
            pause("Setup did not finish cleanly. Press Enter to try again or close the app.")
        elif choice == "2":
            mark_setup_complete()
            print(f"\n{Y}Setup skipped. You can always run the BC pipeline later from option 1.{RST}")
            time.sleep(1.2)
            return
        elif choice == "3":
            raise SystemExit(0)
        else:
            print(f"{Y}Pick 1, 2, or 3.{RST}")
            time.sleep(1.0)


def run_rl_training():
    print(f"\n{B}[PPO]{RST} Starting PPO training. Press Ctrl+C to return to the menu.\n")
    time.sleep(0.5)
    try:
        from cybernoodles.training.train_rl_gpu import train_ppo_gpu

        train_ppo_gpu()
    except KeyboardInterrupt:
        print(f"\n{Y}RL training interrupted. Returning to the terminal.{RST}")
    except Exception as exc:
        print(f"\n{R}RL training crashed: {exc}{RST}")
        traceback.print_exc()


def run_awac_training():
    print(f"\n{B}[AWAC]{RST} Starting AWAC bootstrap training. Press Ctrl+C to return to the menu.\n")
    time.sleep(0.5)
    try:
        from cybernoodles.training.train_awac import train_awac

        train_awac()
    except KeyboardInterrupt:
        print(f"\n{Y}AWAC training interrupted. Returning to the terminal.{RST}")
    except Exception as exc:
        print(f"\n{R}AWAC training crashed: {exc}{RST}")
        traceback.print_exc()


def make_replay():
    print_banner()
    print(f"{C}REPLAY FORGE{RST}")
    print(f"{DIM}{'-' * 68}{RST}")
    print("  Enter a BeatSaver BSR code or a full map hash. The generator will")
    print("  use your best available model unless you point it at a different one.\n")

    default_model = best_available_model()
    if default_model is None:
        print(f"{Y}No local model was auto-detected. You can still type a path manually.{RST}")

    map_id = input(f"  {M}BSR / Hash:{RST} ").strip()
    if not map_id:
        print(f"{Y}Replay generation cancelled.{RST}")
        return

    default_model_text = str(default_model.name) if default_model else model_candidate_paths("rl_model")[0].name
    model_raw = input(f"  {M}Model path:{RST} [{default_model_text}] ").strip()
    model_path = model_raw or str(default_model if default_model else model_candidate_paths("rl_model")[0])

    output_file = input(f"  {M}Output file:{RST} [CyberNoodles_Replay.bsor] ").strip() or "CyberNoodles_Replay.bsor"
    diff_raw = input(f"  {M}Difficulty index:{RST} [-1 = hardest] ").strip()
    diff_index = int(diff_raw) if diff_raw else -1

    try:
        from cybernoodles.replay.generate_replay import generate_bsor

        print()
        generate_bsor(
            map_id,
            output_file=output_file,
            diff_index=diff_index,
            model_path=model_path,
        )
    except KeyboardInterrupt:
        print(f"\n{Y}Replay generation interrupted.{RST}")
    except Exception as exc:
        print(f"\n{R}Replay generation failed: {exc}{RST}")
        traceback.print_exc()


def print_menu():
    print(
        f"""
  {BOLD}Choose your lane{RST}

    {C}[1]{RST} Retrain BC / Pull new data
    {C}[2]{RST} Train AWAC bootstrap
    {C}[3]{RST} Train PPO
    {C}[4]{RST} Make a replay
    {C}[5]{RST} Exit
"""
    )


def main_menu():
    while True:
        print_banner()
        print_status()
        print_menu()
        choice = input(f"  {M}>{RST} ").strip()

        if choice == "1":
            success = guided_data_and_bc(initial=False)
            if success:
                mark_setup_complete()
            pause()
        elif choice == "2":
            run_awac_training()
            pause()
        elif choice == "3":
            run_rl_training()
            pause()
        elif choice == "4":
            make_replay()
            pause()
        elif choice == "5" or choice.lower() in ("q", "quit", "exit"):
            print(f"\n{M}CyberNoodles offline.{RST}\n")
            break
        else:
            print(f"{Y}Pick 1, 2, 3, 4, or 5.{RST}")
            time.sleep(1.0)


def main():
    os.chdir(PROJECT_ROOT)
    ensure_app_state()
    first_run_setup()
    main_menu()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{M}CyberNoodles offline.{RST}\n")
