import argparse
import json
import os
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import requests

from cybernoodles.paths import DATA_DIR, MODEL_FILENAME_ALIASES, PROJECT_ROOT, SCRATCH_DIR


UPDATE_CONFIG_PATH = DATA_DIR / "github_update_config.json"
PRESERVE_DIRS = {
    ".git",
    "__pycache__",
    "data",
    "runs",
    "Past_Models",
    "Progress Replays",
    "scratch",
    "torchinductor_cache",
    ".venv",
    "venv",
}
PRESERVE_FILES = {
    *(filename for aliases in MODEL_FILENAME_ALIASES.values() for filename in aliases),
    "bsai_bc_last.pth",
    "bsai_bc_model.pth",
    "bsai_rl_checkpoint.pth",
    "bsai_rl_model.pth",
    "rl_state.json",
    "curriculum.json",
    "sim_calibration.json",
}


def load_update_config():
    if not UPDATE_CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(UPDATE_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_update_config(repo, ref):
    UPDATE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    UPDATE_CONFIG_PATH.write_text(
        json.dumps({"repo": repo, "ref": ref}, indent=2),
        encoding="utf-8",
    )


def normalize_repo(repo):
    text = str(repo or "").strip().rstrip("/")
    if not text:
        return None
    if re.fullmatch(r"[\w.-]+/[\w.-]+", text):
        return text
    match = re.search(r"github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?$", text, flags=re.IGNORECASE)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return None


def _should_skip(relative_path):
    parts = relative_path.parts
    if not parts:
        return True
    if any(part in PRESERVE_DIRS for part in parts):
        return True
    if relative_path.name in PRESERVE_FILES:
        return True
    return False


def update_from_github(repo, ref="main", token=None):
    normalized_repo = normalize_repo(repo)
    if not normalized_repo:
        raise ValueError(f"Unsupported GitHub repo reference: {repo}")

    save_update_config(normalized_repo, ref)
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "CyberNoodles-Updater/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    archive_url = f"https://api.github.com/repos/{normalized_repo}/zipball/{ref}"
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    backup_root = SCRATCH_DIR / "update_backups" / time.strftime("%Y%m%d-%H%M%S")
    backup_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="cybernoodles-update-", dir=SCRATCH_DIR) as temp_dir:
        temp_dir_path = Path(temp_dir)
        archive_path = temp_dir_path / "repo.zip"
        response = requests.get(archive_url, headers=headers, timeout=60)
        response.raise_for_status()
        archive_path.write_bytes(response.content)

        with zipfile.ZipFile(archive_path, "r") as zip_file:
            zip_file.extractall(temp_dir_path / "extract")

        extracted_roots = [path for path in (temp_dir_path / "extract").iterdir() if path.is_dir()]
        if not extracted_roots:
            raise RuntimeError("Downloaded GitHub archive did not contain a repository root.")
        repo_root = extracted_roots[0]

        copied = []
        for source in repo_root.rglob("*"):
            if source.is_dir():
                continue
            relative_path = source.relative_to(repo_root)
            if _should_skip(relative_path):
                continue

            destination = PROJECT_ROOT / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                backup_target = backup_root / relative_path
                backup_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination, backup_target)
            shutil.copy2(source, destination)
            copied.append(str(relative_path).replace("\\", "/"))

    print(f"Updated {len(copied)} files from {normalized_repo}@{ref}.")
    print(f"Backup written to {backup_root}.")
    for item in copied[:25]:
        print(f"  {item}")
    if len(copied) > 25:
        print(f"  ... and {len(copied) - 25} more")


def main():
    parser = argparse.ArgumentParser(description="Update CyberNoodles code from a GitHub repository archive.")
    parser.add_argument("--repo", default=None, help="GitHub repo URL or owner/repo slug.")
    parser.add_argument("--ref", default=None, help="Branch, tag, or commit-ish to download.")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"), help="Optional GitHub token for private repositories.")
    args = parser.parse_args()

    config = load_update_config()
    repo = args.repo or config.get("repo")
    ref = args.ref or config.get("ref") or "main"
    if not repo:
        raise SystemExit("No GitHub repo configured. Pass --repo owner/repo or a full GitHub URL.")

    update_from_github(repo, ref=ref, token=args.token)


if __name__ == "__main__":
    main()
