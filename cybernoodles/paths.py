import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
SCRATCH_DIR = PROJECT_ROOT / "scratch"

BRAND_NAME = "CyberNoodles"

MODEL_FILENAME_ALIASES = {
    "bc_model": ("cybernoodles_bc_model.pth", "bsai_bc_model.pth"),
    "bc_last": ("cybernoodles_bc_last.pth", "bsai_bc_last.pth"),
    "awac_model": ("cybernoodles_awac_model.pth", "bsai_awac_model.pth"),
    "awac_checkpoint": ("cybernoodles_awac_checkpoint.pth", "bsai_awac_checkpoint.pth"),
    "rl_model": ("cybernoodles_rl_model.pth", "bsai_rl_model.pth"),
    "rl_checkpoint": ("cybernoodles_rl_checkpoint.pth", "bsai_rl_checkpoint.pth"),
}


def model_candidate_paths(kind):
    return tuple(PROJECT_ROOT / filename for filename in MODEL_FILENAME_ALIASES[kind])


def preferred_model_filename(kind):
    return MODEL_FILENAME_ALIASES[kind][0]


def first_existing_model_path(*kinds):
    for kind in kinds:
        for path in model_candidate_paths(kind):
            if path.exists():
                return path
    return None


def existing_or_preferred_model_path(kind):
    return str(first_existing_model_path(kind) or model_candidate_paths(kind)[0])


def _appdata_root():
    candidates = [
        os.getenv("APPDATA"),
        os.getenv("LOCALAPPDATA"),
    ]
    for raw in candidates:
        if raw:
            return Path(raw)
    return Path.home() / "AppData" / "Roaming"


APP_STATE_DIR = _appdata_root() / "CyberNoodles"
SETUP_COMPLETE_PATH = APP_STATE_DIR / "setup_complete"
