import os
from pathlib import Path
import shutil
import subprocess
import warnings

import torch
from torch.utils.cpp_extension import CUDA_HOME, load

from cybernoodles.paths import PROJECT_ROOT


_SIM_CUDA_EXT = None
_SIM_CUDA_EXT_ERROR = None


def _find_vsdevcmd():
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.exists():
        try:
            result = subprocess.run(
                [
                    str(vswhere),
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            install_path = result.stdout.strip()
            if install_path:
                candidate = Path(install_path) / "Common7" / "Tools" / "VsDevCmd.bat"
                if candidate.exists():
                    return candidate
        except Exception:
            pass

    fallbacks = [
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"),
    ]
    for candidate in fallbacks:
        if candidate.exists():
            return candidate
    return None


def _ensure_windows_msvc_env():
    if os.name != "nt":
        return

    if shutil.which("cl") is not None:
        return

    vsdevcmd = _find_vsdevcmd()
    if vsdevcmd is None:
        return

    try:
        cmd = f'cmd /c ""{vsdevcmd}" -arch=amd64 >nul && set"'
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            shell=True,
        )
    except Exception:
        return

    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key:
            os.environ[key] = value


def get_sim_cuda_extension(verbose=False):
    global _SIM_CUDA_EXT, _SIM_CUDA_EXT_ERROR

    if os.environ.get("BSAI_DISABLE_NATIVE_SIM", "").strip() == "1":
        return None

    if _SIM_CUDA_EXT is not None:
        return _SIM_CUDA_EXT

    if _SIM_CUDA_EXT_ERROR is not None:
        return None

    if not torch.cuda.is_available():
        return None

    if CUDA_HOME is None:
        _SIM_CUDA_EXT_ERROR = RuntimeError("CUDA toolkit not found")
        warnings.warn(
            "CUDA toolkit not found; falling back to the PyTorch simulator path.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    _ensure_windows_msvc_env()
    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

    base_dir = Path(__file__).resolve().parent
    build_dir = PROJECT_ROOT / "scratch" / "sim_cuda_ext_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    name = f"bsai_sim_cuda_ext_t{torch.__version__.split('+')[0].replace('.', '_')}_cu{str(torch.version.cuda or 'cpu').replace('.', '_')}"
    sources = [
        str(base_dir / "sim_cuda_kernels.cpp"),
        str(base_dir / "sim_cuda_kernels.cu"),
    ]

    extra_cflags = ["/O2", "/std:c++17", "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"] if os.name == "nt" else ["-O3", "-std=c++17"]
    extra_cuda_cflags = ["-O3", "--use_fast_math", "-lineinfo", "-std=c++17", "-allow-unsupported-compiler", "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"]
    verbose = verbose or (os.environ.get("BSAI_NATIVE_SIM_VERBOSE", "").strip() == "1")

    try:
        _SIM_CUDA_EXT = load(
            name=name,
            sources=sources,
            build_directory=str(build_dir),
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            with_cuda=True,
            verbose=verbose,
        )
    except Exception as exc:
        _SIM_CUDA_EXT_ERROR = exc
        warnings.warn(
            f"Native simulator extension build failed; using the PyTorch fallback. {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    return _SIM_CUDA_EXT
