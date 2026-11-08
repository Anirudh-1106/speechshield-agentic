"""
GPU helpers.

CUDA runtime libraries (cuBLAS, cuDNN) can be installed into the venv as
pip wheels (``pip install -r requirements-gpu.txt``) instead of a system
CUDA toolkit.  Those wheels put their DLLs under
``site-packages/nvidia/*/bin``, which Windows does not search by default —
``register_cuda_libraries()`` adds them so CTranslate2 (Whisper) finds them.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def register_cuda_libraries() -> list[str]:
    """Make pip-installed NVIDIA libraries loadable. Returns directories added."""
    added: list[str] = []
    for base in map(Path, sys.path):
        nvidia = base / "nvidia"
        if not nvidia.is_dir():
            continue
        for sub in ("bin", "lib"):
            for d in nvidia.glob(f"*/{sub}"):
                if not d.is_dir():
                    continue
                path = str(d)
                if sys.platform == "win32":
                    os.add_dll_directory(path)
                    os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
                added.append(path)
    return added


def cuda_device_count() -> int:
    """Number of CUDA devices CTranslate2 can see (0 if none or unusable)."""
    register_cuda_libraries()
    try:
        import ctranslate2
        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0
