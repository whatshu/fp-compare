"""x86-only fp128 helpers backed by GCC __float128 + libquadmath."""

from __future__ import annotations

import ctypes
import platform
import subprocess
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "c_fp128_x86.c"
SO = ROOT / "_fp128_x86.so"


class FP128Unavailable(RuntimeError):
    pass


def _is_x86() -> bool:
    m = platform.machine().lower()
    return m in {"x86_64", "amd64", "i386", "i686"}


def build_shared_lib() -> None:
    if SO.exists() and SO.stat().st_mtime >= SRC.stat().st_mtime:
        return
    if not _is_x86():
        raise FP128Unavailable(f"unsupported architecture: {platform.machine()}")

    cmd = [
        "gcc",
        "-O3",
        "-std=c11",
        "-fPIC",
        "-shared",
        str(SRC),
        "-o",
        str(SO),
        "-lquadmath",
        "-lm",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise FP128Unavailable(f"cannot build fp128 library: {detail}") from exc


def load_lib() -> ctypes.CDLL:
    build_shared_lib()
    lib = ctypes.CDLL(str(SO))
    lib.mean_abs_error_exp_taylor_f64.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.mean_abs_error_exp_taylor_f64.restype = ctypes.c_double
    lib.mean_abs_error_exp_taylor_f128.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.mean_abs_error_exp_taylor_f128.restype = ctypes.c_double
    return lib


def compare_taylor_exp_errors(degree: int = 25, points: int = 80) -> Tuple[float, float]:
    lib = load_lib()
    e64 = float(lib.mean_abs_error_exp_taylor_f64(degree, points))
    e128 = float(lib.mean_abs_error_exp_taylor_f128(degree, points))
    return e64, e128
