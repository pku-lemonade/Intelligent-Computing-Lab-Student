from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import os
from pathlib import Path
import shutil
import sys

import torch
from torch.utils.cpp_extension import load

from microllm.kernels.errors import KernelUnavailable


@lru_cache(maxsize=8)
def load_cuda_extension(name: str, sources: tuple[str, ...]):
    if not torch.cuda.is_available():
        raise KernelUnavailable("CUDA is not available")
    root = Path(__file__).resolve().parents[2]
    try:
        with _python_bin_on_path():
            return load(
                name=name,
                sources=[str(root / source) for source in sources],
                extra_cflags=["-O3"],
                extra_cuda_cflags=_cuda_cflags(root),
                verbose=False,
            )
    except Exception as exc:
        raise KernelUnavailable(str(exc)) from exc


@contextmanager
def _python_bin_on_path():
    old_path = os.environ.get("PATH", "")
    python_bin = str(Path(sys.executable).parent)
    os.environ["PATH"] = python_bin + os.pathsep + old_path
    try:
        yield
    finally:
        os.environ["PATH"] = old_path


def _cuda_cflags(root: Path) -> list[str]:
    flags = ["-O3", "-allow-unsupported-compiler"]
    if compiler := _project_gcc14(root):
        flags.append(f"-ccbin={compiler}")
        flags.extend(_gcc14_include_flags(root))
    return flags


def _project_gcc14(root: Path) -> Path | None:
    candidates = [
        root / "dependence/gcc14-root/usr/bin/x86_64-linux-gnu-g++-14",
        Path("/usr/bin/g++-14"),
        shutil.which("g++-14"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


def _gcc14_include_flags(root: Path) -> list[str]:
    local = root / "dependence/gcc14-root/usr"
    if (local / "include/c++/14").exists():
        paths = [
            local / "lib/gcc/x86_64-linux-gnu/14/include",
            local / "include/c++/14",
            local / "include/x86_64-linux-gnu/c++/14",
            local / "include/c++/14/backward",
        ]
    else:
        paths = [
            Path("/usr/lib/gcc/x86_64-linux-gnu/14/include"),
            Path("/usr/include/c++/14"),
            Path("/usr/include/x86_64-linux-gnu/c++/14"),
            Path("/usr/include/c++/14/backward"),
        ]
    flags: list[str] = []
    for path in paths:
        if path.exists():
            flags.extend(["-isystem", str(path)])
    return flags


def assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 1e-6,
    rtol: float = 0.0,
) -> None:
    diff = (actual.float() - expected.float()).abs()
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(f"{name}: max_abs_diff={diff.max().item():.6f}")


def benchmark_cuda(fn, *, warmup: int = 10, repeat: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat
