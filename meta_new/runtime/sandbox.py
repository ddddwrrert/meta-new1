from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import SandboxConfig

_IS_WIN = os.name == "nt"


@dataclass
class RawRun:
    status: str
    stdout: str
    stderr: str
    elapsed: float


if _IS_WIN:
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_ulonglong)
            for name in (
                "ReadOperationCount",
                "WriteOperationCount",
                "OtherOperationCount",
                "ReadTransferCount",
                "WriteTransferCount",
                "OtherTransferCount",
            )
        ]

    class _BASIC_LIMIT(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BASIC_LIMIT),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _JobObjectExtendedLimitInformation = 9
    _LIMIT_PROCESS_MEMORY = 0x00000100
    _LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _k32.SetInformationJobObject.restype = wintypes.BOOL
    _k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _k32.QueryInformationJobObject.restype = wintypes.BOOL
    _k32.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    _k32.AssignProcessToJobObject.restype = wintypes.BOOL
    _k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _k32.TerminateJobObject.restype = wintypes.BOOL
    _k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _k32.CloseHandle.restype = wintypes.BOOL
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]

    def _attach_job(proc: subprocess.Popen, mem_mb: int):
        job = _k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _EXTENDED_LIMIT()
        info.BasicLimitInformation.LimitFlags = _LIMIT_KILL_ON_JOB_CLOSE | _LIMIT_PROCESS_MEMORY
        info.ProcessMemoryLimit = mem_mb * 1024 * 1024
        ok = _k32.SetInformationJobObject(
            job, _JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)
        )
        if not ok or not _k32.AssignProcessToJobObject(job, wintypes.HANDLE(int(proc._handle))):
            _k32.CloseHandle(job)
            return None
        return job

    def _peak_memory(job) -> int:
        info = _EXTENDED_LIMIT()
        if _k32.QueryInformationJobObject(
            job, _JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info), None
        ):
            return int(info.PeakProcessMemoryUsed)
        return 0

    def _kill(proc: subprocess.Popen, job) -> None:
        if job:
            _k32.TerminateJobObject(job, 1)
        try:
            proc.kill()
        except OSError:
            pass

    def _close(job) -> None:
        if job:
            _k32.CloseHandle(job)

else:
    import resource
    import signal

    def _posix_limits(mem_mb: int):
        limit = mem_mb * 1024 * 1024

        def apply() -> None:
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

        return apply

    def _kill(proc: subprocess.Popen, job) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass


def _child_env(tmp: str, seed: int, cfg: SandboxConfig) -> dict[str, str]:
    env = {k: os.environ[k] for k in ("SYSTEMROOT", "SystemRoot", "PATH", "PATHEXT", "WINDIR") if k in os.environ}
    env.update(
        TEMP=tmp,
        TMP=tmp,
        TMPDIR=tmp,
        HOME=tmp,
        PYTHONHASHSEED="0",
        PYTHONIOENCODING="utf-8",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        RSI_SEED=str(seed),
        RSI_TIME_LIMIT=str(cfg.time_limit_s),
        RSI_ITER_BUDGET=str(cfg.iter_budget),
    )
    return env


def load_check(code: str, mem_mb: int = 1024, timeout_s: float = 5.0) -> str | None:
    """Load a candidate without calling main.

    A top-level exception is a direct crash and is removed before scoring.
    A timeout is not a crash here: the official run records that result.
    """
    cfg = SandboxConfig(max_mem_mb=mem_mb, time_limit_s=timeout_s, tle_factor=1.0, hard_grace_s=0.0, iter_budget=0)
    with tempfile.TemporaryDirectory(prefix="rsi_load_", ignore_cleanup_errors=True) as tmp:
        (Path(tmp) / "candidate.py").write_text(code, encoding="utf-8")
        (Path(tmp) / "load.py").write_text(
            "import pathlib\n"
            "src = pathlib.Path('candidate.py').read_text(encoding='utf-8')\n"
            "exec(compile(src, 'candidate.py', 'exec'), {'__name__': 'candidate'})\n",
            encoding="utf-8",
        )
        kwargs: dict = dict(
            cwd=tmp,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_child_env(tmp, 0, cfg),
        )
        if _IS_WIN:
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            kwargs["start_new_session"] = True
            kwargs["preexec_fn"] = _posix_limits(mem_mb)
        proc = subprocess.Popen([sys.executable, "-I", "load.py"], **kwargs)
        job = _attach_job(proc, mem_mb) if _IS_WIN else None
        try:
            try:
                _out, err = proc.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                _kill(proc, job)
                proc.communicate()
                return None
            if proc.returncode == 0:
                return None
            tail = err.decode("utf-8", "replace").strip().splitlines()
            detail = tail[-1] if tail else "nonzero"
            return f"crash:{detail[:120]}"
        finally:
            if _IS_WIN:
                _close(job)


def run_solver(solver_path: Path, stdin_text: str, seed: int, cfg: SandboxConfig) -> RawRun:
    """Run one solver on one instance in a fresh working directory with time and memory limits."""
    tle = cfg.time_limit_s * cfg.tle_factor
    hard = tle + cfg.hard_grace_s
    with tempfile.TemporaryDirectory(prefix="rsi_run_", ignore_cleanup_errors=True) as tmp:
        shutil.copyfile(solver_path, Path(tmp) / "main.py")
        kwargs: dict = dict(
            cwd=tmp,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_child_env(tmp, seed, cfg),
        )
        if _IS_WIN:
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            kwargs["start_new_session"] = True
            kwargs["preexec_fn"] = _posix_limits(cfg.max_mem_mb)
        start = time.perf_counter()
        proc = subprocess.Popen([sys.executable, "-I", "main.py"], **kwargs)
        job = _attach_job(proc, cfg.max_mem_mb) if _IS_WIN else None
        timed_out = False
        try:
            out, err = proc.communicate(stdin_text.encode("utf-8"), timeout=hard)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill(proc, job)
            out, err = proc.communicate()
        elapsed = time.perf_counter() - start
        peak = _peak_memory(job) if job else 0
        if _IS_WIN:
            _close(job)
    stdout = out.decode("utf-8", "replace")
    stderr = err.decode("utf-8", "replace")
    if timed_out or elapsed > tle:
        status = "timeout"
    elif proc.returncode != 0:
        near_limit = peak and peak >= 0.95 * cfg.max_mem_mb * 1024 * 1024
        status = "memory" if ("MemoryError" in stderr or near_limit) else "runtime_error"
    else:
        status = "ok"
    return RawRun(status=status, stdout=stdout, stderr=stderr, elapsed=elapsed)
