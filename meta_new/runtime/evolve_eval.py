"""Run an organon evolve-task evaluator on C++ (or other) candidate files.

Each candidate is scored by that task's evaluator/evaluator.py, with the timeout
and retries from task.json. ALE-Bench starts a session per call, so a batch runs
one program at a time. On Windows the interpreter is the organon environment
inside WSL, because that environment owns ale_bench and Docker.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .evaluate import Outcome

_RUNNER = Path(__file__).with_name("task_runner.py")


def to_wsl(path: str | Path) -> str:
    text = str(path)
    if text.startswith("/"):
        return text.replace("\\", "/")
    resolved = Path(text).resolve()
    drive = resolved.drive
    if len(drive) < 2:
        return resolved.as_posix()
    letter = drive[0].lower()
    rest = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{letter}{rest}"


class EvolveTaskEvaluator:
    def __init__(self, problem, python: str | None = None, via: str = "native", distro: str | None = None) -> None:
        if via not in ("native", "wsl"):
            raise ValueError("evaluator.via must be native or wsl")
        chosen = python or os.environ.get("META_EVAL_PYTHON")
        if via == "wsl" and not chosen:
            raise ValueError("evaluator.python must be a Linux interpreter when evaluator.via is wsl")
        self.problem = problem
        self.via = via
        self.distro = distro or os.environ.get("META_EVAL_DISTRO")
        self.python = chosen or sys.executable
        self.instance = ""

    def bind(self, instance: str) -> None:
        self.instance = instance

    def run_many(self, codes: dict[str, str]) -> dict[str, Outcome]:
        return {label: self._one(code) for label, code in codes.items()}

    def _one(self, code: str) -> Outcome:
        suffix = Path(self.problem.program_name).suffix or ".cpp"
        started = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="rsi_cpp_") as tmp:
                program = Path(tmp) / f"candidate{suffix}"
                program.write_text(code, encoding="utf-8")
                proc = subprocess.run(
                    self._command(program),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self._cap(),
                )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.perf_counter() - started
            err = (exc.stderr or "")[-4000:]
            return Outcome("timeout", None, stderr=err, elapsed=elapsed)
        except OSError as exc:
            return Outcome("error", None, stderr=f"{type(exc).__name__}: {exc}")
        elapsed = time.perf_counter() - started
        payload = _payload(proc.stdout)
        stderr = (proc.stderr or "")[-4000:]
        if payload is None:
            detail = stderr or (proc.stdout or "")[-1000:]
            return Outcome("error", None, stderr=detail or f"evaluator exited {proc.returncode}", elapsed=elapsed)
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        score = payload.get("fitness")
        try:
            number = None if score is None else float(score)
        except (TypeError, ValueError):
            number = None
        brief = json.dumps(
            {
                "judge_result": metrics.get("judge_result"),
                "combined_score": metrics.get("combined_score"),
                "message": metrics.get("message") or metrics.get("error"),
            },
            ensure_ascii=False,
            default=str,
        )
        return Outcome(_status(metrics), number, brief, stderr, elapsed, _cases(metrics))

    def _cap(self) -> float:
        attempts = self.problem.max_retries + 1
        return self.problem.timeout * attempts + 5 * attempts + 30

    def _command(self, program: Path) -> list[str]:
        if self.via == "wsl":
            command = ["wsl"]
            if self.distro:
                command += ["-d", self.distro]
            command += ["-e", self.python, to_wsl(_RUNNER), to_wsl(self.problem.task_dir), to_wsl(program)]
            return command
        return [self.python, str(_RUNNER), str(self.problem.task_dir), str(program)]


def _payload(stdout: str) -> dict | None:
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "fitness" in obj:
            return obj
    return None


def _cases(metrics: dict) -> list | None:
    raw = metrics.get("cases")
    if not isinstance(raw, list):
        return None
    rows = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        try:
            rows.append({"seed": int(row["seed"]), "score": float(row["score"])})
        except (KeyError, TypeError, ValueError):
            continue
    return rows or None


def _status(metrics: dict) -> str:
    if metrics.get("timeout"):
        return "timeout"
    if "combined_score" not in metrics and metrics.get("error"):
        return "error"
    judge = str(metrics.get("judge_result") or "")
    if judge and judge.upper() not in ("ACCEPTED", "AC"):
        return judge
    return "ok"
