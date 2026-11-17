from __future__ import annotations

import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from ..config import SandboxConfig
from ..problems.base import InvalidOutput
from .sandbox import run_solver


@dataclass
class Outcome:
    status: str
    score: float | None
    stdout: str = ""
    stderr: str = ""
    elapsed: float = 0.0
    cases: list | None = None


class SandboxEvaluator:
    """One batch runs every candidate and the control process together.

    The caller advances the budget index once per batch. Each process has its
    own time and memory limit. The score comes from the problem scorer.
    """

    def __init__(self, problem, cfg: SandboxConfig, seed: int = 0) -> None:
        self.problem = problem
        self.cfg = cfg
        self.seed = seed
        self.instance = ""

    def bind(self, instance: str) -> None:
        self.instance = instance

    def run_many(self, codes: dict[str, str]) -> dict[str, Outcome]:
        workers = max(1, self.cfg.parallel)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {label: pool.submit(self._one, code) for label, code in codes.items()}
            return {label: fut.result() for label, fut in futures.items()}

    def _one(self, code: str) -> Outcome:
        try:
            with tempfile.TemporaryDirectory(prefix="rsi_src_", ignore_cleanup_errors=True) as tmp:
                path = Path(tmp) / "solver.py"
                path.write_text(code, encoding="utf-8")
                raw = run_solver(path, self.instance, self.seed, self.cfg)
        except Exception as exc:
            return Outcome("runtime_error", None, stderr=f"{type(exc).__name__}: {exc}")
        if raw.status != "ok":
            return Outcome(raw.status, None, raw.stdout, raw.stderr, raw.elapsed)
        try:
            score = float(self.problem.score(self.instance, raw.stdout))
        except (InvalidOutput, ValueError, TypeError) as exc:
            return Outcome("invalid_output", None, raw.stdout, f"{raw.stderr}\n{exc}", raw.elapsed)
        return Outcome("ok", score, raw.stdout, raw.stderr, raw.elapsed)
