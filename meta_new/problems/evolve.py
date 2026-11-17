"""An organon evolve-task. The baseline is the task's own program file.

ALE-Bench tasks ship solution.cpp. The score is the evaluator's combined_score,
which is already oriented so that a larger value is better.
"""

from __future__ import annotations

import json
from pathlib import Path

from .base import InvalidOutput, Problem

_CPP = {".cpp", ".cc", ".cxx", ".c++", ".hpp", ".h"}


def resolve_task_dir(config_path: str | Path, task_dir: str) -> Path:
    path = Path(task_dir)
    if not path.is_absolute():
        path = Path(config_path).resolve().parent / path
    return path.resolve()


class EvolveTaskProblem(Problem):
    name = "evolve"
    direction = "max"
    baseline_file = ""
    statement = ""

    def __init__(self, task_dir: str, **params) -> None:
        super().__init__(**params)
        self.task_dir = Path(task_dir).resolve()
        settings_path = self.task_dir / "task.json"
        if not settings_path.is_file():
            raise FileNotFoundError(f"evolve task not found: {self.task_dir}")
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        self.program_name = str(settings["program"])
        self.timeout = float(settings.get("timeout", 10000))
        self.max_retries = int(settings.get("max_retries", 0))
        program = self.task_dir / self.program_name
        self._baseline = program.read_text(encoding="utf-8")
        statement = self.task_dir / "statement.md"
        self.statement = statement.read_text(encoding="utf-8") if statement.is_file() else self.task_dir.name
        self.language = "cpp" if program.suffix.lower() in _CPP else "python"
        self.name = self.task_dir.name
        self.direction = "max"

    def baseline_code(self) -> str:
        return self._baseline

    def generate(self, seed: int) -> str:
        return f"{self.name}\nseed={seed}\n"

    def score(self, instance: str, output: str) -> float:
        raise InvalidOutput("this task is scored by its evaluator")
