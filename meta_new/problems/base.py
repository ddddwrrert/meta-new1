from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Literal

SEEDS_DIR = Path(__file__).parent / "seeds"


class InvalidOutput(Exception):
    pass


class Problem(ABC):
    """An AHC-style task: seeded instance generator plus a scorer for solver stdout."""

    name: ClassVar[str]
    direction: ClassVar[Literal["min", "max"]]
    statement: ClassVar[str]
    baseline_file: ClassVar[str]

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    def generate(self, seed: int) -> str: ...

    @abstractmethod
    def score(self, instance: str, output: str) -> float: ...

    def baseline_code(self) -> str:
        return (SEEDS_DIR / self.baseline_file).read_text(encoding="utf-8")
