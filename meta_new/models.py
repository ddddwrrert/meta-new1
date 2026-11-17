from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Candidate:
    code: str
    kind: str
    edit_class: str
    hypothesis: str = ""
    origin: str = ""


@dataclass
class ProposeContext:
    phase: str
    problem_name: str
    statement: str
    direction: str
    incumbent_code: str
    rho: float
    n: int
    n_explore: int
    n_local: int
    verdict: str | None = None
    official: float | None = None
    repeated_worse: list[str] = field(default_factory=list)
    history: list = field(default_factory=list)
    skill_text: str = ""
    trajectory_excerpt: str = ""
    budget_index: int = 0
    previous_official: float | None = None
    candidate_scores: list = field(default_factory=list)
    language: str = "python"


@dataclass
class ProblemState:
    name: str
    direction: str
    instance: str
    incumbent_code: str
    incumbent_sha: str
    official: float | None
    epsilon: float
    rho: float
    flat_streak: int
    budget: int
    skill: dict
    history: dict
    blacklist: list
    score_history: list
