from __future__ import annotations

from .evolve import EvolveTaskProblem
from .knapsack import KnapsackProblem
from .tsp import TSPProblem

_TABLE = {"tsp": TSPProblem, "knapsack": KnapsackProblem}


def build_problem(name: str, params: dict | None = None):
    params = dict(params or {})
    if name == "evolve":
        task_dir = params.pop("task_dir", None)
        if not task_dir:
            raise KeyError("evolve task requires task_dir")
        return EvolveTaskProblem(str(task_dir), **params)
    if name not in _TABLE:
        raise KeyError(f"unknown problem: {name}")
    return _TABLE[name](**params)
