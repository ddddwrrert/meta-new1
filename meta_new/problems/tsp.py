from __future__ import annotations

import math
import random

from .base import InvalidOutput, Problem


class TSPProblem(Problem):
    name = "tsp"
    direction = "min"
    baseline_file = "tsp_baseline.py"
    statement = (
        "Euclidean TSP. Input: first line N, then N lines 'x y' (integers in [0, 1000]). "
        "Output: one line with a permutation of 0..N-1 (the visiting order of a closed tour). "
        "Score: total closed tour length, lower is better. Any malformed output is invalid."
    )

    def generate(self, seed: int) -> str:
        rng = random.Random(seed)
        n = rng.randint(int(self.params.get("n_min", 60)), int(self.params.get("n_max", 120)))
        pts = [(rng.randint(0, 1000), rng.randint(0, 1000)) for _ in range(n)]
        return "\n".join([str(n)] + [f"{x} {y}" for x, y in pts]) + "\n"

    def score(self, instance: str, output: str) -> float:
        data = instance.split()
        n = int(data[0])
        pts = [(int(data[1 + 2 * i]), int(data[2 + 2 * i])) for i in range(n)]
        tokens = output.split()
        if len(tokens) != n:
            raise InvalidOutput(f"expected {n} indices, got {len(tokens)}")
        try:
            order = [int(t) for t in tokens]
        except ValueError as e:
            raise InvalidOutput(str(e)) from e
        if sorted(order) != list(range(n)):
            raise InvalidOutput("output is not a permutation of 0..N-1")
        return sum(math.dist(pts[order[i]], pts[order[(i + 1) % n]]) for i in range(n))
