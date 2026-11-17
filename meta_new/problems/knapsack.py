from __future__ import annotations

import random

from .base import InvalidOutput, Problem


class KnapsackProblem(Problem):
    name = "knapsack"
    direction = "max"
    baseline_file = "knapsack_baseline.py"
    statement = (
        "0/1 knapsack. Input: first line 'N C', then N lines 'w v' (weight, value). "
        "Output: one line with distinct item indices in 0..N-1 (may be empty) whose total weight is at most C. "
        "Score: total value, higher is better. Overweight or malformed output is invalid."
    )

    def generate(self, seed: int) -> str:
        rng = random.Random(seed)
        n = rng.randint(int(self.params.get("n_min", 60)), int(self.params.get("n_max", 120)))
        items = []
        for _ in range(n):
            w = rng.randint(5, 60)
            items.append((w, max(1, w + rng.randint(-5, 25))))
        cap = sum(w for w, _ in items) // 3
        return "\n".join([f"{n} {cap}"] + [f"{w} {v}" for w, v in items]) + "\n"

    def score(self, instance: str, output: str) -> float:
        data = instance.split()
        n, cap = int(data[0]), int(data[1])
        items = [(int(data[2 + 2 * i]), int(data[3 + 2 * i])) for i in range(n)]
        try:
            chosen = [int(t) for t in output.split()]
        except ValueError as e:
            raise InvalidOutput(str(e)) from e
        if len(set(chosen)) != len(chosen):
            raise InvalidOutput("duplicate item index")
        if any(i < 0 or i >= n for i in chosen):
            raise InvalidOutput("item index out of range")
        if sum(items[i][0] for i in chosen) > cap:
            raise InvalidOutput("total weight exceeds capacity")
        return float(sum(items[i][1] for i in chosen))
