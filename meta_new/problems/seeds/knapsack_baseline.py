# variant: baseline
import os
import sys
import time


def read_input():
    data = sys.stdin.read().split()
    n, cap = int(data[0]), int(data[1])
    items = [(int(data[2 + 2 * i]), int(data[3 + 2 * i])) for i in range(n)]
    return n, cap, items


def feasible(chosen, n, cap, items):
    if len(set(chosen)) != len(chosen):
        return False
    if any(i < 0 or i >= n for i in chosen):
        return False
    return sum(items[i][0] for i in chosen) <= cap


# --- solve begin ---
def solve(n, cap, items, deadline, iter_budget, rng):
    chosen, weight = [], 0
    for i, (w, v) in enumerate(items):
        if weight + w <= cap:
            chosen.append(i)
            weight += w
    return chosen
# --- solve end ---


def main():
    n, cap, items = read_input()
    time_limit = float(os.environ.get("RSI_TIME_LIMIT", "2.0"))
    iter_budget = int(os.environ.get("RSI_ITER_BUDGET", "20000"))
    seed = int(os.environ.get("RSI_SEED", "0"))
    import random
    rng = random.Random(seed)
    deadline = time.perf_counter() + time_limit * 0.8
    chosen = solve(n, cap, items, deadline, iter_budget, rng)
    if not feasible(chosen, n, cap, items):
        chosen = []
    print(" ".join(map(str, sorted(chosen))))


if __name__ == "__main__":
    main()
