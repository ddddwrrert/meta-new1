# variant: baseline
import os
import sys
import time


def read_input():
    data = sys.stdin.read().split()
    n = int(data[0])
    pts = [(int(data[1 + 2 * i]), int(data[2 + 2 * i])) for i in range(n)]
    return n, pts


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def valid(order, n):
    return len(order) == n and sorted(order) == list(range(n))


# --- solve begin ---
def solve(n, pts, deadline, iter_budget, rng):
    return list(range(n))
# --- solve end ---


def main():
    n, pts = read_input()
    time_limit = float(os.environ.get("RSI_TIME_LIMIT", "2.0"))
    iter_budget = int(os.environ.get("RSI_ITER_BUDGET", "20000"))
    seed = int(os.environ.get("RSI_SEED", "0"))
    import random
    rng = random.Random(seed)
    deadline = time.perf_counter() + time_limit * 0.8
    order = solve(n, pts, deadline, iter_budget, rng)
    if not valid(order, n):
        order = list(range(n))
    print(" ".join(map(str, order)))


if __name__ == "__main__":
    main()
