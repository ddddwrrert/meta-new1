"""Adjacent-budget verdicts, tolerance, and the single rho update.

Official score changes only when a candidate is better. q is computed from the
direction-best candidate, including one that was not accepted. The control
score never enters q and never becomes the official score.
"""

from __future__ import annotations

import math

from ..util import clip

Verdict = str  # better | worse | flat


def direction_name(problem_direction: str) -> str:
    if problem_direction in ("max", "higher"):
        return "higher"
    if problem_direction in ("min", "lower"):
        return "lower"
    raise ValueError(f"unknown direction: {problem_direction}")


def iround(value: float) -> int:
    return int(math.floor(value + 0.5))


def mix_counts(rho: float, n: int) -> tuple[int, int]:
    """round(rho N) exploratory candidates; the rest are local. N>=2 keeps both."""
    explore = iround(rho * n)
    if n >= 2:
        explore = min(max(explore, 1), n - 1)
    else:
        explore = 1 if rho >= 0.5 else 0
    return explore, n - explore


def judge(direction: str, previous, candidate, epsilon: float) -> Verdict:
    prev_ok = previous is not None
    cand_ok = candidate is not None
    if prev_ok and not cand_ok:
        return "worse"
    if not prev_ok and cand_ok:
        return "better"
    if not prev_ok and not cand_ok:
        return "flat"
    diff = candidate - previous
    if abs(diff) <= epsilon:
        return "flat"
    if direction == "higher":
        return "better" if diff > 0 else "worse"
    return "better" if diff < 0 else "worse"


def best_score(direction: str, scores: list):
    legal = [s for s in scores if s is not None]
    if not legal:
        return None
    return max(legal) if direction == "higher" else min(legal)


def select_winner(direction: str, judged: list[tuple[int, Verdict, float | None]]) -> int | None:
    """judged items are (index, verdict, score). Return the index of the best better candidate."""
    betters = [(i, s) for i, v, s in judged if v == "better" and s is not None]
    if not betters:
        return None
    pick = max if direction == "higher" else min
    return pick(betters, key=lambda item: item[1])[0]


def quality_q(direction: str, previous, best, epsilon: float, eta: float) -> float:
    if best is None and previous is None:
        return 0.0
    if best is None and previous is not None:
        return -1.0
    if previous is None and best is not None:
        return 1.0
    if abs(best - previous) <= epsilon:
        return 0.0
    denom = max(abs(previous), eta)
    if direction == "higher":
        return (best - previous) / denom
    return (previous - best) / denom


def budget_q(direction: str, previous, ran_scores: list, epsilon: float, eta: float) -> float:
    if not ran_scores:
        return 0.0
    return quality_q(direction, previous, best_score(direction, ran_scores), epsilon, eta)


def update_epsilon(epsilon: float, epsilon_0: float, previous, noise) -> float:
    if previous is None or noise is None:
        return epsilon
    gap = abs(noise - previous)
    return max(epsilon_0, (epsilon + gap) / 2)


def update_rho(rho: float, q: float, flat_streak: int, search) -> tuple[float, int]:
    """The only rho update. A flat budget may add delta once, inside this same update."""
    rho = clip(rho - search.lam * q, search.rho_min, search.rho_max)
    if q == 0:
        flat_streak += 1
    else:
        flat_streak = 0
    if flat_streak >= search.m:
        rho = clip(rho + search.delta, search.rho_min, search.rho_max)
    return rho, flat_streak


def budget_verdict(judged: list[Verdict], q: float) -> str:
    if any(v == "better" for v in judged):
        return "better"
    if q < 0:
        return "worse"
    return "flat"


def seed_attention(
    incumbent_cases,
    candidate_cases=None,
    direction: str = "higher",
    subject: str = "",
    adopted: bool = False,
) -> str:
    """Prompt text only. Official replacement still uses the mean score."""
    incumbent = _seed_map(incumbent_cases)
    candidate = _seed_map(candidate_cases)
    if not incumbent and not candidate:
        return ""
    if adopted and candidate:
        current = candidate
    elif incumbent:
        current = incumbent
    else:
        current = candidate
    low = _low_seeds(current, direction)
    signs: dict[int, str] = {}
    focused: set[int] = set()
    for seed in set(incumbent) & set(candidate):
        signs[seed] = _seed_sign(direction, incumbent[seed], candidate[seed])
        if signs[seed] == "−":
            focused.add(seed)
    focused.update(low)
    seeds = sorted(set(incumbent) | set(candidate))
    title = "逐 seed 符号（写两遍的是 − 和仍低的 seed；不进入官方得分，不参与替换）"
    if subject:
        title += f"，对照 {subject}"
    lines = [title + "："]
    if not focused:
        lines.append(f"{len(seeds)} 个 seed 没有 −，也没有仍低的，合并为一行。")
        return "\n".join(lines)
    improved = 0
    unchanged = 0
    for seed in seeds:
        if seed in focused:
            continue
        sign = signs.get(seed)
        if sign == "+":
            improved += 1
        else:
            unchanged += 1
    for seed in sorted(focused):
        line = _seed_line(seed, incumbent, candidate, signs, low)
        lines.append(line)
        lines.append(line)
    rest = improved + unchanged
    if rest:
        parts = []
        if improved:
            parts.append(f"{improved} 个变好")
        if unchanged:
            parts.append(f"{unchanged} 个相同")
        lines.append(f"其余 {rest} 个 seed 合并：" + "，".join(parts) + "。")
    return "\n".join(lines)


def _seed_map(cases) -> dict[int, float]:
    found: dict[int, float] = {}
    if not cases:
        return found
    for row in cases:
        if not isinstance(row, dict):
            continue
        try:
            found[int(row["seed"])] = float(row["score"])
        except (KeyError, TypeError, ValueError):
            continue
    return found


def _low_seeds(scores: dict[int, float], direction: str) -> set[int]:
    if not scores:
        return set()
    if len(scores) == 1:
        return set(scores)
    values = list(scores.values())
    if min(values) == max(values):
        return set()
    keep = max(1, len(scores) // 3)
    if direction == "higher":
        ranked = sorted(scores, key=lambda seed: (scores[seed], seed))
        cutoff = scores[ranked[keep - 1]]
        return {seed for seed, score in scores.items() if score <= cutoff}
    ranked = sorted(scores, key=lambda seed: (-scores[seed], seed))
    cutoff = scores[ranked[keep - 1]]
    return {seed for seed, score in scores.items() if score >= cutoff}


def _seed_sign(direction: str, previous: float, candidate: float) -> str:
    if candidate == previous:
        return "0"
    if direction == "higher":
        return "+" if candidate > previous else "−"
    return "+" if candidate < previous else "−"


def _seed_line(seed: int, incumbent: dict[int, float], candidate: dict[int, float], signs: dict[int, str], low: set[int]) -> str:
    parts = [f"seed {seed}:"]
    if seed in incumbent:
        parts.append(f"现任 {_num(incumbent[seed])}")
    if seed in candidate:
        parts.append(f"候选 {_num(candidate[seed])}")
    if seed in signs:
        parts.append(signs[seed])
    elif seed in low:
        parts.append("仍低")
    return " ".join(parts)


def _num(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"
