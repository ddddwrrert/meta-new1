"""Offline stand-in. It rewrites the marked solve function of the current incumbent."""

from __future__ import annotations

import re

from ..models import Candidate, ProposeContext

_SOLVE = re.compile(r"# --- solve begin ---\n.*?# --- solve end ---", re.S)

_TSP_NEAREST = """def solve(n, pts, deadline, iter_budget, rng):
    order = [0]
    left = set(range(1, n))
    while left:
        last = pts[order[-1]]
        nxt = min(left, key=lambda j: dist(last, pts[j]))
        order.append(nxt)
        left.remove(nxt)
    return order"""

_TSP_FARTHEST = """def solve(n, pts, deadline, iter_budget, rng):
    order = [0]
    left = set(range(1, n))
    while left:
        last = pts[order[-1]]
        nxt = max(left, key=lambda j: dist(last, pts[j]))
        order.append(nxt)
        left.remove(nxt)
    return order"""

_KS_RATIO = """def solve(n, cap, items, deadline, iter_budget, rng):
    order = sorted(range(n), key=lambda i: items[i][1] / items[i][0], reverse=True)
    chosen, weight = [], 0
    for i in order:
        if weight + items[i][0] <= cap:
            chosen.append(i)
            weight += items[i][0]
    return chosen"""

_KS_HEAVY = """def solve(n, cap, items, deadline, iter_budget, rng):
    order = sorted(range(n), key=lambda i: items[i][0], reverse=True)
    chosen, weight = [], 0
    for i in order:
        if weight + items[i][0] <= cap:
            chosen.append(i)
            weight += items[i][0]
    return chosen"""


def replace_solve(code: str, body: str) -> str:
    block = "# --- solve begin ---\n" + body.strip() + "\n# --- solve end ---"
    if _SOLVE.search(code):
        return _SOLVE.sub(block, code, count=1)
    return code.rstrip() + "\n\n" + block + "\n"


def _behaviors(ctx: ProposeContext) -> list[dict]:
    """Read this budget's trajectory. Do not invent reasoning steps that were not recorded."""
    text = ctx.trajectory_excerpt or ""
    found = []
    if ctx.verdict:
        found.append({"feature": str(ctx.verdict), "step": "score"})
    if '"anomalies":' in text and '"anomalies": []' not in text and '"anomalies":[]' not in text:
        found.append({"feature": "anomaly", "step": "run"})
    if not found:
        found.append({"feature": "baseline", "step": "solve"})
    return found


def _tsp_explore(index: int) -> tuple[str, str]:
    if index == 0:
        return "nearest", _TSP_NEAREST
    if index == 1:
        return "farthest", _TSP_FARTHEST
    return f"start_{index}", f"""def solve(n, pts, deadline, iter_budget, rng):
    start = {index} % n
    order = [start]
    left = set(range(n))
    left.remove(start)
    while left:
        last = pts[order[-1]]
        nxt = min(left, key=lambda j: dist(last, pts[j]))
        order.append(nxt)
        left.remove(nxt)
    return order"""


def _tsp_local(index: int) -> tuple[str, str]:
    pos = index + 1
    return f"swap_{pos}", f"""def solve(n, pts, deadline, iter_budget, rng):
    order = list(range(n))
    if n > {pos}:
        order[0], order[{pos}] = order[{pos}], order[0]
    return order"""


def _ks_explore(index: int) -> tuple[str, str]:
    if index == 0:
        return "ratio", _KS_RATIO
    if index == 1:
        return "heavy", _KS_HEAVY
    return f"value_{index}", f"""def solve(n, cap, items, deadline, iter_budget, rng):
    order = sorted(range(n), key=lambda i: items[i][1] - {index} * items[i][0], reverse=True)
    chosen, weight = [], 0
    for i in order:
        if weight + items[i][0] <= cap:
            chosen.append(i)
            weight += items[i][0]
    return chosen"""


def _ks_local(index: int) -> tuple[str, str]:
    return f"skip_{index}", f"""def solve(n, cap, items, deadline, iter_budget, rng):
    chosen, weight = [], 0
    for i, (w, v) in enumerate(items):
        if i == {index}:
            continue
        if weight + w <= cap:
            chosen.append(i)
            weight += w
    return chosen"""


def _analyze(ctx: ProposeContext) -> dict:
    return {
        "problem": ctx.problem_name,
        "budget": ctx.budget_index,
        "verdict": ctx.verdict,
        "behaviors": _behaviors(ctx),
    }


def _hypotheses(ctx: ProposeContext, analyzed: dict) -> list[dict]:
    must = set(ctx.repeated_worse)
    found = []
    for behavior in analyzed["behaviors"]:
        found.append(
            {
                "edit_class": "local",
                "kind": "local",
                "change": behavior["feature"],
                "step": behavior["step"],
                "must_fix": False,
                "risk": "",
                "repeated": sorted(must),
            }
        )
    return found


class MockGenerator:
    """Offline stand-in for the four serial roles.

    Analyzer reads the trajectory and does not invent a verdict.
    Critic turns those behaviors into hypotheses. Local candidates are emitted
    only when a hypothesis exists, and only inside the marked solve region.
    Explore candidates are not limited to that fragment. The loop verifier
    then drops illegal candidates.
    """

    def propose(self, ctx: ProposeContext) -> list[Candidate]:
        analyzed = _analyze(ctx)
        hypotheses = _hypotheses(ctx, analyzed)
        must = set(ctx.repeated_worse)
        explore_of, local_of = (_tsp_explore, _tsp_local) if ctx.problem_name != "knapsack" else (_ks_explore, _ks_local)
        out: list[Candidate] = []
        for index in range(ctx.n_explore):
            name, body = explore_of(index)
            out.append(_one(ctx, "explore", name, body, index, hypotheses, must))
        if hypotheses:
            for index in range(ctx.n_local):
                name, body = local_of(index)
                out.append(_one(ctx, "local", name, body, index, hypotheses, must))
        return out


def _one(ctx: ProposeContext, kind: str, name: str, body: str, index: int, hypotheses: list[dict], must: set[str]) -> Candidate:
    edited = replace_solve(ctx.incumbent_code, body)
    change = hypotheses[0]["change"] if hypotheses else "baseline"
    hypothesis = f"{name}; feature={change}; must_fix={name in must}"
    return Candidate(code=edited, kind=kind, edit_class=name, hypothesis=hypothesis, origin=f"{kind}-{index}")
