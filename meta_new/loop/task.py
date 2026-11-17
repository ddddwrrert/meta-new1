"""Task loop for one problem.

Budget 0 runs the baseline, records S0, and does not judge or update the meta skill.
The first mutation happens before budget 1. Later budgets judge candidates against
the previous official score with the epsilon from before the control run.
Rho is updated once after those scores. The next mutation uses that rho and does
not update it again. A better, worse, or flat budget all continue while budget remains.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import asdict
from pathlib import Path

from ..generation.verify import dedup, verify
from ..meta import gate
from ..meta.evolve import apply_proposal, default_proposal, meta_trace, neutral_skill, skill_text
from ..models import ProblemState, ProposeContext
from ..util import sha256_text, write_json
from .scoring import (
    budget_q,
    budget_verdict,
    direction_name,
    judge,
    mix_counts,
    seed_attention,
    select_winner,
    update_epsilon,
    update_rho,
)


def run_task(
    *,
    name: str,
    problem,
    instance_seed: int,
    search,
    budgets: int,
    evaluator,
    generator,
    run_dir: str | Path,
    propose_meta=None,
) -> ProblemState:
    instance = problem.generate(instance_seed)
    if hasattr(evaluator, "bind"):
        evaluator.bind(instance)
    code = problem.baseline_code()
    state = ProblemState(
        name=name,
        direction=direction_name(problem.direction),
        instance=instance,
        incumbent_code=code,
        incumbent_sha=sha256_text(code),
        official=None,
        epsilon=search.epsilon_0,
        rho=search.rho_0,
        flat_streak=0,
        budget=0,
        skill=neutral_skill(),
        history={},
        blacklist=[],
        score_history=[],
    )
    root = Path(run_dir) / name
    root.mkdir(parents=True, exist_ok=True)
    baseline = evaluator.run_many({"baseline": code})["baseline"]
    state.official = baseline.score
    state.score_history.append(baseline.score)
    baseline_traj = _traj(instance, baseline)
    seed_note = seed_attention(getattr(baseline, "cases", None), direction=state.direction)
    record = {
        "budget": 0,
        "kind": "baseline",
        "score": baseline.score,
        "status": baseline.status,
        "verdict": None,
        "rho": state.rho,
        "rho_updated": False,
        "epsilon": state.epsilon,
        "meta_updated": False,
        "trajectory": baseline_traj,
    }
    shown = baseline_traj
    if seed_note:
        record["seed_note"] = seed_note
        shown = {**baseline_traj, "seed_note": seed_note}
    write_json(root / "budget_0000.json", record)
    _save(root, state, problem)
    _say(state, "baseline", None, 0)
    if budgets <= 1:
        return state

    meta_fn = propose_meta or (lambda records, skill: default_proposal(records, skill))
    pending = generator.propose(_context(state, problem, search, "first", None, shown))
    excerpt = shown
    for index in range(1, budgets):
        pending, excerpt = _execute(state, problem, search, evaluator, pending, root, meta_fn, excerpt, generator)
        _save(root, state, problem)
        if index < budgets - 1:
            pending = generator.propose(_context(state, problem, search, "next", excerpt.get("budget_verdict"), excerpt))
    return state


def _execute(state, problem, search, evaluator, candidates, root: Path, meta_fn, excerpt, generator) -> tuple[list, dict]:
    epsilon_used = state.epsilon
    official_before = state.official
    rho_before = state.rho
    language = getattr(problem, "language", "python")
    unique, dup_dropped = dedup(candidates, state.incumbent_code, language)
    prompt_dropped: list[dict] = []
    screen = getattr(generator, "screen", None)
    if callable(screen):
        if _accepts_language(screen):
            unique, prompt_dropped = screen(unique, state.incumbent_code, language)
        else:
            unique, prompt_dropped = screen(unique, state.incumbent_code)
    checked, bad = verify(unique, state.incumbent_code, language)
    admitted = []
    skipped = []
    for cand in checked:
        decision = gate.decide(state, cand.edit_class)
        row = _cand_row(cand, decision)
        if decision["run"]:
            admitted.append((cand, row))
        else:
            row["ran"] = False
            row["score"] = None
            row["verdict"] = None
            skipped.append(row)
    codes = {f"c{i}": cand.code for i, (cand, _) in enumerate(admitted)}
    codes["control"] = state.incumbent_code
    outcomes = evaluator.run_many(codes)
    noise = outcomes["control"]
    judged = []
    ran_scores = []
    for i, (cand, row) in enumerate(admitted):
        outcome = outcomes[f"c{i}"]
        verdict = judge(state.direction, official_before, outcome.score, epsilon_used)
        row.update(
            {
                "ran": True,
                "status": outcome.status,
                "score": outcome.score,
                "verdict": verdict,
                "accepted": False,
                "trajectory": _traj(state.instance, outcome),
            }
        )
        judged.append((i, verdict, outcome.score))
        ran_scores.append(outcome.score)
        counted = gate.record(state, cand.edit_class, verdict)
        row["worse_count"] = counted["worse"]
        row["better_count"] = counted["better"]
    winner = select_winner(state.direction, judged)
    replaced = winner is not None
    focus = winner if winner is not None else _best_ran(state.direction, admitted, outcomes)
    focus_cases = None
    focus_name = ""
    if focus is not None:
        focus_cases = getattr(outcomes[f"c{focus}"], "cases", None)
        focus_name = admitted[focus][0].edit_class
    seed_note = seed_attention(
        getattr(noise, "cases", None),
        focus_cases,
        state.direction,
        subject=focus_name,
        adopted=replaced,
    )
    if replaced:
        cand, row = admitted[winner]
        row["accepted"] = True
        state.incumbent_code = cand.code
        state.incumbent_sha = sha256_text(cand.code)
        state.official = row["score"]
        gate.clear_incumbent_memory(state)
    records = [row for _, row in admitted] + skipped
    q = budget_q(state.direction, official_before, ran_scores, epsilon_used, search.eta)
    state.epsilon = update_epsilon(epsilon_used, search.epsilon_0, official_before, noise.score)
    state.rho, state.flat_streak = update_rho(state.rho, q, state.flat_streak, search)
    seen = {row["edit_class"] for row in records if row.get("ran")}
    conclusion = budget_verdict([row["verdict"] for _, row in admitted if row.get("verdict")], q)
    for row in records:
        row["budget_verdict"] = conclusion
        row["incumbent_replaced"] = replaced
    trace = meta_trace(records)
    proposal = meta_fn(records, state.skill)
    state.skill, meta_ok = apply_proposal(state.skill, proposal, seen, has_history=True)
    state.budget += 1
    state.score_history.append(state.official)
    payload = {
        "budget": state.budget,
        "kind": "probe",
        "epsilon_used": epsilon_used,
        "epsilon_after": state.epsilon,
        "noise": {"score": noise.score, "status": noise.status},
        "official_before": official_before,
        "official_after": state.official,
        "q": q,
        "rho_before": rho_before,
        "rho_after": state.rho,
        "rho_updated": True,
        "flat_streak": state.flat_streak,
        "budget_verdict": conclusion,
        "candidates": records,
        "dropped": dup_dropped + prompt_dropped + bad,
        "meta_trace": trace,
        "meta_accepted": meta_ok,
        "incumbent_sha": state.incumbent_sha,
    }
    if seed_note:
        payload["seed_note"] = seed_note
    write_json(root / f"budget_{state.budget:04d}.json", payload)
    _say(state, conclusion, q, len(admitted))
    summary = {
        "budget_verdict": conclusion,
        "budget": state.budget,
        "previous_official": official_before,
        "candidate_scores": _score_rows(records),
        "text": _excerpt(state, records, official_before, conclusion, seed_note),
    }
    return [], summary


def _context(state: ProblemState, problem, search, phase: str, verdict, excerpt) -> ProposeContext:
    n_explore, n_local = mix_counts(state.rho, search.n)
    budget_index = 0
    previous_official = None
    candidate_scores: list = []
    if isinstance(excerpt, str):
        text = excerpt
    elif isinstance(excerpt, dict) and "text" in excerpt:
        text = excerpt["text"]
        budget_index = int(excerpt.get("budget") or 0)
        previous_official = excerpt.get("previous_official")
        candidate_scores = list(excerpt.get("candidate_scores") or [])
    else:
        text = json.dumps(excerpt, ensure_ascii=False)
    return ProposeContext(
        phase=phase,
        problem_name=getattr(problem, "name", state.name),
        statement=getattr(problem, "statement", ""),
        direction=state.direction,
        incumbent_code=state.incumbent_code,
        rho=state.rho,
        n=search.n,
        n_explore=n_explore,
        n_local=n_local,
        verdict=verdict,
        official=state.official,
        repeated_worse=gate.repeated_worse(state),
        history=gate.history_rows(state),
        skill_text=skill_text(state.skill),
        trajectory_excerpt=text,
        budget_index=budget_index,
        previous_official=previous_official,
        candidate_scores=candidate_scores,
        language=getattr(problem, "language", "python"),
    )


def _accepts_language(screen) -> bool:
    try:
        signature = inspect.signature(screen)
    except (TypeError, ValueError):
        return False
    return any(
        name == "language" or param.kind == inspect.Parameter.VAR_POSITIONAL
        for name, param in signature.parameters.items()
    )


def _cand_row(cand, decision) -> dict:
    return {
        "edit_class": cand.edit_class,
        "kind": cand.kind,
        "decision": decision["action"],
        "criterion": decision.get("by"),
        "rule_id": decision.get("rule_id"),
        "hypothesis": cand.hypothesis,
        "origin": cand.origin,
    }


def _traj(instance: str, outcome) -> dict:
    status = outcome.status
    return {
        "input": instance,
        "reasoning_steps": [],
        "tool_calls": [],
        "output": outcome.stdout or "",
        "stderr": outcome.stderr or "",
        "elapsed": outcome.elapsed,
        "anomalies": [] if status == "ok" else [status],
        "contradictions": [],
    }


def _score_rows(records: list[dict]) -> list[dict]:
    rows = []
    for rec in records:
        row = {"edit_class": rec.get("edit_class"), "ran": bool(rec.get("ran"))}
        if rec.get("ran"):
            row["score"] = rec.get("score")
            row["verdict"] = rec.get("verdict")
        rows.append(row)
    return rows


def _best_ran(direction: str, admitted, outcomes) -> int | None:
    found = None
    best = None
    for index, _item in enumerate(admitted):
        score = outcomes[f"c{index}"].score
        if score is None:
            continue
        if best is None or (score > best if direction == "higher" else score < best):
            best = score
            found = index
    return found


def _excerpt(state: ProblemState, records: list[dict], official_before, conclusion: str, seed_note: str = "") -> str:
    lines = [f"budget={state.budget}", f"conclusion={conclusion}", f"S_prev={official_before}"]
    for rec in records:
        if rec.get("ran"):
            lines.append(f"S_cand {rec['edit_class']}={rec.get('score')} {rec.get('verdict')}")
            traj = rec.get("trajectory") or {}
            brief = {
                key: traj.get(key)
                for key in (
                    "input",
                    "reasoning_steps",
                    "tool_calls",
                    "output",
                    "stderr",
                    "elapsed",
                    "anomalies",
                    "contradictions",
                )
            }
            lines.append(json.dumps(brief, ensure_ascii=False))
        else:
            lines.append(f"{rec['edit_class']} 未运行")
    if seed_note:
        lines.append(seed_note)
    return "\n".join(lines)


def _save(root: Path, state: ProblemState, problem=None) -> None:
    data = asdict(state)
    data.pop("incumbent_code", None)
    data.pop("instance", None)
    write_json(root / "state.json", data)
    program = getattr(problem, "program_name", "solver.py")
    suffix = Path(program).suffix or ".py"
    (root / f"incumbent{suffix}").write_text(state.incumbent_code, encoding="utf-8")
    (root / "instance.txt").write_text(state.instance, encoding="utf-8")


def _say(state: ProblemState, verdict, q, admitted: int) -> None:
    print(
        f"[{state.name}] budget {state.budget} official={state.official} "
        f"verdict={verdict} q={q} rho={state.rho:.3f} epsilon={state.epsilon:.4g} admitted={admitted}",
        flush=True,
    )
