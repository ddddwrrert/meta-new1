import json
from pathlib import Path

from meta_new.config import SearchConfig
from meta_new.loop.task import run_task
from meta_new.meta.evolve import Proposal
from meta_new.runtime.evaluate import Outcome
from meta_new.models import Candidate


class Tiny:
    name = "tiny"
    direction = "max"
    statement = "higher is better"

    def generate(self, seed):
        return "x\n"

    def baseline_code(self):
        return "value = 10\n"


class MapEval:
    def __init__(self, fn):
        self.fn = fn
        self.batches = 0

    def bind(self, instance):
        self.instance = instance

    def run_many(self, codes):
        self.batches += 1
        return {label: self.fn(self.batches, label, code) for label, code in codes.items()}


class Script:
    def __init__(self, factory):
        self.factory = factory
        self.calls = 0

    def propose(self, ctx):
        self.calls += 1
        return self.factory(ctx, self.calls)


def _search(**kwargs):
    base = dict(n=2, rho_0=0.5, rho_min=0.1, rho_max=0.9, lam=0.5, eta=1.0, delta=0.1, m=3, epsilon_0=0.1)
    base.update(kwargs)
    return SearchConfig(**base)


def _run(tmp, fn, factory, budgets=2, **search):
    gen = Script(factory)
    ev = MapEval(fn)
    state = run_task(
        name="tiny",
        problem=Tiny(),
        instance_seed=1,
        search=_search(**search),
        budgets=budgets,
        evaluator=ev,
        generator=gen,
        run_dir=tmp,
    )
    return state, gen, ev


def _budget(tmp, index):
    return json.loads((Path(tmp) / "tiny" / f"budget_{index:04d}.json").read_text(encoding="utf-8"))


def test_budget_zero_does_not_judge(tmp_path):
    state, gen, ev = _run(tmp_path, lambda b, label, code: Outcome("ok", 10), lambda ctx, n: [], budgets=1)
    assert ev.batches == 1
    assert gen.calls == 0
    assert state.official == 10
    assert state.rho == 0.5
    assert state.skill["neutral"] is True
    assert _budget(tmp_path, 0)["verdict"] is None
    assert _budget(tmp_path, 0)["rho_updated"] is False


def test_worse_candidate_raises_rho_without_moving_official(tmp_path):
    def fn(batch, label, code):
        if label == "control":
            return Outcome("ok", 1)
        if "LOW" in code:
            return Outcome("ok", 8)
        return Outcome("ok", 10)

    def factory(ctx, n):
        return [Candidate("LOW = 1\n", "explore", "heavy")]

    state, gen, ev = _run(tmp_path, fn, factory, budgets=2)
    row = _budget(tmp_path, 1)
    assert row["official_after"] == 10
    assert row["noise"]["score"] == 1
    assert row["q"] == -0.2
    assert row["budget_verdict"] == "worse"
    assert state.rho == 0.6
    assert row["candidates"][0]["verdict"] == "worse"
    assert state.epsilon == max(0.1, (0.1 + 9) / 2)


def test_better_candidate_becomes_official_and_lowers_rho(tmp_path):
    def fn(batch, label, code):
        if label == "control":
            return Outcome("ok", 1)
        if "HIGH" in code:
            return Outcome("ok", 15)
        return Outcome("ok", 10)

    state, gen, ev = _run(
        tmp_path,
        fn,
        lambda ctx, n: [Candidate("HIGH = 1\n", "local", "ratio")],
        budgets=3,
    )
    assert _budget(tmp_path, 1)["official_after"] == 15
    assert state.score_history[1] == 15
    assert abs(state.rho - 0.25) < 1e-9
    assert gen.calls == 2
    assert state.skill["neutral"] is False


def test_candidates_use_epsilon_from_before_the_control_update(tmp_path):
    def fn(batch, label, code):
        if batch == 1:
            return Outcome("ok", 10)
        if label == "control":
            return Outcome("ok", 20 if batch == 2 else 10)
        if "mark = 1" in code:
            return Outcome("ok", 10)
        if "mark = 2" in code:
            return Outcome("ok", 13)
        return Outcome("ok", 10)

    def factory(ctx, n):
        body = "mark = 1\n" if n == 1 else "mark = 2\n"
        return [Candidate(body, "local", "nudge")]

    state, gen, ev = _run(tmp_path, fn, factory, budgets=3, epsilon_0=0.1)
    first = _budget(tmp_path, 1)
    second = _budget(tmp_path, 2)
    assert first["budget_verdict"] == "flat"
    assert first["official_after"] == 10
    assert first["epsilon_after"] == max(0.1, (0.1 + 10) / 2)
    assert second["epsilon_used"] == first["epsilon_after"]
    assert second["budget_verdict"] == "flat"
    assert second["official_after"] == 10
    assert second["candidates"][0]["score"] == 13


def test_fourth_worse_blacklists_and_the_next_one_does_not_run(tmp_path):
    def fn(batch, label, code):
        if "BAD" in code:
            return Outcome("ok", 0)
        return Outcome("ok", 10)

    state, gen, ev = _run(
        tmp_path,
        fn,
        lambda ctx, n: [Candidate("BAD = 1\n", "local", "bad")],
        budgets=6,
        epsilon_0=0.1,
    )
    actions = [_budget(tmp_path, i)["candidates"][0] for i in range(1, 6)]
    assert [row["decision"] for row in actions] == ["probe", "hold", "hold", "hold", "avoid"]
    assert actions[3]["ran"] is True and actions[3]["verdict"] == "worse"
    assert actions[4]["ran"] is False and actions[4]["score"] is None
    assert state.official == 10
    assert any(key.endswith("|bad") for key in state.blacklist)


def test_baseline_trajectory_records_the_named_fields(tmp_path):
    _run(tmp_path, lambda b, label, code: Outcome("ok", 10), lambda ctx, n: [], budgets=1)
    traj = _budget(tmp_path, 0)["trajectory"]
    assert set(traj) == {
        "input",
        "reasoning_steps",
        "tool_calls",
        "output",
        "stderr",
        "elapsed",
        "anomalies",
        "contradictions",
    }
    assert traj["reasoning_steps"] == []
    assert traj["tool_calls"] == []
    assert traj["contradictions"] == []
    assert traj["anomalies"] == []


def test_baseline_failure_is_an_anomaly_without_a_score(tmp_path):
    state, _gen, _ev = _run(
        tmp_path,
        lambda b, label, code: Outcome("timeout", None, stderr="slow"),
        lambda ctx, n: [],
        budgets=1,
    )
    assert state.official is None
    traj = _budget(tmp_path, 0)["trajectory"]
    assert traj["anomalies"] == ["timeout"]
    assert traj["stderr"] == "slow"


def test_next_gate_reads_the_updated_criterion(tmp_path):
    def meta(records, skill):
        return Proposal(add=[{"id": "h-same", "edit_class": "same", "stance": "hold", "text": "flat, still runnable"}])

    state = run_task(
        name="tiny",
        problem=Tiny(),
        instance_seed=1,
        search=_search(),
        budgets=3,
        evaluator=MapEval(lambda b, label, code: Outcome("ok", 10)),
        generator=Script(lambda ctx, n: [Candidate(f"z = {n}\n", "local", "same")]),
        run_dir=tmp_path,
        propose_meta=meta,
    )
    first = _budget(tmp_path, 1)["candidates"][0]
    second = _budget(tmp_path, 2)["candidates"][0]
    assert first["decision"] == "probe"
    assert first["criterion"] == "no-history"
    assert second["decision"] == "pass"
    assert second["criterion"] == "step3"
    assert second["ran"] is True
    assert state.official == 10


def test_verifier_drop_is_not_scored(tmp_path):
    class Drop:
        def propose(self, ctx):
            return [Candidate("KEEP = 1\n", "local", "keep")]

        def screen(self, candidates, incumbent):
            return [], [{"edit_class": c.edit_class, "kind": c.kind, "reason": "verifier"} for c in candidates]

    def fn(batch, label, code):
        assert label in ("baseline", "control")
        return Outcome("ok", 10)

    state = run_task(
        name="tiny",
        problem=Tiny(),
        instance_seed=1,
        search=_search(),
        budgets=2,
        evaluator=MapEval(fn),
        generator=Drop(),
        run_dir=tmp_path,
    )
    row = _budget(tmp_path, 1)
    assert row["candidates"] == []
    assert row["dropped"][0]["reason"] == "verifier"
    assert row["q"] == 0
    assert state.official == 10


def test_sign_table_is_prompt_text_and_the_mean_still_decides_replacement(tmp_path):
    incumbent = [{"seed": i, "score": score} for i, score in enumerate([90, 80, 70, 60, 20, 10])]
    dropped = [{"seed": i, "score": score} for i, score in enumerate([40, 80, 70, 60, 20, 10])]
    seen = []

    def fn(batch, label, code):
        if batch == 1 or label == "control" or "NEXT" not in code:
            return Outcome("ok", 55, cases=incumbent)
        return Outcome("ok", 40, cases=dropped)

    def factory(ctx, n):
        seen.append(ctx.trajectory_excerpt)
        return [Candidate("NEXT = 1\n", "explore", "drop_seed")]

    state, _gen, _ev = _run(tmp_path, fn, factory, budgets=3, epsilon_0=1e-6)
    assert state.official == 55
    assert _budget(tmp_path, 1)["official_after"] == 55
    assert _budget(tmp_path, 1)["candidates"][0]["verdict"] == "worse"
    assert seen[1].count("seed 0:") == 2
    assert seen[1].count("seed 4:") == 2
    assert "seed 1:" not in seen[1]
    assert "不参与替换" in seen[1]


def test_flat_and_better_both_keep_generating(tmp_path):
    def fn(batch, label, code):
        return Outcome("ok", 10)

    state, gen, ev = _run(tmp_path, fn, lambda ctx, n: [Candidate(f"z = {n}\n", "explore", "same")], budgets=3)
    assert gen.calls == 2
    assert state.official == 10
    assert _budget(tmp_path, 1)["budget_verdict"] == "flat"
