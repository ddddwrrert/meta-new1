from pathlib import Path

from meta_new.config import SandboxConfig
from meta_new.generation.mockgen import MockGenerator
from meta_new.runtime.evaluate import SandboxEvaluator
from meta_new.models import ProposeContext
from meta_new.problems import build_problem


def _eval(problem, instance):
    cfg = SandboxConfig(time_limit_s=0.4, tle_factor=1.0, hard_grace_s=0.4, max_mem_mb=512, iter_budget=200, parallel=2)
    ev = SandboxEvaluator(problem, cfg, seed=0)
    ev.bind(instance)
    return ev


def test_timeout_has_no_score():
    problem = build_problem("knapsack", {"n_min": 8, "n_max": 8})
    ev = _eval(problem, problem.generate(1))
    out = ev.run_many({"c": "import time\ntime.sleep(5)\n"})["c"]
    assert out.status == "timeout"
    assert out.score is None


def test_mock_candidates_run_and_tsp_nearest_is_shorter():
    problem = build_problem("tsp", {"n_min": 20, "n_max": 20})
    instance = problem.generate(1)
    ev = _eval(problem, instance)
    base = problem.baseline_code()
    ctx = ProposeContext(
        phase="first",
        problem_name="tsp",
        statement=problem.statement,
        direction="lower",
        incumbent_code=base,
        rho=0.5,
        n=2,
        n_explore=1,
        n_local=1,
    )
    cands = MockGenerator().propose(ctx)
    assert [c.kind for c in cands] == ["explore", "local"]
    out = ev.run_many({"base": base, **{c.edit_class: c.code for c in cands}})
    assert out["base"].status == "ok"
    assert out["nearest"].status == "ok"
    assert out["nearest"].score < out["base"].score


def test_mock_fills_every_slot_and_keeps_local_edits_inside_solve():
    from meta_new.generation.verify import verify

    problem = build_problem("tsp", {"n_min": 20, "n_max": 20})
    base = problem.baseline_code()
    ctx = ProposeContext(
        phase="first",
        problem_name="tsp",
        statement=problem.statement,
        direction="lower",
        incumbent_code=base,
        rho=0.8,
        n=8,
        n_explore=6,
        n_local=2,
        trajectory_excerpt='{"input":"1","anomalies":[]}',
        repeated_worse=["swap_1"],
    )
    cands = MockGenerator().propose(ctx)
    assert [(c.kind, c.edit_class) for c in cands] == [
        ("explore", "nearest"),
        ("explore", "farthest"),
        ("explore", "start_2"),
        ("explore", "start_3"),
        ("explore", "start_4"),
        ("explore", "start_5"),
        ("local", "swap_1"),
        ("local", "swap_2"),
    ]
    assert len({c.code for c in cands}) == 8
    assert "feature=baseline" in cands[0].hypothesis
    local = next(c for c in cands if c.edit_class == "swap_1")
    assert "must_fix=True" in local.hypothesis
    kept, dropped = verify(cands, base)
    assert dropped == []
    assert len(kept) == 8


def test_source_has_no_held_out_split():
    root = Path(__file__).resolve().parents[1] / "meta_new"
    blob = "\n".join(p.read_text(encoding="utf-8") for p in root.rglob("*.py"))
    assert "train_seed" not in blob
    assert "report_seed" not in blob
