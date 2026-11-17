import json
from pathlib import Path

from meta_new.config import load_config
from meta_new.generation.verify import verify
from meta_new.loop.task import run_task
from meta_new.models import Candidate
from meta_new.problems.evolve import EvolveTaskProblem, resolve_task_dir
from meta_new.runtime.evolve_eval import EvolveTaskEvaluator, to_wsl

ALE = Path(__file__).resolve().parents[1] / "tasks" / "ale_bench"
ORGANON = Path(__file__).resolve().parents[2] / "code" / "organon" / "evolve-tasks" / "ale_bench"
IDS = [
    "ahc008",
    "ahc011",
    "ahc015",
    "ahc016",
    "ahc024",
    "ahc025",
    "ahc026",
    "ahc027",
    "ahc039",
    "ahc046",
]

_BASE = """#EVOLVE_START
#include <iostream>
int main() {
    std::cout << "base\\n";
    return 0;
}
#EVOLVE_END
"""

_BETTER = """#EVOLVE_START
#include <iostream>
int main() {
    std::cout << "BETTER\\n";
    return 0;
}
#EVOLVE_END
"""

_EVALUATOR = '''
from pathlib import Path

def evaluate(program_path):
    text = Path(program_path).read_text(encoding="utf-8")
    Path("seen.txt").write_text(text, encoding="utf-8")
    score = 4.0 if "BETTER" in text else 1.0
    return {
        "combined_score": score,
        "judge_result": "ACCEPTED",
        "leaked": "#EVOLVE_START" in text or "#EVOLVE_END" in text,
    }
'''


def _task(tmp: Path) -> EvolveTaskProblem:
    root = tmp / "toy"
    (root / "evaluator").mkdir(parents=True)
    (root / "task.json").write_text(
        json.dumps({"program": "solution.cpp", "timeout": 30, "max_retries": 0}),
        encoding="utf-8",
    )
    (root / "solution.cpp").write_text(_BASE, encoding="utf-8")
    (root / "statement.md").write_text("Toy contest. Higher combined_score is better.\n", encoding="utf-8")
    (root / "evaluator" / "evaluator.py").write_text(_EVALUATOR, encoding="utf-8")
    return EvolveTaskProblem(root)


def test_cpp_verification_keeps_a_real_edit_and_drops_a_copy():
    incumbent = "#include <iostream>\nint main(){return 0;} // note\n"
    kept, dropped = verify(
        [
            Candidate("#include <iostream>\nint main(){return 0;}\n", "explore", "copy"),
            Candidate("int helper = 1;\n", "explore", "nomain"),
            Candidate("#include <iostream>\nint main(){return 1;}\n", "explore", "edit"),
        ],
        incumbent,
        "cpp",
    )
    assert [cand.edit_class for cand in kept] == ["edit"]
    reasons = {row["edit_class"]: row["reason"] for row in dropped}
    assert reasons["copy"] == "near_copy"
    assert reasons["nomain"] == "flow"


def test_a_local_cpp_edit_outside_the_markers_is_removed():
    incumbent = "int header = 1;\n#EVOLVE_START\nint main(){return 0;}\n#EVOLVE_END\n"
    changed = "int header = 2;\n#EVOLVE_START\nint main(){return 0;}\n#EVOLVE_END\n"
    kept, dropped = verify(
        [
            Candidate(changed, "local", "outside"),
            Candidate(changed, "explore", "wide"),
        ],
        incumbent,
        "cpp",
    )
    assert [cand.edit_class for cand in kept] == ["wide"]
    assert dropped[0]["reason"] == "not_local"


def test_wsl_paths_keep_the_drive_letter():
    assert to_wsl("/mnt/d/RSI/code") == "/mnt/d/RSI/code"
    converted = to_wsl(Path(r"D:\RSI\code\organon"))
    assert converted == "/mnt/d/RSI/code/organon"


def test_the_task_evaluator_scores_solution_cpp_and_strips_markers(tmp_path):
    problem = _task(tmp_path)
    assert problem.language == "cpp"
    assert problem.baseline_code() == _BASE
    outcome = EvolveTaskEvaluator(problem).run_many({"baseline": problem.baseline_code()})["baseline"]
    assert outcome.status == "ok"
    assert outcome.score == 1.0
    seen = (problem.task_dir / "seen.txt").read_text(encoding="utf-8")
    assert "#EVOLVE_START" not in seen
    assert "#EVOLVE_END" not in seen
    assert "base" in seen


def test_a_better_cpp_candidate_replaces_the_baseline(tmp_path):
    from meta_new.config import SearchConfig

    problem = _task(tmp_path)

    class Once:
        def propose(self, ctx):
            assert ctx.language == "cpp"
            assert "solution" not in ctx.incumbent_code or "#include" in ctx.incumbent_code
            return [Candidate(_BETTER, "explore", "bias")]

    state = run_task(
        name="toy",
        problem=problem,
        instance_seed=1,
        search=SearchConfig(n=2, epsilon_0=0.01),
        budgets=2,
        evaluator=EvolveTaskEvaluator(problem),
        generator=Once(),
        run_dir=tmp_path / "run",
    )
    assert state.official == 4.0
    saved = (tmp_path / "run" / "toy" / "incumbent.cpp").read_text(encoding="utf-8")
    assert "BETTER" in saved
    assert not (tmp_path / "run" / "toy" / "incumbent.py").exists()


def test_ale_bench_config_uses_each_solution_cpp(monkeypatch):
    monkeypatch.setenv("META_EVAL_PYTHON", "/opt/ale-bench/bin/python")
    cfg = load_config("configs/ale_bench.yaml")
    assert cfg.evaluator.via == "wsl"
    assert cfg.evaluator.distro == "Ubuntu-24.04"
    assert cfg.evaluator.python == "/opt/ale-bench/bin/python"
    assert cfg.max_budgets == 100
    assert [task.name for task in cfg.tasks] == IDS
    for task in cfg.tasks:
        root = resolve_task_dir("configs/ale_bench.yaml", task.params["task_dir"])
        assert root == (ALE / task.name).resolve()
        problem = EvolveTaskProblem(root)
        disk = (root / "solution.cpp").read_text(encoding="utf-8")
        evaluator = (root / "evaluator" / "evaluator.py").read_text(encoding="utf-8")
        assert problem.program_name == "solution.cpp"
        assert problem.language == "cpp"
        assert problem.direction == "max"
        assert problem.baseline_code() == disk
        assert "main" in disk
        assert len(problem.statement) > 200
        assert "num_public_cases = 50" in evaluator
        origin = ORGANON / task.name / "solution.cpp"
        if origin.is_file():
            assert disk == origin.read_text(encoding="utf-8")
