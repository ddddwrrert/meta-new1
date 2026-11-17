from meta_new.models import Candidate
from meta_new.generation.verify import verify


def test_syntax_near_copy_and_top_level_crash_are_removed():
    kept, dropped = verify(
        [
            Candidate("def (\n", "local", "bad"),
            Candidate("raise RuntimeError('boom')\n", "local", "boom"),
            Candidate("x = 1\n", "local", "ok"),
            Candidate("y = 0\n", "local", "copy"),
        ],
        "y = 0\n",
    )
    assert [cand.edit_class for cand in kept] == ["ok"]
    reasons = {row["edit_class"]: row["reason"] for row in dropped}
    assert reasons["bad"].startswith("syntax")
    assert reasons["boom"].startswith("crash")
    assert reasons["copy"] == "near_copy"
    same, _ = verify([Candidate("y = 0  # note\n", "local", "comment")], "y = 0\n")
    assert same == []


def test_a_solver_missing_its_entry_is_a_flow_failure():
    kept, dropped = verify(
        [Candidate("x = 1\n", "local", "broken")],
        "def solve(n):\n    return n\n",
    )
    assert kept == []
    assert dropped[0]["reason"] == "flow"
    kept_main, dropped_main = verify(
        [Candidate("x = 1\n", "local", "nomain")],
        "def main():\n    return 1\n",
    )
    assert kept_main == []
    assert dropped_main[0]["reason"] == "flow"


def test_a_one_token_difference_is_not_a_near_copy():
    kept, dropped = verify([Candidate("y = 1\n", "local", "near")], "y = 0\n")
    assert [cand.edit_class for cand in kept] == ["near"]
    assert dropped == []


def test_a_local_edit_outside_solve_is_removed_and_an_explore_edit_is_not():
    incumbent = "HEADER = 1\n# --- solve begin ---\ndef solve():\n    return 1\n# --- solve end ---\n"
    changed = "HEADER = 2\n# --- solve begin ---\ndef solve():\n    return 1\n# --- solve end ---\n"
    kept, dropped = verify(
        [
            Candidate(changed, "local", "outside"),
            Candidate(changed, "explore", "wide"),
        ],
        incumbent,
    )
    assert [cand.edit_class for cand in kept] == ["wide"]
    assert dropped[0]["reason"] == "not_local"
