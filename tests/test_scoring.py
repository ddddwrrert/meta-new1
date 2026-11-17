from meta_new.config import SearchConfig
from meta_new.loop.scoring import (
    best_score,
    budget_q,
    iround,
    judge,
    mix_counts,
    quality_q,
    seed_attention,
    update_epsilon,
    update_rho,
)
from meta_new.runtime.evolve_eval import _cases
from meta_new.runtime.task_runner import fitness_of


def test_direction_and_failures():
    assert judge("higher", 10, 12, 1) == "better"
    assert judge("higher", 10, 8, 1) == "worse"
    assert judge("higher", 10, 10.5, 1) == "flat"
    assert judge("lower", 10, 8, 1) == "better"
    assert judge("lower", 10, 12, 1) == "worse"
    assert judge("lower", 10, 10.2, 1) == "flat"
    assert judge("higher", 10, None, 1) == "worse"
    assert judge("higher", None, 4, 1) == "better"
    assert judge("higher", None, None, 1) == "flat"


def test_q_uses_best_candidate_not_acceptance():
    assert quality_q("higher", 90, 85, 1, 1) < 0
    assert quality_q("higher", 90, 90.2, 1, 1) == 0
    assert quality_q("higher", 10, None, 1, 1) == -1
    assert quality_q("higher", None, 10, 1, 1) == 1
    assert quality_q("lower", 10, 7, 0.1, 1) == (10 - 7) / 10
    assert budget_q("higher", 10, [], 0.1, 1) == 0
    assert best_score("lower", [5, 3, None]) == 3
    assert best_score("higher", [None, None]) is None


def test_epsilon_ignores_a_failed_control_and_can_shrink_toward_noise():
    assert update_epsilon(4, 1, 10, None) == 4
    assert update_epsilon(4, 1, None, 8) == 4
    assert update_epsilon(4, 1, 10, 10) == 2
    assert update_epsilon(1, 1, 10, 10) == 1


def test_rho_updates_once_per_formula():
    search = SearchConfig(n=4, rho_0=0.5, rho_min=0.1, rho_max=0.9, lam=0.5, eta=1, delta=0.1, m=2, epsilon_0=0.1)
    down, streak = update_rho(0.5, 0.5, 0, search)
    assert abs(down - 0.25) < 1e-9 and streak == 0
    up, streak = update_rho(0.5, -0.2, 0, search)
    assert abs(up - 0.6) < 1e-9 and streak == 0
    rho, streak = update_rho(0.5, 0, 0, search)
    assert rho == 0.5 and streak == 1
    rho, streak = update_rho(rho, 0, streak, search)
    assert abs(rho - 0.6) < 1e-9 and streak == 2
    capped, _ = update_rho(0.9, -1, 0, search)
    assert capped == 0.9


def test_seed_table_repeats_drops_and_low_seeds_without_changing_the_mean():
    incumbent = [{"seed": i, "score": score} for i, score in enumerate([90, 80, 70, 60, 20, 10])]
    candidate = [{"seed": i, "score": score} for i, score in enumerate([40, 80, 70, 60, 20, 10])]
    text = seed_attention(incumbent, candidate, "higher", subject="drop_seed", adopted=False)
    assert text.count("seed 0:") == 2
    assert "−" in text
    assert text.count("seed 4:") == 2
    assert text.count("seed 5:") == 2
    assert "seed 1:" not in text
    assert "合并" in text
    assert "不参与替换" in text
    assert judge("higher", 55, 40, 1e-6) == "worse"
    assert fitness_of({"combined_score": 55, "cases": candidate}) == 55
    assert _cases({"cases": candidate})[0] == {"seed": 0, "score": 40.0}
    assert _cases({"cases": "nope"}) is None


def test_adopted_table_follows_the_new_low_seeds():
    incumbent = [{"seed": i, "score": score} for i, score in enumerate([90, 80, 70, 60, 20, 10])]
    candidate = [{"seed": i, "score": score} for i, score in enumerate([90, 80, 70, 60, 95, 96])]
    text = seed_attention(incumbent, candidate, "higher", subject="lift", adopted=True)
    assert "seed 4:" not in text
    assert "seed 5:" not in text
    assert text.count("seed 2:") == 2
    assert text.count("seed 3:") == 2
    assert "变好" in text


def test_equal_seeds_collapse_to_one_line():
    same = [{"seed": i, "score": 5} for i in range(6)]
    text = seed_attention(same, same, "higher")
    assert "seed 0:" not in text
    assert "合并为一行" in text
    assert seed_attention(None, None) == ""


def test_mix_keeps_both_kinds_when_n_is_at_least_two():
    assert mix_counts(0.8, 8) == (iround(0.8 * 8), 8 - iround(0.8 * 8))
    assert mix_counts(0.1, 8)[0] == 1
    assert mix_counts(0.95, 4) == (3, 1)
