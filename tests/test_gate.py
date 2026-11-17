from meta_new.meta.evolve import neutral_skill
from meta_new.meta.gate import WORSE_LIMIT, clear_incumbent_memory, decide, record, repeated_worse
from meta_new.models import ProblemState


def _state(sha="inc"):
    return ProblemState(
        name="p",
        direction="higher",
        instance="x",
        incumbent_code="code",
        incumbent_sha=sha,
        official=10,
        epsilon=0.1,
        rho=0.5,
        flat_streak=0,
        budget=1,
        skill=neutral_skill(),
        history={},
        blacklist=[],
        score_history=[10],
    )


def test_hold_allows_reruns_until_the_fourth_worse():
    state = _state()
    assert decide(state, "swap")["action"] == "probe"
    for expected in range(1, WORSE_LIMIT + 1):
        assert decide(state, "swap")["run"] is True
        record(state, "swap", "worse")
        assert state.history["inc|swap"]["worse"] == expected
    assert decide(state, "swap")["run"] is True
    record(state, "swap", "worse")
    assert state.history["inc|swap"]["worse"] == WORSE_LIMIT + 1
    assert decide(state, "swap")["action"] == "avoid"
    assert decide(state, "swap")["run"] is False


def test_flat_does_not_count_and_better_clears_worse():
    state = _state()
    record(state, "swap", "worse")
    record(state, "swap", "flat")
    assert state.history["inc|swap"]["worse"] == 1
    record(state, "swap", "better")
    assert state.history["inc|swap"]["worse"] == 0
    assert state.history["inc|swap"]["better"] == 1
    assert repeated_worse(state) == []
    record(state, "swap", "worse")
    record(state, "swap", "worse")
    assert repeated_worse(state) == ["swap"]


def test_one_flat_without_a_rule_still_runs_and_is_not_act():
    state = _state()
    record(state, "swap", "flat")
    decision = decide(state, "swap")
    assert decision["run"] is True
    assert decision["action"] == "pass"


def test_a_hold_rule_does_not_relabel_a_flat():
    state = _state()
    record(state, "swap", "flat")
    state.skill = {"neutral": False, "rules": [{"id": "h", "edit_class": "swap", "stance": "hold", "text": ""}]}
    decision = decide(state, "swap")
    assert decision["action"] == "pass"
    assert decision["run"] is True


def test_one_better_is_not_act_even_if_the_rule_says_act():
    state = _state()
    record(state, "swap", "better")
    state.skill = {"neutral": False, "rules": [{"id": "a", "edit_class": "swap", "stance": "act", "text": "once"}]}
    decision = decide(state, "swap")
    assert decision["run"] is True
    assert decision["action"] == "pass"


def test_avoid_rule_does_not_skip_before_the_fourth_worse():
    state = _state()
    record(state, "swap", "worse")
    state.skill = {"neutral": False, "rules": [{"id": "a", "edit_class": "swap", "stance": "avoid", "text": ""}]}
    decision = decide(state, "swap")
    assert decision["action"] == "hold"
    assert decision["run"] is True
    state.skill = {"neutral": False, "rules": [{"id": "a", "edit_class": "fresh", "stance": "avoid", "text": ""}]}
    first = decide(state, "fresh")
    assert first["action"] == "probe"
    assert first["run"] is True


def test_repeated_better_stays_act():
    state = _state()
    record(state, "swap", "better")
    record(state, "swap", "better")
    state.skill = {"neutral": False, "rules": [{"id": "h", "edit_class": "swap", "stance": "hold", "text": ""}]}
    decision = decide(state, "swap")
    assert decision["action"] == "act"
    assert decision["run"] is True


def test_blacklist_beats_an_act_rule():
    state = _state()
    for _ in range(WORSE_LIMIT + 1):
        record(state, "swap", "worse")
    state.skill = {"neutral": False, "rules": [{"id": "a", "edit_class": "swap", "stance": "act", "text": "run"}]}
    decision = decide(state, "swap")
    assert decision["action"] == "avoid"
    assert decision["run"] is False


def test_replacement_drops_avoid_rules_and_keeps_the_rest():
    state = _state()
    state.skill = {
        "neutral": False,
        "rules": [
            {"id": "a", "edit_class": "swap", "stance": "avoid", "text": ""},
            {"id": "b", "edit_class": "ratio", "stance": "act", "text": ""},
        ],
    }
    state.blacklist = ["inc|swap"]
    clear_incumbent_memory(state)
    assert state.blacklist == []
    assert state.history == {}
    assert [rule["id"] for rule in state.skill["rules"]] == ["b"]
    assert state.skill["neutral"] is False


def test_blacklist_belongs_to_the_current_incumbent():
    first = _state("aaa")
    second = _state("bbb")
    for _ in range(WORSE_LIMIT + 1):
        record(first, "swap", "worse")
    assert decide(first, "swap")["run"] is False
    assert decide(second, "swap")["run"] is True
    clear_incumbent_memory(first)
    first.incumbent_sha = "ccc"
    assert decide(first, "swap")["action"] == "probe"
    assert first.blacklist == []
