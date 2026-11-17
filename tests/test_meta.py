from meta_new.meta.evolve import Proposal, apply_proposal, meta_trace, neutral_skill, proposal_from_obj


def test_neutral_skill_is_not_vetoed_before_history():
    skill = neutral_skill()
    kept, accepted = apply_proposal(skill, Proposal(allow_all=True), set(), has_history=False)
    assert accepted is True
    assert kept["neutral"] is True
    assert kept["rules"] == []


def test_pass_all_and_block_all_are_rejected_after_history():
    skill = neutral_skill()
    seen = {"a", "b", "c"}
    for proposal in (
        Proposal(allow_all=True),
        Proposal(block_all=True),
        Proposal(touches_strike=True),
        Proposal(add=[{"id": str(i), "edit_class": name, "stance": "avoid", "text": "x"} for i, name in enumerate(seen)]),
    ):
        kept, accepted = apply_proposal(skill, proposal, seen, has_history=True)
        assert accepted is False
        assert kept == skill


def test_default_update_covers_every_measured_class():
    from meta_new.meta.evolve import default_proposal

    proposal = default_proposal(
        [
            {
                "edit_class": "ratio",
                "ran": True,
                "accepted": True,
                "verdict": "better",
                "decision": "probe",
                "better_count": 1,
                "worse_count": 0,
            },
            {
                "edit_class": "heavy",
                "ran": True,
                "accepted": False,
                "verdict": "worse",
                "decision": "probe",
                "better_count": 0,
                "worse_count": 1,
            },
        ],
        {"neutral": True, "rules": [{"id": "act-heavy", "edit_class": "heavy", "stance": "act", "text": "old"}]},
    )
    stances = {rule["edit_class"]: rule["stance"] for rule in proposal.add}
    assert "ratio" not in stances
    assert stances == {"heavy": "hold"}
    assert "act-heavy" in proposal.delete


def test_one_better_removes_an_act_rule_instead_of_keeping_it():
    from meta_new.meta.evolve import default_proposal

    proposal = default_proposal(
        [
            {
                "edit_class": "ratio",
                "ran": True,
                "accepted": True,
                "verdict": "better",
                "decision": "probe",
                "better_count": 1,
                "worse_count": 0,
            }
        ],
        {"neutral": False, "rules": [{"id": "act-ratio", "edit_class": "ratio", "stance": "act", "text": "once"}]},
    )
    assert proposal.add == []
    assert proposal.delete == ["act-ratio"]


def test_a_single_measured_note_is_kept():
    skill = neutral_skill()
    proposal = Proposal(add=[{"id": "act-ratio", "edit_class": "ratio", "stance": "act", "text": "measured"}])
    kept, accepted = apply_proposal(skill, proposal, {"ratio"}, has_history=True)
    assert accepted is True
    assert kept["neutral"] is False
    assert kept["rules"][0]["edit_class"] == "ratio"


def test_meta_trace_keeps_measured_rows_and_writes_no_score_for_avoid():
    trace = meta_trace(
        [
            {
                "edit_class": "swap",
                "decision": "act",
                "ran": True,
                "verdict": "better",
                "score": 3,
                "accepted": True,
                "budget_verdict": "better",
            },
            {
                "edit_class": "bad",
                "decision": "avoid",
                "ran": False,
                "score": None,
                "verdict": None,
                "budget_verdict": "better",
            },
            {
                "edit_class": "held",
                "decision": "hold",
                "ran": True,
                "verdict": "worse",
                "score": 1,
                "accepted": False,
                "budget_verdict": "better",
            },
        ]
    )
    assert trace["success"][0]["kind"] == "act-accepted"
    assert trace["success"][1]["kind"] == "hold-measured"
    assert trace["success"][1]["score"] == 1
    assert trace["avoid"] == [{"edit_class": "bad", "score": None}]
    assert trace["mistakes"] == []


def test_unaccepted_worse_budget_is_a_misrelease():
    trace = meta_trace(
        [
            {
                "edit_class": "swap",
                "decision": "act",
                "ran": True,
                "verdict": "worse",
                "score": 1,
                "accepted": False,
                "budget_verdict": "worse",
            }
        ]
    )
    assert trace["mistakes"] == [{"kind": "mis-release", "detail": "no candidate accepted and the budget is worse"}]


def test_parser_rejects_a_rewritten_strike():
    proposal = proposal_from_obj({"strike": 1, "add": []})
    assert proposal.touches_strike is True
    star = proposal_from_obj({"add": [{"edit_class": "*", "stance": "avoid"}]})
    assert star.block_all is True
