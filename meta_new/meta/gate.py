"""ACT / HOLD / AVOID for one problem and one incumbent.

The gate reads this problem's current criterion.
No run on the current incumbent means one probe, unless this incumbent
already blacklists that class.
Count rules still bound the criterion: a worse count that has not exceeded
3 still runs, and the run that makes the count exceed 3 blacklists this
incumbent only. A flat result does not change the worse count.
A better result clears it.
Replacing the incumbent drops the old blacklist, the old counts, and avoid rules.
"""

from __future__ import annotations

WORSE_LIMIT = 3


def _norm(edit_class: str) -> str:
    name = (edit_class or "unlabeled").strip().lower().replace(" ", "_")
    return name or "unlabeled"


def class_key(incumbent_sha: str, edit_class: str) -> str:
    return f"{incumbent_sha}|{_norm(edit_class)}"


def _rule(state, edit_class: str):
    name = _norm(edit_class)
    found = None
    for rule in state.skill.get("rules") or []:
        if isinstance(rule, dict) and _norm(str(rule.get("edit_class", ""))) == name:
            found = rule
    return found


def class_name(key: str) -> str:
    return key.split("|", 1)[1]


def decide(state, edit_class: str) -> dict:
    """Use the current criterion where the count rules leave a choice.

    Blacklist and a worse count above 3 do not run.
    No history runs once.
    An outstanding worse that has not exceeded 3 is HOLD and still runs.
    Two or more betters with no outstanding worse are ACT and run.
    A criterion that says avoid before that fourth worse does not skip the run.
    """
    key = class_key(state.incumbent_sha, edit_class)
    rec = state.history.get(key)
    worse = 0 if rec is None else int(rec["worse"])
    better = 0 if rec is None else int(rec["better"])
    runs = 0 if rec is None else int(rec["runs"])
    rule = _rule(state, edit_class)
    rule_id = None if rule is None else rule.get("id")
    stance = None if rule is None else rule.get("stance")
    base = {"worse": worse, "key": key, "rule_id": rule_id}
    if key in state.blacklist or worse > WORSE_LIMIT:
        return {**base, "action": "avoid", "run": False, "by": "count"}
    if runs == 0:
        return {**base, "action": "probe", "run": True, "by": "no-history"}
    if worse >= 1:
        return {**base, "action": "hold", "run": True, "by": "skill" if stance == "hold" else "step3"}
    if better >= 2:
        return {**base, "action": "act", "run": True, "by": "skill" if stance == "act" else "step3"}
    # One better, or only flats, is not ACT. A criterion that says otherwise does not rename it.
    return {**base, "action": "pass", "run": True, "by": "step3"}


def record(state, edit_class: str, verdict: str) -> dict:
    key = class_key(state.incumbent_sha, edit_class)
    rec = state.history.setdefault(key, {"runs": 0, "worse": 0, "better": 0, "last": ""})
    rec["runs"] += 1
    rec["last"] = verdict
    if verdict == "worse":
        rec["worse"] += 1
        if rec["worse"] > WORSE_LIMIT and key not in state.blacklist:
            state.blacklist.append(key)
    elif verdict == "better":
        rec["worse"] = 0
        rec["better"] += 1
        if key in state.blacklist:
            state.blacklist.remove(key)
    return rec


def clear_incumbent_memory(state) -> None:
    """Old blacklist, avoid rules, and worse counts expire with the incumbent code."""
    state.history = {}
    state.blacklist = []
    rules = [r for r in (state.skill or {}).get("rules", []) if r.get("stance") != "avoid"]
    state.skill = {"neutral": not rules, "rules": rules}


def repeated_worse(state) -> list[str]:
    """Classes whose worse count is more than one. A single worse is not repeated."""
    prefix = state.incumbent_sha + "|"
    names = []
    for key, rec in state.history.items():
        if key.startswith(prefix) and int(rec["worse"]) > 1:
            names.append(class_name(key))
    return names


def history_rows(state) -> list[dict]:
    """Measured up/down counts for the current incumbent."""
    prefix = state.incumbent_sha + "|"
    rows = []
    for key, rec in state.history.items():
        if not key.startswith(prefix):
            continue
        rows.append(
            {
                "edit_class": class_name(key),
                "runs": int(rec["runs"]),
                "worse": int(rec["worse"]),
                "better": int(rec["better"]),
                "last": rec.get("last", ""),
            }
        )
    return rows
