"""Meta update for one problem.

The numeric strike rule stays in the gate. This step writes the criterion
the next gate reads. A proposal that would pass almost every edit, block
almost every edit, or rewrite the strike count is rejected. The neutral
skill used before any measured history is not subject to that rejection.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Proposal:
    add: list[dict] = field(default_factory=list)
    delete: list[str] = field(default_factory=list)
    allow_all: bool = False
    block_all: bool = False
    touches_strike: bool = False


def meta_trace(records: list[dict]) -> dict:
    """Measured success, mistakes, and avoid entries for this problem only.

    Unrun avoid rows keep a null score. No other problem's scores are included.
    """
    budget_verdict = ""
    if records:
        budget_verdict = records[0].get("budget_verdict") or ""
    success = []
    mistakes = []
    avoided = []
    accepted = False
    for rec in records:
        if rec.get("accepted"):
            accepted = True
        if rec.get("decision") == "avoid" and not rec.get("ran"):
            avoided.append({"edit_class": rec.get("edit_class"), "score": None})
            continue
        if not rec.get("ran"):
            continue
        measured = {
            "edit_class": rec.get("edit_class"),
            "decision": rec.get("decision"),
            "verdict": rec.get("verdict"),
            "score": rec.get("score"),
        }
        if rec.get("decision") == "act" and rec.get("accepted") and rec.get("verdict") == "better":
            success.append({**measured, "kind": "act-accepted"})
        elif rec.get("decision") == "hold":
            success.append({**measured, "kind": "hold-measured"})
        if rec.get("decision") == "act" and rec.get("accepted") and rec.get("verdict") != "better":
            mistakes.append({**measured, "kind": "mis-release"})
    if budget_verdict == "worse" and not accepted:
        mistakes.append({"kind": "mis-release", "detail": "no candidate accepted and the budget is worse"})
    return {"success": success, "mistakes": mistakes, "avoid": avoided}


def neutral_skill() -> dict:
    return {"neutral": True, "rules": []}


def skill_text(skill: dict) -> str:
    if not skill.get("rules"):
        return "中性初值。没有根据实测升降写成的附加规则。"
    lines = []
    for rule in skill["rules"]:
        lines.append(f"- {rule.get('id')}: {rule.get('edit_class')} {rule.get('stance')} {rule.get('text')}")
    return "\n".join(lines)


def _class_name(edit_class: str) -> str:
    name = (edit_class or "unlabeled").strip().lower().replace(" ", "_")
    return name or "unlabeled"


def default_proposal(records: list[dict], skill: dict | None = None) -> Proposal:
    """One criterion update from every measured class in this budget.

    A replaced incumbent does not keep avoid rules from the old code.
    A flat result does not by itself add a rule.
    """
    skill = skill or {"rules": []}
    replaced = any(rec.get("incumbent_replaced") for rec in records)
    grouped: dict[str, dict] = {}
    for rec in records:
        name = _class_name(str(rec.get("edit_class", "")))
        slot = grouped.setdefault(name, {})
        if rec.get("decision") == "avoid" and not rec.get("ran"):
            slot["avoid_unrun"] = True
            continue
        if not rec.get("ran"):
            continue
        slot["worse"] = max(int(slot.get("worse", 0)), int(rec.get("worse_count") or 0))
        slot["better"] = max(int(slot.get("better", 0)), int(rec.get("better_count") or 0))
        if rec.get("verdict") == "better":
            slot["better_now"] = True
        if rec.get("verdict") == "worse":
            slot["worse_now"] = True
        if rec.get("accepted") and rec.get("verdict") == "better":
            slot["accepted_better"] = True
        if rec.get("decision") == "act" and rec.get("verdict") != "better":
            slot["misrelease"] = True
    adds = []
    delete = []
    for name, slot in grouped.items():
        worse = int(slot.get("worse", 0))
        better = int(slot.get("better", 0))
        if slot.get("avoid_unrun") or worse > 3:
            stance, text = "avoid", "累计变差超过 3 次"
        elif worse >= 1:
            stance, text = "hold", "有实测变差，次数未超过 3"
        elif better >= 2 and not slot.get("misrelease"):
            stance, text = "act", "多次实测变好"
        else:
            stance, text = None, ""
        if replaced and stance == "avoid":
            stance, text = None, ""
        for rule in skill.get("rules") or []:
            if _class_name(str(rule.get("edit_class", ""))) != name:
                continue
            if rule.get("stance") != stance:
                delete.append(str(rule.get("id")))
        if stance:
            adds.append({"id": f"{stance}-{name}", "edit_class": name, "stance": stance, "text": text})
    return Proposal(add=adds, delete=list(dict.fromkeys(delete)))


def proposal_from_obj(obj: dict) -> Proposal:
    if not isinstance(obj, dict):
        return Proposal(allow_all=True)
    touches = "strike" in obj or "worse_limit" in obj
    add = []
    for item in obj.get("add") or []:
        if not isinstance(item, dict):
            continue
        class_name = str(item.get("edit_class", "")).strip()
        if class_name == "*":
            return Proposal(block_all=True, touches_strike=touches)
        add.append(
            {
                "id": str(item.get("id") or f"rule-{class_name}"),
                "edit_class": class_name or "unlabeled",
                "stance": item.get("stance") if item.get("stance") in ("act", "hold", "avoid") else "hold",
                "text": str(item.get("text", ""))[:240],
            }
        )
    return Proposal(
        add=add,
        delete=[str(x) for x in (obj.get("delete") or [])],
        allow_all=bool(obj.get("allow_all")),
        block_all=bool(obj.get("block_all")),
        touches_strike=touches,
    )


def apply_proposal(skill: dict, proposal: Proposal, seen: set[str], has_history: bool) -> tuple[dict, bool]:
    """Return the skill to keep, and whether this proposal was accepted.

    Before any measured history, the neutral skill stays and is not vetoed.
    """
    if not has_history:
        return skill, True
    avoid_classes = {r["edit_class"] for r in proposal.add if r.get("stance") == "avoid"}
    covers_seen = len(seen) >= 3 and seen <= avoid_classes
    if proposal.allow_all or proposal.block_all or proposal.touches_strike or covers_seen:
        return skill, False
    rules = [r for r in skill.get("rules", []) if r.get("id") not in set(proposal.delete)]
    known = {r.get("id") for r in rules}
    for rule in proposal.add:
        if rule["id"] in known:
            rules = [rule if r.get("id") == rule["id"] else r for r in rules]
        else:
            rules.append(rule)
    return {"neutral": False, "rules": rules}, True
