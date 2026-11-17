from __future__ import annotations

import json
import re

from ..meta.evolve import Proposal, meta_trace, proposal_from_obj, skill_text
from ..models import Candidate, ProposeContext
from .llm import LLMError

_JSON = re.compile(r"```json\s*(.*?)```", re.S | re.I)
_PY = re.compile(r"```python\s*(.*?)```", re.S | re.I)
_CPP = re.compile(r"```(?:cpp|c\+\+|cc)\s*(.*?)```", re.S | re.I)


def extract_json(text: str) -> dict:
    match = _JSON.search(text)
    raw = match.group(1) if match else text
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no json object")
    return json.loads(raw[start : end + 1])


def extract_python(text: str) -> str:
    match = _PY.search(text)
    if match:
        return match.group(1).strip() + "\n"
    return ""


def extract_cpp(text: str) -> str:
    match = _CPP.search(text)
    if match:
        return match.group(1).strip() + "\n"
    return ""


def extract_code(text: str, language: str) -> str:
    if language == "cpp":
        return extract_cpp(text)
    return extract_python(text)


ANALYZER_FIRST = (
    "你是 Analyzer。阅读 baseline 轨迹，指出限制求解质量的行为特征和所在步骤。"
    "若轨迹里有逐 seed 符号，行为只描述写了两遍的 seed。这些符号不是得分。"
    "这次没有升降结论，不要把 baseline 写成变好或变差，不要给修改方案。"
    '只输出 JSON：{"problem":"...","budget":0,"verdict":null,"behaviors":[{"feature":"...","step":"..."}]}'
)
ANALYZER_NEXT = (
    "你是 Analyzer。只说明本档升降对应的行为，不给修改方案。"
    "若轨迹里有逐 seed 符号，把 − 和仍低的 seed 写进行为。符号和重复行不改变官方得分。"
    "变好时写出要保留的行为，并标出还能改进的位置，不要把变好写成缺陷。"
    "变差时定位与这次变差对应的行为和步骤。"
    "持平时不要写成退化，也不要写缺陷根因，也不要改写官方得分。"
    '只输出 JSON：{"problem":"...","budget":0,"verdict":"better|worse|flat",'
    '"behaviors":[{"feature":"...","step":"..."}],"keep":["..."],"improve":["..."]}'
)
CRITIC = (
    "你是 Critic。根据 Analyzer 的记录写修改假设，不要写代码。"
    "单次变差不写成必须修改的位置。只有反复变差的类别才写成必须处理的局部修改。"
    "若轨迹把某些 seed 写了两遍，假设针对这些 seed，并且只改一份求解过程。不要写成两套搜索或两个 solve。"
    "标出可能让后续档位变差的副作用。去掉明显不能运行或会直接破坏合法输出的改法。"
    "edit_class 用这次改动自己的英文蛇形名，例如 nearest_neighbor 或 swap_two_cities。"
    "不要使用 snake_name、local、explore、unlabeled 作为类别名。"
    '只输出 JSON：{"hypotheses":[{"edit_class":"nearest_neighbor","kind":"explore","change":"...","must_fix":false,"risk":"..."}]}'
)
MUTATOR = (
    "你是 Mutator。基于当前 incumbent 输出一份完整 Python 求解器。"
    "程序从 stdin 读入题目，把答案打印到 stdout。可以用标准库。不能联网、不能开子进程、不能读写文件。"
    "环境变量 RSI_TIME_LIMIT、RSI_ITER_BUDGET、RSI_SEED 可用。"
    "若假设或上一档记录点名了 seed，这一份程序要处理这些 seed。"
    "只保留一份求解过程，不要按 seed 拆成两套搜索。能否替换只看全部 seed 的平均分。"
    "探索候选可以换 solve 里的解法。局部候选只改与假设直接相关的片段。"
    "两行标记 # --- solve begin --- 和 # --- solve end --- 以外的代码必须与 incumbent 逐字相同，包括注释。"
    "edit_class 写这次改动自己的英文蛇形名，不要写 snake_name、local、explore。"
    "不要声称自己的分数。先输出一个 json 代码块，再输出一个 python 代码块。"
    'json 形如 {"edit_class":"nearest_neighbor","kind":"explore","hypothesis":"..."}'
)
MUTATOR_CPP = (
    "你是 Mutator。基于当前 incumbent 输出一份完整 C++20 程序。"
    "程序从 stdin 读入题目，把答案写到 stdout。交互式题目按题面要求 flush。"
    "保留源文件里的 #EVOLVE_START 和 #EVOLVE_END 两行。"
    "若假设或上一档记录点名了 seed，这一份程序要处理这些 seed。"
    "只保留一份求解过程，不要按 seed 拆成两套搜索。能否替换只看全部 seed 的平均分。"
    "探索候选可以更换求解策略。局部候选只改这两个标记之间、且与假设直接相关的片段；标记以外的内容必须与 incumbent 逐字相同。"
    "edit_class 写这次改动自己的英文蛇形名，不要写 snake_name、local、explore。"
    "不要声称自己的分数。先输出一个 json 代码块，再输出一个 cpp 代码块。"
    'json 形如 {"edit_class":"wall_priority","kind":"explore","hypothesis":"..."}'
)
VERIFIER = (
    "你是 Verifier。只做初筛，不要改写代码，不要打分。"
    "去掉语法非法、流程不成立、一加载就会崩溃、以及与 incumbent 几乎相同的候选。"
    '只输出 JSON：{"accept":true,"reason":"..."}'
)
META = (
    "你只修改这道题的元判别准则，不修改求解代码。下一档闸门读取这套准则。"
    "未运行的候选没有分数，不要编造。"
    "次数门槛固定为 3：累计变差超过 3 次才写入 avoid，这个数字不能改。未超过 3 次的变差写成 hold，该类仍可再跑。"
    "变好次数达到 2 次且没有未清零的变差，才写成 act。只有一次变好不要写成 act。"
    "删除和本档实测不符的旧规则，保留仍然相符的规则。"
    "不要给出放行几乎全部变异或拦截几乎全部变异的规则。"
    '只输出 JSON：{"delete":[],"add":[{"id":"...","edit_class":"...","stance":"act|hold|avoid","text":"..."}],'
    '"allow_all":false,"block_all":false}'
)


class LLMGenerator:
    def __init__(self, llm) -> None:
        self.llm = llm

    def propose(self, ctx: ProposeContext) -> list[Candidate]:
        system = ANALYZER_FIRST if ctx.phase == "first" else ANALYZER_NEXT
        try:
            analyzed = extract_json(self.llm.complete("analyzer", system, _analyzer_user(ctx)))
        except (LLMError, ValueError, json.JSONDecodeError):
            return []
        try:
            criticism = extract_json(self.llm.complete("critic", CRITIC, _critic_user(ctx, analyzed)))
        except (LLMError, ValueError, json.JSONDecodeError):
            return []
        hypotheses = list(criticism.get("hypotheses") or [])
        out: list[Candidate] = []
        plan = [("explore", i) for i in range(ctx.n_explore)]
        if hypotheses:
            plan += [("local", i) for i in range(ctx.n_local)]
        for kind, index in plan:
            hint = hypotheses[index % len(hypotheses)] if hypotheses else {}
            if hint.get("kind") not in ("explore", "local"):
                hint = {**hint, "kind": kind}
            else:
                hint = {**hint, "kind": kind}
            try:
                system = MUTATOR_CPP if ctx.language == "cpp" else MUTATOR
                text = self.llm.complete("mutator", system, _mutator_user(ctx, hint, kind, index))
            except LLMError:
                continue
            code = extract_code(text, ctx.language)
            if not code:
                continue
            try:
                header = extract_json(text)
            except (ValueError, json.JSONDecodeError):
                header = {}
            edit_class = str(header.get("edit_class") or hint.get("edit_class") or f"{kind}_{index}")
            out.append(
                Candidate(
                    code=code,
                    kind=kind,
                    edit_class=edit_class,
                    hypothesis=str(header.get("hypothesis") or hint.get("change") or ""),
                    origin=f"{kind}-{index}",
                )
            )
        return out

    def screen(self, candidates: list[Candidate], incumbent: str, language: str = "python") -> tuple[list[Candidate], list[dict]]:
        kept: list[Candidate] = []
        dropped: list[dict] = []
        for cand in candidates:
            try:
                text = self.llm.complete("verifier", VERIFIER, _verifier_user(cand, incumbent, language))
                verdict = extract_json(text)
            except (LLMError, ValueError, json.JSONDecodeError):
                dropped.append(
                    {"edit_class": cand.edit_class, "kind": cand.kind, "reason": "verifier", "detail": "no verdict"}
                )
                continue
            if verdict.get("accept") is not True:
                dropped.append(
                    {
                        "edit_class": cand.edit_class,
                        "kind": cand.kind,
                        "reason": "verifier",
                        "detail": str(verdict.get("reason", ""))[:240],
                    }
                )
                continue
            kept.append(cand)
        return kept, dropped


def proposal_from_llm(llm, records: list[dict], skill: dict) -> Proposal:
    payload = json.dumps(
        {"skill": skill, "trace": meta_trace(records), "measured": _shrink(records)},
        ensure_ascii=False,
    )[:4000]
    try:
        text = llm.complete("meta", META, payload)
        return proposal_from_obj(extract_json(text))
    except (LLMError, ValueError, json.JSONDecodeError):
        return Proposal()


def _analyzer_user(ctx: ProposeContext) -> str:
    lines = [f"题目：{ctx.problem_name}", f"方向：{ctx.direction}"]
    if ctx.phase != "first":
        lines.append(f"budget：{ctx.budget_index}")
        lines.append(f"本档结论：{ctx.verdict}")
        lines.append(f"上一档官方得分：{ctx.previous_official}")
        lines.append("各候选分数：")
        if not ctx.candidate_scores:
            lines.append("- 无")
        for row in ctx.candidate_scores:
            if row.get("ran"):
                lines.append(f"- {row.get('edit_class')}: {row.get('score')} {row.get('verdict')}")
            else:
                lines.append(f"- {row.get('edit_class')}: 未运行")
    lines.append("轨迹：")
    lines.append(ctx.trajectory_excerpt)
    return "\n".join(lines)


def _critic_user(ctx: ProposeContext, analyzed: dict) -> str:
    must = ", ".join(ctx.repeated_worse) or "无"
    return (
        f"Analyzer：{json.dumps(analyzed, ensure_ascii=False)}\n"
        f"本档轨迹：\n{ctx.trajectory_excerpt}\n"
        f"本档结论：{ctx.verdict}\n"
        f"这道题当前 incumbent 上已经积累的升降记录：{json.dumps(ctx.history, ensure_ascii=False)}\n"
        f"变差次数大于 1、应写成局部必须处理的类别：{must}\n"
        "变差次数为 1 的是单次变差，不要写成必须修改，也不要取消下一档候选。"
    )


def _mutator_user(ctx: ProposeContext, hint: dict, kind: str, index: int) -> str:
    fence = "cpp" if ctx.language == "cpp" else "python"
    return (
        f"题面：{ctx.statement}\n当前元判别准则：\n{ctx.skill_text}\n"
        f"程序语言：{ctx.language}\n"
        f"这份候选类别：{kind}，序号：{index}\n假设：{json.dumps(hint, ensure_ascii=False)}\n"
        f"上一档记录：\n{ctx.trajectory_excerpt}\n"
        "当前 incumbent 全文如下，请输出完整替换文件。\n"
        f"```{fence}\n{ctx.incumbent_code}\n```"
    )


def _shrink(records: list[dict]) -> list[dict]:
    out = []
    for rec in records:
        out.append(
            {
                "edit_class": rec.get("edit_class"),
                "decision": rec.get("decision"),
                "ran": rec.get("ran"),
                "verdict": rec.get("verdict"),
                "score": rec.get("score"),
                "accepted": rec.get("accepted"),
            }
        )
    return out


def _verifier_user(cand: Candidate, incumbent: str, language: str = "python") -> str:
    fence = "cpp" if language == "cpp" else "python"
    note = "这是 C++ 程序，按 C++ 判断语法。" if language == "cpp" else "这是 Python 程序。"
    return (
        f"改法类别：{cand.edit_class}\n类别：{cand.kind}\n"
        f"{note}\n"
        "当前 incumbent：\n"
        f"```{fence}\n{incumbent}\n```\n"
        "候选：\n"
        f"```{fence}\n{cand.code}\n```\n"
        "语法非法、流程不成立、一加载就会崩溃、或与 incumbent 几乎相同，则 accept 为 false。不要打分。"
    )


def render_skill(skill: dict) -> str:
    return skill_text(skill)
