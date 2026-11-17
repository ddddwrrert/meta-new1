import json

from meta_new.models import Candidate, ProposeContext
from meta_new.generation.roles import CRITIC, LLMGenerator, MUTATOR_CPP, _critic_user


class Fake:
    def __init__(self, accept=True):
        self.roles = []
        self.accept = accept

    def complete(self, role, system, user):
        self.roles.append(role)
        if role == "analyzer":
            return '{"problem":"p","budget":0,"verdict":null,"behaviors":[{"feature":"long","step":"scan"}]}'
        if role == "critic":
            return '{"hypotheses":[{"edit_class":"swap","kind":"local","change":"shorten","must_fix":false,"risk":"none"}]}'
        if role == "mutator":
            return (
                '```json\n{"edit_class":"swap","kind":"local","hypothesis":"shorten"}\n```\n'
                "```python\nx = 1\n```"
            )
        if role == "verifier":
            return json.dumps({"accept": self.accept, "reason": "checked"})
        raise AssertionError(role)


def _ctx():
    return ProposeContext(
        phase="first",
        problem_name="tiny",
        statement="higher",
        direction="higher",
        incumbent_code="y = 0\n",
        rho=0.8,
        n=1,
        n_explore=1,
        n_local=0,
    )


def test_critic_sees_the_accumulated_history():
    ctx = _ctx()
    ctx.history = [{"edit_class": "swap", "runs": 1, "worse": 1, "better": 0, "last": "worse"}]
    ctx.repeated_worse = []
    text = _critic_user(ctx, {"behaviors": []})
    assert '"worse": 1' in text
    assert "单次变差" in text


def test_a_failed_analyzer_does_not_invent_candidates():
    class Broken:
        def complete(self, role, system, user):
            return "not json"

    assert LLMGenerator(Broken()).propose(_ctx()) == []


def test_roles_run_in_document_order():
    llm = Fake()
    gen = LLMGenerator(llm)
    candidates = gen.propose(_ctx())
    kept, dropped = gen.screen(candidates, "y = 0\n")
    assert [cand.edit_class for cand in kept] == ["swap"]
    assert dropped == []
    assert llm.roles == ["analyzer", "critic", "mutator", "verifier"]


def test_no_hypothesis_skips_local_and_keeps_explore():
    class NoHyp(Fake):
        def complete(self, role, system, user):
            if role == "critic":
                self.roles.append(role)
                return '{"hypotheses":[]}'
            return super().complete(role, system, user)

    ctx = _ctx()
    ctx.n_local = 2
    candidates = LLMGenerator(NoHyp()).propose(ctx)
    assert [cand.kind for cand in candidates] == ["explore"]


def test_verifier_silence_drops_the_candidate():
    class Silent(Fake):
        def complete(self, role, system, user):
            if role == "verifier":
                self.roles.append(role)
                return "not json"
            return super().complete(role, system, user)

    gen = LLMGenerator(Silent())
    candidates = gen.propose(_ctx())
    kept, dropped = gen.screen(candidates, "y = 0\n")
    assert kept == []
    assert dropped[0]["reason"] == "verifier"


def test_cpp_mutator_reads_a_cpp_block():
    class Cpp(Fake):
        def complete(self, role, system, user):
            self.roles.append((role, system))
            if role == "analyzer":
                return '{"problem":"p","budget":0,"verdict":null,"behaviors":[{"feature":"long","step":"scan"}]}'
            if role == "critic":
                return '{"hypotheses":[{"edit_class":"swap","kind":"local","change":"shorten","must_fix":false,"risk":"none"}]}'
            if role == "mutator":
                return (
                    '```json\n{"edit_class":"bias","kind":"explore","hypothesis":"bias"}\n```\n'
                    "```cpp\n#include <iostream>\nint main(){return 1;}\n```"
                )
            raise AssertionError(role)

    ctx = _ctx()
    ctx.language = "cpp"
    ctx.incumbent_code = "#include <iostream>\nint main(){return 0;}\n"
    llm = Cpp()
    candidates = LLMGenerator(llm).propose(ctx)
    assert "int main(){return 1;}" in candidates[0].code
    assert any(role == "mutator" and "C++20" in system for role, system in llm.roles)


def test_prompts_keep_one_solver_and_the_mean():
    assert "两套搜索" in CRITIC
    assert "两套搜索" in MUTATOR_CPP
    assert "平均分" in MUTATOR_CPP


def test_verifier_reject_drops_the_candidate():
    llm = Fake(accept=False)
    gen = LLMGenerator(llm)
    candidates = gen.propose(_ctx())
    kept, dropped = gen.screen(candidates, "y = 0\n")
    assert kept == []
    assert dropped[0]["reason"] == "verifier"
    assert isinstance(candidates[0], Candidate)
