from pathlib import Path

import pytest

from meta_new.config import LLMConfig, load_config
from meta_new.generation.llm import LLMError, OpenAICompatLLM


def test_demo_config_is_per_problem_and_has_one_instance_seed():
    cfg = load_config("configs/demo.yaml")
    text = Path("configs/demo.yaml").read_text(encoding="utf-8")
    assert cfg.llm.provider == "openai"
    assert cfg.llm.base_url == "https://api.deepseek.com"
    assert cfg.llm.model == "deepseek-chat"
    assert "sk-" not in text
    assert {task.problem for task in cfg.tasks} == {"tsp", "knapsack"}
    assert all(task.instance_seed == 1 for task in cfg.tasks)
    assert cfg.search.n >= 2


def test_wsl_evaluator_requires_the_python_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("META_EVAL_PYTHON", raising=False)
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "run_dir: runs/x\nmax_budgets: 2\nevaluator:\n  via: wsl\n  python: ${META_EVAL_PYTHON}\ntasks:\n  - {name: a, problem: tsp}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="META_EVAL_PYTHON"):
        load_config(path)


def test_a_model_run_requires_the_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    cfg = LLMConfig(provider="openai", base_url="https://api.deepseek.com", model="deepseek-chat")
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        OpenAICompatLLM(cfg)
