from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass
class LLMConfig:
    provider: str = "mock"
    base_url: str | None = None
    model: str | None = None
    api_key_env: str = "LLM_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 4000
    timeout_s: float = 180.0
    retries: int = 2


@dataclass
class SandboxConfig:
    time_limit_s: float = 2.0
    tle_factor: float = 1.5
    hard_grace_s: float = 2.0
    max_mem_mb: int = 1024
    iter_budget: int = 20000
    parallel: int = 4


@dataclass
class SearchConfig:
    """Presets from the exploration and tolerance sections. One rho update uses only these."""

    n: int = 16
    rho_0: float = 0.8
    rho_min: float = 0.2
    rho_max: float = 0.9
    lam: float = 0.5
    eta: float = 1.0
    delta: float = 0.1
    m: int = 3
    epsilon_0: float = 1e-6


@dataclass
class EvaluatorConfig:
    """How to run an evolve-task evaluator. via=wsl uses a Linux interpreter.

    distro selects the WSL distribution. The default `wsl` command on this
    machine is docker-desktop, which cannot run the organon environment.
    """

    via: str = "native"
    python: str | None = None
    distro: str | None = None


@dataclass
class TaskConfig:
    name: str
    problem: str
    instance_seed: int = 1
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    run_dir: str = "runs/demo"
    max_budgets: int = 4
    tasks: list[TaskConfig] = field(default_factory=list)
    llm: LLMConfig = field(default_factory=LLMConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)


def _take(data: dict, cls):
    names = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in data.items() if k in names})


def load_config(path: str | Path) -> RunConfig:
    raw = _expand(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    cfg = RunConfig(
        run_dir=raw.get("run_dir", "runs/demo"),
        max_budgets=int(raw.get("max_budgets", 4)),
        tasks=[_take(item, TaskConfig) for item in raw.get("tasks", [])],
        llm=_take(raw.get("llm") or {}, LLMConfig),
        sandbox=_take(raw.get("sandbox") or {}, SandboxConfig),
        search=_take(raw.get("search") or {}, SearchConfig),
        evaluator=_take(raw.get("evaluator") or {}, EvaluatorConfig),
    )
    _check(cfg)
    return cfg


def _check(cfg: RunConfig) -> None:
    if cfg.max_budgets < 1:
        raise ValueError("max_budgets must be at least 1")
    if not cfg.tasks:
        raise ValueError("at least one task is required")
    names = [t.name for t in cfg.tasks]
    if len(names) != len(set(names)):
        raise ValueError("task names must be unique")
    s = cfg.search
    if s.n < 2:
        raise ValueError("search.n must be at least 2 so both kinds can be generated")
    if not (s.rho_min <= s.rho_0 <= s.rho_max):
        raise ValueError("rho_0 must lie between rho_min and rho_max")
    if s.eta <= 0 or s.lam < 0 or s.delta < 0 or s.m < 1 or s.epsilon_0 < 0:
        raise ValueError("search parameters are out of range")
    if cfg.llm.provider not in ("mock", "openai"):
        raise ValueError("llm.provider must be mock or openai")
    if cfg.evaluator.via not in ("native", "wsl"):
        raise ValueError("evaluator.via must be native or wsl")
    if cfg.evaluator.via == "wsl" and not cfg.evaluator.python:
        raise ValueError("evaluator.python is required when evaluator.via is wsl; set META_EVAL_PYTHON in .env")


def _expand(value):
    if isinstance(value, str):
        return _ENV_REF.sub(lambda match: os.environ.get(match.group(1), ""), value)
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def load_dotenv(path: str | Path = ".env") -> None:
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        import os

        os.environ.setdefault(key, value)
