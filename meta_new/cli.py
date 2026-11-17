from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config, load_dotenv
from .generation.mockgen import MockGenerator
from .loop.task import run_task
from .problems import build_problem
from .problems.evolve import resolve_task_dir
from .runtime.evaluate import SandboxEvaluator
from .runtime.evolve_eval import EvolveTaskEvaluator


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="meta-new")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run each problem from its own baseline")
    run.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    resolved = Path(args.config).resolve()
    load_dotenv(resolved.parent.parent / ".env")
    load_dotenv(resolved.parent / ".env")
    load_dotenv()
    cfg = load_config(args.config)
    generator, meta_fn = _backend(cfg, args.config)
    run_dir = Path(cfg.run_dir)
    if not run_dir.is_absolute():
        run_dir = Path.cwd() / run_dir
    for task in cfg.tasks:
        problem, evaluator = _open_task(task, cfg, args.config)
        run_task(
            name=task.name,
            problem=problem,
            instance_seed=task.instance_seed,
            search=cfg.search,
            budgets=cfg.max_budgets,
            evaluator=evaluator,
            generator=generator,
            run_dir=run_dir,
            propose_meta=meta_fn,
        )


def _open_task(task, cfg, config_path: str | Path):
    params = dict(task.params)
    if task.problem == "evolve":
        params["task_dir"] = str(resolve_task_dir(config_path, params["task_dir"]))
        problem = build_problem("evolve", params)
        python = params.get("python") or cfg.evaluator.python
        via = params.get("via") or cfg.evaluator.via
        distro = params.get("distro") or cfg.evaluator.distro
        return problem, EvolveTaskEvaluator(problem, python=python, via=via, distro=distro)
    problem = build_problem(task.problem, params)
    return problem, SandboxEvaluator(problem, cfg.sandbox, seed=0)


def _backend(cfg, config_path: str | Path):
    if cfg.llm.provider == "mock" or cfg.max_budgets <= 1:
        return MockGenerator(), None
    resolved = Path(config_path).resolve()
    load_dotenv(resolved.parent.parent / ".env")
    load_dotenv(resolved.parent / ".env")
    load_dotenv()
    from .generation.llm import OpenAICompatLLM
    from .generation.roles import LLMGenerator, proposal_from_llm

    llm = OpenAICompatLLM(cfg.llm)
    return LLMGenerator(llm), (lambda records, skill: proposal_from_llm(llm, records, skill))


if __name__ == "__main__":
    main()
