"""Score one program with an evolve-task evaluator.

This file is executed by the interpreter that has the task's dependencies.
It stays on the standard library so that interpreter does not need meta-new.

    python task_runner.py <task dir> <program file>

The program is written again without #EVOLVE_START / #EVOLVE_END, then
evaluate(path) from <task>/evaluator/evaluator.py runs under the timeout and
retries in task.json. One JSON line goes to stdout:
{"fitness": ..., "metrics": {...}}
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

MARKERS = {b"#EVOLVE_START", b"#EVOLVE_END"}


class EvaluationTimeout(Exception):
    pass


def load_evaluate(path: Path):
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    name = f"_evolve_eval_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.evaluate


def call_with_timeout(function, argument, timeout: float):
    outcome = {}

    def target():
        try:
            outcome["value"] = function(argument)
        except Exception as error:
            outcome["error"] = error

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise EvaluationTimeout()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def strip_markers(code: bytes) -> bytes:
    kept = [line for line in code.split(b"\n") if line.strip() not in MARKERS]
    return b"\n".join(kept)


def run_evaluate(function, code: bytes, suffix: str, timeout: float, max_retries: int) -> dict:
    for attempt in range(max_retries + 1):
        handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            handle.write(code)
            handle.close()
            result = call_with_timeout(function, handle.name, timeout)
            return result if isinstance(result, dict) else {"error": "evaluator did not return a dict"}
        except EvaluationTimeout:
            return {"error": "timeout", "timeout": True}
        except Exception as error:
            print(
                f"evaluate() failed, attempt {attempt + 1} of {max_retries + 1}: {error!r}",
                file=sys.stderr,
            )
            if attempt < max_retries:
                time.sleep(1.0)
        finally:
            try:
                os.unlink(handle.name)
            except OSError:
                pass
    return {"error": "evaluate failed"}


def fitness_of(metrics: dict) -> float:
    if not metrics:
        return 0.0
    if "combined_score" in metrics:
        try:
            return float(metrics["combined_score"])
        except (TypeError, ValueError):
            pass
    values = [v for v in metrics.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    task, program = Path(sys.argv[1]), Path(sys.argv[2])
    os.chdir(task)
    settings = json.loads((task / "task.json").read_text(encoding="utf-8"))
    result_stream = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    try:
        function = load_evaluate(task / "evaluator" / "evaluator.py")
    except Exception as error:
        print(f"cannot load evaluator: {error!r}", file=sys.stderr)
        result_stream.write(json.dumps({"fitness": None, "metrics": {"error": str(error)}}) + "\n")
        result_stream.flush()
        return
    code = strip_markers(program.read_bytes())
    suffix = program.suffix or ".cpp"
    metrics = run_evaluate(
        function,
        code,
        suffix,
        float(settings.get("timeout", 10000)),
        int(settings.get("max_retries", 0)),
    )
    result_stream.write(json.dumps({"fitness": fitness_of(metrics), "metrics": metrics}, default=str) + "\n")
    result_stream.flush()


if __name__ == "__main__":
    main()
