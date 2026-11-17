from __future__ import annotations

import re
import shutil
import subprocess

from ..models import Candidate
from ..runtime.sandbox import load_check

_FORBIDDEN = (
    re.compile(
        r"(^|\n)\s*(import|from)\s+(socket|urllib|http|requests|httpx|subprocess|multiprocessing|ctypes|shutil|asyncio)\b"
    ),
    re.compile(r"\bopen\s*\("),
    re.compile(r"\bos\.(system|popen|remove|unlink|rmdir)\b"),
    re.compile(r"\bsubprocess\b"),
)


def _missing_entry(code: str, incumbent: str) -> bool:
    """The candidate drops an entry the current program uses to read, solve, or emit."""
    for name in ("read_input", "solve", "main"):
        token = f"def {name}"
        if token in incumbent and token not in code:
            return True
    return False


def normalize(code: str) -> str:
    return "\n".join(line.rstrip() for line in code.strip().splitlines())


def _plain(code: str) -> str:
    """Same program once comments and surrounding whitespace are ignored."""
    lines = []
    for line in code.splitlines():
        if "#" in line:
            line = line.split("#", 1)[0]
        line = line.strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


_SOLVE = re.compile(r"# --- solve begin ---.*?# --- solve end ---", re.S)


def _outside_solve(code: str, incumbent: str) -> bool:
    """A local edit changed something outside the marked solve region."""
    if "# --- solve begin ---" not in incumbent:
        return False
    def shell(text: str) -> str:
        return normalize(_SOLVE.sub("", text))
    return shell(code) != shell(incumbent)


_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_LINE_COMMENT = re.compile(r"//.*?$", re.M)
_MARKERS = {"#EVOLVE_START", "#EVOLVE_END"}
_MAIN = re.compile(r"\bmain\s*\(")
_EVOLVE = re.compile(r"#EVOLVE_START.*?#EVOLVE_END", re.S)
_COMPILER: str | None | bool = False


def strip_markers(code: str) -> str:
    return "\n".join(line for line in code.splitlines() if line.strip() not in _MARKERS)


def _plain_cpp(code: str) -> str:
    text = _BLOCK_COMMENT.sub("", strip_markers(code))
    text = _LINE_COMMENT.sub("", text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _compiler() -> str | None:
    """A C++ compiler that can syntax-check, or nothing when this machine has none."""
    global _COMPILER
    if _COMPILER is not False:
        return _COMPILER or None
    found = shutil.which("g++") or shutil.which("clang++")
    if not found:
        _COMPILER = None
        return None
    try:
        probe = subprocess.run(
            [found, "-std=c++20", "-fsyntax-only", "-x", "c++", "-"],
            input="int main(){return 0;}\n",
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        _COMPILER = None
        return None
    _COMPILER = found if probe.returncode == 0 else None
    return _COMPILER


def cpp_syntax(code: str) -> str | None:
    compiler = _compiler()
    if not compiler:
        return None
    try:
        proc = subprocess.run(
            [compiler, "-std=c++20", "-fsyntax-only", "-x", "c++", "-"],
            input=strip_markers(code),
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode == 0:
        return None
    detail = next((line.strip() for line in (proc.stderr or "").splitlines() if line.strip()), "syntax")
    return f"syntax:{detail[:180]}"


def _outside_evolve(code: str, incumbent: str) -> bool:
    if "#EVOLVE_START" not in incumbent or "#EVOLVE_END" not in incumbent:
        return False

    def shell(text: str) -> str:
        return normalize(_EVOLVE.sub("", text))

    return shell(code) != shell(incumbent)


def near_copy(a: str, b: str, language: str = "python") -> bool:
    if language == "cpp":
        return _plain_cpp(a) == _plain_cpp(b)
    return _plain(a) == _plain(b)


def rejection(code: str, incumbent: str, language: str = "python") -> str | None:
    if not code.strip():
        return "empty"
    if near_copy(code, incumbent, language):
        return "near_copy"
    if language == "cpp":
        if not _MAIN.search(strip_markers(code)):
            return "flow"
        return cpp_syntax(code)
    try:
        compile(code, "candidate.py", "exec")
    except SyntaxError as exc:
        return f"syntax:{exc.lineno}"
    if _missing_entry(code, incumbent):
        return "flow"
    for pattern in _FORBIDDEN:
        if pattern.search(code):
            return "forbidden"
    crash = load_check(code)
    if crash:
        return crash
    return None


def dedup(candidates: list[Candidate], incumbent: str, language: str = "python") -> tuple[list[Candidate], list[dict]]:
    kept: list[Candidate] = []
    dropped: list[dict] = []
    for cand in candidates:
        if near_copy(cand.code, incumbent, language) or any(near_copy(cand.code, prev.code, language) for prev in kept):
            dropped.append({"edit_class": cand.edit_class, "kind": cand.kind, "reason": "duplicate"})
            continue
        kept.append(cand)
    return kept, dropped


def verify(candidates: list[Candidate], incumbent: str, language: str = "python") -> tuple[list[Candidate], list[dict]]:
    kept: list[Candidate] = []
    dropped: list[dict] = []
    for cand in candidates:
        reason = rejection(cand.code, incumbent, language)
        outside = _outside_evolve if language == "cpp" else _outside_solve
        if reason is None and cand.kind == "local" and outside(cand.code, incumbent):
            reason = "not_local"
        if reason:
            dropped.append({"edit_class": cand.edit_class, "kind": cand.kind, "reason": reason})
            continue
        kept.append(cand)
    return kept, dropped
