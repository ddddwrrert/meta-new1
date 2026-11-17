from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from ..config import LLMConfig


class LLMError(RuntimeError):
    pass


class OpenAICompatLLM:
    """Frozen chat model. The loop calls it; it does not train."""

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self.calls = 0
        self.base_url = (cfg.base_url or os.environ.get("LLM_BASE_URL") or "").rstrip("/")
        self.model = cfg.model or os.environ.get("LLM_MODEL_NAME")
        self.api_key = os.environ.get(cfg.api_key_env, "")
        if not self.base_url or not self.model:
            raise LLMError("set LLM_BASE_URL and LLM_MODEL_NAME, or llm.base_url and llm.model")
        if not self.api_key:
            raise LLMError(
                f"set {cfg.api_key_env} in the environment or in .env next to the project. Do not put the key in the yaml."
            )

    def complete(self, role: str, system: str, user: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": self.cfg.temperature,
                "max_tokens": self.cfg.max_tokens,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last = "no attempt"
        for attempt in range(self.cfg.retries + 1):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                self.calls += 1
                return data["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as exc:
                last = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}"
                if exc.code not in (408, 409, 429) and exc.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError, OSError, KeyError, json.JSONDecodeError) as exc:
                last = f"{type(exc).__name__}: {exc}"
            time.sleep(min(8.0, 2.0**attempt))
        raise LLMError(f"{role} call failed: {last}")
