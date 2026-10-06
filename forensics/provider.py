"""LLM provider interface with a deterministic mock and an optional Anthropic client."""
from __future__ import annotations

import os
from typing import Callable, Protocol


class Provider(Protocol):
    def complete(self, prompt: str, system: str | None = None) -> str: ...


class MockProvider:
    """Deterministic provider for tests and the offline demo.

    Give it either a `handler(prompt, system) -> str` or a `script` list of
    responses (consumed in order, the last one repeats). Output is simulated.
    """

    def __init__(self, handler: Callable[[str, str | None], str] | None = None, script: list[str] | None = None):
        self.handler = handler
        self.script = list(script or [])
        self.calls: list[str] = []

    def complete(self, prompt: str, system: str | None = None) -> str:
        self.calls.append(prompt)
        if self.handler:
            return self.handler(prompt, system)
        if not self.script:
            return ""
        idx = min(len(self.calls) - 1, len(self.script) - 1)
        return self.script[idx]


class AnthropicProvider:
    """Real model calls. Untested against the live API (tested with a fake client only)."""

    def __init__(self, model: str | None = None, client=None, max_tokens: int = 512):
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens
        self._client = client

    def _get_client(self):
        if self._client is None:
            import anthropic  # lazy: only needed in live mode

            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        return self._client

    def complete(self, prompt: str, system: str | None = None) -> str:
        kwargs = {"model": self.model, "max_tokens": self.max_tokens, "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system
        resp = self._get_client().messages.create(**kwargs)
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
