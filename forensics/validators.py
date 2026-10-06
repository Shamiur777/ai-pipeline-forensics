"""Pluggable per-step validators. Each returns a Result; a crash inside a validator is a failure."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .provider import Provider

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass
class Result:
    ok: bool
    reason: str
    validator: str
    severity: str = "medium"


def _field(output: Any, field: str | None) -> Any:
    if field is None:
        return output
    if not isinstance(output, dict) or field not in output:
        raise KeyError(f"missing field '{field}'")
    return output[field]


class Validator:
    name = "validator"
    severity = "medium"

    def check(self, output: Any, ctx: Any = None) -> Result:  # pragma: no cover - interface
        raise NotImplementedError

    def _r(self, ok: bool, reason: str) -> Result:
        return Result(ok, reason, self.name, self.severity)


class Schema(Validator):
    """Output must be a dict with required keys of given types; optional allowed values."""

    name = "schema"

    def __init__(self, required: dict[str, type], allowed: dict[str, set] | None = None):
        self.required, self.allowed = required, allowed or {}

    def check(self, output: Any, ctx: Any = None) -> Result:
        if not isinstance(output, dict):
            return self._r(False, f"expected object, got {type(output).__name__}")
        for key, typ in self.required.items():
            if key not in output:
                return self._r(False, f"missing key '{key}'")
            if not isinstance(output[key], typ):
                return self._r(False, f"'{key}' should be {typ.__name__}, got {type(output[key]).__name__}")
        for key, values in self.allowed.items():
            if output.get(key) not in values:
                return self._r(False, f"'{key}'={output.get(key)!r} not in {sorted(values)}")
        return self._r(True, "ok")


class Contains(Validator):
    """Text must contain all/any of `texts`; with negate=True it must contain none of them."""

    def __init__(self, texts: str | list[str], field: str | None = None, mode: str = "all",
                 negate: bool = False, severity: str = "medium", label: str | None = None):
        self.texts = [texts] if isinstance(texts, str) else list(texts)
        self.field, self.mode, self.negate, self.severity = field, mode, negate, severity
        self.name = label or ("not_contains" if negate else "contains")

    def check(self, output: Any, ctx: Any = None) -> Result:
        try:
            text = str(_field(output, self.field)).lower()
        except KeyError as exc:
            return self._r(False, str(exc))
        hits = [t for t in self.texts if t.lower() in text]
        if self.negate:
            return self._r(not hits, "ok" if not hits else f"contains forbidden text {hits}")
        good = len(hits) == len(self.texts) if self.mode == "all" else bool(hits)
        return self._r(good, "ok" if good else f"missing {[t for t in self.texts if t not in hits]}")


class Length(Validator):
    name = "length"

    def __init__(self, min: int = 0, max: int | None = None, field: str | None = None):
        self.min, self.max, self.field = min, max, field

    def check(self, output: Any, ctx: Any = None) -> Result:
        try:
            n = len(str(_field(output, self.field)))
        except KeyError as exc:
            return self._r(False, str(exc))
        if n < self.min:
            return self._r(False, f"{n} chars < min {self.min}")
        if self.max is not None and n > self.max:
            return self._r(False, f"{n} chars > max {self.max}")
        return self._r(True, "ok")


class Custom(Validator):
    """Wrap `fn(output, ctx) -> bool | (bool, reason)`."""

    def __init__(self, fn: Callable[[Any, Any], Any], name: str = "custom", severity: str = "medium"):
        self.fn, self.name, self.severity = fn, name, severity

    def check(self, output: Any, ctx: Any = None) -> Result:
        res = self.fn(output, ctx)
        ok, reason = res if isinstance(res, tuple) else (bool(res), "ok" if res else "check returned False")
        return self._r(ok, reason)


class LLMJudge(Validator):
    """Optional model-based check. Replies must start with PASS or FAIL."""

    name = "llm_judge"

    def __init__(self, provider: Provider, rubric: str):
        self.provider, self.rubric = provider, rubric

    def check(self, output: Any, ctx: Any = None) -> Result:
        reply = self.provider.complete(
            f"Rubric: {self.rubric}\nOutput to judge: {output}\nAnswer with PASS or FAIL, then a short reason."
        ).strip()
        return self._r(reply.upper().startswith("PASS"), reply[:120] or "empty judge reply")


def run_validators(validators: list[Validator], output: Any, ctx: Any = None) -> list[Result]:
    results = []
    for v in validators:
        try:
            results.append(v.check(output, ctx))
        except Exception as exc:  # a broken validator must not hide a failure
            results.append(Result(False, f"validator crashed: {type(exc).__name__}: {exc}", v.name, v.severity))
    return results
