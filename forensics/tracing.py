"""Zero-dependency tracing: nested spans written to JSONL.

    tracer = Tracer("trace.jsonl")
    with tracer.span("step", input=x) as sp:
        sp.set_output(y)

    @tracer.trace()
    def f(a, b): ...

Spans are written when they close, so children appear before their parent.
Exceptions are recorded on the span and then re-raised (never swallowed).
"""
from __future__ import annotations

import contextvars
import functools
import itertools
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator


def _safe(value: Any) -> Any:
    """Round-trip through JSON so spans never hold unserialisable objects."""
    return json.loads(json.dumps(value, default=repr))


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    input: Any = None
    output: Any = None
    start_ts: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    def set_output(self, value: Any) -> None:
        self.output = value

    def to_dict(self) -> dict:
        return asdict(self)


class Tracer:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.spans: list[Span] = []
        self._span_ids = itertools.count(1)
        self._trace_ids = itertools.count(1)
        self._stack: contextvars.ContextVar[tuple[Span, ...]] = contextvars.ContextVar(
            f"span_stack_{id(self)}", default=()
        )
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def span(self, name: str, input: Any = None, **metadata: Any) -> Iterator[Span]:
        stack = self._stack.get()
        parent = stack[-1] if stack else None
        sp = Span(
            trace_id=parent.trace_id if parent else f"t{next(self._trace_ids)}",
            span_id=f"s{next(self._span_ids)}",
            parent_id=parent.span_id if parent else None,
            name=name,
            input=input,
            start_ts=time.time(),
            metadata=dict(metadata),
        )
        token = self._stack.set(stack + (sp,))
        t0 = time.perf_counter()
        try:
            yield sp
        except BaseException as exc:
            sp.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            sp.latency_ms = round((time.perf_counter() - t0) * 1000, 3)
            self._stack.reset(token)
            self._emit(sp)

    def trace(self, name: str | None = None, **metadata: Any) -> Callable:
        """Decorator: record call arguments as input and the return value as output."""

        def wrap(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def inner(*args: Any, **kwargs: Any) -> Any:
                with self.span(name or fn.__name__, input={"args": list(args), "kwargs": kwargs}, **metadata) as sp:
                    result = fn(*args, **kwargs)
                    sp.set_output(result)
                    return result

            return inner

        return wrap

    def _emit(self, sp: Span) -> None:
        sp.input, sp.output, sp.metadata = _safe(sp.input), _safe(sp.output), _safe(sp.metadata)
        self.spans.append(sp)
        if self.path:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(sp.to_dict()) + "\n")


def read_spans(path: str | Path) -> list[Span]:
    spans = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            spans.append(Span(**json.loads(line)))
    return spans
