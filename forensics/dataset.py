"""Eval dataset store: append-only JSONL in golden-dataset shape {id, input, expected, tags, weight}.

`tags` is a list of "key:value" strings. `expected` is null when a human must fill it in
(the case then carries the tag `needs_review` and a `reason:` tag). An extra `meta`
object holds provenance; consumers that only read the five core fields can ignore it.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .localize import BlameReport

SEVERITY_WEIGHT = {"low": 0.5, "medium": 1.0, "high": 2.0}


def normalize_input(text: Any) -> str:
    """Lowercase, drop punctuation, collapse whitespace, so trivial variants dedupe."""
    if not isinstance(text, str):
        text = json.dumps(text, sort_keys=True, default=repr)
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def input_hash(text: Any) -> str:
    return hashlib.sha1(normalize_input(text).encode("utf-8")).hexdigest()[:16]


class EvalStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._hashes = {input_hash(r["input"]) for r in self.records()}

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def __len__(self) -> int:
        return len(self._hashes)

    def has(self, input: Any) -> bool:
        return input_hash(input) in self._hashes

    def add(self, input: Any, expected: Any = None, tags: list[str] | None = None, weight: float = 1.0,
            meta: dict | None = None, needs_review_reason: str | None = None) -> bool:
        """Append a case. Returns False (and writes nothing) if the normalised input already exists."""
        h = input_hash(input)
        if h in self._hashes:
            return False
        tags = list(tags or [])
        if expected is None:
            tags += ["needs_review", f"reason:{needs_review_reason or 'no known-good output'}"]
        record = {"id": f"ev_{h}", "input": input, "expected": expected, "tags": tags, "weight": weight,
                  "meta": meta or {}}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._hashes.add(h)
        return True

    def stats(self) -> dict:
        recs = self.records()
        by = lambda prefix: dict(Counter(t.split(":", 1)[1] for r in recs for t in r["tags"] if t.startswith(prefix)))
        return {"total": len(recs), "needs_review": sum("needs_review" in r["tags"] for r in recs),
                "by_type": by("type:"), "by_source": by("source:"), "by_step": by("step:")}


def add_failure_case(store: EvalStore, ticket: str, report: BlameReport) -> bool:
    """Turn a localised pipeline failure into an eval case. Expected comes from the counterfactual."""
    if report.final_ok:
        return False
    cause = report.root_cause or report.earliest_bad
    failed = report.entry(cause).failed_checks if cause else []
    ftype = failed[0].validator if failed else "unknown"
    expected, reason = None, None
    if report.root_cause:
        fixed = report.entry(report.root_cause).replay_final
        if isinstance(fixed, dict) and fixed.get("reply"):
            expected = fixed["reply"]
    if expected is None:
        reason = "root cause not isolated by single-step counterfactual" if not report.root_cause \
            else "counterfactual produced no usable final output"
    tags = [f"step:{cause}", f"type:{ftype}", f"severity:{report.severity}", "source:pipeline_forensics"]
    meta = {"confidence": report.confidence, "earliest_bad": report.earliest_bad,
            "blame": [{"step": e.step, "role": e.role} for e in report.entries if e.role != "ok"]}
    return store.add(ticket, expected, tags, SEVERITY_WEIGHT[report.severity], meta, reason)
