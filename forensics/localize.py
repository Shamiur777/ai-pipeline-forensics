"""Failure localisation: earliest-bad-step, counterfactual replay, ranked blame report.

Generic over any pipeline: you supply `run_fn(input, overrides) -> RunResult`, per-step
validators, and known-good outputs per step. The last step is the pipeline's final output.

Counterfactual rule: replay with the known-good output substituted at step k. The root
cause is the EARLIEST step whose substitution makes the final output pass. A step that
fails its validators but whose substitution does not help is a false alarm; a later step
that fails and whose substitution helps is a downstream victim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .validators import SEVERITY_ORDER, Result, Validator, run_validators


@dataclass
class RunResult:
    steps: dict[str, Any]  # step name -> output, in execution order
    errors: dict[str, str] = field(default_factory=dict)


@dataclass
class StepSpec:
    name: str
    validators: list[Validator]


@dataclass
class BlameEntry:
    step: str
    role: str  # root_cause | downstream_victim | false_alarm | ok | unresolved_suspect
    score: float
    failed_checks: list[Result] = field(default_factory=list)
    fixes_final: bool | None = None  # None = not tested (no known-good output)
    replay_final: Any = None
    evidence: list[str] = field(default_factory=list)


@dataclass
class BlameReport:
    final_ok: bool
    earliest_bad: str | None
    root_cause: str | None
    confidence: str
    entries: list[BlameEntry]
    severity: str = "medium"

    def entry(self, step: str) -> BlameEntry:
        return next(e for e in self.entries if e.step == step)

    def render(self) -> str:
        if self.final_ok:
            return "Final output passed all checks. Nothing to localise."
        lines = [
            "Final output: BAD",
            f"Earliest suspicious step: {self.earliest_bad or 'none flagged'}",
            f"Root cause (counterfactual): {self.root_cause or 'unresolved'}  [confidence: {self.confidence}]",
            "Counterfactual replay (known-good output substituted at each step):",
        ]
        for e in self.entries:
            verdict = {True: "final becomes GOOD", False: "final still bad", None: "not tested"}[e.fixes_final]
            lines.append(f"  {e.step:<10} -> {verdict}")
        lines.append("Ranked blame:")
        for rank, e in enumerate(sorted(self.entries, key=lambda x: -x.score), 1):
            if e.role == "ok":
                continue
            lines.append(f"  {rank}. {e.step:<10} {e.role:<18} score={e.score:.2f}")
            for ev in e.evidence:
                lines.append(f"       - {ev}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "final_ok": self.final_ok, "earliest_bad": self.earliest_bad, "root_cause": self.root_cause,
            "confidence": self.confidence, "severity": self.severity,
            "entries": [
                {"step": e.step, "role": e.role, "score": e.score, "fixes_final": e.fixes_final,
                 "failed": [f"{r.validator}: {r.reason}" for r in e.failed_checks]}
                for e in self.entries
            ],
        }


def evaluate_steps(run: RunResult, specs: list[StepSpec], ctx: Any) -> dict[str, list[Result]]:
    """Failed checks per step. A step that raised, or never ran, counts as failed."""
    failed: dict[str, list[Result]] = {}
    for spec in specs:
        if spec.name in run.errors:
            failed[spec.name] = [Result(False, run.errors[spec.name], "exception", "high")]
        elif spec.name not in run.steps:
            failed[spec.name] = [Result(False, "step did not run", "missing")]
        else:
            failed[spec.name] = [r for r in run_validators(spec.validators, run.steps[spec.name], ctx) if not r.ok]
    return failed


def earliest_bad_step(run: RunResult, specs: list[StepSpec], ctx: Any) -> str | None:
    failed = evaluate_steps(run, specs, ctx)
    return next((s.name for s in specs if failed[s.name]), None)


def final_passes(run: RunResult, specs: list[StepSpec], ctx: Any) -> bool:
    last = specs[-1]
    return not evaluate_steps(run, [last], ctx)[last.name]


def localize(
    run_fn: Callable[[Any, dict], RunResult],
    input: Any,
    specs: list[StepSpec],
    good_outputs: dict[str, Any],
    bad_run: RunResult | None = None,
) -> BlameReport:
    bad_run = bad_run or run_fn(input, {})
    failed = evaluate_steps(bad_run, specs, input)
    final_ok = final_passes(bad_run, specs, input)
    earliest = next((s.name for s in specs if failed[s.name]), None)
    if final_ok:
        return BlameReport(True, earliest, None, "n/a", [BlameEntry(s.name, "ok", 0.0) for s in specs])

    entries: dict[str, BlameEntry] = {}
    for spec in specs:
        entry = BlameEntry(spec.name, "ok", 0.0, failed_checks=failed[spec.name])
        if spec.name in good_outputs:
            replay = run_fn(input, {spec.name: good_outputs[spec.name]})
            entry.fixes_final = final_passes(replay, specs, input)
            entry.replay_final = list(replay.steps.values())[-1] if replay.steps else None
        entries[spec.name] = entry

    root = next((s.name for s in specs if entries[s.name].fixes_final), None)
    order = [s.name for s in specs]
    victims = 0
    for name in order:
        e = entries[name]
        flagged = bool(e.failed_checks)
        for r in e.failed_checks:
            e.evidence.append(f"validator {r.validator}: {r.reason}")
        if name == root:
            e.role, e.score = "root_cause", 1.0
            e.evidence.append("substituting a known-good output here makes the final output pass")
            if not flagged:
                e.evidence.append("no validator flagged this step (validator gap)")
        elif root and order.index(name) > order.index(root):
            if flagged:
                victims += 1
                e.role, e.score = "downstream_victim", round(0.6 - 0.05 * victims, 2)
                e.evidence.append(f"fails only because of upstream '{root}'")
        elif flagged:
            if root:
                e.role, e.score = "false_alarm", 0.2
                e.evidence.append("replacing it with a known-good output does not fix the final output")
            else:
                e.role, e.score = "unresolved_suspect", 0.5 if name == earliest else 0.3
                e.evidence.append("no single-step substitution fixed the final output")

    if root:
        confidence = "high" if failed[root] else "medium"
    elif earliest:
        confidence = "low"
    else:
        confidence = "none"
    cause = root or earliest
    severity = "medium"
    if cause:
        sev = [r.severity for r in entries[cause].failed_checks]
        severity = max(sev, key=lambda s: SEVERITY_ORDER[s]) if sev else "medium"
    return BlameReport(False, earliest, root, confidence, list(entries.values()), severity)
