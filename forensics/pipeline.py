"""Example pipeline: support-ticket assistant (classify -> retrieve -> draft -> safety).

The "model" is a scripted simulation, so every injected fault is reproducible.
Nothing here measures real model quality.
"""
from __future__ import annotations

from typing import Any

from .localize import RunResult, StepSpec
from .provider import MockProvider, Provider
from .tracing import Tracer
from .validators import Contains, Custom, Length, Schema

STEPS = ("classify", "retrieve", "draft", "safety")
LABELS = ("billing", "shipping", "technical", "other")

POLICIES = {
    "billing": "Refunds are issued within 14 days of purchase to the original payment method.",
    "shipping": "Orders ship within 2 business days and a tracking number is emailed on dispatch.",
    "technical": "Reset your password under Settings > Security; outages are posted on the status page.",
    "other": "A team member will review your request and reply within 3 business days.",
}
KEY_FACT = {"billing": "14 days", "shipping": "2 business days", "technical": "Settings > Security",
            "other": "3 business days"}
KEYWORDS = {
    "billing": ("refund", "charge", "invoice", "billed", "payment"),
    "shipping": ("package", "delivery", "shipped", "tracking", "order"),
    "technical": ("password", "login", "crash", "error", "app"),
}
FORBIDDEN = ("guarantee", "24 hours", "no questions asked")
FALLBACK = "We could not process this reply automatically; it has been sent for manual review."
BOILERPLATE = " Our company was founded to make support simple. Visit our blog for tips, product news and stories."

# fault name -> step where it is injected
FAULTS = {
    "classify_wrong": "classify",
    "retrieve_empty": "retrieve",
    "retrieve_verbose": "retrieve",  # harmless-looking: right policy plus noise. A decoy for earliest-bad.
    "draft_hallucination": "draft",
    "safety_overblock": "safety",
}

SAMPLE_TICKETS = [
    "I was charged twice and want a refund for my last invoice.",
    "My package has not arrived and the tracking page is empty.",
    "I forgot my password and the app keeps crashing at login.",
    "Where is my order? The delivery date passed yesterday.",
    "Please refund the payment, I was billed after cancelling.",
    "The app shows an error when I try to login.",
    "Can you tell me who the account manager for my region is?",
    "My invoice shows a charge I do not recognise.",
]


def true_topic(ticket: str) -> str:
    """Keyword oracle used by validators. A heuristic stand-in for a labelled check."""
    text = ticket.lower()
    for topic, words in KEYWORDS.items():
        if any(w in text for w in words):
            return topic
    return "other"


class SimulatedSupportModel:
    """Scripted stand-in for an LLM; prompt format is 'TASK: <name>\\nTICKET: ...'."""

    def __init__(self, faults: tuple[str, ...] = ()):
        self.faults = set(faults)

    def __call__(self, prompt: str, system: str | None = None) -> str:
        task = prompt.split("\n", 1)[0].removeprefix("TASK: ").strip()
        ticket = prompt.split("TICKET: ", 1)[1].split("\n", 1)[0]
        if task == "classify":
            label = true_topic(ticket)
            if "classify_wrong" in self.faults:
                label = LABELS[(LABELS.index(label) + 1) % len(LABELS)]
            return label
        if task == "draft":
            policy = prompt.split("POLICY: ", 1)[1]
            if "draft_hallucination" in self.faults:
                return "Hi! We guarantee a full refund within 24 hours, no questions asked."
            if not policy.strip():
                return "Hi, thanks for reaching out. We will look into this."
            return f"Hi, thanks for contacting us. {policy.strip()} Let us know if you need anything else."
        return ""


def make_provider(faults: tuple[str, ...] = ()) -> Provider:
    return MockProvider(handler=SimulatedSupportModel(faults))


def run_pipeline(
    ticket: str,
    faults: tuple[str, ...] = (),
    overrides: dict[str, Any] | None = None,
    tracer: Tracer | None = None,
    provider: Provider | None = None,
) -> RunResult:
    """Run the pipeline; `overrides` substitutes a step's output (used for counterfactual replay)."""
    unknown = set(faults) - set(FAULTS)
    if unknown:
        raise ValueError(f"unknown faults {sorted(unknown)}; choose from {sorted(FAULTS)}")
    tracer, overrides = tracer or Tracer(), overrides or {}
    provider = provider or make_provider(faults)
    steps: dict[str, Any] = {}
    errors: dict[str, str] = {}

    def step(name: str, inp: Any, fn):
        with tracer.span(name, input=inp) as sp:
            if name in overrides:
                out = overrides[name]
                sp.metadata["overridden"] = True
            else:
                out = fn()
            sp.set_output(out)
        steps[name] = out
        return out

    with tracer.span("pipeline", input={"ticket": ticket}, faults=list(faults), overrides=sorted(overrides)) as root:
        try:
            label = step("classify", {"ticket": ticket},
                         lambda: {"label": provider.complete(f"TASK: classify\nTICKET: {ticket}").strip()})["label"]

            def retrieve():
                text = "" if "retrieve_empty" in faults else POLICIES.get(label, POLICIES["other"])
                if "retrieve_verbose" in faults:
                    text += BOILERPLATE
                return {"policy_id": label, "text": text}

            policy = step("retrieve", {"label": label}, retrieve)["text"]
            reply = step("draft", {"ticket": ticket, "policy": policy},
                         lambda: {"reply": provider.complete(f"TASK: draft\nTICKET: {ticket}\nPOLICY: {policy}")})["reply"]

            def safety():
                ok = "safety_overblock" not in faults and not any(p in reply.lower() for p in FORBIDDEN)
                return {"approved": ok, "reply": reply if ok else FALLBACK}

            step("safety", {"reply": reply}, safety)
        except Exception as exc:
            failing = next((n for n in STEPS if n not in steps), STEPS[-1])
            errors[failing] = f"{type(exc).__name__}: {exc}"
        root.set_output(steps.get("safety"))
    return RunResult(steps, errors)


def known_good_outputs(ticket: str) -> dict[str, Any]:
    """Outputs of a fault-free run: the 'known good' reference for counterfactual replay."""
    return run_pipeline(ticket).steps


def _has_key_fact(field: str | None):
    def check(output, ticket):
        text = output if field is None else output.get(field, "")
        fact = KEY_FACT[true_topic(ticket)]
        return (fact.lower() in str(text).lower(), f"reply lacks policy fact '{fact}'")
    return check


def support_specs() -> list[StepSpec]:
    def label_matches(output, ticket):
        expected = true_topic(ticket)
        return (output.get("label") == expected, f"label {output.get('label')!r} but ticket looks like {expected!r}")

    def policy_matches(output, ticket):
        expected = true_topic(ticket)
        return (output.get("policy_id") == expected, f"policy {output.get('policy_id')!r}, expected {expected!r}")

    no_forbidden = lambda field: Contains(list(FORBIDDEN), field=field, negate=True, severity="high",
                                          label="forbidden_claim")
    return [
        StepSpec("classify", [Schema({"label": str}, {"label": set(LABELS)}), Custom(label_matches, "label_consistency")]),
        StepSpec("retrieve", [Schema({"policy_id": str, "text": str}), Length(20, 120, field="text"),
                              Custom(policy_matches, "policy_matches_topic")]),
        StepSpec("draft", [Schema({"reply": str}), no_forbidden("reply"),
                           Custom(_has_key_fact("reply"), "mentions_policy_fact")]),
        StepSpec("safety", [Schema({"approved": bool, "reply": str}), no_forbidden("reply"),
                            Custom(_has_key_fact("reply"), "mentions_policy_fact"), Length(20, 500, field="reply")]),
    ]


def make_run_fn(faults: tuple[str, ...] = ()):
    return lambda ticket, overrides: run_pipeline(ticket, faults, overrides)
