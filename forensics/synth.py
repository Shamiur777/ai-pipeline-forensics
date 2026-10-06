"""Seeded synthetic production-log generator with planted failure modes (all data is fictional).

Ground truth is returned separately (log id -> mode) so the miner never sees it.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

PRODUCTS = ["the mobile app", "the web dashboard", "the desktop client", "the browser plugin", "the admin portal",
            "the reporting tool", "the billing page", "the API console"]
NORMAL = [
    "How do I reset my password for {p}?",
    "Where can I find my invoices in {p}?",
    "Can you explain how notifications work in {p}?",
    "How do I export my data from {p}?",
    "What is the difference between the free and paid plan in {p}?",
    "How do I invite a teammate to {p}?",
    "Why is {p} loading slowly today?",
    "How do I change the language settings in {p}?",
    "Can I schedule a weekly summary email from {p}?",
    "How do I delete my account from {p}?",
    "Which browsers does {p} support?",
    "How do I connect {p} to my calendar?",
]
RARE = [
    "Translate this clause into formal Bengali and keep the legal tone: {x}",
    "Write a haiku about {x} in the style of a ship captain.",
    "Convert this recipe for {x} to metric units and halve the servings.",
    "Explain the chess opening {x} to a ten year old.",
    "Give me a regex that validates {x} identifiers with a checksum.",
    "Draft a polite eviction notice mentioning {x}.",
]
RARE_SLOTS = ["the lease agreement", "monsoon tides", "lentil soup", "the Sicilian Defense", "ISBN-like", "unpaid rent"]
MODES = ["error", "thumbs_down", "user_edit", "retry", "refusal", "latency_spike", "runaway_long", "too_short",
         "rare_edge"]
FILLER = ("open settings choose the option you need confirm the change and refresh the page if nothing happens "
          "contact support with the error code shown on screen").split()
MODELS = ["model-a", "model-b"]


def _response(rng: random.Random, p: str) -> str:
    return f"Open {p}, then " + " ".join(rng.choice(FILLER) for _ in range(rng.randint(20, 45))) + "."


def generate_logs(n: int, seed: int, planted_rate: float = 0.10, prefix: str = "L") -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    logs, planted = [], {}
    for i in range(n):
        p = rng.choice(PRODUCTS)
        req = rng.choice(NORMAL).format(p=p)
        log = {"id": f"{prefix}{seed:03d}-{i:05d}", "ts": 1_700_000_000 + seed * 100_000 + i * 30,
               "model": rng.choice(MODELS), "prompt_version": rng.choice(["v1", "v2"]),
               "request": req, "response": _response(rng, p),
               "latency_ms": round(rng.lognormvariate(6.8, 0.25), 1), "error": None,
               "feedback": {"thumbs": rng.choice(["up", None, None]), "edited_response": None}, "retry": False}
        # unplanted noise: real users sometimes downvote fine answers, and some calls are just slow
        if rng.random() < 0.02:
            log["feedback"]["thumbs"] = "down"
        if rng.random() < 0.03:
            log["latency_ms"] = round(rng.uniform(2500, 4500), 1)
        if rng.random() < planted_rate:
            mode = rng.choice(MODES)
            planted[log["id"]] = mode
            if mode == "error":
                log.update(error=rng.choice(["upstream_timeout", "rate_limited", "context_overflow"]), response="")
            elif mode == "thumbs_down":
                log["response"] = f"{p.capitalize()} costs five dollars per year and never needs updating."
                log["feedback"]["thumbs"] = "down"
            elif mode == "user_edit":
                log["feedback"]["edited_response"] = f"Go to Settings > Help in {p}, then choose Contact support."
            elif mode == "retry":
                log["retry"] = True
            elif mode == "refusal":
                log["response"] = "I'm sorry, but I can't help with that request."
            elif mode == "latency_spike":
                log["latency_ms"] = round(rng.uniform(7000, 15000), 1)
            elif mode == "runaway_long":
                log["response"] = " ".join(rng.choice(FILLER) for _ in range(400))
            elif mode == "too_short":
                log["response"] = "Yes."
            elif mode == "rare_edge":
                j = rng.randrange(len(RARE))
                log["request"] = RARE[j].format(x=RARE_SLOTS[j])
                log["feedback"]["thumbs"] = None
        logs.append(log)
    return logs, planted


def write_logs(path: str | Path, logs: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(l) for l in logs) + "\n", encoding="utf-8")
