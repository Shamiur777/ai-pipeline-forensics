"""Measure the miner against planted failures and the localiser against injected faults.

    python -m forensics.evaluate

Everything is synthetic and simulated; the numbers describe these generators, not production.
"""
from __future__ import annotations

import argparse
import tempfile
from collections import Counter
from pathlib import Path

from .dataset import EvalStore, input_hash
from .localize import earliest_bad_step, localize
from .miner import MinerConfig, mine
from .pipeline import SAMPLE_TICKETS, known_good_outputs, make_run_fn, run_pipeline, support_specs
from .synth import generate_logs, write_logs

# (faults, ground-truth root cause step). retrieve_verbose is a harmless decoy.
SCENARIOS = [
    (("classify_wrong",), "classify"),
    (("retrieve_empty",), "retrieve"),
    (("draft_hallucination",), "draft"),
    (("safety_overblock",), "safety"),
    (("retrieve_verbose", "draft_hallucination"), "draft"),
    (("retrieve_verbose", "safety_overblock"), "safety"),
    (("retrieve_verbose", "classify_wrong"), "classify"),
]


def evaluate_localisation() -> dict:
    specs, total, earliest_ok, cf_ok, decoy_total, decoy_earliest_ok = support_specs(), 0, 0, 0, 0, 0
    decoy_cf_ok = 0
    for faults, truth in SCENARIOS:
        for ticket in SAMPLE_TICKETS:
            bad = run_pipeline(ticket, faults)
            report = localize(make_run_fn(faults), ticket, specs, known_good_outputs(ticket), bad)
            if report.final_ok:
                continue
            e_ok = earliest_bad_step(bad, specs, ticket) == truth
            c_ok = report.root_cause == truth
            total += 1
            earliest_ok += e_ok
            cf_ok += c_ok
            if "retrieve_verbose" in faults:
                decoy_total += 1
                decoy_earliest_ok += e_ok
                decoy_cf_ok += c_ok
    return {"cases": total, "earliest_bad_correct": earliest_ok, "counterfactual_correct": cf_ok,
            "decoy_cases": decoy_total, "decoy_earliest_correct": decoy_earliest_ok,
            "decoy_counterfactual_correct": decoy_cf_ok}


def evaluate_miner(n: int = 400, seed: int = 7, cfg: MinerConfig | None = None, workdir: Path | None = None) -> dict:
    cfg = cfg or MinerConfig()
    tmp = None
    if workdir is None:
        tmp = tempfile.TemporaryDirectory()
        workdir = Path(tmp.name)
    store = EvalStore(workdir / "dataset.jsonl")
    state = workdir / "state.json"
    runs, all_planted = [], {}
    for k in range(3):
        logs, planted = generate_logs(n, seed + k, prefix="L")
        all_planted.update(planted)
        by_id = {l["id"]: l for l in logs}
        path = workdir / f"logs_{k + 1}.jsonl"
        write_logs(path, logs)
        before = {r["input"] for r in store.records()}
        rep = mine(path, store, state, cfg)
        added = [r for r in store.records() if r["input"] not in before]
        added_logs = [r["meta"]["log_id"] for r in added]
        owner = {input_hash(r["input"]): r["meta"]["log_id"] for r in store.records()}
        # covered = the planted log's normalised input is in the dataset AND the stored case came from a planted log
        covered = sum(1 for lid in planted if owner.get(input_hash(by_id[lid]["request"])) in all_planted)
        runs.append({
            "run": rep.run, "logs": rep.new_logs, "planted": len(planted), "flagged": rep.flagged,
            "dups_removed": rep.dups_removed, "selected": rep.selected, "added": rep.added,
            "precision_of_added": round(sum(l in planted for l in added_logs) / max(1, len(added_logs)), 3),
            "planted_covered": covered, "recall": round(covered / max(1, len(planted)), 3),
            "dataset_size": rep.dataset_size, "by_type": rep.by_type,
        })
    recs = store.records()
    types = Counter(t for r in recs for t in r["tags"] if t.startswith("type:"))
    out = {"runs": runs, "final_size": len(recs), "type_counts": dict(types),
           "max_type_share": round(max(types.values()) / len(recs), 3) if recs else 0,
           "with_expected": sum(r["expected"] is not None for r in recs),
           "needs_review": sum("needs_review" in r["tags"] for r in recs)}
    if tmp:
        tmp.cleanup()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    loc = evaluate_localisation()
    print("LOCALISATION (simulated pipeline, injected faults)")
    for k, v in loc.items():
        print(f"  {k}: {v}")
    m = evaluate_miner(args.n, args.seed)
    print(f"\nMINER (synthetic logs, n={args.n} per run, seeds {args.seed}..{args.seed + 2})")
    for r in m["runs"]:
        print(" ", r)
    for k in ("final_size", "type_counts", "max_type_share", "with_expected", "needs_review"):
        print(f"  {k}: {m[k]}")


if __name__ == "__main__":
    main()
