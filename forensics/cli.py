"""CLI: python -m forensics analyze|mine|demo"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .dataset import EvalStore, add_failure_case
from .localize import RunResult, localize
from .miner import MinerConfig, mine
from .pipeline import FAULTS, known_good_outputs, make_run_fn, run_pipeline, support_specs
from .synth import generate_logs, write_logs
from .tracing import Tracer, read_spans


def analyze_trace(trace_path: str | Path, store: EvalStore | None = None) -> list:
    """Re-localise every pipeline run in a trace file (replay uses the faults recorded on the root span)."""
    spans = read_spans(trace_path)
    specs, reports = support_specs(), []
    for root in (s for s in spans if s.parent_id is None and s.name == "pipeline"):
        ticket, faults = root.input["ticket"], tuple(root.metadata.get("faults", []))
        steps = {s.name: s.output for s in spans if s.parent_id == root.span_id and s.trace_id == root.trace_id}
        errors = {s.name: s.error for s in spans if s.parent_id == root.span_id and s.error}
        report = localize(make_run_fn(faults), ticket, specs, known_good_outputs(ticket), RunResult(steps, errors))
        reports.append((ticket, faults, report))
        if store is not None:
            add_failure_case(store, ticket, report)
    return reports


def _print_growth(reports) -> None:
    print(f"{'run':>3} {'new logs':>8} {'flagged':>7} {'dups':>5} {'selected':>8} {'added':>5} {'dataset':>7}")
    for r in reports:
        print(f"{r.run:>3} {r.new_logs:>8} {r.flagged:>7} {r.dups_removed:>5} {r.selected:>8} {r.added:>5} {r.dataset_size:>7}")


def cmd_analyze(args) -> None:
    store = EvalStore(args.dataset) if args.dataset else None
    for ticket, faults, report in analyze_trace(args.trace, store):
        print(f"\nTicket: {ticket}\nRecorded faults: {list(faults) or 'none'}\n{report.render()}")
    if store:
        print(f"\nDataset now has {len(store)} cases ({args.dataset})")


def cmd_mine(args) -> None:
    store = EvalStore(args.dataset)
    rep = mine(args.logs, store, args.state, MinerConfig(budget=args.budget))
    _print_growth([rep])
    print(f"by failure type: {rep.by_type}")


def cmd_demo(args) -> None:
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    trace_path, ds_path, state_path = out / "trace.jsonl", out / "eval_dataset.jsonl", out / "miner_state.json"
    store, specs = EvalStore(ds_path), support_specs()
    print("All data and model behaviour below is simulated (scripted mock provider, synthetic logs).\n")

    cases = [
        ("1) Upstream fault: the classifier mislabels a billing ticket",
         "I was charged twice and want a refund for my last invoice.", ("classify_wrong",)),
        ("2) Tricky case: retrieve looks suspicious (decoy) but the draft step is the root cause",
         "My package has not arrived and the tracking page is empty.", ("retrieve_verbose", "draft_hallucination")),
    ]
    tracer = Tracer(trace_path)
    for title, ticket, faults in cases:
        print("=" * 78 + f"\n{title}\nTicket: {ticket}\nInjected faults: {list(faults)}")
        run = run_pipeline(ticket, faults, tracer=tracer)
        print(f"Final reply: {run.steps['safety']['reply']!r}\n")
        report = localize(make_run_fn(faults), ticket, specs, known_good_outputs(ticket), run)
        print(report.render())
        added = add_failure_case(store, ticket, report)
        print(f"-> appended to eval dataset: {added}\n")

    print("=" * 78 + "\nRe-analysing the saved trace file (python -m forensics analyze)")
    for ticket, faults, report in analyze_trace(trace_path):
        print(f"  {ticket[:50]!r}: root cause = {report.root_cause}, earliest bad = {report.earliest_bad}")

    print("\n" + "=" * 78 + "\nMining synthetic production logs, 3 incremental runs (seeded)")
    reports = []
    for k in range(3):
        logs, _ = generate_logs(args.logs, args.seed + k)
        path = out / f"prod_logs_{k + 1}.jsonl"
        write_logs(path, logs)
        reports.append(mine(path, store, state_path))
    _print_growth(reports)
    stats = store.stats()
    print(f"\nDataset: {stats['total']} cases, {stats['needs_review']} need review")
    print(f"by source: {stats['by_source']}\nby type:   {stats['by_type']}")
    print(f"Files in {out}/: trace.jsonl, eval_dataset.jsonl, miner_state.json, prod_logs_*.jsonl")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m forensics")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="localise failures in a trace JSONL file")
    a.add_argument("trace")
    a.add_argument("--dataset", help="append flagged failures to this eval dataset")
    a.set_defaults(fn=cmd_analyze)
    m = sub.add_parser("mine", help="mine a production log file into the eval dataset (incremental)")
    m.add_argument("logs")
    m.add_argument("--dataset", default="data/eval_dataset.jsonl")
    m.add_argument("--state", default="data/miner_state.json")
    m.add_argument("--budget", type=int, default=60)
    m.set_defaults(fn=cmd_mine)
    d = sub.add_parser("demo", help="end-to-end offline demo")
    d.add_argument("--out", default="demo_out")
    d.add_argument("--logs", type=int, default=400, help="synthetic log lines per run")
    d.add_argument("--seed", type=int, default=7)
    d.set_defaults(fn=cmd_demo)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
