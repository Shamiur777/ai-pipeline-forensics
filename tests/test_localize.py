import pytest

from forensics.dataset import EvalStore, add_failure_case
from forensics.localize import RunResult, StepSpec, earliest_bad_step, localize
from forensics.pipeline import known_good_outputs, make_run_fn, run_pipeline, support_specs
from forensics.validators import Custom

CLEAN_TICKET = "I was charged twice and want a refund for my last invoice."
SHIP_TICKET = "My package has not arrived and the tracking page is empty."


def report_for(ticket, faults):
    run = run_pipeline(ticket, faults)
    return run, localize(make_run_fn(faults), ticket, support_specs(), known_good_outputs(ticket), run)


def test_clean_run_passes():
    _, report = report_for(CLEAN_TICKET, ())
    assert report.final_ok and report.root_cause is None


def test_upstream_fault_is_root_and_downstream_are_victims():
    run, report = report_for(CLEAN_TICKET, ("classify_wrong",))
    assert report.root_cause == "classify" and report.earliest_bad == "classify"
    assert report.entry("draft").role == "downstream_victim"
    assert report.entry("safety").role == "downstream_victim"
    assert report.confidence == "high"


def test_tricky_case_first_suspicious_step_is_not_root_cause():
    faults = ("retrieve_verbose", "draft_hallucination")
    run, report = report_for(SHIP_TICKET, faults)
    assert earliest_bad_step(run, support_specs(), SHIP_TICKET) == "retrieve"  # naive answer
    assert report.root_cause == "draft"  # counterfactual answer
    assert report.entry("retrieve").role == "false_alarm"
    assert report.entry("retrieve").fixes_final is False
    assert report.entry("draft").fixes_final is True


@pytest.mark.parametrize("fault,step", [("retrieve_empty", "retrieve"), ("safety_overblock", "safety"),
                                        ("draft_hallucination", "draft"), ("classify_wrong", "classify")])
def test_each_single_fault_is_localised(fault, step):
    _, report = report_for(CLEAN_TICKET, (fault,))
    assert not report.final_ok and report.root_cause == step


def test_harmless_decoy_alone_is_not_a_failure():
    _, report = report_for(SHIP_TICKET, ("retrieve_verbose",))
    assert report.final_ok


def test_unresolved_when_two_independent_faults_are_needed():
    # both a and b break the final output; fixing only one is not enough
    specs = [StepSpec("a", [Custom(lambda o, c: o == "ok", "a_ok")]),
             StepSpec("b", [Custom(lambda o, c: o == "ok", "b_ok")]),
             StepSpec("final", [Custom(lambda o, c: o == "okok", "final_ok")])]

    def run_fn(inp, overrides):
        a = overrides.get("a", "bad")
        b = overrides.get("b", "bad")
        return RunResult({"a": a, "b": b, "final": a + b})

    report = localize(run_fn, None, specs, {"a": "ok", "b": "ok"})
    assert report.root_cause is None and report.earliest_bad == "a" and report.confidence == "low"
    assert report.entry("a").role == "unresolved_suspect"


def test_step_exception_is_reported_as_failure():
    specs = [StepSpec("a", []), StepSpec("final", [Custom(lambda o, c: True)])]
    run = RunResult({"a": 1}, {"final": "RuntimeError: boom"})
    assert earliest_bad_step(run, specs, None) == "final"


def test_failure_is_appended_to_dataset_with_counterfactual_expected(tmp_path):
    store = EvalStore(tmp_path / "ds.jsonl")
    _, report = report_for(CLEAN_TICKET, ("draft_hallucination",))
    assert add_failure_case(store, CLEAN_TICKET, report)
    rec = store.records()[0]
    assert "14 days" in rec["expected"]
    assert {"step:draft", "severity:high", "source:pipeline_forensics"} <= set(rec["tags"])
    assert not add_failure_case(store, CLEAN_TICKET, report)  # deduped


def test_unresolved_failure_is_marked_needs_review(tmp_path):
    from forensics.localize import BlameEntry, BlameReport
    store = EvalStore(tmp_path / "ds.jsonl")
    rep = BlameReport(False, "a", None, "low", [BlameEntry("a", "unresolved_suspect", 0.5)])
    add_failure_case(store, "some input", rep)
    rec = store.records()[0]
    assert rec["expected"] is None and "needs_review" in rec["tags"]
    assert any(t.startswith("reason:") for t in rec["tags"])
