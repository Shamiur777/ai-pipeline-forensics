import json
from collections import Counter

from forensics.dataset import EvalStore
from forensics.evaluate import evaluate_miner
from forensics.miner import MinerConfig, batch_stats, detect_signals, load_state, mine
from forensics.synth import generate_logs, write_logs


def log(**kw):
    base = {"id": "x", "model": "m", "prompt_version": "v1", "request": "how do I reset my password today",
            "response": "Open settings and choose reset " * 3, "latency_ms": 900, "error": None,
            "feedback": {"thumbs": None, "edited_response": None}, "retry": False}
    base.update(kw)
    return base


def sigs(l, population=None):
    pop = population or [log(id=str(i)) for i in range(30)]
    return detect_signals(l, batch_stats(pop))


def test_explicit_signals():
    assert sigs(log()) == []
    assert "error" in sigs(log(error="timeout"))
    assert "thumbs_down" in sigs(log(feedback={"thumbs": "down", "edited_response": None}))
    assert "user_edit" in sigs(log(feedback={"thumbs": None, "edited_response": "better answer"}))
    assert "retry" in sigs(log(retry=True))
    assert "refusal" in sigs(log(response="I'm sorry, but I can't help with that."))
    assert "latency_outlier" in sigs(log(latency_ms=20000))
    assert "length_anomaly" in sigs(log(response="Yes."))
    assert "length_anomaly" in sigs(log(response="word " * 500))


def test_edit_identical_to_response_is_not_a_signal():
    resp = "Open settings"
    assert "user_edit" not in sigs(log(response=resp, feedback={"thumbs": None, "edited_response": resp}))


def test_user_correction_becomes_expected_and_others_need_review(tmp_path):
    logs = [log(id=f"a{i:02d}", request=f"how do I reset password number {i}") for i in range(30)]
    logs[0]["feedback"] = {"thumbs": None, "edited_response": "Use Settings > Security."}
    logs[1]["feedback"] = {"thumbs": "down", "edited_response": None}
    write_logs(tmp_path / "l.jsonl", logs)
    store = EvalStore(tmp_path / "d.jsonl")
    mine(tmp_path / "l.jsonl", store, tmp_path / "s.json")
    by_log = {r["meta"]["log_id"]: r for r in store.records()}
    assert by_log["a00"]["expected"] == "Use Settings > Security."
    assert by_log["a01"]["expected"] is None and "needs_review" in by_log["a01"]["tags"]
    assert by_log["a01"]["meta"]["signals"][0] == "thumbs_down"


def test_duplicates_in_a_run_are_collapsed(tmp_path):
    logs = [log(id=f"a{i:02d}", error="timeout", request="Same prompt, again!") for i in range(10)]
    logs += [log(id=f"b{i:02d}", request=f"routine question {i}") for i in range(30)]
    write_logs(tmp_path / "l.jsonl", logs)
    store = EvalStore(tmp_path / "d.jsonl")
    rep = mine(tmp_path / "l.jsonl", store, tmp_path / "s.json")
    assert sum(1 for r in store.records() if r["input"] == "Same prompt, again!") == 1
    assert rep.dups_removed >= 1


def test_stratification_stops_one_failure_type_dominating(tmp_path):
    logs = [log(id=f"e{i:03d}", error="timeout", request=f"unique error prompt {i} " + "z" * i) for i in range(40)]
    logs += [log(id=f"t{i:03d}", feedback={"thumbs": "down", "edited_response": None},
                 request=f"another thumbs prompt about topic {i} {'q' * i}") for i in range(4)]
    logs += [log(id=f"r{i:03d}", retry=True, request=f"retry prompt {i} {'w' * i}") for i in range(4)]
    logs += [log(id=f"n{i:03d}", request=f"routine {i}") for i in range(100)]
    write_logs(tmp_path / "l.jsonl", logs)
    store = EvalStore(tmp_path / "d.jsonl")
    mine(tmp_path / "l.jsonl", store, tmp_path / "s.json", MinerConfig(budget=20, max_share=0.4))
    types = Counter(t for r in store.records() for t in r["tags"] if t.startswith("type:"))
    assert types["type:error"] <= 8  # 40% of a budget of 20
    assert types["type:thumbs_down"] >= 3 and types["type:retry"] >= 3  # per-cluster cap is 3


def test_incremental_state_skips_seen_logs_and_records_growth(tmp_path):
    store = EvalStore(tmp_path / "d.jsonl")
    state = tmp_path / "s.json"
    logs1, _ = generate_logs(200, 1)
    write_logs(tmp_path / "l1.jsonl", logs1)
    r1 = mine(tmp_path / "l1.jsonl", store, state)
    again = mine(tmp_path / "l1.jsonl", store, state)  # same file: nothing new
    assert again.new_logs == 0 and again.added == 0 and again.dataset_size == r1.dataset_size
    logs2, _ = generate_logs(200, 2)
    write_logs(tmp_path / "l2.jsonl", logs2)
    r3 = mine(tmp_path / "l2.jsonl", store, state)
    assert r3.dataset_size >= r1.dataset_size and r3.new_logs == 200
    assert [r["run"] for r in load_state(state)["runs"]] == [1, 2, 3]
    assert len(json.loads(state.read_text())["seen_ids"]) == 400


def test_provenance_and_tags_recorded(tmp_path):
    logs, _ = generate_logs(200, 3)
    write_logs(tmp_path / "l.jsonl", logs)
    store = EvalStore(tmp_path / "d.jsonl")
    mine(tmp_path / "l.jsonl", store, tmp_path / "s.json")
    rec = store.records()[0]
    assert {"log_id", "run", "signals", "score", "cluster_size"} <= set(rec["meta"])
    assert "source:prod_log" in rec["tags"] and any(t.startswith("model:") for t in rec["tags"])


def test_planted_failure_recall_regression():
    """Regression gate: measured on synthetic logs (3 incremental runs, seeds 7-9)."""
    result = evaluate_miner(n=400, seed=7)
    for run in result["runs"]:
        assert run["recall"] >= 0.75, run
        assert run["precision_of_added"] >= 0.5, run
    assert result["max_type_share"] <= 0.4
    sizes = [r["dataset_size"] for r in result["runs"]]
    assert sizes == sorted(sizes) and sizes[-1] > sizes[0]
