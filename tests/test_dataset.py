from forensics.dataset import EvalStore, input_hash, normalize_input


def test_normalisation_dedupes_trivial_variants(tmp_path):
    assert normalize_input("  How do I RESET my password?! ") == normalize_input("how do i reset my password")
    store = EvalStore(tmp_path / "d.jsonl")
    assert store.add("How do I reset my password?", "x", ["type:a"])
    assert not store.add("how do i  reset my password", "y", ["type:a"])
    assert len(store.records()) == 1


def test_golden_schema_fields_and_reload(tmp_path):
    path = tmp_path / "d.jsonl"
    store = EvalStore(path)
    store.add("q1", "a1", ["type:a"], weight=2.0)
    store.add("q2", None, ["type:b"], needs_review_reason="thumbs_down: no correction")
    rec1, rec2 = store.records()
    assert {"id", "input", "expected", "tags", "weight"} <= set(rec1)
    assert rec1["id"] == f"ev_{input_hash('q1')}" and rec1["weight"] == 2.0
    assert rec2["expected"] is None and "needs_review" in rec2["tags"]
    assert "reason:thumbs_down: no correction" in rec2["tags"]
    # a fresh store over the same file still dedupes
    assert not EvalStore(path).add("Q1!", "z")
