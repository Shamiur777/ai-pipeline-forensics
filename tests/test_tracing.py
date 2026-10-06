import pytest

from forensics.tracing import Tracer, read_spans


def test_nested_spans_have_parent_links():
    t = Tracer()
    with t.span("outer", input=1) as outer:
        with t.span("inner", input=2) as inner:
            inner.set_output("x")
        outer.set_output("y")
    by_name = {s.name: s for s in t.spans}
    assert by_name["inner"].parent_id == by_name["outer"].span_id
    assert by_name["outer"].parent_id is None
    assert by_name["inner"].trace_id == by_name["outer"].trace_id
    assert by_name["inner"].output == "x" and by_name["outer"].latency_ms >= by_name["inner"].latency_ms


def test_error_is_recorded_and_reraised():
    t = Tracer()
    with pytest.raises(ValueError):
        with t.span("boom"):
            raise ValueError("bad input")
    assert t.spans[0].error == "ValueError: bad input"


def test_decorator_records_args_output_and_nesting(tmp_path):
    path = tmp_path / "trace.jsonl"
    t = Tracer(path)

    @t.trace()
    def add(a, b):
        return a + b

    @t.trace("wrapper")
    def run():
        return add(2, b=3)

    assert run() == 5
    spans = read_spans(path)
    add_span = next(s for s in spans if s.name == "add")
    wrapper = next(s for s in spans if s.name == "wrapper")
    assert add_span.input == {"args": [2], "kwargs": {"b": 3}} and add_span.output == 5
    assert add_span.parent_id == wrapper.span_id


def test_unserialisable_values_do_not_break_tracing(tmp_path):
    t = Tracer(tmp_path / "t.jsonl")
    with t.span("s", input=object()) as sp:
        sp.set_output({1, 2})
    assert len(read_spans(tmp_path / "t.jsonl")) == 1


def test_separate_traces_get_separate_ids():
    t = Tracer()
    with t.span("a"):
        pass
    with t.span("b"):
        pass
    assert t.spans[0].trace_id != t.spans[1].trace_id
