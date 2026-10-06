import types

from forensics.provider import AnthropicProvider, MockProvider
from forensics.validators import Contains, Custom, Length, LLMJudge, Schema, run_validators


def test_schema():
    v = Schema({"label": str}, {"label": {"a", "b"}})
    assert v.check({"label": "a"}).ok
    assert "missing key" in v.check({}).reason
    assert not v.check({"label": "z"}).ok
    assert not v.check("text").ok


def test_contains_modes_and_negate():
    assert Contains(["a", "b"]).check("A and B").ok
    assert not Contains(["a", "b"]).check("only a").ok
    assert Contains(["a", "b"], mode="any").check("only a").ok
    assert not Contains("bad", negate=True).check("this is BAD").ok
    assert Contains("x", field="r").check({"r": "x"}).ok
    assert not Contains("x", field="r").check({}).ok


def test_length():
    assert Length(2, 4).check("abc").ok
    assert not Length(2, 4).check("a").ok
    assert not Length(2, 4).check("abcde").ok


def test_custom_and_crash_is_a_failure():
    assert Custom(lambda o, c: o > c).check(3, 2).ok
    assert not Custom(lambda o, c: (False, "nope")).check(1).ok
    res = run_validators([Custom(lambda o, c: 1 / 0, "div")], 1)
    assert not res[0].ok and "crashed" in res[0].reason


def test_llm_judge_with_scripted_provider():
    judge_pass = LLMJudge(MockProvider(script=["PASS: fine"]), "be polite")
    judge_fail = LLMJudge(MockProvider(script=["FAIL: rude"]), "be polite")
    assert judge_pass.check("hello").ok
    assert not judge_fail.check("hello").ok


def test_anthropic_provider_with_fake_client():
    block = types.SimpleNamespace(type="text", text="hi there")
    calls = {}

    class FakeMessages:
        def create(self, **kw):
            calls.update(kw)
            return types.SimpleNamespace(content=[block])

    client = types.SimpleNamespace(messages=FakeMessages())
    p = AnthropicProvider(client=client)
    assert p.complete("hello", system="sys") == "hi there"
    assert calls["model"] == "claude-sonnet-5-5" and calls["system"] == "sys"
