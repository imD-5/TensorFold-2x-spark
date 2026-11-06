"""Keep prose references stable without exposing calls from another call's arguments."""
import json
from unittest import mock

import pytest

from tensorfold import dsml_calls
from tensorfold.cuda.reply_text import parse_tool_calls
from tests.test_cuda_tool_choice import EOS, app_for
from tests.test_dsml_containment import calls, invoke, param, tool


@pytest.mark.parametrize("name", ["tag-read", "tag.foo", "tag:foo", ".read", ":read"])
@pytest.mark.parametrize("limit", [None, 1])
def test_glm_identifier_is_not_a_prose_reference(name, limit):
    nested = calls(invoke("read", param("path", "nested")))
    gap = " " if name.startswith("tag") else ""
    text = f"<tool_call>{gap}{name}<arg_key>q</arg_key><arg_value>{nested}</arg_value></tool_call>"
    content, found = parse_tool_calls(text, [tool(name), tool("read")], max_calls=limit)
    assert [call["function"]["name"] for call in found or []] == [name]
    assert content == ""
    assert json.loads(found[0]["function"]["arguments"]) == {"q": nested}


class CharacterEngine:
    """Emit one character per round so every visibility update is tested."""
    eos = (EOS,)

    def __init__(self, reply):
        self.reply = reply

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        tokens = [ord(character) for character in self.reply][:max_tokens]
        if len(self.reply) < max_tokens:
            tokens.append(EOS)
        for token in tokens:
            if on_tokens([token]):
                break
        return {"rounds": len(tokens)}


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("prose,single_content", [("The <tool_call> tag starts a call.", "The "),
                                                 ("Use `<tool_call>` here.", "Use `"),
                                                 ('Use "<tool_call>", then.', 'Use "')])
def test_prose_and_argument_closer_stream_like_the_whole_reply_at_every_cut(tmp_path, parallel, prose, single_content):
    reply = prose + "\n" + calls(invoke("read", param("path", "</tool_call> suffix")),
                                 invoke("write", param("path", "second")))
    app = app_for(tmp_path, CharacterEngine(reply))
    for cut in range(1, len(reply) + 2):
        body = {"messages": [{"role": "user", "content": "Go."}],
                "tools": [tool("read"), tool("write")], "max_tokens": cut,
                "temperature": 0, "parallel_tool_calls": parallel}
        whole = app.run(body, True, lambda delta: True)
        deltas = []
        streamed = app.run({**body, "stream": True}, True, lambda delta, target=deltas: target.append(delta) or True)
        content = "".join(delta.get("content", "") for delta in deltas) + streamed["final"].get("content", "")
        assert content == whole["content"], (prose, parallel, cut)
        assert streamed["finish"] == whole["finish"], cut
        assert [(c["function"]["name"], c["function"]["arguments"]) for c in streamed["calls"] or []] == [
            (c["function"]["name"], c["function"]["arguments"]) for c in whole["calls"] or []], cut
        if cut >= len(reply):
            # Single-call mode retains its separate base-policy filter for literal tool_call envelopes.
            assert whole["content"] == (prose if parallel else single_content)
            expected = [("read", {"path": "</tool_call> suffix"}), ("write", {"path": "second"})]
            assert [(c["function"]["name"], json.loads(c["function"]["arguments"]))
                    for c in whole["calls"]] == (expected if parallel else expected[:1])


class CountedText(str):
    """Count copied slice characters without using timing as a pass condition."""
    copied = 0

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(key, slice):
            self.copied += len(result)
        return result


@pytest.mark.parametrize("mentions", [4000, 8000, 16000])
def test_prose_reference_scan_work_is_linear(mentions):
    text = CountedText("`<tool_call>` " * mentions + calls(invoke("read", param("path", "a"))))
    with mock.patch.object(dsml_calls, "_close_of", wraps=dsml_calls._close_of) as closer:
        blocks, open_at = dsml_calls.top_level_blocks(text)
    assert [block.kind for block in blocks] == ["dsml"]
    assert open_at == -1
    assert closer.call_count <= 2
    assert text.copied <= 4 * len(text)


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("continuation", ["plain", "dsml", "argument_closer"])
def test_open_and_close_references_stream_like_whole_at_every_cut(tmp_path, parallel, continuation):
    prose = "Use `<tool_call>` and `</tool_call>` for calls."
    if continuation == "plain":
        reply = prose + " Continue with plain text."
        expected_content = reply if parallel else "Use `` for calls. Continue with plain text."
        expected_calls = []
    else:
        value = "plain" if continuation == "dsml" else "</tool_call> suffix"
        reply = prose + "\n" + calls(invoke("read", param("path", value)),
                                     invoke("write", param("path", "second")))
        expected_content = prose if parallel else "Use `` for calls."
        expected_calls = [("read", {"path": value}), ("write", {"path": "second"})]
        if not parallel:
            expected_calls = expected_calls[:1]
    assert_every_cut(tmp_path, reply, parallel, expected_content, expected_calls)


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("name", ["tag", "tags", "tag-read", "tag.foo", "tag:foo"])
def test_unfinished_glm_name_does_not_become_a_prose_reference(tmp_path, parallel, name):
    nested = calls(invoke("read", param("path", "nested")))
    reply = f"<tool_call> {name}<arg_key>q</arg_key><arg_value>{nested}</arg_value></tool_call>"
    assert_every_cut(tmp_path, reply, parallel, "", [(name, {"q": nested})], [tool(name), tool("read")])


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("reference", ["tag", "tags"])
@pytest.mark.parametrize("with_dsml", [True, False])
def test_unquoted_reference_prefixes_keep_stream_and_whole_equal(tmp_path, parallel, reference, with_dsml):
    prose = f"Use <tool_call> {reference} in prose and </tool_call> literally."
    reply = prose + ("\n" + calls(invoke("read", param("path", "</tool_call> suffix")))
                     if with_dsml else " Continue with plain text.")
    expected_content = prose if with_dsml else reply
    if not parallel:
        expected_content = "Use  literally." + ("" if with_dsml else " Continue with plain text.")
    expected_calls = [("read", {"path": "</tool_call> suffix"})] if with_dsml else []
    assert_every_cut(tmp_path, reply, parallel, expected_content, expected_calls)


def assert_every_cut(tmp_path, reply, parallel, expected_content, expected_calls, tools=None):
    app = app_for(tmp_path, CharacterEngine(reply))

    def parsed(result):
        return [(c["function"]["name"], json.loads(c["function"]["arguments"]))
                for c in result["calls"] or []]

    for cut in range(1, len(reply) + 2):
        body = {"messages": [{"role": "user", "content": "Go."}], "max_tokens": cut,
                "tools": tools or [tool("read"), tool("write")], "temperature": 0,
                "parallel_tool_calls": parallel}
        whole = app.run(body, True, lambda delta: True)
        deltas = []
        streamed = app.run({**body, "stream": True}, True, lambda delta, target=deltas: target.append(delta) or True)
        content = "".join(delta.get("content", "") for delta in deltas) + streamed["final"].get("content", "")
        assert content == whole["content"], (parallel, cut)
        assert streamed["finish"] == whole["finish"], cut
        assert parsed(streamed) == parsed(whole), cut
        if cut >= len(reply):
            assert whole["content"] == expected_content
            assert parsed(whole) == expected_calls
