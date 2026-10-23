"""A reply cut inside a DSML calls block (max_tokens, finish_reason "length") gives the same content and the same tool
calls streamed and not streamed, at every cut. The open block is removed from the content and makes no call. Calls
blocks that are complete before it stay calls. Text before the open block stays. Reasoning is not changed.

The fake engine writes a fixed reply, one character for each token, and stops at max_tokens: max_tokens is the cut.
The replies come through the real HTTP handler, so the streamed deltas and the whole reply are the ones a client gets."""

import json
import re

import pytest

from tests.test_cuda_admission import http_server, post
from tests.test_cuda_tool_choice import EOS, app_for, events

D = "｜DSML｜"
OPEN, CLOSE = f"<{D} calls>", f"</{D} calls>"
TOOLS = [{"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {}}}}
         for name in ("read", "write")]
INVOKE_NAME = re.compile(rf'<{D} invoke name="(\w+)">')


def param(key, value):
    return f'<{D} parameter name="{key}" string="true">{value}</{D} parameter>\n'


def invoke(name, *params):
    return f'<{D} invoke name="{name}">\n{"".join(params)}</{D} invoke>\n'


def block(*invokes):
    return f"{OPEN}\n{''.join(invokes)}{CLOSE}"


READ_A = invoke("read", param("path", "a.py"))
WRITE_B = invoke("write", param("path", "b.py"), param("text", "hi"))
ONE = "Checking.\n\n" + block(READ_A)
TWO_INVOKES = "Checking.\n\n" + block(READ_A, WRITE_B)
COMPLETE_THEN_OPEN = "First.\n\n" + block(READ_A) + "\nThen.\n\n" + block(WRITE_B)[:-len(CLOSE)]


class Chars:
    """The fake engine: the characters of ``reply`` as tokens, ``max_tokens`` of them, then the end token when the
    whole reply fits."""

    eos = (EOS,)

    def __init__(self):
        self.reply = ""

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        ids = [ord(c) for c in self.reply][:max_tokens]
        if len(self.reply) < max_tokens:
            ids.append(EOS)
        for at in range(0, len(ids), 3):
            if on_tokens(ids[at:at + 3]):
                break
        return {"rounds": 1}


class Replies:
    """Asks one server for a reply cut at ``cut`` characters, not streamed and streamed."""

    def __init__(self, tmp_path):
        self.engine = Chars()
        self.app = app_for(tmp_path, self.engine)

    def ask(self, port, reply, cut, **body):
        self.engine.reply = reply
        request = {"messages": [{"role": "user", "content": "Go."}], "tools": TOOLS, "max_tokens": cut,
                   "temperature": 0, **body}
        whole = self.whole(port, request)
        streamed = self.streamed(port, request)
        return whole, streamed

    @staticmethod
    def whole(port, request):
        status, text = post(port, {**request, "stream": False}, True)
        assert status == 200, text
        choice = json.loads(text)["choices"][0]
        message = choice["message"]
        calls = [(c["function"]["name"], json.loads(c["function"]["arguments"]))
                 for c in message.get("tool_calls") or []]
        return {"content": message.get("content") or "", "reasoning": message.get("reasoning_content") or "",
                "calls": calls, "finish": choice["finish_reason"]}

    @staticmethod
    def streamed(port, request):
        status, text = post(port, {**request, "stream": True}, True)
        assert status == 200, text
        content, reasoning, finish, calls = "", "", None, {}
        for event in events(text):
            if not event.get("choices"):
                continue
            choice = event["choices"][0]
            delta = choice.get("delta") or {}
            content += delta.get("content") or ""
            reasoning += delta.get("reasoning_content") or ""
            for call in delta.get("tool_calls") or []:
                entry = calls.setdefault(call["index"], {"name": "", "arguments": ""})
                entry["name"] += (call.get("function") or {}).get("name") or ""
                entry["arguments"] += (call.get("function") or {}).get("arguments") or ""
            finish = choice.get("finish_reason") or finish
        return {"content": content, "reasoning": reasoning, "finish": finish,
                "calls": [(c["name"], json.loads(c["arguments"])) for _, c in sorted(calls.items())]}


def expected(reply, cut):
    """(content, call names) that the rule gives for the reply cut after ``cut`` characters: complete calls blocks
    make calls, an open one is removed with the rest of the reply."""

    text = reply[:cut]
    names, content, at = [], "", 0
    while True:
        start = text.find(OPEN, at)
        if start < 0:
            content += text[at:]
            break
        content += text[at:start]
        end = text.find(CLOSE, start)
        if end < 0:
            break
        names += INVOKE_NAME.findall(text[start:end])
        at = end + len(CLOSE)
    return content.strip(), names


@pytest.fixture
def replies(tmp_path):
    return Replies(tmp_path)


def served(replies):
    return http_server(replies.app)


@pytest.mark.parametrize("reply", [ONE, TWO_INVOKES, COMPLETE_THEN_OPEN], ids=["one", "two_invokes", "complete_then_open"])
@pytest.mark.parametrize("parallel", [True, False])
def test_streamed_and_whole_replies_agree_at_every_cut(replies, reply, parallel):
    with served(replies) as port:
        for cut in range(1, len(reply) + 2):
            whole, streamed = replies.ask(port, reply, cut, parallel_tool_calls=parallel)
            assert whole == streamed, cut
            content, names = expected(reply, cut)
            assert whole["content"] == content, cut
            assert [name for name, _ in whole["calls"]] == (names if parallel else names[:1]), cut
            if cut >= len(reply) + 1:                     # the end token came: the reply is whole
                assert whole["finish"] == ("tool_calls" if names else "stop"), cut
            elif not names:
                assert whole["finish"] == "length", cut   # a cut with no complete call is a length stop
            else:
                assert whole["finish"] == "tool_calls", cut


def test_a_cut_in_the_second_invoke_of_a_block_makes_no_call_and_keeps_the_text_before(replies):
    cut = TWO_INVOKES.index(WRITE_B) + len(f'<{D} invoke name="write">\n<{D} parameter name="path" st')
    with served(replies) as port:
        whole, streamed = replies.ask(port, TWO_INVOKES, cut)
    assert whole == streamed
    assert whole["calls"] == []
    assert whole["content"] == "Checking."
    assert whole["finish"] == "length"


def test_a_cut_in_the_first_invoke_of_a_block_makes_no_call_and_keeps_the_text_before(replies):
    cut = TWO_INVOKES.index(READ_A) + len(f'<{D} invoke name="read">\n<{D} parameter name="path" string="true">a.')
    with served(replies) as port:
        whole, streamed = replies.ask(port, TWO_INVOKES, cut)
    assert whole == streamed
    assert whole["calls"] == []
    assert whole["content"] == "Checking."
    assert whole["finish"] == "length"


def test_a_complete_block_before_an_open_block_keeps_its_calls(replies):
    cut = len(COMPLETE_THEN_OPEN) - 3
    with served(replies) as port:
        whole, streamed = replies.ask(port, COMPLETE_THEN_OPEN, cut)
    assert whole == streamed
    assert whole["calls"] == [("read", {"path": "a.py"})]
    assert whole["content"] == "First.\n\n\nThen."
    assert D not in whole["content"]


def test_a_cut_inside_reasoning_changes_nothing(replies):
    reasoning = f"I may write {OPEN}\n{invoke('read', param('path', 'a'))} but first I think."
    thinking = {"chat_template_kwargs": {"enable_thinking": True}}
    with served(replies) as port:
        for cut in (len("I may write "), len("I may write ") + len(OPEN) + 5, len(reasoning)):
            whole, streamed = replies.ask(port, reasoning, cut, **thinking)
            assert whole == streamed, cut
            assert whole["reasoning"] == reasoning[:cut], cut
            assert (whole["content"], whole["calls"]) == ("", []), cut


def test_an_open_block_after_closed_reasoning_is_removed_and_the_reasoning_stays(replies):
    reply = "I will read it.</think>\n\nOK.\n\n" + block(READ_A)
    thinking = {"chat_template_kwargs": {"enable_thinking": True}}
    with served(replies) as port:
        for cut in range(len(reply) - len(CLOSE) - 20, len(reply) - len(CLOSE) + 1):
            whole, streamed = replies.ask(port, reply, cut, **thinking)
            assert whole == streamed, cut
            assert whole["reasoning"] == "I will read it."
            assert (whole["content"], whole["calls"], whole["finish"]) == ("OK.", [], "length"), cut
