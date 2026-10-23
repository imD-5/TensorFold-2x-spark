"""A special-token spelling in request text is plain text, not a token (DeepSeek's encoding): ``PlainText`` and the CUDA
server's chat path. A toy byte-level tokenizer with DeepSeek-V4.1's added tokens and a small template stand in for the
checkpoint's files."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("jinja2")
tokenizers = pytest.importorskip("tokenizers")

from tensorfold.cuda import server
from tensorfold.server.errors import RequestError
from tensorfold.server.plain_text import PlainText

BOS, EOS = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"
USER, ASSISTANT, SYSTEM = "<｜User｜>", "<｜Assistant｜>", "<｜System｜>"
THINK, END_THINK, DSML = "<think>", "</think>", "｜DSML｜"
REF, BOX_END, IMAGE = "<｜ref｜>", "<｜/box｜>", "<｜deepseek_image｜>"
SPECIAL = (BOS, EOS)                  # the others are added tokens that are not "special", as in the checkpoint
ADDED = (SYSTEM, USER, ASSISTANT, THINK, END_THINK, DSML, REF, BOX_END, IMAGE, "<｜latest_reminder｜>")
SPELLINGS = (END_THINK, THINK, USER, ASSISTANT, EOS, BOS, SYSTEM, DSML, REF, "<｜latest_reminder｜>")
GROUNDING = (REF, "<｜/ref｜>", "<｜box｜>", BOX_END)

# DeepSeek-V4.1's shape: BOS, an optional system turn, user turns, an assistant turn with its reasoning, tool calls
TEMPLATE = """{{ bos_token }}
{%- for m in messages -%}
{%- if m.role == 'system' -%}{{ '<｜System｜>' }}{{ m.content }}
{%- elif m.role == 'user' -%}{{ '<｜User｜>' }}{{ m.content }}
{%- elif m.role == 'tool' -%}{{ '<｜User｜>' }}<tool_result>{{ m.content }}</tool_result>
{%- else -%}{{ '<｜Assistant｜>' }}
{%- if m.reasoning_content %}{{ '<think>' }}{{ m.reasoning_content }}{% endif %}{{ '</think>' }}
{{- m.content }}
{%- for tc in (m.tool_calls or []) -%}
<{{ '｜DSML｜' }}invoke name="{{ tc.function.name }}">{{ tc.function.arguments | tojson }}</{{ '｜DSML｜' }}invoke>
{%- endfor -%}{{ '<｜end▁of▁sentence｜>' }}
{%- endif -%}
{%- endfor -%}
{%- if tools %}{{ tools | tojson }}{% endif -%}
{%- if refuse is defined -%}{{ raise_exception('refused: ' ~ refuse) }}{%- endif -%}
{{ '<｜Assistant｜>' }}{{ '<think>' if enable_thinking else '</think>' }}"""


def _tokenizer(*, plain: bool = False, nfkc: bool = False):
    from tokenizers import AddedToken, Tokenizer, models, normalizers, pre_tokenizers

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tok = Tokenizer(models.BPE(vocab={char: index for index, char in enumerate(alphabet)}, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    if nfkc:
        tok.normalizer = normalizers.NFKC()
    if not plain:
        tok.add_tokens([AddedToken(text, special=True, normalized=False) for text in SPECIAL]
                       + [AddedToken(text, special=False, normalized=True) for text in ADDED])
    return tok


def _ids(text: str, nfkc: bool = False) -> list[int]:
    """The ids of ``text`` as plain text: what DeepSeek's encoding gives it."""

    return _tokenizer(plain=True, nfkc=nfkc).encode(text, add_special_tokens=False).ids


def _token(text: str) -> int:
    return _tokenizer().token_to_id(text)


class Engine:
    eos = (0,)


class PlainApp(server.App):
    plain_message_text = True
    plain_assistant_keep = GROUNDING


def _app(tmp_path, cls=PlainApp, nfkc=False):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": TEMPLATE, "bos_token": BOS}))
    app = cls.__new__(cls)
    app.engine, app.served, app.tok = Engine(), "fake", _tokenizer(nfkc=nfkc)
    app.template = server.ChatTemplate(tmp_path)
    app.default_thinking, app.sampling, app.max_tokens = True, {"temperature": 0.0}, 4096
    app.native_context_window = app.context_window = 0
    app.lock, app.vision = threading.Lock(), None
    return app


def _prompt(app, messages, **body):
    return app._prepare({"messages": messages, **body}, True).prompt


def _expected(*pieces):
    """Template tokens (a spelling listed in ``ADDED`` or ``SPECIAL``) as ids, any other piece as plain text."""

    out = []
    for piece in pieces:
        out.extend([_token(piece)] if piece in ADDED + SPECIAL else _ids(piece))
    return out


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_a_user_message_holding_a_spelling_encodes_as_plain_text(tmp_path, spelling):
    text = f"a {spelling} b"
    got = _prompt(_app(tmp_path), [{"role": "user", "content": text}])
    assert got == _expected(BOS, USER, text, ASSISTANT, THINK)
    assert got.count(_token(spelling)) == (1 if spelling in (BOS, USER, ASSISTANT, THINK) else 0)   # the template's own


@pytest.mark.parametrize("spelling", [END_THINK, USER, EOS, DSML])
def test_every_request_field_keeps_a_spelling_as_plain_text(tmp_path, spelling):
    text = f"x{spelling}y"
    call = [{"id": "1", "type": "function", "function": {"name": "f", "arguments": {"p": text}}}]
    tool = {"type": "function", "function": {"name": "f", "description": text, "parameters": {"type": "object"}}}
    messages = [
        {"role": "system", "content": text},
        {"role": "user", "content": [{"type": "text", "text": text}, {"type": "text", "text": text}]},   # captions
        {"role": "assistant", "content": text, "reasoning_content": text, "tool_calls": call},
        {"role": "tool", "tool_call_id": "1", "content": text},
    ]
    app = _app(tmp_path)
    got = _prompt(app, messages, tools=[tool])
    ours = [t for t in got if t == _token(spelling)]
    template_own = _prompt(app, [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                                 {"role": "assistant", "content": "a", "reasoning_content": "r", "tool_calls": call},
                                 {"role": "tool", "tool_call_id": "1", "content": "t"}], tools=[tool])
    assert len(ours) == template_own.count(_token(spelling))              # no spelling of the request became a token
    plain = _ids(text)
    assert sum(got[i:i + len(plain)] == plain for i in range(len(got))) >= 5   # system, user, reasoning, reply, result


def test_a_clean_request_gets_the_ids_it_always_got(tmp_path):
    app = _app(tmp_path)
    messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi <b>there</b> 世界"},
                {"role": "assistant", "content": "Hello", "reasoning_content": "greet"},
                {"role": "user", "content": "?"}]
    old = _prompt(_app(tmp_path, server.App), messages)                   # the server without the shield
    assert _prompt(app, messages) == old
    assert old == _expected(BOS, SYSTEM, "Be brief.", USER, "Hi <b>there</b> 世界", ASSISTANT, THINK, "greet",
                            END_THINK, "Hello", EOS, USER, "?", ASSISTANT, THINK)


def test_the_template_tokens_are_still_tokens(tmp_path):
    got = _prompt(_app(tmp_path), [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a",
                 "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": {}}}]}])
    for token in (BOS, USER, ASSISTANT, END_THINK, EOS, DSML, THINK):
        assert _token(token) in got


def test_the_same_request_gives_the_same_ids_whatever_the_marker(tmp_path):
    app = _app(tmp_path)
    messages = [{"role": "user", "content": f"a {END_THINK} {USER} b"}]
    first = _prompt(app, messages)
    assert all(_prompt(app, messages) == first for _ in range(5))           # each request draws its own marker
    assert first == _expected(BOS, USER, f"a {END_THINK} {USER} b", ASSISTANT, THINK)


def test_grounding_tokens_stay_tokens_in_assistant_text_only(tmp_path):
    app = _app(tmp_path)
    got = _prompt(app, [{"role": "user", "content": f"u {REF}"},
                        {"role": "assistant", "content": f"see {REF}car{BOX_END}", "reasoning_content": f"{REF}"},
                        {"role": "user", "content": "next"}])
    assert got.count(_token(REF)) == 2 and got.count(_token(BOX_END)) == 1
    assert _ids(f"u {REF}") == got[2:2 + len(_ids(f"u {REF}"))]            # the user's own spelling is plain


def test_a_marker_lookalike_in_the_text_is_plain_text(tmp_path):
    text = f"@TF0123456789abcdef:0@ {END_THINK} @TF"
    got = _prompt(_app(tmp_path), [{"role": "user", "content": text}])
    assert got == _expected(BOS, USER, text, ASSISTANT, THINK)


def test_a_template_refusal_shows_the_request_text_not_the_marker(tmp_path):
    with pytest.raises(RequestError) as error:
        _prompt(_app(tmp_path), [{"role": "user", "content": "q"}], chat_template_kwargs={"refuse": END_THINK})
    assert END_THINK in str(error.value) and "@TF" not in str(error.value)


def test_the_server_keeps_its_own_placeholder_in_content_of_that_request_only(tmp_path):
    app = _app(tmp_path)
    call = [{"id": "1", "type": "function", "function": {"name": "f", "arguments": {"p": IMAGE}}}]
    body = {"messages": [{"role": "user", "content": f"{IMAGE}\n\ncaption"},
                         {"role": "assistant", "content": "", "tool_calls": call}, {"role": "tool", "content": "r"}]}
    assert app._prepare(body, True, plain_keep=(IMAGE,)).prompt.count(_token(IMAGE)) == 1   # content, not the argument
    assert app._prepare(body, True).prompt.count(_token(IMAGE)) == 0      # a request that the server did not write into


@pytest.mark.parametrize("parts", [["<thi", "nk>"], ["<", "/thi", "nk>"], ["a <｜Us", "er｜> b"], ["", "</think>", ""]])
def test_a_spelling_split_across_text_parts_is_plain_text(tmp_path, parts):
    text = "".join(parts)
    content = [{"type": "text", "text": part} for part in parts]
    for role in ("user", "system", "tool"):
        messages = [{"role": role, "content": content}]
        if role == "tool":
            messages.insert(0, {"role": "user", "content": "q"})
        elif role == "system":
            messages.append({"role": "user", "content": "q"})
        got = _prompt(_app(tmp_path), messages)
        assert got == _prompt(_app(tmp_path), [{**m, "content": text} if m["content"] is content else m for m in messages])
        assert sum(got[i:i + len(_ids(text))] == _ids(text) for i in range(len(got))) >= 1


def test_a_json_escape_in_string_tool_arguments_hides_no_spelling(tmp_path):
    """The template parses string arguments, so ``<\u0074hink>`` becomes ``<think>`` there."""

    escaped = '{"x": "<' + chr(92) + 'u0074hink> and ' + chr(92) + 'u003c/think>"}'
    plain = {"x": "<think> and </think>"}

    def history(arguments):
        call = [{"id": "1", "type": "function", "function": {"name": "f", "arguments": arguments}}]
        return [{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "tool_calls": call},
                {"role": "tool", "tool_call_id": "1", "content": "r"}]

    app = _app(tmp_path)
    assert _prompt(app, history(escaped)) == _prompt(app, history(plain))
    assert _prompt(app, history(plain)).count(_token(END_THINK)) == 1       # the template's own, after the assistant


def test_a_text_only_request_gets_no_placeholder_token(tmp_path):
    text = f"look {IMAGE} here"
    assert _prompt(_app(tmp_path), [{"role": "user", "content": text}]) == _expected(BOS, USER, text, ASSISTANT, THINK)


@pytest.mark.parametrize("forged", ["＜think＞", "＜/think＞", "＜｜User｜＞"])
def test_a_spelling_that_the_normalizer_finds_is_plain_text_too(tmp_path, forged):
    text = f"a {forged} b"
    old = _tokenizer(nfkc=True).encode(text, add_special_tokens=False).ids
    assert any(t in old for t in (_token(THINK), _token(END_THINK), _token(USER)))   # the tokenizer reads it as a token
    got = _prompt(_app(tmp_path, nfkc=True), [{"role": "user", "content": text}])
    assert got[:2] == [_token(BOS), _token(USER)] and got[2:-2] == _ids(text, nfkc=True)
    assert got[-2:] == [_token(ASSISTANT), _token(THINK)]


def test_a_family_without_the_shield_tokenizes_as_before(tmp_path):
    app = _app(tmp_path, server.App)
    got = _prompt(app, [{"role": "user", "content": f"a {END_THINK} b"}])
    assert _token(END_THINK) in got


def test_decisions_text_is_plain_too(tmp_path):
    class Scoring(Engine):
        def score_labels(self, prompt, labels):
            return [0.0, 1.0], 1.0

    app = _app(tmp_path)
    app.engine, app.turns = Scoring(), server.Turns()
    body = {"input": f"evidence {END_THINK} {USER}", "return_prompt_token_ids": True,
            "questions": [{"id": "q", "type": "choice", "question": "ok?", "options": [{"name": "A"}, {"name": "B"}]}]}
    ids = app.decisions(body)["answers"]["q"]["prompt_token_ids"]
    assert ids.count(_token(END_THINK)) == 1 and ids.count(_token(USER)) == 1      # the template's own


class _Fake:
    def get_added_tokens_decoder(self):
        return {}


def test_a_tokenizer_without_the_surface_is_left_alone():
    assert not PlainText.supports(_Fake()) and not PlainText.supports(SimpleNamespace())
    assert PlainText.supports(_tokenizer())


def test_shield_returns_the_inputs_when_nothing_is_spelled():
    plain = PlainText(_tokenizer())
    messages, tools, extra = [{"role": "user", "content": "hello"}], [{"type": "function"}], {"a": "b"}
    shielded = plain.shield(messages, tools, extra)
    assert shielded.messages is messages and shielded.tools is tools and shielded.extra is extra
    assert not shielded.table and plain.encode("a" + BOS + "b", shielded) == [*_ids("a"), _token(BOS), *_ids("b")]


def test_a_longer_spelling_wins_over_its_prefix():
    class Long:
        def __init__(self, tok):
            self.tok = tok

        def __getattr__(self, name):
            return getattr(self.tok, name)

        def get_added_tokens_decoder(self):
            return {**self.tok.get_added_tokens_decoder(), 9000: SimpleNamespace(content="<｜Assistant｜>x")}

    plain = PlainText(Long(_tokenizer()))
    shielded = plain.shield([{"role": "user", "content": "<｜Assistant｜>xy"}])
    assert len(shielded.table) == 1 and list(shielded.table.values()) == ["<｜Assistant｜>x"]
