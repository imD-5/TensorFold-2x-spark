"""The image placeholder inside a message's own text: escaped (TF #17) instead of refusing an image chat.

DeepSeek's encoding lays images out as placeholder blocks in the rendered prompt, and the placeholder string is the
tokenizer's image token, so a message that merely quotes it would inject stray image token ids (and once in the
history even text-only turns carry them). Following TF #17 by @LeetJoe on the Zig server: when the chat carries an
image part, the placeholder in a message's own text (string ``content``, ``reasoning_content``, the ``text`` of
array parts) becomes the plain ASCII form the tokenizer reads as text; the image part's own rendered placeholder
keeps the special form; a chat without an image part reaches the tokenizer byte-identical."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.families.deepseek_v41.cuda import vision
from tensorfold.families.deepseek_v41.cuda.app import DsApp

SPECIAL = "<｜deepseek_image｜>"
PLAIN = "<|deepseek_image|>"
IMAGE_TOKEN = 7
IMAGE_PART = {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}}   # never decoded below


def config():
    return vision.VisionConfig(layers=1, dim=8, heads=2, inter=16, patch=14, theta=10000.0, ratio=3,
                               max_tokens=64, min_pixels=4, max_wh_ratio=None, image_token_id=IMAGE_TOKEN,
                               model_dim=8)


def picture():
    return vision.Picture(patches=torch.zeros(1, 3, 14, 14), n_vit_h=1, n_vit_w=1, n_llm_h=1, n_llm_w=1)


@pytest.fixture
def frontend(monkeypatch):
    """DsVision with decode faked: the escape tests need one decodable image part, not a vision tower."""

    monkeypatch.setattr(vision, "decode", lambda data, cfg, max_pixels=64 << 20: picture())
    return vision.DsVision(config())


@pytest.mark.torch
@pytest.mark.parametrize("message, field, expected", [
    ({"role": "user", "content": f"open {SPECIAL}.png"}, "content", f"open {PLAIN}.png"),
    ({"role": "user", "content": [{"type": "text", "text": f"quoted earlier turn: {SPECIAL}"}]},
     "content", f"quoted earlier turn: {PLAIN}"),
    ({"role": "assistant", "reasoning_content": f"the file was {SPECIAL}", "content": "ok"},
     "reasoning_content", f"the file was {PLAIN}"),
], ids=["content-string", "array-text-part", "reasoning-content"])
def test_the_placeholder_in_a_message_s_own_text_is_escaped(frontend, message, field, expected):
    out, pictures = frontend.split([message, {"role": "user", "content": [IMAGE_PART]}])
    assert out[0][field] == expected
    assert len(pictures) == 1


@pytest.mark.torch
def test_the_image_part_keeps_its_token(frontend):
    message = {"role": "user", "content": [IMAGE_PART, {"type": "text", "text": f"see {SPECIAL}"}]}
    out, pictures = frontend.split([message])
    assert out[0]["content"] == f"{SPECIAL}\n\nsee {PLAIN}"
    expanded = frontend.expand([1, IMAGE_TOKEN, 2], pictures)
    assert expanded.token_ids.count(IMAGE_TOKEN) == pictures[0].tokens      # the escaped text adds no image token


@pytest.mark.torch
def test_the_escape_is_idempotent(frontend):
    assert vision.escape_placeholder(vision.escape_placeholder(f"a {SPECIAL} b {PLAIN} c")) == \
        vision.escape_placeholder(f"a {SPECIAL} b {PLAIN} c") == f"a {PLAIN} b {PLAIN} c"
    out, _ = frontend.split([{"role": "user", "content": [{"type": "text", "text": f"already {PLAIN}"}]},
                             {"role": "user", "content": [IMAGE_PART]}])
    assert out[0]["content"] == f"already {PLAIN}"                # no double escape, no special form back
    again, _ = frontend.split(out)                                # escaping the split output changes nothing
    assert again == out


class FakeTokenizer:
    """SPECIAL is the tokenizer's image token (IMAGE_TOKEN); PLAIN and every other character are plain text."""

    def __init__(self):
        self.rendered = []

    @staticmethod
    def ids(text: str) -> list[int]:
        out = []
        for chunk in text.split(SPECIAL):
            out.extend(1000 + ord(c) for c in chunk)
            out.append(IMAGE_TOKEN)
        return out[:-1]

    def encode(self, text: str, **kwargs):
        self.rendered.append(text)
        return NS(ids=self.ids(text))


def fake_render(messages, **kwargs):
    return " | ".join(f"{m.get('role')}[{m.get('reasoning_content') or ''}] {m.get('content')}" for m in messages)


def fake_ds_app(frontend) -> tuple[DsApp, FakeTokenizer]:
    """The plumbing DsApp._prepare reads, on the real App code path (no HTTP, no engine)."""

    app = DsApp.__new__(DsApp)
    app.tok = FakeTokenizer()
    app.template = NS(render=fake_render)
    app.engine = NS(vision=frontend, eos=(0,), context_window=32)
    app.context_window, app.native_context_window = 32, 64
    app.max_tokens, app.default_thinking = 8, False
    app.sampling = {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0}
    return app, app.tok


@pytest.mark.torch
def test_a_chat_without_images_reaches_the_tokenizer_byte_identical():
    app, tok = fake_ds_app(vision.DsVision(config()))
    for messages in ([{"role": "user", "content": f"file {SPECIAL}.png"}],
                     [{"role": "user", "content": "plain"}],
                     [{"role": "assistant", "reasoning_content": f"was {SPECIAL}", "content": "ok"}]):
        rendered = fake_render(messages)
        prepared = app._prepare({"messages": messages, "max_tokens": 4}, True)
        assert tok.rendered[-1] == rendered               # the tokenizer heard the request's own text, byte-identical
        assert prepared.prompt == FakeTokenizer.ids(rendered)   # the same ids the base server encoded
        assert (prepared.prompt.count(IMAGE_TOKEN) == 1) == (SPECIAL in rendered)


@pytest.mark.torch
def test_an_image_chat_prepares_instead_of_refusing(monkeypatch):
    monkeypatch.setattr(vision, "decode", lambda data, cfg, max_pixels=64 << 20: picture())
    app, tok = fake_ds_app(vision.DsVision(config()))
    body = {"messages": [{"role": "user", "content": f"open {SPECIAL}.png"},
                         {"role": "user", "content": [IMAGE_PART]}], "max_tokens": 4}
    prepared = app._prepare(body, True)               # base refused this with a 400 (image input: ...)
    plain = FakeTokenizer.ids("user[] open <|deepseek_image|>.png | user[] ")
    assert tok.rendered[-1] == "user[] open <|deepseek_image|>.png | user[] <｜deepseek_image｜>"
    assert prepared.prompt == plain + [IMAGE_TOKEN] * picture().tokens
    assert prepared.vision.spans[0][0] == len(plain)
    assert prepared.vision.spans[0][1].tokens == picture().tokens
