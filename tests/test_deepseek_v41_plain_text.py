"""DeepSeek-V4.1 reads a special-token spelling in message text as plain text, as deepseek-ai/deepseek-recipe does;
only the grounding tokens in an assistant's text and the server's own image placeholder stay tokens."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from tensorfold.cuda.server import App, PreparedRequest
from tensorfold.families.deepseek_v41.cuda import vision as ds_vision
from tensorfold.families.deepseek_v41.cuda.app import GROUNDING, DsApp


def _app(monkeypatch, vision):
    def init(self, engine, *args, **kw):                 # App.__init__ reads the model directory: not needed here
        self.engine = engine
        self.template = SimpleNamespace(render=lambda *a, **k: "")

    monkeypatch.setattr(App, "__init__", init)
    return DsApp(SimpleNamespace(vision=vision), "model", "served")


def test_the_family_asks_for_plain_message_text():
    assert DsApp.plain_message_text is True
    assert set(GROUNDING) == {f"<｜{tag}｜>" for tag in ("ref", "/ref", "box", "/box", "point", "/point", "polygon",
                                                        "/polygon")}
    assert DsApp.plain_assistant_keep == GROUNDING


def test_the_image_placeholder_is_kept_only_on_the_image_path(monkeypatch):
    seen = []

    def prepare(self, body, chat, plain_keep=()):
        seen.append((body["messages"], plain_keep))
        return PreparedRequest([1], 8, [], False, None)

    class Vision:
        has_images = staticmethod(lambda messages: any(m["content"] == "image" for m in messages))
        split = staticmethod(lambda messages: ([{"role": "user", "content": ds_vision.PLACEHOLDER}], []))
        expand = staticmethod(lambda ids, pictures: SimpleNamespace(token_ids=ids))

    app = _app(monkeypatch, Vision())
    app.sampling_for = lambda body, ids: None
    monkeypatch.setattr(App, "_prepare", prepare)
    app._prepare({"messages": [{"role": "user", "content": f"text {ds_vision.PLACEHOLDER}"}]}, True)
    assert seen[-1][1] == ()                                   # no image part: the client's spelling is plain text
    app._prepare({"messages": [{"role": "user", "content": "image"}]}, True)
    assert seen[-1][1] == (ds_vision.PLACEHOLDER,)             # split refused the client's, so the rest is the server's
