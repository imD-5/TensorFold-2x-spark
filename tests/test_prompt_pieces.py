"""PromptPieces: a chat prompt's ids equal the tokenizer's, and a re-sent conversation encodes only its new pieces."""

import json
import random
from types import SimpleNamespace

import pytest

tokenizers = pytest.importorskip("tokenizers")
from tokenizers import AddedToken, Regex, Tokenizer, models, normalizers, pre_tokenizers, trainers

from tensorfold.server.prompt_pieces import PromptPieces

TURN, END, TOOL, CALL, MARK = "<｜User｜>", "<｜end▁of▁sentence｜>", "<tool>", "<tool><call>", "<x>"
WORDS = ["a", "b", " ", "  ", "\n", "\n\n", "\t", "1", "12", "1234", "日本", "語", "🙂", ".", "'s", "def", " x",
         "(", ")", "<", ">", "｜", "<｜", "Call", "call", "tool", "end"]


def tokenizer(**mark) -> Tokenizer:
    """A byte-level BPE like DeepSeek's, with a normalizer and added tokens; ``mark`` sets the flags of ``MARK``."""

    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(r"\p{N}{1,3}"), "isolated"),
                                                 pre_tokenizers.ByteLevel(add_prefix_space=False)])
    trainer = trainers.BpeTrainer(vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    tok.train_from_iterator(["def call(x): return x  # 日本語 🙂 tool end 1234\n"] * 50, trainer)
    tok.add_special_tokens([TURN, END])
    tok.add_tokens([AddedToken(TOOL, normalized=False), AddedToken(CALL, normalized=False),
                    AddedToken("call", normalized=True), AddedToken(MARK, normalized=False, **mark)])
    return tok


class Counting:
    """The tokenizer, recording each text it is asked to encode."""

    def __init__(self, tok):
        self.tok, self.encoded = tok, []

    def get_added_tokens_decoder(self):
        return self.tok.get_added_tokens_decoder()

    @property
    def pre_tokenizer(self):
        return self.tok.pre_tokenizer

    @property
    def model(self):
        return self.tok.model

    def encode_batch(self, texts, add_special_tokens=False):
        self.encoded += texts
        return self.tok.encode_batch(texts, add_special_tokens=add_special_tokens)


def full(tok, text):
    return tok.encode(text, add_special_tokens=False).ids


def pieces_of(tok, **kwargs) -> PromptPieces:
    pieces = PromptPieces.of(tok, **kwargs)
    assert pieces is not None
    return pieces


@pytest.mark.parametrize("mark", [{}, {"lstrip": True}])
def test_ids_equal_the_tokenizers_on_texts_with_added_tokens_anywhere(mark):
    tok, rng = tokenizer(**mark), random.Random(0)
    pieces = pieces_of(tok, budget=200)       # small: pieces are dropped and encoded again along the way
    marks = [TURN, END, TOOL, CALL, MARK, " " + MARK, TOOL + "<call", TOOL + CALL]
    texts = ["", TURN, END + END, " " + TURN + " ", "a" * 3000 + TURN + "b"]
    texts += ["".join(rng.choice(marks) if rng.random() < 0.2 else rng.choice(WORDS)
                      for _ in range(rng.randint(1, 50))) for _ in range(1500)]
    for text in texts:
        assert pieces.ids(text) == full(tok, text), text


def test_a_resent_conversation_encodes_only_its_new_pieces():
    tok = Counting(tokenizer())
    pieces = pieces_of(tok)
    turn1 = TURN + "def call(x):\n  return 12 * x\n" * 200 + END
    turn2 = TURN + "and 日本語? 🙂" + END
    assert pieces.ids(turn1) == full(tok.tok, turn1)
    tok.encoded.clear()
    assert pieces.ids(turn1 + turn2 + "tail") == full(tok.tok, turn1 + turn2 + "tail")
    assert tok.encoded == ["and 日本語? 🙂" + END, "tail"]       # TURN alone is a piece seen before


@pytest.mark.parametrize("mark", [{"rstrip": True}, {"single_word": True}])
def test_a_tokenizer_whose_added_tokens_move_a_split_gets_no_pieces(mark):
    assert PromptPieces.of(tokenizer(**mark)) is None


def test_special_tokens_encoded_as_text_get_no_pieces():
    tok = tokenizer()
    tok.encode_special_tokens = True
    assert PromptPieces.of(tok) is None


@pytest.mark.parametrize("option", ["padding", "truncation"])
def test_whole_input_padding_or_truncation_gets_no_pieces(option):
    tok = tokenizer()
    if option == "padding":
        tok.enable_padding(length=32)
    else:
        tok.enable_truncation(max_length=4)
    assert PromptPieces.of(tok) is None


def test_a_tokenizer_without_added_tokens_or_batches_gets_no_pieces():
    assert PromptPieces.of(Tokenizer(models.BPE())) is None
    assert PromptPieces.of(object()) is None


def test_the_pieces_kept_stay_within_the_budget():
    tok = tokenizer()
    pieces = pieces_of(tok, budget=50)
    for i in range(40):
        text = f"{TURN}turn {i} " + "x" * i + END
        assert pieces.ids(text) == full(tok, text)
        assert pieces._held <= 50 and pieces._held == sum(len(ids) for ids in pieces._kept.values())


def test_text_that_encodes_to_no_tokens_stays_within_the_budget():
    tok = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tok.normalizer = normalizers.Strip()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_tokens([AddedToken(MARK, normalized=False)])
    pieces = pieces_of(tok, budget=256)
    for n in range(1, 100):
        text = MARK + " " * n
        assert pieces.ids(text) == full(tok, text)
        assert pieces._used <= 256
        assert pieces._used == sum(len(piece) + len(ids) + 64 for piece, ids in pieces._kept.items())
    assert len(pieces._kept) < 4


def test_text_larger_than_the_cache_budget_is_not_kept():
    tok = tokenizer()
    pieces = pieces_of(tok, budget=128)
    text = " " * 1000 + TURN
    assert pieces.ids(text) == full(tok, text)
    assert text not in pieces._kept


def test_a_bounded_cache_reuses_tokens_and_encodes_evicted_pieces_again():
    tok = Counting(tokenizer())
    pieces = pieces_of(tok, budget=256)
    first = "turn 0" + END
    assert pieces.ids(first) == full(tok.tok, first)
    assert first in pieces._kept
    tok.encoded.clear()
    assert pieces.ids(first) == full(tok.tok, first)
    assert tok.encoded == []
    for n in range(1, 20):
        text = f"turn {n}" + END
        assert pieces.ids(text) == full(tok.tok, text)
        assert pieces._kept and pieces._used <= 256
    assert first not in pieces._kept
    tok.encoded.clear()
    assert pieces.ids(first) == full(tok.tok, first)
    assert tok.encoded == [first]


def test_the_cuda_server_encodes_chat_prompts_through_its_pieces(tmp_path):
    pytest.importorskip("jinja2")
    from tensorfold.cuda.server import App

    tokenizer().save(str(tmp_path / "tokenizer.json"))
    template = "{% for m in messages %}" + TURN + "{{ m.content }}" + END + "{% endfor %}"
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
    app = App(SimpleNamespace(eos=(0,)), tmp_path, "test")
    messages = [{"role": "user", "content": "def call(x): return 12 * x"}]
    for turn in range(3):
        prompt = app.prepare({"messages": messages, "max_tokens": 1}, True).prompt
        assert prompt == full(app.tok, app.template.render(messages, tools=None, enable_thinking=False))
        messages += [{"role": "assistant", "content": f"answer {turn}"}, {"role": "user", "content": f"more {turn}"}]
    pieces = app._prompt_pieces()
    assert pieces is not None and "def call(x): return 12 * x" + END in pieces._kept


def metaspace_tokenizer(scheme="first", nested=False):
    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "▁hello": 2, "and": 3, "▁and": 4},
                                     unk_token="[UNK]"))
    tok.add_tokens([AddedToken("<s>", normalized=False)])
    state = {"type": "Metaspace", "replacement": "▁", "prepend_scheme": scheme, "split": True}
    if nested:
        state = {"type": "Sequence", "pretokenizers": [
            {"type": "WhitespaceSplit"}, {"type": "Sequence", "pretokenizers": [state]}]}
    document = json.loads(tok.to_str())
    document["pre_tokenizer"] = state
    tok = Tokenizer.from_str(json.dumps(document))
    if nested:
        assert json.loads(tok.pre_tokenizer.__getstate__())["pretokenizers"][1]["type"] == "Sequence"
    return tok


@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested-sequence"])
def test_position_sensitive_metaspace_keeps_whole_input_ids(nested):
    tok = metaspace_tokenizer(nested=nested)
    assert full(tok, "<s>hello") != full(tok, "<s>") + full(tok, "hello")
    pieces = PromptPieces.of(tok)
    for text in ("<s>hello", "<s>and<s>hello", "hello<s>hello"):
        actual = full(tok, text) if pieces is None else pieces.ids(text)
        assert actual == full(tok, text)
    assert pieces is None


def server_for_tokenizer(tmp_path, tok):
    pytest.importorskip("jinja2")
    from tensorfold.cuda.server import App

    tok.save(str(tmp_path / "tokenizer.json"))
    template = "{% for m in messages %}<s>{{ m.content }}{% endfor %}"
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
    return App(SimpleNamespace(eos=(0,)), tmp_path, "test")


@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested-sequence"])
def test_the_server_falls_back_for_position_sensitive_metaspace(tmp_path, nested):
    app = server_for_tokenizer(tmp_path, metaspace_tokenizer(nested=nested))
    messages = [{"role": "user", "content": "hello"}]
    for _ in range(2):
        actual = app.prepare({"messages": messages, "max_tokens": 1}, True).prompt
        text = app.template.render(messages, tools=None, enable_thinking=False)
        assert actual == full(app.tok, text)
        assert app._prompt_pieces() is None
        messages += [{"role": "assistant", "content": "and"}, {"role": "user", "content": "hello"}]


@pytest.mark.parametrize("scheme", ["always", "never"])
def test_position_independent_metaspace_keeps_real_cache_reuse(scheme):
    tok = Counting(metaspace_tokenizer(scheme))
    pieces = pieces_of(tok)
    text = "<s>hello<s>and<s>hello"
    assert pieces.ids(text) == full(tok.tok, text)
    assert tok.encoded
    tok.encoded.clear()
    assert pieces.ids(text) == full(tok.tok, text)
    assert tok.encoded == []


class UnknownPreTokenizer:
    def __init__(self, state):
        self.state = state

    def __getstate__(self):
        if isinstance(self.state, Exception):
            raise self.state
        return self.state


class UnknownPreState:
    def __init__(self, tok, state):
        self.tok = tok
        self.pre_tokenizer = UnknownPreTokenizer(state)

    def __getattr__(self, name):
        return getattr(self.tok, name)


@pytest.mark.parametrize("state", [b"not json", b"[]", b'{}', b'{"type":"FuturePreTokenizer"}',
                                    b'{"type":"Sequence"}', None, RuntimeError("custom state is not serializable")])
def test_unknown_pre_tokenizer_state_keeps_the_normal_server_path(tmp_path, state):
    app = server_for_tokenizer(tmp_path, metaspace_tokenizer("never"))
    app.tok = UnknownPreState(app.tok, state)
    messages = [{"role": "user", "content": "hello"}]
    actual = app.prepare({"messages": messages, "max_tokens": 1}, True).prompt
    assert actual == full(app.tok, "<s>hello")
    assert app._prompt_pieces() is None


def test_a_real_custom_pre_tokenizer_can_encode_without_enabling_pieces():
    class Custom:
        def pre_tokenize(self, text):
            pass

    tok = metaspace_tokenizer("never")
    tok.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(Custom())
    assert PromptPieces.of(tok) is None
    assert full(tok, "<s>hello") == [tok.token_to_id("<s>"), tok.token_to_id("hello")]


@pytest.mark.parametrize("dropout", [0.5, 1.0])
def test_bpe_dropout_does_not_enable_piece_reuse(dropout):
    tok = tokenizer()
    tok.model.dropout = dropout
    assert isinstance(tok.model, models.BPE) and tok.model.dropout == dropout
    assert PromptPieces.of(tok) is None


@pytest.mark.parametrize("dropout", [0.5, 1.0])
def test_bpe_dropout_server_uses_the_whole_input_route(tmp_path, monkeypatch, dropout):
    tok = tokenizer()
    tok.model.dropout = dropout
    app = server_for_tokenizer(tmp_path, tok)
    assert isinstance(app.tok.model, models.BPE) and app.tok.model.dropout == dropout
    encoded = []
    encode = app._encode

    def whole_input(text, add_special_tokens):
        encoded.append((text, add_special_tokens))
        return encode(text, add_special_tokens)

    monkeypatch.setattr(app, "_encode", whole_input)
    messages = [{"role": "user", "content": "hello"}]
    prompt = app.prepare({"messages": messages, "max_tokens": 1}, True).prompt
    text = app.template.render(messages, tools=None, enable_thinking=False)
    assert encoded == [(text, False)] and prompt
    assert app._prompt_pieces() is None


@pytest.mark.parametrize("dropout", [None, 0.0])
def test_bpe_without_dropout_keeps_real_cache_reuse(dropout):
    tok = Counting(tokenizer())
    tok.model.dropout = dropout
    assert tok.model.dropout == dropout
    pieces = pieces_of(tok)
    text = TURN + "hello" + END
    assert pieces.ids(text) == full(tok.tok, text) and tok.encoded
    tok.encoded.clear()
    assert pieces.ids(text) == full(tok.tok, text)
    assert tok.encoded == []


class UnknownModelState:
    def __init__(self, tok, state):
        self.tok, self.state = tok, state

    @property
    def model(self):
        if isinstance(self.state, Exception):
            raise self.state
        return self.state

    def __getattr__(self, name):
        return getattr(self.tok, name)


@pytest.mark.parametrize("state", [None, object(), SimpleNamespace(dropout=None), RuntimeError("unknown model")])
def test_unknown_model_metadata_keeps_the_normal_server_path(tmp_path, state):
    app = server_for_tokenizer(tmp_path, metaspace_tokenizer("never"))
    app.tok = UnknownModelState(app.tok, state)
    messages = [{"role": "user", "content": "hello"}]
    actual = app.prepare({"messages": messages, "max_tokens": 1}, True).prompt
    assert actual == full(app.tok, "<s>hello")
    assert app._prompt_pieces() is None
