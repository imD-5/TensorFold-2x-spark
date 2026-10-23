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
