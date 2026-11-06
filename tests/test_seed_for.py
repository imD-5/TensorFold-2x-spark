"""seed_for over a long prompt: feeding the hash in slices gives the bytes of one join, so the same seed."""

import hashlib
import random
from array import array

import numpy as np
import pytest

from tensorfold.engine import exact_sampling


def one_join(tokens, salt):
    """The bytes a sliced hash must reproduce: every id, a comma apart, then the salt."""

    joined = ",".join(str(int(t)) for t in tokens) + f"|{salt}"
    return int.from_bytes(hashlib.sha256(joined.encode()).digest()[:8], "little") & ((1 << 63) - 1)


@pytest.mark.parametrize("n", [0, 1, 2, 4095, 4096, 4097, 8192, 8193, 50000])
@pytest.mark.parametrize("salt", [0, -7, 2**80 + 3])
def test_a_sliced_seed_equals_the_one_join(n, salt):
    rng = random.Random(n)
    tokens = [rng.randrange(0, 200000) for _ in range(n)]
    assert exact_sampling.seed_for(tokens, salt) == one_join(tokens, salt)


def test_a_changed_token_moves_the_seed():
    tokens = list(range(9000))
    assert exact_sampling.seed_for(tokens, 0) != exact_sampling.seed_for(tokens[:-1] + [1], 0)


def test_a_slice_boundary_between_two_ids_changes_nothing():
    """A comma the slice boundary could swallow would join two ids; 4096 and 4097 must both match the join."""

    tokens = list(range(4097))
    assert [exact_sampling.seed_for(tokens[:n], 7) for n in (4096, 4097)] == [one_join(tokens[:4096], 7),
                                                                             one_join(tokens, 7)]


@pytest.mark.parametrize("n", [4095, 4096, 4097, 8191, 8192, 8193])
def test_signed_ids_at_each_boundary_keep_their_separators(n):
    tokens = [(-1 if at % 2 else 1) * (10**(at % 8) + at) for at in range(n)]
    assert exact_sampling.seed_for(tokens, -2**100) == one_join(tokens, -2**100)


@pytest.mark.parametrize("factory", [list, tuple, lambda values: array("q", values), np.asarray])
def test_sequence_forms_keep_the_same_seed(factory):
    values = list(range(-4097, 4098))
    assert exact_sampling.seed_for(factory(values), 13) == one_join(values, 13)


def test_range_keeps_the_same_seed():
    tokens = range(-9000, 9000, 3)
    assert exact_sampling.seed_for(tokens, -13) == one_join(tokens, -13)


def test_integer_conversion_is_preserved_across_a_boundary():
    tokens = ["-12", 1.9, -1.9, True, False, np.int64(-9), 2**100] * 700
    assert exact_sampling.seed_for(tokens, 31) == one_join(tokens, 31)


def test_invalid_id_at_a_later_boundary_is_still_rejected():
    tokens = list(range(4096)) + ["not-an-id"]
    with pytest.raises(ValueError):
        exact_sampling.seed_for(tokens, 0)


@pytest.mark.parametrize("salt", [0, -5, 2**90])
def test_omitted_and_none_salts_use_the_configured_default(monkeypatch, salt):
    tokens = list(range(8193))
    monkeypatch.setattr(exact_sampling, "SEED_SALT", salt)
    expected = one_join(tokens, salt)
    assert exact_sampling.seed_for(tokens) == expected
    assert exact_sampling.seed_for(tokens, None) == expected
    assert exact_sampling.seed_for(tokens, 0) == one_join(tokens, 0)
