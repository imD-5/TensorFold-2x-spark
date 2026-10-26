"""seed_for over a long prompt: feeding the hash in slices gives the bytes of one join, so the same seed."""

import hashlib
import random

import pytest

from tensorfold.engine import exact_sampling


def one_join(tokens, salt):
    """The bytes a sliced hash must reproduce: every id, a comma apart, then the salt."""

    joined = ",".join(str(int(t)) for t in tokens) + f"|{salt}"
    return int.from_bytes(hashlib.sha256(joined.encode()).digest()[:8], "little") & ((1 << 63) - 1)


@pytest.mark.parametrize("n", [0, 1, 2, 4095, 4096, 4097, 8192, 8193, 50000])
def test_a_sliced_seed_equals_the_one_join(n):
    tokens = [random.Random(n).randrange(0, 200000) for _ in range(n)]
    assert exact_sampling.seed_for(tokens, 0) == one_join(tokens, 0)


def test_a_changed_token_moves_the_seed():
    tokens = list(range(9000))
    assert exact_sampling.seed_for(tokens, 0) != exact_sampling.seed_for(tokens[:-1] + [1], 0)


def test_a_slice_boundary_between_two_ids_changes_nothing():
    """A comma the slice boundary could swallow would join two ids; 4096 and 4097 must both match the join."""

    tokens = list(range(4097))
    assert [exact_sampling.seed_for(tokens[:n], 7) for n in (4096, 4097)] == [one_join(tokens[:4096], 7),
                                                                             one_join(tokens, 7)]
