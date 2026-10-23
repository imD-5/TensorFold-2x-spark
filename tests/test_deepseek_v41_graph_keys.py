"""A concurrent round runs in a captured graph of the next size up, and the warm-up covers every graph it can need."""

import importlib.util
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "src/tensorfold/families/deepseek_v41/cuda/graph_keys.py"
spec = importlib.util.spec_from_file_location("graph_keys_under_test", SRC)
keys = importlib.util.module_from_spec(spec)
spec.loader.exec_module(keys)    # no torch: the module holds only the key arithmetic

CAP = 1_048_576 + 16 + 8         # --context 1048576 with the engine's 16 rows and 8 spare
FINE = keys.row_sizes(16, (1, 2, 3, 4, 6, 8, 12, 16))
LONG = keys.row_sizes(16, (1, 2, 4, 8, 16))
FINE_CONTEXT = 32_768


def test_buckets_double_from_the_minimum_and_stop_at_the_window():
    assert keys.context_buckets(CAP, CAP) == [CAP, 1_048_576, 524_288, 262_144, 131_072, 65_536, 32_768, 16_384,
                                              8_192, 4_096, 2_048, 1_024]
    assert keys.context_buckets(5_000, 5_000) == [5_000, 4_096, 2_048, 1_024]


def test_the_sizes_end_at_the_most_rows():
    assert keys.row_sizes(16, (1, 2, 4, 8, 16)) == [1, 2, 4, 8, 16]
    assert keys.row_sizes(16, (2, 5)) == [2, 5, 16]
    assert keys.row_sizes(12, (1, 2, 4, 8, 16)) == [1, 2, 4, 8, 12]      # none above the engine most rows
    assert keys.row_sizes(16, (0,)) == list(range(1, 17))                # 0: a graph for every count
    assert keys.row_sizes(16, ()) == list(range(1, 17))


def test_the_default_sizes_are_exact_for_the_counts_rounds_have():
    # one stream runs 1 to 6 rows, two to three streams 4 to 12, four streams 11 to 16 (measured on the engine)
    assert keys.ROUND_ROWS == (1, 2, 3, 4, 6, 8, 12, 16) and keys.ROUND_ROWS_LONG == (1, 2, 4, 8, 16)
    assert keys.FINE_CONTEXT == FINE_CONTEXT
    assert [keys.padded_rows(r, FINE) for r in range(1, 17)] == [1, 2, 3, 4, 6, 6, 8, 8, 12, 12, 12, 12, 16, 16, 16, 16]


def test_a_round_takes_the_smallest_size_that_holds_it():
    assert [keys.padded_rows(r, LONG) for r in range(1, 17)] == [1, 2, 4, 4, 8, 8, 8, 8] + [16] * 8
    assert keys.padded_rows(16, LONG) == 16


def test_short_contexts_use_the_fine_sizes_and_long_ones_the_short_ladder():
    assert keys.sizes_for(1_024, FINE, LONG, FINE_CONTEXT) == FINE
    assert keys.sizes_for(32_768, FINE, LONG, FINE_CONTEXT) == FINE
    assert keys.sizes_for(65_536, FINE, LONG, FINE_CONTEXT) == LONG
    assert keys.sizes_for(65_536, FINE) == FINE                           # no short ladder: one ladder everywhere


def test_every_round_the_engine_can_run_has_a_warm_key():
    for fine, long in ((FINE, LONG), (keys.row_sizes(16, (0,)), None), (keys.row_sizes(16, (3, 6, 12)), None)):
        warm = set(keys.round_keys(CAP, CAP, fine, long, FINE_CONTEXT))
        for deepest in list(range(1, 70_000, 97)) + [2**k + d for k in range(10, 21) for d in (-1, 0, 1)] + [CAP]:
            b = keys.bucket_for(deepest, CAP)
            for rows in range(1, 17):
                assert (keys.padded_rows(rows, keys.sizes_for(b, fine, long, FINE_CONTEXT)), b) in warm, (rows, deepest)


def test_the_default_warm_set_is_close_to_the_one_before():
    # the graphs held at ready: 64 before (16 row counts at 4 buckets). Now 8 sizes at the 6 buckets up to 32768 and
    # 5 sizes at the 6 above it (the pool cap bucket included): 78. At the usable extent of a 1M window: 73.
    assert len(keys.round_keys(CAP, CAP, FINE, LONG, FINE_CONTEXT)) == 6 * 8 + 6 * 5
    assert len(keys.round_keys(1_048_576, CAP, FINE, LONG, FINE_CONTEXT)) == 6 * 8 + 5 * 5


def test_warm_order_captures_the_largest_bucket_first():
    order = keys.round_keys(CAP, CAP, [1, 2, 4, 8, 16])
    assert order[0] == (16, CAP) and order[4] == (1, CAP)
    assert [b for _, b in order] == sorted((b for _, b in order), reverse=True)
    assert len(set(order)) == len(order)


def test_a_small_window_has_few_keys():
    assert keys.round_keys(2_000, 2_000, [1, 2, 4]) == [(r, b) for b in (2_000, 1_024) for r in (4, 2, 1)]


def test_the_warm_set_stops_at_the_usable_extent():
    # the window pool is the context rounded down, the pool cap has spare rows above it: no stream reaches the top bucket
    usable = 1_048_576
    warm = keys.round_keys(usable, CAP, [1, 2])
    assert CAP not in [b for _, b in warm] and len(warm) == 11 * 2
    for deepest in list(range(1, usable, 4_099)) + [usable]:
        assert (1, keys.bucket_for(deepest, CAP)) in set(warm)
