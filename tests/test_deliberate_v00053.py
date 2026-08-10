from butterfly.corpus.deliberate import (
    _canonical_math,
    _math_pairs,
    _make_copy_targets,
    build_all_rows,
)
from butterfly.deliberate_trainer import AssistantOnlyDataset
from butterfly.learning.evaluator import (
    BENCHMARK_RESERVED_EXACT_TARGETS,
    BENCHMARK_RESERVED_MATH,
    benchmark_surface_prompts,
    normalize_surface,
)
from butterfly.tokenizer import ByteTokenizer


def test_copy_train_valid_and_benchmark_values_are_three_disjoint_sets():
    train, valid = _make_copy_targets()
    t = {x.casefold() for x in train}
    v = {x.casefold() for x in valid}
    b = {x.casefold() for x in BENCHMARK_RESERVED_EXACT_TARGETS}
    assert not (t & v)
    assert not (t & b)
    assert not (v & b)


def test_arithmetic_pairs_hold_out_benchmark_and_validation_pairs():
    train, valid = _math_pairs()
    train_canon = {_canonical_math(*x) for x in train}
    valid_canon = {_canonical_math(*x) for x in valid}
    bench_canon = {_canonical_math(*x) for x in BENCHMARK_RESERVED_MATH}
    assert not (train_canon & valid_canon)
    assert not (train_canon & bench_canon)
    assert not (valid_canon & bench_canon)


def test_all_stage_surfaces_keep_benchmark_and_validation_families_out():
    stages = build_all_rows()
    bench = benchmark_surface_prompts()
    for train, valid in stages.values():
        assert not ({r["family"] for r in train} & {r["family"] for r in valid})
        assert not any(normalize_surface(r["user"]) in bench for r in train + valid)


def test_assistant_only_dataset_masks_user_context_and_keeps_answer_target():
    tok = ByteTokenizer()
    ds = AssistantOnlyDataset([{"user": "escribi solo rojo", "assistant": "rojo"}], tok, seq_len=96)
    x, y = ds[0]
    visible_targets = y[y != -100].tolist()
    decoded = tok.decode(visible_targets)
    assert "rojo" in decoded
    # User context must not appear among supervised target tokens.
    assert "escribi solo" not in decoded
