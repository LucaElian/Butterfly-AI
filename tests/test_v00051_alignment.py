import json
from pathlib import Path

from butterfly.corpus.alignment_v00051 import (
    BENCHMARK_PROMPTS,
    MANIFEST,
    STAGE_FILES,
    _norm,
    build_alignment_corpus_v00051,
)
from butterfly.learning.evaluator import BENCHMARK_RESERVED_EXACT_TARGETS


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_v00051_corpus_has_no_surface_benchmark_leak_or_family_overlap():
    manifest = build_alignment_corpus_v00051()
    assert manifest["benchmark_surface_leaks"] == 0
    assert manifest["validation_surface_family_overlap"] == 0

    for train_path, valid_path in STAGE_FILES.values():
        train = _rows(train_path)
        valid = _rows(valid_path)
        train_norm = {_norm(r["user"]) for r in train}
        valid_norm = {_norm(r["user"]) for r in valid}
        assert not (train_norm & valid_norm)
        assert not ((train_norm | valid_norm) & BENCHMARK_PROMPTS)
        assert not ({r.get("surface_family") for r in train} & {r.get("surface_family") for r in valid})


def test_benchmark_exact_copy_targets_are_reserved():
    reserved = {x.casefold() for x in BENCHMARK_RESERVED_EXACT_TARGETS}
    for train_path, valid_path in STAGE_FILES.values():
        for row in _rows(train_path) + _rows(valid_path):
            if row.get("category") == "exact_copy":
                assert row["assistant"].casefold() not in reserved


def test_leak_normalization_treats_cosmetic_variants_as_same():
    assert _norm("¿Cómo te llamás?") == _norm("como te llamas")
    assert _norm("buenas q onda") == _norm("Buenas, que onda!!!")
