from butterfly.learning.evaluator import (
    BENCHMARK_RESERVED_EXACT_TARGETS,
    BENCHMARK_RESERVED_MATH,
    BENCHMARK_SUITE_VERSION,
    CASES,
    _case_result,
    benchmark_surface_prompts,
    normalize_surface,
)


def case(case_id):
    return next(c for c in CASES if c["id"] == case_id)


def test_suite_and_case_ids_are_unique():
    assert BENCHMARK_SUITE_VERSION == "0.00043"
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))


def test_surface_normalization_blocks_cosmetic_leaks():
    assert normalize_surface("¿Cuánto es 7 MÁS 6?") == normalize_surface("cuanto es 7 mas 6")
    assert normalize_surface("q onda") == normalize_surface("que onda")
    assert len(benchmark_surface_prompts()) == len({normalize_surface(c["prompt"]) for c in CASES})


def test_new_binding_cases_are_exact():
    c = case("exact_lz4")
    assert _case_result("LZ4", c)["critical_pass"]
    assert not _case_result("LZ4.", c)["critical_pass"]
    assert "lz4" in BENCHMARK_RESERVED_EXACT_TARGETS


def test_new_math_and_false_math_cases_are_strict():
    c = case("math_8_plus_9")
    assert _case_result("17", c)["critical_pass"]
    assert not _case_result("da 17", c)["critical_pass"]
    assert ("+", 8, 9) in BENCHMARK_RESERVED_MATH
    f = case("reject_false_math_new")
    assert _case_result("No. 9 + 6 = 15.", f)["critical_pass"]
    assert not _case_result("No, está mal.", f)["critical_pass"]


def test_comprehension_is_now_critical_regression_surface():
    c = case("token_new")
    assert _case_result("Un token es una pieza de texto que procesa el modelo.", c)["critical_pass"]
    assert not _case_result("Me llamo ButterflyAI.", c)["critical_pass"]
