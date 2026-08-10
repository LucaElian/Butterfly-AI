"""Compatibility filename kept intentionally.
The active evaluator policy is benchmark suite v0.00043.
"""
from butterfly.learning.evaluator import BENCHMARK_SUITE_VERSION, CASES, _case_result


def case(case_id):
    return next(c for c in CASES if c["id"] == case_id)


def test_suite_was_upgraded_without_losing_old_regressions():
    assert BENCHMARK_SUITE_VERSION == "0.00043"
    assert _case_result("¡Hey! Decime qué querés hacer.", case("hello_casual"))["critical_pass"]
    assert _case_result("azul", case("exact_blue"))["critical_pass"]
    assert not _case_result("El color es azul.", case("exact_blue"))["critical_pass"]
