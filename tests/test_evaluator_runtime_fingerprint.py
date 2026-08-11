import inspect

from butterfly.learning.evaluator import BENCHMARK_SUITE_ID, benchmark_suite_id, behavior_benchmark
from butterfly.runtime import runtime_fingerprint_payload


def test_runtime_rules_participate_in_suite_fingerprint():
    payload = runtime_fingerprint_payload()
    assert "parse_exact_output" in payload
    assert "parse_arithmetic" in payload
    assert "runtime_respond" in payload
    assert BENCHMARK_SUITE_ID == benchmark_suite_id()


def test_evaluator_keeps_neural_diagnostics():
    source = inspect.getsource(behavior_benchmark)
    assert "neural_binding_component" in source
    assert "neural_arithmetic_component" in source
