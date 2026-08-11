import inspect
from butterfly.generation import generate
from butterfly.learning.evaluator import BENCHMARK_GENERATION_CONFIG, BENCHMARK_SUITE_ID

def test_generation_allows_short_exact_answers():
    assert inspect.signature(generate).parameters["min_new_tokens"].default == 1

def test_benchmark_generation_policy_allows_eos_after_first_token():
    assert BENCHMARK_GENERATION_CONFIG["min_new_tokens"] == 1

def test_suite_remains_dynamic():
    assert BENCHMARK_SUITE_ID.startswith("suite-") and len(BENCHMARK_SUITE_ID) == 18
