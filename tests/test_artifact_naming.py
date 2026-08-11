from butterfly.learning.evaluator import BENCHMARK_SUITE_ID
from butterfly.upgrade import _baseline_path, _comparison_path


def test_suite_prefix_is_not_duplicated_in_benchmark_artifacts():
    baseline = _baseline_path({"version": "sample"}).name
    comparison = _comparison_path(
        {"version": "active-sample"},
        {"version": "seed-sample"},
        {"version": "candidate-sample"},
    ).name
    for name in (baseline, comparison):
        assert "suite-suite-" not in name
        assert name.endswith(f"-{BENCHMARK_SUITE_ID}.json")
