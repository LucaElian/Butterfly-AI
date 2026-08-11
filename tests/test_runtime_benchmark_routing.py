from butterfly.learning.evaluator import CASES
from butterfly.runtime import route_deterministic


def test_all_final_binding_cases_are_runtime_routed():
    cases = [case for case in CASES if case.get("skill") == "binding"]
    assert cases
    for case in cases:
        result = route_deterministic(case["prompt"])
        assert result is not None, case["id"]
        assert result.route == "exact_binding", case["id"]
        assert result.answer.casefold() == str(case["exact_answer"]).casefold(), case["id"]


def test_all_final_arithmetic_cases_are_runtime_routed():
    cases = [case for case in CASES if case.get("skill") == "arithmetic"]
    assert cases
    for case in cases:
        result = route_deterministic(case["prompt"])
        assert result is not None, case["id"]
        assert result.route == "arithmetic", case["id"]
        assert result.answer == str(case["exact_answer"]), case["id"]


def test_false_math_is_not_swallowed_by_calculator():
    cases = [case for case in CASES if case.get("validator") == "false_math"]
    assert cases
    for case in cases:
        result = route_deterministic(case["prompt"])
        assert result is None or result.route != "arithmetic", case["id"]
