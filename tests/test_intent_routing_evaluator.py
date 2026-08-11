from butterfly.learning.evaluator import CASES, PROMOTION_THRESHOLDS, _case_result


def case(case_id):
    return next(c for c in CASES if c["id"] == case_id)


def test_intent_routing_remains_a_hard_gate():
    assert PROMOTION_THRESHOLDS["intent_routing_component"] == 0.75


def test_wrong_intent_is_not_rewarded():
    assert _case_result("Hola, ¿en qué te ayudo?", case("route_file_2"))["semantic"] == 0.0
    assert _case_result("Soy ButterflyAI.", case("route_state_2"))["semantic"] == 0.0
