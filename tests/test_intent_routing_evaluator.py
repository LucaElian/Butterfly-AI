from butterfly.learning.evaluator import CASES, PROMOTION_THRESHOLDS, _case_result


def case(case_id):
    return next(c for c in CASES if c["id"] == case_id)


def test_intent_routing_is_a_real_hard_gate():
    assert PROMOTION_THRESHOLDS["intent_routing_component"] == 0.75


def test_contextual_api_and_parameter_answers_do_not_require_term_repetition():
    assert _case_result(
        "Es una interfaz para que distintos programas intercambien información.",
        case("api_new"),
    )["semantic"] == 1.0
    assert _case_result(
        "Es un número interno que se ajusta durante el entrenamiento.",
        case("parameter"),
    )["semantic"] == 1.0


def test_wrong_intent_is_not_rewarded_just_for_fluent_language():
    assert _case_result("Hola, ¿en qué te ayudo?", case("route_file_2"))["semantic"] == 0.0
    assert _case_result("Soy ButterflyAI.", case("route_state_2"))["semantic"] == 0.0
    assert _case_result("Una API conecta programas.", case("route_thanks_2"))["semantic"] == 0.0
