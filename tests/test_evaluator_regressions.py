from butterfly.learning.evaluator import CASES, _case_result


def case(case_id):
    return next(c for c in CASES if c["id"] == case_id)


def test_greeting_reply_counts_as_greeting():
    assert _case_result("¡Hey! Decime qué querés hacer.", case("hello_casual"))["critical_pass"]


def test_identity_and_state_are_contrastive():
    assert _case_result("Me llamo ButterflyAI.", case("identity_plain"))["critical_pass"]
    assert not _case_result("Bien, gracias. Estoy lista para ayudarte.", case("identity_plain"))["critical_pass"]
    assert _case_result("Bien, gracias. Estoy lista para ayudarte.", case("how_are_you_plain"))["critical_pass"]
    assert not _case_result("Me llamo ButterflyAI.", case("how_are_you_plain"))["critical_pass"]


def test_exact_binding_rejects_extra_text():
    value = case("exact_cloud")
    assert _case_result("nube", value)["critical_pass"]
    assert not _case_result("La palabra es nube.", value)["critical_pass"]
    assert not _case_result('"nube"', value)["critical_pass"]


def test_false_math_needs_rejection_and_correct_result():
    value = case("reject_false_math_alt")
    assert _case_result("No. 7 + 8 = 15.", value)["critical_pass"]
    assert not _case_result("Sí, 16.", value)["critical_pass"]
    assert not _case_result("No, está mal.", value)["critical_pass"]


def test_unknown_fact_does_not_reward_invention():
    value = case("unknown_fictional_fact_alt")
    assert _case_result("No lo sé con esos datos; necesitaría información adicional.", value)["critical_pass"]
    assert not _case_result("La capital de Ormavia es Neral.", value)["critical_pass"]
