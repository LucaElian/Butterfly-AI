import unittest

from butterfly.learning.evaluator import CASES, _case_result


class EvaluatorV00042Tests(unittest.TestCase):
    def case(self, case_id):
        return next(c for c in CASES if c["id"] == case_id)

    def test_natural_hey_reply_counts_as_greeting(self):
        row = _case_result("¡Hey! Decime qué querés hacer.", self.case("hello_casual"))
        self.assertTrue(row["critical_pass"])

    def test_identity_and_state_are_contrastive(self):
        identity = self.case("identity_plain")
        state = self.case("how_are_you_plain")
        self.assertTrue(_case_result("Me llamo ButterflyAI.", identity)["critical_pass"])
        self.assertFalse(_case_result("Bien, gracias. Estoy lista para ayudarte.", identity)["critical_pass"])
        self.assertTrue(_case_result("Bien, gracias. Estoy lista para ayudarte.", state)["critical_pass"])
        self.assertFalse(_case_result("Me llamo ButterflyAI.", state)["critical_pass"])

    def test_exact_binding_rejects_extra_text(self):
        case = self.case("exact_cloud")
        self.assertTrue(_case_result("nube", case)["critical_pass"])
        self.assertFalse(_case_result("La palabra es nube.", case)["critical_pass"])
        self.assertFalse(_case_result('"nube"', case)["critical_pass"])

    def test_false_math_needs_rejection_and_correct_result(self):
        case = self.case("reject_false_math_alt")
        self.assertTrue(_case_result("No. 7 + 8 = 15.", case)["critical_pass"])
        self.assertFalse(_case_result("Sí, 16.", case)["critical_pass"])
        self.assertFalse(_case_result("No, está mal.", case)["critical_pass"])

    def test_unknown_fact_does_not_reward_invention(self):
        case = self.case("unknown_fictional_fact_alt")
        self.assertTrue(_case_result("No lo sé con esos datos; necesitaría información adicional.", case)["critical_pass"])
        self.assertFalse(_case_result("La capital de Ormavia es Neral.", case)["critical_pass"])


if __name__ == "__main__":
    unittest.main()
