"""Compatibility filename kept so older project checkouts do not fail pytest.
The active evaluator policy is benchmark suite v0.00042.
"""
import unittest

from butterfly.learning.evaluator import BENCHMARK_SUITE_VERSION, CASES, _case_result


class EvaluatorCompatibilityTests(unittest.TestCase):
    def case(self, case_id):
        return next(c for c in CASES if c["id"] == case_id)

    def test_suite_was_upgraded(self):
        self.assertEqual(BENCHMARK_SUITE_VERSION, "0.00042")

    def test_natural_greeting_is_semantic(self):
        self.assertTrue(_case_result("¡Hey! Decime qué querés hacer.", self.case("hello_casual"))["critical_pass"])

    def test_exact_instruction_stays_exact(self):
        case = self.case("exact_blue")
        self.assertTrue(_case_result("azul", case)["critical_pass"])
        self.assertFalse(_case_result("El color es azul.", case)["critical_pass"])


if __name__ == "__main__":
    unittest.main()
