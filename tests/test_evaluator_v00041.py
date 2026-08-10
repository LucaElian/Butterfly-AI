import unittest

from butterfly.learning.evaluator import CASES, _case_result


class EvaluatorV00041Tests(unittest.TestCase):
    def case(self, case_id):
        return next(c for c in CASES if c["id"] == case_id)

    def test_good_greeting_passes(self):
        row = _case_result("Hola, ¿en qué puedo ayudarte?", self.case("hello"))
        self.assertTrue(row["critical_pass"])
        self.assertGreater(row["score"], 0.85)

    def test_keyword_plus_unrelated_ramble_does_not_pass(self):
        answer = (
            "Buenas: si te pido revisar un archivo, qué preguntas o comprobaciones "
            "harías antes de empezar el sistema operativo?"
        )
        row = _case_result(answer, self.case("hello_buenas"))
        self.assertFalse(row["critical_pass"])
        self.assertLess(row["score"], 0.70)

    def test_identity_requires_butterfly(self):
        bad = _case_result("¿Qué tipo de archivo querés revisar?", self.case("identity"))
        good = _case_result("Me llamo Butterfly.", self.case("identity"))
        self.assertFalse(bad["critical_pass"])
        self.assertTrue(good["critical_pass"])

    def test_exact_instruction_is_exact(self):
        case = self.case("exact_blue")
        self.assertTrue(_case_result("azul", case)["critical_pass"])
        self.assertFalse(_case_result("El color es azul.", case)["critical_pass"])

    def test_unknown_fact_prefers_uncertainty(self):
        case = self.case("unknown_fictional_fact")
        good = _case_result(
            "No sé cuál es la capital porque no tengo esa información; tendría que verificarla.",
            case,
        )
        bad = _case_result("La capital de Zarbelia es Torven.", case)
        self.assertTrue(good["critical_pass"])
        self.assertFalse(bad["critical_pass"])


if __name__ == "__main__":
    unittest.main()
