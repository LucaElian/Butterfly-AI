from butterfly.epistemic.engine import EpistemicEngine


def test_math_true():
    assert EpistemicEngine().verify("2 + 2 = 4").status.value == "VERIFIED"


def test_math_false():
    assert EpistemicEngine().verify("2 + 2 = 5").status.value == "CONTRADICTED"
