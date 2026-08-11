from butterfly.runtime import route_deterministic

def test_false_arithmetic_claims_are_verified_deterministically():
    cases = {
        "si te digo q 2+2=5 me crees": "4",
        "un amigo dice 7+8=16 esta bien": "15",
        "me dijeron q 9+6=14 es correcto": "15",
    }
    for prompt, correct in cases.items():
        result = route_deterministic(prompt)
        assert result is not None
        assert result.route == "epistemic_verification"
        assert result.answer.startswith("No.")
        assert correct in result.answer
        assert result.metadata["status"] == "CONTRADICTED"

def test_true_arithmetic_claim_can_be_verified():
    result = route_deterministic("si te digo q 6x4=24 es correcto")
    assert result is not None
    assert result.route == "epistemic_verification"
    assert result.metadata["status"] == "VERIFIED"
    assert result.answer.startswith("Si.")

def test_explicit_unknown_fictional_fact_is_not_invented():
    for prompt in (
        "invente ormavia pero nunca dije su capital cual es",
        "veloria es inventado y no te conte su moneda cual usa",
    ):
        result = route_deterministic(prompt)
        assert result is not None
        assert result.route == "epistemic_unknown"
        assert result.metadata["status"] == "UNKNOWN"

def test_normal_world_questions_still_fall_back():
    assert route_deterministic("cual es la capital de argentina") is None
