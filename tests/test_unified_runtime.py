from butterfly.runtime import parse_arithmetic, parse_exact_output, route_deterministic


def test_exact_output_generalizes_to_unseen_values():
    cases = {
        "responde solamente con la palabra turquesa": "turquesa",
        "sin comillas ni explicacion escribi ZX91": "ZX91",
        "solo devolve K4M8": "K4M8",
        "escribi cedro sin agregar absolutamente nada": "cedro",
        'responde exactamente "hola mundo" y nada mas': "hola mundo",
    }
    for prompt, expected in cases.items():
        assert parse_exact_output(prompt) == expected


def test_exact_router_is_conservative():
    assert parse_exact_output("en una sola oracion explica que es una api") is None
    assert parse_exact_output("en menos de 10 palabras resume este texto") is None
    assert parse_exact_output("decime como estas") is None


def test_arithmetic_direct_requests():
    cases = {
        "cuanto es 23+19 responde solo el numero": "42",
        "15 menos 8 solo resultado": "7",
        "6x7 respuesta solo numero": "42",
        "9 por 5 sin explicar": "45",
        "10 dividido por 4 solo resultado": "2.5",
    }
    for prompt, expected in cases.items():
        assert parse_arithmetic(prompt) == expected


def test_arithmetic_does_not_capture_truth_claims():
    assert parse_arithmetic("si te digo q 2+2=5 me crees") is None
    assert parse_arithmetic("un amigo dice 7+8=16 esta bien") is None
    assert parse_arithmetic("me dijeron q 9+6=14 es correcto") is None


def test_arithmetic_routes_before_binding():
    result = route_deterministic("cuanto es 2+2 responde solo el numero")
    assert result is not None
    assert result.route == "arithmetic"
    assert result.answer == "4"
