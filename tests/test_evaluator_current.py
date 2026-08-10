from butterfly.learning.evaluator import (
    BENCHMARK_SUITE_VERSION,
    CASES,
    _case_result,
    benchmark_surface_prompts,
    normalize_surface,
)


def case(case_id):
    return next(c for c in CASES if c["id"] == case_id)


def test_current_suite_has_valid_version_and_unique_ids():
    # Permanent invariant: evaluator must expose a non-empty semantic suite version.
    # Do NOT hardcode 0.00044/0.00045/etc. here.
    assert isinstance(BENCHMARK_SUITE_VERSION, str)
    assert BENCHMARK_SUITE_VERSION.strip()

    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))
    assert len(benchmark_surface_prompts()) == len(
        {normalize_surface(c["prompt"]) for c in CASES}
    )


def test_old_exact_regressions_remain_strict():
    assert _case_result("azul", case("exact_blue"))["critical_pass"]
    assert not _case_result("El color es azul.", case("exact_blue"))["critical_pass"]
    assert _case_result("17", case("math_8_plus_9"))["critical_pass"]
    assert not _case_result("da 17", case("math_8_plus_9"))["critical_pass"]


def test_api_semantics_accept_natural_spanish():
    c = case("api_new")
    good = (
        "Una API es una interfaz que permite que programas o sistemas "
        "se comuniquen de forma definida."
    )
    assert _case_result(good, c)["critical_pass"]
    assert not _case_result("Soy ButterflyAI.", c)["critical_pass"]


def test_parameter_semantics_use_prompt_context():
    c = case("parameter")
    assert _case_result(
        "Es un número interno de la red que el entrenamiento modifica "
        "para mejorar sus predicciones.",
        c,
    )["critical_pass"]
    assert _case_result(
        "Un valor que la red aprende y ajusta durante el entrenamiento.",
        c,
    )["critical_pass"]
    assert not _case_result("Es una carpeta que ordena archivos.", c)["critical_pass"]


def test_definition_validators_distinguish_file_from_folder():
    assert _case_result(
        "Un archivo guarda información o contenido en la computadora.",
        case("file_new"),
    )["critical_pass"]

    assert not _case_result(
        "Es un contenedor usado para ordenar archivos y subcarpetas.",
        case("file_new"),
    )["critical_pass"]

    assert _case_result(
        "Una carpeta organiza y contiene archivos y otras carpetas.",
        case("folder_new"),
    )["critical_pass"]


def test_focused_intent_routing_surfaces_exist():
    routing = [c for c in CASES if c.get("intent_route")]
    assert len(routing) >= 15
    assert _case_result(
        "Soy ButterflyAI.",
        case("route_identity_2"),
    )["critical_pass"]
    assert not _case_result(
        "Una carpeta organiza archivos.",
        case("route_identity_2"),
    )["critical_pass"]
    assert _case_result(
        "Bien, gracias.",
        case("route_state_2"),
    )["critical_pass"]
