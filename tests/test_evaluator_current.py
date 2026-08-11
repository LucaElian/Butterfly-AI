from butterfly.learning.evaluator import CASES, _case_result, benchmark_surface_prompts, normalize_surface


def case(case_id):
    return next(c for c in CASES if c["id"] == case_id)


def test_case_ids_and_surfaces_are_unique():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))
    assert len(benchmark_surface_prompts()) == len({normalize_surface(c["prompt"]) for c in CASES})


def test_api_semantics_accept_natural_spanish():
    answer = "Una API es una interfaz que permite que programas o sistemas se comuniquen de forma definida."
    assert _case_result(answer, case("api_new"))["critical_pass"]


def test_parameter_semantics_use_prompt_context():
    answer = "Es un número interno de la red que el entrenamiento modifica para mejorar sus predicciones."
    assert _case_result(answer, case("parameter"))["critical_pass"]


def test_file_and_folder_stay_distinct():
    assert _case_result(
        "Un archivo guarda información o contenido en la computadora.",
        case("file_new"),
    )["critical_pass"]
    assert not _case_result(
        "Es un contenedor usado para ordenar archivos y subcarpetas.",
        case("file_new"),
    )["critical_pass"]
