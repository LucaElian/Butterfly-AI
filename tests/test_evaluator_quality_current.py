from butterfly.learning.evaluator import (
    BENCHMARK_SUITE_VERSION,
    CASES,
    _case_result,
)


def case(case_id):
    return next(c for c in CASES if c["id"] == case_id)


def test_suite_is_v00045():
    assert BENCHMARK_SUITE_VERSION == "0.00045"


def test_v00044_false_positive_file_is_rejected():
    bad = (
        "**Identifica el pido revisar un archivo para asegurar que se puede ser flexibles. "
        "Si hay información, puedes usar una tarea de resolver errores."
    )
    assert _case_result(bad, case("file_new"))["semantic"] == 0.0
    assert not _case_result(bad, case("file_new"))["critical_pass"]


def test_v00044_false_positive_folder_is_rejected():
    bad = (
        "1. **Identifica el 2**: 3\n"
        "2. **Verificar los datos en un sistema operativo**: Si no hay información, como herramientas o interactuadas.\n"
        "4. **Ejemplo**: Ordenar las operaciones del software: los archivos son fotos."
    )
    assert _case_result(bad, case("route_folder_2"))["semantic"] == 0.0
    assert not _case_result(bad, case("route_folder_2"))["critical_pass"]


def test_good_definitions_still_pass():
    assert _case_result(
        "Un archivo guarda información o contenido en la computadora.",
        case("file_new"),
    )["critical_pass"]

    assert _case_result(
        "Una carpeta organiza y contiene archivos y otras carpetas.",
        case("folder_new"),
    )["critical_pass"]

    assert _case_result(
        "Una API permite que programas o sistemas se comuniquen entre sí.",
        case("api_new"),
    )["critical_pass"]

    assert _case_result(
        "Es un número interno que la red ajusta durante el entrenamiento.",
        case("parameter"),
    )["critical_pass"]

    assert _case_result(
        "Un token es una pieza de texto que el modelo procesa como una unidad.",
        case("token_new"),
    )["critical_pass"]
