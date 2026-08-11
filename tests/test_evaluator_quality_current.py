from butterfly.learning.evaluator import CASES, _case_result


def case(case_id):
    return next(c for c in CASES if c["id"] == case_id)


def test_procedural_noise_is_not_a_file_definition():
    bad = (
        "**Identifica el pido revisar un archivo para asegurar que se puede ser flexibles. "
        "Si hay información, puedes usar una tarea de resolver errores."
    )
    assert _case_result(bad, case("file_new"))["semantic"] == 0.0


def test_list_noise_is_not_a_folder_definition():
    bad = (
        "1. Identifica el 2: 3\n"
        "2. Verificar los datos en un sistema operativo.\n"
        "4. Ejemplo: Ordenar las operaciones del software: los archivos son fotos."
    )
    assert _case_result(bad, case("route_folder_2"))["semantic"] == 0.0
