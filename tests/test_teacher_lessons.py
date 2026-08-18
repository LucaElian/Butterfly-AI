import json

from butterfly.learning.teacher import (
    TeacherTarget,
    install_teacher_lessons,
    load_teacher_config,
    parse_teacher_lessons,
    select_teacher_targets,
)
from butterfly.memory import MemoryStore


def _fake_response(prompt, cfg):
    assert "SOLO JSON" in prompt
    return json.dumps({
        "lessons": [
            {
                "task": "Explica que hacer cuando falta el nombre del archivo para completar una accion.",
                "lesson": "Cuando falta un dato necesario, la respuesta correcta es pedir ese dato antes de inventarlo o ejecutar una accion ambigua.",
                "result": "Necesito el nombre del archivo para seguir.",
            },
            {
                "task": "Indica como responder una solicitud incompleta sobre borrar un elemento.",
                "lesson": "Si una accion puede afectar datos y la instruccion no identifica el objetivo, hay que pedir aclaracion antes de continuar.",
                "result": "Decime que elemento queres borrar antes de hacerlo.",
            },
        ]
    })


def test_teacher_config_defaults_to_free_local_ollama():
    cfg = load_teacher_config()
    assert cfg["provider"] == "ollama"
    assert "api_key" not in json.dumps(cfg).lower()
    assert cfg["limits"]["lessons_per_target"] <= 8


def test_teacher_target_selection_supports_dynamic_family():
    targets = select_teacher_targets(dynamic_family="missing", limit=1)
    assert targets[0].curriculum_node == "instruction.missing"
    assert targets[0].dynamic_family == "missing"


def test_teacher_parser_rejects_duplicate_and_keeps_target():
    target = TeacherTarget("instruction.missing", "missing", "Missing information")
    rows = parse_teacher_lessons(_fake_response("Responde SOLO JSON", {}), target)
    assert len(rows) == 2
    assert {row["curriculum_node"] for row in rows} == {"instruction.missing"}
    assert {row["dynamic_family"] for row in rows} == {"missing"}


def test_teacher_lessons_install_verified_material_once(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    marked = []
    first = install_teacher_lessons(
        store=store,
        dynamic_family="missing",
        cfg={"limits": {"targets_per_run": 1, "lessons_per_target": 2}},
        request_func=_fake_response,
        mark_material_func=lambda node, status: marked.append((node, status)),
    )
    second = install_teacher_lessons(
        store=store,
        dynamic_family="missing",
        cfg={"limits": {"targets_per_run": 1, "lessons_per_target": 2}},
        request_func=_fake_response,
        mark_material_func=lambda node, status: marked.append((node, status)),
    )

    assert len(first["added"]) == 2
    assert len(second["added"]) == 0
    assert len(second["skipped"]) == 2
    assert marked == [("instruction.missing", "verified_packet"), ("instruction.missing", "verified_packet")]
    rows = store.approved_experiences(limit=10, minimum_quality=0.8)
    assert len(rows) == 2
    assert "ollama_teacher" in rows[0][2]


def test_teacher_lessons_dry_run_does_not_call_teacher_or_write(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    report = install_teacher_lessons(
        store=store,
        dynamic_family="missing",
        dry_run=True,
        request_func=lambda *_: (_ for _ in ()).throw(AssertionError("should not call teacher")),
        mark_material_func=lambda *_: (_ for _ in ()).throw(AssertionError("should not mark")),
    )
    assert report["planned_lessons"] >= 1
    assert report["added"] == []
    assert store.approved_experiences(limit=10) == []

def test_autonomy_installs_local_teacher_material_for_current_gap(monkeypatch):
    from butterfly.learning import autonomy
    from butterfly.learning import teacher

    calls = []

    monkeypatch.setattr(
        autonomy,
        "enrich_legacy_lesson",
        lambda raw: {
            **raw,
            "curriculum_node": "instruction.two_steps",
            "dynamic_family": "two_steps",
        },
    )
    monkeypatch.setattr(teacher, "load_teacher_config", lambda: {"enabled": True})

    def fake_install_teacher_lessons(**kwargs):
        calls.append(kwargs)
        return {
            "added": [{"experience_id": 123}],
            "skipped": [],
            "targets": [{"curriculum_node": "instruction.two_steps", "dynamic_family": "two_steps"}],
        }

    monkeypatch.setattr(teacher, "install_teacher_lessons", fake_install_teacher_lessons)

    report = autonomy._install_local_teacher_material(
        {
            "capabilities": [
                {
                    "capability": "instruction_format",
                    "focus_family": "two_steps",
                    "gap": 0.02,
                    "critical_count": 0,
                    "trainable": True,
                }
            ]
        },
        set(),
    )

    assert report["added"][0]["experience_id"] == 123
    assert calls[0]["node"] == "instruction.two_steps"
    assert calls[0]["dynamic_family"] is None

def test_teacher_lessons_retries_malformed_json(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    calls = []

    def flaky_response(prompt, cfg):
        calls.append(prompt)
        if len(calls) == 1:
            return '{"lessons":[{"task":"Explica una respuesta en dos pasos", "lesson":"texto incompleto'
        return _fake_response(prompt, cfg)

    report = install_teacher_lessons(
        store=store,
        dynamic_family="missing",
        cfg={"limits": {"targets_per_run": 1, "lessons_per_target": 2, "retry_attempts": 2}},
        request_func=flaky_response,
        mark_material_func=lambda *_: None,
    )

    assert len(calls) == 2
    assert "JSON invalido" in calls[1]
    assert len(report["added"]) == 2

def test_teacher_prompt_specializes_two_step_format(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    prompts = []

    def capture_response(prompt, cfg):
        prompts.append(prompt)
        return json.dumps({
            "lessons": [
                {
                    "task": "Da exactamente dos pasos para revisar un error de permisos.",
                    "lesson": "Cuando piden dos pasos, se responde con dos acciones numeradas y nada extra.",
                    "result": "1. Revisa el permiso del archivo.\n2. Prueba de nuevo la accion.",
                }
            ]
        })

    install_teacher_lessons(
        store=store,
        dynamic_family="two_steps",
        cfg={"limits": {"targets_per_run": 1, "lessons_per_target": 1, "retry_attempts": 1}},
        request_func=capture_response,
        mark_material_func=lambda *_: None,
    )

    assert "exactamente dos pasos numerados" in prompts[0]
    assert "solo 1. y 2." in prompts[0]


def test_teacher_lessons_accumulates_partial_valid_responses(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    calls = []

    def partial_response(prompt, cfg):
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps({
                "lessons": [
                    {
                        "task": "Explica que hacer cuando falta el archivo exacto.",
                        "lesson": "Si falta un dato necesario, hay que pedirlo antes de seguir.",
                        "result": "Necesito el nombre del archivo para seguir.",
                    }
                ]
            })
        return json.dumps({
            "lessons": [
                {
                    "task": "Indica que pedir si no se sabe la carpeta destino.",
                    "lesson": "Cuando falta la carpeta destino, la respuesta segura es pedir esa ruta.",
                    "result": "Decime la carpeta destino antes de continuar.",
                }
            ]
        })

    report = install_teacher_lessons(
        store=store,
        dynamic_family="missing",
        cfg={"limits": {"targets_per_run": 1, "lessons_per_target": 2, "retry_attempts": 2}},
        request_func=partial_response,
        mark_material_func=lambda *_: None,
    )

    assert len(calls) == 2
    assert "Faltan 1 lecciones nuevas" in calls[1]
    assert len(report["added"]) == 2

def test_teacher_parser_rejects_bad_two_step_teacher_patterns():
    target = TeacherTarget("instruction.two_steps", "two_steps", "Two-step ordered instructions")
    payload = json.dumps({
        "lessons": [
            {
                "task": "Instrui para limpiar un plato. Instrui para limpiar un plato.",
                "lesson": "Las instrucciones deben ser claras y contar con dos pasos identicos.",
                "result": "1. Listar los ingredientes necesarios. 2. Describir como preparar la tarea.",
            },
            {
                "task": "Da exactamente dos pasos para revisar un error de permisos.",
                "lesson": "Cuando piden dos pasos, la respuesta debe resolver la accion con dos pasos numerados y nada extra.",
                "result": "1. Revisa que el archivo tenga permisos de lectura. 2. Ejecuta de nuevo la accion despues de corregirlos.",
            },
        ]
    })

    rows = parse_teacher_lessons(payload, target)

    assert len(rows) == 1
    assert rows[0]["task"] == "Da exactamente dos pasos para revisar un error de permisos."


def test_teacher_lessons_retry_when_two_step_rows_are_invalid(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    calls = []

    def response(prompt, cfg):
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps({
                "lessons": [
                    {
                        "task": "Instrui para limpiar un plato. Instrui para limpiar un plato.",
                        "lesson": "Las instrucciones deben ser claras y contar con dos pasos identicos.",
                        "result": "1. Listar los ingredientes necesarios. 2. Describir como hacer la tarea.",
                    }
                ]
            })
        return json.dumps({
            "lessons": [
                {
                    "task": "Da exactamente dos pasos para guardar un archivo nuevo.",
                    "lesson": "Cuando piden dos pasos, se responde con dos acciones numeradas que ejecutan la tarea sin texto extra.",
                    "result": "1. Elige la carpeta donde queres guardarlo. 2. Escribi el nombre del archivo y confirma el guardado.",
                }
            ]
        })

    report = install_teacher_lessons(
        store=store,
        dynamic_family="two_steps",
        cfg={"limits": {"targets_per_run": 1, "lessons_per_target": 1, "retry_attempts": 2}},
        request_func=response,
        mark_material_func=lambda *_: None,
    )

    assert len(calls) == 2
    assert len(report["added"]) == 1
    assert "guardar un archivo" in report["added"][0]["task"]

