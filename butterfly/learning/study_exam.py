from __future__ import annotations

from typing import Any

from ..generation import generate
from .evaluator import BENCHMARK_GENERATION_CONFIG, BENCHMARK_MAX_NEW_TOKENS, _case_result

STUDY_RESERVED_EXACT_TARGETS = {
    "Q5N8", "V27M", "HP6", "T4R9", "DK38", "Z6P",
    "marmol", "jazmin", "brisa", "cedro", "L8C4", "FG7",
}


def _binding(case_id: str, prompt: str, answer: str) -> dict[str, Any]:
    return {
        "id": case_id, "group": "binding", "category": "instruction", "skill": "binding",
        "prompt": prompt, "exact_answer": answer, "max_words": 1, "direct": True, "no_list": True,
    }


def _fmt(case_id: str, prompt: str, **kwargs) -> dict[str, Any]:
    return {"id": case_id, "group": "instruction_format", "category": "instruction", "prompt": prompt, **kwargs}


def _keep(group: str, case_id: str, category: str, prompt: str, **kwargs) -> dict[str, Any]:
    return {"id": case_id, "group": group, "category": category, "prompt": prompt, **kwargs}


STUDY_CASES: list[dict[str, Any]] = [
    _binding("sb01", "devolve solamente Q5N8", "Q5N8"),
    _binding("sb02", "escribi V27M sin ningun agregado", "V27M"),
    _binding("sb03", "tu salida exacta tiene que ser HP6", "HP6"),
    _binding("sb04", "solo responde T4R9", "T4R9"),
    _binding("sb05", "contesta nada mas DK38", "DK38"),
    _binding("sb06", "pone exactamente Z6P y termina", "Z6P"),
    _binding("sb07", "quiero solo marmol como respuesta", "marmol"),
    _binding("sb08", "sin explicacion escribi jazmin", "jazmin"),
    _binding("sb09", "la unica palabra debe ser brisa", "brisa"),
    _binding("sb10", "responde unicamente cedro", "cedro"),
    _binding("sb11", "nada mas que L8C4", "L8C4"),
    _binding("sb12", "devolve FG7 y no agregues texto", "FG7"),


    _fmt("sf01", "en exactamente una oracion explica: una ruta indica donde esta un archivo",
         format_family="sentence", required_groups=[["ruta"], ["archivo", "ubicacion", "donde"]], max_sentences=1, max_words=32, no_list=True),
    _fmt("sf02", "usa una sola oracion para decir que un registro guarda eventos",
         format_family="sentence", required_groups=[["registro"], ["evento", "guard"]], max_sentences=1, max_words=28, no_list=True),
    _fmt("sf03", "solo una oracion: explica que una cache reutiliza datos temporales",
         format_family="sentence", required_groups=[["cache"], ["dato", "temporal", "reutil"]], max_sentences=1, max_words=32, no_list=True),
    _fmt("sf04", "una sola oracion para explicar que un proceso ejecuta una tarea",
         format_family="sentence", required_groups=[["proceso"], ["ejecut", "tarea", "programa"]], max_sentences=1, max_words=32, no_list=True),
    _fmt("sf05", "sin lista y en exactamente una oracion explica que hace un buffer",
         format_family="sentence", required_groups=[["buffer"], ["dato", "temporal", "guard"]], max_sentences=1, max_words=34, no_list=True),

    _fmt("sf06", "dame exactamente dos pasos numerados: primero revisa el error y segundo verifica la ruta",
         format_family="two_steps", required_groups=[["error"], ["ruta"]], expected_items=2, max_words=42),
    _fmt("sf07", "solo dos pasos numerados: comprobar la red y luego revisar el servicio",
         format_family="two_steps", required_groups=[["red"], ["servicio"]], expected_items=2, max_words=42),
    _fmt("sf08", "quiero dos pasos exactos: identifica el archivo y despues comprueba sus permisos",
         format_family="two_steps", required_groups=[["archivo"], ["permiso"]], expected_items=2, max_words=42),
    _fmt("sf09", "exactamente dos pasos para revisar un paquete que no se instala",
         format_family="two_steps", required_groups=[["error", "paquete"], ["dependencia", "version", "configuracion"]], expected_items=2, max_words=55),
    _fmt("sf10", "dos pasos numerados y nada mas para revisar un script que no encuentra una ruta",
         format_family="two_steps", required_groups=[["script", "error"], ["ruta", "archivo", "ubicacion"]], expected_items=2, max_words=55),

    _fmt("sf11", "quiero mover un archivo pero no dije a que carpeta, que haces",
         format_family="missing", required_groups=[["pregunt", "pedir", "necesit", "falta"], ["carpeta", "destino", "donde"]], max_words=48, no_list=True),
    _fmt("sf12", "quiero enviar un mensaje pero no dije el destinatario, como seguis",
         format_family="missing", required_groups=[["pregunt", "pedir", "necesit", "falta"], ["destinatario", "quien"]], max_words=48, no_list=True),
    _fmt("sf13", "quiero filtrar datos pero no te di ningun criterio, que deberias hacer",
         format_family="missing", required_groups=[["pregunt", "pedir", "necesit", "falta"], ["criterio", "filtro"]], max_words=48, no_list=True),
    _fmt("sf14", "te pido crear un recordatorio pero no te di fecha ni hora, como seguis",
         format_family="missing", required_groups=[["pregunt", "pedir", "necesit", "falta"], ["fecha", "hora", "cuando"]], max_words=52, no_list=True),
    _fmt("sf15", "quiero comparar dos cosas pero solo te pase una, que necesitas",
         format_family="missing", required_groups=[["pregunt", "pedir", "necesit", "falta"], ["segunda", "otro", "compar"]], max_words=52, no_list=True),

    _fmt("sf16", "en menos de 8 palabras: para que sirve un backup",
         format_family="short", required_groups=[["backup", "copia"], ["recuper", "dato"]], max_words=7, no_list=True),
    _fmt("sf17", "menos de 8 palabras: para que sirve una ruta",
         format_family="short", required_groups=[["ruta"], ["ubic", "archivo", "carpeta"]], max_words=7, no_list=True),
    _fmt("sf18", "en menos de 8 palabras: que hace un monitor",
         format_family="short", required_groups=[["monitor"], ["muestra", "visual", "informacion"]], max_words=7, no_list=True),
    _fmt("sf19", "menos de 8 palabras: para que sirve un permiso",
         format_family="short", required_groups=[["permiso"], ["autor", "acceso", "accion"]], max_words=7, no_list=True),
    _fmt("sf20", "en menos de 8 palabras: para que sirve un log",
         format_family="short", required_groups=[["log", "registro"], ["evento", "ocurr"]], max_words=7, no_list=True),



    _keep("retention_conversation", "sc01", "conversation", "holaa todo bien",
          retention_family="greeting", validator="greeting_or_state", intent_route=True, direct=True, no_list=True, max_words=20),
    _keep("retention_conversation", "sc02", "conversation", "ey butterfly buenas",
          retention_family="greeting", validator="greeting", intent_route=True, direct=True, no_list=True, max_words=20),
    _keep("retention_conversation", "sc03", "conversation", "buenas butterfly todo tranqui",
          retention_family="greeting", validator="greeting_or_state", intent_route=True, direct=True, no_list=True, max_words=20),
    _keep("retention_conversation", "sc04", "conversation", "ey buenas como va",
          retention_family="greeting", validator="greeting_or_state", intent_route=True, direct=True, no_list=True, max_words=20),

    _keep("retention_conversation", "sc05", "conversation", "gracias che me sirvio",
          retention_family="thanks", validator="thanks", intent_route=True, direct=True, no_list=True, max_words=20),
    _keep("retention_conversation", "sc06", "conversation", "joya muchas gracias",
          retention_family="thanks", validator="thanks", intent_route=True, direct=True, no_list=True, max_words=20),
    _keep("retention_conversation", "sc07", "conversation", "graciass posta me ayudo",
          retention_family="thanks", validator="thanks", intent_route=True, direct=True, no_list=True, max_words=20),
    _keep("retention_conversation", "sc08", "conversation", "mil gracias me re sirvio",
          retention_family="thanks", validator="thanks", intent_route=True, direct=True, no_list=True, max_words=20),

    _keep("retention_conversation", "sc09", "conversation", "como era tu nombre",
          retention_family="identity", validator="identity", intent_route=True, direct=True, no_list=True, max_words=24),
    _keep("retention_conversation", "sc10", "conversation", "decime quien sos",
          retention_family="identity", validator="identity", intent_route=True, direct=True, no_list=True, max_words=24),
    _keep("retention_conversation", "sc11", "conversation", "vos como te llamabas",
          retention_family="identity", validator="identity", intent_route=True, direct=True, no_list=True, max_words=24),
    _keep("retention_conversation", "sc12", "conversation", "quien sos butterfly",
          retention_family="identity", validator="identity", intent_route=True, direct=True, no_list=True, max_words=24),

    _keep("retention_conversation", "sc13", "conversation", "como andas por ahi",
          retention_family="state", validator="state", intent_route=True, direct=True, no_list=True, max_words=28),
    _keep("retention_conversation", "sc14", "conversation", "todo tranquilo de tu lado",
          retention_family="state", validator="state", intent_route=True, direct=True, no_list=True, max_words=28),
    _keep("retention_conversation", "sc15", "conversation", "como venis por ahi hoy",
          retention_family="state", validator="state", intent_route=True, direct=True, no_list=True, max_words=28),
    _keep("retention_conversation", "sc16", "conversation", "todo bien de tu lado hoy",
          retention_family="state", validator="state", intent_route=True, direct=True, no_list=True, max_words=28),



    _keep("retention_comprehension", "sd01", "comprehension", "explicame facil que guarda un archivo",
          retention_family="file", validator="definition_file", intent_route=True, direct=True, no_list=True, max_sentences=3, max_words=55),
    _keep("retention_comprehension", "sd02", "comprehension", "archivo en windows que es exactamente",
          retention_family="file", validator="definition_file", intent_route=True, direct=True, no_list=True, max_sentences=3, max_words=55),
    _keep("retention_comprehension", "sd03", "comprehension", "un archivo de pc que vendria a ser",
          retention_family="file", validator="definition_file", intent_route=True, direct=True, no_list=True, max_sentences=3, max_words=55),

    _keep("retention_comprehension", "sd04", "comprehension", "para que usas una carpeta en pc",
          retention_family="folder", validator="definition_folder", intent_route=True, direct=True, no_list=True, max_sentences=3, max_words=55),
    _keep("retention_comprehension", "sd05", "comprehension", "directorio o carpeta que hace",
          retention_family="folder", validator="definition_folder", intent_route=True, direct=True, no_list=True, max_sentences=3, max_words=55),
    _keep("retention_comprehension", "sd06", "comprehension", "carpeta en windows que vendria a ser",
          retention_family="folder", validator="definition_folder", intent_route=True, direct=True, no_list=True, max_sentences=3, max_words=55),
    _keep("retention_comprehension", "sd07", "comprehension", "una carpeta del sistema para q sirve",
          retention_family="folder", validator="definition_folder", intent_route=True, direct=True, no_list=True, max_sentences=3, max_words=55),

    _keep("retention_comprehension", "sd08", "comprehension", "api explicada simple para que sirve",
          retention_family="api", validator="definition_api", intent_route=True, direct=True, no_list=True, max_sentences=3, max_words=65),
    _keep("retention_comprehension", "sd09", "comprehension", "que permite hacer una api entre programas",
          retention_family="api", validator="definition_api", intent_route=True, direct=True, no_list=True, max_sentences=3, max_words=65),
    _keep("retention_comprehension", "sd10", "comprehension", "api entre sistemas que hace",
          retention_family="api", validator="definition_api", intent_route=True, direct=True, no_list=True, max_sentences=3, max_words=65),

    _keep("retention_comprehension", "sd11", "comprehension", "parametro de red neuronal explicado facil",
          retention_family="parameter", validator="definition_parameter", intent_route=True, direct=True, no_list=True, max_sentences=3, max_words=75),
    _keep("retention_comprehension", "sd12", "comprehension", "en una red neuronal un parametro que es",
          retention_family="parameter", validator="definition_parameter", intent_route=True, direct=True, no_list=True, max_sentences=3, max_words=75),

    _keep("retention_comprehension", "sd13", "comprehension", "token en ia explicado corto",
          retention_family="token", validator="definition_token", intent_route=True, direct=True, no_list=True, max_sentences=3, max_words=65),
    _keep("retention_comprehension", "sd14", "comprehension", "token en un modelo que vendria a ser",
          retention_family="token", validator="definition_token", intent_route=True, direct=True, no_list=True, max_sentences=3, max_words=65),


    _keep("retention_epistemic", "se01", "epistemic_dialogue", "si una pagina cualquiera dice algo ya lo das por verdadero",
          required_groups=[["no"], ["fuente", "verific", "evidencia", "contrastar"]], max_words=55, no_list=True),
    _keep("retention_epistemic", "se02", "epistemic_dialogue", "si no sabes de donde sale un dato lo afirmas igual",
          required_groups=[["no"], ["fuente", "verific", "evidencia", "dato"]], max_words=55, no_list=True),
    _keep("retention_epistemic", "se03", "epistemic_dialogue", "invente noravia y nunca te dije su moneda cual usa",
          validator="unknown", forbidden=["noravia usa", "la moneda de noravia es"], max_words=55, no_list=True),
    _keep("retention_epistemic", "se04", "epistemic_dialogue", "talmeria es inventado y no dije su capital cual es",
          validator="unknown", forbidden=["capital de talmeria es", "talmeria tiene como capital"], max_words=55, no_list=True),
    _keep("retention_epistemic", "se05", "epistemic_dialogue", "dos documentos confiables se contradicen que haces",
          required_groups=[["compar", "contrastar", "verific", "revis"], ["fuente", "evidencia", "fecha", "contexto"]], max_words=65, no_list=True),
    _keep("retention_epistemic", "se06", "epistemic_dialogue", "si falta evidencia para una afirmacion concreta que haces",
          required_groups=[["no", "falta", "insuficiente"], ["evidencia", "verific", "fuente", "dato"]], max_words=60, no_list=True),
]


def study_surface_prompts() -> set[str]:
    return {case["prompt"] for case in STUDY_CASES}


def _extract(output: str, shaped: str) -> str:
    answer = output[len(shaped):]
    for marker in ("<END>", "\nUser:"):
        if marker in answer:
            answer = answer.split(marker, 1)[0]
    return answer.strip()


def _avg(rows, key="score"):
    return sum(float(row[key]) for row in rows) / max(1, len(rows))


def study_microbenchmark(
    model,
    tokenizer,
    max_new_tokens: int = BENCHMARK_MAX_NEW_TOKENS,
    stop_requested=None,
):
    rows = []
    for case in STUDY_CASES:
        if stop_requested is not None and stop_requested():
            raise StopIteration("STOP_AUTONOMY requested during study exam")
        shaped = f"User: {case['prompt']}\nButterfly:"
        out = generate(
            model,
            shaped,
            tokenizer,
            max_new_tokens=max_new_tokens,
            **BENCHMARK_GENERATION_CONFIG,
        )
        row = _case_result(_extract(out, shaped), case)
        row["group"] = case["group"]
        if case.get("format_family"):
            row["format_family"] = case["format_family"]
        if case.get("retention_family"):
            row["retention_family"] = case["retention_family"]
        rows.append(row)
        if stop_requested is not None and stop_requested():
            raise StopIteration("STOP_AUTONOMY requested during study exam")

    binding = [r for r in rows if r["group"] == "binding"]
    fmt = [r for r in rows if r["group"] == "instruction_format"]
    conv = [r for r in rows if r["group"] == "retention_conversation"]
    comp = [r for r in rows if r["group"] == "retention_comprehension"]
    epi = [r for r in rows if r["group"] == "retention_epistemic"]
    intent = [r for r in conv + comp if r.get("intent_route")]
    retained = conv + comp + epi

    def family_score(source, field, value, key="score"):
        subset = [r for r in source if r.get(field) == value]
        return _avg(subset, key)

    binding_score = _avg(binding, "semantic")
    format_score = _avg(fmt, "score")
    conv_score = _avg(conv, "score")
    comp_score = _avg(comp, "score")
    epi_score = _avg(epi, "score")
    intent_score = _avg(intent, "semantic")
    quality = sum(
        (float(r["language"]) + float(r["style"]) + float(r["cleanliness"])) / 3.0
        for r in retained
    ) / max(1, len(retained))
    retention = sum([conv_score, comp_score, epi_score, intent_score, quality]) / 5.0

    sentence = family_score(fmt, "format_family", "sentence")
    two_steps = family_score(fmt, "format_family", "two_steps")
    missing = family_score(fmt, "format_family", "missing")
    short = family_score(fmt, "format_family", "short")

    greeting = family_score(conv, "retention_family", "greeting")
    thanks = family_score(conv, "retention_family", "thanks")
    identity = family_score(conv, "retention_family", "identity")
    state = family_score(conv, "retention_family", "state")

    file_score = family_score(comp, "retention_family", "file")
    folder_score = family_score(comp, "retention_family", "folder")
    api_score = family_score(comp, "retention_family", "api")
    parameter_score = family_score(comp, "retention_family", "parameter")
    token_score = family_score(comp, "retention_family", "token")

    return {
        "study_score": 0.45 * format_score + 0.35 * retention + 0.20 * binding_score,
        "binding_exact_component": binding_score,
        "instruction_format_component": format_score,
        "instruction_sentence_component": sentence,
        "instruction_two_steps_component": two_steps,
        "instruction_missing_component": missing,
        "instruction_short_component": short,
        "instruction_weakest_family_component": min(sentence, two_steps, missing, short),
        "retention_component": retention,
        "retention_conversation_component": conv_score,
        "retention_greeting_component": greeting,
        "retention_thanks_component": thanks,
        "retention_identity_component": identity,
        "retention_state_component": state,
        "retention_comprehension_component": comp_score,
        "retention_file_component": file_score,
        "retention_folder_component": folder_score,
        "retention_api_component": api_score,
        "retention_parameter_component": parameter_score,
        "retention_token_component": token_score,
        "retention_epistemic_component": epi_score,
        "retention_intent_component": intent_score,
        "retention_quality_component": quality,
        "cases": rows,
    }

def study_suite_id() -> str:
    import hashlib
    import inspect
    import json
    payload = {
        "cases": STUDY_CASES,
        "generation": BENCHMARK_GENERATION_CONFIG,
        "max_new_tokens": BENCHMARK_MAX_NEW_TOKENS,
        "study_microbenchmark": inspect.getsource(study_microbenchmark),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "study-" + hashlib.sha256(raw).hexdigest()[:12]
