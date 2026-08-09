from __future__ import annotations
import json
from pathlib import Path
import time
from .curriculum import build_curriculum
from .teacher import Teacher, TEACHER_MODEL
from ..config import DATA_DIR, INHERITED_DIR, ensure_dirs

IDENTITY_EXAMPLES = [
    ("Como te llamas?", "Me llamo ButterflyAI. Soy una inteligencia artificial local en desarrollo."),
    ("Quien sos?", "Soy ButterflyAI. Estoy aprendiendo a conversar, razonar, verificar informacion y usar herramientas de computadora."),
    ("Que haces si no sabes algo?", "No deberia inventarlo. Intento verificarlo con la herramienta o fuente adecuada y, si falta informacion esencial, lo digo o pido una aclaracion."),
    ("Podes equivocarte?", "Si. Por eso debo comprobar resultados, conservar evidencia y aprender de las correcciones."),
    ("Hola", "Hola! Como estas?"), ("Buenas", "Buenas! En que te puedo ayudar?"),
    ("Que estas diciendo?", "Creo que mi respuesta anterior no fue clara. Puedo explicarlo de nuevo de una forma mas simple."),
]


def _quality_ok(answer: str) -> bool:
    if not answer or len(answer) < 2 or len(answer) > 900: return False
    lower = answer.lower()
    bad_runs = ["calcalcal", "inininin", "qunqun", "bur.\nbur."]
    if any(x in lower for x in bad_runs): return False
    words = lower.split()
    if len(words) >= 12:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.30: return False
    return True


def _load_inherited_rows():
    rows = []
    if not INHERITED_DIR.exists(): return rows
    for path in INHERITED_DIR.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                row = json.loads(line)
                if row.get("user") and row.get("assistant") and _quality_ok(row["assistant"]): rows.append(row)
            except Exception: pass
    return rows


def build_distilled_dataset(limit=600, batch_size=4):
    ensure_dirs(); output_jsonl = DATA_DIR / "consolidated.jsonl"; output_text = DATA_DIR / "consolidated.txt"
    prompts = build_curriculum(limit=limit); teacher = Teacher(); rows=[]; chunks=[]; seen=set()

    def add(user, assistant, **meta):
        key=(user.strip().lower(), assistant.strip().lower())
        if key in seen or not _quality_ok(assistant): return
        seen.add(key); row={"user":user,"assistant":assistant,**meta}; rows.append(row)
        chunks.append(f"User: {user}\nButterfly: {assistant}\n<END>\n")

    for u,a in IDENTITY_EXAMPLES: add(u,a,source="butterfly_identity",verified_policy=True)
    for r in _load_inherited_rows(): add(r["user"],r["assistant"],source="inherited_verified_or_teacher")

    started=time.time()
    for start in range(0,len(prompts),batch_size):
        batch=prompts[start:start+batch_size]
        try: answers=teacher.answer_batch(batch)
        except Exception as e:
            print(f"Batch ERROR: {e}"); continue
        for prompt,answer in zip(batch,answers):
            add(prompt,answer,source=TEACHER_MODEL,teacher_only=True,factual_status="UNVERIFIED_UNLESS_DETERMINISTIC")
        done=min(start+len(batch),len(prompts)); print(f"[{done:>4}/{len(prompts)}] {time.time()-started:>7.0f}s | kept {len(rows)}")

    output_jsonl.write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in rows)+"\n",encoding="utf-8")
    output_text.write_text("\n".join(chunks),encoding="utf-8")
    print(f"Created {output_jsonl}\nCreated {output_text}\nExamples: {len(rows)}\nChars: {len(output_text.read_text(encoding='utf-8')):,}")
    return output_text
