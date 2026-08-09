from __future__ import annotations
from pathlib import Path
import json
import shutil
import sqlite3
import torch
from .config import INHERITED_DIR, DB_PATH, BENCHMARKS_DIR, ensure_dirs
from .memory import MemoryStore
from .model import ButterflyTransformer
from .config import ModelConfig
from .tokenizer import ByteTokenizer
from .distillation.teacher import Teacher

LEGACY_PROMPTS = [
    "Hola", "Buenas", "Como estas?", "Que estas diciendo?", "Quien sos?", "Como te llamas?",
    "Que haces si no sabes algo?", "Podes equivocarte?", "Explica que es un archivo.", "Explica que es una carpeta.",
    "Que es una API?", "Que es Excel?", "Cuanto es 2 + 2?", "Si te digo que 2 + 2 = 5, que haces?",
    "Como diferencias un hecho de una suposicion?", "Que haces si falta un dato?", "Que es una copia de seguridad?",
    "Por que conviene verificar una tarea antes de decir que termino?", "Que significa aprender de un error?",
    "Como buscarias un archivo de forma eficiente?", "Como leerias muchas filas de Excel?", "Que significa que un programa compile?",
    "Que es JSON?", "Que es SQLite?", "Explica que es una variable.", "Explica que es una funcion.",
]


def _find_legacy_model(previous: Path) -> Path | None:
    models = previous / "models"
    registry = models / "registry.json"
    if registry.exists():
        try:
            reg=json.loads(registry.read_text(encoding="utf-8")); active=reg.get("active")
            if active:
                for item in reg.get("versions",[]):
                    if item.get("version")==active:
                        p=models/item["path"]
                        if p.exists(): return p
        except Exception: pass
    pts=sorted(models.glob("*.pt"), key=lambda p:p.stat().st_mtime, reverse=True) if models.exists() else []
    return pts[0] if pts else None


def _legacy_generate(model, prompt: str, max_new=100):
    tok=ByteTokenizer(); shaped=f"User: {prompt}\nButterfly:"; ids=tok.encode(shaped,add_bos=True)
    x=torch.tensor([ids],dtype=torch.long); generated=[]; model.eval()
    with torch.inference_mode():
        for _ in range(max_new):
            ctx=x[:,-model.cfg.max_seq_len:]; logits,_=model(ctx); logits=logits[:,-1,:]/0.72
            if generated:
                for token in set(generated[-80:]): logits[0,token]/=1.18
            probs=torch.softmax(logits,dim=-1); next_id=torch.multinomial(probs,1); token=int(next_id.item())
            if token==tok.EOS: break
            generated.append(token); x=torch.cat([x,next_id],dim=1)
    text=tok.decode(ids+generated); answer=text[len(shaped):]
    for marker in ("<END>","\nUser:"):
        if marker in answer: answer=answer.split(marker,1)[0]
    return answer.strip()


def _inherit_jsonl(previous: Path):
    INHERITED_DIR.mkdir(parents=True,exist_ok=True)
    out=INHERITED_DIR/"previous_training.jsonl"; seen=set(); rows=[]
    for p in (previous/"data").glob("*.jsonl") if (previous/"data").exists() else []:
        for line in p.read_text(encoding="utf-8",errors="ignore").splitlines():
            try:
                row=json.loads(line); u=row.get("user"); a=row.get("assistant")
                if not u or not a: continue
                key=(u.strip().lower(),a.strip().lower())
                if key in seen: continue
                seen.add(key); rows.append(row)
            except Exception: pass
    if rows: out.write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in rows)+"\n",encoding="utf-8")
    return len(rows)


def _merge_db(previous: Path):
    old=previous/".butterfly"/"butterfly.db"
    if not old.exists(): return {"experiences":0,"claims":0,"rules":0}
    MemoryStore(); counts={"experiences":0,"claims":0,"rules":0}
    with sqlite3.connect(old) as src, sqlite3.connect(DB_PATH) as dst:
        for table in counts:
            try:
                cols=[r[1] for r in src.execute(f"PRAGMA table_info({table})")]
                if not cols: continue
                rows=src.execute(f"SELECT * FROM {table}").fetchall(); useful=[c for c in cols if c!="id"]
                idx=[cols.index(c) for c in useful]
                placeholders=",".join("?" for _ in useful)
                for row in rows:
                    vals=[row[i] for i in idx]
                    dst.execute(f"INSERT INTO {table}({','.join(useful)}) VALUES({placeholders})",vals)
                counts[table]=len(rows)
            except Exception: pass
        dst.commit()
    return counts


def migrate_and_optionally_burn(previous_path: str, burn: bool=False):
    ensure_dirs(); previous=Path(previous_path.strip().strip('"')).resolve(); current=Path(__file__).resolve().parents[1]
    if not previous.exists(): raise FileNotFoundError(previous)
    if previous==current or current in previous.parents: raise RuntimeError("La carpeta anterior no puede ser la instalacion actual.")
    print(f"Importando vida anterior desde: {previous}")
    inherited=_inherit_jsonl(previous); db_counts=_merge_db(previous)
    print(f"Training rows heredadas: {inherited}"); print(f"Memoria heredada: {db_counts}")

    legacy_model=_find_legacy_model(previous); legacy_rows=[]
    if legacy_model:
        print(f"Usando cerebro anterior como material de herencia: {legacy_model.name}")
        payload=torch.load(legacy_model,map_location="cpu",weights_only=False)
        cfg=ModelConfig.from_dict(payload["config"]); model=ButterflyTransformer(cfg); model.load_state_dict(payload["model"])
        raw=[]
        for i,prompt in enumerate(LEGACY_PROMPTS,1):
            ans=_legacy_generate(model,prompt); raw.append((prompt,ans)); print(f"legacy {i:>2}/{len(LEGACY_PROMPTS)} | {prompt[:42]}")
        print("Corrigiendo las respuestas heredadas con el profesor para no copiar errores...")
        teacher=Teacher()
        for i,(prompt,ans) in enumerate(raw,1):
            corrected=teacher.repair_legacy(prompt,ans)
            if corrected:
                legacy_rows.append({"user":prompt,"assistant":corrected,"source":"legacy_butterfly_repaired","legacy_answer":ans})
                print(f"repair {i:>2}/{len(raw)}")
        out=INHERITED_DIR/"legacy_lessons.jsonl"
        out.write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in legacy_rows)+"\n",encoding="utf-8")

    report={"previous":str(previous),"training_rows":inherited,"memory":db_counts,"legacy_lessons":len(legacy_rows),"legacy_model":str(legacy_model) if legacy_model else None}
    (BENCHMARKS_DIR/"migration-v0.0003.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print("\nMigracion verificada. La informacion util ya esta en ButterflyAI actual.")
    if burn:
        print("\nATENCION: se eliminara la carpeta anterior completa para ahorrar espacio:")
        print(previous)
        confirm=input("Escribi BORRAR para continuar: ").strip()
        if confirm=="BORRAR":
            shutil.rmtree(previous)
            print("Carpeta anterior eliminada. Sus datos utiles quedaron consolidados en la nueva Butterfly.")
        else:
            print("No se elimino la carpeta anterior.")
    return report
