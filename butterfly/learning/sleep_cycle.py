from __future__ import annotations
from copy import deepcopy
import json
from ..memory import MemoryStore
from ..checkpoint import load_active, save_checkpoint
from ..config import MODELS_DIR, DATA_DIR
from ..registry import register_model, promote, get_active_entry, compact_to_active
from ..trainer import continue_training, best_device
from .evaluator import composite_score


def experiences_to_text(rows):
    chunks=[]; ids=[]
    for row in rows:
        id_,task,context,actions_json,result,lesson,quality=row; ids.append(id_); actions=json.loads(actions_json or "[]")
        chunks.append(f"Task: {task}\nContext: {context}\nActions: {actions}\nResult: {result}\nLesson: {lesson}\nQuality: {quality}\n")
    return "\n".join(chunks),ids

def next_version(active):
    if not active:return "0.0003"
    parts=active.split("."); return ".".join(parts[:-1]+[f"{int(parts[-1])+1:04d}"])

def run_sleep_cycle(steps=120):
    memory=MemoryStore(); rows=memory.approved_experiences()
    if not rows: print("No verified high-quality unused experiences. Nothing to learn tonight."); return False
    model,payload,tokenizer=load_active(device=best_device()); active=get_active_entry()["version"]; baseline=composite_score(model,DATA_DIR/"eval.txt",tokenizer)
    candidate=deepcopy(model); text,ids=experiences_to_text(rows); candidate,loss=continue_training(candidate,text,tokenizer,steps=steps)
    metrics=composite_score(candidate,DATA_DIR/"eval.txt",tokenizer); version=next_version(active); path=MODELS_DIR/f"butterfly-v{version}.pt"
    save_checkpoint(path,candidate,extra={"sleep_cycle":True,"source_experiences":ids,"baseline":baseline,"candidate":metrics,"train_loss":loss})
    promoted=metrics["score"]>baseline["score"]; register_model(path,version,score=metrics["score"],active=False,metadata={"baseline":baseline,"candidate":metrics,"promoted":promoted})
    print("Baseline:",baseline);print("Candidate:",metrics)
    if promoted: promote(version); compact_to_active(); memory.mark_used(ids); print(f"PROMOTED Butterfly v{version}; previous checkpoint burned after passing evaluation."); return True
    path.unlink(missing_ok=True); print(f"REJECTED Butterfly v{version}; candidate file deleted to save space.");return False
