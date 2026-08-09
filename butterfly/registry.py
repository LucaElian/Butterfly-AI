from pathlib import Path
import json
from .config import REGISTRY_PATH, MODELS_DIR, ensure_dirs


def load_registry():
    ensure_dirs()
    if not REGISTRY_PATH.exists(): return {"active":None,"versions":[]}
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

def save_registry(reg): REGISTRY_PATH.write_text(json.dumps(reg,indent=2,ensure_ascii=False),encoding="utf-8")

def register_model(path:Path,version:str,score=None,active=False,metadata=None):
    reg=load_registry(); entry={"version":version,"path":str(path.relative_to(MODELS_DIR)),"score":score,"metadata":metadata or {}}
    reg["versions"]=[v for v in reg["versions"] if v["version"]!=version];reg["versions"].append(entry)
    if active: reg["active"]=version
    save_registry(reg)

def get_active_entry():
    reg=load_registry(); active=reg.get("active")
    return next((v for v in reg["versions"] if v["version"]==active),None) if active else None

def promote(version:str):
    reg=load_registry()
    if not any(v["version"]==version for v in reg["versions"]): raise KeyError(version)
    reg["active"]=version;save_registry(reg)

def compact_to_active():
    """Butterfly's 'burn the old book' policy: after a candidate passed evaluation,
    keep only the active checkpoint. Training data, memory and inherited lessons stay.
    """
    reg=load_registry(); active=reg.get("active")
    if not active:return
    keep=[]
    for item in reg.get("versions",[]):
        p=MODELS_DIR/item["path"]
        if item["version"]==active: keep.append(item)
        else: p.unlink(missing_ok=True)
    reg["versions"]=keep; save_registry(reg)
