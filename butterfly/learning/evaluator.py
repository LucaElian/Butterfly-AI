from __future__ import annotations
from pathlib import Path
import json, math, re, unicodedata
from ..generation import generate
from ..epistemic.engine import EpistemicEngine


def _norm(s):
    s=unicodedata.normalize("NFKD",s.lower())
    s="".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+"," ",s).strip()

def _repetition_score(text):
    words=re.findall(r"[a-záéíóúüñ0-9]+",text.lower())
    if len(words)<8: return 1.0
    grams=[tuple(words[i:i+3]) for i in range(len(words)-2)]
    if not grams: return 1.0
    unique=len(set(grams))/len(grams)
    # Also catch character/syllable loops such as calcalcal / ininin.
    bad_char=bool(re.search(r"(.{2,6})\1\1",text.lower()))
    return max(0.0,min(1.0,unique-(0.35 if bad_char else 0.0)))

def _language_quality(text):
    if not text.strip(): return 0.0
    if "�" in text: return 0.0
    words=re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+",text)
    if not words: return 0.15
    bizarre=0
    vowels=set("aeiouáéíóúüAEIOUÁÉÍÓÚÜ")
    for w in words:
        if len(w)>28: bizarre+=1
        elif len(w)>6 and not any(c in vowels for c in w): bizarre+=1
        elif re.search(r"(.{2,4})\1\1",w.lower()): bizarre+=1
    alpha=sum(ch.isalpha() or ch.isspace() for ch in text)/max(1,len(text))
    score=1.0-bizarre/max(1,len(words))
    if alpha<0.45: score-=0.25
    return max(0.0,min(1.0,score))*_repetition_score(text)

def _form_score(text, casual=False):
    n=len(text.strip())
    score=1.0
    if n<2: return 0.0
    if n>900: score-=0.25
    if casual and re.match(r"\s*(?:1[.)]|[-*]\s)",text): score-=0.45
    if text.count("*")>8: score-=0.2
    return max(0.0,score)

CASES=[
    ("conversation","Hola",["hola","buenas"],True),
    ("conversation","Buenas",["hola","buenas"],True),
    ("conversation","Gracias",["de nada","placer","cuando quieras"],True),
    ("conversation","¿Cómo te llamás?",["butterfly"],True),
    ("conversation","¿Qué estás diciendo?",["perdon","discul","acl","explic","mal"],True),
    ("conversation","¿Cómo estás?",["bien","lista","ayud","hola"],True),
    ("comprehension","Explica qué es un archivo.",["archivo","datos","informacion","guard"],False),
    ("comprehension","¿Qué es una carpeta?",["carpeta","archivo","organiz"],False),
    ("comprehension","¿Qué es una API?",["api","interfaz","comunic"],False),
    ("comprehension","¿Qué es un parámetro de una red neuronal?",["parametro","valor","ajust","entren"],False),
    ("instruction","Quiero organizar muchos documentos sin perder archivos.",["archivo","organiz","revis","verific","carpeta"],False),
    ("instruction","Tengo un proyecto que no compila. ¿Qué harías?",["compil","error","codigo","archivo"],False),
    ("instruction","Tengo muchos Excel y quiero buscar errores.",["excel","datos","hoja","verific","estructur"],False),
    ("epistemic_dialogue","¿Qué hacés si no sabés algo?",["no se","verific","pregunt","no invent","fuente"],False),
    ("epistemic_dialogue","Si te digo que 2 + 2 = 5, ¿me creés?",["4","no","incorrect"],False),
    ("epistemic_dialogue","¿Todo lo que aparece en Internet es verdad?",["no","fuente","verific","evidencia"],False),
]


def behavior_benchmark(model,tokenizer,max_new_tokens=80):
    rows=[]; cats={}
    for category,prompt,expected,casual in CASES:
        shaped=f"User: {prompt}\nButterfly:"
        out=generate(model,shaped,tokenizer,max_new_tokens=max_new_tokens,temperature=.42,top_k=40,repetition_penalty=1.28)
        ans=out[len(shaped):]
        for marker in ("<END>","\nUser:"):
            if marker in ans: ans=ans.split(marker,1)[0]
        ans=ans.strip(); n=_norm(ans); expected_hit=1.0 if any(_norm(x) in n for x in expected) else 0.0
        lang=_language_quality(ans); rep=_repetition_score(ans); form=_form_score(ans,casual=casual)
        case_score=.50*expected_hit+.25*lang+.15*rep+.10*form
        row={"category":category,"prompt":prompt,"answer":ans,"expected_hit":expected_hit,"language":lang,"repetition":rep,"form":form,"score":case_score}
        rows.append(row); cats.setdefault(category,[]).append(case_score)
    cat_scores={k:sum(v)/len(v) for k,v in cats.items()}
    language=sum(r["language"] for r in rows)/len(rows)
    repetition=sum(r["repetition"] for r in rows)/len(rows)
    engine=EpistemicEngine(); epi_tests=[("2 + 2 = 4","VERIFIED"),("2 + 2 = 5","CONTRADICTED"),("10 / 2 = 5","VERIFIED"),("3 * 7 = 20","CONTRADICTED"),("9 - 4 = 5","VERIFIED"),("6 * 6 = 35","CONTRADICTED")]
    epistemic=sum(engine.verify(c).status.value==e for c,e in epi_tests)/len(epi_tests)
    overall=(.30*language+.20*cat_scores.get("conversation",0)+.15*cat_scores.get("comprehension",0)+.15*cat_scores.get("instruction",0)+.10*cat_scores.get("epistemic_dialogue",0)+.05*repetition+.05*epistemic)
    return {"score":overall,"language_component":language,"repetition_component":repetition,"conversation_component":cat_scores.get("conversation",0),"comprehension_component":cat_scores.get("comprehension",0),"instruction_component":cat_scores.get("instruction",0),"epistemic_dialogue_component":cat_scores.get("epistemic_dialogue",0),"epistemic_engine_component":epistemic,"cases":rows}


def print_benchmark(metrics):
    keys=["score","language_component","conversation_component","comprehension_component","instruction_component","epistemic_dialogue_component","epistemic_engine_component","repetition_component"]
    for k in keys: print(f"{k:32s}: {metrics.get(k,0):.4f}")
    print("\nConversation samples:")
    for r in metrics["cases"][:8]: print(f"You > {r['prompt']}\nButterfly > {r['answer']}\n")
