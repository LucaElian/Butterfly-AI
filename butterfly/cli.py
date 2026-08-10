from __future__ import annotations
import argparse,json
from pathlib import Path
from .config import ensure_dirs,APP_VERSION,LANG_TRAIN,CONV_TRAIN,INST_TRAIN,V4_TOKENIZER_PATH,CORPUS_DIR
from .memory import MemoryStore
from .trainer import train_curriculum,best_device
from .checkpoint import load_active
from .generation import generate
from .tokenizer import ButterflyTokenizer
from .epistemic.engine import EpistemicEngine
from .agent.preflight import PreflightEvaluator
from .learning.sleep_cycle import run_sleep_cycle
from .learning.evaluator import behavior_benchmark,print_benchmark


def command_init(_args): ensure_dirs();MemoryStore();print(f"ButterflyAI v{APP_VERSION} initialized.")
def command_prepare(_args):
    from .upgrade import prepare_v0004;prepare_v0004()
def command_corpus(args):
    from .corpus.builder import build_all;build_all(target_wiki_mb=args.wiki_mb,conversation_mb=args.conversation_mb)
def command_tokenizer(args):
    files=[LANG_TRAIN,CONV_TRAIN,INST_TRAIN]
    tok=ButterflyTokenizer.train_bpe(files,vocab_size=args.vocab,min_frequency=2);tok.save(V4_TOKENIZER_PATH)
    total_bytes=sum(p.stat().st_size for p in files if p.exists()); total_tokens=sum(len(tok.encode(p.read_text(encoding="utf-8",errors="ignore"))) for p in files if p.exists())
    print(f"Tokenizer saved: {V4_TOKENIZER_PATH}\nVocab: {tok.vocab_size:,}\nTraining bytes: {total_bytes:,} -> tokens: {total_tokens:,} ({total_tokens/max(1,total_bytes):.2%})")
def command_train(args): train_curriculum(version="0.0004",preset=args.preset)
def command_compare(_args):
    from .upgrade import compare_and_promote;compare_and_promote("0.0004")
def command_chat(args):
    model,_,tokenizer=load_active(device=best_device());print("ButterflyAI local chat. Ctrl+C to exit.");history=[]
    while True:
        try: prompt=input("You > ").strip()
        except (EOFError,KeyboardInterrupt): print();return
        if not prompt: continue
        context="".join(f"User: {u}\nButterfly: {a}\n<END>\n" for u,a in history[-4:]);shaped=context+f"User: {prompt}\nButterfly:"
        output=generate(model,shaped,tokenizer,max_new_tokens=args.max_tokens,temperature=args.temperature,repetition_penalty=args.repetition_penalty);answer=output[len(shaped):]
        for marker in ("<END>","\nUser:"):
            if marker in answer:answer=answer.split(marker,1)[0]
        answer=answer.strip();print("Butterfly >",answer);history.append((prompt,answer))
def command_evaluate(_args):
    model,_,tok=load_active(device=best_device());m=behavior_benchmark(model,tok);print_benchmark(m)
def command_verify(args):
    r=EpistemicEngine().verify(args.claim,allow_web=args.web);print(json.dumps({"claim":r.claim,"status":r.status.value,"confidence":r.confidence,"method":r.method,"explanation":r.explanation,"evidence":[e.__dict__ for e in r.evidence]},indent=2,ensure_ascii=False))
def command_preflight(args): print(json.dumps(PreflightEvaluator().evaluate(args.task).to_dict(),indent=2,ensure_ascii=False))
def command_experience(args): MemoryStore().add_experience(task=args.task,result=args.result,lesson=args.lesson,context=args.context,verified=args.verified,quality=args.quality);print("Experience stored.")
def command_sleep(args): run_sleep_cycle(steps=args.steps)
def command_status(_args):
    from .registry import load_registry,get_active_entry,load_history
    from .storage import storage_status
    reg=load_registry();print(json.dumps({"app_version":APP_VERSION,"active":reg.get("active"),"models":reg.get("versions",[]),"history":load_history().get("versions",[])[-10:]},indent=2,ensure_ascii=False))

def command_export_release(_args):
    from .storage import export_active_release
    export_active_release()

def command_storage(_args):
    from .storage import storage_status
    print(json.dumps(storage_status(),indent=2,ensure_ascii=False))


def build_parser():
    p=argparse.ArgumentParser(prog="butterfly");sub=p.add_subparsers(dest="command",required=True)
    sp=sub.add_parser("init");sp.set_defaults(func=command_init)
    sp=sub.add_parser("prepare-v0004");sp.set_defaults(func=command_prepare)
    sp=sub.add_parser("build-corpus");sp.add_argument("--wiki-mb",type=float,default=20.0);sp.add_argument("--conversation-mb",type=float,default=2.0);sp.set_defaults(func=command_corpus)
    sp=sub.add_parser("train-tokenizer");sp.add_argument("--vocab",type=int,default=8192);sp.set_defaults(func=command_tokenizer)
    sp=sub.add_parser("train-v0004");sp.add_argument("--preset",choices=["auto","ryzen3600","light"],default="auto");sp.set_defaults(func=command_train)
    sp=sub.add_parser("compare-promote");sp.set_defaults(func=command_compare)
    sp=sub.add_parser("chat");sp.add_argument("--max-tokens",type=int,default=150);sp.add_argument("--temperature",type=float,default=.58);sp.add_argument("--repetition-penalty",type=float,default=1.24);sp.set_defaults(func=command_chat)
    sp=sub.add_parser("evaluate");sp.set_defaults(func=command_evaluate)
    sp=sub.add_parser("status");sp.set_defaults(func=command_status)
    sp=sub.add_parser("storage");sp.set_defaults(func=command_storage)
    sp=sub.add_parser("export-release");sp.set_defaults(func=command_export_release)
    sp=sub.add_parser("verify");sp.add_argument("claim");sp.add_argument("--web",action="store_true");sp.set_defaults(func=command_verify)
    sp=sub.add_parser("preflight");sp.add_argument("task");sp.set_defaults(func=command_preflight)
    sp=sub.add_parser("experience");sp.add_argument("--task",required=True);sp.add_argument("--result",required=True);sp.add_argument("--lesson",default="");sp.add_argument("--context",default="");sp.add_argument("--verified",action="store_true");sp.add_argument("--quality",type=float,default=.8);sp.set_defaults(func=command_experience)
    sp=sub.add_parser("sleep");sp.add_argument("--steps",type=int,default=120);sp.set_defaults(func=command_sleep)
    return p

def main():
    args=build_parser().parse_args();args.func(args)
