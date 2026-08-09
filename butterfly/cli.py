from __future__ import annotations
import argparse
import json
from pathlib import Path
from .config import ensure_dirs, DATA_DIR, TOKENIZER_PATH, config_for_preset
from .memory import MemoryStore
from .trainer import train_new, best_device
from .checkpoint import load_active
from .generation import generate
from .tokenizer import ButterflyTokenizer
from .epistemic.engine import EpistemicEngine
from .agent.preflight import PreflightEvaluator
from .learning.sleep_cycle import run_sleep_cycle
from .learning.evaluator import composite_score


def command_init(_args): ensure_dirs();MemoryStore();print("ButterflyAI v0.0003 initialized.")
def command_migrate(args):
    from .migration import migrate_and_optionally_burn
    migrate_and_optionally_burn(args.previous,burn=args.burn)
def command_build_data(args):
    from .distillation.build_dataset import build_distilled_dataset
    build_distilled_dataset(limit=args.examples,batch_size=args.batch)
def command_tokenizer(args):
    path=Path(args.data); text=path.read_text(encoding="utf-8"); tok=ButterflyTokenizer.train(text,target_vocab=args.vocab);tok.save(TOKENIZER_PATH)
    before=len(text.encode("utf-8"));after=len(tok.encode(text));print(f"Tokenizer saved: {TOKENIZER_PATH}");print(f"Vocab: {tok.vocab_size:,}");print(f"UTF-8 bytes: {before:,} -> tokens: {after:,} ({after/max(1,before):.2%})")
def command_train(args):
    tok=ButterflyTokenizer.load(TOKENIZER_PATH); cfg,preset=config_for_preset(args.preset,tok.vocab_size);print(f"Model preset: {preset}");train_new(Path(args.data),steps=args.steps,version=args.version,cfg=cfg)
def command_chat(args):
    model,_,tokenizer=load_active(device=best_device());print("ButterflyAI local chat. Ctrl+C to exit.");history=[]
    while True:
        try: prompt=input("You > ").strip()
        except (EOFError,KeyboardInterrupt): print();return
        if not prompt:continue
        context="".join(f"User: {u}\nButterfly: {a}\n<END>\n" for u,a in history[-4:]);shaped=context+f"User: {prompt}\nButterfly:"
        output=generate(model,shaped,tokenizer,max_new_tokens=args.max_tokens,temperature=args.temperature,repetition_penalty=args.repetition_penalty);answer=output[len(shaped):]
        for marker in ("<END>","\nUser:"):
            if marker in answer:answer=answer.split(marker,1)[0]
        answer=answer.strip();print("Butterfly >",answer);history.append((prompt,answer))
def command_evaluate(_args):
    model,_,tok=load_active(device=best_device());metrics=composite_score(model,DATA_DIR/"eval.txt",tok);print(json.dumps(metrics,indent=2,ensure_ascii=False));print("\nConversation samples:")
    for p in ["Hola","Buenas","Que estas diciendo?","Que haces si no sabes algo?","Explica que es un archivo."]:
        shaped=f"User: {p}\nButterfly:";out=generate(model,shaped,tok,max_new_tokens=80,temperature=.55,repetition_penalty=1.24);ans=out[len(shaped):].split("<END>",1)[0].split("\nUser:",1)[0].strip();print(f"You > {p}\nButterfly > {ans}\n")
def command_verify(args):
    r=EpistemicEngine().verify(args.claim,allow_web=args.web);print(json.dumps({"claim":r.claim,"status":r.status.value,"confidence":r.confidence,"method":r.method,"explanation":r.explanation,"evidence":[e.__dict__ for e in r.evidence]},indent=2,ensure_ascii=False))
def command_preflight(args): print(json.dumps(PreflightEvaluator().evaluate(args.task).to_dict(),indent=2,ensure_ascii=False))
def command_experience(args): MemoryStore().add_experience(task=args.task,result=args.result,lesson=args.lesson,context=args.context,verified=args.verified,quality=args.quality);print("Experience stored.")
def command_sleep(args): run_sleep_cycle(steps=args.steps)

def build_parser():
    p=argparse.ArgumentParser(prog="butterfly");sub=p.add_subparsers(dest="command",required=True)
    sp=sub.add_parser("init");sp.set_defaults(func=command_init)
    sp=sub.add_parser("migrate");sp.add_argument("--previous",required=True);sp.add_argument("--burn",action="store_true");sp.set_defaults(func=command_migrate)
    sp=sub.add_parser("build-data");sp.add_argument("--examples",type=int,default=600);sp.add_argument("--batch",type=int,default=4);sp.set_defaults(func=command_build_data)
    sp=sub.add_parser("tokenizer");sp.add_argument("--data",default=str(DATA_DIR/"consolidated.txt"));sp.add_argument("--vocab",type=int,default=4096);sp.set_defaults(func=command_tokenizer)
    sp=sub.add_parser("train");sp.add_argument("--data",default=str(DATA_DIR/"consolidated.txt"));sp.add_argument("--steps",type=int,default=1400);sp.add_argument("--version",default="0.0003");sp.add_argument("--preset",choices=["auto","ryzen3600","light"],default="auto");sp.set_defaults(func=command_train)
    sp=sub.add_parser("chat");sp.add_argument("--max-tokens",type=int,default=150);sp.add_argument("--temperature",type=float,default=.64);sp.add_argument("--repetition-penalty",type=float,default=1.22);sp.set_defaults(func=command_chat)
    sp=sub.add_parser("evaluate");sp.set_defaults(func=command_evaluate)
    sp=sub.add_parser("verify");sp.add_argument("claim");sp.add_argument("--web",action="store_true");sp.set_defaults(func=command_verify)
    sp=sub.add_parser("preflight");sp.add_argument("task");sp.set_defaults(func=command_preflight)
    sp=sub.add_parser("experience");sp.add_argument("--task",required=True);sp.add_argument("--result",required=True);sp.add_argument("--lesson",default="");sp.add_argument("--context",default="");sp.add_argument("--verified",action="store_true");sp.add_argument("--quality",type=float,default=.8);sp.set_defaults(func=command_experience)
    sp=sub.add_parser("sleep");sp.add_argument("--steps",type=int,default=120);sp.set_defaults(func=command_sleep)
    return p

def main():
    args=build_parser().parse_args();args.func(args)
