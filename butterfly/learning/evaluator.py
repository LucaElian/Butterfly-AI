from pathlib import Path
import torch
from ..data import NextTokenDataset, load_text
from ..epistemic.engine import EpistemicEngine
from ..config import TOKENIZER_PATH
from ..tokenizer import ButterflyTokenizer

@torch.no_grad()
def validation_loss(model, eval_path: Path, tokenizer=None, max_batches=50):
    model.eval(); device=next(model.parameters()).device; tokenizer=tokenizer or ButterflyTokenizer.load(TOKENIZER_PATH)
    ds=NextTokenDataset(load_text(eval_path),min(model.cfg.max_seq_len,128),tokenizer=tokenizer); losses=[]
    for i in range(min(len(ds),max_batches)):
        x,y=ds[i]; _,loss=model(x.unsqueeze(0).to(device),y.unsqueeze(0).to(device)); losses.append(float(loss.item()))
    return sum(losses)/max(1,len(losses))

def epistemic_regression_score():
    engine=EpistemicEngine(); tests=[("2 + 2 = 4","VERIFIED"),("2 + 2 = 5","CONTRADICTED"),("10 / 2 = 5","VERIFIED"),("3 * 7 = 20","CONTRADICTED")]
    return sum(engine.verify(c).status.value==e for c,e in tests)/len(tests)

def composite_score(model,eval_path:Path,tokenizer=None):
    loss=validation_loss(model,eval_path,tokenizer=tokenizer); language=1/(1+loss); epistemic=epistemic_regression_score(); score=.80*language+.20*epistemic
    return {"score":score,"validation_loss":loss,"language_component":language,"epistemic_component":epistemic}
