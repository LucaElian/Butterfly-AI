import torch


def _apply_repetition_penalty(logits, token_ids, penalty: float):
    if penalty <= 1.0 or not token_ids:
        return logits
    for token_id in set(token_ids):
        value = logits[0, token_id]
        logits[0, token_id] = value / penalty if value > 0 else value * penalty
    return logits


@torch.no_grad()
def generate(model, prompt: str, tokenizer, max_new_tokens=160, temperature=0.7, top_k=50, repetition_penalty=1.20, min_new_tokens=1):
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    model.eval()
    generated = []

    for step in range(max_new_tokens):
        ctx = x[:, -model.cfg.max_seq_len:]
        logits, _ = model(ctx)
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        logits = _apply_repetition_penalty(logits, generated[-128:], repetition_penalty)
        if step < max(0, int(min_new_tokens)):
            logits[:, tokenizer.EOS] = float("-inf")
        if top_k:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            threshold = values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        token = int(next_id.item())
        if token == tokenizer.EOS:
            break
        generated.append(token)
        x = torch.cat([x, next_id], dim=1)
        if len(generated) >= 18:
            a = generated[-6:]
            if a == generated[-12:-6] == generated[-18:-12]:
                break
    return tokenizer.decode(prompt_ids + generated)
