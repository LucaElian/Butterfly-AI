from pathlib import Path

from butterfly.alignment_data import AssistantOnlyDialogueDataset
from butterfly.corpus.alignment_v0005 import BENCHMARK_PROMPTS, _norm, build_alignment_corpus_v0005, STAGE_FILES
from butterfly.tokenizer import ByteTokenizer
from butterfly.model import ButterflyTransformer
from butterfly.config import ModelConfig


def test_assistant_only_loss_masks_user_prompt():
    tok = ByteTokenizer()
    rows = [{"user": "Hola", "assistant": "Buenas.", "category": "test"}]
    ds = AssistantOnlyDialogueDataset(rows, tok, seq_len=96)
    x, labels = ds[0]
    prefix = "User: Hola\nButterfly:"
    prefix_len = len(tok.encode(prefix, add_bos=True))
    assert all(int(v) == -100 for v in labels[: prefix_len - 1])
    assert any(int(v) != -100 for v in labels[prefix_len - 1 :])
    assert ds.answer_tokens > 0


def test_alignment_corpus_keeps_benchmark_exact_prompts_held_out(tmp_path, monkeypatch):
    # Build in the normal project test tree; generation is deterministic and small.
    manifest = build_alignment_corpus_v0005()
    assert manifest["exact_benchmark_prompt_leaks"] == 0
    for _, (train_path, valid_path) in STAGE_FILES.items():
        for path in (train_path, valid_path):
            text = Path(path).read_text(encoding="utf-8")
            # Exact prompt text should not appear as a JSON user field.
            import json
            for line in text.splitlines():
                row = json.loads(line)
                assert _norm(row["user"]) not in BENCHMARK_PROMPTS


def test_masked_targets_produce_finite_model_loss():
    tok = ByteTokenizer()
    ds = AssistantOnlyDialogueDataset(
        [{"user": "Decime hola", "assistant": "hola", "category": "test"}],
        tok,
        seq_len=64,
    )
    x, labels = ds[0]
    cfg = ModelConfig(vocab_size=tok.vocab_size, max_seq_len=64, d_model=32, n_layers=1, n_heads=4, d_ff=64)
    model = ButterflyTransformer(cfg)
    _, loss = model(x.unsqueeze(0), labels.unsqueeze(0))
    assert loss is not None
    assert loss.isfinite().item()
