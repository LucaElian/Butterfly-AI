from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .checkpoint import load_checkpoint, load_entry, metadata_path, save_stable_model
from .config import MODELS_DIR, ROOT
from .corpus.deliberate import STAGE_FILES, load_manifest
from .registry import get_active_entry, get_entry, register_model
from .trainer import best_device, configure_cpu

TRAINING_ROOT = ROOT / "training_state" / "deliberate"
PROGRESS_PATH = TRAINING_ROOT / "progress.json"
RESUME_MODEL = TRAINING_ROOT / "resume.safetensors"
BEST_STAGE_MODEL = TRAINING_ROOT / "best-stage.safetensors"
AUTOSAVE_SECONDS = 600


class AssistantOnlyDataset(Dataset):
    """One conversation example per item with loss only on Butterfly's answer.

    Critical boundary rule: the context `...\nButterfly:` is tokenized separately,
    exactly like inference. The leading answer space is tokenized as answer output,
    so no tokenizer piece can silently cross the context/answer boundary.
    """

    def __init__(self, rows: list[dict], tokenizer, seq_len: int = 160):
        self.rows = rows
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.target_token_count = 0
        for row in rows:
            self.target_token_count += len(tokenizer.encode(" " + row["assistant"] + "\n<END>\n", add_eos=True))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        prefix = f"User: {row['user']}\nButterfly:"
        prefix_ids = self.tokenizer.encode(prefix, add_bos=True)
        answer_ids = self.tokenizer.encode(" " + row["assistant"] + "\n<END>\n", add_eos=True)
        ids = prefix_ids + answer_ids
        if len(ids) < 2:
            raise RuntimeError("Encoded deliberate example is empty.")
        if len(ids) - 1 > self.seq_len:
            # Keep the full answer and the most recent context tokens.
            max_ids = self.seq_len + 1
            answer_keep = min(len(answer_ids), max_ids - 1)
            prefix_keep = max_ids - answer_keep
            prefix_ids = prefix_ids[-prefix_keep:]
            answer_ids = answer_ids[-answer_keep:]
            ids = prefix_ids + answer_ids

        x = torch.full((self.seq_len,), int(self.tokenizer.PAD), dtype=torch.long)
        y = torch.full((self.seq_len,), -100, dtype=torch.long)
        raw_x = ids[:-1]
        raw_y = ids[1:]
        n = min(self.seq_len, len(raw_x))
        x[:n] = torch.tensor(raw_x[:n], dtype=torch.long)
        y[:n] = torch.tensor(raw_y[:n], dtype=torch.long)
        # y[i] predicts ids[i+1]. The first answer token is predicted at
        # i=len(prefix_ids)-1, so every earlier target must be ignored.
        mask_before = max(0, min(n, len(prefix_ids) - 1))
        y[:mask_before] = -100
        return x, y


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_weights(path: Path, model, extra: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.stem + ".tmp.safetensors")
    tmp_meta = metadata_path(tmp)
    tmp.unlink(missing_ok=True)
    tmp_meta.unlink(missing_ok=True)
    save_stable_model(tmp, model, extra=extra)
    os.replace(tmp, path)
    os.replace(tmp_meta, metadata_path(path))


def _save_resume(model, progress: dict, message: str | None = None):
    _atomic_weights(RESUME_MODEL, model, {"deliberate_resume": progress, "created_at": time.time()})
    _atomic_json(PROGRESS_PATH, progress)
    if message:
        print(f"  -> autosave: {message} | {RESUME_MODEL.stat().st_size / (1024*1024):.1f} MB weights-only")


def _save_best(model, progress: dict):
    _atomic_weights(BEST_STAGE_MODEL, model, {"deliberate_stage_best": progress, "created_at": time.time()})


def _read_progress():
    if not PROGRESS_PATH.exists() or not RESUME_MODEL.exists():
        return None
    try:
        return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _clean_best():
    BEST_STAGE_MODEL.unlink(missing_ok=True)
    metadata_path(BEST_STAGE_MODEL).unlink(missing_ok=True)


def _clean_training_state():
    if TRAINING_ROOT.exists():
        shutil.rmtree(TRAINING_ROOT, ignore_errors=True)


def _load_weights_into(model, path: Path, device: str):
    loaded, _ = load_checkpoint(path, device=device)
    model.load_state_dict(loaded.state_dict(), strict=True)
    del loaded


def _set_frozen_blocks(model, frozen_first_blocks: int):
    for p in model.parameters():
        p.requires_grad = True
    n = max(0, min(int(frozen_first_blocks), len(model.blocks)))
    for block in model.blocks[:n]:
        for p in block.parameters():
            p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total, n


@torch.no_grad()
def _validation_loss(model, ds: AssistantOnlyDataset, batch_size: int):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    device = next(model.parameters()).device
    vals = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        vals.append(float(loss.item()))
    return sum(vals) / max(1, len(vals))


def _recipe_hash(recipe: dict) -> str:
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode("utf-8")).hexdigest()


def _train_stage(model, tokenizer, target_version: str, stage_cfg: dict, recipe_hash: str, resume_progress, completed_stages):
    name = stage_cfg["name"]
    train_path, valid_path = STAGE_FILES[name]
    train_rows = _load_jsonl(train_path)
    valid_rows = _load_jsonl(valid_path)
    seq_len = int(stage_cfg.get("seq_len", 160))
    batch_size = int(stage_cfg.get("batch_size", 4))
    max_epochs = int(stage_cfg["max_epochs"])
    lr = float(stage_cfg["lr"])
    patience = int(stage_cfg.get("patience", 1))
    frozen = int(stage_cfg.get("frozen_first_blocks", 0))

    train_ds = AssistantOnlyDataset(train_rows, tokenizer, seq_len=seq_len)
    valid_ds = AssistantOnlyDataset(valid_rows, tokenizer, seq_len=seq_len)
    trainable, total_params, frozen = _set_frozen_blocks(model, frozen)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=float(stage_cfg.get("weight_decay", 0.06)))

    best_loss = float("inf")
    bad_epochs = 0
    start_epoch = 1
    resume_step = 0
    epochs_completed = 0
    completed_stages = list(completed_stages or [])

    if resume_progress and resume_progress.get("stage") == name and not resume_progress.get("stage_complete", False):
        start_epoch = max(1, int(resume_progress.get("epoch", 1)))
        resume_step = max(0, int(resume_progress.get("step", 0)))
        best_loss = float(resume_progress.get("best_validation_loss", float("inf")))
        bad_epochs = int(resume_progress.get("bad_epochs", 0))
        epochs_completed = max(0, start_epoch - 1)
        print(f"\nRESUME {name}: epoch {start_epoch}, after batch {resume_step:,}. Optimizer momentum restarts; weights do not.")
    else:
        _clean_best()

    print(f"\n=== STAGE {name.upper()} ===")
    print(f"train rows: {len(train_rows):,} | valid rows: {len(valid_rows):,}")
    print(f"assistant target tokens: {train_ds.target_token_count:,} train | {valid_ds.target_token_count:,} valid")
    print(f"batch: {batch_size} | max epochs: {max_epochs} | lr: {lr:g}")
    print(f"trainable parameters: {trainable:,}/{total_params:,} | frozen first blocks: {frozen}")
    print("Objective: USER tokens are context-only; loss is computed ONLY on Butterfly answer tokens.")
    print("Recovery: weights-only autosave about every 10 minutes + every epoch/stage.")

    device = next(model.parameters()).device
    stage_start = time.time()
    last_autosave = time.time()

    for epoch in range(start_epoch, max_epochs + 1):
        generator = torch.Generator()
        generator.manual_seed(530000 + sum(ord(c) for c in name) * 100 + epoch)
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
        model.train()
        running = []
        started = time.time()
        total = len(loader)
        skip_through = resume_step if epoch == start_epoch else 0
        if skip_through:
            print(f"Skipping first {skip_through:,}/{total:,} batches already represented in resumed weights.")

        for step, (x, y) in enumerate(loader, 1):
            if step <= skip_through:
                continue
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            running.append(float(loss.item()))

            if step == skip_through + 1 or step % max(1, total // 12) == 0 or step == total:
                elapsed = time.time() - started
                done = max(1, step - skip_through)
                rate = done / max(elapsed, 1e-6)
                eta = (total - step) / max(rate, 1e-6)
                avg = sum(running[-50:]) / max(1, len(running[-50:]))
                print(f"{name} epoch {epoch}/{max_epochs} | {step:>5}/{total} | answer-loss {avg:.4f} | ETA {eta/60:.1f}m")

            if time.time() - last_autosave >= AUTOSAVE_SECONDS:
                progress = {
                    "target_version": target_version,
                    "recipe_hash": recipe_hash,
                    "stage": name,
                    "stage_complete": False,
                    "epoch": epoch,
                    "step": step,
                    "total_steps": total,
                    "best_validation_loss": best_loss,
                    "bad_epochs": bad_epochs,
                    "completed_stages": completed_stages,
                    "updated_at": time.time(),
                }
                _save_resume(model, progress, f"{name} epoch {epoch} batch {step}/{total}")
                last_autosave = time.time()

        val = _validation_loss(model, valid_ds, batch_size=batch_size)
        train_avg = sum(running) / max(1, len(running)) if running else float("nan")
        print(f"{name} epoch {epoch}: answer-train={train_avg:.4f} answer-valid={val:.4f}")
        epochs_completed = epoch

        if val < best_loss - float(stage_cfg.get("min_delta", 1e-3)):
            best_loss = val
            bad_epochs = 0
            _save_best(model, {"target_version": target_version, "stage": name, "epoch": epoch, "validation_loss": best_loss})
            print("  -> new best checkpoint for this stage (saved to disk)")
        else:
            bad_epochs += 1
            print(f"  -> no validation improvement ({bad_epochs}/{patience})")

        progress = {
            "target_version": target_version,
            "recipe_hash": recipe_hash,
            "stage": name,
            "stage_complete": False,
            "epoch": epoch + 1,
            "step": 0,
            "total_steps": total,
            "best_validation_loss": best_loss,
            "bad_epochs": bad_epochs,
            "completed_stages": completed_stages,
            "updated_at": time.time(),
        }
        _save_resume(model, progress, f"{name} epoch {epoch} complete")
        last_autosave = time.time()
        resume_step = 0
        if bad_epochs >= patience:
            print("  -> early stopping")
            break

    if BEST_STAGE_MODEL.exists():
        _load_weights_into(model, BEST_STAGE_MODEL, device)
    result = {
        "stage": name,
        "best_validation_loss": best_loss,
        "seconds": time.time() - stage_start,
        "epochs_completed": epochs_completed,
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
    }
    completed_stages = completed_stages + [result]
    progress = {
        "target_version": target_version,
        "recipe_hash": recipe_hash,
        "stage": name,
        "stage_complete": True,
        "epoch": epochs_completed,
        "step": 0,
        "best_validation_loss": best_loss,
        "bad_epochs": bad_epochs,
        "completed_stages": completed_stages,
        "updated_at": time.time(),
    }
    _save_resume(model, progress, f"STAGE {name.upper()} COMPLETE")
    _clean_best()
    return result, completed_stages


def train_candidate(target_version: str, recipe: dict):
    random.seed(530053)
    torch.manual_seed(530053)

    manifest = load_manifest()
    if str(manifest.get("target_version")) != str(target_version):
        raise RuntimeError(f"Corpus target is v{manifest.get('target_version')}, expected v{target_version}.")
    if str(manifest.get("benchmark_suite")) != str(recipe.get("benchmark_suite")):
        raise RuntimeError("Corpus benchmark suite does not match deliberate recipe.")

    active = get_active_entry()
    if not active:
        raise RuntimeError("No active Butterfly brain.")
    expected_active = str(recipe.get("expected_active"))
    if active["version"] != expected_active:
        raise RuntimeError(f"Expected active seed v{expected_active}, but active is v{active['version']}. Nothing changed.")
    existing = get_entry(target_version)
    if existing and existing.get("status") == "candidate":
        raise RuntimeError(f"Candidate v{target_version} is already registered. Evaluate it or remove it deliberately before retraining.")

    device = best_device()
    threads = configure_cpu()
    recipe_hash = _recipe_hash(recipe)
    progress = _read_progress()
    if progress:
        if str(progress.get("target_version")) != str(target_version) or progress.get("recipe_hash") != recipe_hash:
            raise RuntimeError(
                "Existing deliberate recovery state belongs to another target/recipe. "
                "It was preserved for inspection and was NOT overwritten."
            )
        model, _ = load_checkpoint(RESUME_MODEL, device=device)
        _, _, tokenizer = load_entry(active, device="cpu")
        print("\nRECOVERY FOUND")
        print(f"Stage: {progress.get('stage')} | complete: {progress.get('stage_complete')} | epoch: {progress.get('epoch')} | step: {progress.get('step')}")
        print("Continuing from last atomic weights-only autosave. Adam momentum intentionally restarts.")
    else:
        model, _, tokenizer = load_entry(active, device=device)
        progress = None
        print(f"\nStarting v{target_version} FROM the accepted v{active['version']} weights.")
        print("This is continued learning, NOT a random-weight restart.")

    if model.cfg.vocab_size != tokenizer.vocab_size:
        raise RuntimeError("Model/tokenizer vocabulary mismatch.")
    tokenizer_rel = (active.get("metadata") or {}).get("tokenizer_path")
    if not tokenizer_rel:
        raise RuntimeError("Active brain has no tokenizer lineage metadata.")

    print(f"Seed brain: v{active['version']}")
    print(f"Device: {device}")
    print(f"CPU threads: {threads} (safe cap for Windows/USB headroom)" if device == "cpu" else "CPU threads: n/a")
    print(f"Parameters: {model.parameter_count():,} (same architecture as v{active['version']})")
    print(f"Tokenizer vocab: {tokenizer.vocab_size:,} (same accepted tokenizer)")
    print(f"v{active['version']} remains ACTIVE and untouched until v{target_version} passes benchmark v{recipe['benchmark_suite']}.")

    completed = list((progress or {}).get("completed_stages", []))
    completed_names = {x.get("stage") for x in completed}
    for stage_cfg in recipe["training_stages"]:
        name = stage_cfg["name"]
        if name in completed_names:
            print(f"\n=== STAGE {name.upper()} === already completed in recovery; skipping.")
            continue
        stage_resume = progress if progress and progress.get("stage") == name and not progress.get("stage_complete", False) else None
        _, completed = _train_stage(model, tokenizer, target_version, stage_cfg, recipe_hash, stage_resume, completed)
        completed_names.add(name)
        progress = _read_progress()

    candidate_path = MODELS_DIR / f"butterfly-v{target_version}-candidate.safetensors"
    extra = {
        "candidate": True,
        "version": target_version,
        "seed_brain": active["version"],
        "benchmark_suite": recipe["benchmark_suite"],
        "stages": completed,
        "tokenizer_path": tokenizer_rel,
        "created_at": time.time(),
        "recipe_hash": recipe_hash,
        "objective": "assistant-only-loss; boundary-safe; staged replay",
    }
    save_stable_model(candidate_path, model, extra=extra)
    register_model(
        candidate_path,
        target_version,
        active=False,
        status="candidate",
        metadata={
            "parameters": model.parameter_count(),
            "tokenizer_vocab": tokenizer.vocab_size,
            "tokenizer_path": tokenizer_rel,
            "seed_brain": active["version"],
            "benchmark_suite": recipe["benchmark_suite"],
            "stages": completed,
            "recipe_hash": recipe_hash,
        },
    )
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    training_report = {
        "target_version": target_version,
        "seed_brain": active["version"],
        "benchmark_suite": recipe["benchmark_suite"],
        "candidate_path": str(candidate_path),
        "stages": completed,
        "created_at": time.time(),
    }
    text = json.dumps(training_report, indent=2, ensure_ascii=False)
    (reports_dir / "latest-training.json").write_text(text, encoding="utf-8")
    (reports_dir / f"v{target_version}-training.json").write_text(text, encoding="utf-8")

    print(f"\nSaved candidate: {candidate_path}")
    print(f"It is NOT active. v{active['version']} is still the accepted brain.")
    print("Candidate saved successfully; temporary recovery checkpoints are now removed.")
    _clean_training_state()
    return candidate_path
