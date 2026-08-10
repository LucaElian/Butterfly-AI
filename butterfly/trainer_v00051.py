from __future__ import annotations

from pathlib import Path
import json
import os
import random
import shutil
import time

import torch
from torch.utils.data import DataLoader

from .alignment_data import AssistantOnlyDialogueDataset, load_jsonl
from .checkpoint import load_checkpoint, load_entry, metadata_path, save_stable_model
from .config import MODELS_DIR, ROOT, ensure_dirs
from .corpus.alignment_v00051 import MANIFEST, STAGE_FILES
from .registry import get_active_entry, get_entry, register_model

VERSION = "0.00051"
TRAINING_ROOT = ROOT / "training_state" / "v0.00051"
PROGRESS_PATH = TRAINING_ROOT / "progress.json"
RESUME_MODEL = TRAINING_ROOT / "resume.safetensors"
BEST_STAGE_MODEL = TRAINING_ROOT / "best-stage.safetensors"
AUTOSAVE_SECONDS = 600


def best_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def configure_cpu():
    if torch.cuda.is_available():
        return None
    threads = max(1, min(8, os.cpu_count() or 6))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, min(4, threads // 2)))
    except RuntimeError:
        pass
    return threads


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
    _atomic_weights(RESUME_MODEL, model, {"training_resume": progress, "created_at": time.time()})
    _atomic_json(PROGRESS_PATH, progress)
    if message:
        print(f"  -> autosave: {message} | {RESUME_MODEL.stat().st_size/1024/1024:.1f} MB weights-only")


def _save_best(model, progress: dict):
    _atomic_weights(BEST_STAGE_MODEL, model, {"stage_best": progress, "created_at": time.time()})


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


def _set_trainable(model, freeze_first_blocks: int):
    for p in model.parameters():
        p.requires_grad = True
    for block in model.blocks[:freeze_first_blocks]:
        for p in block.parameters():
            p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


@torch.no_grad()
def _validation_loss(model, dataset, batch_size=4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    device = next(model.parameters()).device
    vals = []
    for x, labels in loader:
        _, loss = model(x.to(device), labels.to(device))
        vals.append(float(loss.item()))
    return sum(vals) / max(1, len(vals))


def _load_weights_into(model, path: Path, device):
    loaded, _ = load_checkpoint(path, device=device)
    model.load_state_dict(loaded.state_dict(), strict=True)
    del loaded


def _train_stage(
    model,
    tokenizer,
    name: str,
    train_path: Path,
    valid_path: Path,
    *,
    epochs: int,
    lr: float,
    freeze_first_blocks: int,
    patience: int,
    seq_len: int = 192,
    batch_size: int = 4,
    resume_progress: dict | None = None,
    completed_stages: list | None = None,
):
    train_rows = load_jsonl(train_path)
    valid_rows = load_jsonl(valid_path)
    train_ds = AssistantOnlyDialogueDataset(train_rows, tokenizer, seq_len=seq_len)
    valid_ds = AssistantOnlyDialogueDataset(valid_rows, tokenizer, seq_len=seq_len)
    device = next(model.parameters()).device

    trainable, total_params = _set_trainable(model, freeze_first_blocks)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.05,
    )

    completed_stages = list(completed_stages or [])
    best_loss = float("inf")
    bad_epochs = 0
    start_epoch = 1
    resume_step = 0
    epochs_completed = 0

    if resume_progress and resume_progress.get("stage") == name and not resume_progress.get("stage_complete", False):
        start_epoch = max(1, int(resume_progress.get("epoch", 1)))
        resume_step = max(0, int(resume_progress.get("step", 0)))
        best_loss = float(resume_progress.get("best_validation_loss", float("inf")))
        bad_epochs = int(resume_progress.get("bad_epochs", 0))
        epochs_completed = max(0, start_epoch - 1)
        print(f"\nRESUME {name}: epoch {start_epoch}, after batch {resume_step:,}. Adam momentum restarts; learned weights remain.")
        if BEST_STAGE_MODEL.exists() and best_loss < float("inf"):
            print(f"Preserved best validation for this stage: {best_loss:.4f}")
    else:
        _clean_best()

    if start_epoch > epochs:
        print(f"\n{name}: all {epochs} epoch(s) were already completed before interruption.")
        print("Restoring this stage's best validation weights; it will NOT be retrained.")

    print(f"\n=== STAGE {name.upper()} ===")
    print(f"train rows: {len(train_ds):,} | valid rows: {len(valid_ds):,}")
    print(f"assistant target tokens: {train_ds.answer_tokens:,} train | {valid_ds.answer_tokens:,} valid")
    print(f"batch: {batch_size} | max epochs: {epochs} | lr: {lr:g}")
    print(f"trainable parameters: {trainable:,}/{total_params:,} | frozen first blocks: {freeze_first_blocks}")
    if train_ds.truncated or valid_ds.truncated:
        print(f"truncated long rows: train={train_ds.truncated}, valid={valid_ds.truncated}")
    print("Objective: USER tokens are context-only; loss is computed ONLY on Butterfly answer tokens.")
    print("Recovery: weights-only autosave about every 10 minutes + every epoch/stage.")

    stage_start = time.time()
    last_autosave = time.time()

    for epoch in range(start_epoch, epochs + 1):
        generator = torch.Generator()
        generator.manual_seed(5100 + sum(ord(c) for c in name) * 100 + epoch)
        loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )
        model.train()
        running = []
        started = time.time()
        total = len(loader)
        skip_through = resume_step if epoch == start_epoch else 0

        if skip_through:
            print(f"Skipping {skip_through:,}/{total:,} batches already represented in resumed weights.")

        for step, (x, labels) in enumerate(loader, 1):
            if step <= skip_through:
                continue
            x, labels = x.to(device), labels.to(device)
            _, loss = model(x, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            running.append(float(loss.item()))

            if step == skip_through + 1 or step % max(1, total // 12) == 0 or step == total:
                elapsed = time.time() - started
                effective = max(1, step - skip_through)
                rate = effective / max(elapsed, 1e-6)
                eta = (total - step) / max(rate, 1e-6)
                avg = sum(running[-50:]) / max(1, len(running[-50:]))
                print(f"{name} epoch {epoch}/{epochs} | {step:>5}/{total} | answer-loss {avg:.4f} | ETA {eta/60:.1f}m")

            if time.time() - last_autosave >= AUTOSAVE_SECONDS:
                progress = {
                    "version": VERSION,
                    "seed_brain": "0.0004",
                    "stage": name,
                    "stage_complete": False,
                    "epoch": epoch,
                    "step": step,
                    "total_steps": total,
                    "best_validation_loss": best_loss,
                    "bad_epochs": bad_epochs,
                    "completed_stages": completed_stages,
                    "updated_at": time.time(),
                    "objective": "assistant-only-loss",
                }
                _save_resume(model, progress, f"{name} epoch {epoch} batch {step}/{total}")
                last_autosave = time.time()

        val = _validation_loss(model, valid_ds, batch_size=batch_size)
        train_avg = sum(running) / max(1, len(running)) if running else float("nan")
        print(f"{name} epoch {epoch}: answer-train={train_avg:.4f} answer-valid={val:.4f}")
        epochs_completed = epoch

        if val < best_loss - 1e-3:
            best_loss = val
            bad_epochs = 0
            _save_best(model, {
                "version": VERSION,
                "stage": name,
                "epoch": epoch,
                "validation_loss": best_loss,
            })
            print("  -> new best checkpoint for this stage (saved to disk)")
        else:
            bad_epochs += 1
            print(f"  -> no validation improvement ({bad_epochs}/{patience})")

        progress = {
            "version": VERSION,
            "seed_brain": "0.0004",
            "stage": name,
            "stage_complete": False,
            "epoch": epoch + 1,
            "step": 0,
            "total_steps": total,
            "best_validation_loss": best_loss,
            "bad_epochs": bad_epochs,
            "completed_stages": completed_stages,
            "updated_at": time.time(),
            "objective": "assistant-only-loss",
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
        "freeze_first_blocks": freeze_first_blocks,
        "objective": "assistant-only-loss",
    }
    completed_stages = completed_stages + [result]
    progress = {
        "version": VERSION,
        "seed_brain": "0.0004",
        "stage": name,
        "stage_complete": True,
        "epoch": epochs_completed,
        "step": 0,
        "best_validation_loss": best_loss,
        "bad_epochs": bad_epochs,
        "completed_stages": completed_stages,
        "updated_at": time.time(),
        "objective": "assistant-only-loss",
    }
    _save_resume(model, progress, f"STAGE {name.upper()} COMPLETE")
    _clean_best()
    return result, completed_stages


def train_v00051(preset="ryzen3600"):
    del preset  # v0.00051 inherits the exact accepted architecture; no resize is allowed.
    ensure_dirs()
    random.seed(5100)
    torch.manual_seed(5100)

    if not MANIFEST.exists():
        raise FileNotFoundError("Missing v0.00051 corrective corpus. Run 11_BUILD_CORRECTIVE_CORPUS_V00051.bat first.")
    for stage, paths in STAGE_FILES.items():
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Missing v0.00051 {stage} file: {path}")

    active = get_active_entry()
    if not active:
        raise RuntimeError("No active Butterfly brain to inherit from.")
    if active["version"] == VERSION:
        print("Butterfly v0.00051 is already active. There is nothing to retrain.")
        return MODELS_DIR / active["path"]
    if active["version"] != "0.0004":
        raise RuntimeError(
            f"v0.00051 was designed to inherit the accepted v0.0004 brain, but active is v{active['version']}. "
            "Do not overwrite lineage; inspect STATUS first."
        )

    existing_candidate = get_entry(VERSION)
    if existing_candidate and not _read_progress():
        raise RuntimeError(
            "A registered v0.00051 candidate already exists. Compare/promote it or remove it through the normal rejection flow; "
            "training will not overwrite a finished candidate."
        )

    device = best_device()
    threads = configure_cpu()
    progress = _read_progress()

    if progress:
        model, _ = load_checkpoint(RESUME_MODEL, device=device)
        _, _, tokenizer = load_entry(active, device="cpu")
        if progress.get("seed_brain") != "0.0004":
            raise RuntimeError("Recovery state has unexpected seed-brain lineage. Nothing was overwritten.")
        print("\nRECOVERY FOUND for v0.00051")
        print(f"Stage: {progress.get('stage')} | complete: {progress.get('stage_complete')} | epoch: {progress.get('epoch')} | step: {progress.get('step')}")
        print("Continuing from the last atomic weights-only autosave.")
    else:
        model, _, tokenizer = load_entry(active, device=device)
        progress = None
        print("\nStarting v0.00051 FROM the accepted v0.0004 weights.")
        print("This is continued learning, NOT a random-weight restart.")

    tokenizer_path = (active.get("metadata") or {}).get("tokenizer_path")
    if not tokenizer_path:
        raise RuntimeError("Active v0.0004 registry entry has no tokenizer lineage.")
    if tokenizer.vocab_size != model.cfg.vocab_size:
        raise RuntimeError("Tokenizer/model vocabulary mismatch. Training stopped safely.")

    print(f"Seed brain: v{active['version']}")
    print(f"Device: {device}")
    print(f"CPU threads: {threads} (safe cap for Windows/USB headroom)" if device == "cpu" else "GPU training enabled")
    print(f"Parameters: {model.parameter_count():,} (same architecture as v0.0004)")
    print(f"Tokenizer vocab: {tokenizer.vocab_size:,} (same tokenizer as v0.0004)")
    print("v0.0004 remains ACTIVE and untouched until v0.00051 passes benchmark v0.00042.")

    stage_defs = [
        # stage, epochs, lr, frozen lower blocks, patience
        # Robust dialogue first repairs casual/no-punctuation Spanish while
        # protecting most of the language base. Binding/math then gets a little
        # more plasticity; final mixed generalization unlocks the whole brain.
        ("robust_dialogue", 3, 5e-5, 3, 2),
        ("binding_math", 3, 5e-5, 2, 2),
        ("epistemic_contrast", 3, 4e-5, 2, 2),
        ("mixed_generalization", 2, 1.5e-5, 0, 1),
    ]

    completed = list((progress or {}).get("completed_stages", []))
    completed_names = {x.get("stage") for x in completed}
    stages = list(completed)

    for name, epochs, lr, freeze_first, patience in stage_defs:
        train_path, valid_path = STAGE_FILES[name]
        if name in completed_names:
            print(f"\n=== STAGE {name.upper()} === already completed in recovery state; skipping.")
            continue
        stage_resume = progress if progress and progress.get("stage") == name and not progress.get("stage_complete", False) else None
        _, completed = _train_stage(
            model,
            tokenizer,
            name,
            train_path,
            valid_path,
            epochs=epochs,
            lr=lr,
            freeze_first_blocks=freeze_first,
            patience=patience,
            seq_len=min(128, model.cfg.max_seq_len),
            batch_size=4,
            resume_progress=stage_resume,
            completed_stages=completed,
        )
        stages = list(completed)
        completed_names.add(name)
        progress = _read_progress()

    # Ensure all blocks are enabled before serializing the inference brain.
    _set_trainable(model, 0)
    candidate_path = MODELS_DIR / "butterfly-v0.00051-candidate.safetensors"
    extra = {
        "candidate": True,
        "version": VERSION,
        "seed_brain": "0.0004",
        "training_objective": "robust-contrastive-assistant-only-alignment-v2",
        "stages": stages,
        "tokenizer_path": tokenizer_path,
        "created_at": time.time(),
        "recovery": "weights-only autosaves; optimizer momentum intentionally not retained",
    }
    save_stable_model(candidate_path, model, extra=extra)
    register_model(
        candidate_path,
        VERSION,
        active=False,
        status="candidate",
        metadata={
            "parameters": model.parameter_count(),
            "tokenizer_vocab": tokenizer.vocab_size,
            "tokenizer_path": tokenizer_path,
            "seed_brain": "0.0004",
            "training_objective": "robust-contrastive-assistant-only-alignment-v2",
            "corpus_manifest": str(MANIFEST.relative_to(ROOT)),
            "stages": stages,
        },
    )
    print(f"\nSaved candidate: {candidate_path}")
    print("It is NOT active. v0.0004 is still the accepted brain.")
    print("Next: run 13_COMPARE_AND_PROMOTE_V00051.bat.")
    print("Candidate saved successfully; temporary recovery checkpoints are now removed.")
    _clean_training_state()
    return candidate_path
