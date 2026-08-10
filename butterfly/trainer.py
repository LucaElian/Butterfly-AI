from __future__ import annotations
from pathlib import Path
import copy, json, os, random, shutil, time
import torch
from torch.utils.data import DataLoader
from .config import ModelConfig, MODELS_DIR, ROOT, V4_TOKENIZER_PATH, ensure_dirs
from .model import ButterflyTransformer
from .data import BlockTokenDataset, load_text
from .checkpoint import save_stable_model, load_checkpoint, metadata_path
from .registry import register_model
from .tokenizer import load_tokenizer

TRAINING_ROOT = ROOT / "training_state" / "v0.0004"
PROGRESS_PATH = TRAINING_ROOT / "progress.json"
RESUME_MODEL = TRAINING_ROOT / "resume.safetensors"
BEST_STAGE_MODEL = TRAINING_ROOT / "best-stage.safetensors"
AUTOSAVE_SECONDS = 600  # weights-only, about the size of the model rather than Adam x3


def best_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def configure_cpu():
    if torch.cuda.is_available():
        return None
    # Safe default for the Ryzen 5 3600: keep headroom for Windows/USB instead
    # of pinning all 12 logical CPUs at 100% for hours. This changes throughput,
    # not model capacity or the training data.
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


def _atomic_weights(path: Path, model: ButterflyTransformer, extra: dict):
    """Write a weights-only safetensors checkpoint atomically.

    If an external disk disappears during the write, the previous checkpoint remains
    intact because only the temporary file is replaced after a complete save.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.stem + ".tmp.safetensors")
    tmp_meta = metadata_path(tmp)
    tmp.unlink(missing_ok=True)
    tmp_meta.unlink(missing_ok=True)
    save_stable_model(tmp, model, extra=extra)
    os.replace(tmp, path)
    os.replace(tmp_meta, metadata_path(path))


def _save_resume(model: ButterflyTransformer, progress: dict, message: str | None = None):
    TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
    extra = {"training_resume": progress, "created_at": time.time()}
    _atomic_weights(RESUME_MODEL, model, extra)
    _atomic_json(PROGRESS_PATH, progress)
    if message:
        size_mb = RESUME_MODEL.stat().st_size / (1024 * 1024)
        print(f"  -> autosave: {message} | {size_mb:.1f} MB weights-only")


def _save_best_stage(model: ButterflyTransformer, progress: dict):
    extra = {"stage_best": progress, "created_at": time.time()}
    _atomic_weights(BEST_STAGE_MODEL, model, extra)


def _read_progress() -> dict | None:
    if not PROGRESS_PATH.exists() or not RESUME_MODEL.exists():
        return None
    try:
        return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _clean_stage_best():
    BEST_STAGE_MODEL.unlink(missing_ok=True)
    metadata_path(BEST_STAGE_MODEL).unlink(missing_ok=True)


def _clean_training_state():
    if TRAINING_ROOT.exists():
        shutil.rmtree(TRAINING_ROOT, ignore_errors=True)


@torch.no_grad()
def validation_loss(model, text, tokenizer, seq_len=192, batch_size=4, max_batches=None):
    ds = BlockTokenDataset(text, seq_len, tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    device = next(model.parameters()).device
    vals = []
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        _, loss = model(x.to(device), y.to(device))
        vals.append(float(loss.item()))
    return sum(vals) / max(1, len(vals))


def _load_model_weights_into(model: ButterflyTransformer, path: Path, device: str):
    loaded, _ = load_checkpoint(path, device=device)
    model.load_state_dict(loaded.state_dict(), strict=True)
    del loaded


def _train_stage(
    model,
    tokenizer,
    name,
    train_path,
    valid_path,
    epochs,
    lr,
    seq_len=192,
    batch_size=4,
    patience=2,
    resume_progress: dict | None = None,
    completed_stages: list | None = None,
):
    train_text = load_text(train_path)
    valid_text = load_text(valid_path)
    ds = BlockTokenDataset(train_text, seq_len, tokenizer)
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.08)

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
        print(f"\nRESUME {name}: epoch {start_epoch}, after batch {resume_step:,}. Optimizer momentum restarts, learned weights do not.")
        if BEST_STAGE_MODEL.exists() and best_loss < float("inf"):
            print(f"Preserved best validation for this stage: {best_loss:.4f}")
    else:
        _clean_stage_best()

    if start_epoch > epochs:
        print(f"\n{name}: all {epochs} epoch(s) were already completed before the interruption.")
        print("Restoring the saved best-stage weights; this stage will NOT be retrained.")

    print(f"\n=== STAGE {name.upper()} ===")
    print(f"train tokens: {len(tokenizer.encode(train_text)):,} | valid tokens: {len(tokenizer.encode(valid_text)):,}")
    print(f"blocks: {len(ds):,} | batch: {batch_size} | max epochs: {epochs}")
    print("Recovery: weights-only autosave every ~10 min + every completed epoch/stage.")

    stage_start = time.time()
    last_autosave = time.time()

    for epoch in range(start_epoch, epochs + 1):
        # Deterministic shuffle per epoch lets us skip already-processed batches after a restart.
        generator = torch.Generator()
        generator.manual_seed(1337 + sum(ord(c) for c in name) * 100 + epoch)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
        model.train()
        running = []
        started = time.time()
        total = len(loader)
        skip_through = resume_step if epoch == start_epoch else 0

        if skip_through:
            print(f"Skipping first {skip_through:,}/{total:,} batches: they are already represented in the resumed weights.")

        for step, (x, y) in enumerate(loader, 1):
            if step <= skip_through:
                continue
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running.append(float(loss.item()))

            if step == skip_through + 1 or step % max(1, total // 12) == 0 or step == total:
                elapsed = time.time() - started
                effective_steps = max(1, step - skip_through)
                rate = effective_steps / max(elapsed, 1e-6)
                eta = (total - step) / max(rate, 1e-6)
                avg = sum(running[-50:]) / max(1, len(running[-50:]))
                print(f"{name} epoch {epoch}/{epochs} | {step:>5}/{total} | train {avg:.4f} | ETA {eta/60:.1f}m")

            if time.time() - last_autosave >= AUTOSAVE_SECONDS:
                progress = {
                    "version": "0.0004",
                    "stage": name,
                    "stage_complete": False,
                    "epoch": epoch,
                    "step": step,
                    "total_steps": total,
                    "best_validation_loss": best_loss,
                    "bad_epochs": bad_epochs,
                    "completed_stages": completed_stages,
                    "updated_at": time.time(),
                    "resume_note": "Weights are exact at this batch; Adam momentum intentionally restarts after interruption.",
                }
                _save_resume(model, progress, f"{name} epoch {epoch} batch {step}/{total}")
                last_autosave = time.time()

        # Once an epoch reaches validation it is considered fully processed.
        val = validation_loss(model, valid_text, tokenizer, seq_len=seq_len, batch_size=batch_size)
        train_avg = sum(running) / max(1, len(running)) if running else float("nan")
        print(f"{name} epoch {epoch}: train={train_avg:.4f} valid={val:.4f}")
        epochs_completed = epoch

        if val < best_loss - 1e-3:
            best_loss = val
            bad_epochs = 0
            best_progress = {
                "version": "0.0004",
                "stage": name,
                "epoch": epoch,
                "validation_loss": best_loss,
            }
            _save_best_stage(model, best_progress)
            print("  -> new best checkpoint for this stage (saved to disk)")
        else:
            bad_epochs += 1
            print(f"  -> no validation improvement ({bad_epochs}/{patience})")

        # Save after every completed epoch, even if this epoch was not the best.
        progress = {
            "version": "0.0004",
            "stage": name,
            "stage_complete": False,
            "epoch": epoch + 1,
            "step": 0,
            "total_steps": total,
            "best_validation_loss": best_loss,
            "bad_epochs": bad_epochs,
            "completed_stages": completed_stages,
            "updated_at": time.time(),
            "resume_note": "Epoch completed. Resume starts from the next epoch if needed.",
        }
        _save_resume(model, progress, f"{name} epoch {epoch} complete")
        last_autosave = time.time()
        resume_step = 0

        if bad_epochs >= patience:
            print("  -> early stopping")
            break

    # Always finish a stage with the best validation weights, not merely the last epoch.
    if BEST_STAGE_MODEL.exists():
        _load_model_weights_into(model, BEST_STAGE_MODEL, device)

    stage_result = {
        "stage": name,
        "best_validation_loss": best_loss,
        "seconds": time.time() - stage_start,
        "epochs_completed": epochs_completed,
    }
    completed_stages = completed_stages + [stage_result]
    progress = {
        "version": "0.0004",
        "stage": name,
        "stage_complete": True,
        "epoch": epochs_completed,
        "step": 0,
        "best_validation_loss": best_loss,
        "bad_epochs": bad_epochs,
        "completed_stages": completed_stages,
        "updated_at": time.time(),
        "resume_note": f"Stage {name} completed with its best validation weights.",
    }
    _save_resume(model, progress, f"STAGE {name.upper()} COMPLETE")
    _clean_stage_best()
    return stage_result, completed_stages


def train_curriculum(version="0.0004", preset="ryzen3600"):
    from .config import LANG_TRAIN, LANG_VALID, CONV_TRAIN, CONV_VALID, INST_TRAIN, INST_VALID, config_for_preset

    ensure_dirs()
    random.seed(1337)
    torch.manual_seed(1337)
    required = [V4_TOKENIZER_PATH, LANG_TRAIN, LANG_VALID, CONV_TRAIN, CONV_VALID, INST_TRAIN, INST_VALID]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing v0.0004 preparation files:\n" + "\n".join(missing))

    tokenizer = load_tokenizer(V4_TOKENIZER_PATH)
    cfg, preset_name = config_for_preset(preset, tokenizer.vocab_size)
    device = best_device()
    threads = configure_cpu()

    progress = _read_progress()
    if progress:
        model, payload = load_checkpoint(RESUME_MODEL, device=device)
        if model.cfg.to_dict() != cfg.to_dict():
            raise RuntimeError("Resume checkpoint model config differs from current v0.0004 config. Keep training_state for inspection and do not overwrite it.")
        print("\nRECOVERY FOUND")
        print(f"Stage: {progress.get('stage')} | complete: {progress.get('stage_complete')} | epoch: {progress.get('epoch')} | step: {progress.get('step')}")
        print("Continuing from the last atomic weights-only autosave. No language weights are discarded.")
    else:
        model = ButterflyTransformer(cfg).to(device)
        progress = None
        print("\nNo resumable v0.0004 training found. Starting fresh weights.")

    thread_note = " (safe cap for Windows/USB headroom)" if device == "cpu" else ""
    print(f"Model preset: {preset_name}\nDevice: {device}\nCPU threads: {threads}{thread_note}\nParameters: {model.parameter_count():,}\nTokenizer vocab: {tokenizer.vocab_size:,}")
    print("Previous verified data and memories are inherited through the corpus; v0.0003 remains active until promotion.")

    stage_defs = [
        ("language", LANG_TRAIN, LANG_VALID, 1, 3e-4, 1),
        ("conversation", CONV_TRAIN, CONV_VALID, 4, 1.5e-4, 2),
        ("instruction", INST_TRAIN, INST_VALID, 4, 1.0e-4, 2),
    ]
    completed = list((progress or {}).get("completed_stages", []))
    completed_names = {x.get("stage") for x in completed}
    stages = list(completed)

    for name, train_path, valid_path, epochs, lr, patience in stage_defs:
        if name in completed_names:
            print(f"\n=== STAGE {name.upper()} === already completed in recovery state; skipping.")
            continue

        # If the preceding stage is marked complete, resume.safetensors already contains its best weights.
        stage_resume = progress if progress and progress.get("stage") == name and not progress.get("stage_complete", False) else None
        result, completed = _train_stage(
            model,
            tokenizer,
            name,
            train_path,
            valid_path,
            epochs=epochs,
            lr=lr,
            seq_len=192,
            batch_size=4,
            patience=patience,
            resume_progress=stage_resume,
            completed_stages=completed,
        )
        stages = list(completed)
        completed_names.add(name)
        # refresh recovery metadata from disk; this is what a restart will see
        progress = _read_progress()

    path = MODELS_DIR / f"butterfly-v{version}-candidate.safetensors"
    extra = {
        "candidate": True,
        "version": version,
        "stages": stages,
        "tokenizer_path": str(V4_TOKENIZER_PATH.relative_to(ROOT)),
        "created_at": time.time(),
        "recovery": "weights-only autosaves; optimizer state intentionally not retained",
    }
    save_stable_model(path, model, extra=extra)
    register_model(
        path,
        version,
        active=False,
        status="candidate",
        metadata={
            "parameters": model.parameter_count(),
            "tokenizer_vocab": tokenizer.vocab_size,
            "tokenizer_path": str(V4_TOKENIZER_PATH.relative_to(ROOT)),
            "stages": stages,
        },
    )
    print(f"\nSaved candidate: {path}")
    print("It is NOT active yet. Run 04_COMPARE_AND_PROMOTE.bat.")
    print("Candidate saved successfully; removing temporary recovery checkpoints to save space.")
    _clean_training_state()
    return path


def continue_training(model, text, tokenizer, steps=120, lr=8e-5, batch_size=4):
    device = next(model.parameters()).device
    ds = BlockTokenDataset(text, min(model.cfg.max_seq_len, 160), tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.1)
    model.train()
    it = iter(loader)
    last = 0.0
    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = float(loss.item())
    return model, last
