from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .alignment_data import AssistantOnlyDialogueDataset, load_jsonl
from .checkpoint import load_checkpoint, load_entry, metadata_path, save_stable_model
from .config import MODELS_DIR, ROOT
from .corpus.deliberate import load_manifest
from .learning.study_exam import study_microbenchmark
from .learning.dynamic_exam import (
    evaluate_bank, fresh_bank_excluding, fresh_pair, record_bank_use,
)
from .learning.evaluator import normalize_surface
from .registry import get_entry, get_candidate_entry, register_model
from .training.runtime import best_device, configure_cpu

TRAINING_ROOT = ROOT / "training_state" / "deliberate"
PROGRESS_PATH = TRAINING_ROOT / "progress.json"
RESUME_MODEL = TRAINING_ROOT / "resume.safetensors"
BEST_STAGE_MODEL = TRAINING_ROOT / "best-stage.safetensors"
ENTRY_STAGE_MODEL = TRAINING_ROOT / "entry-stage.safetensors"
AUTOSAVE_SECONDS = 600
DYNAMIC_FOCUS_KEY = "dynamic_selection_component"


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
        print(f"  -> autosave: {message} | {RESUME_MODEL.stat().st_size / (1024 * 1024):.1f} MB weights-only")


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


def _clean_entry():
    ENTRY_STAGE_MODEL.unlink(missing_ok=True)
    metadata_path(ENTRY_STAGE_MODEL).unlink(missing_ok=True)


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
    count = max(0, min(int(frozen_first_blocks), len(model.blocks)))
    for block in model.blocks[:count]:
        for p in block.parameters():
            p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total, count


@torch.no_grad()
def _validation_loss(model, dataset, batch_size: int):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    device = next(model.parameters()).device
    values = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        values.append(float(loss.item()))
    return sum(values) / max(1, len(values))


def _stable_stage_seed(experiment_seed: int, stage: str, epoch: int) -> int:
    raw = f"{experiment_seed}:{stage}:{epoch}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def _study_avg(exam: dict, names: list[str]) -> float:
    if not names:
        return float(exam.get("study_score", 0.0))
    return sum(float(exam.get(name, 0.0)) for name in names) / len(names)


def _study_protected_ok(candidate: dict, entry: dict, stage_cfg: dict) -> tuple[bool, list[str]]:
    default_budget = float(stage_cfg.get("max_study_protected_regression", 0.08))
    per_metric = dict(stage_cfg.get("study_protected_regression_budgets", {}))
    blockers = []
    for name in stage_cfg.get("study_protected_metrics", []):
        cur = float(candidate.get(name, 0.0))
        base = float(entry.get(name, 0.0))
        budget = float(per_metric.get(name, default_budget))
        if cur < base - budget:
            blockers.append(f"{name} {cur:.4f} < entry {base:.4f} - {budget:.4f}")
    return not blockers, blockers


def _study_focus_ok(candidate: dict, entry: dict, stage_cfg: dict) -> tuple[bool, list[str]]:
    required = float(stage_cfg.get("min_each_study_focus_delta", 0.0))
    blockers = []
    for name in stage_cfg.get("study_focus_metrics", []):
        cur = float(candidate.get(name, 0.0))
        base = float(entry.get(name, 0.0))
        if cur < base + required:
            blockers.append(f"{name} {cur:.4f} < entry {base:.4f} + {required:.4f}")
    return not blockers, blockers


def _study_focus_floor(exam: dict, entry: dict, names: list[str]) -> float:
    if not names:
        return float(exam.get("study_score", 0.0)) - float(entry.get("study_score", 0.0))
    return min(float(exam.get(name, 0.0)) - float(entry.get(name, 0.0)) for name in names)


def _study_is_better(candidate: dict, best: dict, entry: dict, candidate_val: float, best_val: float, stage_cfg: dict) -> bool:
    focus = list(stage_cfg.get("study_focus_metrics", []))
    protected = list(stage_cfg.get("study_protected_metrics", []))
    delta = float(stage_cfg.get("min_study_delta", 0.01))

    candidate_floor = _study_focus_floor(candidate, entry, focus)
    best_floor = _study_focus_floor(best, entry, focus)
    if candidate_floor > best_floor + delta:
        return True
    if candidate_floor < best_floor - delta:
        return False

    candidate_focus = _study_avg(candidate, focus)
    best_focus = _study_avg(best, focus)
    if candidate_focus > best_focus + delta:
        return True
    if candidate_focus < best_focus - delta:
        return False

    candidate_protected = _study_avg(candidate, protected) if protected else 0.0
    best_protected = _study_avg(best, protected) if protected else 0.0
    if candidate_protected > best_protected + 1e-9:
        return True
    return abs(candidate_protected - best_protected) <= 1e-9 and candidate_val < best_val


def _print_study(label: str, exam: dict):
    print(
        f"{label}: study={float(exam.get('study_score',0)):.4f} | "
        f"binding-exact={float(exam.get('binding_exact_component',0)):.4f} | "
        f"instruction-format={float(exam.get('instruction_format_component',0)):.4f} | "
        f"ret-conv={float(exam.get('retention_conversation_component',0)):.4f} | "
        f"ret-comp={float(exam.get('retention_comprehension_component',0)):.4f} | "
        f"ret-epi={float(exam.get('retention_epistemic_component',0)):.4f} | "
        f"ret-intent={float(exam.get('retention_intent_component',0)):.4f} | "
        f"ret-quality={float(exam.get('retention_quality_component',0)):.4f}"
    )
    if "instruction_sentence_component" in exam:
        print(
            "    fmt families: "
            f"sentence={float(exam.get('instruction_sentence_component',0)):.4f} | "
            f"two-steps={float(exam.get('instruction_two_steps_component',0)):.4f} | "
            f"missing={float(exam.get('instruction_missing_component',0)):.4f} | "
            f"short={float(exam.get('instruction_short_component',0)):.4f} | "
            f"weakest={float(exam.get('instruction_weakest_family_component',0)):.4f}"
        )
        print(
            "    conv families: "
            f"greeting={float(exam.get('retention_greeting_component',0)):.4f} | "
            f"thanks={float(exam.get('retention_thanks_component',0)):.4f} | "
            f"identity={float(exam.get('retention_identity_component',0)):.4f} | "
            f"state={float(exam.get('retention_state_component',0)):.4f}"
        )
        print(
            "    comp families: "
            f"file={float(exam.get('retention_file_component',0)):.4f} | "
            f"folder={float(exam.get('retention_folder_component',0)):.4f} | "
            f"api={float(exam.get('retention_api_component',0)):.4f} | "
            f"parameter={float(exam.get('retention_parameter_component',0)):.4f} | "
            f"token={float(exam.get('retention_token_component',0)):.4f}"
        )
    _print_dynamic(exam)

def _dynamic_stage_banks(experiment: dict, stage_name: str, count: int = 12):
    target = dict(experiment.get("focus_target") or {})
    family = target.get("dynamic_family")
    if not family:
        return None, None
    seed = _stable_stage_seed(int(experiment["random_seed"]), f"dynamic:{stage_name}", 0)
    return fresh_pair(str(family), seed, count=count)


def _study_with_dynamic_focus(model, tokenizer, experiment: dict, stage_cfg: dict, *, include_transfer: bool = False):
    exam = study_microbenchmark(model, tokenizer)
    target = dict(experiment.get("focus_target") or {})
    family = target.get("dynamic_family")
    if not family:
        return exam
    count = int(target.get("selection_cases", 12))
    selection, transfer = _dynamic_stage_banks(experiment, stage_cfg["name"], count=count)
    selection_result = evaluate_bank(model, tokenizer, selection)
    exam[DYNAMIC_FOCUS_KEY] = float(selection_result["score"])
    exam["dynamic_selection"] = {
        "family": family,
        "fingerprint": selection["fingerprint"],
        "score": float(selection_result["score"]),
        "semantic": float(selection_result["semantic"]),
    }
    if include_transfer:
        transfer_result = evaluate_bank(model, tokenizer, transfer)
        exam["dynamic_transfer_entry"] = {
            "family": family,
            "fingerprint": transfer["fingerprint"],
            "score": float(transfer_result["score"]),
            "semantic": float(transfer_result["semantic"]),
        }
        record_bank_use(selection, purpose="checkpoint_selection", experiment_id=experiment.get("experiment_id"))
        record_bank_use(transfer, purpose="stage_transfer", experiment_id=experiment.get("experiment_id"))
    return exam


def _print_dynamic(exam: dict):
    dynamic = exam.get("dynamic_selection") or {}
    if dynamic:
        print(
            f"    dynamic selection: family={dynamic.get('family')} "
            f"score={float(dynamic.get('score',0)):.4f} "
            f"bank={dynamic.get('fingerprint')}"
        )
    transfer = exam.get("dynamic_transfer_entry") or {}
    if transfer:
        print(
            f"    transfer entry   : score={float(transfer.get('score',0)):.4f} "
            f"bank={transfer.get('fingerprint')}"
        )


def _failure_driven_stage_cfg(stage_cfg: dict, experiment: dict) -> dict:
    cfg = dict(stage_cfg)
    target = dict(experiment.get("focus_target") or {})
    if target.get("dynamic_family"):
        cfg["study_focus_metrics"] = [DYNAMIC_FOCUS_KEY]
        cfg["min_each_study_focus_delta"] = float(
            target.get("selection_min_delta", cfg.get("failure_focus_min_delta", 0.02))
        )
        cfg["lr"] = float(cfg.get("lr", 0.0)) * float(target.get("lr_scale", 1.0))
        return cfg
    metric = target.get("study_metric")
    if not metric:
        return cfg
    cfg["study_focus_metrics"] = [str(metric)]
    cfg["study_protected_metrics"] = [name for name in cfg.get("study_protected_metrics", []) if name != metric]
    if target.get("reason") in {"critical_failure", "weakest_family"}:
        cfg["min_each_study_focus_delta"] = float(cfg.get("failure_focus_min_delta", cfg.get("min_each_study_focus_delta", 0.0)))
    return cfg


def _train_stage(model, tokenizer, experiment, stage_cfg, manifest_stage, resume_progress, completed):
    stage_cfg = _failure_driven_stage_cfg(stage_cfg, experiment)
    name = stage_cfg["name"]
    train_path = ROOT / manifest_stage["train_file"]
    valid_path = ROOT / manifest_stage["valid_file"]
    train_rows = load_jsonl(train_path)
    valid_rows = load_jsonl(valid_path)
    seq_len = int(stage_cfg.get("seq_len", 160))
    batch_size = int(stage_cfg.get("batch_size", 4))
    max_epochs = int(stage_cfg["max_epochs"])
    lr = float(stage_cfg["lr"])
    patience = int(stage_cfg.get("patience", 1))
    min_delta = float(stage_cfg.get("min_delta", 1e-3))
    weight_decay = float(stage_cfg.get("weight_decay", 0.05))
    frozen = int(stage_cfg.get("frozen_first_blocks", 0))
    study_enabled = bool(stage_cfg.get("study_focus_metrics"))
    effective_patience = int(stage_cfg.get("study_patience", patience)) if study_enabled else patience

    train_ds = AssistantOnlyDialogueDataset(train_rows, tokenizer, seq_len=seq_len)
    valid_ds = AssistantOnlyDialogueDataset(valid_rows, tokenizer, seq_len=seq_len)
    trainable, total_params, frozen = _set_frozen_blocks(model, frozen)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    bad_epochs = 0
    start_epoch = 1
    resume_step = 0
    epochs_completed = 0
    completed = list(completed or [])
    entry_study = None
    best_study = None
    selected_epoch = None

    if resume_progress and resume_progress.get("stage") == name and not resume_progress.get("stage_complete"):
        start_epoch = max(1, int(resume_progress.get("epoch", 1)))
        resume_step = max(0, int(resume_progress.get("step", 0)))
        raw = resume_progress.get("best_validation_loss")
        best_loss = float(raw) if raw is not None else float("inf")
        bad_epochs = int(resume_progress.get("bad_epochs", 0))
        epochs_completed = max(0, start_epoch - 1)
        if study_enabled:
            entry_study = resume_progress.get("stage_entry_study")
            best_study = resume_progress.get("best_study_exam")
            selected_epoch = resume_progress.get("selected_study_epoch", 0)
            if not entry_study or not best_study:
                raise RuntimeError(f"Recovery for study-enabled stage {name} lacks study state")
            if (experiment.get("focus_target") or {}).get("dynamic_family") and not ENTRY_STAGE_MODEL.exists():
                raise RuntimeError(f"Recovery for dynamic stage {name} lacks entry-stage weights")
            if not BEST_STAGE_MODEL.exists():
                _save_best(model, {"experiment_id":experiment["experiment_id"],"stage":name,"reconstructed_from_resume":True,"study_exam":best_study})
        print(f"\nRESUME {name}: epoch {start_epoch}, after batch {resume_step:,}. Optimizer momentum restarts; weights do not.")
    else:
        _clean_best()
        if study_enabled:
            print(f"\n=== STUDY ENTRY EXAM: {name} ===")
            entry_study = _study_with_dynamic_focus(
                model, tokenizer, experiment, stage_cfg, include_transfer=True
            )
            best_study = entry_study
            selected_epoch = 0
            _print_study("entry", entry_study)
            _save_best(model, {"experiment_id":experiment["experiment_id"],"stage":name,"epoch":0,"stage_entry":True,"study_exam":entry_study})
            if (experiment.get("focus_target") or {}).get("dynamic_family"):
                _atomic_weights(ENTRY_STAGE_MODEL, model, {"experiment_id":experiment["experiment_id"],"stage":name,"dynamic_stage_entry":True})

    target = experiment.get("focus_target") or {}
    if target.get("study_metric"):
        print(f"Failure-driven focus: family={target.get('family')} | metric={target.get('study_metric')} | reason={target.get('reason')}")
    print(f"\n=== STAGE {name.upper()} ===")
    print(f"train rows: {len(train_rows):,} | valid rows: {len(valid_rows):,}")
    print(f"assistant target tokens: {train_ds.answer_tokens:,} train | {valid_ds.answer_tokens:,} valid")
    print(f"batch: {batch_size} | max epochs: {max_epochs} | lr: {lr:g}")
    print(f"trainable parameters: {trainable:,}/{total_params:,} | frozen first blocks: {frozen}")
    print("Objective: USER tokens are context-only; loss is computed ONLY on Butterfly answer tokens.")
    if study_enabled:
        print("Checkpoint policy: held-out STUDY EXAM is primary; token validation loss is only a tiebreaker.")
    print("Recovery: weights-only autosave about every 10 minutes + every epoch/stage.")

    device = next(model.parameters()).device
    stage_start = time.time()
    last_autosave = time.time()

    def extra_state():
        if not study_enabled:
            return {}
        return {"stage_entry_study":entry_study,"best_study_exam":best_study,"selected_study_epoch":selected_epoch}

    for epoch in range(start_epoch, max_epochs + 1):
        generator = torch.Generator()
        generator.manual_seed(_stable_stage_seed(int(experiment["random_seed"]), name, epoch))
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
        model.train(); running=[]; started=time.time(); total=len(loader)
        skip_through = resume_step if epoch == start_epoch else 0
        if skip_through:
            print(f"Skipping first {skip_through:,}/{total:,} batches already represented in resumed weights.")

        for step, (x, y) in enumerate(loader, 1):
            if step <= skip_through:
                continue
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); running.append(float(loss.item()))
            if step == skip_through + 1 or step % max(1,total//12) == 0 or step == total:
                elapsed=time.time()-started; done=max(1,step-skip_through); rate=done/max(elapsed,1e-6)
                eta=(total-step)/max(rate,1e-6); avg=sum(running[-50:])/max(1,len(running[-50:]))
                print(f"{name} epoch {epoch}/{max_epochs} | {step:>5}/{total} | answer-loss {avg:.4f} | ETA {eta/60:.1f}m")
            if time.time()-last_autosave >= AUTOSAVE_SECONDS:
                progress={"experiment_id":experiment["experiment_id"],"target_version":experiment["target_version"],"recipe_hash":experiment["recipe_hash"],
                          "stage":name,"stage_complete":False,"epoch":epoch,"step":step,"total_steps":total,
                          "best_validation_loss":None if best_loss == float("inf") else best_loss,"bad_epochs":bad_epochs,
                          "completed_stages":completed,"updated_at":time.time(),**extra_state()}
                _save_resume(model, progress, f"{name} epoch {epoch} batch {step}/{total}"); last_autosave=time.time()

        val = _validation_loss(model, valid_ds, batch_size=batch_size)
        train_avg = sum(running)/max(1,len(running)) if running else float("nan")
        print(f"{name} epoch {epoch}: answer-train={train_avg:.4f} answer-valid={val:.4f}")
        epochs_completed = epoch

        if study_enabled:
            exam = _study_with_dynamic_focus(model, tokenizer, experiment, stage_cfg, include_transfer=False)
            _print_study(f"{name} epoch {epoch} exam", exam)
            protected_ok, protected_blockers = _study_protected_ok(exam, entry_study, stage_cfg)
            focus_ok, focus_blockers = _study_focus_ok(exam, entry_study, stage_cfg)
            selected = (
                protected_ok
                and focus_ok
                and _study_is_better(exam, best_study, entry_study, val, best_loss, stage_cfg)
            )
            if selected:
                best_study=exam; best_loss=val; selected_epoch=epoch; bad_epochs=0
                _save_best(model,{"experiment_id":experiment["experiment_id"],"stage":name,"epoch":epoch,"validation_loss":val,"study_exam":exam})
                print("  -> new BEST STUDY checkpoint")
            else:
                bad_epochs += 1
                if not protected_ok:
                    print("  -> protected capability gate blocked epoch:")
                    for blocker in protected_blockers:
                        print(f"     - {blocker}")
                if not focus_ok:
                    print("  -> focus gate blocked epoch:")
                    for blocker in focus_blockers:
                        print(f"     - {blocker}")
                if protected_ok and focus_ok:
                    print(f"  -> no balanced held-out study improvement ({bad_epochs}/{effective_patience})")
                if BEST_STAGE_MODEL.exists():
                    _load_weights_into(model, BEST_STAGE_MODEL, device)
                    print("  -> rollback to previous best study checkpoint")
        else:
            if val < best_loss - min_delta:
                best_loss=val; bad_epochs=0
                _save_best(model,{"experiment_id":experiment["experiment_id"],"stage":name,"epoch":epoch,"validation_loss":best_loss})
                print("  -> new best checkpoint for this stage (saved to disk)")
            else:
                bad_epochs += 1
                print(f"  -> no validation improvement ({bad_epochs}/{effective_patience})")

        progress={"experiment_id":experiment["experiment_id"],"target_version":experiment["target_version"],"recipe_hash":experiment["recipe_hash"],
                  "stage":name,"stage_complete":False,"epoch":epoch+1,"step":0,"total_steps":total,
                  "best_validation_loss":None if best_loss == float("inf") else best_loss,"bad_epochs":bad_epochs,
                  "completed_stages":completed,"updated_at":time.time(),**extra_state()}
        _save_resume(model, progress, f"{name} epoch {epoch} complete"); last_autosave=time.time(); resume_step=0
        if bad_epochs >= effective_patience:
            print("  -> early stopping"); break

    if BEST_STAGE_MODEL.exists():
        _load_weights_into(model, BEST_STAGE_MODEL, device)
    dynamic_transfer = None
    if study_enabled:
        _print_study(f"{name} selected checkpoint", best_study)
        if selected_epoch == 0:
            print("  -> STAGE ROLLBACK: no epoch beat the stage-entry study checkpoint")
        elif (experiment.get("focus_target") or {}).get("dynamic_family"):
            target = dict(experiment.get("focus_target") or {})
            count = int(target.get("selection_cases", 12))
            _, transfer_bank = _dynamic_stage_banks(experiment, name, count=count)
            transfer_exam = evaluate_bank(model, tokenizer, transfer_bank)
            base_transfer = float((entry_study.get("dynamic_transfer_entry") or {}).get("score", 0.0))
            final_transfer = float(transfer_exam.get("score", 0.0))
            required_transfer = float(target.get("transfer_min_delta", 0.015))
            transfer_passed = final_transfer >= base_transfer + required_transfer
            dynamic_transfer = {
                "fingerprint": transfer_bank["fingerprint"],
                "entry_score": base_transfer,
                "final_score": final_transfer,
                "delta": final_transfer - base_transfer,
                "required_delta": required_transfer,
                "passed": transfer_passed,
            }
            print(
                f"  -> fresh TRANSFER exam: {base_transfer:.4f} -> {final_transfer:.4f} "
                f"delta={final_transfer-base_transfer:+.4f} required={required_transfer:+.4f}"
            )
            if not transfer_passed:
                if not ENTRY_STAGE_MODEL.exists():
                    raise RuntimeError("Dynamic transfer failed but entry-stage checkpoint is missing")
                _load_weights_into(model, ENTRY_STAGE_MODEL, device)
                best_study = entry_study
                selected_epoch = 0
                print("  -> TRANSFER GATE FAILED: rollback to unseen stage-entry weights")

    result={"stage":name,"best_validation_loss":None if best_loss == float("inf") else best_loss,
            "seconds":time.time()-stage_start,"epochs_completed":epochs_completed,
            "train_rows":len(train_rows),"valid_rows":len(valid_rows)}
    if study_enabled:
        result.update({"stage_entry_study":entry_study,"best_study_exam":best_study,
                       "selected_study_epoch":selected_epoch,"stage_rolled_back":selected_epoch == 0})
        if dynamic_transfer is not None:
            result["dynamic_transfer"] = dynamic_transfer
    completed = completed + [result]
    progress={"experiment_id":experiment["experiment_id"],"target_version":experiment["target_version"],"recipe_hash":experiment["recipe_hash"],
              "stage":name,"stage_complete":True,"epoch":epochs_completed,"step":0,
              "best_validation_loss":None if best_loss == float("inf") else best_loss,"bad_epochs":bad_epochs,
              "completed_stages":completed,"updated_at":time.time(),**extra_state()}
    _save_resume(model, progress, f"STAGE {name.upper()} COMPLETE")
    _clean_best()
    _clean_entry()
    return result, completed

def _candidate_lifelong_acceptance(model, tokenizer, seed: dict, experiment: dict, recipe: dict, device: str):
    target = dict(experiment.get("focus_target") or {})
    family = target.get("dynamic_family")
    node_id = target.get("curriculum_node")
    if not family or not node_id:
        return None

    count = int(target.get("acceptance_cases", 16))
    forbidden = set()
    for stage_cfg in recipe.get("training_stages", []):
        selection, transfer = _dynamic_stage_banks(experiment, stage_cfg["name"], count=int(target.get("selection_cases", 12)))
        for bank in (selection, transfer):
            forbidden |= {normalize_surface(row["prompt"]) for row in bank.get("cases", [])}

    acceptance_seed = _stable_stage_seed(int(experiment["random_seed"]), "dynamic:candidate-acceptance", 0)
    bank = fresh_bank_excluding(
        str(family), acceptance_seed, count=count, mode="acceptance", forbidden_surfaces=forbidden
    )
    record_bank_use(bank, purpose="candidate_acceptance", experiment_id=experiment.get("experiment_id"))

    seed_model, _, _ = load_entry(seed, device=device)
    seed_exam = evaluate_bank(seed_model, tokenizer, bank)
    del seed_model
    candidate_exam = evaluate_bank(model, tokenizer, bank)
    base = float(seed_exam["score"])
    current = float(candidate_exam["score"])
    minimum_delta = float(target.get("acceptance_min_delta", 0.02))
    minimum_score = float(target.get("acceptance_min_score", 0.60))
    passed = current >= base + minimum_delta and current >= minimum_score
    result = {
        "curriculum_node": node_id,
        "family": family,
        "fingerprint": bank["fingerprint"],
        "seed_score": base,
        "candidate_score": current,
        "delta": current - base,
        "required_delta": minimum_delta,
        "required_score": minimum_score,
        "passed": passed,
    }
    print("\n=== FRESH CANDIDATE ACCEPTANCE EXAM ===")
    print(f"node   : {node_id}")
    print(f"family : {family}")
    print(f"bank   : {bank['fingerprint']}")
    print(f"score  : {base:.4f} -> {current:.4f} delta={current-base:+.4f}")
    print(f"gate   : delta>={minimum_delta:+.4f} and score>={minimum_score:.4f} -> {'PASS' if passed else 'FAIL'}")
    return result


def train_candidate(experiment: dict, recipe: dict):
    manifest = load_manifest()
    for key in ("experiment_id", "target_version", "recipe_hash", "suite_id"):
        if str(manifest.get(key)) != str(experiment.get(key)):
            raise RuntimeError(f"Corpus manifest mismatch for {key}.")

    existing = get_candidate_entry()
    if existing:
        raise RuntimeError(f"A candidate is already registered: {existing['version']}")

    seed = get_entry(experiment["seed_version"])
    if not seed:
        raise RuntimeError("Experiment seed is no longer registered.")

    random.seed(int(experiment["random_seed"]))
    torch.manual_seed(int(experiment["random_seed"]))

    device = best_device()
    threads = configure_cpu()
    progress = _read_progress()

    if progress:
        if (
            progress.get("experiment_id") != experiment["experiment_id"]
            or progress.get("recipe_hash") != experiment["recipe_hash"]
        ):
            raise RuntimeError("Recovery state belongs to another experiment. It was not overwritten.")
        model, _ = load_checkpoint(RESUME_MODEL, device=device)
        _, _, tokenizer = load_entry(seed, device="cpu")
        print("\nRECOVERY FOUND")
        print(
            f"Stage: {progress.get('stage')} | complete: {progress.get('stage_complete')} | "
            f"epoch: {progress.get('epoch')} | step: {progress.get('step')}"
        )
        print("Continuing from last atomic weights-only autosave. Adam momentum intentionally restarts.")
    else:
        model, _, tokenizer = load_entry(seed, device=device)
        progress = None
        print(f"\nStarting candidate {experiment['target_version']} from seed {seed['version']} weights.")
        print("This is continued learning, NOT a random-weight restart.")

    if model.cfg.vocab_size != tokenizer.vocab_size:
        raise RuntimeError("Model/tokenizer vocabulary mismatch.")
    tokenizer_rel = (seed.get("metadata") or {}).get("tokenizer_path")
    if not tokenizer_rel:
        raise RuntimeError("Seed brain has no tokenizer lineage metadata.")

    print(f"Seed brain: {seed['version']} ({seed.get('status')})")
    print(f"Device: {device}")
    print(f"CPU threads: {threads}" if device == "cpu" else "CPU threads: n/a")
    print(f"Parameters: {model.parameter_count():,}")
    print(f"Tokenizer vocab: {tokenizer.vocab_size:,}")

    completed = list((progress or {}).get("completed_stages", []))
    completed_names = {x.get("stage") for x in completed}

    for stage_cfg in recipe.get("training_stages", []):
        name = stage_cfg["name"]
        if name in completed_names:
            print(f"\n=== STAGE {name.upper()} === already completed in recovery; skipping.")
            continue
        manifest_stage = manifest.get("stages", {}).get(name)
        if not manifest_stage:
            raise RuntimeError(f"Manifest is missing stage {name}.")
        stage_resume = (
            progress
            if progress and progress.get("stage") == name and not progress.get("stage_complete")
            else None
        )
        _, completed = _train_stage(
            model, tokenizer, experiment, stage_cfg, manifest_stage, stage_resume, completed
        )
        completed_names.add(name)
        progress = _read_progress()

    lifelong_acceptance = _candidate_lifelong_acceptance(
        model, tokenizer, seed, experiment, recipe, device
    )

    target = experiment["target_version"]
    candidate_path = MODELS_DIR / f"butterfly-v{target}-candidate.safetensors"
    extra = {
        "candidate": True,
        "version": target,
        "experiment_id": experiment["experiment_id"],
        "seed_brain": seed["version"],
        "seed_slot": experiment["seed_slot"],
        "suite_id": experiment["suite_id"],
        "stages": completed,
        "tokenizer_path": tokenizer_rel,
        "created_at": time.time(),
        "recipe_hash": experiment["recipe_hash"],
        "recipe_name": experiment["recipe_name"],
        "focus_target": experiment.get("focus_target"),
        "lifelong_acceptance": lifelong_acceptance,
    }
    save_stable_model(candidate_path, model, extra=extra)
    register_model(
        candidate_path,
        target,
        active=False,
        status="candidate",
        metadata={
            "parameters": model.parameter_count(),
            "tokenizer_vocab": tokenizer.vocab_size,
            "tokenizer_path": tokenizer_rel,
            "seed_brain": seed["version"],
            "seed_slot": experiment["seed_slot"],
            "suite_id": experiment["suite_id"],
            "stages": completed,
            "recipe_hash": experiment["recipe_hash"],
            "recipe_name": experiment["recipe_name"],
            "experiment_id": experiment["experiment_id"],
            "focus_target": experiment.get("focus_target"),
            "lifelong_acceptance": lifelong_acceptance,
        },
    )

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    training_report = {
        "experiment_id": experiment["experiment_id"],
        "target_version": target,
        "seed_version": seed["version"],
        "recipe_name": experiment["recipe_name"],
        "stages": completed,
        "focus_target": experiment.get("focus_target"),
        "lifelong_acceptance": lifelong_acceptance,
    }
    text = json.dumps(training_report, indent=2, ensure_ascii=False)
    (reports / "latest-training.json").write_text(text, encoding="utf-8")
    (reports / f"brain-{target}-training.json").write_text(text, encoding="utf-8")

    print(f"\nSaved candidate: {candidate_path}")
    print("ACTIVE and LAB remain unchanged until evaluation.")
    _clean_training_state()
    return candidate_path
