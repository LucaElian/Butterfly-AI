ButterflyAI v0.0004 - SafeTensor CPU/Resume hotfix

Fixes the crash seen immediately after STAGE LANGUAGE:
  SafetensorError: device device(type='cpu') is invalid

Cause:
- Butterfly passed a torch.device('cpu') object into safetensors while restoring
  best-stage.safetensors. safetensors expects a device string such as "cpu".

What this hotfix does:
1. Normalizes safetensors device arguments to strings in butterfly/checkpoint.py.
2. Preserves and reuses training_state/v0.0004 recovery checkpoints.
3. Detects the case where LANGUAGE already finished its only epoch and restores
   best-stage weights without retraining LANGUAGE.
4. Caps PyTorch CPU threads at 8 on CPU training to leave Windows/USB headroom.

IMPORTANT:
- DO NOT delete training_state/v0.0004.
- DO NOT rerun 00, 01, or 02.
- Extract this ZIP over the permanent ButterflyAI folder and replace files.
- Then run 03_TRAIN_BUTTERFLY_V0004.bat again.

Expected recovery after the reported crash:
  RECOVERY FOUND
  Stage: language ... epoch: 2 ...
  language: all 1 epoch(s) were already completed before the interruption.
  Restoring the saved best-stage weights; this stage will NOT be retrained.
  -> STAGE LANGUAGE COMPLETE
  === STAGE CONVERSATION ===

The saved LANGUAGE weights and validation result are not discarded.
