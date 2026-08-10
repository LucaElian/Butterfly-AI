# ButterflyAI + GitHub

## Código vs cerebro

GitHub repository:
- source code
- scripts
- configs
- tests
- benchmarks
- small manifests/history

GitHub Release asset:
- active Butterfly brain package produced by `EXPORT_ACTIVE_MODEL_FOR_RELEASE.bat`

Local only:
- `.venv`
- live SQLite memory
- generated multi-MB corpus
- temporary training state
- candidate brains

## v0.0003 -> v0.0004 suggested flow

Before update:

    git add .
    git commit -m "ButterflyAI v0.0003"
    git tag -a v0.0003 -m "ButterflyAI v0.0003"
    git push origin main
    git push origin v0.0003

After v0.0004 is promoted:

1. Run `GITHUB_STOP_TRACKING_RUNTIME.bat` once if old runtime/model files are tracked.
2. Inspect `git status`.
3. Commit source + benchmark/history changes.
4. Tag v0.0004.
5. Run `EXPORT_ACTIVE_MODEL_FOR_RELEASE.bat`.
6. Create GitHub Release v0.0004 and upload the ZIP from `release\`.

Do not `git add -f` the model unless intentionally creating a special LFS milestone.
