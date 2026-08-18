#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

SLEEP_SECONDS="${BUTTERFLY_AUTONOMY_LOOP_SLEEP_SECONDS:-900}"
export BUTTERFLY_AUTONOMY_CONSOLE="${BUTTERFLY_AUTONOMY_CONSOLE:-compact}"
export BUTTERFLY_AUTONOMY_PAUSE="never"

printf '%s\n' "ButterflyAI autonomy loop started."
printf '%s\n' "Sleep between sessions: ${SLEEP_SECONDS}s"
printf '%s\n' "Create .butterfly/STOP_AUTONOMY or run ./STOP_AUTONOMY.sh to stop before the next session."

while true; do
  if [ -f .butterfly/STOP_AUTONOMY ]; then
    printf '%s\n' "STOP_AUTONOMY present before next session; loop exits."
    exit 0
  fi

  if command -v ollama >/dev/null 2>&1; then
    ./TEACHER_LESSONS.sh || true
  else
    printf '%s\n' "Ollama no esta instalado en esta VM; Autonomy seguira sin teacher local externo."
  fi

  ./RUN_AUTONOMY.sh || true

  if [ -f .butterfly/STOP_AUTONOMY ]; then
    printf '%s\n' "STOP_AUTONOMY present after session; loop exits."
    exit 0
  fi

  printf '%s\n' "Loop sleeping ${SLEEP_SECONDS}s. Latest report: reports/autonomy-latest.json"
  sleep "$SLEEP_SECONDS"
done