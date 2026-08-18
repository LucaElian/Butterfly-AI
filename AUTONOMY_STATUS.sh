#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  printf '%s\n' "ERROR: falta .venv. Ejecuta bash SETUP_LINUX.sh primero."
  exit 1
fi

"$PY" -m butterfly.learning.curriculum_graph --status
printf '\n%s\n' "Dynamic Exam quick check:"
"$PY" -m butterfly.learning.dynamic_exam --self-test