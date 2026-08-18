#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  printf '%s\n' "ERROR: falta .venv. Ejecuta bash SETUP_LINUX.sh primero."
  exit 1
fi

printf '%s\n\n' "ButterflyAI - carga automatica de material verificado base"
printf '%s\n' "Esto agrega experiencias verificadas curadas con procedencia."
printf '%s\n\n' "No descarga internet ni entrena texto crudo directamente."
"$PY" -m butterfly research-seed "$@"
printf '\n%s\n' "Listo. El proximo RUN_AUTONOMY puede usar este material cuando toque esos nodos."