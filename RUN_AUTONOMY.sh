#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if command -v clear >/dev/null 2>&1 && [ -t 1 ]; then
  clear
fi

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  printf '%s\n' "ERROR: falta .venv. Ejecuta bash SETUP_LINUX.sh primero."
  exit 1
fi

export BUTTERFLY_AUTONOMY_CONSOLE="${BUTTERFLY_AUTONOMY_CONSOLE:-live}"
export BUTTERFLY_AUTONOMY_PAUSE="${BUTTERFLY_AUTONOMY_PAUSE:-never}"

printf '%s\n' "=================================================="
printf '%s\n' " ButterflyAI - AUTONOMY"
printf '%s\n' "=================================================="
printf '\n'
printf '%s\n' "Consola: ${BUTTERFLY_AUTONOMY_CONSOLE} (el log guarda todo completo)."
printf '%s\n' "Objetivo: estudiar capacidades pendientes sin supervision constante."
printf '%s\n' "Seed/LAB/ACTIVE, suite y target se resuelven dinamicamente."
printf '%s\n' "Limites: config/autonomy_learning.json (0 = ilimitado)."
printf '%s\n' "STOP seguro: ./STOP_AUTONOMY.sh"
printf '\n'
printf '%s\n' "Durante la sesion vas a ver progreso vivo y eventos importantes:"
printf '%s\n' "  BLOCK        intento autonomo actual."
printf '%s\n' "  EPOCH        avance de entrenamiento y answer-loss en vivo."
printf '%s\n' "  LAB_ACCEPTED mejora conservada como nuevo LAB."
printf '%s\n' "  REJECTED     pesos descartados; evidencia queda en benchmark."
printf '%s\n' "  SKILL CREDIT rescate parcial seguro sin aceptar pesos rotos."
printf '%s\n' "  TEACHER      material local verificado generado con Ollama si falta corpus."
printf '\n%s\n\n' "Starting..."

set +e
"$PY" -m butterfly autonomy
ERR=$?
set -e

printf '\n'
if [ "$ERR" -ne 0 ]; then
  printf '%s\n' "Autonomy termino con error ${ERR}."
else
  printf '%s\n' "Autonomy termino. Revisa reports/autonomy-latest.json o logs/autonomy-*.log para el detalle completo."
fi
exit "$ERR"