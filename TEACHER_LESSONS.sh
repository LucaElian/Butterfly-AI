#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="python3"
fi

printf '%s\n' "=================================================="
printf '%s\n' " ButterflyAI - LOCAL TEACHER LESSONS"
printf '%s\n' "=================================================="
printf '\n'
printf '%s\n' "Usa Ollama local. No usa OpenAI, Gemini ni APIs pagas."
printf '%s\n' "Genera material chico y lo guarda como experiencias verificadas."
printf '\n'

"$PY" -m butterfly teacher-lessons "$@" || {
  printf '\n%s\n' "Teacher termino con error."
  printf '%s\n' "Si Ollama no esta listo, instala/abre Ollama y ejecuta:"
  printf '%s\n' "  ollama pull qwen2.5:3b"
  printf '%s\n' "Despues volve a ejecutar ./TEACHER_LESSONS.sh"
  exit 1
}