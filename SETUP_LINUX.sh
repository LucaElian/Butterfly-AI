#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

printf '%s\n' "========================================"
printf '%s\n' " ButterflyAI - LINUX SETUP"
printf '%s\n' "========================================"
printf '\n'

if ! command -v python3 >/dev/null 2>&1; then
  printf '%s\n' "ERROR: python3 no esta instalado."
  printf '%s\n' "En Ubuntu: sudo apt update && sudo apt install -y python3 python3-venv python3-pip"
  exit 1
fi

if ! python3 -m venv --help >/dev/null 2>&1; then
  printf '%s\n' "ERROR: falta python3-venv."
  printf '%s\n' "En Ubuntu: sudo apt update && sudo apt install -y python3-venv"
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  printf '%s\n' "Creando .venv..."
  python3 -m venv .venv
fi

PY=".venv/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r requirements.txt
"$PY" -m butterfly init
"$PY" -m butterfly health
"$PY" -m butterfly audit-hardcodes

chmod +x ./*.sh 2>/dev/null || true

printf '\n%s\n' "Setup listo. No se modificaron modelos, memoria ni corpus."
printf '%s\n' "Para correr una sesion: ./RUN_AUTONOMY.sh"
printf '%s\n' "Para modo cloud 24/7: ./INSTALL_AUTONOMY_SERVICE.sh"