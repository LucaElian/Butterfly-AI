#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p .butterfly
: > .butterfly/STOP_AUTONOMY
printf '%s\n' "Stop requested."
printf '%s\n' "Butterfly terminara de forma segura apenas llegue al proximo punto seguro."