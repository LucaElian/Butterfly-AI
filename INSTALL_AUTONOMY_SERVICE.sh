#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT_DIR="$(pwd)"
SERVICE_DIR="$HOME/.config/systemd/user"
SERVICE_PATH="$SERVICE_DIR/butterfly-autonomy.service"

mkdir -p "$SERVICE_DIR"
chmod +x "$ROOT_DIR"/*.sh 2>/dev/null || true

cat > "$SERVICE_PATH" <<SERVICE
[Unit]
Description=ButterflyAI Autonomy Loop
After=network-online.target

[Service]
Type=simple
WorkingDirectory=$ROOT_DIR
Environment=BUTTERFLY_AUTONOMY_CONSOLE=compact
Environment=BUTTERFLY_AUTONOMY_PAUSE=never
Environment=BUTTERFLY_AUTONOMY_LOOP_SLEEP_SECONDS=900
ExecStart=$ROOT_DIR/RUN_AUTONOMY_LOOP.sh
Restart=on-failure
RestartSec=30

[Install]
WantedBy=default.target
SERVICE

systemctl --user daemon-reload
systemctl --user enable butterfly-autonomy.service
systemctl --user restart butterfly-autonomy.service

if command -v loginctl >/dev/null 2>&1; then
  loginctl enable-linger "$USER" >/dev/null 2>&1 || true
fi

printf '%s\n' "Servicio instalado y arrancado: butterfly-autonomy.service"
printf '%s\n' "Ver estado:  systemctl --user status butterfly-autonomy.service"
printf '%s\n' "Ver logs:    journalctl --user -u butterfly-autonomy.service -f"
printf '%s\n' "Detener:     systemctl --user stop butterfly-autonomy.service"
printf '%s\n' "Stop seguro: ./STOP_AUTONOMY.sh"