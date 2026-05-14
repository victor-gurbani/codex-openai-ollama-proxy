#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

declare -a pass_args=("$@")

LOG_FILE="$SCRIPT_DIR/logs/proxy.stdout.log"
PID_FILE="$SCRIPT_DIR/proxy.pid"

echo "Restarting codex-openai-ollama-proxy"
echo "Working directory: $PWD"
echo

"$SCRIPT_DIR/stop-proxy.sh" "${pass_args[@]}"

echo
echo "Starting fresh proxy process in background..."
echo "stdout/stderr: $LOG_FILE"
echo "pid file: $PID_FILE"
echo

rm -f "$PID_FILE"

nohup "$SCRIPT_DIR/run-proxy.sh" "${pass_args[@]}" > "$LOG_FILE" 2>&1 &
proxy_pid=$!
printf '%s\n' "$proxy_pid" > "$PID_FILE"

echo "[OK] Proxy started in background with PID $proxy_pid"
echo "[INFO] You can safely close this terminal."
