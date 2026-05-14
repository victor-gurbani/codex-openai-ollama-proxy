#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

declare -a pass_args=("$@")

LOG_FILE="$SCRIPT_DIR/logs/interceptor.stdout.log"
PID_FILE="$SCRIPT_DIR/interceptor.pid"

echo "Restarting ollama-interceptor"
echo "Working directory: $PWD"
echo

"$SCRIPT_DIR/stop-interceptor.sh" "${pass_args[@]}"

echo
echo "Starting fresh interceptor process in background..."
echo "stdout/stderr: $LOG_FILE"
echo "pid file: $PID_FILE"
echo

rm -f "$PID_FILE"

nohup "$SCRIPT_DIR/run-interceptor.sh" "${pass_args[@]}" > "$LOG_FILE" 2>&1 &
interceptor_pid=$!
printf '%s\n' "$interceptor_pid" > "$PID_FILE"

echo "[OK] Interceptor started in background with PID $interceptor_pid"
echo "[INFO] You can safely close this terminal."
