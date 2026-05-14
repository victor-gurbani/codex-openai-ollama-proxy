#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_PORT="8888"
SHUTDOWN_WAIT_SECONDS="10"

trim() {
  local value="$1"
  value="${value#${value%%[![:space:]]*}}"
  value="${value%${value##*[![:space:]]}}"
  printf '%s' "$value"
}

trim_matching_quotes() {
  local value
  value="$(trim "$1")"
  if [[ ${#value} -ge 2 ]]; then
    local first_char="${value:0:1}"
    local last_char="${value: -1}"
    if [[ "$first_char" == "$last_char" && ( "$first_char" == '"' || "$first_char" == "'" ) ]]; then
      value="${value:1:${#value}-2}"
    fi
  fi
  printf '%s' "$value"
}

resolve_port() {
  if [[ $# -ge 1 && -n "${1:-}" ]]; then
    printf '%s' "$1"
    return 0
  fi

  if [[ -n "${PORT:-}" ]]; then
    printf '%s' "$PORT"
    return 0
  fi

  if [[ -f ".env" ]]; then
    while IFS= read -r raw_line; do
      local line
      line="$(trim "$raw_line")"
      [[ -n "$line" ]] || continue
      [[ "$line" == \#* ]] && continue
      [[ "$line" == *=* ]] || continue

      local key="${line%%=*}"
      local value="${line#*=}"
      key="$(trim "$key")"
      if [[ "$key" == "PORT" ]]; then
        printf '%s' "$(trim_matching_quotes "$value")"
        return 0
      fi
    done < ".env"
  fi

  printf '%s' "$DEFAULT_PORT"
}

PORT_TO_STOP="$(resolve_port "${1:-}")"

echo "Stopping codex-openai-ollama-proxy"
echo "Working directory: $PWD"
echo "Target port: $PORT_TO_STOP"
echo

mapfile -t listening_pids < <(lsof -nP -iTCP:"$PORT_TO_STOP" -sTCP:LISTEN -t 2>/dev/null || true)

if (( ${#listening_pids[@]} == 0 )); then
  echo "[INFO] No listening process found on port $PORT_TO_STOP."
  exit 0
fi

matching_pids=()
non_matching_pids=()

for pid in "${listening_pids[@]}"; do
  command_line="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  if [[ "$command_line" == *"codex_openai_ollama_proxy"* ]]; then
    matching_pids+=("$pid")
  else
    non_matching_pids+=("$pid")
  fi
done

if (( ${#matching_pids[@]} == 0 )); then
  echo "[ERROR] Found listener(s) on port $PORT_TO_STOP, but none appear to be codex-openai-ollama-proxy. Refusing to stop unrelated processes."
  for pid in "${non_matching_pids[@]}"; do
    echo "  PID $pid: $(ps -p "$pid" -o command= 2>/dev/null || true)"
  done
  exit 1
fi

echo "Sending SIGTERM to: ${matching_pids[*]}"
kill "${matching_pids[@]}"

deadline=$((SECONDS + SHUTDOWN_WAIT_SECONDS))
while (( SECONDS < deadline )); do
  still_running=0
  for pid in "${matching_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      still_running=1
      break
    fi
  done

  if (( still_running == 0 )); then
    echo "[OK] Proxy stopped cleanly."
    exit 0
  fi

  sleep 1
done

still_running_pids=()
for pid in "${matching_pids[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    still_running_pids+=("$pid")
  fi
done

if (( ${#still_running_pids[@]} == 0 )); then
  echo "[OK] Proxy stopped cleanly."
  exit 0
fi

if [[ "${FORCE:-0}" == "1" ]]; then
  echo "[WARN] Proxy still running after ${SHUTDOWN_WAIT_SECONDS}s. Sending SIGKILL because FORCE=1 was set."
  kill -9 "${still_running_pids[@]}"
  echo "[OK] Proxy force-stopped."
  exit 0
fi

echo "[WARN] Proxy still running after ${SHUTDOWN_WAIT_SECONDS}s: ${still_running_pids[*]}"
echo "[INFO] Re-run with FORCE=1 to send SIGKILL if you really need to force-stop it."
exit 1
