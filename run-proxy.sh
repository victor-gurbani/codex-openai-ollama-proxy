#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

declare -a pass_args=("$@")

if [[ $# -ge 1 && -n "${1:-}" ]]; then
  export PORT="$1"
  pass_args=("${pass_args[@]:1}")
fi

if [[ $# -ge 2 && -n "${2:-}" ]]; then
  export CODEX_AUTH_PATH="$2"
  pass_args=("${pass_args[@]:1}")
fi

if [[ ! -d ".venv" ]]; then
  echo "[ERROR] .venv not found. Run install-proxy.sh first."
  exit 1
fi

print_env_flag_status() {
  local name="$1"
  local value="${!name-}"
  local normalized
  normalized="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"

  if [[ -z "$value" ]]; then
    echo "$name: not set in shell; Python may load it from .env/defaults"
    return
  fi

  case "$normalized" in
    1|true|yes|on)
      echo "$name: on in shell; passed to Python"
      ;;
    *)
      echo "$name: off in shell ('$value'); passed to Python"
      ;;
  esac
}

echo "Starting codex-openai-ollama-proxy"
echo "Working directory: $PWD"
echo "Config sources: CLI args > environment variables > .env > built-in defaults"
print_env_flag_status "DEBUG"
print_env_flag_status "DISABLE_COPILOT_ADAPTATIONS"
echo

if (( ${#pass_args[@]} )); then
  exec "$SCRIPT_DIR/.venv/bin/python" -m codex_openai_ollama_proxy "${pass_args[@]}"
else
  exec "$SCRIPT_DIR/.venv/bin/python" -m codex_openai_ollama_proxy
fi
