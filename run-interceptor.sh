#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

declare -a pass_args=("$@")

if [[ $# -ge 1 && -n "${1:-}" ]]; then
  export OLLAMA_INTERCEPTOR_PORT="$1"
  pass_args=("${pass_args[@]:1}")
fi

if [[ $# -ge 2 && -n "${2:-}" ]]; then
  export OLLAMA_INTERCEPTOR_UPSTREAM="$2"
  pass_args=("${pass_args[@]:1}")
fi

export OLLAMA_INTERCEPTOR_DEBUG="${OLLAMA_INTERCEPTOR_DEBUG:-true}"

if [[ ! -d ".venv" ]]; then
  echo "[ERROR] .venv not found. Run install-proxy.sh first."
  exit 1
fi

echo "Starting ollama-interceptor"
echo "Working directory: $PWD"
echo "Config sources: CLI args > environment variables > .env > built-in defaults"
echo

if (( ${#pass_args[@]} )); then
  exec "$SCRIPT_DIR/.venv/bin/python" -m codex_openai_ollama_proxy.ollama_interceptor "${pass_args[@]}"
else
  exec "$SCRIPT_DIR/.venv/bin/python" -m codex_openai_ollama_proxy.ollama_interceptor
fi
