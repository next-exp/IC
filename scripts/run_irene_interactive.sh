#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ICTDIR="${ICTDIR:-$ROOT_DIR}"

PORT="8501"
if [[ $# -gt 0 ]]; then
  PORT="$1"
fi

if ! command -v streamlit >/dev/null 2>&1; then
  echo "Error: streamlit command not found. Activate the IC environment and install streamlit." >&2
  exit 1
fi

echo "Starting interactive app on http://127.0.0.1:$PORT"
cd "$ROOT_DIR"

exec streamlit run "scripts/irene_interactive_app.py" \
  --server.port "$PORT" \
  --server.address 127.0.0.1
