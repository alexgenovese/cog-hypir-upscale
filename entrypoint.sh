#!/usr/bin/env bash
set -euo pipefail

# If models directory is missing or empty, download models
MODELS_DIR=/src/models
if [ ! -d "$MODELS_DIR" ] || [ -z "$(ls -A $MODELS_DIR 2>/dev/null)" ]; then
  echo "Models not found in $MODELS_DIR — downloading..."
  python3 /src/download_models.py
else
  echo "Models present, skipping download"
fi

# Exec the passed command or sleep (so container doesn't exit immediately)
if [ $# -gt 0 ]; then
  exec "$@"
else
  exec sleep infinity
fi
