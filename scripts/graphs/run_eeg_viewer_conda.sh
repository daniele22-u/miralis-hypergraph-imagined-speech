#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run the EEG viewer inside the Conda env `daniele_dl_thesis`.
# Usage: ./run_eeg_viewer_conda.sh [args passed to the python script]

ENV_NAME="daniele_dl_thesis"
PY_SCRIPT="scripts/graphs/eeg_viewer.py"

echo "Preparing to run $PY_SCRIPT in conda env: $ENV_NAME"

# If a virtualenv is active, attempt to deactivate to avoid Python conflicts.
if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "Deactivating active virtualenv: $VIRTUAL_ENV"
  # `deactivate` is a shell function available in many venvs; try to call it safely.
  if command -v deactivate >/dev/null 2>&1; then
    deactivate || true
  else
    # Fallback: unset VIRTUAL_ENV and PATH adjustments (best-effort)
    unset VIRTUAL_ENV || true
  fi
fi

echo "Running viewer with: conda run -n $ENV_NAME python $PY_SCRIPT $*"
conda run -n "$ENV_NAME" python "$PY_SCRIPT" "$@"
