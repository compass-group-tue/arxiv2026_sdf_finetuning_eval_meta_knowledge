#!/usr/bin/env bash
# Generic finetuning launcher. Reads a YAML config and calls ft_train.py.
# Usage: bash src/scripts/run_finetune.sh configs/finetune/my_experiment.yaml
export SOFT_FILELOCK=1
set -euo pipefail

CONFIG="${1:?Usage: run_finetune.sh <config.yaml>}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ROOT_DIR}/.env"
  set +a
fi

# ── Activate venv (one grep line — before any python call) ────────────────────
VENV_SUFFIX="$(grep -m1 '^venv:' "${ROOT_DIR}/${CONFIG}" | awk '{print $2}')"
VENV_PATH="${VENV_PATH:-${ROOT_DIR}/${VENV_SUFFIX}}"
if [[ -d "${VENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
else
  echo "Warning: venv not found at ${VENV_PATH}; continuing without activation."
fi

echo "ROOT_DIR=${ROOT_DIR}"
echo "CONFIG=${CONFIG}"
echo "VENV_PATH=${VENV_PATH}"
echo "HF_HOME=${HF_HOME:-<unset>}"
echo "SOFT_FILELOCK=${SOFT_FILELOCK}"
echo "python=$(command -v python || echo '<not found>')"

# ── Build argument list via Python ────────────────────────────────────────────
echo "--- building args from config ---"
mapfile -t ARGS < <(python "${ROOT_DIR}/src/scripts/config_to_args.py" "${ROOT_DIR}/${CONFIG}")
echo "--- args built (${#ARGS[@]} args) ---"

echo "Running: python src/ft_train.py ${ARGS[*]}"
python "${ROOT_DIR}/src/ft_train.py" "${ARGS[@]}"
echo "--- ft_train.py finished ---"
