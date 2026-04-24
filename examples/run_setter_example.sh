#!/usr/bin/env bash
# =============================================================================
# Example: iterate2 with --param-setter
#
# Some scripts (e.g. those using Hydra, MMCV, or custom key-value CLIs) do not
# accept traditional named flags:
#
#   python script.py --learning-rate 0.001 --batch-size 32
#
# Instead they expect a setter-style interface:
#
#   python script.py --set learning_rate 0.001 --set batch_size 32
#
# Pass --param-setter <flag> to iterate2 to switch to this style.
# Every HPO and static parameter will be forwarded as:
#   --<flag> key value
#
# This example uses examples/bumpy_setter.py which accepts --set key value.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

iterate2 \
  --script        "${SCRIPT_DIR}/bumpy_setter.py" \
  --root-dir      "${SCRIPT_DIR}" \
  --venv          ""                              \
  --param-setter  set                             \
  --optuna-study-name  bumpy_setter_study         \
  --optuna-db-path     "sqlite:///bumpy_setter_hpo.db" \
  --optuna-n-trials    20                         \
  --hpo-yaml      "${SCRIPT_DIR}/bumpy_setter_hpo.yaml" \
  --metric        "yval"
