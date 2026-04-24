#!/usr/bin/env bash
# =============================================================================
# Example: iterate2 with the Vela WLM plugin (OpenShift / MLBatch PyTorchJob)
#
# Each Optuna trial is submitted as a PyTorchJob by the plugin script
# examples/wlm_plugins/vela_plugin.py.  That script reads all Vela/oc
# settings from ITERATE_WLM_* env vars which are populated from the
# ``wlm:`` section in the HPO YAML.
#
# What changed vs the old --wlm vela approach
# -------------------------------------------
# * --wlm vela, --vela-job-template, --vela-chart-path, --vela-namespace,
#   --vela-cmd-placeholder, --vela-pod-ready-timeout, --vela-job-timeout
#   are gone; all of these now live in the  wlm:  block of the HPO YAML.
# * --wlm-plugin points to the Vela plugin script (user-owned).
# * The plugin can be customised freely without touching iterate2 itself.
#
# Prerequisites
# -------------
#   * helm CLI installed and on PATH
#   * oc CLI logged in to the target cluster
#   * mlbatch/tools/pytorchjob-generator/chart checked out locally
#   * The gridfm HPO YAML (configs/gridfm_graphkit_hpo.yaml) present
#     and the wlm: section filled in (see configs/gridfm_graphkit_hpo.yaml)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

iterate2 \
  --script            "gridfm_graphkit train"                           \
  --interpreter       ""                                                \
  --wlm-plugin        "${SCRIPT_DIR}/wlm_plugins/vela_plugin.py"        \
  --no-underscore-to-hyphen                                             \
  --optuna-study-name  gridfm_vela_hpo                                  \
  --optuna-db-path     "js:///gridfm_vela_hpo.journal"                  \
  --parallelism        16                                               \
  --optuna-n-trials    20                                               \
  --hpo-yaml           "${REPO_ROOT}/configs/gridfm_graphkit_hpo.yaml"
