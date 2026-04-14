#!/usr/bin/env bash
# =============================================================================
# Example: iterate2 with --wlm vela (OpenShift / MLBatch PyTorchJob)
#
# Prerequisites
# -------------
#   * helm CLI installed and on PATH
#   * oc CLI logged in to the target cluster
#   * mlbatch/tools/pytorchjob-generator/chart checked out locally
#   * The gridfm HPO YAML (configs/gridfm_graphkit_hpo.yaml) present
#
# How it works
# ------------
# 1. For each Optuna trial iterate2:
#    a. Samples hyperparameters from gridfm_graphkit_hpo.yaml
#    b. Builds the gridfm_graphkit CLI invocation from static + sampled params
#    c. Patches vela_gridfm_template.yaml:
#         - appends  "-trial-<N>"  to jobName  (unique resource per trial)
#         - sets     numGpusPerPod = gpu_num    (from the HPO space)
#         - replaces {{HPO_COMMAND}}            (the actual CLI call)
#    d. Runs:  helm template -f <patched.yaml> <chart> | oc create -f-
#    e. Polls until <jobName>-master-0 pod is Running
#    f. Streams:  oc logs -f <jobName>-master-0
#       (blocks until container exits; output captured for metric extraction)
#    g. Checks pod exit code; deletes the PyTorchJob resource
# 2. Metrics are extracted from the captured log and returned to Optuna.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Path to the mlbatch pytorchjob-generator helm chart.
# Clone mlbatch first:  git clone https://github.com/project-codeflare/mlbatch
CHART_PATH="${MLBATCH_CHART_PATH:-${REPO_ROOT}/../mlbatch/tools/pytorchjob-generator/chart}"

iterate2 \
  --script            "gridfm_graphkit train" \
  --interpreter       ""                                       \
  --wlm               vela                                    \
  --vela-job-template "${SCRIPT_DIR}/vela_gridfm_template.yaml" \
  --vela-chart-path   "${CHART_PATH}"                         \
  --vela-namespace    "${OC_NAMESPACE:-}"                     \
  --vela-cmd-placeholder "{{HPO_COMMAND}}"                    \
  --vela-pod-ready-timeout 600                                 \
  --vela-job-timeout  86400                                    \
  --gpu-count         1                                        \
  --optuna-study-name  gridfm_vela_hpo                        \
  --optuna-db-path     "sqlite:///gridfm_vela_hpo.db"         \
  --optuna-n-trials    20                                      \
  --hpo-yaml          "${REPO_ROOT}/configs/gridfm_graphkit_hpo.yaml"
