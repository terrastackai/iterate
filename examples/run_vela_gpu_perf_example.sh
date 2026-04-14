#!/usr/bin/env bash
# =============================================================================
# Example: iterate2 GPU performance benchmark HPO on Vela (OpenShift / MLBatch)
#
# What iterate2 does per trial
# ----------------------------
# 1. Samples hyperparameters from configs/gpu_perf_hpo.yaml
# 2. Builds the CLI call:
#      python /app/.local/.../gpu_performance_test.py \
#        --mode single_gpu --batch-size <N> --num-workers <N> ...
# 3. Patches examples/vela_gpu_perf_template.yaml:
#      - appends  -trial-<N>  to jobName
#      - sets     numGpusPerPod = gpu_num
#      - replaces {{HPO_COMMAND}} with the CLI call (awk wrapper stays intact)
# 4. Submits:
#      helm template -f <patched.yaml> <chart> | oc create [-n <ns>] -f-
# 5. Streams:  oc logs -f <jobName>-master-0  (blocks until container exits)
# 6. Extracts metrics from the renamed [performance] lines:
#      dataloader_samples_per_sec, training_samples_per_sec,
#      inference_samples_per_sec, gflops
# 7. Deletes the PyTorchJob resource.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Path to the mlbatch pytorchjob-generator helm chart.
# Clone first:  git clone https://github.com/project-codeflare/mlbatch
CHART_PATH="${MLBATCH_CHART_PATH:-${REPO_ROOT}/../mlbatch/tools/pytorchjob-generator/chart}"

iterate2 \
  --script      "/app/.local/lib/python3.12/site-packages/claimed/components/util/gpu_performance_test.py" \
  --interpreter "python"                                                   \
  --wlm         vela                                                       \
  --vela-job-template  "${SCRIPT_DIR}/vela_gpu_perf_template.yaml"        \
  --vela-chart-path    "${CHART_PATH}"                                     \
  --vela-namespace     "${OC_NAMESPACE:-}"                                 \
  --vela-pod-ready-timeout 300                                             \
  --vela-job-timeout       7200                                            \
  --gpu-count              1                                               \
  --optuna-study-name      gpu_perf_hpo                                    \
  --optuna-db-path         "sqlite:///gpu_perf_hpo.db"                     \
  --optuna-n-trials        30                                              \
  --hpo-yaml               "${REPO_ROOT}/configs/gpu_perf_hpo.yaml"
