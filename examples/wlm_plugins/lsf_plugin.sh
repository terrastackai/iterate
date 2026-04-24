#!/usr/bin/env bash
# =============================================================================
# iterate2 WLM plugin – IBM Spectrum LSF
#
# Submits each HPO trial as an LSF job via ``bsub -K`` (blocking).
#
# Environment variables provided by iterate2
# ------------------------------------------
#   ITERATE_TRIAL_NUMBER   integer trial ID
#   ITERATE_TRIAL_CMD      full shell command to execute (includes cd, venv activation, etc.)
#   ITERATE_OUT_FILE       path where trial stdout must be written
#   ITERATE_ERR_FILE       path where trial stderr must be written
#
# WLM configuration (from the ``wlm:`` section in the HPO YAML)
# -------------------------------------------------------------
# All keys from the ``wlm:`` block are available as
# ``ITERATE_WLM_<KEY_UPPERCASE>`` (hyphens → underscores).
# Recognised keys (with defaults):
#
#   gpu-count          (ITERATE_WLM_GPU_COUNT)          default: 1
#   cpu-count          (ITERATE_WLM_CPU_COUNT)          default: 4
#   mem-gb             (ITERATE_WLM_MEM_GB)             default: 32
#   lsf-gpu-config     (ITERATE_WLM_LSF_GPU_CONFIG)     default: auto from gpu-count
#   queue              (ITERATE_WLM_QUEUE)              optional
#
# Usage in HPO YAML
# -----------------
#   wlm:
#     gpu-count: 1
#     cpu-count: 16
#     mem-gb: 32
#     lsf-gpu-config: "num=1:mode=exclusive_process:mps=no:gmodel=NVIDIAA100_SXM4_80GB"
#     # queue: normal       # uncomment to specify an LSF queue
#
# Usage in the launch script
# --------------------------
#   iterate2 \
#     --wlm-plugin "$(dirname "$0")/wlm_plugins/lsf_plugin.sh" \
#     --hpo-yaml   my_hpo.yaml \
#     ...
#
# Exit code
# ---------
# The script exits 0 on success, non-zero on failure.
# iterate2 marks the Optuna trial as FAILED when exit code != 0.
# =============================================================================

set -euo pipefail

# ── Read iterate2 env vars ────────────────────────────────────────────────────
TRIAL_NUMBER="${ITERATE_TRIAL_NUMBER:?ITERATE_TRIAL_NUMBER not set}"
TRIAL_CMD="${ITERATE_TRIAL_CMD:?ITERATE_TRIAL_CMD not set}"
OUT_FILE="${ITERATE_OUT_FILE:?ITERATE_OUT_FILE not set}"
ERR_FILE="${ITERATE_ERR_FILE:?ITERATE_ERR_FILE not set}"

# ── WLM config from YAML wlm: section ────────────────────────────────────────
GPU_COUNT="${ITERATE_WLM_GPU_COUNT:-1}"
CPU_COUNT="${ITERATE_WLM_CPU_COUNT:-4}"
MEM_GB="${ITERATE_WLM_MEM_GB:-32}"
QUEUE="${ITERATE_WLM_QUEUE:-}"
# If a custom GPU resource string is provided use it; otherwise derive from gpu-count.
if [[ -n "${ITERATE_WLM_LSF_GPU_CONFIG:-}" ]]; then
    GPU_FRAGMENT="-gpu \"${ITERATE_WLM_LSF_GPU_CONFIG}\""
elif [[ "${GPU_COUNT}" -gt 0 ]]; then
    GPU_FRAGMENT="-gpu num=${GPU_COUNT}"
else
    GPU_FRAGMENT=""
fi

QUEUE_FRAGMENT=""
if [[ -n "${QUEUE}" ]]; then
    QUEUE_FRAGMENT="-q ${QUEUE}"
fi

# ── Build bsub command ────────────────────────────────────────────────────────
# -K  : block until the job finishes (iterate2 calls this per-trial in a thread)
# -o/-e: LSF writes stdout/stderr directly to these files
BSUB_CMD="bsub ${GPU_FRAGMENT} ${QUEUE_FRAGMENT} \
  -K \
  -o ${OUT_FILE} \
  -e ${ERR_FILE} \
  -R \"rusage[ngpus=${GPU_COUNT}, cpu=${CPU_COUNT}, mem=${MEM_GB}GB]\" \
  -J hpo_trial_${TRIAL_NUMBER} \
  \"${TRIAL_CMD}\""

echo "[lsf_plugin] Trial ${TRIAL_NUMBER}: submitting → ${BSUB_CMD}"
eval "${BSUB_CMD}"
echo "[lsf_plugin] Trial ${TRIAL_NUMBER}: job finished"
