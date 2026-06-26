#!/usr/bin/env bash
# =============================================================================
# iterate2 WLM plugin  -  CCC (IBM Spectrum LSF)
#
# Invocation contract (set by iterate2 when --wlm-plugin is used):
#
#   ccc_plugin.sh <script> [--<wlm-key> <value>]...
#
# where <script> is the executable passed as --script, and the --<wlm-key>
# flags are the keys of the YAML 'wlm:' section, e.g.
#   --gpu-count 1 --cpu-count 16 --mem-gb 32 \
#   --lsf-gpu-config "num=1:mode=exclusive_process:mps=no"
#
# iterate2 also exports the per-trial env vars (ITERATE_TRIAL_NUMBER,
# ITERATE_OUT_FILE, ITERATE_ERR_FILE, ITERATE_PARAM_*) which bsub forwards
# to the job so the wrapped <script> can read them.
# =============================================================================

set -euo pipefail

SCRIPT="${1:?usage: ccc_plugin.sh <script> [--key value]...}"
shift

# Defaults; overridden by --<key> <value> argv pairs below.
GPU_COUNT=1
CPU_COUNT=4
MEM_GB=16
LSF_GPU_CONFIG=""
QUEUE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-count)       GPU_COUNT="$2";       shift 2 ;;
    --cpu-count)       CPU_COUNT="$2";       shift 2 ;;
    --mem-gb)          MEM_GB="$2";          shift 2 ;;
    --lsf-gpu-config)  LSF_GPU_CONFIG="$2";  shift 2 ;;
    --queue)           QUEUE="$2";           shift 2 ;;
    *) echo "ccc_plugin: ignoring unknown flag $1 $2" >&2; shift 2 ;;
  esac
done

TRIAL_NUMBER="${ITERATE_TRIAL_NUMBER:?ITERATE_TRIAL_NUMBER not set}"
OUT_FILE="${ITERATE_OUT_FILE:?ITERATE_OUT_FILE not set}"
ERR_FILE="${ITERATE_ERR_FILE:?ITERATE_ERR_FILE not set}"

MEM_MB=$(( MEM_GB * 1024 ))
GPU_STRING="${LSF_GPU_CONFIG:-num=${GPU_COUNT}:mode=exclusive_process:mps=no}"

QUEUE_FLAG=()
[[ -n "$QUEUE" ]] && QUEUE_FLAG=(-q "$QUEUE")

GPU_FLAG=()
if [[ "$GPU_COUNT" -gt 0 || -n "$LSF_GPU_CONFIG" ]]; then
  GPU_FLAG=(-gpu "$GPU_STRING")
fi

bsub \
  -K \
  "${GPU_FLAG[@]}" \
  "${QUEUE_FLAG[@]}" \
  -n    "${CPU_COUNT}" \
  -R    "rusage[mem=${MEM_MB}]" \
  -o    "${OUT_FILE}" \
  -e    "${ERR_FILE}" \
  -J    "hpo_trial_${TRIAL_NUMBER}" \
  "${SCRIPT}"

echo "[ccc_plugin] trial ${TRIAL_NUMBER} finished"
