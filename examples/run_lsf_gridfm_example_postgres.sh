#!/usr/bin/env bash
# =============================================================================
# Example: iterate2 with the LSF WLM plugin and PostgreSQL coordinator
#
# Each Optuna trial is submitted as an LSF job by the plugin script
# examples/wlm_plugins/lsf_plugin.sh.  That script reads all LSF
# resource settings from ITERATE_WLM_* env vars which are populated from
# the ``wlm:`` section in the HPO YAML (configs/gridfm_graphkit_hpo.yaml).
#
# What changed vs the old --wlm lsf approach
# ------------------------------------------
# * --wlm lsf, --gpu-count, --cpu-count, --mem-gb, --lsf-gpu-config-string
#   are gone; all of these now live in the  wlm:  block of the HPO YAML.
# * --wlm-plugin points to the LSF plugin script (user-owned).
# * The plugin can be customised freely without touching iterate2 itself.
#
# Prerequisites
# -------------
#   * LSF bsub/bjobs available on PATH
#   * gridfm-graphkit installed in the venv below
#   * configs/gridfm_graphkit_hpo.yaml present  (with wlm: section filled in)
#   * psycopg2-binary installed:  pip install 'terratorch-iterate[postgresql]'
#   * POSTGRES_URL set (or hard-code it in --optuna-db-path below)
#
# PostgreSQL coordinator
# ----------------------
# Using PostgreSQL instead of SQLite / JournalFS is the recommended backend
# for high-parallelism HPO on a cluster.
#
#   export POSTGRES_URL="postgresql://user:password@host:5432/optuna_studies"
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Required: PostgreSQL connection URL
# ---------------------------------------------------------------------------
: "${POSTGRES_URL:?Please set POSTGRES_URL=postgresql://user:password@host:port/dbname}"

# ---------------------------------------------------------------------------
# Customisable paths – override via environment variables
# ---------------------------------------------------------------------------
GRIDFM_ROOT="${GRIDFM_ROOT:-/dccstor/terratorch/users/rkie/gitco/gridfm-graphkit}"
GRIDFM_VENV="${GRIDFM_VENV:-/u/rkie/venvs/venv_gridfm-graphkit}"
CUDA_BASE="${CUDA_BASE:-/opt/share/cuda-12.8.1}"
DATA_PATH="${DATA_PATH:-/u/rkie/}"

# ---------------------------------------------------------------------------
# Pre-run commands executed inside every bsub job before the training script.
# ---------------------------------------------------------------------------
PRE_RUN="\
export PATH='${CUDA_BASE}/bin:\$PATH' && \
export CUDA_HOME='${CUDA_BASE}' && \
export LD_LIBRARY_PATH='${CUDA_BASE}/lib64:\$LD_LIBRARY_PATH' && \
cd '${GRIDFM_ROOT}' && \
source '${GRIDFM_VENV}/bin/activate'"

# ---------------------------------------------------------------------------
# Launch iterate2
#
# Resource settings (gpu-count, cpu-count, mem-gb, lsf-gpu-config) are
# defined in the  wlm:  section of gridfm_graphkit_hpo.yaml – not here.
# ---------------------------------------------------------------------------
iterate2 \
  --script              "gridfm_graphkit train"                         \
  --interpreter         ""                                              \
  --root-dir            "${GRIDFM_ROOT}"                                \
  --wlm-plugin          "${SCRIPT_DIR}/wlm_plugins/lsf_plugin.sh"      \
  --pre-run-commands    "${PRE_RUN}"                                    \
  --no-underscore-to-hyphen                                             \
  --optuna-study-name   gridfm_lsf_postgres_hpo                         \
  --optuna-db-path      "${POSTGRES_URL}"                               \
  --parallelism         4                                               \
  --optuna-n-trials     20                                              \
  --hpo-yaml            "${REPO_ROOT}/configs/gridfm_graphkit_hpo.yaml"
