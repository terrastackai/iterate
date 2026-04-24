#!/usr/bin/env python3

import argparse
import json
import logging
import os
import subprocess
import sys
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, Literal, List

import optuna
import yaml
from terratorch_iterate.iterate2.plugin.coordinator import load_builtin_plugins, resolve_storage

# Load built-in coordinator plugins (sqlite, journalfs, postgresql)
load_builtin_plugins()

logger = logging.getLogger("iterate2")

# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic Optuna HPO launcher with Multi-Metric support"
    )

    # ------------------------
    # Execution config
    # ------------------------
    parser.add_argument("--script", required=True, help="Training script to execute")
    parser.add_argument("--root-dir", default=None, help="Root dir (derived if omitted)")
    parser.add_argument("--venv", default=".venv", help="Virtualenv dir (shortcut for source <venv>/bin/activate)")
    parser.add_argument(
        "--pre-run-commands",
        default=None,
        help=(
            "Shell commands to run before the training script, joined with ' && '. "
            "Useful for sourcing bashrc, activating conda/mamba envs, loading modules, etc. "
            "Example: 'source ~/.bashrc && micromamba activate gridfm'. "
            "When set, --venv is ignored."
        ),
    )
    parser.add_argument("--interpreter", default="python", help="Interpreter to use")
    parser.add_argument("--param-setter", type=str, default=None)
    parser.add_argument(
        "--wlm-plugin",
        type=str,
        default=None,
        help=(
            "Path to an executable WLM plugin script that submits/runs each trial. "
            "When omitted, trials run locally (equivalent to the old --wlm none). "
            "The plugin receives trial information as environment variables: "
            "ITERATE_TRIAL_NUMBER, ITERATE_TRIAL_CMD, ITERATE_OUT_FILE, ITERATE_ERR_FILE, "
            "and ITERATE_WLM_<KEY> for every key in the 'wlm:' YAML section. "
            "Exit 0 to signal success; any other exit code marks the trial as failed. "
            "See examples/wlm_plugins/ for LSF and Vela reference implementations."
        ),
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="Number of trials to run in parallel (default: 1 = sequential). "
             "Each parallel trial runs in its own thread. "
             "For SQLite storage, values >4 may cause locking contention; "
             "consider PostgreSQL for high parallelism.",
    )
    parser.add_argument(
        "--no-underscore-to-hyphen",
        dest="underscore_to_hyphen",
        action="store_false",
        default=True,
        help="Do not convert underscores to hyphens in arg names (default: convert)",
    )

    # ------------------------
    # Optuna config
    # ------------------------
    parser.add_argument("--optuna-study-name", required=True)
    parser.add_argument("--optuna-db-path", required=True)
    parser.add_argument("--optuna-n-trials", type=int, default=100)

    # ------------------------
    # HPO space
    # ------------------------
    parser.add_argument("--hpo-json", type=str, default=None)
    parser.add_argument("--hpo-yaml", type=str, default=None)
    parser.add_argument("--static-args-json", type=str, default=None)
    parser.add_argument("--static-args-yaml", type=str, default=None)

    # ------------------------
    # Metric extraction (Supports comma-separated list)
    # ------------------------
    parser.add_argument(
        "--metrics",
        default="score_combined",
        help="Comma-separated metric names to extract (e.g. score_linear_acc,score_modality_leak,score_combined)",
    )

    # ------------------------
    # Logging
    # ------------------------
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )

    return parser.parse_args()


# ============================================================
# HELPERS & COMMAND BUILDERS
# ============================================================

def resolve_paths(script: str, root_dir: Optional[str]):
    if root_dir is None: root_dir = '.'
    resolved = Path(root_dir).resolve()
    logger.debug("Resolved root_dir '%s' → '%s'", root_dir, resolved)
    return script, resolved

def build_shell_command(interpreter, root_dir, script_path, venv, script_args, param_setter, underscore_to_hyphen=True, pre_run_commands=None):
    parts = [f"cd {root_dir}"]
    if pre_run_commands:
        parts.append(pre_run_commands)
        logger.debug("Pre-run commands: %s", pre_run_commands)
    elif venv:
        parts.append(f"source {venv}/bin/activate")
        logger.debug("Activating venv: %s", venv)
    arg_list = [f"{interpreter} {script_path}"]
    for key, value in script_args.items():
        arg_name = key.replace("_", "-") if underscore_to_hyphen else key
        if value is None:
            logger.debug("Skipping arg '%s': value is None (flag omitted)", key)
            continue
        if param_setter:
            if isinstance(value, bool):
                if value:
                    arg_list.append(f"--{param_setter} {key}")
                    logger.debug("Setter flag: --%s %s (store_true)", param_setter, key)
                else:
                    logger.debug("Skipping flag '%s': False → omitted", key)
            else:
                arg_list.append(f"--{param_setter} {key} {value}")
                logger.debug("Setter arg: --%s %s %s", param_setter, key, value)
        else:
            if isinstance(value, bool):
                if value:
                    arg_list.append(f"--{arg_name}")
                    logger.debug("Flag present: --%s", arg_name)
                else:
                    logger.debug("Skipping flag '--%s': False → omitted", arg_name)
            else:
                arg_list.append(f"--{arg_name} {value}")
                logger.debug("Arg: --%s %s", arg_name, value)
    cmd = " && ".join(parts + [" ".join(arg_list)])
    logger.debug("Shell command: %s", cmd)
    return cmd


def build_container_command(interpreter: str, script_path: str, script_args: dict, param_setter: Optional[str], underscore_to_hyphen: bool = True) -> str:
    """Build a bare CLI invocation suitable for running inside a container.

    Unlike :func:`build_shell_command` this function does **not** prepend
    ``cd`` or ``source venv`` – those are not needed (or available) inside an
    already-running container image.
    """
    prefix = f"{interpreter} " if interpreter else ""
    arg_list = [f"{prefix}{script_path}".strip()]
    for key, value in script_args.items():
        arg_name = key.replace("_", "-") if underscore_to_hyphen else key
        if value is None:
            logger.debug("Container cmd: skipping '%s' (None)", key)
            continue
        if param_setter:
            if isinstance(value, bool):
                if value:
                    arg_list.append(f"--{param_setter} {key}")
                else:
                    pass  # omit
            else:
                arg_list.append(f"--{param_setter} {key} {value}")
        else:
            if isinstance(value, bool):
                if value:
                    arg_list.append(f"--{arg_name}")
                # else omit
            else:
                arg_list.append(f"--{arg_name} {value}")
    cmd = " ".join(arg_list)
    logger.debug("Container command: %s", cmd)
    return cmd


_print_lock = threading.Lock()

def _stream_pipe(pipe, dest_file, trial_id: int, stream_name: str, dest_stream):
    """Read lines from *pipe*, write to *dest_file* and print prefixed to *dest_stream*."""
    prefix = f"[trial-{trial_id}]"
    with open(dest_file, "w", encoding="utf-8", errors="replace") as fh:
        for raw in pipe:
            line = raw.decode("utf-8", errors="replace")
            fh.write(line)
            fh.flush()
            with _print_lock:
                dest_stream.write(f"{prefix} {line}")
                dest_stream.flush()

def run_and_stream(launcher_cmd: str, trial_id: int, out_file: str, err_file: str):
    """Run *launcher_cmd* locally, capturing stdout/stderr to files and streaming
    every line to the console prefixed with ``[trial-N]``."""
    logger.debug("Trial %d: run_and_stream cmd=%s", trial_id, launcher_cmd)
    proc = subprocess.Popen(
        launcher_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    t_out = threading.Thread(
        target=_stream_pipe,
        args=(proc.stdout, out_file, trial_id, "stdout", sys.stdout),
        daemon=True,
    )
    t_err = threading.Thread(
        target=_stream_pipe,
        args=(proc.stderr, err_file, trial_id, "stderr", sys.stderr),
        daemon=True,
    )
    t_out.start()
    t_err.start()
    proc.wait()
    t_out.join()
    t_err.join()

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, launcher_cmd)

# ============================================================
# MULTI-METRIC EXTRACTION
# ============================================================

def extract_metrics_from_log(path: str, metric_names: List[str], err_path: Optional[str] = None) -> List[float]:
    """Extract metric values from a log file.

    Each entry in *metric_names* is either a plain name (uses the **last**
    match) or ``name#N`` to select the **N-th occurrence** (0-based). This
    lets you disambiguate scripts that print the same metric key multiple
    times, e.g.::

        metrics:
          - "Samples/sec#0"   # DataLoader throughput (first occurrence)
          - "Samples/sec#1"   # Training throughput   (second occurrence)
          - "Samples/sec#2"   # Inference throughput  (third occurrence)
          - GFLOPS
    """
    logger.debug("Extracting metrics %s from '%s'", metric_names, path)
    results = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    logger.debug("Log file '%s': %d characters read", path, len(text))
    # Also read stderr — Lightning/rich writes test result tables there
    if err_path:
        try:
            with open(err_path, "r", encoding="utf-8", errors="ignore") as f:
                err_text = f.read()
            logger.debug("Err file '%s': %d characters read", err_path, len(err_text))
            text = text + "\n" + err_text
        except FileNotFoundError:
            logger.debug("Err file '%s' not found, skipping", err_path)

    for metric in metric_names:
        # Support  name#N  syntax for Nth-occurrence selection (0-based)
        occurrence: Optional[int] = None
        bare_metric = metric
        idx_match = re.fullmatch(r'(.+)#(\d+)', metric)
        if idx_match:
            bare_metric = idx_match.group(1)
            occurrence = int(idx_match.group(2))

        # Matches: key: value | key=value | [performance] key : value | Lightning table │ key │ value │
        pattern = re.compile(
            rf"(?:\[\w+\]\s*)?{re.escape(bare_metric)}\s*(?:[:=│])\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        )
        matches = pattern.findall(text)
        if not matches:
            logger.warning("Metric '%s' not found in '%s' — defaulting to 0.0", metric, path)
            results.append(0.0)
        elif occurrence is not None:
            if occurrence >= len(matches):
                logger.warning(
                    "Metric '%s' occurrence #%d requested but only %d match(es) found — defaulting to 0.0",
                    metric, occurrence, len(matches),
                )
                results.append(0.0)
            else:
                value = float(matches[occurrence])
                logger.debug("Metric '%s': using occurrence #%d = %s", metric, occurrence, value)
                results.append(value)
        else:
            value = float(matches[-1])
            logger.debug("Metric '%s': found %d match(es), using last value %s", metric, len(matches), value)
            results.append(value)
    return results

# ============================================================
# MAIN
# ============================================================

def load_hpo_space(args):
    data = {}
    if args.hpo_json:
        logger.debug("Loading HPO space from JSON string")
        data = json.loads(args.hpo_json)
    elif args.hpo_yaml:
        logger.debug("Loading HPO space from YAML file: %s", args.hpo_yaml)
        with open(args.hpo_yaml, "r") as f: data = yaml.safe_load(f)
    space = data.get("hpo", {})
    logger.info("HPO space loaded: %d parameter(s): %s", len(space), list(space.keys()))
    return space

def load_metrics_from_yaml(args):
    """Return metrics list from YAML 'metrics:' key, or None if not present."""
    data = {}
    if args.hpo_json:
        data = json.loads(args.hpo_json)
    elif args.hpo_yaml:
        with open(args.hpo_yaml, "r") as f: data = yaml.safe_load(f)
    elif args.static_args_yaml:
        with open(args.static_args_yaml, "r") as f: data = yaml.safe_load(f)
    metrics = data.get("metrics", None)
    if metrics is None:
        return None
    if isinstance(metrics, list):
        return [m.strip() for m in metrics]
    return [m.strip() for m in str(metrics).split(",")]

def load_static_args(args):
    data = {}
    if args.static_args_json:
        logger.debug("Loading static args from JSON string")
        data = json.loads(args.static_args_json)
    elif args.static_args_yaml:
        logger.debug("Loading static args from YAML file: %s", args.static_args_yaml)
        with open(args.static_args_yaml, "r") as f: data = yaml.safe_load(f)
    elif args.hpo_yaml:
        logger.debug("Loading static args from HPO YAML file: %s", args.hpo_yaml)
        with open(args.hpo_yaml, "r") as f: data = yaml.safe_load(f)
    static = data.get("static", data if data else {})
    logger.info("Static args loaded: %d key(s): %s", len(static), list(static.keys()))
    return static

def load_wlm_config(args) -> dict:
    """Load the optional ``wlm:`` section from the HPO YAML.

    Returns a dict that is forwarded to the WLM plugin as
    ``ITERATE_WLM_<KEY>`` environment variables.  Keys are taken verbatim
    from the YAML (e.g. ``gpu-count``, ``mem-gb``) so plugin scripts can
    use familiar names.
    """
    data = {}
    if args.hpo_yaml:
        with open(args.hpo_yaml, "r") as f:
            data = yaml.safe_load(f) or {}
    elif args.static_args_yaml:
        with open(args.static_args_yaml, "r") as f:
            data = yaml.safe_load(f) or {}
    wlm_cfg = data.get("wlm", {})
    logger.info("WLM config from YAML: %s", wlm_cfg)
    return wlm_cfg


def suggest_from_spec(trial, name, spec):
    t = spec["type"]
    if t == "float":
        val = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=spec.get("log", False))
    elif t == "int":
        val = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), log=spec.get("log", False))
    elif t == "categorical":
        val = trial.suggest_categorical(name, spec["choices"])
    elif t == "flag":
        val = trial.suggest_categorical(name, [True, False])
    elif t == "group":
        val = trial.suggest_categorical(name, list(spec["choices"].keys()))
    else:
        raise ValueError(f"Unknown param type: {t}")
    logger.debug("Suggested '%s' (%s) = %r", name, t, val)
    return val

def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Suppress noisy optuna INFO logs unless user asked for DEBUG
    logging.getLogger("optuna").setLevel(
        logging.WARNING if args.log_level == "INFO" else getattr(logging, args.log_level)
    )

    logger.info("iterate2 starting")
    logger.info("Log level: %s", args.log_level)
    logger.info(
        "WLM plugin: %s | interpreter: %s | script: %s",
        args.wlm_plugin or "(local)", args.interpreter, args.script,
    )
    logger.info("Optuna study: '%s' | db: %s | n_trials: %d", args.optuna_study_name, args.optuna_db_path, args.optuna_n_trials)

    hpo_space = load_hpo_space(args)
    static_args = load_static_args(args)
    wlm_config = load_wlm_config(args)
    yaml_metrics = load_metrics_from_yaml(args)
    metric_list = yaml_metrics if yaml_metrics is not None else [m.strip() for m in args.metrics.split(",")]
    logger.info("Optimising metrics: %s (source: %s)", metric_list, "yaml" if yaml_metrics else "cli")

    script_path, root_dir = resolve_paths(args.script, args.root_dir)

    def objective(trial):
        script_args = static_args.copy()
        for name, spec in hpo_space.items():
            val = suggest_from_spec(trial, name, spec)
            if spec["type"] == "group":
                # Expand the chosen group's key→value pairs directly into script_args
                script_args.update(spec["choices"][val])
            else:
                script_args[name] = val

        # gpu_num can appear in hpo or static space; pull it out and add to
        # wlm_config so plugins can use ITERATE_WLM_GPU_NUM for resource allocation.
        trial_wlm_config = dict(wlm_config)
        if "gpu_num" in script_args:
            trial_wlm_config.setdefault("gpu-count", script_args.pop("gpu_num"))

        logger.info("Trial %d: sampled parameters: %s", trial.number, script_args)
        logger.debug("Trial %d: effective WLM config: %s", trial.number, trial_wlm_config)

        out_file = f"trial_{trial.number}.out"
        err_file = f"trial_{trial.number}.err"
        logger.debug("Trial %d: stdout → %s | stderr → %s", trial.number, out_file, err_file)

        # Build the shell command that the plugin (or local runner) shall execute.
        shell_cmd = build_shell_command(
            args.interpreter, root_dir, script_path, args.venv,
            script_args, args.param_setter, args.underscore_to_hyphen,
            pre_run_commands=args.pre_run_commands,
        )
        # Container-safe command: no cd / source – for plugins running inside
        # an already-configured container image (e.g. Vela/OpenShift).
        container_cmd = build_container_command(
            args.interpreter, script_path, script_args,
            args.param_setter, args.underscore_to_hyphen,
        )

        if args.wlm_plugin:
            # ── User-provided WLM plugin ───────────────────────────────────
            # The plugin is responsible for submitting the trial, waiting for
            # completion, and writing stdout/stderr to out_file/err_file.
            # It signals success via exit code 0; any other value fails the trial.
            env = os.environ.copy()
            env["ITERATE_TRIAL_NUMBER"]        = str(trial.number)
            env["ITERATE_TRIAL_CMD"]           = shell_cmd
            env["ITERATE_TRIAL_CONTAINER_CMD"] = container_cmd
            env["ITERATE_OUT_FILE"]            = out_file
            env["ITERATE_ERR_FILE"]            = err_file
            for k, v in trial_wlm_config.items():
                env_key = "ITERATE_WLM_" + k.upper().replace("-", "_").replace(" ", "_")
                env[env_key] = str(v)
            logger.info("Trial %d: invoking WLM plugin %s", trial.number, args.wlm_plugin)
            result = subprocess.run(
                [args.wlm_plugin],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Stream plugin output to console so operators see submission status.
            plugin_out = result.stdout.decode("utf-8", errors="replace")
            plugin_err = result.stderr.decode("utf-8", errors="replace")
            prefix = f"[trial-{trial.number}][plugin]"
            for line in plugin_out.splitlines():
                sys.stdout.write(f"{prefix} {line}\n")
            for line in plugin_err.splitlines():
                sys.stderr.write(f"{prefix} {line}\n")
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, args.wlm_plugin)
        else:
            # ── Local execution (no WLM plugin) ───────────────────────────
            launcher_cmd = f'bash -c "{shell_cmd}"'
            logger.info("Trial %d: running locally → %s", trial.number, launcher_cmd)
            run_and_stream(launcher_cmd, trial.number, out_file, err_file)

        logger.info("Trial %d: job finished", trial.number)

        values = extract_metrics_from_log(out_file, metric_list, err_path=err_file)
        logger.info("Trial %d: results %s", trial.number, dict(zip(metric_list, values)))

        return tuple(values)

    # Multi-objective direction
    directions = ["maximize"] * len(metric_list)
    logger.info("Creating Optuna study (directions: %s)", directions)

    storage = resolve_storage(args.optuna_db_path)
    logger.debug("Optuna storage: %s", storage)

    study = optuna.create_study(
        study_name=args.optuna_study_name,
        storage=storage,
        directions=directions,
        load_if_exists=True,
    )
    logger.info("Study '%s' ready (existing trials: %d)", args.optuna_study_name, len(study.trials))

    # ── Re-queue failed trials (25 % retry / 75 % new) ────────────────────
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    n_total = args.optuna_n_trials
    if failed_trials:
        n_retry = max(1, round(0.25 * n_total))
        n_retry = min(n_retry, len(failed_trials))   # can't retry more than we have
        n_new   = n_total - n_retry
        # enqueue the most-recent failed trials first
        trials_to_retry = failed_trials[-n_retry:]
        logger.info(
            "Found %d failed trial(s). Re-queuing %d (25%%) and running %d new (75%%).",
            len(failed_trials), n_retry, n_new,
        )
        for ft in trials_to_retry:
            if ft.params:               # skip trials that had no params at all
                study.enqueue_trial(ft.params)
                logger.info("  Enqueued params from failed trial %d: %s", ft.number, ft.params)
            else:
                logger.info("  Skipped failed trial %d (no params recorded).", ft.number)
        # adjust total so we run exactly n_new *additional* new trials on top
        n_total = n_new + n_retry       # enqueued slots count toward n_trials
    else:
        logger.info("No failed trials found – running %d fresh trials.", n_total)
    # ── end retry logic ───────────────────────────────────────────────────

    logger.info("Parallelism: %d worker(s)", args.parallelism)
    study.optimize(
        objective,
        n_trials=n_total,
        n_jobs=args.parallelism,
        catch=(Exception,),   # mark trial as FAILED and continue; never crash the study
    )

    logger.info("=" * 60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("Pareto Front Trials: %d", len(study.best_trials))
    for t in study.best_trials:
        logger.info("  Trial %d: Values=%s  Params=%s", t.number, t.values, t.params)

if __name__ == "__main__":
    main()