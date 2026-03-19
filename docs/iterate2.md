# iterate2 – HPO Launcher

`iterate2` is a generic Optuna-based hyperparameter optimisation (HPO) launcher with a pluggable workload-manager backend. It submits one trial per Optuna suggestion, waits for the job to finish, extracts a scalar metric from the job's log file, and returns it to Optuna.

## Quick start

```sh
iterate2 \
  --script train.py \
  --wlm lsf \
  --gpu-count 1 \
  --cpu-count 20 \
  --mem-gb 512 \
  --optuna-study-name my_study \
  --optuna-db-path sqlite:///hpo.db \
  --optuna-n-trials 50 \
  --hpo-yaml hpo_space.yaml
```

## CLI reference

### Execution options

| Option | Default | Description |
|---|---|---|
| `--script` | *(required)* | Training script to execute |
| `--root-dir` | `.` | Working directory; derived from `--script` if omitted |
| `--venv` | `.venv` | Virtual-environment directory to activate. Set to empty string to disable |
| `--interpreter` | `python` | Python interpreter to invoke |
| `--param-setter` | `None` | Use setter-style argument passing (see [Setter-style arguments](#setter-style-arguments)) |
| `--wlm` | `none` | Workload manager: `lsf`, `slurm`, `openshift`, or `none` |
| `--gpu-count` | `1` | Number of GPUs per trial |
| `--cpu-count` | `4` | Number of CPUs per trial |
| `--mem-gb` | `128` | Memory (GB) per trial |
| `--lsf-gpu-config-string` | `None` | Optional verbatim LSF `-gpu` option string (see [GPU configuration](#gpu-configuration-on-lsf)) |

### Optuna options

| Option | Default | Description |
|---|---|---|
| `--optuna-study-name` | *(required)* | Name of the Optuna study |
| `--optuna-db-path` | *(required)* | Storage URL for the Optuna database, e.g. `sqlite:///hpo.db` |
| `--optuna-n-trials` | `100` | Number of trials to run |

### HPO search space

Provide either a JSON string or a YAML file that defines the search space under the key `hpo`:

| Option | Description |
|---|---|
| `--hpo-json` | HPO search space as a JSON string |
| `--hpo-yaml` | Path to a YAML file containing the search space (and optionally static args) |

#### Search space YAML format

```yaml
hpo:
  learning_rate:
    type: float
    low: 1e-5
    high: 1e-2
    log: true
  batch_size:
    type: categorical
    choices: [16, 32, 64]
  encoder_depth:
    type: int
    low: 2
    high: 6

static:
  max_epochs: 50
  dataset_path: /data/my_dataset
```

Supported parameter types: `float`, `int`, `categorical`.

### Static arguments

Arguments passed unchanged to every trial. Can be supplied inline or via file:

| Option | Description |
|---|---|
| `--static-args-json` | Static arguments as a JSON string |
| `--static-args-yaml` | Path to a YAML file with static arguments |

If neither is provided, `iterate2` falls back to the `static` section of `--hpo-yaml`.

### Metric extraction

| Option | Default | Description |
|---|---|
| `--metric` | `val/F1_Score` | Metric name to extract from the trial's stdout log |

The last occurrence of the pattern `<metric_name>: <value>` or `<metric_name>= <value>` is used.

---

## Setter-style arguments

Some scripts (e.g. those using [Hydra](https://hydra.cc/) overrides or custom key-value CLIs) do not accept named flags like `--learning-rate 0.001`. Instead they expect:

```sh
--set learning_rate 0.001 --set batch_size 32
```

Pass `--param-setter set` (or whatever flag name the target script uses) to switch `iterate2` to this style:

```sh
iterate2 --param-setter set ...
```

| Mode | Generated argument style |
|---|---|
| default (`--param-setter` omitted) | `--learning-rate 0.001 --batch-size 32` |
| `--param-setter set` | `--set learning_rate 0.001 --set batch_size 32` |

!!! note
    In setter style, boolean parameters are passed explicitly as `--set flag true` / `--set flag false` rather than as bare flags (`--flag`), since there is no named flag to toggle.

---

## GPU configuration on LSF

When `--wlm lsf` is selected, `iterate2` constructs a `bsub` command for each trial.

### Default behaviour

| `--gpu-count` | Generated fragment |
|---|---|
| `> 0` (default `1`) | `-gpu num=<N>` |
| `0` | *(no `-gpu` flag, CPU-only job)* |

### `--lsf-gpu-config-string`

For advanced LSF GPU scheduling you can supply the full value of the `-gpu` option as a string. When set, it **completely replaces** the auto-generated `-gpu num=<N>` fragment.

```sh
iterate2 \
  --wlm lsf \
  --lsf-gpu-config-string "num=1:mode=exclusive_process:mps=yes:gmodel=NVIDIAA100_SXM4_80GB" \
  --cpu-count 20 \
  --mem-gb 512 \
  ...
```

This produces a `bsub` submission resembling:

```sh
bsub -n 20 -R "span[hosts=1]" \
     -gpu "num=1:mode=exclusive_process:mps=yes:gmodel=NVIDIAA100_SXM4_80GB" \
     -M 512G -J hpo_trial_0 \
     "cd /my/root && source .venv/bin/activate && python train.py ..."
```

!!! note
    `--gpu-count` is still used for the `rusage` memory/CPU reservation string even when `--lsf-gpu-config-string` is set. Set it to match the `num=` value in your GPU string.

!!! tip
    Use exclusive process mode (`mode=exclusive_process`) together with MPS (`mps=yes`) to share a single A100 across multiple MPS clients while still pinning the job to one physical GPU.

---

## Workload managers

### LSF

`iterate2` waits for each `bsub` job synchronously (`-K` flag). Output is written to `trial_<N>.out` / `trial_<N>.err` in the working directory.

### Slurm

Uses `srun` with `--gres=gpu:<N>`, `--cpus-per-task`, and `--mem` flags.

### none

Runs the command directly in a local shell, redirecting stdout/stderr to `trial_<N>.out` / `trial_<N>.err`.

### openshift

Not yet implemented.

---

## Example HPO YAML

```yaml
# hpo_space.yaml
hpo:
  learning_rate:
    type: float
    low: 1e-5
    high: 1e-2
    log: true
  weight_decay:
    type: float
    low: 1e-6
    high: 1e-3
    log: true

static:
  max_epochs: 30
  config: configs/my_model.yaml
```

Launch with:

```sh
iterate2 \
  --script terratorch_iterate/main.py \
  --wlm lsf \
  --lsf-gpu-config-string "num=1:mode=exclusive_process:mps=yes:gmodel=NVIDIAA100_SXM4_80GB" \
  --cpu-count 20 \
  --mem-gb 512 \
  --optuna-study-name geobench_hpo \
  --optuna-db-path sqlite:///geobench_hpo.db \
  --optuna-n-trials 40 \
  --hpo-yaml hpo_space.yaml \
  --metric "val/F1_Score"
```
