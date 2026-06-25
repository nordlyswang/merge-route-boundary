# Setup

This project is Conda-only. Docker and devcontainers are intentionally not
supported.

## Create an Environment

CUDA environment:

```bash
bash scripts/create_conda_env.sh cuda
conda activate moe
```

CPU environment:

```bash
bash scripts/create_conda_env.sh cpu
conda activate moe-cpu
```

The helper uses `mamba` when available and falls back to `conda`. The canonical
environment for project checks and diagnostics is `moe`; do not validate from
`base`.

## Configure Resource Paths

Copy `.env.example` to `.env`, edit paths for the server, and source it:

```bash
cp .env.example .env
source .env
```

Required resource variables:

- `MRB_DATA_ROOT`
- `MRB_PROJECT_DATA`
- `MRB_MODEL_ROOT`
- `MRB_CACHE_ROOT`
- `MRB_OUTPUT_ROOT`
- `MRB_FEATURE_ROOT`
- `HF_HOME`
- `HF_HUB_CACHE`
- `HF_DATASETS_CACHE`
- `TORCH_HOME`

Recommended local links:

```text
data -> /root/rivermind-data/datasets
features -> /root/rivermind-data/datasets/_derived/merge-route-boundary/features
```

Feature banks should use:

```text
MRB_FEATURE_ROOT/<dataset_id>/<split>/<backbone_id>/
```

## Optional Pip Mirror

If the server has slow package downloads, configure pip outside the repository:

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

If the server requires a proxy, enable it in the shell before creating the
environment. Do not commit proxy settings or credentials.

## Checks

```bash
conda run -n moe pytest -q
conda run -n moe python scripts/check_env.py
conda run -n moe python scripts/inspect_datasets.py --no-status-json
conda run -n moe python scripts/verify_task_streams.py --stream controlled_cifar100_split_10x10 --seed 0 --skip-batch
conda run -n moe python scripts/verify_task_streams.py --stream real_tier0_tier2_small --seed 0 --skip-batch
conda run -n moe python scripts/download_datasets.py --dry-run
conda run -n moe python scripts/download_models.py --dry-run
conda run -n moe python scripts/verify_resources.py
conda run -n moe python scripts/smoke_check.py --quick
```
