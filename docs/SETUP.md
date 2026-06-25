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

The helper uses `mamba` when available and falls back to `conda`.
The canonical environment for project checks and diagnostics is `moe`.

## Configure Resource Paths

Copy `.env.example` to `.env`, edit the paths for your server, and source it:

```bash
cp .env.example .env
source .env
```

All large files must be kept outside git and controlled through these
environment variables:

- `MRB_DATA_ROOT`
- `MRB_MODEL_ROOT`
- `MRB_CACHE_ROOT`
- `MRB_OUTPUT_ROOT`
- `HF_HOME`
- `HF_HUB_CACHE`
- `HF_DATASETS_CACHE`
- `TORCH_HOME`

## Optional Pip Mirror

If the server has slow package downloads, configure pip outside the repository:

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

If the server requires a proxy, enable it in the shell before creating the
environment. Do not commit proxy settings or credentials.

## Checks

```bash
conda run -n moe python scripts/check_env.py
conda run -n moe python scripts/download_datasets.py --dry-run
conda run -n moe python scripts/download_models.py --dry-run
conda run -n moe python scripts/verify_resources.py
conda run -n moe python scripts/smoke_check.py --quick
conda run -n moe pytest tests/
```
