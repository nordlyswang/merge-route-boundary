# merge-route-boundary

Conda-only infrastructure for studying the boundary between continual learning,
model merging, and routing. The current phase only sets up environments,
resource manifests, download helpers, verification, and smoke checks.

Docker, devcontainers, training loops, model merging, MoE routing, adapter/LoRA
training, and experiment logic are intentionally out of scope for this phase.

## Setup

```bash
bash scripts/create_conda_env.sh cuda
conda activate moe
cp .env.example .env
source .env
```

For CPU-only servers:

```bash
bash scripts/create_conda_env.sh cpu
conda activate moe-cpu
```

## Checks

```bash
conda run -n moe python scripts/check_env.py
conda run -n moe python scripts/download_datasets.py --dry-run
conda run -n moe python scripts/download_models.py --dry-run
conda run -n moe python scripts/verify_resources.py
conda run -n moe python scripts/smoke_check.py --quick
conda run -n moe pytest tests/
```

See `docs/SETUP.md` and `docs/RESOURCE_POLICY.md` for details.
