# AGENTS.md

## Project

This repository studies the boundary between continual learning, model
merging, and MoE-style routing. The first phase is infrastructure only:
Conda environment files, resource manifests, download scripts, verification,
and smoke checks.

## Hard Constraints

- Do not add Docker, devcontainer, docker-compose, or container deployment
  files.
- Use Conda/Mamba environment files only.
- Do not implement training, model merging, routing, adapter logic, or
  experiments in the initial setup phase.
- Do not commit datasets, model weights, checkpoints, cache files, logs, or
  secrets.
- All large resources must live outside git and be controlled by environment
  variables.

## Required Checks

Run these before opening a PR:

```bash
conda run -n moe python scripts/check_env.py
conda run -n moe python scripts/download_datasets.py --dry-run
conda run -n moe python scripts/download_models.py --dry-run
conda run -n moe python scripts/verify_resources.py
conda run -n moe python scripts/smoke_check.py --quick
conda run -n moe pytest tests/
```

If CUDA is unavailable, CPU smoke checks may pass with a warning.

## Coding Style

- Use type hints for new Python functions.
- Prefer small, testable scripts.
- Use clear warnings when optional dependencies or resources are missing.
- Keep resource manifests declarative.
