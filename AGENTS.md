# AGENTS.md

## Project

This repository studies the boundary between continual learning, model
merging, and MoE-style routing. Current code covers Conda/resource
infrastructure, dataset registry and task streams, frozen feature banks, and
boundary diagnostics v0.

## Current Allowed Scope

- Bug fixes and QA for data, feature, and diagnostics layers.
- Documentation and status alignment.
- Smoke checks and lightweight validation.
- Manifest, split protocol, and CLI quality-gate fixes.

## Still Forbidden Unless Explicitly Requested

- Docker, devcontainer, docker-compose, or container deployment files.
- Deep model training.
- Adapter or LoRA training.
- Model merging.
- MoE routing.
- Feature-level merge-vs-route baselines.
- Phase diagrams.
- Downloading new datasets or extracting full feature banks during QA-only work.

## Resource Rules

- Use Conda/Mamba environment files only.
- Do not commit datasets, model weights, checkpoints, cache files, logs,
  generated artifacts, feature banks, or secrets.
- All large resources must live outside git and be controlled by environment
  variables.
- Canonical feature bank layout is
  `MRB_FEATURE_ROOT/<dataset_id>/<split>/<backbone_id>/`.

## Required Checks

Run project checks through the `moe` Conda environment:

```bash
conda run -n moe pytest -q
conda run -n moe python scripts/inspect_datasets.py --no-status-json
conda run -n moe python scripts/verify_task_streams.py --stream controlled_cifar100_split_10x10 --seed 0 --skip-batch
conda run -n moe python scripts/verify_task_streams.py --stream real_tier0_tier2_small --seed 0 --skip-batch
```

If CUDA is unavailable, CPU smoke checks may pass with a warning.

## Coding Style

- Use type hints for new Python functions.
- Prefer small, testable scripts.
- Use clear warnings when optional dependencies or resources are missing.
- Keep resource manifests declarative.
