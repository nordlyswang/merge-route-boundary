# merge-route-boundary

## Project Goal

This repository studies the boundary between continual learning, model merging,
and MoE-style routing. The working question is when tasks should share merged
representations and when they should remain separated behind a route.

## Current Phase

The repository currently contains infrastructure and diagnostics layers:

- Conda-only environment and resource management.
- Dataset registry and deterministic task streams.
- Frozen feature bank extraction, storage, and verification.
- Boundary diagnostics v0 over frozen feature banks.

The project has not entered these stages yet:

- Adapter or LoRA training.
- Deep model training.
- Model merging.
- MoE routing.
- Feature-level merge-vs-route baselines.
- Phase diagrams.

## Server Layout

Large resources live outside git and are controlled by environment variables.

- `MRB_DATA_ROOT=/root/rivermind-data/datasets`
- `MRB_FEATURE_ROOT=/root/rivermind-data/datasets/_derived/merge-route-boundary/features`
- Project data link: `data -> /root/rivermind-data/datasets`
- Project feature link: `features -> $MRB_FEATURE_ROOT`

Canonical feature bank layout:

```text
MRB_FEATURE_ROOT/<dataset_id>/<split>/<backbone_id>/
```

Legacy feature banks under `<backbone_id>/<dataset_id>/<split>/` are readable
for compatibility, but new outputs should use the canonical layout.

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

## Required Checks

Use the `moe` Conda environment for project validation:

```bash
conda run -n moe pytest -q
conda run -n moe python scripts/inspect_datasets.py --no-status-json
conda run -n moe python scripts/verify_task_streams.py --stream controlled_cifar100_split_10x10 --seed 0 --skip-batch
conda run -n moe python scripts/verify_task_streams.py --stream real_tier0_tier2_small --seed 0 --skip-batch
```

Resource and smoke checks:

```bash
conda run -n moe python scripts/check_env.py
conda run -n moe python scripts/download_datasets.py --dry-run
conda run -n moe python scripts/download_models.py --dry-run
conda run -n moe python scripts/verify_resources.py
conda run -n moe python scripts/smoke_check.py --quick
```

Boundary diagnostics in strict mode fail if any pair fails:

```bash
conda run -n moe python scripts/compute_boundary_diagnostics.py \
  --stream controlled_cifar100_split_10x10 \
  --seed 0 \
  --backbone clip_vit_b32 \
  --split train \
  --strict-features \
  --fail-on-any-failed-pair \
  --overwrite
```

See `docs/SETUP.md`, `docs/RESOURCE_POLICY.md`, and
`docs/PROJECT_STATUS.md` for details.
