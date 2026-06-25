# Project Status

## Project Goal

`merge-route-boundary` studies the boundary between continual learning, model
merging, and MoE-style routing. The current repository prepares the data,
frozen-feature, and diagnostics infrastructure needed before implementing
feature-level merge-vs-route baselines.

## Current Branch And Phase

- Branch: `codex/0625-dataset-registry`
- Phase: Phase Alignment + QA fixes for data, feature, and diagnostics layers.
- This phase does not add new research methods or train models.

## Completed Layers

- Phase 0: Conda-only environment files, resource manifests, download dry-runs,
  verification scripts, and smoke checks.
- Phase 1: Dataset registry, read-only dataset loaders, deterministic task
  streams, and split manifests.
- Phase 2: Frozen feature backbone manifests, feature extraction, feature bank
  storage, listing, and verification.
- Phase 3: Boundary diagnostics v0 over frozen feature banks, including
  prototype, distance, separability, overlap, matrix export, and CLI quality
  gates.

## Local Server Layout

- `MRB_DATA_ROOT=/root/rivermind-data/datasets`
- `MRB_PROJECT_DATA=/root/rivermind-data/projects/merge-route-boundary/data`
- `MRB_MODEL_ROOT=/root/rivermind-data/models`
- `MRB_CACHE_ROOT=/root/rivermind-data/datasets/_cache`
- `MRB_OUTPUT_ROOT=/root/rivermind-data/outputs`
- `MRB_FEATURE_ROOT=/root/rivermind-data/datasets/_derived/merge-route-boundary/features`
- Project data link: `data -> /root/rivermind-data/datasets`
- Project feature link: `features -> $MRB_FEATURE_ROOT`

Canonical feature bank layout:

```text
MRB_FEATURE_ROOT/<dataset_id>/<split>/<backbone_id>/
```

## Required Environment Variables

Use `.env.example` as the template for:

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

## Current Validation Commands

```bash
conda run -n moe pytest -q
conda run -n moe python scripts/inspect_datasets.py --no-status-json
conda run -n moe python scripts/verify_task_streams.py --stream controlled_cifar100_split_10x10 --seed 0 --skip-batch
conda run -n moe python scripts/verify_task_streams.py --stream real_tier0_tier2_small --seed 0 --skip-batch
conda run -n moe python scripts/verify_feature_bank.py --dataset cifar100 --split train --backbone clip_vit_b32
conda run -n moe python scripts/compute_boundary_diagnostics.py --stream controlled_cifar100_split_10x10 --seed 0 --backbone clip_vit_b32 --split train --strict-features --fail-on-any-failed-pair --overwrite
```

## Known Risks And Open Issues

- `no_split` datasets need deterministic holdout before formal evaluation; this
  QA phase adds the protocol, but downstream evaluation must still audit each
  dataset.
- Feature diagnostics are feature-side proxies, not parameter-level
  incompatibility `I`.
- Diagnostics quality depends on a full feature bank rather than smoke feature
  banks.
- `artifacts/diagnostics` outputs are local and ignored by git.
- Strict diagnostics should fail when required feature banks are missing; a
  failed strict run is expected on machines without those feature banks.

## Next Recommended Step

After this QA PR passes validation and is merged, the next phase can start
Feature-level Merge-vs-Route Toy Baselines. That next phase should implement
prototype or classifier baselines only after the current data, feature bank, and
diagnostics contracts remain stable.
