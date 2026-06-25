# Resource Policy

Git stores source code, configuration manifests, tests, and documentation only.
Large resources must not be committed.

## Required Environment Variables

- `MRB_DATA_ROOT`: shared datasets.
- `MRB_PROJECT_DATA`: project-local data link path.
- `MRB_MODEL_ROOT`: local model metadata or snapshots.
- `MRB_CACHE_ROOT`: shared cache root.
- `MRB_OUTPUT_ROOT`: generated reports and future experiment outputs.
- `MRB_FEATURE_ROOT`: frozen feature bank root.
- `HF_HOME`: Hugging Face home directory.
- `HF_HUB_CACHE`: Hugging Face hub cache.
- `HF_DATASETS_CACHE`: Hugging Face datasets cache.
- `TORCH_HOME`: Torch and TorchVision cache.

## Ignored Local Resources

The repository ignores local resource directories and generated artifacts,
including:

- `data/`, `features/`, `models/`, `cache/`, `outputs/`, `artifacts/`
- `checkpoints/`, `logs/`, `wandb/`
- `*.npy`, `*.npz`, `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.bin`

Do not commit dataset status JSON, split manifests, diagnostics CSV files,
feature banks, model weights, checkpoints, or cache files.

## Manifests

Download resources are declared in:

- `configs/resources/datasets.yaml`
- `configs/resources/models.yaml`

Dataset registry and task-stream resources are declared in:

- `configs/datasets/registry.yaml`
- `configs/task_streams/*.yaml`

Model downloads default to metadata/tokenizer/config files. Full weights are
allowed only when `scripts/download_models.py --include-weights` is passed.

## Feature Banks

Canonical layout:

```text
MRB_FEATURE_ROOT/<dataset_id>/<split>/<backbone_id>/
```

The diagnostics loader can read legacy layout
`MRB_FEATURE_ROOT/<backbone_id>/<dataset_id>/<split>/`, but new feature banks
must use the canonical layout.
