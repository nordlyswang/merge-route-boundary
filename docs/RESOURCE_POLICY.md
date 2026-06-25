# Resource Policy

Git stores source code, configuration manifests, tests, and documentation only.
Large resources must not be committed.

## Required Environment Variables

- `MRB_DATA_ROOT`: datasets.
- `MRB_MODEL_ROOT`: local model metadata or snapshots.
- `MRB_CACHE_ROOT`: shared cache root.
- `MRB_OUTPUT_ROOT`: generated reports and future experiment outputs.
- `HF_HOME`: Hugging Face home directory.
- `HF_HUB_CACHE`: Hugging Face hub cache.
- `HF_DATASETS_CACHE`: Hugging Face datasets cache.
- `TORCH_HOME`: Torch and TorchVision cache.

## Ignored Local Resources

The repository ignores local resource directories and model artifacts, including:

- `data/`, `models/`, `cache/`, `outputs/`
- `checkpoints/`, `logs/`, `wandb/`
- `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.bin`

## Manifests

Datasets are declared in `configs/resources/datasets.yaml`.
Models are declared in `configs/resources/models.yaml`.

Download scripts must read these manifests and default to required resources
only. Model downloads default to metadata/tokenizer/config files. Full weights
are allowed only when `scripts/download_models.py --include-weights` is passed.
