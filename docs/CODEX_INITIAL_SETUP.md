# Codex Initial Setup

This repository is initialized from `docs/CODEX_INITIAL_SETUP_CONDA.md`.
The first phase is limited to infrastructure:

- Conda/Mamba environment files.
- Resource directory policy.
- Manifest-driven model and dataset download scripts.
- Environment and resource verification scripts.
- Quick smoke checks and minimal tests.

Do not add Docker, devcontainer, docker-compose, training loops, model merging,
MoE routing, adapter/LoRA training, benchmarks, or experiment logic during this
phase.

## Acceptance Commands

```bash
python scripts/check_env.py
python scripts/download_datasets.py --dry-run
python scripts/download_models.py --dry-run
python scripts/verify_resources.py
python scripts/smoke_check.py --quick
pytest tests/
```
