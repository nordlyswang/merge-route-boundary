# Datasets

This project keeps real datasets outside git under:

```bash
/root/rivermind-data/datasets
```

The repository-local `data/` path is a real directory. Its managed children are
symlinks into the shared dataset root, for example:

```text
data/torchvision -> /root/rivermind-data/datasets/torchvision
data/huggingface -> /root/rivermind-data/datasets/huggingface
data/manual -> /root/rivermind-data/datasets/manual
data/_cache -> /root/rivermind-data/datasets/_cache
```

Do not commit datasets, archives, caches, model weights, checkpoints, logs, or
generated outputs.

## Environment

```bash
export MRB_PROJECT_ROOT=/root/rivermind-data/projects/merge-route-boundary
export MRB_DATA_ROOT=/root/rivermind-data/datasets
export MRB_PROJECT_DATA=/root/rivermind-data/projects/merge-route-boundary/data
export MRB_CACHE_ROOT=/root/rivermind-data/datasets/_cache
export HF_HOME=/root/rivermind-data/datasets/_cache/huggingface
export HF_HUB_CACHE=/root/rivermind-data/datasets/_cache/huggingface/hub
export HF_DATASETS_CACHE=/root/rivermind-data/datasets/_cache/huggingface/datasets
export TORCH_HOME=/root/rivermind-data/datasets/_cache/torch
```

Create the shared directories and project symlinks:

```bash
bash scripts/create_data_symlink.sh
```

If a managed child such as `data/torchvision` already exists as a real file or
directory, the script refuses to overwrite it.

## Tier Strategy

Datasets are declared in `configs/resources/datasets.yaml`.

- Tier 0 is the default download set: MNIST, FashionMNIST, CIFAR10, CIFAR100.
- Tier 1 is optional controlled-boundary data: SVHN, STL10, Caltech101, DTD,
  EuroSAT, OxfordIIITPet, Flowers102, FGVCAircraft.
- Tier 2 is larger real continual-classification data: Food101, SUN397,
  StanfordCars.
- Tier 3 is optional domain/OOD data: PACS, OfficeHome, ImageNetR,
  ImageNetSketch, DomainNetFull, ImageNet1K.

Large, manual, and gated datasets are never downloaded by default.

## Commands

```bash
cd /root/rivermind-data/projects/merge-route-boundary
conda activate mrb

python scripts/report_storage.py
python scripts/download_datasets.py --dry-run
python scripts/download_datasets.py --tier 0 --max-gb 10
python scripts/verify_datasets.py --tier 0 --write-status
```

Optional Tier 1 example:

```bash
python scripts/download_datasets.py --name SVHN --name Caltech101 --name DTD --name EuroSAT --max-gb 20
python scripts/verify_datasets.py --name SVHN --name Caltech101 --name DTD --name EuroSAT --write-status
```

## Large And Manual Data

`SUN397`, `DomainNetFull`, and `ImageNet1K` are large and require explicit
intent. Use `--include-large` only after checking storage:

```bash
python scripts/report_storage.py
python scripts/download_datasets.py --name SUN397 --include-large --max-gb 120
```

Manual or gated datasets are manifest entries only in this phase. Place them
under the documented `manual/` or `huggingface/` subdirectory after obtaining
access through the correct source. `scripts/download_datasets.py` prints a skip
reason instead of downloading them.

Suggested manual locations:

```text
/root/rivermind-data/datasets/manual/stanford_cars/
/root/rivermind-data/datasets/manual/imagenet1k/
```

## Verification Outputs

With `--write-status`, verification writes:

```text
/root/rivermind-data/datasets/_metadata/dataset_status.json
/root/rivermind-data/datasets/_metadata/verification_log.jsonl
```

The verifier does not download missing datasets. For torchvision datasets,
missing files are reported as errors. For manual/Hugging Face registration-only
entries, the verifier records a skipped/manual status.

## Common Issues

- `WORLD_SIZE > 1`: dataset downloads are refused because torchvision downloads
  must run serially.
- Insufficient disk space: the downloader estimates peak space as
  `estimated_gb * 2.2 + min_free_gb_after_download` and refuses early.
- Existing `data/<entry>` real directory: move it manually before running
  `scripts/create_data_symlink.sh`; the script will not overwrite it.
- Missing optional dependencies: activate the Conda environment from
  `environment.yml` or `environment.cpu.yml`.
