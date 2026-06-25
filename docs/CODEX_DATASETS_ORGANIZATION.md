# Codex 指导文件：数据集组织、下载与验证（共享目录 + 软链接）

> 适用仓库：`/root/rivermind-data/projects/merge-route-boundary`  
> GitHub：`https://github.com/nordlyswang/merge-route-boundary`  
> 目标阶段：只完成数据集目录组织、manifest、下载脚本、磁盘审计和验证脚本；**暂不实现具体实验、训练、merging、routing 或 adapter 逻辑**。  
> 服务器约束：不使用 Docker；项目环境使用 Conda；数据集集中安装到共享目录，并通过软链接映射到项目目录。

---

## 1. 项目背景

`merge-route-boundary` 是一个独立研究项目，目标是分析 **continual learning、model merging 与 MoE routing 的适用边界**。项目的核心问题不是先追某个 benchmark SOTA，而是建立一个可复现的实验平台，用于回答：

```text
给定任务流、task-specific updates/adapters/experts 和资源预算，
什么时候应该 merge，什么时候应该 route，什么时候应该 expand / continue learning？
```

因此，数据集准备阶段需要服务于三类后续实验：

1. **Controlled boundary analysis**：用小型、可控、低成本数据集构造 task incompatibility 与 input separability 的 phase diagram。
2. **Real continual classification**：用多数据集任务序列验证 merge / route / expand 的真实边界。
3. **Domain / OOD transfer analysis**：用跨域、风格迁移或 ImageNet 派生数据集评估 routing 与 static merging 的边界。

本阶段 Codex 只负责搭建数据底座：目录、manifest、下载、验证、磁盘报告和安全策略。

---

## 2. 固定路径约定

### 2.1 共享数据根目录

所有真实数据集必须放在：

```bash
/root/rivermind-data/datasets
```

项目内的 `data/` 不存放真实数据，而是软链接：

```bash
/root/rivermind-data/projects/merge-route-boundary/data -> /root/rivermind-data/datasets
```

这样以后多个项目可以共享同一份数据集。

### 2.2 推荐环境变量

Codex 需要更新 `.env.example` 和 `docs/DATASETS.md`，加入：

```bash
export MRB_PROJECT_ROOT=/root/rivermind-data/projects/merge-route-boundary
export MRB_DATA_ROOT=/root/rivermind-data/datasets
export MRB_PROJECT_DATA=/root/rivermind-data/projects/merge-route-boundary/data

# Shared cache directories. These are intentionally placed under the shared dataset tree.
export MRB_CACHE_ROOT=/root/rivermind-data/datasets/_cache
export HF_HOME=/root/rivermind-data/datasets/_cache/huggingface
export HF_HUB_CACHE=/root/rivermind-data/datasets/_cache/huggingface/hub
export HF_DATASETS_CACHE=/root/rivermind-data/datasets/_cache/huggingface/datasets
export TORCH_HOME=/root/rivermind-data/datasets/_cache/torch
```

### 2.3 目录结构

Codex 需要创建或确保存在：

```text
/root/rivermind-data/datasets/
  _cache/
    huggingface/
      hub/
      datasets/
    torch/
  _metadata/
    dataset_status.json
    storage_report.json
    download_log.jsonl
    verification_log.jsonl
  torchvision/
  huggingface/
  manual/
  external/
  indices/
  manifests/
```

项目目录中：

```text
/root/rivermind-data/projects/merge-route-boundary/
  data -> /root/rivermind-data/datasets
  configs/resources/datasets.yaml
  docs/DATASETS.md
  scripts/create_data_symlink.sh
  scripts/download_datasets.py
  scripts/verify_datasets.py
  scripts/report_storage.py
  scripts/inspect_dataset.py
  tests/test_dataset_manifest.py
  tests/test_data_paths.py
```

---

## 3. Codex 必须实现的第一组文件

### 3.1 `scripts/create_data_symlink.sh`

职责：创建共享数据目录和项目软链接。

要求：

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${MRB_PROJECT_ROOT:-/root/rivermind-data/projects/merge-route-boundary}
DATA_ROOT=${MRB_DATA_ROOT:-/root/rivermind-data/datasets}
PROJECT_DATA=${MRB_PROJECT_DATA:-${PROJECT_ROOT}/data}

mkdir -p "${DATA_ROOT}" \
  "${DATA_ROOT}/_cache/huggingface/hub" \
  "${DATA_ROOT}/_cache/huggingface/datasets" \
  "${DATA_ROOT}/_cache/torch" \
  "${DATA_ROOT}/_metadata" \
  "${DATA_ROOT}/torchvision" \
  "${DATA_ROOT}/huggingface" \
  "${DATA_ROOT}/manual" \
  "${DATA_ROOT}/external" \
  "${DATA_ROOT}/indices" \
  "${DATA_ROOT}/manifests"

mkdir -p "${PROJECT_ROOT}"

if [ -e "${PROJECT_DATA}" ] && [ ! -L "${PROJECT_DATA}" ]; then
  echo "ERROR: ${PROJECT_DATA} exists and is not a symlink. Refusing to overwrite."
  exit 1
fi

ln -sfn "${DATA_ROOT}" "${PROJECT_DATA}"

echo "Created symlink: ${PROJECT_DATA} -> ${DATA_ROOT}"
readlink -f "${PROJECT_DATA}"
```

验收：

```bash
bash scripts/create_data_symlink.sh
readlink -f /root/rivermind-data/projects/merge-route-boundary/data
# expected: /root/rivermind-data/datasets
```

---

## 4. 数据集选择原则

本阶段不要盲目下载所有数据。按 tier 管理。

### Tier 0：立即下载，smoke / CI / 快速验证

这些数据集小、稳定、下载快，用于验证路径、缓存、torchvision API、DataLoader 和基础 split 检查。

| Dataset | 用途 | 来源 | 估计磁盘 | 默认下载 |
|---|---|---|---:|---|
| MNIST | digit smoke test | torchvision | <100 MB | yes |
| FashionMNIST | MNIST-like domain shift | torchvision | <100 MB | yes |
| CIFAR10 | small object classification | torchvision | ~200 MB | yes |
| CIFAR100 | class-incremental / fine-grained small task | torchvision | ~200 MB | yes |

### Tier 1：建议下载，controlled boundary analysis

这些数据集仍可控，适合构造早期 phase diagram：低冲突/高冲突、输入可分/不可分、任务顺序切分。

| Dataset | 用途 | 来源 | 估计磁盘 | 默认下载 |
|---|---|---|---:|---|
| SVHN | digit domain shift vs MNIST/FashionMNIST | torchvision | ~1.5-2.5 GB（不含 extra 时更小） | no |
| STL10 | CIFAR-like but 96×96, small labeled train | torchvision | ~2.5-3 GB（含 unlabeled 会增大） | no |
| Caltech101 | object categories, low/moderate size | torchvision | <1 GB | no |
| DTD | texture task | torchvision | <1 GB | no |
| EuroSAT | satellite/remote sensing | torchvision | <0.2 GB for RGB | no |
| OxfordIIITPet | fine-grained pets | torchvision | <1 GB | no |
| Flowers102 | fine-grained flowers | torchvision | <1 GB | no |
| FGVCAircraft | fine-grained aircraft | torchvision/manual fallback | ~2-3 GB | no |

### Tier 2：real multi-dataset continual benchmark

这些数据集可组成视觉任务序列，用于真实 merge-route 边界分析。它们也接近 CLIP/MoE-Adapter 类工作中常用的 multi-dataset continual setup。

推荐 order：

```text
Aircraft,
Caltech101,
CIFAR100,
DTD,
EuroSAT,
Flowers102,
Food101,
MNIST,
OxfordIIITPet,
StanfordCars,
SUN397
```

| Dataset | 来源 | 估计磁盘 | 注意事项 |
|---|---|---:|---|
| FGVCAircraft | torchvision/manual | ~2-3 GB | 作为 Aircraft |
| Caltech101 | torchvision | <1 GB | 可能需要 `gdown` |
| CIFAR100 | torchvision | ~200 MB | Tier 0 已含 |
| DTD | torchvision | <1 GB | texture domain |
| EuroSAT | torchvision | <0.2 GB RGB | remote-sensing domain |
| Flowers102 | torchvision | <1 GB | 需要 `scipy` 加载 `.mat` target |
| Food101 | torchvision | ~5-6 GB download/extracted | 中等体积 |
| MNIST | torchvision | <100 MB | Tier 0 已含 |
| OxfordIIITPet | torchvision | <1 GB | fine-grained pets |
| StanfordCars | manual/HF/Kaggle fallback | ~2-3 GB | torchvision 官方原 URL 可能不可用，不作为默认自动下载 |
| SUN397 | torchvision/HF fallback | ~40 GB 级别 | **大数据集，需单独确认磁盘** |

### Tier 3：optional domain / OOD / robustness datasets

这些数据集适合后续扩展，不在第一轮默认下载。

| Dataset | 用途 | 来源 | 估计磁盘 | 默认下载 |
|---|---|---|---:|---|
| PACS | domain generalization, small | HF/manual | <1 GB | no |
| OfficeHome | 4 domains, 65 categories | official/HF/manual | ~1-2 GB | no |
| ImageNet-R | ImageNet renditions, OOD robustness | official/HF/manual | ~2-3 GB | no |
| ImageNet-Sketch | sketch shift, 1000 classes | official/HF/manual | ~2-4 GB | no |
| DomainNet-mini | domain generalization subset | manual/HF if available | variable | no |
| DomainNet-full | large multi-domain | HF/manual | 18.5 GB compressed on one HF mirror; original/extracted can be much larger | no |
| ImageNet-1K | major robustness / transfer baseline | official/manual | ~130 GB | no |

**Hard rule:** 第一轮不要自动下载 `SUN397`、`DomainNet-full`、`ImageNet-1K`。这些必须通过显式 `--include-large` 或 `--name SUN397` 触发，并要求 `--max-gb` 检查通过。

---

## 5. Dataset manifest 设计

Codex 需要创建：

```text
configs/resources/datasets.yaml
```

建议 schema：

```yaml
version: 1
root_env: MRB_DATA_ROOT
project_data_env: MRB_PROJECT_DATA

policies:
  default_tiers: [0]
  default_keep_archives: false
  min_free_gb_after_download: 20
  require_explicit_large_download: true
  large_dataset_threshold_gb: 10
  use_file_lock: true

paths:
  shared_root: /root/rivermind-data/datasets
  project_data_symlink: /root/rivermind-data/projects/merge-route-boundary/data
  torchvision_root: /root/rivermind-data/datasets/torchvision
  hf_root: /root/rivermind-data/datasets/huggingface
  manual_root: /root/rivermind-data/datasets/manual
  metadata_root: /root/rivermind-data/datasets/_metadata

datasets:
  - name: MNIST
    tier: 0
    source: torchvision
    torchvision_class: MNIST
    root_subdir: torchvision
    splits: [train, test]
    estimated_gb: 0.1
    default_download: true
    verify:
      min_total_examples: 70000
      num_classes: 10

  - name: FashionMNIST
    tier: 0
    source: torchvision
    torchvision_class: FashionMNIST
    root_subdir: torchvision
    splits: [train, test]
    estimated_gb: 0.1
    default_download: true
    verify:
      min_total_examples: 70000
      num_classes: 10

  - name: CIFAR10
    tier: 0
    source: torchvision
    torchvision_class: CIFAR10
    root_subdir: torchvision
    splits: [train, test]
    estimated_gb: 0.2
    default_download: true
    verify:
      min_total_examples: 60000
      num_classes: 10

  - name: CIFAR100
    tier: 0
    source: torchvision
    torchvision_class: CIFAR100
    root_subdir: torchvision
    splits: [train, test]
    estimated_gb: 0.2
    default_download: true
    verify:
      min_total_examples: 60000
      num_classes: 100

  - name: SVHN
    tier: 1
    source: torchvision
    torchvision_class: SVHN
    root_subdir: torchvision
    splits: [train, test]
    include_extra_split: false
    estimated_gb: 2.5
    default_download: false
    verify:
      min_total_examples: 99000
      num_classes: 10

  - name: STL10
    tier: 1
    source: torchvision
    torchvision_class: STL10
    root_subdir: torchvision
    splits: [train, test]
    include_unlabeled_split: false
    estimated_gb: 3.0
    default_download: false
    verify:
      min_total_examples: 13000
      num_classes: 10

  - name: Caltech101
    tier: 1
    source: torchvision
    torchvision_class: Caltech101
    root_subdir: torchvision
    estimated_gb: 1.0
    default_download: false
    verify:
      min_total_examples: 8000
      min_num_classes: 101

  - name: DTD
    tier: 1
    source: torchvision
    torchvision_class: DTD
    root_subdir: torchvision
    splits: [train, val, test]
    estimated_gb: 1.0
    default_download: false
    verify:
      min_total_examples: 5640
      num_classes: 47

  - name: EuroSAT
    tier: 1
    source: torchvision
    torchvision_class: EuroSAT
    root_subdir: torchvision
    estimated_gb: 0.2
    default_download: false
    verify:
      min_total_examples: 27000
      num_classes: 10

  - name: OxfordIIITPet
    tier: 1
    source: torchvision
    torchvision_class: OxfordIIITPet
    root_subdir: torchvision
    splits: [trainval, test]
    estimated_gb: 1.0
    default_download: false
    verify:
      min_total_examples: 7000
      num_classes: 37

  - name: Flowers102
    tier: 1
    source: torchvision
    torchvision_class: Flowers102
    root_subdir: torchvision
    splits: [train, val, test]
    estimated_gb: 1.0
    default_download: false
    verify:
      min_total_examples: 8000
      num_classes: 102

  - name: FGVCAircraft
    alias: Aircraft
    tier: 1
    source: torchvision
    torchvision_class: FGVCAircraft
    root_subdir: torchvision
    splits: [train, val, test]
    estimated_gb: 3.0
    default_download: false
    verify:
      min_total_examples: 10000
      num_classes: 100

  - name: Food101
    tier: 2
    source: torchvision
    torchvision_class: Food101
    root_subdir: torchvision
    splits: [train, test]
    estimated_gb: 6.0
    default_download: false
    verify:
      min_total_examples: 101000
      num_classes: 101

  - name: SUN397
    tier: 2
    source: torchvision
    torchvision_class: SUN397
    root_subdir: torchvision
    estimated_gb: 40.0
    default_download: false
    large: true
    verify:
      min_total_examples: 108000
      num_classes: 397

  - name: StanfordCars
    tier: 2
    source: manual
    root_subdir: manual/stanford_cars
    estimated_gb: 3.0
    default_download: false
    manual_reason: "torchvision docs indicate the original Stanford Cars URL is no longer available; require manual/HF/Kaggle source."
    verify:
      min_total_examples: 16000
      num_classes: 196

  - name: PACS
    tier: 3
    source: huggingface
    hf_id: flwrlabs/pacs
    root_subdir: huggingface/pacs
    estimated_gb: 1.0
    default_download: false
    verify:
      min_total_examples: 9900
      num_classes: 7
      domains: [art_painting, cartoon, photo, sketch]

  - name: OfficeHome
    tier: 3
    source: manual_or_huggingface
    hf_id: flwrlabs/office-home
    root_subdir: huggingface/office-home
    estimated_gb: 2.0
    default_download: false
    verify:
      min_total_examples: 15000
      num_classes: 65
      domains: [Art, Clipart, Product, Real_World]

  - name: ImageNetR
    tier: 3
    source: manual_or_huggingface
    hf_id: axiong/imagenet-r
    root_subdir: huggingface/imagenet-r
    estimated_gb: 3.0
    default_download: false
    verify:
      min_total_examples: 30000
      num_classes: 200

  - name: ImageNetSketch
    tier: 3
    source: manual_or_huggingface
    hf_id: songweig/imagenet_sketch
    root_subdir: huggingface/imagenet_sketch
    estimated_gb: 4.0
    default_download: false
    verify:
      min_total_examples: 50000
      num_classes: 1000

  - name: DomainNetFull
    tier: 3
    source: huggingface_or_manual
    hf_id: wltjr1007/DomainNet
    root_subdir: huggingface/domainnet
    estimated_gb: 25.0
    default_download: false
    large: true
    verify:
      min_total_examples: 500000
      domains: [clipart, infograph, painting, quickdraw, real, sketch]

  - name: ImageNet1K
    tier: 3
    source: manual_gated
    root_subdir: manual/imagenet1k
    estimated_gb: 130.0
    default_download: false
    large: true
    manual_reason: "Requires official ImageNet access and non-commercial research terms. Do not auto-download."
    verify:
      min_total_examples: 1280000
      num_classes: 1000
```

---

## 6. 下载脚本要求

### 6.1 `scripts/download_datasets.py`

必须 manifest-driven，不允许在脚本里写死数据集列表。

支持 CLI：

```bash
python scripts/download_datasets.py --dry-run
python scripts/download_datasets.py --tier 0
python scripts/download_datasets.py --tier 0 --tier 1
python scripts/download_datasets.py --name CIFAR100
python scripts/download_datasets.py --name Food101 --max-gb 100
python scripts/download_datasets.py --name SUN397 --include-large --max-gb 120
python scripts/download_datasets.py --source torchvision
```

参数要求：

```text
--manifest configs/resources/datasets.yaml
--dry-run
--tier <int>，可重复
--name <dataset_name>，可重复
--all-small            # 下载所有 large=false 且 tier<=2 的数据集
--include-large        # 允许下载 large=true 的数据集
--max-gb <float>       # 下载前估算需要空间；不足则拒绝
--keep-archives        # 默认 false，验证成功后可清理 archives
--force                # 已存在时重新下载/重新验证
--skip-manual          # 默认 true；manual 数据集只打印说明
```

### 6.2 磁盘保护逻辑

下载任何数据集前必须检查：

```python
free_gb = shutil.disk_usage(MRB_DATA_ROOT).free / 1024**3
required_gb = estimated_gb * 2.2 + min_free_gb_after_download
```

解释：很多数据集会先下载 archive，再解压，因此下载峰值可能接近最终体积的 2 倍。

拒绝条件：

```text
1. free_gb < required_gb
2. dataset.large == true and --include-large not set
3. estimated_gb > --max-gb
4. source == manual and --skip-manual true
```

### 6.3 不允许并发下载

使用文件锁：

```text
/root/rivermind-data/datasets/_metadata/download.lock
```

原因：torchvision 的下载逻辑在多进程/分布式场景下可能产生 race condition。所有下载脚本必须串行下载。

---

## 7. 验证脚本要求

### 7.1 `scripts/verify_datasets.py`

支持：

```bash
python scripts/verify_datasets.py --tier 0
python scripts/verify_datasets.py --name CIFAR100
python scripts/verify_datasets.py --all-present
python scripts/verify_datasets.py --write-status
```

每个数据集验证至少包括：

```text
1. root path 是否存在；
2. dataset object 是否能实例化；
3. split 长度是否满足 manifest 中的 min_total_examples；
4. class count 是否满足 num_classes 或 min_num_classes；
5. DataLoader 是否能读取一个 batch；
6. image tensor/PIL image shape 是否合理；
7. label 是否为 int 或可映射 class id；
8. 不修改数据集文件；
9. 不触发重新下载，除非用户显式传 --download-if-missing。
```

### 7.2 验证输出

写入：

```text
/root/rivermind-data/datasets/_metadata/dataset_status.json
/root/rivermind-data/datasets/_metadata/verification_log.jsonl
```

`dataset_status.json` 示例：

```json
{
  "CIFAR100": {
    "status": "ok",
    "source": "torchvision",
    "root": "/root/rivermind-data/datasets/torchvision",
    "verified_at": "2026-06-23T12:34:56+08:00",
    "splits": {
      "train": 50000,
      "test": 10000
    },
    "num_classes": 100,
    "disk_gb": 0.17,
    "notes": []
  },
  "SUN397": {
    "status": "not_downloaded",
    "reason": "large dataset; requires explicit --include-large"
  }
}
```

---

## 8. 存储报告脚本

### 8.1 `scripts/report_storage.py`

支持：

```bash
python scripts/report_storage.py
python scripts/report_storage.py --json /root/rivermind-data/datasets/_metadata/storage_report.json
python scripts/report_storage.py --top 30
```

输出：

```text
MRB_DATA_ROOT: /root/rivermind-data/datasets
Project symlink: /root/rivermind-data/projects/merge-route-boundary/data -> /root/rivermind-data/datasets
Filesystem free: xxx GB
Dataset directories:
  torchvision/CIFAR100: 0.17 GB
  torchvision/Food101: 5.8 GB
  _cache/huggingface: 2.1 GB
Largest files/directories:
  ...
```

要求：

- 使用 `du -sh` 或 Python `os.scandir` 统计目录大小。
- 不跟随项目内 `data` 软链接重复计算。
- 不删除任何文件。

---

## 9. `docs/DATASETS.md` 内容要求

Codex 需要新增文档：

```text
docs/DATASETS.md
```

必须包括：

1. 共享数据目录说明；
2. 软链接创建方式；
3. 环境变量；
4. 数据集 tier 策略；
5. 默认下载命令；
6. 大数据集下载警告；
7. manual/gated 数据集说明；
8. 验证命令；
9. 常见故障。

建议用户命令：

```bash
cd /root/rivermind-data/projects/merge-route-boundary
conda activate mrb

# 1. Create shared data root and project symlink
bash scripts/create_data_symlink.sh

# 2. Check storage
python scripts/report_storage.py

# 3. Dry-run default downloads
python scripts/download_datasets.py --dry-run

# 4. Download Tier 0 only
python scripts/download_datasets.py --tier 0 --max-gb 10

# 5. Verify Tier 0
python scripts/verify_datasets.py --tier 0 --write-status

# 6. Optional: download Tier 1 when storage is sufficient
python scripts/download_datasets.py --tier 1 --max-gb 40
python scripts/verify_datasets.py --tier 1 --write-status
```

---

## 10. 立即下载建议

第一轮只执行：

```bash
python scripts/download_datasets.py --tier 0 --max-gb 10
python scripts/verify_datasets.py --tier 0 --write-status
```

如果 Tier 0 成功，再下载 Tier 1 的一小组：

```bash
python scripts/download_datasets.py --name SVHN --name Caltech101 --name DTD --name EuroSAT --max-gb 20
python scripts/verify_datasets.py --name SVHN --name Caltech101 --name DTD --name EuroSAT --write-status
```

暂时不要下载：

```text
SUN397
DomainNetFull
ImageNet1K
```

除非你确认：

```bash
df -h /root/rivermind-data/datasets
```

至少还有 100GB 以上可用空间。

---

## 11. Manual / gated 数据集处理原则

有些数据集不能稳定通过 torchvision 自动下载或存在访问条款要求。

### 11.1 StanfordCars

- 不作为默认自动下载项。
- `datasets.yaml` 标记为 `source: manual`。
- `download_datasets.py` 遇到它只打印说明，不应报错退出。
- 验证脚本支持用户手动放置后的检查。

建议手动结构：

```text
/root/rivermind-data/datasets/manual/stanford_cars/
  cars_train/
  cars_test/
  devkit/
```

### 11.2 ImageNet-1K

- 需要官方访问权限。
- 不允许 Codex 自动下载。
- 只提供目录规范和验证脚本。

建议结构：

```text
/root/rivermind-data/datasets/manual/imagenet1k/
  train/
    n01440764/
    ...
  val/
    n01440764/
    ...
```

### 11.3 DomainNetFull

- 很大，不作为默认下载。
- 如果后续只需要 domain separability 初步实验，优先使用 PACS、OfficeHome 或 DomainNet subset。

---

## 12. 单元测试要求

新增：

```text
tests/test_dataset_manifest.py
tests/test_data_paths.py
```

### 12.1 `test_dataset_manifest.py`

检查：

```python
def test_dataset_manifest_loads(): ...
def test_required_fields_exist(): ...
def test_tier0_has_default_downloads(): ...
def test_large_datasets_not_default_download(): ...
def test_manual_datasets_not_default_download(): ...
```

### 12.2 `test_data_paths.py`

检查：

```python
def test_expected_env_defaults(): ...
def test_project_data_symlink_is_not_required_for_unit_tests(): ...
def test_no_dataset_paths_inside_git_by_default(): ...
```

单元测试不得下载数据。

---

## 13. Codex 执行边界

Codex 本阶段可以做：

```text
- 创建/修改 manifest；
- 创建目录和软链接脚本；
- 实现 dry-run 下载脚本；
- 实现 torchvision/HF 下载适配器；
- 实现 verify 脚本；
- 实现 storage report；
- 写文档和测试。
```

Codex 本阶段禁止做：

```text
- 不要实现 continual learning 训练；
- 不要实现 model merging；
- 不要实现 MoE routing；
- 不要实现 adapter / LoRA training；
- 不要提交任何下载后的数据；
- 不要把 data/ 变成真实目录；必须是 symlink；
- 不要自动下载 SUN397 / DomainNetFull / ImageNet1K；
- 不要修改 Conda 环境为 Docker/devcontainer。
```

---

## 14. 推荐给 Codex 的直接任务描述

可以直接复制给 Codex：

```text
请在 merge-route-boundary 仓库中实现数据集组织、下载和验证的第二阶段基础设施。服务器不支持 Docker，项目使用 Conda。所有数据集必须集中安装到 /root/rivermind-data/datasets，并通过软链接映射到 /root/rivermind-data/projects/merge-route-boundary/data。不要在 git 中提交任何数据、模型权重、archive、cache 或日志。

请完成：

1. 新增 scripts/create_data_symlink.sh，创建共享数据目录结构，并确保项目 data/ 是指向 /root/rivermind-data/datasets 的软链接。如果 data/ 已存在且不是软链接，必须报错并拒绝覆盖。
2. 新增 configs/resources/datasets.yaml，使用 manifest 管理数据集。至少包含 Tier 0: MNIST, FashionMNIST, CIFAR10, CIFAR100；Tier 1: SVHN, STL10, Caltech101, DTD, EuroSAT, OxfordIIITPet, Flowers102, FGVCAircraft；Tier 2: Food101, SUN397, StanfordCars；Tier 3: PACS, OfficeHome, ImageNetR, ImageNetSketch, DomainNetFull, ImageNet1K。
3. 新增 scripts/download_datasets.py，manifest-driven，支持 --dry-run, --tier, --name, --max-gb, --include-large, --skip-manual, --force。默认只下载 Tier 0。大数据集和 manual/gated 数据集不得默认下载。
4. 新增 scripts/verify_datasets.py，验证路径、split 长度、class 数、一个 batch 是否可读取，并写入 /root/rivermind-data/datasets/_metadata/dataset_status.json 和 verification_log.jsonl。
5. 新增 scripts/report_storage.py，报告 MRB_DATA_ROOT 的剩余空间、各数据集目录大小、最大目录/文件。不要跟随项目 data/ 软链接重复计算。
6. 新增 docs/DATASETS.md，说明共享目录、软链接、环境变量、tier 策略、下载/验证命令、大数据集警告和 manual 数据集处理。
7. 新增 tests/test_dataset_manifest.py 和 tests/test_data_paths.py。测试不允许下载数据。

硬性要求：
- .gitignore 必须忽略 data/, _cache/, datasets/, outputs/, checkpoints/, *.zip, *.tar, *.tgz, *.gz, *.pt, *.pth, *.ckpt。
- 使用文件锁避免并发下载。
- 下载前必须检查磁盘剩余空间。估算空间不足时直接拒绝。
- torchvision 下载必须串行，不要在分布式/多进程中触发。
- --dry-run 必须不下载任何文件，只打印计划。
- 初始验收命令为：
  bash scripts/create_data_symlink.sh
  python scripts/report_storage.py
  python scripts/download_datasets.py --dry-run
  python scripts/download_datasets.py --tier 0 --max-gb 10
  python scripts/verify_datasets.py --tier 0 --write-status
  pytest tests/test_dataset_manifest.py tests/test_data_paths.py
```

---

## 15. 参考资料

这些资料用于确定第一阶段数据集范围和实现注意事项：

- Torchvision built-in datasets list and download warning: https://docs.pytorch.org/vision/stable/datasets.html
- Hugging Face Datasets cache docs: https://huggingface.co/docs/datasets/en/cache
- Hugging Face Hub environment variables: https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables
- CIFAR official page: https://cave.cs.toronto.edu/kriz/cifar.html
- Food-101 official page: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
- FGVC-Aircraft official page: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
- Oxford-IIIT Pet official page: https://www.robots.ox.ac.uk/~vgg/data/pets/
- Oxford Flowers 102 official page: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
- DTD official page: https://www.robots.ox.ac.uk/~vgg/data/dtd/
- SVHN official page: https://ufldl.stanford.edu/housenumbers/
- STL10 official page: https://cs.stanford.edu/~acoates/stl10/
- ImageNet-R official repository: https://github.com/hendrycks/imagenet-r
- ImageNet-Sketch official repository: https://github.com/HaohanWang/ImageNet-Sketch
- Office-Home official page: https://www.hemanthdv.org/OfficeHome-Dataset/
