时间戳（UTC+8）：2026-06-25

# Codex 指导文档：Dataset Registry 与 Task Stream Layer

## 0. 当前阶段目标

本项目 `merge-route-boundary` 旨在独立研究 **continual learning、model merging 与 MoE routing 之间的 merge-route-expand 边界**。后续论文的核心不是追单一 SOTA，而是回答：

> 给定一组持续到来的任务更新、有限专家/adapter 预算与推理成本约束，什么时候应该 merge，什么时候应该 route，什么时候应该 expand 或继续学习？

当前服务器已经完成 Tier 0–2 数据集下载与基础组织。本阶段 Codex 只需要把这些数据集变成后续实验可复现调用的资产：

1. 建立统一 Dataset Registry；
2. 验证共享数据目录与项目软链接；
3. 定义 controlled / real task streams；
4. 固定 split manifest 与 dataset fingerprint；
5. 实现只读验证脚本和单元测试。

**本阶段不允许训练模型，不允许抽取 frozen features，不允许实现 LoRA/adapter、model merging 或 MoE routing。**

---

## 1. 服务器路径约束

数据集必须共用以下服务器目录：

```bash
/root/rivermind-data/datasets
```

项目目录中只保留软链接：

```bash
/root/rivermind-data/projects/merge-route-boundary/data \
  -> /root/rivermind-data/datasets
```

Codex 必须遵守：

```text
- 不要复制数据集到项目目录；
- 不要把 data/、cache/、artifacts/ 下的大文件提交进 git；
- 不要重新下载已经存在的数据集；
- 本阶段所有 loader 必须 download=False；
- 如果数据缺失，只报告 missing，不自动下载。
```

建议使用环境变量：

```bash
export MRB_PROJECT_ROOT=/root/rivermind-data/projects/merge-route-boundary
export MRB_DATA_ROOT=/root/rivermind-data/datasets
export MRB_CACHE_ROOT=/root/rivermind-data/datasets/.cache/mrb
export HF_HOME=/root/rivermind-data/datasets/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TORCH_HOME=/root/rivermind-data/datasets/.cache/torch
```

---

## 2. 本 PR 的边界

### 必须实现

```text
mrb/data/registry.py
mrb/data/datasets.py
mrb/data/transforms.py
mrb/data/splits.py
mrb/data/task_streams.py
mrb/data/fingerprint.py

configs/datasets/registry.yaml
configs/task_streams/controlled.yaml
configs/task_streams/real_tier0_tier2.yaml

scripts/inspect_datasets.py
scripts/build_split_manifest.py
scripts/verify_task_streams.py
scripts/report_data_storage.py

tests/test_dataset_registry.py
tests/test_task_streams.py
tests/test_split_manifest.py
tests/test_data_paths.py
```

### 明确禁止

```text
- 不要训练任何模型；
- 不要抽取或缓存 CLIP/ViT/ResNet features；
- 不要实现 task delta、adapter、LoRA、TIES、DARE、routing；
- 不要下载 Tier 3 数据集；
- 不要创建 resized image copy 或重复数据副本；
- 不要使用 Docker；服务器环境采用 Conda；
- 不要把真实大数据路径写死到 Python 逻辑中，必须从环境变量和 config 解析。
```

---

## 3. Dataset Registry 设计

创建：

```text
configs/datasets/registry.yaml
```

每个数据集条目建议格式：

```yaml
cifar100:
  tier: 0
  source: torchvision
  root: ${MRB_DATA_ROOT}/CIFAR100
  task_type: classification
  num_classes: 100
  splits: [train, test]
  default_image_size: 224
  expected_available: true
  allow_in_smoke: true
  large_dataset: false
  manual_download: false
  notes: "Torchvision CIFAR100 dataset. Loader must use download=False."
```

Registry 至少需要记录：

```text
dataset_id
tier
source
root path
task_type
num_classes
available / missing
train size / test size, if cheaply available
estimated disk usage
image mode, if known
whether manual download is required
whether allowed in default smoke tests
```

请优先覆盖当前已下载的 Tier 0–2 数据集。若某些数据集目录结构与 torchvision 默认结构不一致，不要强行修复数据；先在 registry 中标注 `custom_layout: true`，并在 `inspect_datasets.py` 中报告。

---

## 4. Dataset Loader Interface

创建：

```text
mrb/data/datasets.py
```

目标是提供统一的只读接口：

```python
def get_dataset(dataset_id: str, split: str, transform=None, root: str | None = None):
    ...
```

要求：

```text
- 优先支持 torchvision datasets；
- 所有 torchvision loader 必须 download=False；
- 数据缺失时抛出清晰错误，而不是自动下载；
- 返回对象应兼容 torch.utils.data.Dataset；
- 不要在 dataset loader 中执行随机 split；
- train/val/test 划分由 split manifest 控制。
```

对于暂时无法稳定自动加载的数据集，允许先实现 registry inspection，不强制实现完整 loader。但需要在 `registry.yaml` 中标出：

```yaml
loader_status: inspect_only
```

---

## 5. Transform 设计

创建：

```text
mrb/data/transforms.py
```

提供两个基础 transform：

```python
def build_train_transform(image_size: int = 224):
    ...

def build_eval_transform(image_size: int = 224):
    ...
```

本阶段 transform 只用于 DataLoader smoke check，不服务真实训练。保持简单即可：

```text
train: Resize / RandomResizedCrop / RandomHorizontalFlip / ToTensor / Normalize
eval: Resize / CenterCrop / ToTensor / Normalize
```

不要在本阶段引入复杂 augmentation policy。

---

## 6. Task Stream 设计

这篇论文后续要画 merge-route boundary phase diagram，因此 task stream 必须先固定。

创建：

```text
configs/task_streams/controlled.yaml
configs/task_streams/real_tier0_tier2.yaml
```

### 6.1 Controlled streams

用于后续边界分析：控制 task incompatibility `I` 与 input separability `S`。

建议先定义：

```yaml
controlled_cifar100_split_10x10:
  stream_type: class_incremental
  base_dataset: cifar100
  num_tasks: 10
  classes_per_task: 10
  class_order_seed: 0
  val_ratio: 0.1
  image_size: 224

controlled_cifar10_split_5x2:
  stream_type: class_incremental
  base_dataset: cifar10
  num_tasks: 5
  classes_per_task: 2
  class_order_seed: 0
  val_ratio: 0.1
  image_size: 224

controlled_domain_mixed_small:
  stream_type: dataset_incremental
  datasets: [cifar10, svhn, stl10]
  val_ratio: 0.1
  image_size: 224
```

### 6.2 Real streams

用于后续真实多数据集 continual setting。

示例：

```yaml
real_tier0_tier2_small:
  stream_type: dataset_incremental
  datasets:
    - cifar10
    - cifar100
    - svhn
    - stl10
    - dtd
    - eurosat
    - oxford_iiit_pet
    - flowers102
  val_ratio: 0.1
  image_size: 224
```

如果某些数据集尚未验证 loader，可先将其标记为 `disabled: true`，但不要删除配置。

---

## 7. Split Manifest 设计

创建：

```text
mrb/data/splits.py
scripts/build_split_manifest.py
```

目标：同一个 `stream_id + seed + config` 必须生成完全一致的 train/val/test 划分。

输出目录建议：

```text
artifacts/manifests/splits/
```

注意：`artifacts/` 默认不提交 git。若需要提交一个极小 demo manifest，可以放在：

```text
tests/fixtures/manifests/
```

Manifest 示例：

```json
{
  "stream_id": "controlled_cifar100_split_10x10",
  "seed": 0,
  "data_root": "/root/rivermind-data/datasets",
  "created_by": "scripts/build_split_manifest.py",
  "tasks": [
    {
      "task_id": 0,
      "dataset_id": "cifar100",
      "classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
      "num_train": 4500,
      "num_val": 500,
      "num_test": 1000
    }
  ]
}
```

要求：

```text
- class-incremental split 必须按 class_order_seed 固定类别顺序；
- val split 只从当前 task 的 train indices 中划分；
- 当前 task loader 不允许访问 future task 的训练样本；
- manifest 中记录每个 task 的 class list、样本数和 split hash；
- 不要在训练脚本中临时随机划分。
```

---

## 8. Task Stream Loader

创建：

```text
mrb/data/task_streams.py
```

建议接口：

```python
def build_task_stream(stream_id: str, seed: int, registry_path: str | None = None):
    ...
```

每个 task 对象至少包含：

```text
task_id
dataset_id
classes
train_indices
val_indices
test_indices
num_train
num_val
num_test
image_size
```

本阶段只需要支持构建元信息和简单 DataLoader smoke check，不需要训练模型。

---

## 9. Dataset Fingerprint

创建：

```text
mrb/data/fingerprint.py
```

不要 hash 全量图像，避免浪费时间。只做轻量 fingerprint：

```text
- dataset root absolute path；
- number of files；
- total disk size；
- first N relative file paths hash；
- train/test sample count；
- class count；
- mtime summary；
- registry version hash。
```

输出：

```text
artifacts/dataset_status.json
```

该文件默认不提交 git。它用于服务器本地审计。

---

## 10. 脚本要求

### 10.1 inspect_datasets.py

命令：

```bash
python scripts/inspect_datasets.py --registry configs/datasets/registry.yaml
```

功能：

```text
- 检查 MRB_DATA_ROOT 是否存在；
- 检查项目 data 是否为软链接；
- 列出每个 dataset 的 available/missing 状态；
- 输出估计磁盘占用；
- 对 allow_in_smoke=true 的数据集尝试加载 train/test 元信息；
- 不下载任何数据。
```

### 10.2 build_split_manifest.py

命令：

```bash
python scripts/build_split_manifest.py \
  --stream controlled_cifar100_split_10x10 \
  --seed 0 \
  --output artifacts/manifests/splits/controlled_cifar100_split_10x10_seed0.json
```

功能：

```text
- 从 configs/task_streams/*.yaml 读取 stream config；
- 从 registry 解析 dataset；
- 固定 seed 生成 split；
- 写出 manifest JSON；
- 不训练，不抽 feature。
```

### 10.3 verify_task_streams.py

命令：

```bash
python scripts/verify_task_streams.py \
  --stream controlled_cifar100_split_10x10 \
  --seed 0
```

功能：

```text
- 验证每个 task 的 train/val/test 非空；
- 验证 class-incremental 任务之间 class 不重叠；
- 验证 split 可复现；
- 验证当前 task 不包含 future task 的 train indices；
- 对每个 task 取一个 batch 做 DataLoader smoke check。
```

### 10.4 report_data_storage.py

命令：

```bash
python scripts/report_data_storage.py --data-root /root/rivermind-data/datasets
```

功能：

```text
- 报告 data root 总占用；
- 报告各 dataset 子目录估计大小；
- 报告是否存在明显重复目录；
- 不删除任何文件。
```

---

## 11. 测试要求

测试应尽量不依赖真实大数据集。优先使用临时目录和 synthetic metadata。

必须包含：

```text
tests/test_dataset_registry.py
- registry config 能加载；
- 环境变量路径能解析；
- missing dataset 状态能正确报告。

tests/test_task_streams.py
- class-incremental stream task 数正确；
- classes_per_task 正确；
- 同一 seed 可复现；
- 不同 seed 类别顺序不同。

tests/test_split_manifest.py
- manifest JSON schema 正确；
- split hash 稳定；
- train/val/test 不重叠。

tests/test_data_paths.py
- data symlink 检查函数可工作；
- 不要求 CI 环境真的存在服务器路径。
```

建议添加 pytest marker：

```text
@pytest.mark.data
```

真实数据加载测试默认不在普通 CI 中运行；服务器上可通过脚本执行。

---

## 12. 验收命令

在服务器项目目录中运行：

```bash
cd /root/rivermind-data/projects/merge-route-boundary
conda activate merge-route-boundary

# 路径与数据状态
python scripts/report_data_storage.py --data-root /root/rivermind-data/datasets
python scripts/inspect_datasets.py --registry configs/datasets/registry.yaml

# controlled stream manifest
python scripts/build_split_manifest.py \
  --stream controlled_cifar100_split_10x10 \
  --seed 0 \
  --output artifacts/manifests/splits/controlled_cifar100_split_10x10_seed0.json

python scripts/verify_task_streams.py \
  --stream controlled_cifar100_split_10x10 \
  --seed 0

# tests
pytest tests/test_dataset_registry.py \
       tests/test_task_streams.py \
       tests/test_split_manifest.py \
       tests/test_data_paths.py
```

验收标准：

```text
- 所有命令成功执行；
- 不下载任何新数据；
- 不训练任何模型；
- 不抽取 features；
- data/ 是指向 /root/rivermind-data/datasets 的软链接；
- artifacts/ 下只生成小型 JSON manifest/status 文件；
- pytest 通过；
- git status 不包含数据集、模型权重、缓存或大文件。
```

---

## 13. Codex 执行 Prompt

可以直接复制给 Codex：

```text
请在当前仓库 /root/rivermind-data/projects/merge-route-boundary 中实现 Dataset Registry + Task Stream Layer。

背景：
本项目研究 continual learning、model merging、MoE routing 之间的 merge-route-expand 边界。当前服务器已经下载并组织好了 Tier 0–2 数据集，数据集共享目录为：
/root/rivermind-data/datasets
项目目录中的 data 应该是软链接：
/root/rivermind-data/projects/merge-route-boundary/data -> /root/rivermind-data/datasets

本 PR 的目标不是训练模型，而是把已下载数据集变成可复现、可验证、可复用的实验资产。

请实现：
1. mrb/data/registry.py
2. mrb/data/datasets.py
3. mrb/data/transforms.py
4. mrb/data/splits.py
5. mrb/data/task_streams.py
6. mrb/data/fingerprint.py
7. configs/datasets/registry.yaml
8. configs/task_streams/controlled.yaml
9. configs/task_streams/real_tier0_tier2.yaml
10. scripts/inspect_datasets.py
11. scripts/build_split_manifest.py
12. scripts/verify_task_streams.py
13. scripts/report_data_storage.py
14. tests/test_dataset_registry.py
15. tests/test_task_streams.py
16. tests/test_split_manifest.py
17. tests/test_data_paths.py

硬性要求：
- 不允许训练模型；
- 不允许抽取或缓存 CLIP/ViT/ResNet features；
- 不允许实现 adapter、LoRA、model merging、MoE routing；
- 不允许下载 Tier 3 数据集；
- 所有 dataset loader 必须 download=False；
- 不允许复制数据集到项目目录；
- 不允许提交 data/、cache/、artifacts/ 下的大文件；
- 路径必须通过 MRB_DATA_ROOT 等环境变量或 config 解析；
- 如果真实数据缺失，脚本应给出清晰报告，不要自动下载。

验收：
cd /root/rivermind-data/projects/merge-route-boundary
conda activate merge-route-boundary
python scripts/report_data_storage.py --data-root /root/rivermind-data/datasets
python scripts/inspect_datasets.py --registry configs/datasets/registry.yaml
python scripts/build_split_manifest.py --stream controlled_cifar100_split_10x10 --seed 0 --output artifacts/manifests/splits/controlled_cifar100_split_10x10_seed0.json
python scripts/verify_task_streams.py --stream controlled_cifar100_split_10x10 --seed 0
pytest tests/test_dataset_registry.py tests/test_task_streams.py tests/test_split_manifest.py tests/test_data_paths.py

完成后请总结：
- 已实现文件；
- 数据集 available/missing 状态；
- 生成的 stream manifest；
- 是否存在路径或 loader 问题；
- 没有执行训练/feature extraction/merging/routing。
```

---

## 14. 完成后的下一步

本 PR 完成后，下一阶段才进入：

```text
Frozen Feature Bank Layer
```

届时再实现：

```text
- 使用 frozen CLIP / ViT / ResNet 提取 features；
- 缓存 feature bank；
- 计算 input separability S；
- 为后续 task incompatibility I 和 route-vs-merge gap 分析做准备。
```

当前阶段只完成数据层，不进入模型层。
