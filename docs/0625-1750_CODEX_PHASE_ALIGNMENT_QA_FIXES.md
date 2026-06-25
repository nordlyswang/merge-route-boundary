> 时间戳：06/25 17:48（东八区）

# Codex 阶段任务指导：Phase Alignment + QA 修复

## 0. 当前任务定位

项目仓库：

```text
/root/rivermind-data/projects/merge-route-boundary
https://github.com/nordlyswang/merge-route-boundary
```

当前工作分支：

```text
codex/0625-dataset-registry
```

本阶段不是继续新增研究模块，而是做一次 **Phase Alignment + QA 修复**。当前仓库已经包含：

```text
mrb.data
mrb.features
mrb.diagnostics
```

但 README / AGENTS / docs 仍停留在 initial setup 阶段，同时部分模块之间存在 API/import 不一致、split 协议不清、diagnostics 失败门槛不足等问题。

本 PR 的目标是：

```text
1. 修复当前代码在 fresh clone / conda moe 环境下的可运行性；
2. 对齐 README、AGENTS、docs 与当前阶段；
3. 明确 dataset split、feature bank layout、diagnostics 质量门槛；
4. 生成 PROJECT_STATUS.md，记录当前已完成内容、验收命令、风险和下一步；
5. 不新增实验方法，不进入 feature-level baseline。
```

建议 PR 标题：

```text
fix: align project phases and harden dataset diagnostic checks
```

---

## 1. 强制环境约束

正式运行环境为 Conda 环境：

```bash
moe
```

所有验收命令使用：

```bash
conda run -n moe python ...
conda run -n moe pytest ...
```

不要在 `base` 环境直接运行测试或脚本。

如需交互式调试：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate moe
```

---

## 2. 硬性禁止事项

本 PR 严禁：

```text
- 不下载新数据；
- 不抽取新的全量 feature bank；
- 不训练深度模型；
- 不训练 adapter / LoRA；
- 不实现 model merging；
- 不实现 MoE routing；
- 不实现 feature-level merge-vs-route baselines；
- 不生成 phase diagram；
- 不提交 data/、features/、artifacts/、models/、checkpoints/ 等大文件。
```

本 PR 只允许做：

```text
- API 对齐；
- import 修复；
- docs / README / AGENTS 更新；
- split/manifest/feature-bank 协议修复；
- diagnostics CLI 质量门槛；
- pytest / smoke check 修复；
- PROJECT_STATUS.md。
```

---

## 3. 先运行基线检查并记录

在修改前先运行：

```bash
cd /root/rivermind-data/projects/merge-route-boundary

conda run -n moe pytest -q

conda run -n moe python scripts/inspect_datasets.py --no-status-json

conda run -n moe python scripts/verify_task_streams.py \
  --stream controlled_cifar100_split_10x10 \
  --seed 0 \
  --skip-batch
```

如果有失败，先不要大改架构，记录失败类型。优先修复 ImportError、API mismatch、路径协议不一致、CLI exit code 不合理等基础问题。

---

## 4. 修复项 A：Registry API 与下游 import 对齐

### 4.1 问题

当前下游文件可能依赖以下 registry API：

```python
dataset_size_bytes
inspect_dataset
inspect_registry
check_project_data_symlink
```

例如：

```text
mrb/data/fingerprint.py
scripts/inspect_datasets.py
tests/test_dataset_registry.py
```

但 `mrb/data/registry.py` 中可能没有完整导出这些函数，导致 fresh clone 后出现 ImportError。

### 4.2 修复要求

在 `mrb/data/registry.py` 中补齐并测试以下函数，或统一修改所有 import 到现有 API。建议补齐函数，保持脚本接口稳定。

#### `dataset_size_bytes`

```python
def dataset_size_bytes(
    registry: DatasetRegistry,
    entry: DatasetEntry,
    *,
    root_override: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    ...
```

行为：

```text
- 基于 dataset_storage_paths(registry, entry) 统计磁盘占用；
- 对缺失路径返回 0；
- 使用 directory_size_bytes；
- 不 follow symlink 进入无限递归。
```

#### `check_project_data_symlink`

```python
def check_project_data_symlink(project_data: Path, data_root: Path) -> dict[str, Any]:
    ...
```

输出至少包含：

```json
{
  "mode": "full_symlink | missing | directory | other_symlink | file",
  "path": "...",
  "target": "... or null",
  "expected_target": "...",
  "warnings": []
}
```

判断标准：

```text
- project_data 是 symlink 且 resolve 后等于 data_root => mode=full_symlink；
- project_data 不存在 => mode=missing；
- project_data 是普通目录 => mode=directory；
- project_data 是 symlink 但指向其他位置 => mode=other_symlink；
- 其他情况 => mode=file 或 unknown。
```

#### `inspect_dataset`

```python
def inspect_dataset(
    registry: DatasetRegistry,
    entry: DatasetEntry,
    *,
    data_root: Path | None = None,
    env: Mapping[str, str] | None = None,
    load_smoke_metadata: bool = False,
) -> dict[str, Any]:
    ...
```

输出至少包含：

```text
dataset_id
tier
source
available
expected_available
disk_usage_bytes
loader_status
splits
split_sizes
warnings
```

要求：

```text
- 不下载数据；
- 对 loader_status=inspect_only 只做 storage inspection；
- load_smoke_metadata=True 时可以尝试读取 dataset length / split size，但失败时只记录 warning。
```

#### `inspect_registry`

```python
def inspect_registry(
    registry: DatasetRegistry,
    *,
    data_root: Path | None = None,
    load_smoke_metadata: bool = False,
) -> list[dict[str, Any]]:
    ...
```

要求：

```text
- 对 registry.datasets.values() 逐个调用 inspect_dataset；
- 不因为某个 dataset 失败而中断全部；
- 单个失败要记录 warning。
```

### 4.3 测试要求

新增或修复测试：

```text
tests/test_dataset_registry.py
tests/test_data_paths.py
```

至少覆盖：

```text
- registry_config_loads；
- env path resolution；
- missing dataset reports missing；
- dataset_size_bytes 对缺失路径返回 0；
- check_project_data_symlink 对 full symlink / missing / directory 有正确 mode；
- inspect_registry 对 inspect_only dataset 不报错。
```

---

## 5. 修复项 B：README / AGENTS / docs 阶段描述对齐

### 5.1 问题

当前 README 和 AGENTS 仍把项目描述为 initial setup only。但当前仓库已经包含：

```text
Dataset Registry + Task Stream
Frozen Feature Bank
Boundary Diagnostics v0
```

继续保留旧描述会误导后续 Codex。

### 5.2 修复要求

更新：

```text
README.md
AGENTS.md
docs/SETUP.md
docs/RESOURCE_POLICY.md
```

新增：

```text
docs/PROJECT_STATUS.md
```

### 5.3 README 应包含

建议结构：

```text
# merge-route-boundary

## Project Goal
研究 continual learning、model merging、MoE routing 之间的 merge-route boundary。

## Current Phase
当前已完成：
- Conda-only environment
- Dataset registry and task streams
- Frozen feature bank infrastructure
- Boundary diagnostics v0

当前未进入：
- adapter / LoRA training
- model merging
- MoE routing
- feature-level merge-vs-route baselines

## Server Layout
- MRB_DATA_ROOT=/root/rivermind-data/datasets
- MRB_FEATURE_ROOT=/root/rivermind-data/datasets/_derived/merge-route-boundary/features
- project data symlink: data -> /root/rivermind-data/datasets
- project feature symlink: features -> MRB_FEATURE_ROOT

## Required Checks
conda run -n moe pytest -q
...
```

### 5.4 AGENTS.md 应更新

AGENTS 中保留硬约束，但阶段描述改成：

```text
Current allowed scope:
- bug fixes and QA for data / feature / diagnostics layers
- docs/status alignment
- smoke checks

Still forbidden unless explicitly requested:
- deep model training
- adapter / LoRA
- model merging
- MoE routing
- feature-level baselines
- phase diagrams
```

并明确：

```text
All commands should use conda run -n moe ...
```

### 5.5 PROJECT_STATUS.md 内容要求

新增 `docs/PROJECT_STATUS.md`，包含：

```text
1. Project goal
2. Current branch / phase
3. Completed layers
   - Phase 0: Conda environment and resource management
   - Phase 1: Dataset registry and task streams
   - Phase 2: Frozen feature bank
   - Phase 3: Boundary diagnostics v0
4. Local server layout
5. Required environment variables
6. Current validation commands
7. Known risks / open issues
8. Next recommended step after this QA PR
```

Known risks 至少写：

```text
- no_split datasets need deterministic holdout policy before formal evaluation;
- feature diagnostics are feature-side proxies, not parameter-level incompatibility I;
- diagnostics quality depends on full feature bank rather than smoke feature bank;
- artifacts/diagnostics are local and ignored by git.
```

---

## 6. 修复项 C：统一 Feature Bank canonical layout

### 6.1 问题

当前 feature 写入路径和读取候选路径可能同时存在两种 layout：

```text
A: MRB_FEATURE_ROOT/<dataset_id>/<split>/<backbone_id>/
B: MRB_FEATURE_ROOT/<backbone_id>/<dataset_id>/<split>/
```

为了避免后续混乱，需要明确 canonical layout。

### 6.2 修复要求

将 canonical layout 明确为当前写入逻辑：

```text
MRB_FEATURE_ROOT/<dataset_id>/<split>/<backbone_id>/
```

要求：

```text
- `mrb/features/storage.py::feature_bank_dir` 保持 canonical layout；
- `mrb/diagnostics/feature_io.py` 可以继续兼容 legacy layout，但必须优先 canonical layout；
- README / PROJECT_STATUS / docs 中写明 canonical layout；
- `inspect_feature_bank.py` 输出 layout 时使用 canonical 表述；
- 测试覆盖 canonical path 和 legacy compatibility。
```

### 6.3 测试要求

新增或修复：

```text
tests/test_feature_storage.py
tests/test_feature_diagnostics.py
```

至少验证：

```text
- feature_bank_dir(root, dataset=cifar10, split=train, backbone=clip_vit_b32)
  == root/cifar10/train/clip_vit_b32
- load_feature_bank 优先读取 canonical layout
- legacy layout 存在时仍可读取，但 docs 不把 legacy 当默认
```

---

## 7. 修复项 D：class_order_seed 语义对齐

### 7.1 问题

`configs/task_streams/controlled.yaml` 中包含：

```yaml
class_order_seed: 0
```

但当前 class-incremental split 构建可能只使用 CLI 的 `seed`，没有真正读取 `class_order_seed`。这会造成实验语义混乱。

### 7.2 目标语义

建议采用：

```text
class_order_seed: 控制 class order；
seed: 控制 train/val split、sampling、manifest seed。
```

具体：

```python
class_order_seed = int(config.get("class_order_seed", seed))
split_seed = seed
```

### 7.3 实现要求

修改：

```text
mrb/data/task_streams.py
mrb/data/splits.py
```

让 `build_class_incremental_splits` 支持：

```python
class_order_seed: int
split_seed: int
```

或保持函数参数最小化，但内部明确区分。

Manifest 中应记录：

```json
{
  "class_order_seed": 0,
  "split_seed": 0
}
```

可以在 task-level 或 top-level `stream_config` 中记录，但必须可追踪。

### 7.4 测试要求

新增测试：

```text
- 同一个 class_order_seed、不同 seed：class order 不变，但 train/val split 可变；
- 不同 class_order_seed、同一个 seed：class order 改变；
- manifest 中能看到 class_order_seed / split_seed。
```

---

## 8. 修复项 E：no_split 数据集的 deterministic holdout policy

### 8.1 问题

`EuroSAT`、`Caltech101`、`SUN397` 等可能是 no_split dataset。当前 dataset-incremental stream 如果把同一个 `all` split 同时当 train/test，会造成后续评估泄漏。

### 8.2 推荐策略

本 PR 先实现明确协议，不一定要支持所有 no_split dataset 的正式训练。

推荐加入：

```yaml
split_policy: deterministic_holdout
holdout:
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
```

适用于：

```text
loader: no_split
splits: [all]
```

### 8.3 实现要求

在 dataset-incremental 构建中：

```text
- 如果 train_split 和 test_split 指向同一个 source split，例如 all/all；
- 且 entry.loader == no_split；
- 则必须使用 deterministic holdout 生成 train/val/test indices；
- source split 应记录为 all；
- task manifest 中记录 split_policy=deterministic_holdout；
- train_indices / val_indices / test_indices 互不重叠。
```

如果不实现 holdout，则必须把 no_split dataset 从默认 real stream 中禁用，并在 docs/PROJECT_STATUS.md 中记录原因。建议实现 holdout，因为后续 EuroSAT/Caltech101 有研究价值。

### 8.4 注意 feature bank 映射

如果 no_split 的 feature bank source split 是：

```text
all
```

那么 diagnostics 的 logical split 映射需要支持：

```text
logical_split=train -> feature_bank_split=all, index_field=train_indices
logical_split=val   -> feature_bank_split=all, index_field=val_indices
logical_split=test  -> feature_bank_split=all, index_field=test_indices
```

不能固定认为 train/val 都来自 `train`，test 来自 `test`。

### 8.5 测试要求

新增测试：

```text
- no_split synthetic dataset 经过 deterministic_holdout 后 train/val/test 非空且互不重叠；
- 同 seed 可复现；
- 不同 seed split_hash 改变；
- diagnostics logical split 能映射到 source_split=all。
```

---

## 9. 修复项 F：diagnostics CLI 质量门槛

### 9.1 问题

当前 `compute_boundary_diagnostics.py` 可能在全部 pair failed 时仍返回 exit code 0。这样 CI / smoke check 会误判成功。

### 9.2 修复要求

为 CLI 增加参数：

```text
--fail-on-any-failed-pair
--min-ok-ratio FLOAT
```

建议默认行为：

```text
strict_features=true:
  min_ok_ratio 默认为 1.0
  failed_pairs > 0 返回非零 exit code

allow_partial_features:
  min_ok_ratio 默认为 0.8
  输出 warning，但达到阈值可返回 0
```

也可更简单：

```text
--strict-features 时，只要 failed_pairs > 0，return 1。
```

### 9.3 输出要求

CLI 打印：

```text
rows=<N> ok=<K> failed=<F> ok_ratio=<K/N>
```

如果失败：

```text
ERROR boundary diagnostics quality gate failed:
  failed_pairs=...
  min_ok_ratio=...
  top warnings:
  - ...
```

### 9.4 测试要求

新增测试：

```text
- 全部 ok -> exit 0；
- 有 failed pair 且 --fail-on-any-failed-pair -> exit 1；
- ok_ratio < min_ok_ratio -> exit 1；
- allow_partial_features 下达到阈值 -> exit 0。
```

---

## 10. 修复项 G：diagnostics config 需要被真正使用

### 10.1 问题

`configs/diagnostics/boundary_v0.yaml` 中包含：

```yaml
linear_probe:
  enabled: true
  test_size: 0.3
  max_iter: 1000
  class_weight: balanced
knn:
  enabled: true
  k: 5
```

但当前 `compute_pairwise_boundary_matrix` 未必真正把这些 config 参数传入 separability 函数。

### 10.2 修复要求

最小修复：

```text
- compute_boundary_diagnostics.py 读取 config；
- 将 linear_probe.test_size / max_iter / class_weight 传入；
- 将 knn.k 传入；
- 将 enabled=false 时对应 metric 置为 nan 并记录 warning/config status；
- summary.json 记录 config_path 和 config hash。
```

如果暂时不想扩展函数签名，至少在 docs/PROJECT_STATUS.md 中记录：

```text
boundary_v0.yaml 目前只有部分字段生效。
```

但建议本 PR 直接修复，让 config 生效。

### 10.3 测试要求

```text
- linear_probe.enabled=false 时 linear metrics 为 nan 或 skipped；
- knn.k 修改后传入 knn_separability；
- summary.json 记录 config_hash。
```

---

## 11. 修复项 H：no large artifacts / gitignore 再确认

检查 `.gitignore` 是否包含：

```gitignore
/data/
/features/
/artifacts/
/models/
/cache/
/checkpoints/
/outputs/
*.npy
*.npz
*.pt
*.pth
*.ckpt
*.safetensors
*.bin
```

要求：

```text
- 不提交 artifacts/dataset_status.json；
- 不提交 artifacts/manifests/*.json；
- 不提交 artifacts/diagnostics/*.csv；
- 不提交 feature banks；
- 可以提交 configs、scripts、tests、docs。
```

---

## 12. 最终验收命令

在服务器上运行：

```bash
cd /root/rivermind-data/projects/merge-route-boundary

conda run -n moe pytest -q
```

数据层：

```bash
conda run -n moe python scripts/inspect_datasets.py --no-status-json

conda run -n moe python scripts/verify_task_streams.py \
  --stream controlled_cifar100_split_10x10 \
  --seed 0 \
  --skip-batch

conda run -n moe python scripts/verify_task_streams.py \
  --stream real_tier0_tier2_small \
  --seed 0 \
  --skip-batch
```

manifest：

```bash
mkdir -p artifacts/manifests/splits

conda run -n moe python scripts/build_split_manifest.py \
  --stream controlled_cifar100_split_10x10 \
  --seed 0 \
  --output artifacts/manifests/splits/controlled_cifar100_split_10x10_seed0.json
```

feature bank：

```bash
conda run -n moe python scripts/verify_feature_bank.py \
  --dataset cifar100 \
  --split train \
  --backbone clip_vit_b32
```

diagnostics：

```bash
conda run -n moe python scripts/compute_boundary_diagnostics.py \
  --stream controlled_cifar100_split_10x10 \
  --seed 0 \
  --backbone clip_vit_b32 \
  --split train \
  --max-samples-per-task 500 \
  --strict-features \
  --fail-on-any-failed-pair \
  --overwrite
```

inspect/export：

```bash
conda run -n moe python scripts/inspect_boundary_diagnostics.py

conda run -n moe python scripts/export_boundary_matrix.py \
  --input artifacts/diagnostics/boundary_v0/controlled_cifar100_split_10x10_seed0_clip_vit_b32_train.csv \
  --metric linear_probe_auc_symmetric
```

最低通过标准：

```text
- pytest 全部通过；
- inspect_datasets.py 能运行；
- controlled_cifar100_split_10x10 task stream 验证通过；
- real_tier0_tier2_small 要么验证通过，要么明确跳过/禁用 no_split 风险数据集；
- verify_feature_bank.py 能验证本地已有 feature bank；
- compute_boundary_diagnostics.py 在 strict mode 下不应静默成功但全部 failed；
- docs/PROJECT_STATUS.md 已更新；
- git status 不包含 data/features/artifacts 大文件。
```

---

## 13. Codex 最终回复要求

完成后请 Codex 回复：

```text
1. 修改了哪些文件；
2. 修复了哪些 import/API 不一致；
3. no_split 数据集采用了什么策略；
4. feature bank canonical layout 是什么；
5. class_order_seed / seed 的最终语义；
6. diagnostics CLI 失败门槛是什么；
7. 运行了哪些验收命令，结果如何；
8. 仍有哪些已知限制；
9. 下一步是否可以进入 Feature-level Merge-vs-Route Toy Baselines。
```

---

## 14. 完成后下一步

如果本 PR 通过上述验收，下一阶段才进入：

```text
Feature-level Merge-vs-Route Toy Baselines
```

届时再实现：

```text
- merge-all prototype classifier
- task-specific classifier + oracle route
- task-specific classifier + learned route
- cluster-merge classifier
- route-vs-merge gap table
```

本 PR 不做这些内容。
