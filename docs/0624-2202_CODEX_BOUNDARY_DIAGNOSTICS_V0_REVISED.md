> 时间戳：06/24 22:00（东八区）

# Codex 阶段任务指导：Boundary Diagnostics v0（修订版）

## 0. 修订说明

本版用于替换 `CODEX_BOUNDARY_DIAGNOSTICS_V0.md` 中若干容易导致 Codex 实现偏差的细节。核心修订点：

1. 明确本阶段只计算 **feature-side separability / overlap diagnostics**，不要声称已经得到真正的 task incompatibility `I`。
2. 增加 `logical_split -> feature_bank_split` 映射，避免 `val_indices` 被错误地映射到不存在的 `features/<dataset>/val/`。
3. 增加 feature bank 预检，避免 `max-samples-per-task=500` 但本地只有 smoke feature bank。
4. 统一 task manifest 字段：优先使用 `dataset_id`，兼容 `dataset`。
5. 说明 sklearn linear probe 属于 frozen-feature diagnostic probe，不是深度模型训练，也不保存模型权重。
6. 增加 multi-dataset label namespace 规则，避免不同数据集的 class id 冲突。
7. 增加 deterministic / stratified sampling、缺失 index、有效样本数、metric version 等工程细节。

---

## 1. 当前阶段定位

项目：`merge-route-boundary`

当前已完成：

- Conda 环境：正式运行环境为 `moe`
- 数据根目录：`/root/rivermind-data/datasets`
- 项目目录：`/root/rivermind-data/projects/merge-route-boundary`
- 项目数据软链接：`data -> /root/rivermind-data/datasets`
- Tier 0–2 数据集组织与验证
- Dataset Registry + Task Stream Layer
- Frozen Feature Bank Layer

本阶段目标：

```text
Task Stream Manifest
  + Frozen Feature Bank
  -> pairwise feature diagnostics
  -> input separability S
  -> feature overlap / prototype distance
  -> first feature-side boundary matrix
```

注意：本阶段尚未训练 adapter / LoRA / task-specific model，因此不计算真正的 parameter-level `I = task incompatibility`。可以输出 `feature_overlap_proxy`、`prototype_distance`、`domain_separability_S` 等指标，为后续 `I/S phase diagram` 做准备。

---

## 2. 环境约束

所有验收命令使用：

```bash
conda run -n moe python ...
conda run -n moe pytest ...
```

不要在 `base` 环境直接运行 pytest 或脚本。

如需交互式调试：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate moe
```

Codex 需要更新 `AGENTS.md` 或 `docs/SETUP.md`，说明正式运行环境为 `moe`。

---

## 3. 硬性禁止事项

本 PR 严禁：

```text
- 下载新数据集
- 训练深度模型
- 训练 adapter / LoRA
- 实现 model merging
- 实现 MoE routing
- 实现 continual learning 训练循环
- 生成 phase diagram 大图
- 提交 artifacts/、features/、data/ 下的大文件
```

允许使用 sklearn 的轻量 probe，例如 LogisticRegression、LinearSVC、kNN，但必须满足：

```text
- 只运行在 frozen features 上
- 只作为 diagnostic probe
- 不保存 probe 权重
- 不作为正式模型训练 baseline
- 默认启用 max_samples_per_task
```

---

## 4. 必须先做的 preflight check

在计算 diagnostics 之前，脚本必须检查 feature bank 是否存在且样本数足够。

示例：

```bash
conda run -n moe python scripts/verify_feature_bank.py \
  --dataset cifar100 \
  --split train \
  --backbone clip_vit_b32
```

如果目标 task stream 需要 `max-samples-per-task=500`，但本地 feature bank 只是 smoke extraction，例如只有 128 个样本，脚本应：

```text
1. 给出清晰错误或 warning；
2. 自动降级到可用样本数时必须在 output 中记录 effective_num_samples；
3. 不应静默产生看似完整但实际样本不足的 matrix。
```

建议在 `compute_boundary_diagnostics.py` 中增加：

```text
--min-samples-per-task 50
--allow-partial-features
--strict-features
```

默认建议：

```text
--strict-features true
```

---

## 5. 目录结构

请新增：

```text
mrb/
  diagnostics/
    __init__.py
    feature_io.py
    prototypes.py
    distances.py
    separability.py
    overlap.py
    matrices.py
    report.py

configs/
  diagnostics/
    boundary_v0.yaml

scripts/
  compute_boundary_diagnostics.py
  inspect_boundary_diagnostics.py
  export_boundary_matrix.py

tests/
  test_feature_diagnostics.py
  test_boundary_matrix.py
```

---

## 6. 输入约定

### 6.1 Feature Bank 位置

优先读取：

```bash
MRB_FEATURE_ROOT=/root/rivermind-data/datasets/_derived/merge-route-boundary/features
```

如果缺失，应报错：

```text
MRB_FEATURE_ROOT is not set. Please export it or create the project feature symlink.
```

不要硬编码用户 home 目录。

### 6.2 Feature Bank 格式

兼容：

```text
features/
  <backbone_id>/
    <dataset_id>/
      <source_split>/
        features.npy
        labels.npy
        indices.npy
        sample_ids.jsonl
        metadata.json
```

其中 `source_split` 通常是 torchvision 的原始 split，例如：

```text
train
test
```

### 6.3 Logical split 与 source split 映射

Task stream manifest 通常有逻辑 split：

```text
train_indices
val_indices
test_indices
```

但 feature bank 可能只有原始 split：

```text
train/
test/
```

因此必须显式实现映射：

```text
logical_split=train -> feature_bank_split=train, index_field=train_indices
logical_split=val   -> feature_bank_split=train, index_field=val_indices
logical_split=test  -> feature_bank_split=test,  index_field=test_indices
```

不要假设存在：

```text
features/<backbone>/<dataset>/val/
```

除非 metadata 明确说明前一阶段已经生成了 val feature bank。

### 6.4 Task Manifest 字段

优先使用：

```json
{
  "task_id": 0,
  "dataset_id": "cifar100",
  "classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  "train_indices": [],
  "val_indices": [],
  "test_indices": []
}
```

兼容旧字段：

```json
{
  "dataset": "cifar100"
}
```

但内部标准化后应统一成 `dataset_id`。

对于 dataset-incremental stream，`classes` 可能为空、缺失或表示全部类别。此时不要用 class filter 取样，应优先使用 manifest 中的 indices。

### 6.5 Manifest 读取策略

`artifacts/manifests/splits/*.json` 可能被 `.gitignore` 忽略，因此脚本不能只依赖仓库中已有 artifact。

`compute_boundary_diagnostics.py` 应支持：

```text
--manifest path/to/manifest.json
```

并且当 `--manifest` 未提供时，尝试通过已有 `mrb.data.task_streams` / `configs/task_streams/*.yaml` 构建或定位 manifest。

---

## 7. 模块实现要求

### 7.1 `mrb/diagnostics/feature_io.py`

实现：

```python
load_feature_bank(
    dataset_id: str,
    split: str,
    backbone_id: str,
    feature_root: str | Path | None = None,
) -> FeatureBank
```

建议 dataclass：

```python
@dataclass
class FeatureBank:
    features: np.ndarray
    labels: np.ndarray
    indices: np.ndarray
    metadata: dict
    dataset_id: str
    split: str
    backbone_id: str
```

实现：

```python
validate_feature_bank(bank: FeatureBank) -> None
select_by_indices(
    bank: FeatureBank,
    indices: Sequence[int],
    *,
    strict: bool = True,
) -> FeatureBank

select_by_classes(
    bank: FeatureBank,
    classes: Sequence[int],
) -> FeatureBank
```

要求：

```text
- 计算前将 features 转为 fp32；
- 检查 NaN / Inf；
- 检查 features / labels / indices 长度一致；
- 检查 metadata.feature_dim 与 features.shape[1] 一致；
- 使用 bank.indices 建立原始 index -> row position 映射；
- 不要假设 row position 等于 dataset index；
- 若 manifest indices 中有缺失，strict=true 时直接报错；strict=false 时记录 missing count。
```

---

### 7.2 `mrb/diagnostics/prototypes.py`

实现：

```python
l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray

compute_class_prototypes(
    features: np.ndarray,
    labels: np.ndarray,
    normalize: bool = True,
) -> dict[int, np.ndarray]

compute_task_prototype(
    features: np.ndarray,
    normalize: bool = True,
) -> np.ndarray
```

对于 multi-dataset prototypes，不要只用裸 `class_id` 作为全局 key。应使用：

```text
(dataset_id, class_id)
```

或在每个 task 内部局部计算 class prototypes，避免 CIFAR10 class 0 与 SVHN class 0 被误认为同一类。

---

### 7.3 `mrb/diagnostics/distances.py`

实现：

```python
cosine_distance(a: np.ndarray, b: np.ndarray) -> float
l2_distance(a: np.ndarray, b: np.ndarray) -> float
pairwise_cosine_matrix(prototypes: Sequence[np.ndarray]) -> np.ndarray
intra_task_variance(features: np.ndarray, prototype: np.ndarray) -> float
inter_task_distance(proto_i: np.ndarray, proto_j: np.ndarray) -> dict[str, float]
```

建议指标：

```text
centroid_l2
centroid_cosine_distance
intra_variance_i
intra_variance_j
separation_ratio = centroid_l2 / sqrt(intra_var_i + intra_var_j + eps)
```

---

### 7.4 `mrb/diagnostics/separability.py`

实现三类 frozen-feature separability metrics。

#### A. Centroid separability

```python
centroid_separability(features_i, features_j) -> dict
```

输出：

```text
centroid_l2
centroid_cosine_distance
separation_ratio
```

#### B. Linear probe separability

```python
linear_probe_separability(
    features_i: np.ndarray,
    features_j: np.ndarray,
    seed: int = 0,
    max_samples_per_task: int | None = None,
    test_size: float = 0.3,
) -> dict
```

要求：

```text
- 使用 deterministic StratifiedShuffleSplit；
- domain label: task_i=0, task_j=1；
- 推荐 LogisticRegression(max_iter=1000, class_weight="balanced")；
- 只输出 diagnostic scores，不保存模型；
- 如果样本数不足，返回 nan 并给出 reason；
- 输出 effective_num_samples_i/j。
```

输出：

```text
linear_probe_auc
linear_probe_auc_symmetric
linear_probe_acc
linear_probe_num_train
linear_probe_num_eval
```

其中：

```text
linear_probe_auc_symmetric = max(auc, 1 - auc)
```

避免小样本或标签方向导致 AUROC 低于 0.5 时误判可分性。

#### C. kNN separability

```python
knn_separability(
    features_i: np.ndarray,
    features_j: np.ndarray,
    k: int = 5,
    seed: int = 0,
    max_samples_per_task: int | None = None,
) -> dict
```

输出：

```text
knn_domain_acc
knn_domain_auc 或 nan
```

---

### 7.5 `mrb/diagnostics/overlap.py`

实现：

```python
nearest_task_centroid_confusion(features_i, features_j, proto_i, proto_j) -> dict
prototype_margin(features, own_proto, other_proto) -> dict
nearest_class_prototype_confusion(features, labels, class_prototypes) -> dict
```

输出：

```text
nearest_task_centroid_acc
task_i_to_j_confusion_rate
task_j_to_i_confusion_rate
mean_prototype_margin_i
mean_prototype_margin_j
min_prototype_margin_i
min_prototype_margin_j
```

注意：

```text
- 如果 prototypes 与 evaluated features 来自同一批样本，应在 metadata 中记录 resubstitution=true；
- 若要更严格，可后续用 train prototypes + val/test features，但本 PR 先不强制。
```

---

### 7.6 `mrb/diagnostics/matrices.py`

核心函数：

```python
compute_pairwise_boundary_matrix(
    stream_id: str,
    seed: int,
    backbone_id: str,
    split: str = "train",
    max_samples_per_task: int | None = 500,
    output_dir: Path | None = None,
    manifest_path: Path | None = None,
    strict_features: bool = True,
) -> pd.DataFrame
```

每个 task pair 输出一行：

```csv
stream_id,
seed,
backbone_id,
logical_split,
feature_bank_split_i,
feature_bank_split_j,
task_i,
task_j,
dataset_i,
dataset_j,
num_samples_i,
num_samples_j,
effective_num_samples_i,
effective_num_samples_j,
num_classes_i,
num_classes_j,
centroid_l2,
centroid_cosine_distance,
separation_ratio,
linear_probe_auc,
linear_probe_auc_symmetric,
linear_probe_acc,
knn_domain_acc,
nearest_task_centroid_acc,
task_i_to_j_confusion_rate,
task_j_to_i_confusion_rate,
mean_margin_i,
mean_margin_j,
resubstitution,
metric_version,
status,
warning
```

要求：

```text
- T 个 task 输出 T*(T-1)/2 行；
- 某个 pair 失败不能导致全表失败；
- failed pair 要记录 status=failed 和 warning；
- ok pair 记录 status=ok；
- summary.json 需要记录 feature metadata hash / manifest path / config path。
```

---

### 7.7 `mrb/diagnostics/report.py`

实现：

```python
summarize_boundary_matrix(df: pd.DataFrame) -> dict
```

summary 至少包含：

```text
num_tasks
num_pairs
num_ok_pairs
num_failed_pairs
mean_linear_probe_auc_symmetric
mean_knn_domain_acc
mean_centroid_l2
mean_separation_ratio
most_separable_pairs
least_separable_pairs
feature_bank_metadata
manifest_path
metric_version
```

---

## 8. 配置文件

新增：

```text
configs/diagnostics/boundary_v0.yaml
```

建议内容：

```yaml
diagnostics:
  default_split: train
  max_samples_per_task: 500
  min_samples_per_task: 50
  strict_features: true
  normalize_features: true
  sample_seed: 0
  linear_probe:
    enabled: true
    test_size: 0.3
    max_iter: 1000
    class_weight: balanced
  knn:
    enabled: true
    k: 5
  silhouette:
    enabled: false
    max_samples_total: 2000

outputs:
  root: artifacts/diagnostics/boundary_v0
  save_csv: true
  save_summary_json: true
```

---

## 9. CLI 脚本

### 9.1 `scripts/compute_boundary_diagnostics.py`

示例：

```bash
conda run -n moe python scripts/compute_boundary_diagnostics.py \
  --stream controlled_cifar100_split_10x10 \
  --seed 0 \
  --backbone clip_vit_b32 \
  --split train \
  --max-samples-per-task 500 \
  --strict-features
```

参数至少包括：

```text
--stream
--seed
--backbone
--split
--max-samples-per-task
--min-samples-per-task
--manifest
--output-dir
--config
--strict-features / --allow-partial-features
--overwrite
```

### 9.2 `scripts/inspect_boundary_diagnostics.py`

功能：

```text
- 列出已有 diagnostics 文件；
- 打印 task pair 数；
- 打印 mean AUROC-symmetric / kNN acc / separation ratio；
- 打印 most/least separable pairs；
- 显示 failed pair 数量与 warning 摘要。
```

### 9.3 `scripts/export_boundary_matrix.py`

示例：

```bash
conda run -n moe python scripts/export_boundary_matrix.py \
  --input artifacts/diagnostics/boundary_v0/controlled_cifar100_split_10x10_seed0_clip_vit_b32_train.csv \
  --metric linear_probe_auc_symmetric
```

---

## 10. 测试要求

新增：

```text
tests/test_feature_diagnostics.py
tests/test_boundary_matrix.py
```

测试只用 synthetic features 或 tiny arrays，不依赖真实数据、不下载数据、不跑大模型。

至少测试：

```text
- l2_normalize 输出单位范数；
- compute_class_prototypes 输出类别数正确；
- cosine_distance 范围合理；
- linear_probe_separability 对可分 synthetic data 给出高 AUROC-symmetric；
- nearest_task_centroid_confusion 对明显可分数据给出高 accuracy；
- 2 个 synthetic task 生成 1 行 pairwise matrix；
- 3 个 synthetic task 生成 3 行 pairwise matrix；
- 缺失 task / feature 时不导致全表崩溃，而是记录 failed status；
- logical_split=val 正确映射到 feature_bank_split=train；
- indices.npy 顺序打乱时仍能按原始 index 正确选择样本。
```

验收：

```bash
conda run -n moe pytest tests/test_feature_diagnostics.py tests/test_boundary_matrix.py
```

---

## 11. 输出与 Git 规则

允许生成但不要提交：

```text
artifacts/diagnostics/
```

`.gitignore` 应包含：

```gitignore
artifacts/
features/
data/
*.npy
*.npz
*.pt
*.pth
*.ckpt
```

可以提交：

```text
configs/diagnostics/boundary_v0.yaml
mrb/diagnostics/*.py
scripts/*.py
tests/*.py
docs/*.md
```

---

## 12. 最小验收流程

```bash
cd /root/rivermind-data/projects/merge-route-boundary

conda run -n moe python scripts/compute_boundary_diagnostics.py \
  --stream controlled_cifar100_split_10x10 \
  --seed 0 \
  --backbone clip_vit_b32 \
  --split train \
  --max-samples-per-task 500 \
  --strict-features

conda run -n moe python scripts/inspect_boundary_diagnostics.py

conda run -n moe python scripts/export_boundary_matrix.py \
  --input artifacts/diagnostics/boundary_v0/controlled_cifar100_split_10x10_seed0_clip_vit_b32_train.csv \
  --metric linear_probe_auc_symmetric

conda run -n moe pytest tests/test_feature_diagnostics.py tests/test_boundary_matrix.py
```

最低通过标准：

```text
- compute_boundary_diagnostics.py 成功输出 CSV 和 summary.json；
- controlled_cifar100_split_10x10 有 10 个 task，因此输出 45 个 task pairs；
- ok pair 的 linear_probe_auc_symmetric / centroid distance / kNN acc 字段存在；
- inspect 脚本显示最高/最低 separability pair；
- export 脚本生成 task × task matrix；
- pytest 通过。
```

---

## 13. 完成后的下一步

下一阶段建议做：

```text
Feature-level Merge-vs-Route Toy Baselines
```

用 frozen features 和简单 prototype / linear classifiers 模拟：

```text
1. merge-all classifier
2. task-specific classifier + oracle route
3. task-specific classifier + learned route
4. cluster-merge classifier
```

目的是在不训练大模型的前提下验证：

```text
high separability / low overlap -> routing advantage likely larger
low separability / high overlap -> static merge or shared classifier may be enough
```

---

## 14. Codex 最终回复要求

完成后请列出：

```text
1. 修改了哪些文件；
2. 新增了哪些 CLI 命令；
3. 测试是否通过；
4. 是否成功生成 boundary diagnostics CSV；
5. 输出文件路径；
6. failed pair / nan metric 的原因；
7. 当前 feature bank 是否为 full extraction 还是 smoke extraction。
```
