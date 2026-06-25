> 时间戳：06/24 21:56（东八区）

# Codex 阶段任务指导：Boundary Diagnostics v0（Frozen Feature Bank → I/S 边界诊断）

## 0. 当前阶段定位

项目：`merge-route-boundary`

当前已完成：

- Conda 环境：正式运行环境为 `moe`
- 数据根目录：`/root/rivermind-data/datasets`
- 项目目录：`/root/rivermind-data/projects/merge-route-boundary`
- 项目数据软链接：`data -> /root/rivermind-data/datasets`
- Tier 0–2 数据集组织与验证
- Dataset Registry + Task Stream Layer
- Frozen Feature Bank Layer

本阶段目标是实现 **Boundary Diagnostics v0**：

```text
Task Stream Manifest
  + Frozen Feature Bank
  -> pairwise task diagnostics
  -> input separability S
  -> feature overlap / prototype distance
  -> first I-S boundary matrix
```

本阶段只做 **frozen feature 上的统计诊断**，不进入训练、adapter、LoRA、model merging、MoE routing 或 phase diagram 绘图。

---

## 1. 研究目标说明

本项目的论文主题是研究：

```text
When to Merge, When to Route?
```

即在 continual learning / model merging / MoE routing 之间建立可量化边界。

后续完整边界需要两个核心变量：

```text
I = task incompatibility
S = input separability
```

当前阶段还没有训练 task adapters 或 LoRA，因此暂时无法计算 parameter-level 的 task-vector conflict、sign conflict、Fisher overlap 等真正的 `I`。因此本阶段先实现 **feature-side boundary diagnostics**，用 frozen features 计算：

- 输入分布可分性 `S`
- task/domain overlap
- class/task prototype distance
- nearest-prototype confusion
- feature-space task distance matrix

这些结果将为下一阶段的 simple merge-vs-route prototype baselines 和后续 adapter-level boundary analysis 提供基础。

---

## 2. 强制环境约束

后续所有命令必须使用 `moe` Conda 环境运行：

```bash
conda run -n moe python ...
conda run -n moe pytest ...
```

不要在 `base` 环境直接运行 pytest 或脚本。

如需在交互式 shell 中调试：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate moe
```

Codex 应更新 `AGENTS.md` 或相关 docs，明确正式运行环境为 `moe`。

---

## 3. 硬性禁止事项

本 PR 严禁实现或运行以下内容：

```text
- 不下载新数据集
- 不抽取新的全量 feature bank，除非已有 smoke feature 缺失且用户明确要求
- 不训练深度模型
- 不训练 adapter / LoRA
- 不实现 model merging
- 不实现 MoE routing
- 不实现 continual learning 训练循环
- 不生成 phase diagram 大图
- 不提交 artifacts/、features/、data/ 下的大文件
```

本阶段允许使用 `sklearn` 中的轻量 probe，例如 LogisticRegression、LinearSVC、kNN，但它们只能运行在 frozen features 上，并且只作为诊断指标，不作为正式模型训练方法。

---

## 4. 本阶段 PR 名称

建议 PR 标题：

```text
feat: add boundary diagnostics from frozen feature banks
```

PR 范围：

```text
mrb/diagnostics/*
configs/diagnostics/boundary_v0.yaml
scripts/compute_boundary_diagnostics.py
scripts/inspect_boundary_diagnostics.py
scripts/export_boundary_matrix.py
tests/test_feature_diagnostics.py
tests/test_boundary_matrix.py
docs/CODEX_BOUNDARY_DIAGNOSTICS_V0.md
```

---

## 5. 目录结构要求

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

可选新增：

```text
docs/
  CODEX_BOUNDARY_DIAGNOSTICS_V0.md
```

---

## 6. 输入数据约定

### 6.1 Feature Bank 位置

默认读取：

```bash
MRB_FEATURE_ROOT=/root/rivermind-data/datasets/_derived/merge-route-boundary/features
```

如果环境变量缺失，应给出清晰错误信息：

```text
MRB_FEATURE_ROOT is not set. Please export it or create the project feature symlink.
```

不要硬编码用户 home 目录。可以在默认配置中建议路径，但实际脚本应优先读取环境变量。

### 6.2 Feature Bank 格式

应兼容前一阶段生成的格式：

```text
features/
  <backbone_id>/
    <dataset_id>/
      <split>/
        features.npy
        labels.npy
        indices.npy
        sample_ids.jsonl
        metadata.json
```

每个 split 至少包含：

```text
features.npy       # shape: [N, D]
labels.npy         # shape: [N]
indices.npy        # shape: [N], dataset 原始 index
sample_ids.jsonl
metadata.json
```

### 6.3 Task Stream Manifest

从此前阶段生成的 split manifest 读取 task 信息，例如：

```text
artifacts/manifests/splits/
  controlled_cifar100_split_10x10_seed0.json
```

Manifest 中每个 task 至少应包含：

```json
{
  "task_id": 0,
  "dataset": "cifar100",
  "classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  "train_indices": [...],
  "val_indices": [...],
  "test_indices": [...]
}
```

如果当前实际 manifest 字段命名略有不同，请在 `mrb/diagnostics/matrices.py` 中做兼容读取，但不要破坏原有 manifest schema。

---

## 7. 模块实现要求

### 7.1 `mrb/diagnostics/feature_io.py`

职责：读取 feature bank 并按 task manifest 筛选样本。

需要实现：

```python
load_feature_bank(
    dataset_id: str,
    split: str,
    backbone_id: str,
    feature_root: str | Path | None = None,
) -> FeatureBank
```

建议定义数据结构：

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

需要实现：

```python
validate_feature_bank(bank: FeatureBank) -> None
select_by_indices(bank: FeatureBank, indices: Sequence[int]) -> FeatureBank
select_by_classes(bank: FeatureBank, classes: Sequence[int]) -> FeatureBank
```

要求：

- `features.npy` 可为 fp16，但计算前应可转换为 fp32。
- 检查 NaN / Inf。
- 检查 `features.shape[0] == labels.shape[0] == indices.shape[0]`。
- 检查 metadata 中的 `feature_dim` 与 `features.shape[1]` 一致。
- 对 manifest index 和 feature bank `indices.npy` 做可靠映射，不要假设数组顺序等于原始 index。

---

### 7.2 `mrb/diagnostics/prototypes.py`

职责：计算 class/task prototypes。

需要实现：

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

compute_task_class_prototypes(
    task_features: np.ndarray,
    task_labels: np.ndarray,
    normalize: bool = True,
) -> dict[int, np.ndarray]
```

注意：

- class prototype 用每类样本均值。
- task prototype 用 task 内所有样本均值。
- 若某类样本为空，应跳过并记录 warning，不要直接崩溃。
- 输出向量默认 L2 normalize，便于 cosine distance。

---

### 7.3 `mrb/diagnostics/distances.py`

职责：提供 feature/prototype 距离指标。

至少实现：

```python
cosine_distance(a: np.ndarray, b: np.ndarray) -> float
l2_distance(a: np.ndarray, b: np.ndarray) -> float
pairwise_cosine_matrix(prototypes: Sequence[np.ndarray]) -> np.ndarray
intra_task_variance(features: np.ndarray, prototype: np.ndarray) -> float
inter_task_distance(proto_i: np.ndarray, proto_j: np.ndarray) -> dict[str, float]
```

输出指标至少包括：

```text
task_centroid_cosine_distance
task_centroid_l2_distance
intra_variance_i
intra_variance_j
variance_ratio_or_separation_ratio
```

建议计算：

```text
separation_ratio = inter_centroid_l2 / sqrt(intra_var_i + intra_var_j + eps)
```

这个指标可作为 feature-side separability 的粗略 proxy。

---

### 7.4 `mrb/diagnostics/separability.py`

职责：估计 input separability `S`。

至少实现三类指标。

#### A. Centroid-based separability

```python
centroid_separability(features_i, features_j) -> dict
```

输出：

```text
centroid_l2
centroid_cosine_distance
separation_ratio
```

#### B. Linear probe AUROC / accuracy

```python
linear_probe_separability(
    features_i: np.ndarray,
    features_j: np.ndarray,
    seed: int = 0,
    max_samples_per_task: int | None = None,
) -> dict
```

实现要求：

- 只用 frozen features。
- domain label：task_i = 0，task_j = 1。
- 使用 deterministic train/validation split。
- 推荐 `sklearn.linear_model.LogisticRegression`。
- 若样本数太少或类别单一，应返回 `nan` 并记录 reason。
- 输出：

```text
linear_probe_auc
linear_probe_acc
linear_probe_num_train
linear_probe_num_eval
```

#### C. kNN domain classification

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

如果实现 AUROC 不稳定，可先只输出 accuracy。

注意：

- 所有 probe 都必须有 `max_samples_per_task` 限制。
- 特征计算用 fp32。
- 对大样本数据不要默认全量训练 sklearn probe。

---

### 7.5 `mrb/diagnostics/overlap.py`

职责：计算任务之间的 feature overlap 与 prototype confusion。

至少实现：

```python
nearest_task_centroid_confusion(features_i, features_j, proto_i, proto_j) -> dict
prototype_margin(features, own_proto, other_proto) -> dict
nearest_class_prototype_confusion(features, labels, class_prototypes) -> dict
```

输出建议：

```text
nearest_task_centroid_acc
task_i_to_j_confusion_rate
task_j_to_i_confusion_rate
mean_prototype_margin_i
mean_prototype_margin_j
min_prototype_margin_i
min_prototype_margin_j
```

解释：

- 如果 nearest-task-centroid accuracy 高，说明两个任务在 feature space 中容易区分。
- 如果 margin 低，说明样本靠近边界，后续 route 可能不稳定。
- 如果 confusion rate 高，说明任务高度重叠，静态 merge 或 shared learning 可能比 hard routing 更合适。

可选实现：

```python
silhouette_score_pair(...)
```

注意 silhouette 可能是 O(N²)，必须使用采样，默认最多几千个样本。

---

### 7.6 `mrb/diagnostics/matrices.py`

职责：根据 stream manifest 生成 pairwise task diagnostic matrix。

核心函数：

```python
compute_pairwise_boundary_matrix(
    stream_id: str,
    seed: int,
    backbone_id: str,
    split: str = "train",
    max_samples_per_task: int | None = 500,
    output_dir: Path | None = None,
) -> pd.DataFrame
```

每个 task pair 输出一行：

```csv
stream_id,
seed,
backbone_id,
split,
task_i,
task_j,
dataset_i,
dataset_j,
num_samples_i,
num_samples_j,
num_classes_i,
num_classes_j,
centroid_l2,
centroid_cosine_distance,
separation_ratio,
linear_probe_auc,
linear_probe_acc,
knn_domain_acc,
nearest_task_centroid_acc,
task_i_to_j_confusion_rate,
task_j_to_i_confusion_rate,
mean_margin_i,
mean_margin_j,
status,
warning
```

要求：

- 对 `T` 个 task，输出 `T * (T - 1) / 2` 行。
- 每个 pair 的异常不能导致整张表失败；应记录 `status=failed` 和 warning。
- 正常 pair 记录 `status=ok`。
- 输出同时保存 `.csv` 和 `summary.json`。

---

### 7.7 `mrb/diagnostics/report.py`

职责：输出 summary 和可读报告。

至少实现：

```python
summarize_boundary_matrix(df: pd.DataFrame) -> dict
```

summary 至少包含：

```text
num_tasks
num_pairs
num_ok_pairs
num_failed_pairs
mean_linear_probe_auc
mean_knn_domain_acc
mean_centroid_l2
mean_separation_ratio
most_separable_pairs
least_separable_pairs
```

---

## 8. 配置文件要求

新增：

```text
configs/diagnostics/boundary_v0.yaml
```

示例：

```yaml
diagnostics:
  default_split: train
  max_samples_per_task: 500
  normalize_features: true
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

## 9. 脚本要求

### 9.1 `scripts/compute_boundary_diagnostics.py`

示例命令：

```bash
conda run -n moe python scripts/compute_boundary_diagnostics.py \
  --stream controlled_cifar100_split_10x10 \
  --seed 0 \
  --backbone clip_vit_b32 \
  --split train \
  --max-samples-per-task 500
```

参数：

```text
--stream
--seed
--backbone
--split
--max-samples-per-task
--output-dir
--config
--overwrite
```

输出：

```text
artifacts/diagnostics/boundary_v0/
  <stream>_seed<seed>_<backbone>_<split>.csv
  <stream>_seed<seed>_<backbone>_<split>_summary.json
```

---

### 9.2 `scripts/inspect_boundary_diagnostics.py`

功能：

- 列出已有 diagnostics 文件；
- 打印每个文件的 task pair 数；
- 打印平均 AUROC / kNN acc / separation ratio；
- 打印最容易区分和最难区分的 pair。

示例：

```bash
conda run -n moe python scripts/inspect_boundary_diagnostics.py
```

---

### 9.3 `scripts/export_boundary_matrix.py`

功能：

- 将 CSV 转成更易读的 matrix 形式；
- 支持导出 `linear_probe_auc`、`centroid_l2`、`separation_ratio` 等指标的 task × task matrix。

示例：

```bash
conda run -n moe python scripts/export_boundary_matrix.py \
  --input artifacts/diagnostics/boundary_v0/controlled_cifar100_split_10x10_seed0_clip_vit_b32_train.csv \
  --metric linear_probe_auc
```

输出：

```text
artifacts/diagnostics/boundary_v0/matrices/
  controlled_cifar100_split_10x10_seed0_clip_vit_b32_train_linear_probe_auc.csv
```

---

## 10. 测试要求

新增：

```text
tests/test_feature_diagnostics.py
tests/test_boundary_matrix.py
```

测试必须使用 synthetic features 或 tiny arrays，不依赖真实大数据集，不下载数据，不跑大模型。

### 10.1 `test_feature_diagnostics.py`

至少测试：

```text
- l2_normalize 输出单位范数
- compute_class_prototypes 输出类别数正确
- cosine_distance 范围合理
- linear_probe_separability 对可分 synthetic data 给出高 AUROC
- nearest_task_centroid_confusion 对明显可分数据给出高 accuracy
```

### 10.2 `test_boundary_matrix.py`

至少测试：

```text
- 两个 synthetic task 能生成 1 行 pairwise matrix
- 三个 synthetic task 能生成 3 行 pairwise matrix
- 某个 task 缺失时不会导致整个 matrix 崩溃，而是记录 failed status
- 输出 CSV / summary JSON 字段完整
```

验收命令：

```bash
conda run -n moe pytest tests/test_feature_diagnostics.py tests/test_boundary_matrix.py
```

---

## 11. 输出文件与 Git 规则

允许生成但不要提交：

```text
artifacts/diagnostics/
```

`.gitignore` 应确认包含：

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

Codex 完成后，请按以下命令验证：

```bash
cd /root/rivermind-data/projects/merge-route-boundary

conda run -n moe python scripts/compute_boundary_diagnostics.py \
  --stream controlled_cifar100_split_10x10 \
  --seed 0 \
  --backbone clip_vit_b32 \
  --split train \
  --max-samples-per-task 500

conda run -n moe python scripts/inspect_boundary_diagnostics.py

conda run -n moe python scripts/export_boundary_matrix.py \
  --input artifacts/diagnostics/boundary_v0/controlled_cifar100_split_10x10_seed0_clip_vit_b32_train.csv \
  --metric linear_probe_auc

conda run -n moe pytest tests/test_feature_diagnostics.py tests/test_boundary_matrix.py
```

最低通过标准：

```text
- compute_boundary_diagnostics.py 能成功输出 CSV 和 summary.json
- controlled_cifar100_split_10x10 有 10 个 task，因此应输出 45 个 task pairs
- 所有 ok pair 的 linear_probe_auc / centroid distance / kNN acc 字段存在
- inspect 脚本能显示最高/最低 separability pair
- export 脚本能生成 task × task matrix
- pytest 通过
```

---

## 13. 完成后的下一步

本阶段完成后，不要马上训练 adapter。下一阶段建议做：

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

这样可以在不训练大模型的前提下先验证核心假设：

```text
low overlap / high separability -> routing advantage larger
high overlap / low separability -> static merge or shared classifier may be enough
```

---

## 14. Codex 最终回复要求

Codex 完成任务后，请在回复中列出：

```text
1. 修改了哪些文件；
2. 新增了哪些 CLI 命令；
3. 测试是否通过；
4. 是否成功生成 boundary diagnostics CSV；
5. 输出文件路径；
6. 已知限制，例如某些 metric 暂时为 nan 的情况。
```
