> 时间戳：06/26 15:17（东八区）

# Codex 阶段任务指导：Feature-level Merge-vs-Route Toy Baselines v0

## 0. 当前阶段定位

项目仓库：

```text
/root/rivermind-data/projects/merge-route-boundary
https://github.com/nordlyswang/merge-route-boundary
```

当前工作分支：

```text
codex/0625-dataset-registry
```

当前已完成：

```text
- Dataset Registry + Task Stream Layer
- Frozen Feature Bank Layer
- Boundary Diagnostics v0
- CIFAR100 train/test 的真实 CLIP ViT-B/32 feature bank
- controlled_cifar100_split_10x10 的第一张 boundary diagnostics matrix
```

当前 feature bank：

```text
MRB_FEATURE_ROOT/cifar100/train/clip_vit_b32/
  N=50000, feature_dim=512, dtype=float16

MRB_FEATURE_ROOT/cifar100/test/clip_vit_b32/
  N=10000, feature_dim=512, dtype=float16
```

当前 diagnostics：

```text
stream = controlled_cifar100_split_10x10
seed = 0
backbone = clip_vit_b32
pairs = 45
ok = 45
failed = 0
mean linear_probe_auc_symmetric ≈ 0.9606
mean knn_acc ≈ 0.8931
```

本阶段目标是实现第一版 **Feature-level Merge-vs-Route Toy Baselines**。

这一步仍然不是训练深度模型，不是 adapter，不是 LoRA，不是 parameter-space model merging，也不是 MoE routing。它只是在 **frozen CLIP features** 上用 prototype / lightweight sklearn classifier 模拟：

```text
static merge
vs
oracle route
vs
learned route
vs
clustered merge-route under budget K
```

目的是验证最基础的问题：

```text
在 frozen feature space 中，任务可分性 S 很高时，route 是否明显优于 merge-all？
```

---

## 1. 强制环境约束

所有命令使用：

```bash
conda run -n moe python ...
conda run -n moe pytest ...
```

不要在 `base` 环境直接运行测试或脚本。

如需交互调试：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate moe
source .env
```

---

## 2. 硬性禁止事项

本 PR 严禁：

```text
- 不下载新数据集；
- 不抽取新的全量 feature bank；
- 不训练深度模型；
- 不训练 adapter / LoRA；
- 不实现 parameter-space model merging；
- 不实现 MoE router；
- 不接入 GDA-MoE-Adapter 项目；
- 不生成正式论文图；
- 不提交 data/、features/、artifacts/、models/、checkpoints/ 等大文件。
```

本 PR 允许：

```text
- 在 frozen features 上拟合 prototype classifier；
- 在 frozen features 上拟合 sklearn logistic/linear classifier；
- 在 frozen features 上拟合 lightweight task/domain router；
- 基于 task clusters 训练 cluster-level classifier；
- 输出 CSV / JSON summary 到 artifacts/feature_toy_v0/；
- 添加 tests，使用 synthetic features，不依赖真实 CIFAR100。
```

---

## 3. 核心实验设定

第一版只做一个最小真实实验：

```yaml
stream: controlled_cifar100_split_10x10
seed: 0
backbone: clip_vit_b32
dataset: cifar100
train feature bank: cifar100/train/clip_vit_b32
test feature bank: cifar100/test/clip_vit_b32
```

task stream：

```text
10 tasks
每个 task 10 个 CIFAR100 classes
train/val 来自 CIFAR100 train split
test 来自 CIFAR100 test split
```

当前 diagnostics 已显示 task-level input separability 很高，因此本阶段要估计：

```text
route-vs-merge gap
cluster budget frontier
learned route 与 oracle route 的差距
```

---

## 4. 方法定义

### 4.1 Merge-all classifier

静态合并基线。

训练数据：

```text
所有 task 的 train features 合并
```

预测空间：

```text
所有已见 classes 的 union，即 CIFAR100 的 100 个 classes
```

必须实现：

```text
merge_all_prototype
```

建议可选实现：

```text
merge_all_linear
```

解释：

```text
这是 K=1 的静态 merge 极端，不需要 task id，不需要 router。
```

---

### 4.2 Task-specific classifier + Oracle route

动态路由上界。

训练：

```text
每个 task 独立训练一个 classifier
每个 classifier 只覆盖该 task 的 10 个 classes
```

测试：

```text
使用真实 task_id 选择对应 task classifier
```

必须实现：

```text
task_oracle_prototype
```

建议可选实现：

```text
task_oracle_linear
```

解释：

```text
这是 K=T 的 oracle route 上界，衡量“如果 task routing 完全正确，保留任务专属 classifier 能比 merge-all 强多少”。
```

---

### 4.3 Task-specific classifier + Learned route

实际 routing 模拟。

训练：

```text
1. 每个 task 一个 task-specific classifier；
2. 一个 task router，用 train 或 val features 预测 task_id。
```

测试：

```text
先预测 task_id，再调用对应 task classifier。
```

第一版 router 必须实现：

```text
centroid_task_router
```

建议可选实现：

```text
linear_task_router
```

解释：

```text
如果 learned route 接近 oracle route，说明 feature-side S 足够支持 routing。
```

---

### 4.4 Cluster-merge classifier under budget K

预算化 merge-route 中间态。

给定：

```text
K ∈ {1, 2, 4, 5, 10}
```

把 10 个 tasks 分成 K 个 clusters。

每个 cluster 训练一个 classifier，覆盖该 cluster 内所有 task classes。

需要实现两种 cluster evaluation：

```text
cluster_oracle_prototype:
  使用真实 task_id 对应的 cluster 选择 classifier。

cluster_learned_prototype:
  先预测 cluster_id，再调用 cluster classifier。
```

可选：

```text
cluster_oracle_linear
cluster_learned_linear
```

解释：

```text
K=1 等价 merge-all；
K=10 等价 task-specific route；
1<K<10 是 budgeted hybrid。
```

---

## 5. Classifier 设计

### 5.1 Prototype classifier（必需）

实现：

```python
class PrototypeClassifier:
    def fit(features: np.ndarray, labels: np.ndarray) -> "PrototypeClassifier": ...
    def predict(features: np.ndarray) -> np.ndarray: ...
    def decision_scores(features: np.ndarray) -> np.ndarray: ...
```

要求：

```text
- 默认 L2 normalize features；
- 每个 class prototype = normalized class mean；
- prediction = nearest class prototype by cosine similarity；
- 输出 global CIFAR100 labels，不要输出 local class index；
- 如果某个 class 无样本，记录 warning，不要静默失败。
```

### 5.2 Linear classifier（建议实现，但可配置关闭）

实现：

```python
class SklearnLinearClassifier:
    def fit(features, labels): ...
    def predict(features): ...
```

建议使用：

```python
sklearn.linear_model.LogisticRegression
```

默认参数：

```yaml
linear:
  enabled: false
  max_iter: 1000
  class_weight: balanced
  solver: lbfgs
  max_train_per_task: null
```

注意：

```text
- linear classifier 可能较慢，第一版 config 默认可以关闭；
- 如果实现 linear，必须支持 max_train_per_task 采样；
- linear 只训练 sklearn classifier，不训练深度模型；
- 不保存 sklearn 权重，只保存结果。
```

---

## 6. Router 设计

### 6.1 Oracle router

不需要训练。使用真实 task_id / cluster_id。

### 6.2 Centroid task router（必需）

实现：

```python
class CentroidRouter:
    def fit(features, task_ids): ...
    def predict(features): ...
```

要求：

```text
- 每个 task/cluster 计算一个 centroid；
- 测试时选择 cosine similarity 最大的 centroid；
- 输出 task_id 或 cluster_id；
- 记录 router_acc。
```

### 6.3 Linear task router（可选）

使用 sklearn LogisticRegression 预测 task_id 或 cluster_id。

默认可配置：

```yaml
router:
  linear_enabled: false
  centroid_enabled: true
```

---

## 7. Cluster 设计

第一版至少实现两种 clustering 策略。

### 7.1 Sequential clusters（必需）

按 task_id 顺序平均分组：

```text
K=1: [0,1,2,3,4,5,6,7,8,9]
K=2: [0,1,2,3,4], [5,6,7,8,9]
K=4: 尽量均匀划分
K=5: 每组 2 个 task
K=10: 每个 task 一个 cluster
```

用途：

```text
作为确定性 baseline。
```

### 7.2 Diagnostics-aware clusters（建议实现）

基于已有 diagnostics matrix：

```text
artifacts/diagnostics/boundary_v0/
  controlled_cifar100_split_10x10_seed0_clip_vit_b32_train.csv
```

可用指标：

```text
linear_probe_auc_symmetric
centroid_cosine_distance
separation_ratio
```

建议第一版用简单规则：

```text
distance_ij = linear_probe_auc_symmetric_ij - 0.5
```

解释：

```text
AUC 越高，两个 task 越容易区分，越不应该 merge；
AUC 越接近 0.5，两个 task 越难区分，更可能适合 merge。
```

实现可以用 sklearn AgglomerativeClustering，也可以自写 greedy clustering。注意 sklearn 版本兼容：`metric="precomputed"` 和 `affinity="precomputed"` 在不同版本中可能不同，需要兼容处理。

如果 diagnostics-aware clustering 实现复杂，至少保留 sequential clusters，并把 diagnostics-aware 标记为 TODO，不要阻塞本阶段。

---

## 8. 指标定义

每个 method 至少输出：

```text
method
classifier_type
router_type
cluster_strategy
K
uses_task_id
uses_learned_router
num_classifiers
overall_acc
macro_task_acc
mean_task_acc
last_task_acc
router_acc
route_vs_merge_gap
oracle_gap
num_test_samples
status
warning
```

定义：

```text
overall_acc:
  所有 test samples 上的全局准确率。

mean_task_acc:
  10 个 task accuracy 的平均。

macro_task_acc:
  与 mean_task_acc 相同，保留字段方便后续扩展。

last_task_acc:
  task 9 的 test accuracy。

router_acc:
  learned router 的 task/cluster 预测准确率；
  oracle route 设为 1.0；
  merge-all 设为 NaN。

route_vs_merge_gap:
  method overall_acc - merge_all_prototype overall_acc。

oracle_gap:
  oracle route overall_acc - merge_all_prototype overall_acc。
```

同时输出 per-task accuracy：

```csv
method,K,task_id,task_acc,num_test_samples,router_acc_task
```

---

## 9. 输出路径

所有输出保存到：

```text
artifacts/baselines/feature_toy_v0/
```

建议文件：

```text
artifacts/baselines/feature_toy_v0/
  controlled_cifar100_split_10x10_seed0_clip_vit_b32_results.csv
  controlled_cifar100_split_10x10_seed0_clip_vit_b32_per_task.csv
  controlled_cifar100_split_10x10_seed0_clip_vit_b32_route_confusion.csv
  controlled_cifar100_split_10x10_seed0_clip_vit_b32_cluster_assignments.json
  controlled_cifar100_split_10x10_seed0_clip_vit_b32_summary.json
```

不要提交这些 artifacts。

---

## 10. 推荐代码结构

新增：

```text
mrb/
  baselines/
    __init__.py
    feature_data.py
    classifiers.py
    routers.py
    clustering.py
    evaluate.py
    report.py

configs/
  baselines/
    feature_toy_v0.yaml

scripts/
  run_feature_toy_baselines.py
  inspect_feature_toy_results.py

tests/
  test_feature_toy_classifiers.py
  test_feature_toy_routing.py
  test_feature_toy_evaluate.py
```

---

## 11. 模块职责

### 11.1 `mrb/baselines/feature_data.py`

职责：

```text
- 读取 task stream manifest；
- 读取 train/val/test feature banks；
- 根据 task indices 选择 features；
- 构造 feature-level train/val/test tables。
```

建议数据结构：

```python
@dataclass
class TaskFeatureSplit:
    task_id: int
    dataset_id: str
    classes: tuple[int, ...]
    train_features: np.ndarray
    train_labels: np.ndarray
    val_features: np.ndarray
    val_labels: np.ndarray
    test_features: np.ndarray
    test_labels: np.ndarray
```

注意：

```text
- train/val/test 的 source split 必须从 manifest 的 source_splits 读取；
- 不要假设 val feature bank 存在；
- CIFAR100 controlled stream:
  train/val -> feature bank train
  test      -> feature bank test
```

---

### 11.2 `mrb/baselines/classifiers.py`

实现：

```text
PrototypeClassifier
SklearnLinearClassifier
```

要求：

```text
- 输入 label 是 global class label；
- predict 输出 global class label；
- 支持 fit/predict；
- synthetic tests 覆盖明显可分数据。
```

---

### 11.3 `mrb/baselines/routers.py`

实现：

```text
OracleRouter
CentroidRouter
SklearnLinearRouter（可选）
```

要求：

```text
- router fit 使用 train 或 val features；
- 第一版默认用 train features；
- 记录 route predictions；
- 输出 router_acc。
```

---

### 11.4 `mrb/baselines/clustering.py`

实现：

```text
make_sequential_clusters(task_ids, K)
make_diagnostics_clusters(task_ids, diagnostics_csv, K, metric)
task_to_cluster_map(clusters)
```

要求：

```text
- K=1 和 K=T 必须正确；
- 每个 task 必须且只能属于一个 cluster；
- cluster assignment 写入 JSON。
```

---

### 11.5 `mrb/baselines/evaluate.py`

实现核心评估：

```text
evaluate_merge_all(...)
evaluate_task_oracle_route(...)
evaluate_task_learned_route(...)
evaluate_cluster_oracle(...)
evaluate_cluster_learned(...)
```

要求：

```text
- 所有方法使用同一份 train/test split；
- accuracy 比较 global labels；
- 如果 learned router 路由错误，调用错误 task/cluster classifier 后得到的 global label 仍按原始 label 比较；
- 记录 per-task acc；
- 记录 router confusion。
```

---

### 11.6 `mrb/baselines/report.py`

输出：

```text
results.csv
per_task.csv
route_confusion.csv
cluster_assignments.json
summary.json
```

summary 至少包含：

```text
best_method_by_overall_acc
merge_all_prototype_acc
task_oracle_prototype_acc
learned_route_prototype_acc
oracle_route_gap
learned_route_gap
cluster_frontier
config_hash
feature_bank_metadata
diagnostics_csv
```

---

## 12. 配置文件

新增：

```text
configs/baselines/feature_toy_v0.yaml
```

建议内容：

```yaml
experiment:
  stream: controlled_cifar100_split_10x10
  seed: 0
  backbone: clip_vit_b32
  split_manifest: artifacts/manifests/splits/controlled_cifar100_split_10x10_seed0.json
  diagnostics_csv: artifacts/diagnostics/boundary_v0/controlled_cifar100_split_10x10_seed0_clip_vit_b32_train.csv

data:
  max_train_per_task: null
  max_val_per_task: null
  max_test_per_task: null
  normalize_features: true

classifiers:
  prototype:
    enabled: true
  linear:
    enabled: false
    max_iter: 1000
    class_weight: balanced
    solver: lbfgs
    max_train_per_task: 5000

routers:
  centroid:
    enabled: true
  linear:
    enabled: false
    max_iter: 1000
    class_weight: balanced

clusters:
  ks: [1, 2, 4, 5, 10]
  strategies:
    - sequential
    - diagnostics_auc

outputs:
  root: artifacts/baselines/feature_toy_v0
  save_predictions: false
  save_confusion: true
```

第一版默认关闭 linear classifier/router，先保证 prototype/centroid 跑通。后续再开启 linear。

---

## 13. CLI 脚本

### 13.1 `scripts/run_feature_toy_baselines.py`

示例：

```bash
conda run -n moe python scripts/run_feature_toy_baselines.py \
  --config configs/baselines/feature_toy_v0.yaml \
  --stream controlled_cifar100_split_10x10 \
  --seed 0 \
  --backbone clip_vit_b32 \
  --overwrite
```

参数至少包含：

```text
--config
--stream
--seed
--backbone
--manifest
--diagnostics-csv
--output-dir
--cluster-ks
--overwrite
--quick
```

`--quick` 行为：

```text
max_train_per_task=1000
max_test_per_task=500
linear disabled
只跑 prototype + centroid router
```

正式默认行为：

```text
使用全量 train/test features
prototype required
linear optional, governed by config
```

---

### 13.2 `scripts/inspect_feature_toy_results.py`

功能：

```text
- 读取 results.csv；
- 打印 merge_all_prototype；
- 打印 task_oracle_prototype；
- 打印 task_learned_prototype；
- 打印 K frontier；
- 打印 route-vs-merge gap；
- 打印 learned route 与 oracle route 差距。
```

示例：

```bash
conda run -n moe python scripts/inspect_feature_toy_results.py \
  --input artifacts/baselines/feature_toy_v0/controlled_cifar100_split_10x10_seed0_clip_vit_b32_results.csv
```

---

## 14. 测试要求

新增测试不得依赖真实 CIFAR100 feature bank，不下载数据，不加载 CLIP。

### 14.1 `tests/test_feature_toy_classifiers.py`

测试：

```text
- PrototypeClassifier 能在 synthetic separable data 上达到高准确率；
- PrototypeClassifier predict 输出原始 global labels，而不是 0..C-1 local labels；
- LinearClassifier disabled 时不会被调用。
```

### 14.2 `tests/test_feature_toy_routing.py`

测试：

```text
- CentroidRouter 在明显可分 task features 上 route accuracy 高；
- OracleRouter 输出真实 task id；
- cluster map 每个 task 只属于一个 cluster。
```

### 14.3 `tests/test_feature_toy_evaluate.py`

测试：

```text
- 2-task synthetic data 下 merge_all 和 oracle_route 都能运行；
- route_vs_merge_gap 字段存在；
- K=1 等价 merge-all cluster；
- K=T 等价 task-specific cluster；
- learned route 错误时不会崩溃。
```

验收：

```bash
conda run -n moe pytest tests/test_feature_toy_classifiers.py \
  tests/test_feature_toy_routing.py \
  tests/test_feature_toy_evaluate.py
```

也要运行：

```bash
conda run -n moe pytest -q
conda run -n moe ruff check .
```

---

## 15. 最小验收流程

### 15.1 先 quick run

```bash
cd /root/rivermind-data/projects/merge-route-boundary

conda run -n moe python scripts/run_feature_toy_baselines.py \
  --config configs/baselines/feature_toy_v0.yaml \
  --stream controlled_cifar100_split_10x10 \
  --seed 0 \
  --backbone clip_vit_b32 \
  --quick \
  --overwrite

conda run -n moe python scripts/inspect_feature_toy_results.py \
  --input artifacts/baselines/feature_toy_v0/controlled_cifar100_split_10x10_seed0_clip_vit_b32_results.csv
```

### 15.2 再 full prototype run

```bash
conda run -n moe python scripts/run_feature_toy_baselines.py \
  --config configs/baselines/feature_toy_v0.yaml \
  --stream controlled_cifar100_split_10x10 \
  --seed 0 \
  --backbone clip_vit_b32 \
  --overwrite
```

最低通过标准：

```text
- results.csv 存在；
- per_task.csv 存在；
- summary.json 存在；
- merge_all_prototype 有结果；
- task_oracle_prototype 有结果；
- task_learned_prototype 有结果；
- cluster K={1,2,4,5,10} 有结果；
- 所有 method status=ok；
- no NaN overall_acc；
- inspect script 能打印 route-vs-merge gap。
```

---

## 16. 预期研究性判断

本阶段不要求一定得到 route 大幅超过 merge。需要如实输出。

可能结果解释：

```text
Case A:
  task_oracle_prototype >> merge_all_prototype
  learned_route_prototype 接近 task_oracle_prototype
说明 frozen feature space 中 route 有明显优势，且 S 足以支持 learned routing。

Case B:
  task_oracle_prototype ≈ merge_all_prototype
说明这些 CIFAR100 class splits 在 CLIP feature 上可共享分类边界，static merge 已经足够；这对论文也有价值，说明 high S 不是 route 必要条件，必须结合 incompatibility I。

Case C:
  learned_route_prototype << task_oracle_prototype
说明 oracle route 有潜力，但 learned router 不稳，需要改 router 或重新评估 S 指标。
```

注意：当前 diagnostics 只证明 `S` 高，不证明 `I` 高。所以即使 merge-all 很强，也不是失败，而是说明还需要后续 adapter/task-delta 层面的 incompatibility 实验。

---

## 17. 输出结果解读要求

Codex 完成后请报告：

```text
1. 当前分支和 commit；
2. 是否有代码修改，修改哪些文件；
3. quick run 是否通过；
4. full prototype run 是否通过；
5. results.csv 路径；
6. summary.json 路径；
7. merge_all_prototype overall_acc；
8. task_oracle_prototype overall_acc；
9. task_learned_prototype overall_acc；
10. learned router_acc；
11. cluster K frontier；
12. route_vs_merge_gap；
13. 是否有 NaN / failed method；
14. 下一步建议：是否开启 linear classifier/router，或者是否扩展到 real_tier0_tier2_small。
```

---

## 18. Git 规则

可以提交：

```text
mrb/baselines/*.py
scripts/run_feature_toy_baselines.py
scripts/inspect_feature_toy_results.py
configs/baselines/feature_toy_v0.yaml
tests/test_feature_toy_*.py
docs/*.md
```

不要提交：

```text
artifacts/baselines/
artifacts/diagnostics/
artifacts/manifests/
features/
data/
*.npy
*.npz
*.csv
*.jsonl
*.pt
*.pth
*.ckpt
```

如果 `.gitignore` 尚未覆盖 `artifacts/baselines/`，请补充。
