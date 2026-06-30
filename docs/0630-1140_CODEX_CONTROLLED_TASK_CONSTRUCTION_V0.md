> 时间戳：06/30 11:25（东八区）

# Codex 阶段任务指导：Controlled Task Construction v0

## 0. 当前任务目标

项目仓库：

```text
/root/rivermind-data/projects/merge-route-boundary
https://github.com/nordlyswang/merge-route-boundary
```

当前工作分支：

```text
codex/0625-dataset-registry
```

本阶段有两个目标：

```text
A. 先把 pairwise_policy 和此前未 push 的 routing analysis 代码整理、提交并 push 到 GitHub；
B. 再实现 Controlled Task Construction v0，用不同 CIFAR100 task construction 验证 merge-route boundary 是否随任务结构系统变化。
```

当前已知机制结论：

```text
1. CIFAR100 random split 上 oracle task mask 很强：0.8838 vs merge_all 0.6603；
2. classifier_specialization_gain = 0，oracle 收益来自 label-space mask；
3. learned hard route 低于 merge_all；
4. top-k / fallback 只能小幅超过 merge_all；
5. pairwise diagnostics AUC 能预测 router confusion，但 rule-based pairwise policy 没超过 fallback；
6. 因此不应继续在同一个 random split 上手调 routing policy，而应构造不同 task regimes，验证 boundary 是否系统变化。
```

---

## 1. Step A：先整理并 push 现有未同步代码

开始前运行：

```bash
cd /root/rivermind-data/projects/merge-route-boundary
git status
git branch --show-current
git log --oneline -8
```

必须先处理已有未 push / 未提交内容。尤其包括以下阶段的代码，如果尚未同步，请一起提交：

```text
Router Calibration + Mask-size Cost Frontier v0
Router Error Anatomy v0
Pairwise-aware Routing Policy v0
```

可能涉及文件：

```text
mrb/baselines/calibration.py
mrb/baselines/router_calibration_frontier.py
mrb/baselines/router_error_anatomy.py
mrb/baselines/pairwise_policy.py

configs/baselines/router_calibration_v0.yaml
configs/baselines/router_error_anatomy_v0.yaml
configs/baselines/pairwise_policy_v0.yaml

scripts/run_router_calibration_frontier.py
scripts/inspect_router_calibration_results.py
scripts/run_router_error_anatomy.py
scripts/inspect_router_error_anatomy.py
scripts/run_pairwise_policy.py
scripts/inspect_pairwise_policy_results.py

tests/test_router_calibration_frontier.py
tests/test_router_error_anatomy.py
tests/test_pairwise_policy.py
```

如果这些代码已 commit 但未 push：

```bash
git push origin codex/0625-dataset-registry
```

如果这些代码还未 commit，请只 stage 代码、配置、脚本、测试、docs，严禁 stage artifacts：

```bash
git add mrb/baselines         configs/baselines         scripts/run_router_calibration_frontier.py         scripts/inspect_router_calibration_results.py         scripts/run_router_error_anatomy.py         scripts/inspect_router_error_anatomy.py         scripts/run_pairwise_policy.py         scripts/inspect_pairwise_policy_results.py         tests/test_router_calibration_frontier.py         tests/test_router_error_anatomy.py         tests/test_pairwise_policy.py         docs

git status
```

确认没有以下内容被 stage：

```text
artifacts/
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

然后运行：

```bash
conda run -n moe pytest -q
conda run -n moe ruff check .
```

通过后提交并 push：

```bash
git commit -m "feat: add routing analysis checkpoints"
git push origin codex/0625-dataset-registry
```

完成 Step A 后，再开始 Controlled Task Construction v0。不要把 Step A 和 Step B 混成同一个 commit。

---

## 2. Controlled Task Construction v0 的研究目标

本阶段不再继续优化 routing policy，而是构造不同 CIFAR100 task regimes，观察：

```text
1. task separability S 是否变化；
2. oracle task mask gap 是否变化；
3. learned route / top-k / fallback 是否随任务结构变化；
4. merge-all 是否在某些 construction 下更合适；
5. route 是否只在某些 high-S / high-mask-gain regime 下有效。
```

核心问题：

```text
同一个 CIFAR100 + CLIP frozen features，不同 task grouping 是否会形成不同 merge-route boundary？
```

本阶段仍然只使用 frozen features，不训练深度模型，不训练 adapter / LoRA，不做 parameter-space merging。

---

## 3. 硬性禁止事项

本 PR 严禁：

```text
- 不下载新数据；
- 不重新抽取 feature bank；
- 不训练深度模型；
- 不训练 adapter / LoRA；
- 不实现 parameter-space model merging；
- 不实现 MoE router；
- 不扩展到 real_tier0_tier2_small；
- 不提交 artifacts、features、data、模型权重或大文件；
- 不生成正式论文图。
```

本 PR 允许：

```text
- 基于已有 CIFAR100 train/test feature bank 重新构造 task manifests；
- 基于不同 task manifests 运行已有 diagnostics / feature toy / soft routing / router calibration pipeline；
- 输出 task construction summary 到 artifacts/controlled_tasks/v0/；
- 添加 synthetic tests；
- 提交代码、配置、脚本、tests、docs。
```

---

## 4. 固定资源

使用已有 feature bank：

```text
MRB_FEATURE_ROOT/cifar100/train/clip_vit_b32/
MRB_FEATURE_ROOT/cifar100/test/clip_vit_b32/
```

数据集：

```text
cifar100
```

backbone：

```text
clip_vit_b32
```

每个 controlled stream 仍采用：

```text
10 tasks × 10 classes/task = 100 CIFAR100 classes
train/val from CIFAR100 train split
test from CIFAR100 test split
```

---

## 5. 必须实现的 task construction strategies

本阶段至少实现 4 类 construction。

### 5.1 random_10x10_seed0

目的：

```text
复现当前 baseline construction，作为对照。
```

要求：

```text
应与 controlled_cifar100_split_10x10 seed=0 等价或高度一致。
```

stream id 建议：

```text
cifar100_random_10x10_seed0
```

### 5.2 random_10x10_seed1 / seed2 / seed3

目的：

```text
检查当前结论是否依赖随机 class split。
```

stream ids：

```text
cifar100_random_10x10_seed1
cifar100_random_10x10_seed2
cifar100_random_10x10_seed3
```

要求：

```text
- 每个 seed 使用不同 class order；
- 每个 task 10 classes；
- 所有 100 classes 恰好出现一次；
- train/val/test split 可复现。
```

### 5.3 feature_near_pc1_10x10

目的：

```text
构造 feature-space 中相对相近/连续的 class group，使每个 task 内部更局部。
```

建议实现：

```text
1. 读取 CIFAR100 train feature bank；
2. 计算每个 fine class 的 class prototype；
3. 对 100 个 class prototypes 做 PCA 或第一主成分投影；
4. 按 PC1 排序；
5. 每连续 10 个 classes 组成一个 task。
```

stream id：

```text
cifar100_feature_near_pc1_10x10
```

预期：

```text
可能形成 feature-space 上较局部的 task，task 间可能更可分；
oracle mask 和 learned routing 的关系可能不同于 random split。
```

### 5.4 feature_far_roundrobin_pc1_10x10

目的：

```text
构造 feature-space 上多峰/混合的 task，让每个 task 包含来自不同 feature 区域的 classes。
```

建议实现：

```text
1. 使用与 feature_near 相同的 PC1 sorted class order；
2. round-robin 分配到 10 个 tasks；
   task j = sorted_classes[j], sorted_classes[j+10], sorted_classes[j+20], ...
3. 每个 task 10 classes。
```

stream id：

```text
cifar100_feature_far_roundrobin_pc1_10x10
```

预期：

```text
每个 task 更分散、多峰；
task routing 更难；
merge-all 可能更稳；
route oracle 仍可能有 label-mask 上界。
```

---

## 6. 可选 construction：semantic coarse split

如果实现成本低，可以加 CIFAR100 coarse-label construction；如果实现复杂，不要阻塞 v0。

### 6.1 semantic_coarse_pair_10x10

CIFAR100 有 20 个 coarse superclasses，每个 superclass 包含 5 个 fine classes。可构造：

```text
每个 task = 2 个 coarse superclasses = 10 fine classes。
```

stream id：

```text
cifar100_semantic_coarse_pair_10x10
```

实现方式：

```text
优先从 CIFAR100 原始 pickle/meta 中读取 coarse_labels 和 fine/coarse mapping；
如果 torchvision loader 不方便暴露 coarse labels，可新增 utility 读取 cifar-100-python/train/test pickle；
如果仍复杂，则标记为 TODO，不要阻塞本阶段。
```

---

## 7. Manifest 要求

每个 construction 输出一个 split manifest：

```text
artifacts/controlled_tasks/v0/manifests/
  <stream_id>.json
```

manifest schema 应兼容当前 task stream / feature_data / diagnostics 代码：

```json
{
  "schema_version": 1,
  "stream_id": "...",
  "seed": 0,
  "stream_type": "controlled_class_groups",
  "dataset_id": "cifar100",
  "backbone_id": "clip_vit_b32",
  "construction_strategy": "...",
  "tasks": [
    {
      "task_id": 0,
      "dataset_id": "cifar100",
      "classes": [ ... 10 class ids ... ],
      "train_indices": [ ... ],
      "val_indices": [ ... ],
      "test_indices": [ ... ],
      "source_splits": {
        "train": "train",
        "val": "train",
        "test": "test"
      },
      "split_policy": "controlled_class_groups",
      "num_train": ...,
      "num_val": ...,
      "num_test": ...
    }
  ],
  "manifest_hash": "..."
}
```

要求：

```text
- 每个 stream 恰好 10 tasks；
- 每个 task 恰好 10 classes；
- 所有 100 classes 恰好出现一次；
- train/val 不重叠；
- test 非空；
- source_splits 正确；
- manifest 可被现有 diagnostics/baselines 脚本通过 --manifest 读取。
```

---

## 8. 推荐代码结构

新增：

```text
mrb/data/controlled_task_construction.py
configs/task_construction/controlled_cifar100_v0.yaml
scripts/build_controlled_task_streams.py
scripts/run_controlled_task_suite.py
scripts/inspect_controlled_task_suite.py
tests/test_controlled_task_construction.py
```

如果不想新增 `configs/task_construction`，也可以使用 `configs/baselines/controlled_task_construction_v0.yaml`，但建议单独目录。

---

## 9. 配置文件建议

新增：

```text
configs/task_construction/controlled_cifar100_v0.yaml
```

建议内容：

```yaml
experiment:
  dataset: cifar100
  backbone: clip_vit_b32
  seed: 0
  num_tasks: 10
  classes_per_task: 10
  val_ratio: 0.1

feature_bank:
  train_split: train
  test_split: test

constructions:
  - id: cifar100_random_10x10_seed0
    strategy: random
    class_order_seed: 0

  - id: cifar100_random_10x10_seed1
    strategy: random
    class_order_seed: 1

  - id: cifar100_random_10x10_seed2
    strategy: random
    class_order_seed: 2

  - id: cifar100_random_10x10_seed3
    strategy: random
    class_order_seed: 3

  - id: cifar100_feature_near_pc1_10x10
    strategy: feature_near_pc1

  - id: cifar100_feature_far_roundrobin_pc1_10x10
    strategy: feature_far_roundrobin_pc1

  # optional
  # - id: cifar100_semantic_coarse_pair_10x10
  #   strategy: semantic_coarse_pair
  #   enabled: false

pipeline:
  run_boundary_diagnostics: true
  run_feature_toy: true
  run_soft_routing: true
  run_router_calibration: true
  run_router_error_anatomy: false
  run_pairwise_policy: false

diagnostics:
  max_samples_per_task: 500

outputs:
  root: artifacts/controlled_tasks/v0
```

---

## 10. CLI 设计

### 10.1 `scripts/build_controlled_task_streams.py`

用途：

```text
只构造 manifests，不运行实验。
```

示例：

```bash
conda run -n moe python scripts/build_controlled_task_streams.py \
  --config configs/task_construction/controlled_cifar100_v0.yaml \
  --overwrite
```

输出：

```text
artifacts/controlled_tasks/v0/manifests/*.json
artifacts/controlled_tasks/v0/controlled_task_manifest_summary.csv
```

### 10.2 `scripts/run_controlled_task_suite.py`

用途：

```text
对每个 constructed stream 运行已有 pipeline，并汇总关键指标。
```

示例：

```bash
conda run -n moe python scripts/run_controlled_task_suite.py \
  --config configs/task_construction/controlled_cifar100_v0.yaml \
  --overwrite
```

建议行为：

```text
for each manifest:
  1. compute_boundary_diagnostics.py --manifest <manifest>
  2. run_feature_toy_baselines.py --manifest <manifest>
  3. run_soft_routing_controls.py --manifest <manifest>
  4. run_router_calibration_frontier.py --manifest <manifest>
```

如果某个现有 CLI 缺少 `--manifest` 支持，请补齐，而不是硬编码 default manifest。

### 10.3 `scripts/inspect_controlled_task_suite.py`

用途：

```text
读取 artifacts/controlled_tasks/v0/summary.csv，打印不同 construction 的边界差异。
```

示例：

```bash
conda run -n moe python scripts/inspect_controlled_task_suite.py \
  --input artifacts/controlled_tasks/v0/controlled_task_suite_summary.csv
```

---

## 11. Suite Summary 指标

最终汇总 CSV：

```text
artifacts/controlled_tasks/v0/controlled_task_suite_summary.csv
```

每行一个 construction，至少包含：

```text
stream_id
construction_strategy
class_order_seed
num_tasks
classes_per_task

mean_pairwise_auc_sym
mean_knn_acc
mean_separation_ratio

merge_all_acc
oracle_task_mask_acc
oracle_gap

hard_linear_route_acc
topk_k2_acc
fallback_acc
best_learned_control_acc
best_learned_control_method

route_vs_merge_gap_best
oracle_gap_closure_best

router_top1_acc
router_top2_recall
router_top3_recall
router_top5_recall

best_topk_k
best_topk_acc
fallback_rate
mean_mask_size

status
warning
```

需要额外输出：

```text
artifacts/controlled_tasks/v0/controlled_task_suite_summary.json
```

包含：

```text
best_construction_by_route_vs_merge_gap
best_construction_by_oracle_gap
worst_construction_by_router_top1
random_seed_variance
interpretation
```

---

## 12. 测试要求

新增：

```text
tests/test_controlled_task_construction.py
```

测试必须使用 synthetic labels/features，不依赖真实 CIFAR100 feature bank，不加载 CLIP。

至少覆盖：

```text
1. random strategy 生成 10 tasks × 10 classes；
2. 所有 classes 恰好出现一次；
3. 不同 class_order_seed 产生不同 grouping；
4. feature_near_pc1 使用 sorted order chunk；
5. feature_far_roundrobin 使用 sorted order round-robin；
6. train/val split 可复现且不重叠；
7. manifest schema 包含 source_splits；
8. manifest 可被已有 load_manifest 读取；
9. suite summary 能处理某个 stream 失败并记录 warning；
10. 不产生 data/features/artifacts 以外的可提交大文件。
```

验收命令：

```bash
conda run -n moe pytest tests/test_controlled_task_construction.py -q
conda run -n moe pytest -q
conda run -n moe ruff check .
```

---

## 13. 最小运行流程

先 build manifests：

```bash
cd /root/rivermind-data/projects/merge-route-boundary

conda run -n moe python scripts/build_controlled_task_streams.py \
  --config configs/task_construction/controlled_cifar100_v0.yaml \
  --overwrite
```

然后 quick suite：

```bash
conda run -n moe python scripts/run_controlled_task_suite.py \
  --config configs/task_construction/controlled_cifar100_v0.yaml \
  --quick \
  --overwrite
```

再 full suite：

```bash
conda run -n moe python scripts/run_controlled_task_suite.py \
  --config configs/task_construction/controlled_cifar100_v0.yaml \
  --overwrite
```

Inspect：

```bash
conda run -n moe python scripts/inspect_controlled_task_suite.py \
  --input artifacts/controlled_tasks/v0/controlled_task_suite_summary.csv
```

最低通过标准：

```text
- pytest 全部通过；
- ruff check 通过；
- build_controlled_task_streams.py 通过；
- 每个 required construction 都生成 manifest；
- 每个 manifest 通过 schema validation；
- quick suite 通过；
- full suite 至少对 required constructions 全部完成；
- summary.csv 存在；
- summary 中每个 construction 都有 merge_all/oracle/topk/fallback/router 指标；
- inspect 脚本能输出主要结论。
```

---

## 14. 结果判断标准

重点不是单个方法提升多少，而是不同 construction 之间是否呈现规律：

```text
1. random seeds:
   检查当前结论是否稳定，route_vs_merge_gap_best 的方差是否小。

2. feature_near_pc1:
   如果 mean_pairwise_auc 更高、router_top1 更高、learned route 更好，
   说明 feature-local task construction 更适合 route。

3. feature_far_roundrobin_pc1:
   如果 router_top1 下降、learned route 接近或低于 merge_all，
   说明多峰 task construction 更适合 merge 或 fallback。

4. oracle_gap:
   如果 oracle_gap 高但 learned closure 低，说明 route 上界存在但 router 无法利用。

5. learned closure:
   如果某个 construction 的 oracle_gap_closure 明显更高，
   说明该 construction 更接近 high-S routing-friendly regime。
```

可能结论：

```text
Case A:
  feature_near 明显提升 learned route
说明 task construction 可以系统改变 merge-route boundary。

Case B:
  所有 construction learned route 都只小幅超过 merge
说明 CIFAR100 frozen-feature setting 中 learned route 受限于 prototype/mask机制。

Case C:
  random seeds 方差很大
说明此前结论依赖 class split，需要多 seed 报告。

Case D:
  feature_far 明显降低 router recall
说明任务多峰性是 routing 失败的关键因素。
```

---

## 15. Git 规则

可以提交：

```text
mrb/data/controlled_task_construction.py
configs/task_construction/controlled_cifar100_v0.yaml
scripts/build_controlled_task_streams.py
scripts/run_controlled_task_suite.py
scripts/inspect_controlled_task_suite.py
tests/test_controlled_task_construction.py
docs/*.md
必要的现有 CLI manifest 支持小修复
```

不要提交：

```text
artifacts/controlled_tasks/
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

---

## 16. Codex 最终回复要求

完成后请报告：

```text
1. 是否已先提交并 push pairwise_policy / router_error / previous unpushed checkpoint；
2. 当前分支和最新 commit；
3. 修改了哪些文件；
4. pytest / ruff 是否通过；
5. build manifests 是否通过；
6. quick suite / full suite 是否通过；
7. 生成了哪些 construction；
8. 每个 construction 的核心结果：
   - mean_pairwise_auc_sym
   - merge_all_acc
   - oracle_task_mask_acc
   - hard_linear_route_acc
   - topk_k2_acc
   - fallback_acc
   - best_learned_control_acc
   - router_top1_acc
   - router_top2_recall
   - oracle_gap_closure_best
9. random seed 方差；
10. feature_near vs feature_far 的差异；
11. 是否出现更 routing-friendly 的 construction；
12. 下一步建议。
```
