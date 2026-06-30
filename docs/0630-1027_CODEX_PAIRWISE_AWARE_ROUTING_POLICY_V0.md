> 时间戳：06/30 10:26（东八区）

# Codex 阶段任务指导：Pairwise-aware Routing Policy v0

## 0. 当前阶段定位

项目仓库：

```text
/root/rivermind-data/projects/merge-route-boundary
https://github.com/nordlyswang/merge-route-boundary
```

当前分支：

```text
codex/0625-dataset-registry
```

当前项目已经完成以下阶段：

```text
1. Dataset Registry + Task Stream Layer
2. Frozen Feature Bank Layer
3. Boundary Diagnostics v0
4. Feature-level Merge-vs-Route Toy Baselines
5. Router Sanity + Label-space Control v1
6. Soft / Top-k / Fallback Routing Controls v0
7. Router Calibration + Mask-size Cost Frontier v0
8. Router Error Anatomy v0
```

本阶段目标是实现：

```text
Pairwise-aware Routing Policy v0
```

核心问题：

```text
能否利用 router confidence、top1-top2 task pair 的 pairwise diagnostics，以及 task-pair confusion prior，
在 top-1 hard mask、top-2 mask、fallback merge-all 之间做更有解释性的选择？
```

当前已知关键结果：

```text
merge_all_prototype = 0.6603
oracle_task_mask = 0.8838

hard learned linear route = 0.6489
best top-k learned route = 0.6716
best fallback learned route = 0.6732

router recall:
  top1 = 0.7021
  top2 = 0.8444
  top3 = 0.9096
  top5 = 0.9606

confidence:
  mean_confidence_correct = 0.8001
  mean_confidence_wrong   = 0.5286
  confidence AUROC for correct route = 0.8273

diagnostics AUC vs confusion:
  Pearson  = -0.6936
  Spearman = -0.6533
```

当前判断：

```text
1. top-1 hard route 太脆弱；
2. top-2 可以修复部分 top-1 错误；
3. fallback 可以避免低置信错误 route；
4. pairwise diagnostics AUC 能预测真实 task-pair confusion；
5. 因此下一步应使用 pairwise diagnostics 构造更精细的 routing policy。
```

---

## 1. 强制环境约束

所有命令必须通过 `moe` Conda 环境运行：

```bash
conda run -n moe python ...
conda run -n moe pytest ...
```

不要在 `base` 环境直接运行。

如需交互式调试：

```bash
cd /root/rivermind-data/projects/merge-route-boundary
source ~/miniconda3/etc/profile.d/conda.sh
conda activate moe
source .env
```

---

## 2. 硬性禁止事项

本 PR 严禁：

```text
- 不下载新数据；
- 不重新抽取 feature bank；
- 不训练深度模型；
- 不训练 adapter / LoRA；
- 不实现 parameter-space model merging；
- 不实现 MoE router；
- 不扩展到 real_tier0_tier2_small；
- 不生成正式论文图；
- 不提交 artifacts、features、data、模型权重或大文件。
```

本阶段只允许：

```text
- 在 frozen CLIP features 上复用已有 linear router / prototype classifier；
- 基于已有 diagnostics matrix 和 router error anatomy 输出构造 rule-based pairwise-aware routing policy；
- 输出 CSV / JSON 到 artifacts/baselines/pairwise_policy_v0/；
- 添加 synthetic tests；
- 保持所有结果可复现。
```

---

## 3. 固定实验设定

使用同一个 controlled setting：

```yaml
stream: controlled_cifar100_split_10x10
seed: 0
backbone: clip_vit_b32
dataset: cifar100
```

依赖的本地资源：

```text
MRB_FEATURE_ROOT/cifar100/train/clip_vit_b32/
MRB_FEATURE_ROOT/cifar100/test/clip_vit_b32/
```

依赖的 artifact：

```text
artifacts/diagnostics/boundary_v0/
  controlled_cifar100_split_10x10_seed0_clip_vit_b32_train.csv

artifacts/baselines/feature_toy_v0/
artifacts/baselines/soft_routing_v0/
artifacts/baselines/router_calibration_v0/
artifacts/baselines/router_error_anatomy_v0/
```

如果 artifacts 缺失，脚本可以提示用户先运行对应阶段，也可以自动调用轻量脚本重建；但不要抽取新的 feature bank。

---

## 4. 实验核心思想

已有策略：

```text
hard top-1:
  直接使用 router top1 task mask。
  问题：top1 错误时代价极高。

top-k:
  使用 top-k task label union。
  问题：k 变大时 mask-size cost 增大。

fallback:
  低置信时退回 merge-all。
  问题：无法利用 top2 中可能包含正确 task 的信息。
```

Pairwise-aware policy 希望做：

```text
if top1 confidence high and top1/top2 pair is reliable:
    use top1 task mask
elif top2 appears necessary and pairwise risk is high:
    use top2 task union mask
else:
    fallback to merge-all
```

这里的 pairwise risk 来自：

```text
1. pairwise diagnostics AUC：AUC 越低，两个 task 越容易混淆；
2. observed confusion prior：历史/val 上 top1->top2 或 task-pair confusion 越高，越应该避免 hard top1；
3. top1-top2 margin：margin 越低，越应该 top2 或 fallback；
4. confidence：置信度越低，越应该 fallback。
```

---

## 5. 本阶段新增方法

### 5.1 Pairwise-aware Top1/Top2/Fallback Policy

方法名建议：

```text
pairwise_policy_linear_auc_margin
pairwise_policy_linear_confusion_margin
pairwise_policy_linear_auc_confidence
pairwise_policy_linear_combined
```

最小必做一个：

```text
pairwise_policy_linear_combined
```

对每个测试样本，linear router 输出：

```text
top1_task
top2_task
top1_confidence
top2_confidence
margin = top1_confidence - top2_confidence
```

从 diagnostics matrix 查：

```text
pair_auc = diagnostics_linear_probe_auc_symmetric(top1_task, top2_task)
```

从 router error anatomy 或 val router confusion 估计：

```text
pair_confusion_prior = confusion_rate(top1_task <-> top2_task)
```

然后根据规则选择 action：

```text
action ∈ {top1_mask, top2_mask, fallback_merge_all}
```

建议规则初版：

```text
if confidence < tau_fallback:
    action = fallback_merge_all

elif margin < tau_margin and pair_auc < tau_pair_auc:
    action = top2_mask

elif pair_confusion_prior > tau_pair_confusion and margin < tau_margin_high:
    action = top2_mask

else:
    action = top1_mask
```

可调整为更简单：

```text
if confidence < tau_fallback:
    fallback
elif margin < tau_margin or pair_auc < tau_pair_auc:
    top2
else:
    top1
```

关键要求：

```text
所有 tau / threshold 必须只在 val split 上选择；
test split 只用于最终报告。
```

---

### 5.2 Val-selected Pairwise Policy Grid

需要在 val split 上扫描阈值组合：

```yaml
pairwise_policy:
  tau_fallback_grid: [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8]
  tau_margin_grid: [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
  tau_pair_auc_grid: [0.90, 0.93, 0.95, 0.97, 0.99]
  tau_pair_confusion_grid: [0.02, 0.04, 0.06, 0.08]
```

可以先实现一个精简 grid，避免组合爆炸：

```text
tau_fallback_grid × tau_margin_grid × tau_pair_auc_grid
```

如果运行太慢：

```text
1. 先固定 tau_fallback=0.6；
2. 扫 tau_margin 和 tau_pair_auc；
3. 再细扫 tau_fallback。
```

选择标准：

```text
selected thresholds = val overall_acc 最大；
如果 val overall_acc 相同，优先 mean_mask_size 更小；
如果仍相同，优先 fallback_rate 更小。
```

---

### 5.3 Action-specific Result Logging

每个 policy 必须记录：

```text
action_top1_rate
action_top2_rate
action_fallback_rate
mean_mask_size
top1_route_acc
top2_recall
overall_acc
val_acc
route_vs_merge_gap
oracle_gap_closure
```

解释：

```text
action_top1_rate：使用 top1 hard mask 的样本比例；
action_top2_rate：使用 top2 union mask 的样本比例；
action_fallback_rate：回退 merge-all 的样本比例；
mean_mask_size：平均候选 class 数；
oracle_gap_closure：回收 oracle mask gap 的比例。
```

---

## 6. 需要保留的对照方法

本阶段结果表必须包含以下对照：

```text
merge_all_prototype
oracle_task_mask
hard_top1_linear
topk_linear_k2
topk_linear_k3
fallback_linear_tau
previous_best_learned_control
pairwise_policy_linear_combined
```

建议同时保留：

```text
topk_linear_k4
topk_linear_k5
topk_linear_k7
topk_linear_k10
```

但不作为主结论。

---

## 7. 关键指标

`results.csv` 每行至少包含：

```text
method
policy_type
router_type
selected_on
val_acc
overall_acc
mean_task_acc
last_task_acc
route_vs_merge_gap
oracle_gap_closure

tau_fallback
tau_margin
tau_pair_auc
tau_pair_confusion

action_top1_rate
action_top2_rate
action_fallback_rate
mean_mask_size

router_top1_acc
router_top2_recall
confidence_auc_for_correct_route

status
warning
```

`per_task.csv` 每行至少包含：

```text
method
task_id
task_acc
num_test_samples
action_top1_rate_task
action_top2_rate_task
action_fallback_rate_task
mean_mask_size_task
router_top1_acc_task
router_top2_recall_task
gain_vs_merge_task
gain_vs_hard_top1_task
```

---

## 8. Pairwise Diagnostics Join

需要构造一个 pairwise table，用于解释 policy 选择：

```text
top1_task
top2_task
num_samples
pair_auc
pair_separation_ratio
pair_confusion_prior
mean_margin
mean_confidence
action_top1_rate
action_top2_rate
action_fallback_rate
accuracy_when_pair_appears
```

输出文件：

```text
controlled_cifar100_split_10x10_seed0_clip_vit_b32_pairwise_policy_pairs.csv
```

用途：

```text
验证 pairwise diagnostics 是否真的影响 action selection，以及哪些 pair 被策略改成 top2/fallback。
```

---

## 9. 推荐代码结构

新增：

```text
mrb/baselines/pairwise_policy.py
configs/baselines/pairwise_policy_v0.yaml
scripts/run_pairwise_policy.py
scripts/inspect_pairwise_policy_results.py
tests/test_pairwise_policy.py
```

也可在已有 `routing_controls.py` 基础上扩展，但建议新增独立模块，避免与 soft routing 混杂。

---

## 10. 配置文件

新增：

```text
configs/baselines/pairwise_policy_v0.yaml
```

建议内容：

```yaml
experiment:
  stream: controlled_cifar100_split_10x10
  seed: 0
  backbone: clip_vit_b32
  split_manifest: artifacts/manifests/splits/controlled_cifar100_split_10x10_seed0.json

inputs:
  diagnostics_csv: artifacts/diagnostics/boundary_v0/controlled_cifar100_split_10x10_seed0_clip_vit_b32_train.csv
  router_error_anatomy_dir: artifacts/baselines/router_error_anatomy_v0
  soft_routing_results: artifacts/baselines/soft_routing_v0/controlled_cifar100_split_10x10_seed0_clip_vit_b32_results.csv
  router_calibration_results: artifacts/baselines/router_calibration_v0/controlled_cifar100_split_10x10_seed0_clip_vit_b32_results.csv

router:
  type: linear
  use_temperature: true
  selected_temperature: auto
  max_iter: 1000
  class_weight: balanced
  solver: lbfgs

policy:
  enabled: true
  select_on: val
  candidate_actions: [top1_mask, top2_mask, fallback_merge_all]
  tau_fallback_grid: [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8]
  tau_margin_grid: [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
  tau_pair_auc_grid: [0.90, 0.93, 0.95, 0.97, 0.99]
  tie_break:
    - higher_val_acc
    - lower_mean_mask_size
    - lower_fallback_rate

outputs:
  root: artifacts/baselines/pairwise_policy_v0
  save_predictions: false
  save_pairwise_table: true
```

---

## 11. CLI

新增：

```text
scripts/run_pairwise_policy.py
scripts/inspect_pairwise_policy_results.py
```

运行示例：

```bash
conda run -n moe python scripts/run_pairwise_policy.py \
  --config configs/baselines/pairwise_policy_v0.yaml \
  --stream controlled_cifar100_split_10x10 \
  --seed 0 \
  --backbone clip_vit_b32 \
  --overwrite
```

Inspect：

```bash
conda run -n moe python scripts/inspect_pairwise_policy_results.py \
  --input artifacts/baselines/pairwise_policy_v0/controlled_cifar100_split_10x10_seed0_clip_vit_b32_results.csv
```

Inspect 输出至少包括：

```text
best pairwise policy
selected thresholds
overall_acc
val_acc
route_vs_merge_gap
oracle_gap_closure
action_top1/top2/fallback rates
mean_mask_size
best/worst task gains
top pairwise policy pairs
comparison against fallback and top-k k=2
```

---

## 12. 输出文件

保存到：

```text
artifacts/baselines/pairwise_policy_v0/
```

至少输出：

```text
controlled_cifar100_split_10x10_seed0_clip_vit_b32_results.csv
controlled_cifar100_split_10x10_seed0_clip_vit_b32_per_task.csv
controlled_cifar100_split_10x10_seed0_clip_vit_b32_pairwise_policy_pairs.csv
controlled_cifar100_split_10x10_seed0_clip_vit_b32_summary.json
```

可选输出：

```text
controlled_cifar100_split_10x10_seed0_clip_vit_b32_action_counts.csv
controlled_cifar100_split_10x10_seed0_clip_vit_b32_threshold_grid.csv
```

不要提交这些 artifacts。

---

## 13. 测试要求

新增：

```text
tests/test_pairwise_policy.py
```

测试使用 synthetic features 和 synthetic diagnostics，不依赖真实 CIFAR100 feature bank，不加载 CLIP。

至少覆盖：

```text
1. pairwise diagnostics lookup 对 (i,j) 和 (j,i) 对称；
2. policy 在 low confidence 时选择 fallback；
3. policy 在 low margin + low pair_auc 时选择 top2；
4. policy 在 high confidence + high margin + high pair_auc 时选择 top1；
5. threshold selection 只使用 val metrics；
6. tie-break 在 val_acc 相同情况下优先 lower mean_mask_size；
7. action rates 和 mean_mask_size 计算正确；
8. oracle_gap_closure 计算正确；
9. pairwise policy 输出包含 required columns；
10. 缺失 diagnostics pair 时不崩溃，使用 safe default 并记录 warning。
```

验收命令：

```bash
conda run -n moe pytest tests/test_pairwise_policy.py -q
conda run -n moe pytest tests/test_router_error_anatomy.py -q
conda run -n moe pytest tests/test_router_calibration_frontier.py -q
conda run -n moe pytest tests/test_soft_routing_controls.py -q
conda run -n moe pytest tests/test_feature_toy_*.py -q
conda run -n moe pytest -q
conda run -n moe ruff check .
```

---

## 14. 运行要求

先 quick：

```bash
conda run -n moe python scripts/run_pairwise_policy.py \
  --config configs/baselines/pairwise_policy_v0.yaml \
  --stream controlled_cifar100_split_10x10 \
  --seed 0 \
  --backbone clip_vit_b32 \
  --quick \
  --overwrite
```

再 full：

```bash
conda run -n moe python scripts/run_pairwise_policy.py \
  --config configs/baselines/pairwise_policy_v0.yaml \
  --stream controlled_cifar100_split_10x10 \
  --seed 0 \
  --backbone clip_vit_b32 \
  --overwrite
```

Inspect：

```bash
conda run -n moe python scripts/inspect_pairwise_policy_results.py \
  --input artifacts/baselines/pairwise_policy_v0/controlled_cifar100_split_10x10_seed0_clip_vit_b32_results.csv
```

最低通过标准：

```text
- pytest 全部通过；
- ruff check 通过；
- quick run 通过；
- full run 通过；
- no failed methods；
- no NaN overall_acc；
- pairwise_policy_linear_combined 有结果；
- per_task.csv 存在；
- pairwise_policy_pairs.csv 存在；
- summary.json 存在；
- inspect 脚本能打印 selected thresholds 和 action rates。
```

---

## 15. 结果判断标准

比较基准：

```text
merge_all_prototype = 0.6603
topk_linear_k2 = 0.6716
fallback_linear = 0.6732
oracle_task_mask = 0.8838
```

希望观察：

```text
1. pairwise_policy 是否超过 fallback_linear 0.6732；
2. 如果不能超过，是否能用更低 mean_mask_size 达到接近 fallback 的准确率；
3. action_top2_rate 是否集中在高混淆 task pair；
4. diagnostics pair_auc 是否真正影响 top2/fallback decision；
5. per-task gains 是否集中在 task 5 / task 8 / task 2 等已知困难任务。
```

可能解释：

```text
Case A:
  pairwise_policy > fallback
说明 pairwise diagnostics 能改善 routing decision，是本项目的重要正结果。

Case B:
  pairwise_policy ≈ fallback but lower mean_mask_size
说明 pairwise diagnostics 能降低候选 label budget，是 accuracy-cost frontier 的正结果。

Case C:
  pairwise_policy < fallback
说明当前规则或 pairwise signal 不够，需要转向更强 router 或 controlled task construction。

Case D:
  pairwise_policy 主要改善 task 5/8/2
说明 routing policy 可针对结构化混淆 task 生效。
```

---

## 16. Git 规则

可以提交：

```text
mrb/baselines/pairwise_policy.py
configs/baselines/pairwise_policy_v0.yaml
scripts/run_pairwise_policy.py
scripts/inspect_pairwise_policy_results.py
tests/test_pairwise_policy.py
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

---

## 17. Codex 最终回复要求

完成后请报告：

```text
1. 当前分支和 commit；
2. 修改了哪些文件；
3. pytest / ruff / quick / full 是否通过；
4. selected thresholds；
5. pairwise_policy overall_acc；
6. route_vs_merge_gap；
7. oracle_gap_closure；
8. action_top1_rate / action_top2_rate / action_fallback_rate；
9. mean_mask_size；
10. best/worst per-task gains；
11. top pairwise policy pairs；
12. 是否超过 fallback 0.6732；
13. 是否降低 mean_mask_size；
14. 下一步建议。
```
