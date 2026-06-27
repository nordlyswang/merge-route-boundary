> 时间戳：06/26 16:56（东八区）

# Codex 阶段任务指导：Soft / Top-k / Fallback Routing Controls v0

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

当前 GitHub HEAD 已包含：

```text
Router Sanity + Label-space Control v1
```

上一轮核心结果：

```text
merge_all_prototype overall_acc = 0.6603
merge_all_prototype_oracle_task_mask overall_acc = 0.8838
task_oracle_prototype overall_acc = 0.8838

task_learned_centroid_router_prototype acc = 0.3545, router_acc = 0.3947
task_learned_linear_router_prototype acc = 0.6489, router_acc = 0.7021
task_learned_energy_router_prototype acc = 0.6603, router_acc = 0.6956

label_mask_gain = +0.2235
classifier_specialization_gain = 0.0000
routing_error_cost:
  centroid = 0.5293
  linear   = 0.2349
  energy   = 0.2235
```

当前结论：

```text
1. oracle route 的收益完全来自 label-space mask，而不是 classifier specialization。
2. hard learned routing 过于脆弱；一旦 task 预测错，样本会被限制到错误 label space。
3. linear / energy router 明显强于 centroid，但 hard route 仍未稳定超过 merge_all。
```

本阶段目标：

```text
实现 soft / top-k / fallback routing controls，判断“软化 hard route”是否能超过 merge-all，并量化从 merge 到 route 的连续边界。
```

---

## 1. 强制环境约束

所有命令使用：

```bash
conda run -n moe python ...
conda run -n moe pytest ...
```

不要在 `base` 环境直接运行。

如需交互式调试：

```bash
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

本 PR 允许：

```text
- 在 frozen CLIP features 上拟合 sklearn / prototype-level router；
- 基于已有 prototype classifier 和 learned router 实现 top-k label mask；
- 基于 router confidence 实现 fallback-to-merge-all；
- 基于 router soft prior 实现 soft routing score；
- 输出 CSV / JSON 到 artifacts/baselines/soft_routing_v0/；
- 添加 synthetic tests。
```

---

## 3. 实验数据与协议

固定使用：

```yaml
stream: controlled_cifar100_split_10x10
seed: 0
backbone: clip_vit_b32
dataset: cifar100
```

feature banks：

```text
MRB_FEATURE_ROOT/cifar100/train/clip_vit_b32/
MRB_FEATURE_ROOT/cifar100/test/clip_vit_b32/
```

task stream：

```text
10 tasks
每个 task 10 个 CIFAR100 classes
train/val 来自 CIFAR100 train split
test 来自 CIFAR100 test split
```

重要协议：

```text
1. train split：训练 class prototypes、global prototypes、routers；
2. val split：选择 fallback threshold、soft-prior beta 等超参数；
3. test split：最终评估；
4. 严禁用 test accuracy 选择 tau / beta / top-k。
```

如果当前 feature_data 层还没有显式区分 train/val/test，请优先补齐，不要在 test 上调参。

---

## 4. 本阶段新增方法

### 4.1 Top-k task union mask

核心思想：

```text
hard top-1 route 错误会完全排除正确类别；
top-k route 允许多个候选 task 的 class union，从而降低 routing error cost。
```

方法名建议：

```text
merge_all_prototype_topk_task_mask_linear_k2
merge_all_prototype_topk_task_mask_linear_k3
merge_all_prototype_topk_task_mask_linear_k5

merge_all_prototype_topk_task_mask_energy_k2
merge_all_prototype_topk_task_mask_energy_k3
merge_all_prototype_topk_task_mask_energy_k5
```

定义：

```text
1. 使用 merge_all 的 global class prototypes；
2. router 输出每个样本的 task scores；
3. 选 top-k predicted tasks；
4. 允许预测的 label space = top-k tasks 的 class union；
5. 在该 union 内选择 prototype score 最高的 class；
6. 预测输出仍为 global CIFAR100 label。
```

需要记录：

```text
topk
router_type
topk_task_recall
mean_mask_size
overall_acc
per_task_acc
```

其中：

```text
topk_task_recall = true task_id 是否在 top-k predicted tasks 中的比例。
mean_mask_size = 平均候选 class 数，CIFAR100 每个 task 10 类，因此 k=2/3/5 对应约 20/30/50。
```

---

### 4.2 Fallback-to-merge-all control

核心思想：

```text
当 router 低置信时，不执行 hard task mask，而是回退到 merge_all_prototype。
```

方法名建议：

```text
merge_all_prototype_fallback_linear_val_tau
merge_all_prototype_fallback_energy_val_tau
```

定义：

```text
1. 学习 router；
2. 对每个样本计算 router confidence；
3. 如果 confidence >= tau：
     使用 top-1 learned task mask；
   否则：
     使用 merge_all_prototype 预测；
4. tau 只允许在 val split 上选择；
5. test split 只用于最终评估。
```

router confidence 定义：

```text
linear router:
  confidence = max predicted probability
  optional margin = top1_prob - top2_prob

energy router:
  confidence = top1_task_score - top2_task_score
  或 softmax 后的 top1 probability

centroid router:
  可选，不作为本轮优先项。
```

tau grid 建议：

```yaml
fallback:
  tau_grid:
    linear_prob: [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95]
    energy_margin_quantiles: [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
```

实现要求：

```text
- 对 linear router 优先使用 probability threshold；
- 对 energy router 如果 absolute margin 不稳定，可以用 val confidence quantile 选择阈值；
- summary 中必须记录 selected_tau、val_acc_at_selected_tau、fallback_rate_test。
```

---

### 4.3 Soft task-prior scoring

核心思想：

```text
不要把 label space 硬切成一个 task，也不要只做 top-k union；
而是把 router score 作为 task prior 加到 class prototype score 上。
```

方法名建议：

```text
merge_all_prototype_soft_prior_linear_beta
merge_all_prototype_soft_prior_energy_beta
```

定义：

对每个 class `c`，其所属 task 为 `t(c)`。对样本 `x`：

```text
class_score(c | x) =
  prototype_cosine_score(c | x)
  + beta * router_log_prior(t(c) | x)
```

其中：

```text
router_log_prior = log softmax(router_task_scores)
```

beta 控制 merge 到 route 的连续程度：

```text
beta = 0      => 完全等价 merge_all_prototype
beta 较大    => 更接近 hard routing / task mask
```

beta grid 建议：

```yaml
soft_prior:
  beta_grid: [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
```

要求：

```text
- beta 只在 val split 上选择；
- test 只报告 selected beta 对应结果；
- summary 中记录 selected_beta、val_acc_at_selected_beta。
```

---

### 4.4 Optional：Top-k + fallback hybrid

如果实现简单，可以加：

```text
merge_all_prototype_topk_fallback_linear_k2
merge_all_prototype_topk_fallback_energy_k2
```

定义：

```text
if confidence >= tau:
  use top-k union mask
else:
  use merge_all_prototype
```

该项不是必需，不要阻塞本阶段。

---

## 5. 必须保留的对照方法

本轮输出必须继续包含上一轮核心对照：

```text
merge_all_prototype
merge_all_prototype_oracle_task_mask
merge_all_prototype_learned_task_mask_linear
merge_all_prototype_learned_task_mask_energy
task_oracle_prototype
task_learned_linear_router_prototype
task_learned_energy_router_prototype
```

这样才能比较：

```text
hard route
vs
top-k route
vs
fallback route
vs
soft prior route
vs
merge-all
vs
oracle mask
```

---

## 6. 关键指标

results.csv 每行至少包含：

```text
method
control_type
router_type
topk
tau
beta
selected_on
val_acc
overall_acc
mean_task_acc
last_task_acc
router_acc_top1
router_recall_topk
fallback_rate
mean_mask_size
route_vs_merge_gap
oracle_gap_closure
status
warning
```

定义：

```text
route_vs_merge_gap =
  method overall_acc - merge_all_prototype overall_acc

oracle_gap_closure =
  (method overall_acc - merge_all_prototype overall_acc)
  / (merge_all_prototype_oracle_task_mask overall_acc - merge_all_prototype overall_acc)

fallback_rate =
  使用 merge_all fallback 的样本比例

mean_mask_size =
  每个样本平均候选 class 数
```

解释：

```text
oracle_gap_closure = 0 表示没有超过 merge-all；
oracle_gap_closure = 1 表示达到 oracle task mask；
oracle_gap_closure < 0 表示差于 merge-all。
```

per-task 输出至少包含：

```text
method
task_id
task_acc
num_test_samples
router_acc_top1_task
router_recall_topk_task
fallback_rate_task
mean_mask_size_task
```

---

## 7. 推荐代码结构

优先在已有 `mrb/baselines` 内最小增量实现。

建议新增：

```text
mrb/baselines/routing_controls.py
```

建议更新：

```text
mrb/baselines/routers.py
mrb/baselines/evaluate.py
mrb/baselines/report.py
scripts/run_feature_toy_baselines.py
scripts/inspect_feature_toy_results.py
configs/baselines/feature_toy_v0.yaml
```

如希望隔离本轮实验，也可以新增：

```text
configs/baselines/soft_routing_v0.yaml
scripts/run_soft_routing_controls.py
scripts/inspect_soft_routing_results.py
```

优先推荐新增 `soft_routing_v0.yaml`，但可以复用原 CLI。

---

## 8. Router API 要求

当前 routers 至少应支持：

```python
predict(features) -> np.ndarray
```

本轮需要新增或补充：

```python
decision_scores(features) -> np.ndarray
predict_topk(features, k: int) -> np.ndarray
confidence(features) -> np.ndarray
```

建议语义：

```text
decision_scores:
  shape [N, num_tasks]
  越大表示越可能属于该 task

predict_topk:
  shape [N, k]
  每行是 top-k task ids

confidence:
  shape [N]
  linear router 用 top1 probability；
  energy router 用 top1-top2 margin 或 top1 softmax probability。
```

对 linear router：

```text
decision_scores 可返回 predict_proba 或 decision_function；
若有 predict_proba，优先用 probability。
```

对 energy router：

```text
decision_scores 为每个 task 的 task energy score；
可由 task 内 class prototype max similarity 或 top-k mean similarity 计算。
```

---

## 9. 配置文件建议

新增：

```text
configs/baselines/soft_routing_v0.yaml
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
  normalize_features: true
  max_train_per_task: null
  max_val_per_task: null
  max_test_per_task: null

base_methods:
  include_previous_controls: true

routers:
  linear:
    enabled: true
    max_iter: 1000
    class_weight: balanced
    solver: lbfgs
    max_train_per_task: null
  energy:
    enabled: true
    score: max
    top_k: 3

topk_controls:
  enabled: true
  routers: [linear, energy]
  ks: [2, 3, 5]

fallback_controls:
  enabled: true
  routers: [linear, energy]
  select_tau_on: val
  linear_tau_grid: [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95]
  energy_tau_quantiles: [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]

soft_prior_controls:
  enabled: true
  routers: [linear, energy]
  select_beta_on: val
  beta_grid: [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]

outputs:
  root: artifacts/baselines/soft_routing_v0
  save_predictions: false
  save_confusion: true
```

---

## 10. CLI 要求

可以复用：

```text
scripts/run_feature_toy_baselines.py
```

也可以新增：

```text
scripts/run_soft_routing_controls.py
scripts/inspect_soft_routing_results.py
```

推荐新增，避免影响已有 toy baseline。

运行示例：

```bash
conda run -n moe python scripts/run_soft_routing_controls.py \
  --config configs/baselines/soft_routing_v0.yaml \
  --stream controlled_cifar100_split_10x10 \
  --seed 0 \
  --backbone clip_vit_b32 \
  --overwrite
```

inspect：

```bash
conda run -n moe python scripts/inspect_soft_routing_results.py \
  --input artifacts/baselines/soft_routing_v0/controlled_cifar100_split_10x10_seed0_clip_vit_b32_results.csv
```

---

## 11. 输出文件

保存到：

```text
artifacts/baselines/soft_routing_v0/
```

至少输出：

```text
controlled_cifar100_split_10x10_seed0_clip_vit_b32_results.csv
controlled_cifar100_split_10x10_seed0_clip_vit_b32_per_task.csv
controlled_cifar100_split_10x10_seed0_clip_vit_b32_summary.json
```

可选输出：

```text
controlled_cifar100_split_10x10_seed0_clip_vit_b32_route_confusion.csv
controlled_cifar100_split_10x10_seed0_clip_vit_b32_selected_hparams.json
```

不要提交这些 artifacts。

---

## 12. 测试要求

新增：

```text
tests/test_soft_routing_controls.py
```

或扩展：

```text
tests/test_feature_toy_routing.py
tests/test_feature_toy_evaluate.py
```

测试必须使用 synthetic features，不依赖 CIFAR100 feature bank。

至少覆盖：

```text
1. top-k task mask 的候选 label space 等于 top-k tasks 的 class union；
2. top-k recall 正确计算；
3. fallback 控制在 low confidence 样本上调用 merge_all prediction；
4. fallback_rate 正确计算；
5. soft prior beta=0 等价 merge_all prediction；
6. beta 较大时更偏向 high-prior task；
7. tau / beta 只在 val split 上选择；
8. oracle_gap_closure 字段存在且数值合理；
9. learned router top-k / fallback 出错时不会崩溃。
```

验收命令：

```bash
conda run -n moe pytest tests/test_soft_routing_controls.py -q
conda run -n moe pytest tests/test_feature_toy_*.py -q
conda run -n moe pytest -q
conda run -n moe ruff check .
```

---

## 13. 最小验收流程

先 quick：

```bash
cd /root/rivermind-data/projects/merge-route-boundary

conda run -n moe python scripts/run_soft_routing_controls.py \
  --config configs/baselines/soft_routing_v0.yaml \
  --stream controlled_cifar100_split_10x10 \
  --seed 0 \
  --backbone clip_vit_b32 \
  --quick \
  --overwrite
```

再 full：

```bash
conda run -n moe python scripts/run_soft_routing_controls.py \
  --config configs/baselines/soft_routing_v0.yaml \
  --stream controlled_cifar100_split_10x10 \
  --seed 0 \
  --backbone clip_vit_b32 \
  --overwrite
```

Inspect：

```bash
conda run -n moe python scripts/inspect_soft_routing_results.py \
  --input artifacts/baselines/soft_routing_v0/controlled_cifar100_split_10x10_seed0_clip_vit_b32_results.csv
```

最低通过标准：

```text
- pytest 全部通过；
- ruff check 通过；
- quick run 通过；
- full run 通过；
- results.csv / per_task.csv / summary.json 存在；
- no failed methods；
- no NaN overall_acc；
- top-k controls 有结果；
- fallback controls 有结果；
- soft-prior controls 有结果；
- inspect 脚本能打印 best method、route_vs_merge_gap、oracle_gap_closure。
```

---

## 14. 结果判断标准

基准：

```text
merge_all_prototype = 0.6603
oracle_task_mask = 0.8838
hard learned linear route = 0.6489
hard learned energy route = 0.6603
```

本轮希望看到：

```text
1. top-k linear/energy 至少超过 hard top-1 route；
2. fallback linear/energy 至少不低于 merge_all_prototype；
3. soft-prior linear/energy 尽量超过 merge_all_prototype；
4. oracle_gap_closure > 0 表示开始回收 oracle mask 上界；
5. 如果所有 learned controls 仍不能超过 0.6603，应报告 routing 不足，而不是扩大数据集。
```

可能解释：

```text
Case A:
  top-k/fallback/soft-prior > merge_all
说明 hard route 失败主要来自过度限制 label space，软化 routing 有效。

Case B:
  fallback ≈ merge_all，但 top-k/soft-prior 不提升
说明 router 信号还不够可用，merge_all 是更稳 boundary。

Case C:
  soft-prior 最优，且 beta 不为 0
说明 merge-route continuum 存在，后续可画 beta frontier。

Case D:
  top-k k=5 接近 oracle，但 mask size 大
说明路由收益依赖候选 label budget，需要报告 accuracy-mask-size frontier。
```

---

## 15. Codex 最终回复要求

完成后请报告：

```text
1. 当前分支和 commit；
2. 修改了哪些文件；
3. 是否新增 soft_routing_v0.yaml；
4. 是否新增 run_soft_routing_controls.py / inspect_soft_routing_results.py；
5. pytest / ruff / quick / full 是否通过；
6. best method by overall_acc；
7. merge_all_prototype acc；
8. oracle_task_mask acc；
9. hard learned linear / energy acc；
10. top-k linear/energy 最优结果；
11. fallback linear/energy 最优结果；
12. soft-prior linear/energy 最优结果；
13. selected tau / selected beta；
14. fallback_rate / mean_mask_size；
15. oracle_gap_closure；
16. 是否超过 merge_all；
17. 下一步建议。
```

---

## 16. Git 规则

可以提交：

```text
mrb/baselines/*.py
configs/baselines/soft_routing_v0.yaml
scripts/run_soft_routing_controls.py
scripts/inspect_soft_routing_results.py
tests/test_soft_routing_controls.py
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

如果 `.gitignore` 尚未覆盖 `artifacts/baselines/soft_routing_v0/`，请补充。
