# Codex 初始配置计划：Merge–Route Boundary 项目（Conda-only）

> 适用仓库：`nordlyswang/merge-route-boundary`  
> 当前阶段：初步环境配置、资源目录规范、模型/数据集拉取脚本、smoke checks  
> 明确限制：服务器不支持 Docker / devcontainer；本项目一律使用 Conda / Mamba 管理环境。

---

## 1. 项目背景

本仓库服务于一篇独立论文方向：

> **When to Merge, When to Route? A Boundary Analysis of Continual Learning and Model Merging**

核心问题不是实现某个特定 GDA router、MoE-Adapter 方法或 CLIP 分支融合技巧，而是建立一个可复现实验平台，系统研究：

```text
给定任务流、task-specific updates/adapters/experts、资源预算 K，
什么时候应该 merge，什么时候应该 route，什么时候应该 expand / continue learning？
```

本项目与之前的 GDA-MoE-Adapter 工作保持解耦。之前的工作更聚焦于在 `MoE-Adapters4CL` 上实现 GDA-calibrated routing/fusion、保持 DDAS baseline 可运行、MTIL 11-dataset 评估和 few-shot/full-shot 消融。本仓库不继承那套方法实现，也不以 GDA-MoE 的 SOTA 数字为第一目标。

本仓库第一阶段只建立三类基础能力：

```text
1. 可复现 Conda 环境；
2. 可控的数据集/模型缓存与验证；
3. 后续 merge / route / expand 实验需要的目录、配置和 smoke-check 基础。
```

---

## 2. 当前阶段非目标

Codex 第一轮不要实现以下内容：

```text
- 不实现 continual learning 训练循环；
- 不实现 model merging 算法；
- 不实现 MoE routing；
- 不实现 adapter / LoRA 训练；
- 不下载大规模数据集，例如 DomainNet、ImageNet-R、ImageNet-A、VTAB 全量数据；
- 不引入 Docker、devcontainer、compose、k8s、slurm；
- 不接入 wandb；
- 不提交模型权重、数据集、缓存、checkpoint、日志。
```

第一轮目标是让服务器上的 clone 仓库可以稳定执行：

```bash
conda activate mrb
python scripts/check_env.py
python scripts/download_datasets.py --dry-run
python scripts/download_models.py --dry-run
python scripts/verify_resources.py
python scripts/smoke_check.py --quick
pytest tests/
```

---

## 3. 建议让 Codex 完成的最小 PR

建议只做一个初始化 PR，标题：

```text
PR: initialize conda environment and resource management
```

该 PR 只新增工程底座，不写实验代码。

### 3.1 需要新增的目录结构

```text
merge-route-boundary/
  AGENTS.md
  README.md
  .gitignore
  .env.example
  pyproject.toml
  environment.yml
  environment.cpu.yml
  configs/
    default.yaml
    resources/
      models.yaml
      datasets.yaml
  docs/
    CODEX_INITIAL_SETUP.md
    SETUP.md
    RESOURCE_POLICY.md
  mrb/
    __init__.py
  scripts/
    create_conda_env.sh
    check_env.py
    report_env.py
    download_models.py
    download_datasets.py
    verify_resources.py
    smoke_check.py
  tests/
    test_imports.py
    test_resource_config.py
```

---

## 4. Conda 环境设计

### 4.1 基本原则

本项目不使用 Docker。所有脚本都应假设用户在服务器上通过 Conda/Mamba 创建环境：

```bash
conda env create -f environment.yml
conda activate mrb
```

如果服务器 CUDA / driver 与默认 CUDA 环境不兼容，则使用 CPU 环境先完成 smoke checks：

```bash
conda env create -f environment.cpu.yml
conda activate mrb-cpu
```

### 4.2 `environment.yml` 建议内容

Codex 应创建一个保守、研究友好的 CUDA 环境文件。不要锁死过细 patch version，避免服务器求解失败。

```yaml
name: mrb
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults

dependencies:
  - python=3.10
  - pip
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - matplotlib
  - pyyaml
  - tqdm
  - pytest
  - ruff
  - ipython
  - pytorch
  - torchvision
  - torchaudio
  # 若服务器驱动支持，可按实际情况保留或修改为 12.1 / 12.4 / 11.8。
  - pytorch-cuda=12.1
  - pip:
      - transformers
      - datasets
      - huggingface_hub
      - accelerate
      - timm
      - open_clip_torch
      - peft
```

### 4.3 `environment.cpu.yml` 建议内容

```yaml
name: mrb-cpu
channels:
  - pytorch
  - conda-forge
  - defaults

dependencies:
  - python=3.10
  - pip
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - matplotlib
  - pyyaml
  - tqdm
  - pytest
  - ruff
  - ipython
  - pytorch
  - torchvision
  - torchaudio
  - cpuonly
  - pip:
      - transformers
      - datasets
      - huggingface_hub
      - accelerate
      - timm
      - open_clip_torch
      - peft
```

### 4.4 `scripts/create_conda_env.sh`

该脚本只负责辅助创建环境，不应该强行覆盖已有环境。

建议行为：

```bash
bash scripts/create_conda_env.sh cuda
bash scripts/create_conda_env.sh cpu
```

逻辑：

```text
1. 检查 conda 是否存在；
2. 如果存在 mamba，则优先用 mamba；否则用 conda；
3. cuda 模式使用 environment.yml；
4. cpu 模式使用 environment.cpu.yml；
5. 若环境已存在，提示用户使用 --force 才能重建；
6. 不自动运行训练或下载大资源。
```

---

## 5. 缓存与路径规范

### 5.1 `.env.example`

Codex 应创建 `.env.example`：

```bash
# Repository root
export MRB_ROOT=/path/to/merge-route-boundary

# Large files should live outside git.
export MRB_DATA_ROOT=/data/$USER/mrb/data
export MRB_MODEL_ROOT=/data/$USER/mrb/models
export MRB_CACHE_ROOT=/data/$USER/mrb/cache
export MRB_OUTPUT_ROOT=/data/$USER/mrb/outputs

# Hugging Face / Torch cache
export HF_HOME=$MRB_CACHE_ROOT/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export TORCH_HOME=$MRB_CACHE_ROOT/torch

# Optional: set only if private HF resources are needed later.
# export HF_TOKEN=...
```

### 5.2 `.gitignore`

必须忽略：

```gitignore
# environments
.conda/
.venv/
__pycache__/
*.pyc

# local env files
.env
.env.local

# large local resources
data/
models/
cache/
outputs/
checkpoints/
logs/
wandb/

# model artifacts
*.pt
*.pth
*.ckpt
*.safetensors
*.bin

# generated reports
*.jsonl
*.csv.tmp
```

原则：**Git 只存代码、配置 manifest、文档和小型测试文件；不存模型权重、数据集、checkpoint 和缓存。**

---

## 6. 初始资源清单

### 6.1 `configs/resources/datasets.yaml`

第一阶段只放小数据集，用于下载与 dataloader smoke check。

```yaml
datasets:
  - id: CIFAR10
    source: torchvision
    required: true
    split: train

  - id: CIFAR100
    source: torchvision
    required: true
    split: train

  - id: MNIST
    source: torchvision
    required: false
    split: train

  - id: FashionMNIST
    source: torchvision
    required: false
    split: train
```

### 6.2 `configs/resources/models.yaml`

第一阶段不强制下载大模型权重。默认只下载 tokenizer/config/processor 等轻量文件；如果显式传 `--include-weights`，再下载完整模型快照。

```yaml
models:
  - id: openai/clip-vit-base-patch32
    source: huggingface
    required: true
    default_mode: metadata_only
    allow_patterns:
      - "config.json"
      - "preprocessor_config.json"
      - "tokenizer.json"
      - "tokenizer_config.json"
      - "vocab.json"
      - "merges.txt"
      - "special_tokens_map.json"

  - id: google/vit-base-patch16-224
    source: huggingface
    required: false
    default_mode: metadata_only
    allow_patterns:
      - "config.json"
      - "preprocessor_config.json"
```

后续真正开始实验时，再新增完整模型权重、LoRA/adapter checkpoint 和大数据集配置。

---

## 7. 脚本功能要求

### 7.1 `scripts/check_env.py`

检查：

```text
- Python version；
- conda env name；
- torch / torchvision / transformers / datasets / open_clip_torch 是否可 import；
- torch.cuda.is_available()；
- GPU 名称、CUDA version、device count；
- MRB_DATA_ROOT / MRB_MODEL_ROOT / MRB_CACHE_ROOT / HF_HOME 是否存在；
- 当前 git commit hash。
```

如果 CUDA 不可用，不要直接失败；输出 warning，并提示可使用 CPU smoke check。

### 7.2 `scripts/report_env.py`

输出 JSON 格式环境报告：

```bash
python scripts/report_env.py --out outputs/env_report.json
```

字段建议：

```json
{
  "python": "3.10.x",
  "torch": "...",
  "torchvision": "...",
  "cuda_available": true,
  "cuda_version": "...",
  "gpu_names": ["..."],
  "mrb_data_root": "...",
  "mrb_model_root": "...",
  "hf_home": "...",
  "git_commit": "..."
}
```

### 7.3 `scripts/download_datasets.py`

要求：

```text
- 从 configs/resources/datasets.yaml 读取 manifest；
- 默认只处理 required=true；
- 支持 --all 下载 optional 资源；
- 支持 --dry-run，只打印计划，不下载；
- torchvision 数据集下载到 MRB_DATA_ROOT；
- 不创建实验输出。
```

示例：

```bash
python scripts/download_datasets.py --dry-run
python scripts/download_datasets.py
python scripts/download_datasets.py --all
```

### 7.4 `scripts/download_models.py`

要求：

```text
- 从 configs/resources/models.yaml 读取 manifest；
- 默认 metadata_only，只按 allow_patterns 下载轻量文件；
- 支持 --include-weights 下载完整 snapshot；
- 支持 --dry-run；
- 下载到 HF_HUB_CACHE / MRB_MODEL_ROOT，不写入 git；
- 如 HF_TOKEN 缺失但资源为公开模型，不报错。
```

示例：

```bash
python scripts/download_models.py --dry-run
python scripts/download_models.py
python scripts/download_models.py --include-weights
```

### 7.5 `scripts/verify_resources.py`

检查：

```text
- required datasets 是否存在；
- required model metadata 是否存在；
- cache 目录是否可写；
- 磁盘空间是否低于安全阈值；
- 输出 readable summary。
```

### 7.6 `scripts/smoke_check.py`

`--quick` 模式只做最小检查：

```text
1. import 核心包；
2. 加载 CIFAR10 前 1 个 batch；
3. 加载 CLIP config / processor / tokenizer；
4. 随机 tensor 通过一个小型 torchvision model 或简单 nn.Linear；
5. 如果 CUDA 可用，把一个随机 tensor 放到 GPU 上做一次 matmul；
6. 不训练、不保存 checkpoint、不跑 benchmark。
```

示例：

```bash
python scripts/smoke_check.py --quick
```

---

## 8. `AGENTS.md` 建议内容

Codex 应创建仓库根目录 `AGENTS.md`，内容建议如下：

```markdown
# AGENTS.md

## Project

This repository studies the boundary between continual learning, model merging, and MoE-style routing. The first phase is infrastructure only: Conda environment, resource manifests, download scripts, and smoke checks.

## Hard constraints

- Do not use Docker, devcontainer, docker-compose, or container deployment files.
- Use Conda/Mamba environment files only.
- Do not implement training, model merging, routing, adapter logic, or experiments in the initial setup PR.
- Do not commit datasets, model weights, checkpoints, cache files, logs, or secrets.
- All large resources must live outside git and be controlled by environment variables.

## Required checks before opening a PR

Run:

```bash
python scripts/check_env.py
python scripts/download_datasets.py --dry-run
python scripts/download_models.py --dry-run
python scripts/verify_resources.py
python scripts/smoke_check.py --quick
pytest tests/
```

If CUDA is unavailable, CPU smoke checks may pass with a warning.

## Coding style

- Use type hints for new Python functions.
- Prefer small, testable scripts.
- Use clear error messages when required environment variables are missing.
- Keep resource manifests declarative.
```

---

## 9. Codex 直接任务描述

可以直接把下面这段发给 Codex：

```text
请在仓库 nordlyswang/merge-route-boundary 中完成第一阶段初始化。注意：我的服务器不支持 Docker / devcontainer，因此本项目必须使用 Conda/Mamba 配置环境。当前阶段只做环境配置、资源目录规范、模型/数据集 manifest、下载脚本、verify 脚本和 smoke checks，不允许实现具体 continual learning、model merging、MoE routing、adapter/LoRA 训练或实验逻辑。

请新增以下文件：

1. 项目与 Codex 指导：
   - AGENTS.md
   - docs/CODEX_INITIAL_SETUP.md
   - docs/SETUP.md
   - docs/RESOURCE_POLICY.md
   - README.md 如已有则最小补充，不要大改无关内容

2. Conda 环境：
   - environment.yml
   - environment.cpu.yml
   - scripts/create_conda_env.sh

3. 配置与资源 manifest：
   - .env.example
   - .gitignore
   - configs/default.yaml
   - configs/resources/models.yaml
   - configs/resources/datasets.yaml

4. 基础 Python 包与脚本：
   - pyproject.toml
   - mrb/__init__.py
   - scripts/check_env.py
   - scripts/report_env.py
   - scripts/download_models.py
   - scripts/download_datasets.py
   - scripts/verify_resources.py
   - scripts/smoke_check.py

5. 测试：
   - tests/test_imports.py
   - tests/test_resource_config.py

设计要求：
- 不要添加 Dockerfile、docker-compose.yml、.devcontainer 或容器相关配置。
- 所有大文件路径必须通过环境变量控制：MRB_DATA_ROOT、MRB_MODEL_ROOT、MRB_CACHE_ROOT、MRB_OUTPUT_ROOT、HF_HOME、HF_HUB_CACHE、HF_DATASETS_CACHE、TORCH_HOME。
- 下载脚本必须 manifest-driven，从 configs/resources/*.yaml 读取资源列表。
- 下载脚本默认只下载 required=true 的小资源。
- 模型下载脚本默认只下载 metadata/tokenizer/config 等轻量文件；只有显式 --include-weights 时才下载完整权重。
- 必须支持 --dry-run。
- smoke_check.py 只能做 import、路径检查、GPU 检查、小数据 batch 加载、模型 config/tokenizer 加载、随机 tensor forward；不能训练、不能保存 checkpoint、不能跑 benchmark。
- .gitignore 必须忽略 data/、models/、cache/、outputs/、checkpoints/、logs/、wandb/、*.pt、*.pth、*.ckpt、*.safetensors、*.bin。
- pytest tests/ 必须通过。

验收命令：

python scripts/check_env.py
python scripts/download_datasets.py --dry-run
python scripts/download_models.py --dry-run
python scripts/verify_resources.py
python scripts/smoke_check.py --quick
pytest tests/
```

---

## 10. 第一阶段完成定义

完成后，仓库应满足：

```text
- 服务器可以用 Conda/Mamba 创建环境；
- 不依赖 Docker；
- 缓存路径、数据路径、模型路径全部可配置；
- 小数据集和模型 metadata 可以按 manifest 拉取；
- smoke_check 能验证 import、CUDA、cache、CIFAR batch、CLIP config/tokenizer；
- pytest 通过；
- git status 中没有大文件、数据集、模型权重、checkpoint、cache。
```

只有这一阶段完成后，才进入第二阶段：`mrb/metrics.py`、`mrb/merging.py`、`mrb/frontier.py` 和 controlled benchmark。
