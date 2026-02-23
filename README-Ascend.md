# MiniMind 昇腾 910B (Atlas 800I A2) 训练指南

本文档介绍如何在华为昇腾 910B（Atlas 800I A2）8卡环境下，使用 Docker 完成 MiniMind 的全流程训练。

## 环境要求

| 项目 | 要求 |
|------|------|
| 硬件 | Atlas 800I A2（昇腾 910B × 8） |
| 驱动 | Ascend NPU 驱动已安装（`/usr/local/Ascend/driver`） |
| 工具 | `npu-smi`（`/usr/local/sbin/npu-smi`） |
| 软件 | Docker |
| 基础镜像 | `ascend-pretrain:latest` (torch 2.1 + CANN 8.0) |

## 第1步：准备数据集

从 [ModelScope 数据集](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) 下载所需数据文件，放入 `./dataset/` 目录。

最小可用组合（快速复现聊天模型）：
- `pretrain_hq.jsonl` — 预训练数据
- `sft_mini_512.jsonl` — SFT微调数据

完整训练流程还需：
- `dpo.jsonl` — DPO偏好对齐数据
- `r1_mix_1024.jsonl` — 推理蒸馏数据
- `rlaif-mini.jsonl` — 强化学习(PPO/GRPO/SPO)数据

```
dataset/
├── pretrain_hq.jsonl
├── sft_mini_512.jsonl
├── dpo.jsonl
├── r1_mix_1024.jsonl
└── rlaif-mini.jsonl
```

## 第2步：构建 Docker 镜像

```bash
docker build -f Dockerfile.ascend -t minimind-npu .
```

镜像内容：
- 基于华为官方 PyTorch 昇腾镜像（含 CANN 8.1.RC1 + PyTorch 2.6.0）
- 自动安装 `torch_npu==2.6.0.post3` 及所有依赖
- pip 源使用清华镜像，无需外网

## 第3步：验证环境

```bash
# 查看 NPU 设备状态
docker run -it --rm --network=host \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
    minimind-npu npu-smi info
```

```bash
# 验证 torch_npu 可用
docker run -it --rm --network=host \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
    minimind-npu python -c "import torch; import torch_npu; print('NPU available:', torch.npu.is_available()); print('NPU count:', torch.npu.device_count())"
```

预期输出：
```
NPU available: True
NPU count: 8
```

## 一键全流程训练（推荐）

使用 `scripts/run_all_npu.sh` 可一键完成从数据下载到模型评测的完整流水线，自动处理 Docker 启动、阶段依赖和权重检查：

```bash
# 运行完整流程（下载数据 → 构建镜像 → pretrain → sft → dpo → reason → eval）
bash scripts/run_all_npu.sh all

# 运行最小核心流程（pretrain → sft → eval）
bash scripts/run_all_npu.sh core

# 运行单个阶段
bash scripts/run_all_npu.sh pretrain

# 运行多个指定阶段
bash scripts/run_all_npu.sh pretrain full_sft eval

# 启用断点续训
bash scripts/run_all_npu.sh --resume all

# 使用 768 维度模型
bash scripts/run_all_npu.sh --hidden-size 768 core

# 仅打印命令，不实际执行（调试用）
bash scripts/run_all_npu.sh --dry-run all

# 已在容器内时，跳过 Docker 启动
bash scripts/run_all_npu.sh --inside-docker all
```

脚本会自动记录每个阶段的耗时，失败时立即终止并报错。详细用法见 `bash scripts/run_all_npu.sh --help`。

> 如果需要更细粒度的控制（如自定义参数），可参考下面的手动分步执行方式。

## 第4步：启动训练（手动分步执行）

下面所有命令均通过 Docker 运行。通用的 Docker 启动前缀为：

```bash
DOCKER_RUN="docker run -it --rm --network=host --shm-size=500g \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
    -v $(pwd)/dataset:/workspace/minimind/dataset \
    -v $(pwd)/out:/workspace/minimind/out \
    -v $(pwd)/checkpoints:/workspace/minimind/checkpoints \
    minimind-npu"
```

> 说明：`dataset`、`out`、`checkpoints` 三个目录通过 `-v` 挂载到宿主机，训练产物持久化保存。

---

### 4.1 预训练（学知识）

```bash
$DOCKER_RUN bash scripts/run_train_npu.sh pretrain \
    --epochs 1 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --data_path ./dataset/pretrain_hq.jsonl
```

训练完成后产出 `out/pretrain_512.pth`。

### 4.2 监督微调 SFT（学对话）

```bash
$DOCKER_RUN bash scripts/run_train_npu.sh full_sft \
    --epochs 2 \
    --batch_size 16 \
    --learning_rate 1e-6 \
    --data_path ./dataset/sft_mini_512.jsonl
```

训练完成后产出 `out/full_sft_512.pth`。

> 到这里已经可以得到一个能正常对话的模型。以下步骤为可选的进阶训练。

---

### 4.3（可选）LoRA 微调

```bash
$DOCKER_RUN bash scripts/run_train_npu.sh lora \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --data_path ./dataset/lora_identity.jsonl
```

### 4.4（可选）DPO 偏好对齐

```bash
$DOCKER_RUN bash scripts/run_train_npu.sh dpo \
    --epochs 1 \
    --batch_size 4 \
    --learning_rate 4e-8 \
    --data_path ./dataset/dpo.jsonl
```

训练完成后产出 `out/dpo_512.pth`。

### 4.5（可选）推理模型训练

```bash
$DOCKER_RUN bash scripts/run_train_npu.sh reason \
    --epochs 1 \
    --batch_size 8 \
    --learning_rate 1e-6 \
    --from_weight dpo \
    --data_path ./dataset/r1_mix_1024.jsonl
```

训练完成后产出 `out/reason_512.pth`。

### 4.6（可选）强化学习

三种算法可任选其一，均需要外部 Reward 模型（如 `internlm2-1_8b-reward`）。

**GRPO:**
```bash
$DOCKER_RUN bash scripts/run_train_npu.sh grpo \
    --epochs 1 \
    --batch_size 2 \
    --learning_rate 8e-8 \
    --data_path ./dataset/rlaif-mini.jsonl \
    --reward_model_path /path/to/internlm2-1_8b-reward
```

**PPO:**
```bash
$DOCKER_RUN bash scripts/run_train_npu.sh ppo \
    --epochs 1 \
    --batch_size 2 \
    --learning_rate 8e-8 \
    --data_path ./dataset/rlaif-mini.jsonl \
    --reward_model_path /path/to/internlm2-1_8b-reward
```

**SPO:**
```bash
$DOCKER_RUN bash scripts/run_train_npu.sh spo \
    --epochs 1 \
    --batch_size 2 \
    --learning_rate 1e-7 \
    --data_path ./dataset/rlaif-mini.jsonl \
    --reward_model_path /path/to/internlm2-1_8b-reward
```

> 注意：使用强化学习时，需将 Reward 模型路径也挂载到容器内，并在 `docker run` 中增加对应的 `-v` 参数。

### 4.7（可选）知识蒸馏

```bash
$DOCKER_RUN bash scripts/run_train_npu.sh distillation \
    --epochs 6 \
    --batch_size 32 \
    --learning_rate 5e-6 \
    --data_path ./dataset/sft_mini_512.jsonl \
    --student_hidden_size 512 \
    --teacher_hidden_size 768
```

## 第5步：测试模型

训练完成后，在宿主机上测试（需安装 Python 依赖）：

```bash
python eval_llm.py --weight full_sft --device npu:0
```

也可以在容器内测试：

```bash
$DOCKER_RUN python eval_llm.py --weight full_sft --device npu:0
```

## 单卡训练

如果只想使用单卡训练（不使用分布式），直接运行训练脚本：

```bash
# 只挂载 davinci0 设备
docker run -it --rm --network=host \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
    -v $(pwd)/dataset:/workspace/minimind/dataset \
    -v $(pwd)/out:/workspace/minimind/out \
    minimind-npu python trainer/train_pretrain.py \
        --device npu:0 --epochs 1 --batch_size 4
```

## 断点续训

所有训练脚本均支持断点续训，添加 `--from_resume 1` 即可自动恢复：

```bash
$DOCKER_RUN bash scripts/run_train_npu.sh pretrain \
    --from_resume 1 \
    --epochs 1 \
    --batch_size 32
```

续训机制说明：
- 检查点自动保存在 `checkpoints/` 目录（包含模型、优化器、训练进度）
- 支持跨不同卡数恢复（自动调整 step）

## 推荐训练流程

最小流程（得到可对话模型）：

```
预训练 → SFT
```

完整流程：

```
预训练 → SFT → DPO → 推理训练 → 强化学习(GRPO/PPO/SPO)
```

## 常用参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--epochs` | 训练轮数 | 各阶段不同 |
| `--batch_size` | 批大小 | 各阶段不同 |
| `--learning_rate` | 学习率 | 各阶段不同 |
| `--hidden_size` | 模型隐藏层维度 | 512 (26M模型) |
| `--num_hidden_layers` | 隐藏层数 | 8 |
| `--use_moe` | 使用MoE架构 | 0 |
| `--dtype` | 混合精度类型 | NPU默认float16 |
| `--accumulation_steps` | 梯度累积步数 | 各阶段不同 |
| `--save_interval` | 模型保存间隔(步) | 1000 |
| `--log_interval` | 日志打印间隔(步) | 100 |
| `--from_resume` | 断点续训 | 0 |
| `--use_wandb` | 启用SwanLab记录 | 关闭 |

## 与 CUDA 版本的差异

| 项目 | CUDA | 昇腾 NPU |
|------|------|----------|
| 通信后端 | NCCL | HCCL |
| 默认精度 | bfloat16 | float16 |
| torch.compile | 支持 | 跳过（兼容性有限） |
| GradScaler | `torch.cuda.amp.GradScaler` | `torch.npu.amp.GradScaler` |
| 设备名称 | `cuda:0` | `npu:0` |

代码已做自动检测，无需手动切换。当 `torch_npu` 可导入时自动使用 NPU 路径，否则回退到 CUDA。

## 文件清单

```
Dockerfile.ascend          — Docker 构建文件
requirements_npu.txt       — NPU 环境依赖
scripts/run_all_npu.sh     — 全流程训练编排脚本（一键运行）
scripts/run_train_npu.sh   — 8卡分布式训练启动脚本
trainer/trainer_utils.py   — 核心工具模块（NPU/CUDA 双路径）
trainer/train_*.py         — 各阶段训练脚本（均已适配 NPU）
```
