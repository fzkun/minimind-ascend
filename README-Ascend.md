# MiniMind 昇腾 910B (Atlas 800I A2) 训练指南

本文档介绍如何在华为昇腾 910B（Atlas 800I A2）8卡环境下，使用 Docker 完成 MiniMind 的全流程训练、模型转换和 vLLM 部署。

## 环境要求

| 项目 | 要求 |
|------|------|
| 硬件 | Atlas 800I A2（昇腾 910B × 8） |
| 驱动 | Ascend NPU 驱动已安装（`/usr/local/Ascend/driver`） |
| 工具 | `npu-smi`（`/usr/local/sbin/npu-smi`） |
| 软件 | Docker |
| 基础镜像 | `ascend-pretrain:latest` (torch 2.1 + CANN 8.0) |

## 模型规格

| 类型 | 参数量 | hidden_size | num_hidden_layers | 说明 |
|------|--------|-------------|-------------------|------|
| Dense-Small | 26M | 512 | 8 | 默认配置 |
| Dense-Base | 104M | 768 | 16 | 大模型 |
| MoE | 145M | 768 | 16 | 4路由专家 + 1共享专家, top-2 |

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

## 一键全流程（推荐）

使用 `scripts/run_all_npu.sh` 可一键完成从数据下载到 vLLM 部署的完整流水线：

### Dense 模型

```bash
# 完整流程（下载数据 → 构建镜像 → pretrain → sft → dpo → reason → eval）
bash scripts/run_all_npu.sh all

# 最小核心流程（pretrain → sft → eval）
bash scripts/run_all_npu.sh core

# 训练完成后转换并部署 vLLM
bash scripts/run_all_npu.sh serve

# 使用 768 维度模型
bash scripts/run_all_npu.sh --hidden-size 768 --num-hidden-layers 16 all
```

### MoE 模型

```bash
# MoE 完整训练（pretrain → sft → dpo → reason → eval）
bash scripts/run_all_npu.sh --use-moe --hidden-size 768 --num-hidden-layers 16 all

# MoE 转换 + vLLM 部署
bash scripts/run_all_npu.sh --use-moe --hidden-size 768 --num-hidden-layers 16 serve

# 指定权重名称（默认 full_sft）
bash scripts/run_all_npu.sh --use-moe --hidden-size 768 --num-hidden-layers 16 --weight reason serve
```

### 通用选项

```bash
# 运行单个/多个阶段
bash scripts/run_all_npu.sh pretrain full_sft eval

# 启用断点续训
bash scripts/run_all_npu.sh --resume all

# 仅打印命令，不实际执行（调试用）
bash scripts/run_all_npu.sh --dry-run all

# 已在容器内时，跳过 Docker 启动
bash scripts/run_all_npu.sh --inside-docker all
```

### 阶段名与预设组合

| 预设 | 展开 |
|------|------|
| `all` | download build pretrain full_sft dpo reason eval |
| `core` | download build pretrain full_sft eval |
| `serve` | convert vllm |
| `rl` | ppo grpo spo |

可用阶段：`download` `build` `tokenizer` `pretrain` `full_sft` `lora` `dpo` `reason` `ppo` `grpo` `spo` `distillation` `eval` `convert` `vllm`

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

> MoE 训练只需在训练参数中追加 `--use_moe 1 --hidden_size 768 --num_hidden_layers 16`，产出权重自动带 `_moe` 后缀。

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
# Dense 模型
$DOCKER_RUN python eval_llm.py --weight full_sft --device npu:0

# MoE 模型
$DOCKER_RUN python eval_llm.py --weight full_sft --device npu:0 \
    --hidden_size 768 --num_hidden_layers 16 --use_moe 1
```

## 第6步：模型转换（供 vLLM 部署）

`scripts/convert_to_hf.py` 将 `.pth` 权重转换为 HuggingFace 格式：

- **Dense 模型** → `LlamaForCausalLM` 格式
- **MoE 模型** → `Qwen2MoeForCausalLM` 格式（完整保留共享专家）

### 一键转换

```bash
# Dense 模型转换
bash scripts/run_all_npu.sh convert

# MoE 模型转换
bash scripts/run_all_npu.sh --use-moe --hidden-size 768 --num-hidden-layers 16 convert

# 指定权重（默认 full_sft）
bash scripts/run_all_npu.sh --use-moe --hidden-size 768 --num-hidden-layers 16 --weight reason convert
```

### 手动转换

```bash
# Dense 模型 → out/minimind-hf/
$DOCKER_RUN python scripts/convert_to_hf.py \
    --save_dir out \
    --weight full_sft \
    --hidden_size 512 \
    --num_hidden_layers 8 \
    --output_dir out/minimind-hf

# MoE 模型 → out/minimind-moe-hf/
$DOCKER_RUN python scripts/convert_to_hf.py \
    --save_dir out \
    --weight reason \
    --hidden_size 768 \
    --num_hidden_layers 16 \
    --use_moe 1 \
    --output_dir out/minimind-moe-hf
```

### 转换产物

```
out/minimind-hf/           — Dense 模型 (LlamaForCausalLM)
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json

out/minimind-moe-hf/       — MoE 模型 (Qwen2MoeForCausalLM)
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

### MoE 转换说明

MiniMind MoE 包含共享专家（`shared_experts`），而 vLLM 不支持 MiniMind 原生架构。转换脚本将 MoE 映射为 Qwen2MoE 格式，通过以下方式保留共享专家：

1. `shared_experts.0` → `shared_expert`（重命名）
2. 添加全零 `shared_expert_gate`（`sigmoid(0) = 0.5`）
3. 共享专家的 `down_proj` 权重 ×2 补偿（`0.5 × 2 = 1`，数学等价）
4. 补充零 attention bias（Qwen2MoE 需要，MiniMind 没有）

## 第7步：vLLM 部署

### 一键部署

```bash
# Dense 模型部署
bash scripts/run_all_npu.sh serve

# MoE 模型部署
bash scripts/run_all_npu.sh --use-moe --hidden-size 768 --num-hidden-layers 16 serve

# 自定义端口和镜像
bash scripts/run_all_npu.sh --vllm-port 8998 --vllm-image quay.io/ascend/vllm-ascend:v0.13.0 serve
```

`serve` 预设组合会自动执行 `convert` + `vllm` 两个阶段。

### 手动部署 vLLM

```bash
# Dense 模型
docker run -d --rm \
    --name vllm-minimind \
    --shm-size=1g --network=host \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/:ro \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v $(pwd)/out/minimind-hf:/models/minimind:ro \
    quay.io/ascend/vllm-ascend:v0.13.0 \
    vllm serve /models/minimind \
        --served-model-name minimind \
        --host 0.0.0.0 --port 8000 \
        --dtype float16 --max-model-len 2048

# MoE 模型
docker run -d --rm \
    --name vllm-minimind-moe \
    --shm-size=1g --network=host \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/:ro \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v $(pwd)/out/minimind-moe-hf:/models/minimind:ro \
    quay.io/ascend/vllm-ascend:v0.13.0 \
    vllm serve /models/minimind \
        --served-model-name minimind-moe \
        --host 0.0.0.0 --port 8000 \
        --dtype float16 --max-model-len 2048
```

### 测试 API

```bash
# 测试 Dense 模型
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"minimind","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'

# 测试 MoE 模型
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"minimind-moe","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'

# 停止服务
docker stop vllm-minimind       # Dense
docker stop vllm-minimind-moe   # MoE
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

## 完整复现脚本

### Dense 模型（26M）从训练到部署

```bash
# 1. 构建镜像 + 下载数据
bash scripts/run_all_npu.sh download build

# 2. 训练全流程
bash scripts/run_all_npu.sh pretrain full_sft dpo reason

# 3. 评测
bash scripts/run_all_npu.sh eval

# 4. 转换 + vLLM 部署
bash scripts/run_all_npu.sh serve
```

或一条命令：
```bash
bash scripts/run_all_npu.sh all && bash scripts/run_all_npu.sh serve
```

### Dense 模型（104M）从训练到部署

```bash
bash scripts/run_all_npu.sh --hidden-size 768 --num-hidden-layers 16 all
bash scripts/run_all_npu.sh --hidden-size 768 --num-hidden-layers 16 serve
```

### MoE 模型（145M）从训练到部署

```bash
bash scripts/run_all_npu.sh --use-moe --hidden-size 768 --num-hidden-layers 16 all
bash scripts/run_all_npu.sh --use-moe --hidden-size 768 --num-hidden-layers 16 --weight reason serve
```

### 训练产物目录

```
out/
├── pretrain_512.pth            — Dense 预训练权重
├── full_sft_512.pth            — Dense SFT 权重
├── dpo_512.pth                 — Dense DPO 权重
├── reason_512.pth              — Dense 推理训练权重
├── pretrain_768_moe.pth        — MoE 预训练权重
├── full_sft_768_moe.pth        — MoE SFT 权重
├── dpo_768_moe.pth             — MoE DPO 权重
├── reason_768_moe.pth          — MoE 推理训练权重
├── minimind-hf/                — Dense HF 格式 (LlamaForCausalLM)
└── minimind-moe-hf/            — MoE HF 格式 (Qwen2MoeForCausalLM)
```

## 常用参数说明

### run_all_npu.sh 选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--hidden-size N` | 模型隐藏层维度 | 512 |
| `--num-hidden-layers N` | 隐藏层数量 | 8 |
| `--use-moe` | 启用 MoE 架构 | 关闭 |
| `--resume` | 启用断点续训 | 关闭 |
| `--weight NAME` | 权重名称前缀 (convert/vllm 用) | full_sft |
| `--reward-model PATH` | Reward 模型路径 (RL 阶段) | - |
| `--teacher-hidden-size N` | 教师模型维度 (蒸馏用) | 768 |
| `--vllm-image IMAGE` | vLLM Docker 镜像 | quay.io/ascend/vllm-ascend:v0.13.0 |
| `--vllm-port PORT` | vLLM 服务端口 | 8000 |
| `--max-model-len N` | vLLM 最大序列长度 | 2048 |
| `--force-build` | 强制重建 Docker 镜像 | 关闭 |
| `--inside-docker` | 跳过 Docker 启动 | 关闭 |
| `--dry-run` | 仅打印命令 | 关闭 |

### 训练脚本参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--epochs` | 训练轮数 | 各阶段不同 |
| `--batch_size` | 批大小 | 各阶段不同 |
| `--learning_rate` | 学习率 | 各阶段不同 |
| `--hidden_size` | 模型隐藏层维度 | 512 |
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
Dockerfile.ascend              — Docker 构建文件
requirements_npu.txt           — NPU 环境依赖
scripts/run_all_npu.sh         — 全流程编排脚本（训练 + 转换 + 部署）
scripts/run_train_npu.sh       — 8卡分布式训练启动脚本
scripts/convert_to_hf.py       — 模型转换（Dense→Llama / MoE→Qwen2MoE）
scripts/convert_model.py       — 模型转换（MiniMind原生HF格式，供transformers使用）
scripts/serve_openai_api.py    — OpenAI 兼容 API 服务（不依赖 vLLM）
trainer/trainer_utils.py       — 核心工具模块（NPU/CUDA 双路径）
trainer/train_*.py             — 各阶段训练脚本（均已适配 NPU）
eval_llm.py                    — 交互式推理评测
```
