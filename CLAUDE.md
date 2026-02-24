# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MiniMind is a lightweight LLM (Large Language Model) training framework built with pure PyTorch. It implements the complete pipeline from pre-training to alignment, trainable on a single RTX 3090. Models range from 26M to 145M parameters. All core algorithms are implemented from scratch without relying on third-party training abstractions.

## Commands

### Single-GPU Training
```bash
python trainer/train_pretrain.py --epochs 1 --batch_size 32
python trainer/train_full_sft.py --epochs 2 --batch_size 16
python trainer/train_lora.py --epochs 50 --batch_size 32
python trainer/train_dpo.py --epochs 1 --batch_size 4
python trainer/train_ppo.py --epochs 1 --batch_size 2
python trainer/train_grpo.py --epochs 1 --batch_size 2
python trainer/train_spo.py --epochs 1 --batch_size 2
python trainer/train_distillation.py --epochs 6 --batch_size 32
python trainer/train_reason.py --epochs 1 --batch_size 8
```

### Multi-GPU Training (DDP)
```bash
torchrun --nproc_per_node 4 trainer/train_pretrain.py --epochs 1 --batch_size 32
```

### Ascend NPU 8-Card Training
```bash
bash scripts/run_train_npu.sh pretrain --epochs 1 --batch_size 32
```

### Inference
```bash
python eval_llm.py --weight full_sft --device cuda:0
```

### OpenAI-Compatible API Server
```bash
python scripts/serve_openai_api.py
```

### Tool Calling (工具调用)

**数据准备 & 训练 & 评估（在 Docker 容器内执行）：**
```bash
# 数据准备（从HuggingFace下载并转换，需联网）
python scripts/data_prepare_toolcall.py --source hermes --max_samples 10000

# 工具调用SFT训练（基于full_sft权重继续训练）
python trainer/train_full_sft.py \
    --data_path ../dataset/sft_tool_call.jsonl \
    --from_weight full_sft --save_weight tool_sft --max_seq_len 512

# 自动评估（需挂载 scripts/ 和 model/ 目录到容器内）
python scripts/eval_tool_call.py --weight tool_sft --mode auto
```

**MoE 模型转换 + vLLM 部署（宿主机执行）：**
```bash
# 转换 MoE 权重为 HuggingFace 格式（输出到 out/minimind-moe-hf/）
bash scripts/run_all_npu.sh --use-moe --weight tool_sft --hidden-size 768 --num-hidden-layers 16 convert

# 启动 vLLM Docker 服务（端口 8000）
bash scripts/run_all_npu.sh --use-moe --weight tool_sft --hidden-size 768 --num-hidden-layers 16 vllm

# 一键转换+启动
bash scripts/run_all_npu.sh --use-moe --weight tool_sft --hidden-size 768 --num-hidden-layers 16 serve

# Dense 模型同理（去掉 --use-moe）
bash scripts/run_all_npu.sh --weight tool_sft --hidden-size 768 --num-hidden-layers 16 serve
```

**Web 测试页面（独立 HTML / React App）：**
```bash
# 独立 HTML 版：scripts/web_tool_test.html（浏览器直接打开）
# React 版（推荐）：docs/tutorial/react-app/ 中的 "10.工具测试" Tab
cd docs/tutorial/react-app && npm run dev
# Vite 反向代理已配置：/v1/* → vLLM:8000，/api/* → 训练管理:8999
```

### 训练管理后端

`scripts/serve_train_manager.py` — 运行在 minimind-npu 容器内的 FastAPI 服务（端口 8999），
提供训练控制 REST API + 实时日志 SSE 流。对应 React App 中的 "11.训练管理" Tab。

```bash
# 启动训练管理后端
docker run -d --rm --name minimind-manager --network=host \
  --device /dev/davinci0 ... --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  -v /data/code/minimind:/workspace/minimind \
  minimind-npu python /workspace/minimind/scripts/serve_train_manager.py
```

API 接口：`GET /api/status`、`GET /api/weights`、`GET /api/datasets`、
`POST /api/train/start`、`POST /api/train/stop`、`GET /api/train/logs`（SSE）、`GET /api/npu`

### Docker 环境说明

本项目的训练和推理依赖 Docker 容器（宿主机无 Python 环境）：

- **训练/评估容器**: `minimind-npu`（基于 `Dockerfile.ascend` 构建）
- **vLLM 推理容器**: `quay.io/ascend/vllm-ascend:v0.13.0`

宿主机可用命令仅限 `bash scripts/run_all_npu.sh`，所有 Python 操作需在容器内执行。

常用 Docker 操作：
```bash
# 查看运行中的 vLLM 容器
docker ps | grep vllm-minimind

# 查看 vLLM 日志
docker logs vllm-minimind-moe

# 停止 vLLM 服务
docker stop vllm-minimind-moe

# 进入训练容器执行命令（挂载 scripts/model/out/dataset）
docker run -it --rm --network=host --shm-size=500g \
    --device /dev/davinci0 --device /dev/davinci_manager \
    --device /dev/devmm_svm --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
    -v /data/code/minimind/dataset:/workspace/minimind/dataset \
    -v /data/code/minimind/out:/workspace/minimind/out \
    -v /data/code/minimind/scripts:/workspace/minimind/scripts \
    -v /data/code/minimind/model:/workspace/minimind/model \
    minimind-npu bash
```

### No test suite or linter configured.

## Architecture

### Training Pipeline Flow
```
Pretrain → SFT → [LoRA | DPO | PPO | GRPO | SPO | Distillation | Reason] → Inference
```
Each stage produces a `.pth` checkpoint in `out/` named `{stage}_{hidden_size}.pth` (e.g., `full_sft_512.pth`). Each subsequent stage loads from the previous via `--from_weight`.

### Key Module Relationships

**`trainer/trainer_utils.py`** is the foundation module imported by all 9 training scripts. It provides:
- `init_distributed_mode()` — DDP setup (NCCL for CUDA, HCCL for NPU)
- `init_model()` — loads `MiniMindConfig` + weights + tokenizer, returns `(model, tokenizer)`
- `lm_checkpoint()` — bidirectional: saves when `model` is provided, loads when `model=None`
- `SkipBatchSampler` — enables resume with automatic step-skip and cross-GPU-count scaling
- `is_npu_available()` — detects Ascend NPU (torch_npu) availability

**`model/model_minimind.py`** defines the model architecture:
- `MiniMindConfig` (extends `PretrainedConfig`) — all hyperparameters
- `MiniMindForCausalLM` (extends `PreTrainedModel` + `GenerationMixin`) — the LLM
- Supports MoE via `use_moe=True` with routed + shared experts and auxiliary load-balancing loss
- RoPE with YaRN scaling for long-context extrapolation
- Flash attention toggle via `flash_attn` config parameter

**`dataset/lm_dataset.py`** provides 4 dataset classes:
- `PretrainDataset` — raw text JSONL for pre-training
- `SFTDataset` — conversation JSONL with chat template formatting
- `DPODataset` — chosen/rejected preference pairs
- `RLAIFDataset` — prompts for on-policy RL (PPO/GRPO/SPO)

### Training Script Structure (all 9 scripts follow this pattern)
1. Parse args (device, dtype, batch_size, learning_rate, etc.)
2. `init_distributed_mode()` → DDP setup
3. Configure mixed precision (`autocast` context + `GradScaler`)
4. `init_model()` → load model and tokenizer
5. Optional `lm_checkpoint(model=None)` → resume from checkpoint
6. Training loop with gradient accumulation, logging, periodic saves
7. `dist.destroy_process_group()` cleanup

### NPU/CUDA Dual Support Pattern
All training scripts use a consistent pattern for NPU compatibility:
- Device default: `npu:0` if NPU available, else `cuda:0`, else `cpu`
- DDP device: `npu:{local_rank}` or `cuda:{local_rank}`
- Autocast: `torch.amp.autocast(device_type='npu')` or `torch.cuda.amp.autocast()`
- GradScaler: `torch.amp.GradScaler('npu')` or `torch.cuda.amp.GradScaler()`
- Default dtype on NPU is `float16` (instead of `bfloat16`)
- `torch.compile` is skipped on NPU

### Checkpoint Format
- Model weights: `out/{weight}_{hidden_size}.pth` — half-precision state dict
- Resume data: `checkpoints/{weight}_{hidden_size}_resume.pth` — includes model, optimizer, scaler, epoch, step, world_size, wandb_id
- MoE models append `_moe` suffix

### Model Sizes
| Config | Params | hidden_size | num_hidden_layers |
|--------|--------|-------------|-------------------|
| Small  | 26M    | 512         | 8                 |
| Base   | 104M   | 768         | 16                |
| MoE    | 145M   | 768 + MoE   | 16                |

## Language

This is a Chinese-first project. Code comments, log messages, and documentation are primarily in Chinese. Maintain this convention when modifying existing files.
