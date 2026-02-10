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
