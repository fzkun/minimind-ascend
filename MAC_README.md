# Mac 本地复现指南

在 Mac（Apple Silicon M1/M2/M3/M4）上从零跑通 MiniMind 的完整流程。

## 1. 环境搭建

```bash
# 克隆项目
git clone http://10.8.9.81:3000/llm/minimind.git
cd minimind
git checkout trainer/mac

# 用 uv 创建虚拟环境（Python 3.11）
uv venv --python 3.11 .venv
source .venv/bin/activate

# 安装依赖（阿里云加速）
uv pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## 2. PyCharm 配置

1. 用 PyCharm 打开项目
2. `Settings → Project → Python Interpreter → Add Interpreter → Existing`
3. 选择 `.venv/bin/python`
4. 右上角运行配置会自动出现以下选项（按训练流程排序）：

| 编号 | 配置名 | 用途 |
|------|--------|------|
| 01 | **`01_debug_pretrain`** | 预训练调试 |
| 02 | **`02_debug_sft`** | SFT 微调调试 |
| 03 | **`03_debug_lora`** | LoRA 微调调试 |
| 04 | **`04_debug_reason`** | 推理蒸馏调试 |
| 05 | **`05_debug_dpo`** | DPO 偏好优化调试 |
| 06 | **`06_debug_distill`** | 知识蒸馏调试 |
| 07 | **`07_debug_ppo`** | PPO 强化学习调试 * |
| 08 | **`08_debug_grpo`** | GRPO 强化学习调试 * |
| 09 | **`09_debug_spo`** | SPO 强化学习调试 * |
| 10 | **`10_test_pretrain`** | 测试预训练模型（续写） |
| 11 | **`11_test_sft`** | 测试 SFT 模型（对话） |
| 12 | **`12_test_lora`** | 测试 LoRA 模型（对话） |
| 13 | **`13_test_reason`** | 测试推理模型（思维链） |

> \* PPO/GRPO/SPO 默认使用 `--reward_mode mock`（随机分数），Mac CPU 可直接运行。

## 3. 统一训练脚本

所有训练任务合并为一个入口 `trainer/train.py`，通过 `--task` 切换：

```bash
cd trainer

# 预训练
python train.py --task pretrain --device cpu ...

# SFT 监督微调
python train.py --task sft --device cpu ...

# LoRA 微调
python train.py --task lora --device cpu ...

# 推理蒸馏
python train.py --task reason --device cpu ...
```

原有的薄入口脚本（`train_pretrain.py`、`train_full_sft.py`、`train_lora.py`、`train_reason.py`）仍然可用，向后兼容。

### 一键烟雾测试

```bash
# 跑通全部 9 种训练任务（CPU 上约 2-3 分钟）
# 包含: pretrain, sft, lora, reason, dpo, distillation, ppo, grpo, spo
bash trainer/test_train.sh
```

## 4. 预训练调试

### 命令行方式

```bash
cd trainer
python train.py --task pretrain \
    --data_path ../dataset/pretrain_hq_debug.jsonl \
    --batch_size 2 \
    --accumulation_steps 1 \
    --num_workers 0 \
    --epochs 1 \
    --log_interval 1 \
    --save_interval 50 \
    --hidden_size 512 \
    --num_hidden_layers 2 \
    --max_seq_len 128 \
    --device cpu
```

### PyCharm 方式

右上角选 **`01_debug_pretrain`**，直接点 Debug 按钮即可打断点。

### 参数说明

| 参数 | 调试值 | 正式训练值 | 说明 |
|------|--------|-----------|------|
| `--data_path` | `pretrain_hq_debug.jsonl` | `pretrain_hq.jsonl` | 调试用 100 条，正式 140 万条 |
| `--batch_size` | 2 | 32 | 调试用极小 batch |
| `--accumulation_steps` | 1 | 8 | 调试不累积梯度 |
| `--num_workers` | 0 | 8 | **0 才能在 PyCharm 打断点** |
| `--num_hidden_layers` | 2 | 8 | 调试用 2 层，跑得快 |
| `--max_seq_len` | 128 | 340 | 调试缩短序列 |
| `--device` | cpu | cuda:0 | Mac 用 cpu |
| `--log_interval` | 1 | 100 | 调试每步都打印 |
| `--save_interval` | 50 | 1000 | 调试减少保存频率 |

## 5. 测试预训练模型

预训练模型只会**续写**（接着你的开头往下写），不会回答问题。

```bash
# 在项目根目录运行（自动从权重文件推断层数）
python scripts/test_pretrain.py --device cpu

# 自定义续写开头
python scripts/test_pretrain.py --device cpu --prompt "从前有座山"
```

PyCharm 方式：右上角选 **`10_test_pretrain`**。

## 6. SFT 微调调试

SFT（Supervised Fine-Tuning）让模型从"续写"变成"回答问题"。

### 命令行方式

```bash
cd trainer
python train.py --task sft \
    --data_path ../dataset/sft_mini_512_debug.jsonl \
    --from_weight none \
    --batch_size 2 \
    --accumulation_steps 1 \
    --num_workers 0 \
    --epochs 1 \
    --log_interval 1 \
    --save_interval 50 \
    --hidden_size 512 \
    --num_hidden_layers 2 \
    --max_seq_len 128 \
    --device cpu
```

### PyCharm 方式

右上角选 **`02_debug_sft`**，直接点 Debug 按钮即可打断点。

> `--from_weight none` 表示不加载预训练权重，直接从随机初始化开始 SFT。调试时这样最方便，正式训练应该用 `--from_weight pretrain` 加载预训练好的权重。

## 7. 测试 SFT 模型

SFT 模型能**理解指令并回答问题**（区别于预训练模型只会续写）。

```bash
# 自动模式，跑内置问题
python scripts/test_sft.py --device cpu

# 手动对话模式
python scripts/test_sft.py --device cpu --mode chat
```

PyCharm 方式：右上角选 **`11_test_sft`**。

## 8. LoRA 微调调试

LoRA 在冻结基座模型的基础上，只训练轻量适配器参数（约 0.5M），适合小数据快速微调。

### 命令行方式

```bash
cd trainer
python train.py --task lora \
    --data_path ../dataset/lora_identity_debug.jsonl \
    --from_weight none \
    --batch_size 2 \
    --accumulation_steps 1 \
    --num_workers 0 \
    --epochs 1 \
    --log_interval 1 \
    --save_interval 50 \
    --hidden_size 512 \
    --num_hidden_layers 2 \
    --max_seq_len 128 \
    --device cpu
```

### PyCharm 方式

右上角选 **`03_debug_lora`**，直接点 Debug 按钮即可打断点。

### LoRA 专用参数

| 参数 | 说明 |
|------|------|
| `--lora_name lora_medical` | 指定 LoRA 权重名称（覆盖 `--save_weight`） |
| `--from_weight full_sft` | 正式训练时基于 SFT 权重微调 |

## 9. 测试 LoRA 模型

```bash
# 自动模式
python scripts/test_lora.py --device cpu

# 指定 LoRA 权重名称
python scripts/test_lora.py --device cpu --lora_name lora_medical

# 手动对话
python scripts/test_lora.py --device cpu --mode chat
```

PyCharm 方式：右上角选 **`12_test_lora`**。

## 10. 推理蒸馏调试

推理蒸馏训练模型先"思考"再"回答"，输出包含 `<think>...</think>` 和 `<answer>...</answer>` 标签。

### 命令行方式

```bash
cd trainer
python train.py --task reason \
    --data_path ../dataset/r1_mix_1024_debug.jsonl \
    --from_weight none \
    --batch_size 2 \
    --accumulation_steps 1 \
    --num_workers 0 \
    --epochs 1 \
    --log_interval 1 \
    --save_interval 50 \
    --hidden_size 512 \
    --num_hidden_layers 2 \
    --max_seq_len 128 \
    --device cpu
```

### PyCharm 方式

右上角选 **`04_debug_reason`**，直接点 Debug 按钮即可打断点。

## 11. 测试推理模型

```bash
# 自动模式（观察 <think>/<answer> 标签输出）
python scripts/test_reason.py --device cpu

# 手动对话
python scripts/test_reason.py --device cpu --mode chat
```

PyCharm 方式：右上角选 **`13_test_reason`**。

## 12. DPO 偏好优化调试

DPO（Direct Preference Optimization）通过 chosen/rejected 对比数据让模型学会偏好。需要策略模型和冻结的参考模型。

### 命令行方式

```bash
cd trainer
python train_dpo.py \
    --data_path ../dataset/dpo_debug.jsonl \
    --from_weight none \
    --batch_size 2 \
    --accumulation_steps 1 \
    --num_workers 0 \
    --epochs 1 \
    --log_interval 1 \
    --save_interval 50 \
    --hidden_size 512 \
    --num_hidden_layers 2 \
    --max_seq_len 128 \
    --device cpu
```

### PyCharm 方式

右上角选 **`05_debug_dpo`**，直接点 Debug 按钮即可打断点。

## 13. 知识蒸馏调试

知识蒸馏（Knowledge Distillation）用大模型（教师）指导小模型（学生）学习，混合 CE Loss 和 KL 散度。

### 命令行方式

```bash
cd trainer
python train_distillation.py \
    --data_path ../dataset/sft_mini_512_debug.jsonl \
    --from_student_weight none \
    --from_teacher_weight none \
    --batch_size 2 \
    --accumulation_steps 1 \
    --num_workers 0 \
    --epochs 1 \
    --log_interval 1 \
    --save_interval 50 \
    --student_hidden_size 512 \
    --student_num_layers 2 \
    --teacher_hidden_size 512 \
    --teacher_num_layers 2 \
    --max_seq_len 128 \
    --device cpu
```

### PyCharm 方式

右上角选 **`06_debug_distill`**，直接点 Debug 按钮即可打断点。

> 调试时教师和学生用同样大小的模型（512/2 层），正式训练教师应更大（如 768/16 层）。

## 14. PPO / GRPO / SPO 强化学习调试

三个强化学习训练脚本通过 `--reward_mode` 参数切换奖励模型：

| 模式 | 用途 | 依赖 |
|------|------|------|
| `mock` | 随机分数，调试训练循环 | 无（Mac CPU 可跑） |
| `api` | 调用 OpenAI 兼容 API 打分 | LiteLLM / GLM-4.7 等 |
| `local` | 加载本地 HF 模型（默认） | internlm2-1_8b-reward |

### 命令行方式（Mock 模式）

```bash
cd trainer

# PPO
python train_ppo.py --data_path ../dataset/rlaif_debug.jsonl --reward_mode mock \
    --batch_size 2 --num_workers 0 --epochs 1 --num_hidden_layers 2 \
    --max_seq_len 32 --max_gen_len 64 --device cpu

# GRPO
python train_grpo.py --data_path ../dataset/rlaif_debug.jsonl --reward_mode mock \
    --batch_size 2 --num_workers 0 --epochs 1 --num_hidden_layers 2 \
    --max_seq_len 32 --max_gen_len 64 --num_generations 2 --device cpu

# SPO
python train_spo.py --data_path ../dataset/rlaif_debug.jsonl --reward_mode mock \
    --batch_size 2 --num_workers 0 --epochs 1 --num_hidden_layers 2 \
    --max_seq_len 32 --max_gen_len 64 --device cpu
```

### API 模式（需要 LiteLLM 代理在线）

```bash
python train_grpo.py --data_path ../dataset/rlaif_debug.jsonl --reward_mode api \
    --reward_api_url http://192.168.0.81:4000/v1 \
    --reward_api_key sk-fsl123456 \
    --reward_api_model GLM-4.7 \
    --batch_size 2 --num_workers 0 --epochs 1 --num_hidden_layers 2 \
    --max_seq_len 32 --max_gen_len 64 --num_generations 2 --device cpu
```

### PyCharm 方式

右上角选 **`07_debug_ppo`**、**`08_debug_grpo`** 或 **`09_debug_spo`**，已配置 `--reward_mode mock`，直接点 Debug 即可。

## 15. 完整流程

```
预训练 → SFT → LoRA → 推理蒸馏 → DPO → PPO/GRPO/SPO
  ↓       ↓      ↓       ↓        ↓
  01      02     03      04       05~09  (debug_*)
  ↓       ↓      ↓       ↓
  10      11     12      13              (test_*)
```

1. **`01_debug_pretrain`** — 训练预训练模型（学会"语言"）
2. **`02_debug_sft`** — SFT 微调（学会"回答问题"）
3. **`03_debug_lora`** — LoRA 微调（轻量适配特定场景）
4. **`04_debug_reason`** — 推理蒸馏（学会"先想后答"）
5. **`05_debug_dpo`** — DPO 偏好优化（学会"人类偏好"）
6. **`06_debug_distill`** — 知识蒸馏（大模型教小模型）
7. **`07~09_debug_ppo/grpo/spo`** — 强化学习（mock 模式，CPU 可跑）
8. **`10~13_test_*`** — 分别测试各阶段模型的推理效果

> 调试模式下各步骤可独立运行（`--from_weight none`），无需等前一步完成。

## 16. 注意事项

- Mac 没有 CUDA，只能用 `--device cpu`，速度较慢但功能完整
- `num_workers=0` 是 PyCharm 断点调试的关键，否则 DataLoader 子进程里的断点不会触发
- 训练产出的权重保存在 `out/` 目录（LoRA 权重在 `out/lora/`）
- 正式训练流程：预训练 → SFT（`--from_weight pretrain`）→ LoRA（`--from_weight full_sft`）→ 推理蒸馏（`--from_weight dpo`）
