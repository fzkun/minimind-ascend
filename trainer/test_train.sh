#!/bin/bash
# 训练烟雾测试：验证所有训练任务都能正常跑通
# 用法: bash trainer/test_train.sh
#
# 每个任务只训练 1 epoch、2 batch，CPU 上几秒钟跑完
# 主要验证代码路径没有报错，不关心 loss 收敛
#
# 注意: PPO/GRPO/SPO 需要外部 reward model（internlm2-1_8b-reward），
#       Mac 上无法测试，不包含在烟雾测试中。

set -eo pipefail  # 任何命令失败就退出（含管道）

cd "$(dirname "$0")/../trainer"

COMMON_ARGS="--batch_size 2 --num_workers 0 --epochs 1 --num_hidden_layers 2 --max_seq_len 128 --device cpu --log_interval 1 --save_interval 999 --accumulation_steps 1"
# 蒸馏脚本用 student/teacher 参数名，不包含 --num_hidden_layers
DISTILL_ARGS="--batch_size 2 --num_workers 0 --epochs 1 --max_seq_len 128 --device cpu --log_interval 1 --save_interval 999 --accumulation_steps 1"

echo "=========================================="
echo "  MiniMind 训练烟雾测试"
echo "=========================================="

# ---------- 1. 预训练 ----------
echo ""
echo "[1/6] 测试预训练 (train.py --task pretrain) ..."
python train.py --task pretrain \
    --data_path ../dataset/pretrain_hq_debug.jsonl \
    $COMMON_ARGS 2>&1 | tail -3
echo "  ✓ 预训练通过"

# ---------- 2. SFT ----------
echo ""
echo "[2/6] 测试 SFT (train.py --task sft) ..."
python train.py --task sft \
    --data_path ../dataset/sft_mini_512_debug.jsonl \
    --from_weight none \
    $COMMON_ARGS 2>&1 | tail -3
echo "  ✓ SFT 通过"

# ---------- 3. LoRA ----------
echo ""
echo "[3/6] 测试 LoRA (train.py --task lora) ..."
python train.py --task lora \
    --data_path ../dataset/lora_identity_debug.jsonl \
    --from_weight none \
    $COMMON_ARGS 2>&1 | tail -3
echo "  ✓ LoRA 通过"

# ---------- 4. 推理蒸馏 ----------
echo ""
echo "[4/6] 测试推理蒸馏 (train.py --task reason) ..."
python train.py --task reason \
    --data_path ../dataset/r1_mix_1024_debug.jsonl \
    --from_weight none \
    $COMMON_ARGS 2>&1 | tail -3
echo "  ✓ 推理蒸馏通过"

# ---------- 5. DPO ----------
echo ""
echo "[5/6] 测试 DPO (train_dpo.py) ..."
python train_dpo.py \
    --data_path ../dataset/dpo_debug.jsonl \
    --from_weight none \
    $COMMON_ARGS 2>&1 | tail -3
echo "  ✓ DPO 通过"

# ---------- 6. 知识蒸馏 ----------
echo ""
echo "[6/6] 测试知识蒸馏 (train_distillation.py) ..."
python train_distillation.py \
    --data_path ../dataset/sft_mini_512_debug.jsonl \
    --from_student_weight none --from_teacher_weight none \
    --student_hidden_size 512 --student_num_layers 2 \
    --teacher_hidden_size 512 --teacher_num_layers 2 \
    $DISTILL_ARGS 2>&1 | tail -3
echo "  ✓ 知识蒸馏通过"

# ---------- 7. 薄入口兼容性 ----------
echo ""
echo "[bonus] 测试薄入口脚本兼容性 ..."
python train_pretrain.py --data_path ../dataset/pretrain_hq_debug.jsonl $COMMON_ARGS 2>&1 | tail -1
python train_full_sft.py --data_path ../dataset/sft_mini_512_debug.jsonl --from_weight none $COMMON_ARGS 2>&1 | tail -1
python train_lora.py --data_path ../dataset/lora_identity_debug.jsonl --from_weight none $COMMON_ARGS 2>&1 | tail -1
python train_reason.py --data_path ../dataset/r1_mix_1024_debug.jsonl --from_weight none $COMMON_ARGS 2>&1 | tail -1
echo "  ✓ 所有薄入口通过"

echo ""
echo "=========================================="
echo "  全部通过！（PPO/GRPO/SPO 需要 reward model，跳过）"
echo "=========================================="
