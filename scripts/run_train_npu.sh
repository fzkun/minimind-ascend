#!/bin/bash
# MiniMind 昇腾910B 8卡分布式训练启动脚本
# 用法: bash scripts/run_train_npu.sh pretrain --epochs 1 --batch_size 32
#
# Docker启动命令示例:
# docker run -it --rm --network=host --shm-size=500g \
#     --device /dev/davinci0 --device /dev/davinci1 \
#     --device /dev/davinci2 --device /dev/davinci3 \
#     --device /dev/davinci4 --device /dev/davinci5 \
#     --device /dev/davinci6 --device /dev/davinci7 \
#     --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
#     -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
#     -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
#     -v $(pwd)/dataset:/workspace/minimind/dataset \
#     -v $(pwd)/out:/workspace/minimind/out \
#     minimind-npu bash scripts/run_train_npu.sh pretrain --epochs 1 --batch_size 32

TRAIN_MODE=${1:-pretrain}
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR/trainer"

if [ "$TRAIN_MODE" = "tokenizer" ]; then
    # 分词器训练为单进程，不需要 torchrun
    python train_tokenizer.py "$@"
else
    torchrun --nnodes=1 --nproc_per_node=8 --master_port=29500 \
        train_${TRAIN_MODE}.py "$@"
fi
