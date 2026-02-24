#!/bin/bash
# MiniMind 昇腾910B 全流程训练编排脚本
# 支持一键运行完整训练流水线（数据下载 → 构建镜像 → 各阶段训练 → 推理测试）
#
# 用法:
#   bash scripts/run_all_npu.sh [选项] <阶段...>
#
# 阶段名:
#   download    — 从 ModelScope 下载数据集
#   build       — 构建 Docker 镜像
#   tokenizer   — 训练分词器（单进程）
#   pretrain    — 预训练（8卡）
#   full_sft    — 监督微调（8卡）
#   lora        — LoRA 微调（8卡）
#   dpo         — DPO 偏好对齐（8卡）
#   reason      — 推理训练（8卡）
#   ppo         — PPO 强化学习（8卡，需 reward 模型）
#   grpo        — GRPO 强化学习（8卡，需 reward 模型）
#   spo         — SPO 强化学习（8卡，需 reward 模型）
#   distillation — 知识蒸馏（8卡，需教师模型）
#   eval        — 推理测试（单卡）
#   convert     — 转换模型为 HuggingFace 格式（供 vLLM 使用）
#   vllm        — 使用 vLLM 启动 OpenAI 兼容 API 服务
#   web         — 构建并启动前端 Web 服务（Nginx Docker）
#
# 预设组合:
#   all   = download build pretrain full_sft dpo reason eval
#   core  = download build pretrain full_sft eval
#   serve = convert vllm web
#   rl    = ppo grpo spo
#
# 选项:
#   --resume              各训练阶段启用断点续训
#   --hidden-size N       模型隐藏层维度（默认 512）
#   --num-hidden-layers N 隐藏层数量（默认 8，768维度建议16）
#   --reward-model PATH   Reward 模型路径（RL 阶段必需）
#   --teacher-hidden-size N  教师模型维度（蒸馏用，默认 768）
#   --inside-docker       跳过 Docker 启动，直接在容器内执行
#   --force-build         强制重建 Docker 镜像
#   --weight NAME         权重名称前缀（convert/vllm 用，默认 full_sft）
#   --vllm-image IMAGE    vLLM Docker 镜像（默认 quay.io/ascend/vllm-ascend:v0.13.0）
#   --vllm-port PORT      vLLM 服务端口（默认 8000）
#   --max-model-len N     vLLM 最大序列长度（默认 2048）
#   --use-moe             启用 MoE 架构（145M 参数，权重带 _moe 后缀）
#   --dry-run             仅打印将要执行的命令，不实际执行
#   --help                显示帮助信息

set -eo pipefail

# ============================================================
# 配置变量
# ============================================================
IMAGE_NAME="minimind-npu"
HIDDEN_SIZE=512
NUM_HIDDEN_LAYERS=8
TEACHER_HIDDEN_SIZE=768
REWARD_MODEL_PATH=""
USE_MOE=0
RESUME_FLAG=""
INSIDE_DOCKER=0
FORCE_BUILD=0
DRY_RUN=0
WEIGHT_NAME="full_sft"
VLLM_IMAGE="quay.io/ascend/vllm-ascend:v0.13.0"
VLLM_PORT=8000
MAX_MODEL_LEN=2048
VLLM_CONTAINER_NAME="vllm-minimind"
WEB_PORT=8080
WEB_IMAGE_NAME="192.168.0.81:3001/ascend/minimind-web:latest"
WEB_CONTAINER_NAME="minimind-web"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ModelScope 数据集 ID
DATASET_ID="gongjy/minimind_dataset"

# 需要下载的数据文件
DATA_FILES=(
    pretrain_hq.jsonl
    sft_mini_512.jsonl
    lora_identity.jsonl
    dpo.jsonl
    r1_mix_1024.jsonl
    rlaif-mini.jsonl
)

# ============================================================
# 工具函数
# ============================================================
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2
}

show_help() {
    head -n 45 "$0" | tail -n +2 | sed 's/^# \?//'
    exit 0
}

# 检查数据文件是否存在
check_data() {
    local file="$1"
    if [ ! -f "$PROJECT_DIR/dataset/$file" ]; then
        log_error "数据文件不存在: dataset/$file"
        log_error "请先运行: bash scripts/run_all_npu.sh download"
        return 1
    fi
}

# 检查前置权重是否存在
check_weight() {
    local weight_name="$1"
    local hidden="$2"
    local moe_suffix=""
    [ "$USE_MOE" -eq 1 ] && moe_suffix="_moe"
    local pth="$PROJECT_DIR/out/${weight_name}_${hidden}${moe_suffix}.pth"
    if [ ! -f "$pth" ]; then
        log_error "前置权重不存在: $pth"
        log_error "请先完成 $weight_name 阶段的训练"
        return 1
    fi
}

# Docker 容器启动封装
run_docker() {
    local extra_mounts=("$@")
    # 检测是否有 TTY，有则用 -it，否则只用 -i
    local tty_flag="-i"
    if [ -t 0 ]; then
        tty_flag="-it"
    fi
    local cmd_args=(
        docker run $tty_flag --rm --network=host --shm-size=500g
        --device /dev/davinci0
        --device /dev/davinci1
        --device /dev/davinci2
        --device /dev/davinci3
        --device /dev/davinci4
        --device /dev/davinci5
        --device /dev/davinci6
        --device /dev/davinci7
        --device /dev/davinci_manager
        --device /dev/devmm_svm
        --device /dev/hisi_hdc
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
        -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro
        -v "$PROJECT_DIR/dataset":/workspace/minimind/dataset
        -v "$PROJECT_DIR/out":/workspace/minimind/out
        -v "$PROJECT_DIR/checkpoints":/workspace/minimind/checkpoints
    )
    for m in "${extra_mounts[@]}"; do
        cmd_args+=(-v "$m")
    done
    cmd_args+=("$IMAGE_NAME")
    echo "${cmd_args[@]}"
}

# 执行命令（支持 dry-run）
run_cmd() {
    log_info "执行: $*"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run] $*"
        return 0
    fi
    eval "$@"
}

# 记录阶段耗时
run_stage() {
    local stage_name="$1"
    shift
    log_info "========== 开始阶段: $stage_name =========="
    local start_time=$SECONDS
    local rc=0
    "$@" || rc=$?
    local elapsed=$(( SECONDS - start_time ))
    local minutes=$(( elapsed / 60 ))
    local seconds=$(( elapsed % 60 ))
    if [ "$rc" -ne 0 ]; then
        log_error "========== 阶段失败: $stage_name (耗时 ${minutes}分${seconds}秒) =========="
    else
        log_info "========== 阶段完成: $stage_name (耗时 ${minutes}分${seconds}秒) =========="
    fi
    return $rc
}

# ============================================================
# 各阶段函数
# ============================================================

stage_download() {
    log_info "下载数据集到 $PROJECT_DIR/dataset/"
    mkdir -p "$PROJECT_DIR/dataset"
    for file in "${DATA_FILES[@]}"; do
        if [ -f "$PROJECT_DIR/dataset/$file" ]; then
            log_info "已存在，跳过: dataset/$file"
            continue
        fi
        run_cmd modelscope download --dataset "$DATASET_ID" "$file" --local_dir "$PROJECT_DIR/dataset"
    done
}

stage_build() {
    if [ "$FORCE_BUILD" -eq 0 ] && docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
        log_info "镜像 $IMAGE_NAME 已存在，跳过构建（使用 --force-build 强制重建）"
        return 0
    fi
    run_cmd docker build -f "$PROJECT_DIR/Dockerfile.ascend" -t "$IMAGE_NAME" "$PROJECT_DIR"
}

stage_tokenizer() {
    check_data "pretrain_hq.jsonl" || return 1
    if [ "$INSIDE_DOCKER" -eq 1 ]; then
        run_cmd bash "$PROJECT_DIR/scripts/run_train_npu.sh" tokenizer
    else
        local docker_cmd
        docker_cmd=$(run_docker)
        run_cmd "$docker_cmd bash scripts/run_train_npu.sh tokenizer"
    fi
}

stage_pretrain() {
    check_data "pretrain_hq.jsonl" || return 1
    local train_args="--epochs 1 --batch_size 32 --learning_rate 5e-4 --hidden_size $HIDDEN_SIZE --num_hidden_layers $NUM_HIDDEN_LAYERS --data_path ../dataset/pretrain_hq.jsonl $MOE_FLAG $RESUME_FLAG"
    if [ "$INSIDE_DOCKER" -eq 1 ]; then
        run_cmd bash "$PROJECT_DIR/scripts/run_train_npu.sh" pretrain $train_args
    else
        local docker_cmd
        docker_cmd=$(run_docker)
        run_cmd "$docker_cmd bash scripts/run_train_npu.sh pretrain $train_args"
    fi
}

stage_full_sft() {
    check_data "sft_mini_512.jsonl" || return 1
    check_weight "pretrain" "$HIDDEN_SIZE" || return 1
    local train_args="--epochs 2 --batch_size 16 --learning_rate 1e-6 --hidden_size $HIDDEN_SIZE --num_hidden_layers $NUM_HIDDEN_LAYERS --data_path ../dataset/sft_mini_512.jsonl $MOE_FLAG $RESUME_FLAG"
    if [ "$INSIDE_DOCKER" -eq 1 ]; then
        run_cmd bash "$PROJECT_DIR/scripts/run_train_npu.sh" full_sft $train_args
    else
        local docker_cmd
        docker_cmd=$(run_docker)
        run_cmd "$docker_cmd bash scripts/run_train_npu.sh full_sft $train_args"
    fi
}

stage_lora() {
    check_data "lora_identity.jsonl" || return 1
    check_weight "full_sft" "$HIDDEN_SIZE" || return 1
    local train_args="--epochs 50 --batch_size 32 --learning_rate 1e-4 --hidden_size $HIDDEN_SIZE --num_hidden_layers $NUM_HIDDEN_LAYERS --data_path ../dataset/lora_identity.jsonl $MOE_FLAG $RESUME_FLAG"
    if [ "$INSIDE_DOCKER" -eq 1 ]; then
        run_cmd bash "$PROJECT_DIR/scripts/run_train_npu.sh" lora $train_args
    else
        local docker_cmd
        docker_cmd=$(run_docker)
        run_cmd "$docker_cmd bash scripts/run_train_npu.sh lora $train_args"
    fi
}

stage_dpo() {
    check_data "dpo.jsonl" || return 1
    check_weight "full_sft" "$HIDDEN_SIZE" || return 1
    local train_args="--epochs 1 --batch_size 4 --learning_rate 4e-8 --hidden_size $HIDDEN_SIZE --num_hidden_layers $NUM_HIDDEN_LAYERS --data_path ../dataset/dpo.jsonl $MOE_FLAG $RESUME_FLAG"
    if [ "$INSIDE_DOCKER" -eq 1 ]; then
        run_cmd bash "$PROJECT_DIR/scripts/run_train_npu.sh" dpo $train_args
    else
        local docker_cmd
        docker_cmd=$(run_docker)
        run_cmd "$docker_cmd bash scripts/run_train_npu.sh dpo $train_args"
    fi
}

stage_reason() {
    check_data "r1_mix_1024.jsonl" || return 1
    check_weight "dpo" "$HIDDEN_SIZE" || return 1
    local train_args="--epochs 1 --batch_size 8 --learning_rate 1e-6 --hidden_size $HIDDEN_SIZE --num_hidden_layers $NUM_HIDDEN_LAYERS --from_weight dpo --data_path ../dataset/r1_mix_1024.jsonl $MOE_FLAG $RESUME_FLAG"
    if [ "$INSIDE_DOCKER" -eq 1 ]; then
        run_cmd bash "$PROJECT_DIR/scripts/run_train_npu.sh" reason $train_args
    else
        local docker_cmd
        docker_cmd=$(run_docker)
        run_cmd "$docker_cmd bash scripts/run_train_npu.sh reason $train_args"
    fi
}

stage_ppo() {
    check_data "rlaif-mini.jsonl" || return 1
    check_weight "reason" "$HIDDEN_SIZE" || return 1
    if [ -z "$REWARD_MODEL_PATH" ]; then
        log_error "PPO 阶段需要 Reward 模型，请通过 --reward-model 指定路径"
        return 1
    fi
    local train_args="--epochs 1 --batch_size 2 --learning_rate 8e-8 --hidden_size $HIDDEN_SIZE --num_hidden_layers $NUM_HIDDEN_LAYERS --from_weight reason --data_path ../dataset/rlaif-mini.jsonl --reward_model_path $REWARD_MODEL_PATH $MOE_FLAG $RESUME_FLAG"
    if [ "$INSIDE_DOCKER" -eq 1 ]; then
        run_cmd bash "$PROJECT_DIR/scripts/run_train_npu.sh" ppo $train_args
    else
        local docker_cmd
        docker_cmd=$(run_docker "$REWARD_MODEL_PATH:$REWARD_MODEL_PATH:ro")
        run_cmd "$docker_cmd bash scripts/run_train_npu.sh ppo $train_args"
    fi
}

stage_grpo() {
    check_data "rlaif-mini.jsonl" || return 1
    check_weight "reason" "$HIDDEN_SIZE" || return 1
    if [ -z "$REWARD_MODEL_PATH" ]; then
        log_error "GRPO 阶段需要 Reward 模型，请通过 --reward-model 指定路径"
        return 1
    fi
    local train_args="--epochs 1 --batch_size 2 --learning_rate 8e-8 --hidden_size $HIDDEN_SIZE --num_hidden_layers $NUM_HIDDEN_LAYERS --from_weight reason --data_path ../dataset/rlaif-mini.jsonl --reward_model_path $REWARD_MODEL_PATH $MOE_FLAG $RESUME_FLAG"
    if [ "$INSIDE_DOCKER" -eq 1 ]; then
        run_cmd bash "$PROJECT_DIR/scripts/run_train_npu.sh" grpo $train_args
    else
        local docker_cmd
        docker_cmd=$(run_docker "$REWARD_MODEL_PATH:$REWARD_MODEL_PATH:ro")
        run_cmd "$docker_cmd bash scripts/run_train_npu.sh grpo $train_args"
    fi
}

stage_spo() {
    check_data "rlaif-mini.jsonl" || return 1
    check_weight "reason" "$HIDDEN_SIZE" || return 1
    if [ -z "$REWARD_MODEL_PATH" ]; then
        log_error "SPO 阶段需要 Reward 模型，请通过 --reward-model 指定路径"
        return 1
    fi
    local train_args="--epochs 1 --batch_size 2 --learning_rate 1e-7 --hidden_size $HIDDEN_SIZE --num_hidden_layers $NUM_HIDDEN_LAYERS --from_weight reason --data_path ../dataset/rlaif-mini.jsonl --reward_model_path $REWARD_MODEL_PATH $MOE_FLAG $RESUME_FLAG"
    if [ "$INSIDE_DOCKER" -eq 1 ]; then
        run_cmd bash "$PROJECT_DIR/scripts/run_train_npu.sh" spo $train_args
    else
        local docker_cmd
        docker_cmd=$(run_docker "$REWARD_MODEL_PATH:$REWARD_MODEL_PATH:ro")
        run_cmd "$docker_cmd bash scripts/run_train_npu.sh spo $train_args"
    fi
}

stage_distillation() {
    check_data "sft_mini_512.jsonl" || return 1
    check_weight "full_sft" "$TEACHER_HIDDEN_SIZE" || return 1
    check_weight "full_sft" "$HIDDEN_SIZE" || return 1
    local train_args="--epochs 6 --batch_size 32 --learning_rate 5e-6 --hidden_size $HIDDEN_SIZE --num_hidden_layers $NUM_HIDDEN_LAYERS --data_path ../dataset/sft_mini_512.jsonl --student_hidden_size $HIDDEN_SIZE --teacher_hidden_size $TEACHER_HIDDEN_SIZE $MOE_FLAG $RESUME_FLAG"
    if [ "$INSIDE_DOCKER" -eq 1 ]; then
        run_cmd bash "$PROJECT_DIR/scripts/run_train_npu.sh" distillation $train_args
    else
        local docker_cmd
        docker_cmd=$(run_docker)
        run_cmd "$docker_cmd bash scripts/run_train_npu.sh distillation $train_args"
    fi
}

stage_convert() {
    check_weight "$WEIGHT_NAME" "$HIDDEN_SIZE" || return 1
    local moe_suffix=""
    [ "$USE_MOE" -eq 1 ] && moe_suffix="-moe"
    local hf_dir="$PROJECT_DIR/out/minimind${moe_suffix}-hf"
    local moe_arg=""
    [ "$USE_MOE" -eq 1 ] && moe_arg="--use_moe 1"
    log_info "转换权重 ${WEIGHT_NAME}_${HIDDEN_SIZE}${moe_suffix//-/_}.pth → HuggingFace 格式"
    log_info "输出目录: $hf_dir"
    if [ "$INSIDE_DOCKER" -eq 1 ]; then
        run_cmd "python $PROJECT_DIR/scripts/convert_to_hf.py \
            --save_dir $PROJECT_DIR/out \
            --weight $WEIGHT_NAME \
            --hidden_size $HIDDEN_SIZE \
            --num_hidden_layers $NUM_HIDDEN_LAYERS \
            $moe_arg \
            --output_dir $hf_dir"
    else
        local docker_cmd
        docker_cmd=$(run_docker \
            "$PROJECT_DIR/model:/workspace/minimind/model" \
            "$PROJECT_DIR/scripts:/workspace/minimind/scripts")
        run_cmd "$docker_cmd python scripts/convert_to_hf.py \
            --save_dir out \
            --weight $WEIGHT_NAME \
            --hidden_size $HIDDEN_SIZE \
            --num_hidden_layers $NUM_HIDDEN_LAYERS \
            $moe_arg \
            --output_dir out/minimind${moe_suffix}-hf"
    fi
}

stage_vllm() {
    local moe_suffix=""
    [ "$USE_MOE" -eq 1 ] && moe_suffix="-moe"
    local hf_dir="$PROJECT_DIR/out/minimind${moe_suffix}-hf"
    local model_name="minimind${moe_suffix}"
    local container_name="${VLLM_CONTAINER_NAME}${moe_suffix}"

    if [ ! -f "$hf_dir/config.json" ]; then
        log_error "HuggingFace 模型不存在: $hf_dir/config.json"
        log_error "请先运行 convert 阶段: bash scripts/run_all_npu.sh$([ "$USE_MOE" -eq 1 ] && echo ' --use-moe') convert"
        return 1
    fi

    # 停止已有容器
    if docker ps -q --filter "name=$container_name" | grep -q .; then
        log_info "停止已有 vLLM 容器: $container_name"
        docker stop "$container_name" >/dev/null 2>&1 || true
        sleep 2
    fi
    # 清理已退出的同名容器
    docker rm -f "$container_name" >/dev/null 2>&1 || true

    log_info "启动 vLLM 服务 (模型: $model_name, 端口: $VLLM_PORT, 镜像: $VLLM_IMAGE)"
    run_cmd docker run -d --rm \
        --name "$container_name" \
        --shm-size=1g \
        --network=host \
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
        -v "$hf_dir":/models/minimind:ro \
        "$VLLM_IMAGE" \
        vllm serve /models/minimind \
            --served-model-name "$model_name" \
            --host 0.0.0.0 \
            --port "$VLLM_PORT" \
            --dtype float16 \
            --max-model-len "$MAX_MODEL_LEN"

    if [ "$DRY_RUN" -eq 1 ]; then
        return 0
    fi

    # 等待服务就绪
    log_info "等待 vLLM 服务启动..."
    local max_wait=150
    local elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        if curl -s "http://localhost:$VLLM_PORT/health" >/dev/null 2>&1; then
            log_info "vLLM 服务已就绪!"
            log_info "API 地址: http://localhost:$VLLM_PORT/v1/chat/completions"
            log_info "模型名称: $model_name"
            log_info "容器名称: $container_name"
            log_info "停止服务: docker stop $container_name"

            # 发送测试请求
            log_info "发送测试请求..."
            local response
            response=$(curl -s "http://localhost:$VLLM_PORT/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -d "{
                    \"model\": \"$model_name\",
                    \"messages\": [{\"role\": \"user\", \"content\": \"你好\"}],
                    \"max_tokens\": 64
                }" 2>&1)
            local content
            content=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "(解析失败)")
            log_info "测试回复: $content"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        # 检查容器是否还在运行
        if ! docker ps -q --filter "name=$container_name" | grep -q .; then
            log_error "vLLM 容器已退出，查看日志:"
            docker logs "$container_name" 2>&1 | tail -20
            return 1
        fi
    done
    log_error "vLLM 服务在 ${max_wait}秒 内未就绪"
    docker logs "$container_name" 2>&1 | tail -20
    return 1
}

stage_eval() {
    check_weight "full_sft" "$HIDDEN_SIZE" || return 1
    local eval_moe=""
    [ "$USE_MOE" -eq 1 ] && eval_moe="--use_moe 1"
    if [ "$INSIDE_DOCKER" -eq 1 ]; then
        run_cmd "cd $PROJECT_DIR && echo 0 | python eval_llm.py --weight full_sft --device npu:0 --hidden_size $HIDDEN_SIZE --num_hidden_layers $NUM_HIDDEN_LAYERS $eval_moe"
    else
        local docker_cmd
        docker_cmd=$(run_docker "$PROJECT_DIR/model:/workspace/minimind/model" "$PROJECT_DIR/eval_llm.py:/workspace/minimind/eval_llm.py")
        run_cmd "$docker_cmd bash -c 'echo 0 | python eval_llm.py --weight full_sft --device npu:0 --hidden_size $HIDDEN_SIZE --num_hidden_layers $NUM_HIDDEN_LAYERS $eval_moe'"
    fi
}

stage_web() {
    local web_dir="$PROJECT_DIR/docs/tutorial/react-app"
    if [ ! -f "$web_dir/package.json" ]; then
        log_error "前端项目不存在: $web_dir"
        return 1
    fi

    # 安装依赖 & 构建前端
    log_info "构建前端项目..."
    (cd "$web_dir" && npm install --no-audit --no-fund && npm run build)

    if [ ! -f "$web_dir/dist/index.html" ]; then
        log_error "前端构建失败: $web_dir/dist/index.html 不存在"
        return 1
    fi

    # 构建 Docker 镜像
    log_info "构建前端 Docker 镜像: $WEB_IMAGE_NAME"
    run_cmd docker build -t "$WEB_IMAGE_NAME" "$web_dir"

    # 停止已有容器
    if docker ps -q --filter "name=$WEB_CONTAINER_NAME" | grep -q .; then
        log_info "停止已有前端容器: $WEB_CONTAINER_NAME"
        docker stop "$WEB_CONTAINER_NAME" >/dev/null 2>&1 || true
        sleep 1
    fi
    docker rm -f "$WEB_CONTAINER_NAME" >/dev/null 2>&1 || true

    # 启动容器（--network=host 模式下 nginx 可直接代理 127.0.0.1:8999/8000）
    log_info "启动前端服务 (端口: $WEB_PORT, 镜像: $WEB_IMAGE_NAME)"
    run_cmd docker run -d --rm \
        --name "$WEB_CONTAINER_NAME" \
        --network=host \
        -e WEB_PORT="$WEB_PORT" \
        -v "$web_dir/nginx.conf:/etc/nginx/templates/default.conf.template:ro" \
        "$WEB_IMAGE_NAME"

    if [ "$DRY_RUN" -eq 1 ]; then
        return 0
    fi

    sleep 2
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$WEB_PORT" | grep -q "200"; then
        log_info "前端服务已就绪: http://localhost:$WEB_PORT"
    else
        log_info "前端容器已启动（等待 nginx 就绪...）: http://localhost:$WEB_PORT"
    fi
    log_info "停止服务: docker stop $WEB_CONTAINER_NAME"
}

# ============================================================
# 参数解析
# ============================================================
STAGES=()

while [ $# -gt 0 ]; do
    case "$1" in
        --resume)
            RESUME_FLAG="--from_resume 1"
            shift
            ;;
        --hidden-size)
            HIDDEN_SIZE="$2"
            shift 2
            ;;
        --num-hidden-layers)
            NUM_HIDDEN_LAYERS="$2"
            shift 2
            ;;
        --teacher-hidden-size)
            TEACHER_HIDDEN_SIZE="$2"
            shift 2
            ;;
        --reward-model)
            REWARD_MODEL_PATH="$2"
            shift 2
            ;;
        --inside-docker)
            INSIDE_DOCKER=1
            shift
            ;;
        --force-build)
            FORCE_BUILD=1
            shift
            ;;
        --use-moe)
            USE_MOE=1
            shift
            ;;
        --weight)
            WEIGHT_NAME="$2"
            shift 2
            ;;
        --vllm-image)
            VLLM_IMAGE="$2"
            shift 2
            ;;
        --vllm-port)
            VLLM_PORT="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --web-port)
            WEB_PORT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            show_help
            ;;
        -*)
            log_error "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
        *)
            STAGES+=("$1")
            shift
            ;;
    esac
done

if [ ${#STAGES[@]} -eq 0 ]; then
    log_error "请指定至少一个阶段名或预设组合"
    echo "使用 --help 查看帮助"
    exit 1
fi

# ============================================================
# 展开预设组合
# ============================================================
expand_stages() {
    local expanded=()
    for stage in "$@"; do
        case "$stage" in
            all)
                expanded+=(download build pretrain full_sft dpo reason eval)
                ;;
            core)
                expanded+=(download build pretrain full_sft eval)
                ;;
            serve)
                expanded+=(convert vllm web)
                ;;
            rl)
                expanded+=(ppo grpo spo)
                ;;
            download|build|tokenizer|pretrain|full_sft|lora|dpo|reason|ppo|grpo|spo|distillation|eval|convert|vllm|web)
                expanded+=("$stage")
                ;;
            *)
                log_error "未知阶段: $stage"
                echo "可用阶段: download build tokenizer pretrain full_sft lora dpo reason ppo grpo spo distillation eval convert vllm web"
                echo "预设组合: all core serve rl"
                exit 1
                ;;
        esac
    done
    echo "${expanded[@]}"
}

EXPANDED_STAGES=($(expand_stages "${STAGES[@]}"))

# ============================================================
# 主函数
# ============================================================
# 构建 MoE 参数
MOE_FLAG=""
if [ "$USE_MOE" -eq 1 ]; then
    MOE_FLAG="--use_moe 1"
fi

log_info "MiniMind 昇腾910B 全流程训练"
log_info "模型维度: $HIDDEN_SIZE, 层数: $NUM_HIDDEN_LAYERS$([ "$USE_MOE" -eq 1 ] && echo ', MoE: 开启')"
log_info "执行阶段: ${EXPANDED_STAGES[*]}"
if [ -n "$RESUME_FLAG" ]; then
    log_info "断点续训: 已启用"
fi
if [ "$DRY_RUN" -eq 1 ]; then
    log_info "模式: dry-run（仅打印命令）"
fi
echo ""

# 确保输出目录存在
mkdir -p "$PROJECT_DIR/out" "$PROJECT_DIR/checkpoints"

TOTAL_START=$SECONDS
FAILED=0

for stage in "${EXPANDED_STAGES[@]}"; do
    if ! run_stage "$stage" "stage_$stage"; then
        log_error "阶段 $stage 执行失败，终止流程"
        FAILED=1
        break
    fi
    echo ""
done

TOTAL_ELAPSED=$(( SECONDS - TOTAL_START ))
TOTAL_MIN=$(( TOTAL_ELAPSED / 60 ))
TOTAL_SEC=$(( TOTAL_ELAPSED % 60 ))

if [ "$FAILED" -eq 0 ]; then
    log_info "全部阶段执行完成！总耗时 ${TOTAL_MIN}分${TOTAL_SEC}秒"
else
    log_error "流程中断，总耗时 ${TOTAL_MIN}分${TOTAL_SEC}秒"
    exit 1
fi
