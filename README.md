<div align="center">

![logo](./images/logo.png)

# MiniMind-Ascend

**基于华为昇腾 NPU 的 MiniMind 训练与部署方案**

</div>

<div align="center">

[![GitHub Repo stars](https://img.shields.io/github/stars/fzkun/minimind-ascend?style=social)](https://github.com/fzkun/minimind-ascend/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/fzkun/minimind-ascend)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/fzkun/minimind-ascend)](https://github.com/fzkun/minimind-ascend/commits/master)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/fzkun/minimind-ascend/pulls)
[![Upstream](https://img.shields.io/badge/upstream-jingyaogong%2Fminimind-orange)](https://github.com/jingyaogong/minimind)
[![Demo](https://img.shields.io/badge/demo-GitHub%20Pages-blue)](https://fzkun.github.io/minimind-ascend/)

</div>

<div align="center">
  <h3>"大道至简，国产芯片也能训大模型"</h3>
</div>

<div align="center">

**在线体验：https://fzkun.github.io/minimind-ascend/**

</div>

---

> **本项目是 [MiniMind](https://github.com/jingyaogong/minimind) 的昇腾 NPU 适配分支**，在原项目基础上完成了华为昇腾 910B（Atlas 800I A2）的全流程适配，实现了国产硬件上从预训练到部署的完整 LLM 训练链路。

## 相比上游的主要变化

- **昇腾 NPU 全流程适配**：所有训练脚本（Pretrain / SFT / LoRA / DPO / PPO / GRPO / SPO / 蒸馏 / 推理训练）均已适配华为昇腾 910B，支持单卡和 8 卡分布式训练（HCCL 通信后端）
- **一键训练部署脚本**：`scripts/run_all_npu.sh` 编排全流程（数据下载 → 镜像构建 → 多阶段训练 → 模型转换 → vLLM 部署），一条命令完成
- **Docker 容器化**：提供 `Dockerfile.ascend` 构建 NPU 训练镜像，开箱即用
- **模型转换与 vLLM 部署**：支持 Dense → LlamaForCausalLM、MoE → Qwen2MoeForCausalLM 格式转换，直接用 vLLM-Ascend 推理
- **工具调用 (Tool Calling)**：新增工具调用数据准备、训练、评估全流程，含 OpenAI 兼容 API
- **交互式可视化教学**：基于 React 的 LLM 架构可视化教学页面（含部署与昇腾实战章节）
- **训练管理后端**：FastAPI 训练控制 REST API + 实时日志 SSE 流，配合 Web 页面使用
- **CUDA/NPU 自动切换**：代码自动检测 `torch_npu`，有则走 NPU 路径，否则回退 CUDA，无需手动改代码

## 硬件要求

| 项目 | 要求 |
|------|------|
| 硬件 | Atlas 800I A2（昇腾 910B × 8）或单卡 |
| 驱动 | Ascend NPU 驱动（`/usr/local/Ascend/driver`） |
| 软件 | Docker |

## 模型规格

| 类型 | 参数量 | hidden_size | num_hidden_layers | 说明 |
|------|--------|-------------|-------------------|------|
| Dense-Small | 26M | 512 | 8 | 默认配置 |
| Dense-Base | 104M | 768 | 16 | 大模型 |
| MoE | 145M | 768 + MoE | 16 | 4路由专家 + 1共享专家, top-2 |

## 快速开始

```bash
# 克隆项目
git clone https://github.com/fzkun/minimind-ascend.git
cd minimind-ascend

# 一键完整训练（下载数据 → 构建镜像 → pretrain → sft → dpo → reason → eval）
bash scripts/run_all_npu.sh all

# 模型转换 + vLLM 部署
bash scripts/run_all_npu.sh serve

# MoE 模型训练 + 部署
bash scripts/run_all_npu.sh --use-moe --hidden-size 768 --num-hidden-layers 16 all
bash scripts/run_all_npu.sh --use-moe --hidden-size 768 --num-hidden-layers 16 serve
```

### 阶段预设

| 预设 | 展开 |
|------|------|
| `all` | download build pretrain full_sft dpo reason eval |
| `core` | download build pretrain full_sft eval |
| `serve` | convert vllm |
| `rl` | ppo grpo spo |

可用阶段：`download` `build` `tokenizer` `pretrain` `full_sft` `lora` `dpo` `reason` `ppo` `grpo` `spo` `distillation` `eval` `convert` `vllm`

详细训练指南请参考 **[README-Ascend.md](./README-Ascend.md)**。

## 训练流程

```
Pretrain → SFT → [LoRA | DPO | PPO | GRPO | SPO | 蒸馏 | 推理训练] → 模型转换 → vLLM 部署
```

所有阶段均通过 Docker 容器在昇腾 NPU 上运行，8 卡分布式训练使用 HCCL 通信后端。

### NPU 8 卡训练

```bash
bash scripts/run_all_npu.sh pretrain --epochs 1 --batch_size 32
```

### 单卡训练

```bash
docker run -it --rm --network=host \
    --device /dev/davinci0 --device /dev/davinci_manager \
    --device /dev/devmm_svm --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
    -v $(pwd)/dataset:/workspace/minimind/dataset \
    -v $(pwd)/out:/workspace/minimind/out \
    minimind-npu python trainer/train_pretrain.py --device npu:0 --epochs 1 --batch_size 4
```

### 模型转换 + vLLM 部署

```bash
# Dense 模型
bash scripts/run_all_npu.sh serve

# MoE 模型（转换为 Qwen2MoE 格式，保留共享专家）
bash scripts/run_all_npu.sh --use-moe --hidden-size 768 --num-hidden-layers 16 serve
```

## 工具调用 (Tool Calling)

新增工具调用训练能力，模型学会在对话中调用外部工具（天气查询、计算、搜索等）。

```bash
# 数据准备（从 HuggingFace 下载并转换）
python scripts/data_prepare_toolcall.py --source glaive --max_samples 10000

# 训练（基于 SFT 权重继续训练）
python trainer/train_full_sft.py \
    --data_path ../dataset/sft_tool_call.jsonl \
    --from_weight full_sft --save_weight tool_sft --max_seq_len 512

# 自动评估
python scripts/eval_tool_call.py --weight tool_sft --mode auto
```

API 服务已支持 OpenAI 兼容的 `tools` 字段：

```bash
python scripts/serve_openai_api.py --weight tool_sft
```

## Web 可视化教学

在线体验：**https://fzkun.github.io/minimind-ascend/**

基于 React + TypeScript 的交互式 LLM 架构可视化教学页面，包含：
- Tokenization、Embedding、注意力机制、RoPE、FFN/MoE 等核心概念
- 部署与昇腾实战章节
- 工具调用测试 Playground
- 训练管理面板（启停训练、实时日志）

```bash
# 开发模式
cd docs/tutorial/react-app && npm run dev

# Docker 部署
bash scripts/run_all_npu.sh web

# 一键部署全部（vLLM + Web）
docker compose up -d
```

## 与 CUDA 版本的差异

| 项目 | CUDA | 昇腾 NPU |
|------|------|----------|
| 通信后端 | NCCL | HCCL |
| 默认精度 | bfloat16 | float16 |
| torch.compile | 支持 | 跳过 |
| GradScaler | `torch.cuda.amp.GradScaler` | `torch.npu.amp.GradScaler` |
| 设备名称 | `cuda:0` | `npu:0` |

代码已做自动检测，无需手动切换。

## 项目文件清单

```
Dockerfile.ascend              — NPU 训练 Docker 构建文件
docker-compose.yml             — 一键部署 vLLM + Web
requirements_npu.txt           — NPU 环境依赖
scripts/run_all_npu.sh         — 全流程编排（训练 + 转换 + 部署）
scripts/run_train_npu.sh       — 8 卡分布式训练启动脚本
scripts/convert_to_hf.py       — 模型转换（Dense→Llama / MoE→Qwen2MoE）
scripts/data_prepare_toolcall.py — 工具调用数据准备
scripts/eval_tool_call.py      — 工具调用评估
scripts/serve_openai_api.py    — OpenAI 兼容 API（含 Tool Calling）
scripts/serve_train_manager.py — 训练管理后端（REST API + SSE）
trainer/trainer_utils.py       — 核心工具模块（NPU/CUDA 双路径）
trainer/train_*.py             — 各阶段训练脚本（均已适配 NPU）
docs/tutorial/react-app/       — 交互式可视化教学 Web 应用
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=fzkun/minimind-ascend&type=Date)](https://star-history.com/#fzkun/minimind-ascend&Date)

## 致谢

本项目基于 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 开发，感谢原作者的开源贡献。

## License

[Apache-2.0](LICENSE)
