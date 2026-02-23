# MiniMind 从零开始的 LLM 教程

> 用最少的代码，理解最核心的原理。

## 这套教程适合谁？

- 有 Python 基础，了解 PyTorch 基本用法（Tensor、nn.Module、反向传播）
- 对大语言模型（LLM）感兴趣，但还没有从头训练过
- 希望通过**真实可运行的代码**理解原理，而不是只看论文和公式

## 推荐阅读顺序

```
01 入门概述 ──→ 02 分词器 ──→ 03 模型架构 ──→ 04 预训练
                                                    │
                                          ┌─────────┴─────────┐
                                          ▼                   ▼
                                      05 监督微调          07 进阶技术
                                          │
                                          ▼
                                      06 对齐训练
```

建议按顺序从 01 到 07 阅读。其中 03（模型架构）是核心章节，篇幅最长，建议配合源码仔细阅读。

## 各章节概览

| 章节 | 主题 | 关键内容 |
|------|------|----------|
| [01-introduction](01-introduction.md) | LLM 入门概述 | 什么是大语言模型、下一个词预测、MiniMind 定位、训练全流程鸟瞰 |
| [02-tokenizer](02-tokenizer.md) | 分词器原理 | BPE 算法、6400 词表、特殊 token、Chat Template |
| [03-model-architecture](03-model-architecture.md) | 模型架构详解 | Transformer 结构、RMSNorm、自注意力、RoPE、SwiGLU、MoE |
| [04-pretrain](04-pretrain.md) | 预训练 | 因果语言建模、混合精度、梯度累积、学习率调度、DDP |
| [05-sft](05-sft.md) | 监督微调 | 对话数据格式、选择性损失掩码、SFT 与预训练的区别 |
| [06-alignment](06-alignment.md) | 对齐训练 | DPO、PPO、GRPO、SPO 四种方法对比 |
| [07-advanced](07-advanced.md) | 进阶技术 | LoRA、知识蒸馏、推理训练、部署 |

## 涉及的核心源码

```
model/
├── model_minimind.py          # 模型架构（Config + Model + Attention + FFN + MoE）
├── model_lora.py              # LoRA 低秩适配实现
└── tokenizer_config.json      # 分词器配置

dataset/
└── lm_dataset.py              # 4 种数据集类（Pretrain/SFT/DPO/RLAIF）

trainer/
├── trainer_utils.py           # 训练工具（分布式初始化、模型加载、checkpoint）
├── train_pretrain.py          # 预训练
├── train_full_sft.py          # 全参数监督微调
├── train_lora.py              # LoRA 微调
├── train_dpo.py               # DPO 偏好对齐
├── train_ppo.py               # PPO 强化学习对齐
├── train_grpo.py              # GRPO 组内相对优化
├── train_spo.py               # SPO 自适应基线优化
├── train_distillation.py      # 知识蒸馏
└── train_reason.py            # 推理训练

eval_llm.py                    # 推理与对话
```

## 如何使用本教程

1. **先读文档**：每个章节先用通俗类比解释概念
2. **再看代码**：文档中会引用源码片段，标注了文件路径和行号
3. **动手运行**：每章末尾有实操命令，可以在自己的机器上跑起来
4. **对照完整实现**：根据行号引用，打开对应源文件查看完整上下文
