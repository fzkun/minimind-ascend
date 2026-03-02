# MiniMind Model 模块说明

## 文件结构

```
model/
├── model_config.py      # 模型配置（设计图纸）
├── model_norm.py         # 归一化层
├── model_attention.py    # 位置编码 + 注意力层
├── model_ffn.py          # 前馈网络 + MoE 混合专家
├── model_block.py        # 单层 Transformer
├── model_minimind.py     # 最终模型（组装）
├── model_lora.py         # LoRA 微调
└── __init__.py
```

## 各文件职责

### model_config.py — 模型配置
`MiniMindConfig`：控制模型大小的所有超参数。

关键参数：
- `hidden_size` — 模型宽度（默认 512）
- `num_hidden_layers` — 模型深度/层数（默认 8）
- `num_attention_heads` — Q 头数（默认 8）
- `num_key_value_heads` — KV 头数，GQA 用（默认 2）
- `vocab_size` — 词表大小（默认 6400）
- `use_moe` — 是否启用混合专家

### model_norm.py — 归一化层
`RMSNorm`：比 LayerNorm 更简单更快的归一化，保持每层输入数值稳定。

### model_attention.py — 位置编码 + 注意力
- `precompute_freqs_cis` — 预计算 RoPE 旋转位置编码
- `apply_rotary_pos_emb` — 把位置信息旋转到 Q/K 中
- `repeat_kv` — GQA 辅助，复制 KV 头
- `Attention` — 多头注意力层，支持 Flash Attention 和 KV Cache

### model_ffn.py — 前馈网络 + MoE
- `FeedForward` — SwiGLU 前馈网络（gate + up + down）
- `MoEGate` — MoE 门控，给专家打分选 top_k
- `MOEFeedForward` — MoE 前馈，多个专家 + 门控 + 共享专家

### model_block.py — 单层 Transformer
`MiniMindBlock`：一层完整的 Transformer = RMSNorm → 注意力 → 残差 → RMSNorm → FFN → 残差

### model_minimind.py — 最终模型
- `MiniMindModel` — 堆叠 N 层 Block，加上词嵌入和 RoPE
- `MiniMindForCausalLM` — 在主干上加 lm_head，预测下一个 token

## 依赖关系

```
config ──→ norm ──→ block ──→ minimind
  │          ↑         ↑          ↑
  ├──→ attention ──────┘          │
  └──→ ffn ───────────────────────┘
```

## 数据流向

```
输入 token ID
    ↓
Embedding（查词表，变成向量）
    ↓
N × MiniMindBlock:
    ├→ RMSNorm → Attention（Q/K/V + RoPE + 因果掩码）→ 残差相加
    └→ RMSNorm → FFN 或 MoE（非线性变换）→ 残差相加
    ↓
RMSNorm
    ↓
lm_head（映射回词表大小）
    ↓
logits（每个词的概率）→ 交叉熵损失
```
