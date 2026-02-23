# 第三章：模型架构 — MiniMind 的大脑

> [上一章：分词器](02-tokenizer.md) | [下一章：预训练](04-pretrain.md)

这是整个教程中**最核心、最重要**的一章。我们将从头到尾拆解 MiniMind 的模型架构——一个完整的 Decoder-Only Transformer。所有代码都在一个文件中：`model/model_minimind.py`，不到 470 行，却实现了现代大语言模型的所有关键技术。

读完这一章，你将理解：
- 一段文本如何变成一串数字（Token IDs），再变成"智能"的输出
- 注意力机制到底在做什么
- 为什么位置编码如此重要
- MoE（混合专家）如何让模型更强

---

## 1. Transformer 整体结构

在深入每个组件之前，先从全局鸟瞰 MiniMind 的数据流。下面这张 ASCII 流程图展示了一段文本从输入到输出的完整路径：

```
Input IDs → Embedding → Dropout → [MiniMindBlock × N] → RMSNorm → LM Head → Logits
            (6400→512)              │                     (归一化)   (512→6400)  (词表概率)
                                    │
                             ┌──────┴──────┐
                             │  RMSNorm    │  ← 输入归一化
                             │  Attention  │  ← 自注意力（含 RoPE、GQA、KV Cache）
                             │  + Residual │  ← 第一个残差连接
                             │  RMSNorm    │  ← 注意力后归一化
                             │  FFN / MoE  │  ← 前馈网络 或 混合专家
                             │  + Residual │  ← 第二个残差连接
                             └─────────────┘
                              （重复 N 次）
                              Small: N=8
                              Base:  N=16
```

**用一句话概括**：把 Token ID 变成向量 -> 经过 N 个 Block 反复"思考" -> 归一化 -> 预测下一个词。

这就是 GPT、LLaMA、Qwen 等所有 Decoder-Only 模型的通用结构。MiniMind 的实现非常精简，让我们逐一拆解每个组件。

---

## 2. Token Embedding —— 从数字到向量

### 2.1 基本概念

上一章讲了分词器如何将文字转换成 Token ID（整数）。但神经网络不能直接处理整数，我们需要将每个整数映射成一个**高维向量**。这就是 Embedding 的作用。

打个比方：Token ID 就像一个学生的学号（比如 `3456`），而 Embedding 向量就像这个学生的"能力画像"——一个包含了语文、数学、英语等各项能力分数的向量。网络通过这些"能力画像"来理解每个词。

### 2.2 MiniMind 的实现

```python
# model/model_minimind.py:L381
self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
```

- `vocab_size = 6400`：词表大小，即有 6400 个不同的 Token
- `hidden_size = 512`（Small 模型）：每个 Token 被映射成一个 512 维的向量

所以 Embedding 层本质上是一个 `6400 x 512` 的查找表（矩阵）。给定一个 Token ID，就去这个矩阵里取对应那一行，得到一个 512 维的向量。

### 2.3 权重共享（Weight Tying）

MiniMind 使用了一个经典技巧——**Embedding 层和输出层共享权重**：

```python
# model/model_minimind.py:L434-435
self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
self.model.embed_tokens.weight = self.lm_head.weight
```

这意味着：
- **输入端**（Embedding）：Token ID -> 512 维向量（查表）
- **输出端**（LM Head）：512 维向量 -> 6400 维 logits（矩阵乘法）
- 两者使用**同一个** `6400 x 512` 的权重矩阵！

为什么要这样做？
1. **节省参数**：一个 `6400 x 512` 矩阵就是 327 万参数，共享后直接省了一半
2. **语义一致性**：输入和输出使用相同的"词向量空间"，模型更容易学习
3. **业界惯例**：GPT-2、LLaMA、Qwen 等都用了这个技巧

---

## 3. RMSNorm —— 轻量级归一化

### 3.1 为什么需要归一化？

深度神经网络的一个经典问题是：随着层数加深，中间的数值可能变得极大或极小，导致训练不稳定。归一化（Normalization）就是把数值"拉回"到一个合理的范围。

### 3.2 RMSNorm vs LayerNorm

传统 Transformer（如 BERT、GPT-2）使用 **LayerNorm**，它会计算均值和方差：

```
LayerNorm(x) = (x - mean) / sqrt(var + eps) * weight + bias
```

MiniMind 使用的是 **RMSNorm**（Root Mean Square Normalization），它更简单：

```
RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
```

区别在于：
- **不减均值**：RMSNorm 跳过了"去中心化"步骤
- **不加偏置**：只有 `weight`，没有 `bias`
- **更快**：计算量更小，在 GPU 上更高效

实验表明两者效果几乎相同，但 RMSNorm 速度更快，所以 LLaMA、Qwen 等现代模型都选择了 RMSNorm。

### 3.3 MiniMind 的实现

```python
# model/model_minimind.py:L96-106
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)
```

逐行解读：

1. **`self.weight = nn.Parameter(torch.ones(dim))`**：可学习的缩放参数，初始全为 1
2. **`x.pow(2).mean(-1, keepdim=True)`**：计算每个位置向量的均方值（RMS 的核心）
3. **`torch.rsqrt(...)`**：取倒数平方根，即 `1 / sqrt(mean(x²) + eps)`
4. **`x * torch.rsqrt(...)`**：将原始向量除以 RMS 值，完成归一化
5. **`self.weight * ...`**：乘以可学习的缩放参数
6. **`x.float()` 和 `.type_as(x)`**：先转 float32 做归一化（保证精度），再转回原始类型

`eps = 1e-5` 是一个很小的数，防止除以零。

### 3.4 RMSNorm 在模型中的位置

RMSNorm 在 MiniMind 中出现了三次（Per Block 两次 + 最终输出一次）：

```
Block 内:
  input_layernorm  → 注意力之前
  post_attention_layernorm → FFN 之前

输出:
  model.norm → 最后一个 Block 之后、LM Head 之前
```

这种"先归一化再计算"的方式叫 **Pre-Norm**，相比 Post-Norm（先计算再归一化），训练更稳定，是现代 LLM 的标配。

---

## 4. 自注意力机制（Self-Attention）

这是 Transformer 最核心的部分，也是 MiniMind 最精彩的代码段。

### 4.1 直觉理解：查字典

想象你在图书馆查资料。你有一个**问题**（Query），图书馆有很多书，每本书有一个**标题**（Key）和**内容**（Value）。查资料的过程是：

1. 拿你的**问题**（Q）和每本书的**标题**（K）做对比
2. 找出最相关的几本书（注意力权重）
3. 把这些书的**内容**（V）按相关度加权组合，得到答案

自注意力的"自"是指：Q、K、V 都来自**同一个**输入序列。也就是说，序列中的每个词都在"查看"其他所有词，决定该关注谁。

### 4.2 Q/K/V 投影

```python
# model/model_minimind.py:L159-162
self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
```

四个线性层的作用：
- **`q_proj`**：生成 Query（查询向量），投影到 `num_attention_heads * head_dim` 维
- **`k_proj`**：生成 Key（键向量），投影到 `num_key_value_heads * head_dim` 维
- **`v_proj`**：生成 Value（值向量），投影到 `num_key_value_heads * head_dim` 维
- **`o_proj`**：输出投影，把多头注意力的结果合并回 `hidden_size` 维

注意 Q 和 K/V 的维度可能不同——这就是 GQA 的核心。

### 4.3 多头注意力

为什么要"多头"？因为一个词在不同语境下关注的东西不同。比如"苹果"这个词：
- 一个头可能关注"是水果还是公司"（语义头）
- 另一个头可能关注"在句子中是主语还是宾语"（语法头）

多头注意力就是让模型同时从多个角度去"看"输入序列。

```python
# model/model_minimind.py:L176-179
xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)     # [B, S, 8, 64]
xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  # [B, S, 2, 64]
xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  # [B, S, 2, 64]
```

以 Small 模型为例（`hidden_size=512, num_attention_heads=8`）：
- `head_dim = 512 / 8 = 64`（每个头处理 64 维）
- Q 被拆成 8 个头，每个头 64 维
- K、V 被拆成 2 个头，每个头 64 维（GQA 的原因，下面详解）

### 4.4 GQA：分组查询注意力（Grouped-Query Attention）

这是 MiniMind 的一个重要设计。传统多头注意力中，Q、K、V 的头数相同（比如都是 8 头）。但 GQA 让 **K/V 的头数更少**：

```
MiniMind 配置:
  num_attention_heads = 8      ← Q 有 8 个头
  num_key_value_heads = 2      ← K/V 只有 2 个头
  n_rep = 8 / 2 = 4            ← 每个 KV 头被 4 个 Q 头共享
```

用一张图来理解：

```
Q 头:  Q0  Q1  Q2  Q3  Q4  Q5  Q6  Q7    （8 个头）
       │   │   │   │   │   │   │   │
K/V:  KV0 KV0 KV0 KV0 KV1 KV1 KV1 KV1   （2 个头，每个被复制 4 次）
```

也就是说，Q0、Q1、Q2、Q3 共享同一组 K0/V0，Q4、Q5、Q6、Q7 共享同一组 K1/V1。

这样做的好处：
- **节省显存**：KV 的参数量和缓存量减少到 1/4
- **推理更快**：KV Cache 更小，内存带宽压力更低
- **效果不减**：实验表明 GQA 的性能与标准 MHA 几乎相同

`repeat_kv` 函数负责把 2 个 KV 头"复制扩展"成 8 个，以便和 Q 做计算：

```python
# model/model_minimind.py:L140-147
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
         .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
         .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )
```

这个函数的巧妙之处在于：
1. `x[:, :, :, None, :]` — 在第 4 维插入一个新维度
2. `.expand(...)` — 沿新维度复制 `n_rep` 次（不实际分配内存）
3. `.reshape(...)` — 合并维度，从 `[B, S, 2, 4, 64]` 变成 `[B, S, 8, 64]`

### 4.5 注意力计算

经过 RoPE 位置编码（下一节详解）后，就是核心的注意力计算了：

```python
# model/model_minimind.py:L190-209
xq, xk, xv = (
    xq.transpose(1, 2),                         # [B, 8, S, 64]
    repeat_kv(xk, self.n_rep).transpose(1, 2),  # [B, 8, S, 64]
    repeat_kv(xv, self.n_rep).transpose(1, 2)   # [B, 8, S, 64]
)

if self.flash and ...:
    # Flash Attention（快速路径）
    output = F.scaled_dot_product_attention(xq, xk, xv, ...)
else:
    # 标准注意力（慢速路径）
    scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
    # 因果掩码：防止看到未来的 token
    scores[:, :, :, -seq_len:] += torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
        diagonal=1
    )
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    scores = self.attn_dropout(scores)
    output = scores @ xv
```

标准注意力的数学公式：

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

步骤分解：
1. **Q @ K^T**：计算每对 token 之间的"相似度分数"，shape 为 `[B, 8, S, S]`
2. **/ sqrt(d_k)**：除以 `sqrt(64) = 8`，防止分数过大导致 softmax 饱和
3. **因果掩码**：用上三角矩阵（值为 `-inf`）遮住"未来"的位置，确保每个 token 只能看到自己和之前的 token
4. **softmax**：将分数转换为概率分布（所有权重之和为 1）
5. **@ V**：按注意力权重对 Value 做加权求和

以一个简单例子说明因果掩码：

```
假设序列为 "我 爱 中 国"（4 个 token）

注意力矩阵（softmax 之前）:
        我    爱    中    国
  我  [ 0.8  -inf  -inf  -inf ]   ← "我" 只能看到自己
  爱  [ 0.3   0.9  -inf  -inf ]   ← "爱" 能看到"我"和自己
  中  [ 0.1   0.4   0.7  -inf ]   ← "中" 能看到前三个
  国  [ 0.2   0.5   0.3   0.6 ]   ← "国" 能看到所有
```

这就是**自回归**（auto-regressive）生成的核心：每个 token 只能基于之前的上下文来预测。

### 4.6 Flash Attention

MiniMind 支持 Flash Attention 加速：

```python
# model/model_minimind.py:L196-197
if self.flash and (seq_len > 1) and (past_key_value is None) and \
   (attention_mask is None or torch.all(attention_mask == 1)):
    output = F.scaled_dot_product_attention(
        xq, xk, xv,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=True
    )
```

Flash Attention 的核心思想是：不显式构造 `S x S` 的注意力矩阵，而是用**分块计算 + 在线 softmax**的方式，大幅减少显存占用。

使用条件（必须同时满足）：
- `self.flash = True`：配置中开启了 flash_attn
- `seq_len > 1`：序列长度大于 1（单 token 生成时没必要）
- `past_key_value is None`：没有使用 KV Cache（即训练阶段或第一次前向）
- `attention_mask` 全为 1 或为 None：没有 padding
- 不在 NPU 上运行（NPU 暂不支持）

不满足条件时，会退回到标准注意力计算。

### 4.7 KV Cache：推理加速的关键

在自回归生成时，模型每次只预测一个 token。如果每次都重新计算所有 token 的 K 和 V，那就太浪费了——因为之前的 K/V 没有变化。

KV Cache 的思路很简单：**把之前算过的 K/V 缓存起来，新 token 只算自己的 K/V，然后拼接上去**。

```python
# model/model_minimind.py:L184-188
# kv_cache 实现
if past_key_value is not None:
    xk = torch.cat([past_key_value[0], xk], dim=1)  # 拼接历史 K
    xv = torch.cat([past_key_value[1], xv], dim=1)  # 拼接历史 V
past_kv = (xk, xv) if use_cache else None
```

配合 `prepare_inputs_for_generation` 使用：

```python
# model/model_minimind.py:L437-440
def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
    if past_key_values and past_key_values[0] is not None:
        input_ids = input_ids[:, -1:]  # 只传最后一个 token！
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "past_key_values": past_key_values, "use_cache": True}
```

用一个时间线来理解 KV Cache 的工作过程：

```
第 1 步：输入 "我"
  Q = Q("我"),  K = K("我"),  V = V("我")
  cache = {K: [K_我], V: [V_我]}

第 2 步：输入 "爱"（只传这一个 token！）
  Q = Q("爱"),  K = concat([K_我, K_爱]),  V = concat([V_我, V_爱])
  cache = {K: [K_我, K_爱], V: [V_我, V_爱]}

第 3 步：输入 "中"
  Q = Q("中"),  K = concat([K_我, K_爱, K_中]),  V = concat([V_我, V_爱, V_中])
  cache = {K: [K_我, K_爱, K_中], V: [V_我, V_爱, V_中]}
```

这样每一步只需要为**1 个新 token** 计算 Q/K/V，然后从 cache 中取出之前的 K/V 拼接即可。生成速度大幅提升。

---

## 5. 位置编码 RoPE（旋转位置编码）

### 5.1 为什么需要位置编码？

注意力机制本身是**位置无关**的——"我爱你"和"你爱我"在注意力看来是一样的（只是三个 token 两两之间的相似度）。但语言显然是有顺序的！所以我们需要某种方式告诉模型"每个 token 在哪个位置"。

### 5.2 RoPE 的直觉

RoPE（Rotary Position Embedding）的核心思想非常优雅：**给每个位置一个独特的旋转角度**。

想象一个时钟：
- 位置 0 的指针指向 12 点
- 位置 1 的指针旋转了一个小角度
- 位置 2 旋转了两倍的角度
- ...

当两个 token 做注意力计算（Q 点乘 K）时，旋转角度的**差值**就自然编码了它们之间的**相对距离**。距离近的 token，角度差小，点乘值大；距离远的 token，角度差大，点乘值小。

这比绝对位置编码（给每个位置分配一个固定向量）更灵活，因为：
- 它编码的是**相对位置**而非绝对位置
- 天然支持**任意长度**的序列（只要角度能算出来）
- 点乘值随距离增大而**自然衰减**

### 5.3 预计算频率

```python
# model/model_minimind.py:L109-128
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    attn_factor = 1.0
    # ... YaRN 部分（后面详解）...
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin
```

分步理解：

1. **基础频率**：`freqs = 1 / (base^(2i/d))`，其中 `i` 是维度索引，`d` 是总维度
   - `rope_base = 1000000`（MiniMind 使用 100 万作为基数）
   - 低维度的频率高（变化快），高维度的频率低（变化慢）
   - 这样不同维度以不同的"速度"旋转，编码不同尺度的位置信息

2. **外积**：`torch.outer(t, freqs)` 计算"位置 x 频率"的矩阵
   - `t` 是位置索引 `[0, 1, 2, ..., max_len-1]`
   - 结果 shape 为 `[max_len, dim//2]`，每个元素是 `position * frequency`

3. **三角函数**：对外积结果取 cos 和 sin，并拼接成 `[max_len, dim]`

4. **注册为 buffer**：这些频率在初始化时计算好，推理时直接查表

```python
# model/model_minimind.py:L386-390
freqs_cos, freqs_sin = precompute_freqs_cis(
    dim=config.hidden_size // config.num_attention_heads,  # head_dim = 64
    end=config.max_position_embeddings,                     # 32768
    rope_base=config.rope_theta,                            # 1000000
    rope_scaling=config.rope_scaling
)
self.register_buffer("freqs_cos", freqs_cos, persistent=False)
self.register_buffer("freqs_sin", freqs_sin, persistent=False)
```

### 5.4 应用旋转

```python
# model/model_minimind.py:L131-137
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed
```

`rotate_half` 的作用是将向量的前后两半交换并取负：

```
原始:        [a, b, c, d, e, f, g, h]
rotate_half: [-e, -f, -g, -h, a, b, c, d]
```

然后按照旋转公式：

```
x_rotated = x * cos + rotate_half(x) * sin
```

这等价于把相邻的两个维度看作一个二维平面，在该平面上做旋转。每个位置旋转不同的角度，从而编码了位置信息。

**关键点**：RoPE 只应用于 Q 和 K，不应用于 V。因为位置信息需要影响"谁注意谁"（Q 和 K 的点乘），但不需要影响"注意的内容"（V）。

### 5.5 YaRN 长度外推

MiniMind 的训练序列长度是 2048，但通过 YaRN 技术可以在推理时扩展到 32768（16 倍）！

```python
# model/model_minimind.py:L112-122
if rope_scaling is not None:
    orig_max, factor, beta_fast, beta_slow, attn_factor = (
        rope_scaling.get("original_max_position_embeddings", 2048),
        rope_scaling.get("factor", 16),
        rope_scaling.get("beta_fast", 32.0),
        rope_scaling.get("beta_slow", 1.0),
        rope_scaling.get("attention_factor", 1.0)
    )
    if end / orig_max > 1.0:
        # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
        low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
        ramp = torch.clamp(
            (torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1
        )
        freqs = freqs * (1 - ramp + ramp / factor)
```

YaRN 的核心思路：

1. **问题**：如果直接在更长的序列上使用原始 RoPE，高频维度的旋转角会"溢出"，导致性能崩溃
2. **解决方案**：根据频率高低，使用不同程度的缩放
   - **高频维度**（`ramp ≈ 0`）：保持不变，因为它们编码的是"近距离"信息，不需要外推
   - **低频维度**（`ramp ≈ 1`）：频率除以 `factor=16`，相当于"拉伸"时间轴
   - **中间维度**：在两者之间线性插值
3. **`beta_fast=32, beta_slow=1`**：控制高频和低频的边界

配置中通过 `inference_rope_scaling` 标志控制是否启用：

```python
# model/model_minimind.py:L58-65
self.rope_scaling = {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 16,
    "original_max_position_embeddings": 2048,
    "attention_factor": 1.0,
    "type": "yarn"
} if self.inference_rope_scaling else None
```

这意味着训练时用标准 RoPE（2048 长度），推理时可选开启 YaRN 外推到 32768。

---

## 6. 前馈网络 SwiGLU

### 6.1 前馈网络的角色

如果说注意力层是"信息交换"——让每个 token 看到其他 token；那前馈网络就是"信息加工"——在每个 token 的位置上独立地进行非线性变换。

### 6.2 门控机制的直觉

传统的前馈网络很简单：`FFN(x) = W2(ReLU(W1(x)))`。但 MiniMind 使用了更先进的 **SwiGLU**（Swish-Gated Linear Unit）。

打个比方：想象一个工厂有两条流水线：
- **流水线 A**（`gate_proj`）：评估"这个信息有多重要"，输出一个 0-1 之间的"门控信号"
- **流水线 B**（`up_proj`）：处理实际信息
- 两条线的输出相乘：重要的信息被放大，不重要的被抑制
- 最后通过 `down_proj` 压缩回原始维度

### 6.3 MiniMind 的实现

```python
# model/model_minimind.py:L216-229
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))
```

数据流图：

```
                    ┌─── gate_proj ──→ SiLU ───┐
                    │    (512→1365)    (激活)    │
input (512) ────────┤                           × ──→ down_proj ──→ output (512)
                    │                           │     (1365→512)
                    └─── up_proj ──────────────┘
                         (512→1365)
```

三个线性层的维度（以 Small 模型为例）：

- `gate_proj`: `512 → 1365`
- `up_proj`: `512 → 1365`
- `down_proj`: `1365 → 512`

### 6.4 intermediate_size 的计算

```python
intermediate_size = int(config.hidden_size * 8 / 3)
config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
```

- `512 * 8 / 3 = 1365.33`
- 向上对齐到 64 的倍数：`64 * ceil(1365.33 / 64) = 64 * 22 = 1408`

等等，让我们验算一下：`(1365 + 64 - 1) // 64 = 1428 // 64 = 22`，所以 `64 * 22 = 1408`。

对齐到 64 的倍数是为了**GPU 计算效率**——GPU 的矩阵运算在维度对齐到 64（或 128）时最高效。

### 6.5 为什么是 8/3？

传统 FFN 使用 `4 * hidden_size` 作为中间维度。SwiGLU 有 3 个矩阵（而非 2 个），为了保持参数量大致不变，使用 `8/3 * hidden_size ≈ 2.67 * hidden_size`。计算如下：

```
传统 FFN 参数量:  2 × hidden_size × 4 × hidden_size = 8 × hidden_size²
SwiGLU 参数量:    3 × hidden_size × (8/3) × hidden_size = 8 × hidden_size²
```

两者参数量相同，但 SwiGLU 的效果更好。

---

## 7. MoE 混合专家系统

### 7.1 核心思想

如果说普通 FFN 是"一个全科医生"，那 MoE（Mixture of Experts）就是"一个医院"——有多个"专家"（每个专家是一个独立的 FFN），由一个"分诊台"（路由器/门控网络）决定每个 token 应该去找哪个专家。

MoE 的优势：**增大模型容量（参数量），但不增加计算量**。因为每个 token 只激活部分专家，而非全部。

### 7.2 MiniMind 的 MoE 配置

```python
# model/model_minimind.py:L32-38 (MiniMindConfig)
use_moe: bool = False,
num_experts_per_tok: int = 2,      # 每个 token 选择 2 个专家
n_routed_experts: int = 4,          # 共 4 个路由专家
n_shared_experts: int = 1,          # 1 个共享专家
scoring_func: str = 'softmax',      # 评分函数
aux_loss_alpha: float = 0.01,       # 负载均衡损失系数
```

整体架构图：

```
                        ┌─── Expert 0 (FFN) ──┐
                        │                      │
input ──→ MoEGate ──────┤─── Expert 1 (FFN) ──┤──→ 加权求和 ──→ + ──→ output
          (路由器)       │                      │               │
          选 top-2       │─── Expert 2 (FFN)   │               │
                        │                      │               │
                        └─── Expert 3 (FFN) ──┘               │
                                                               │
                        共享专家 (FFN) ─────────────────────────┘
```

每个 token 经过路由器评分后，只选择得分最高的 2 个专家进行计算，然后加上共享专家的输出。

### 7.3 MoEGate：路由器

路由器是 MoE 的"大脑"，它决定每个 token 该去找哪个专家。

```python
# model/model_minimind.py:L232-285
class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok        # 2
        self.n_routed_experts = config.n_routed_experts  # 4
        self.gating_dim = config.hidden_size             # 512
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        # weight shape: [4, 512]
```

前向过程：

```python
# model/model_minimind.py:L251-260
def forward(self, hidden_states):
    bsz, seq_len, h = hidden_states.shape
    hidden_states = hidden_states.view(-1, h)              # [B*S, 512]
    logits = F.linear(hidden_states, self.weight, None)    # [B*S, 4]  每个 token 对 4 个专家的评分
    scores = logits.softmax(dim=-1)                        # [B*S, 4]  softmax 归一化
    topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
    # topk_weight: [B*S, 2]  选中的 2 个专家的权重
    # topk_idx:    [B*S, 2]  选中的 2 个专家的编号
```

路由过程简单来说：
1. 每个 token 的 512 维向量与路由权重矩阵相乘，得到 4 个分数
2. softmax 归一化为概率
3. 选择概率最高的 2 个专家

如果选择了多个专家（top_k > 1），还会**归一化权重**：

```python
# model/model_minimind.py:L262-264
if self.top_k > 1 and self.norm_topk_prob:
    denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
    topk_weight = topk_weight / denominator
```

确保选中的 2 个专家的权重之和为 1。

### 7.4 负载均衡辅助损失

MoE 有一个经典问题：**路由崩塌**——如果放任不管，路由器可能会把所有 token 都发给同一个专家（"马太效应"），其他专家被"饿死"。

为了防止这种情况，MiniMind 引入了**负载均衡辅助损失**（Auxiliary Load Balancing Loss）：

```python
# model/model_minimind.py:L266-284
if self.training and self.alpha > 0.0:
    scores_for_aux = scores
    aux_topk = self.top_k
    topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
    if self.seq_aux:
        # 序列级别的辅助损失
        scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
        ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
        ce.scatter_add_(1, topk_idx_for_aux_loss,
                        torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)
                       ).div_(seq_len * aux_topk / self.n_routed_experts)
        aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
```

直觉理解：
- `ce`（count of experts）：统计每个专家被分配了多少 token
- `scores.mean(dim=1)`：每个专家的平均评分
- 辅助损失 = 两者的点积 * `alpha`

当所有专家被均匀分配时，这个损失最小。当某个专家过于"热门"时，损失增大，迫使路由器更均匀地分配 token。

`aux_loss_alpha = 0.01` 意味着辅助损失只占总损失的 1%，不会主导训练方向。

### 7.5 MOEFeedForward：专家计算

```python
# model/model_minimind.py:L288-326
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.experts = nn.ModuleList([
            FeedForward(config) for _ in range(config.n_routed_experts)  # 4 个专家
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config) for _ in range(config.n_shared_experts)  # 1 个共享专家
            ])
```

训练时的前向过程：

```python
# model/model_minimind.py:L303-319
def forward(self, x):
    identity = x
    orig_shape = x.shape
    bsz, seq_len, _ = x.shape
    topk_idx, topk_weight, aux_loss = self.gate(x)
    x = x.view(-1, x.shape[-1])
    flat_topk_idx = topk_idx.view(-1)
    if self.training:
        x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
        y = torch.empty_like(x, dtype=x.dtype)
        for i, expert in enumerate(self.experts):
            expert_out = expert(x[flat_topk_idx == i])
            if expert_out.shape[0] > 0:
                y[flat_topk_idx == i] = expert_out.to(y.dtype)
            else:
                y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
        y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
        y = y.view(*orig_shape)
```

训练时的步骤：
1. 路由器为每个 token 选择 2 个专家及其权重
2. 将每个 token 复制 2 份（因为要送给 2 个专家）
3. 遍历每个专家，只处理分配给它的 token
4. 将 2 个专家的输出按权重加权求和
5. 加上共享专家的输出

注意第 317 行的一个细节：`0 * sum(p.sum() for p in expert.parameters())`。当某个专家没有被分配到任何 token 时，这行代码确保该专家的参数仍然参与计算图，避免 DDP（分布式训练）同步时出错。

### 7.6 推理时的优化

推理时使用更高效的 `moe_infer` 方法：

```python
# model/model_minimind.py:L328-349
@torch.no_grad()
def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
    expert_cache = torch.zeros_like(x)
    idxs = flat_expert_indices.argsort()
    tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
    token_idxs = idxs // self.config.num_experts_per_tok
    for i, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
        if start_idx == end_idx:
            continue
        expert = self.experts[i]
        exp_token_idx = token_idxs[start_idx:end_idx]
        expert_tokens = x[exp_token_idx]
        expert_out = expert(expert_tokens).to(expert_cache.dtype)
        expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
        expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)
    return expert_cache
```

推理优化的关键：**先排序再分组**。
- 先按专家编号排序所有 token，这样同一个专家的 token 在内存中是连续的
- 然后每个专家批量处理属于自己的所有 token
- 最后用 `scatter_add_` 将结果累加回原始位置

这比训练时的逐个判断 `flat_topk_idx == i` 高效得多。

### 7.7 共享专家

```python
# model/model_minimind.py:L322-324
if self.config.n_shared_experts > 0:
    for expert in self.shared_experts:
        y = y + expert(identity)
```

共享专家对**所有 token**都生效，不受路由器控制。它的作用是捕获**所有 token 都需要的通用知识**（比如语法规则），而路由专家则处理**特定类型的知识**（比如不同领域的专业知识）。

---

## 8. MiniMindBlock — 一个完整的 Transformer 层

理解了各个组件后，让我们看它们如何组合成一个完整的 Transformer 层：

```python
# model/model_minimind.py:L352-373
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings,
                past_key_value=None, use_cache=False, attention_mask=None):
        # ① 保存输入用于残差连接
        residual = hidden_states

        # ② Pre-Norm → 注意力
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value, use_cache, attention_mask
        )

        # ③ 第一个残差连接
        hidden_states += residual

        # ④ Pre-Norm → FFN/MoE → 第二个残差连接
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )

        return hidden_states, present_key_value
```

数据流：

```
input ──────────────────────────────────────── (+) ─────────────────────── (+) ──→ output
   │                                            ↑                          ↑
   └→ RMSNorm → Attention (含 RoPE, GQA) ──────┘   └→ RMSNorm → FFN/MoE ─┘
      (input_layernorm)                                (post_attention_layernorm)
```

**残差连接（Residual Connection）** 是深度网络的生命线。它让梯度能够直接"穿越"多层网络回传，避免梯度消失问题。没有残差连接，8 层或 16 层的网络几乎无法训练。

注意 `mlp` 的选择：
```python
self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
```
当 `use_moe=True` 时使用 MoE，否则使用普通 FFN。两者的接口完全相同（输入输出维度一致），可以无缝切换。

---

## 9. 完整前向传播路径

### 9.1 MiniMindModel：骨干网络

```python
# model/model_minimind.py:L376-424
class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            MiniMindBlock(l, config) for l in range(self.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 预计算 RoPE 频率
        freqs_cos, freqs_sin = precompute_freqs_cis(...)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
```

前向传播：

```python
# model/model_minimind.py:L392-424
def forward(self, input_ids, attention_mask=None,
            past_key_values=None, use_cache=False, **kwargs):
    batch_size, seq_length = input_ids.shape

    # ① 处理 KV Cache 的起始位置
    past_key_values = past_key_values or [None] * len(self.layers)
    start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

    # ② Embedding + Dropout
    hidden_states = self.dropout(self.embed_tokens(input_ids))

    # ③ 获取位置编码（根据 start_pos 切片）
    position_embeddings = (
        self.freqs_cos[start_pos:start_pos + seq_length],
        self.freqs_sin[start_pos:start_pos + seq_length]
    )

    # ④ 逐层通过 Transformer Block
    presents = []
    for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
        hidden_states, present = layer(
            hidden_states, position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask
        )
        presents.append(present)

    # ⑤ 最终归一化
    hidden_states = self.norm(hidden_states)

    # ⑥ 收集 MoE 辅助损失
    aux_loss = sum(
        [l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
        hidden_states.new_zeros(1).squeeze()
    )
    return hidden_states, presents, aux_loss
```

注意第 ③ 步的 `start_pos` 处理：当使用 KV Cache 时，`start_pos` 等于已缓存的序列长度。这样新 token 获取的是正确的位置编码——比如已生成了 10 个 token，新 token 的位置编码是第 11 个位置的。

### 9.2 MiniMindForCausalLM：完整的因果语言模型

```python
# model/model_minimind.py:L427-468
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight  # 权重共享

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                past_key_values=None, use_cache=False, logits_to_keep=0, **args):
        # ① 通过骨干网络
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            past_key_values=past_key_values, use_cache=use_cache, **args
        )

        # ② LM Head: hidden_states → logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # ③ 计算损失（如果提供了 labels）
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        # ④ 返回结果
        output = CausalLMOutputWithPast(
            loss=loss, logits=logits,
            past_key_values=past_key_values, hidden_states=hidden_states
        )
        output.aux_loss = aux_loss
        return output
```

几个关键点：

**`logits_to_keep`**：在某些训练场景（如 GRPO）中，不需要所有位置的 logits，只需要最后几个。这个参数可以减少 LM Head 的计算量。

**损失计算的"移位"**：

```
logits:  [位置0, 位置1, 位置2, 位置3]  → shift: [位置0, 位置1, 位置2]
labels:  [位置0, 位置1, 位置2, 位置3]  → shift: [位置1, 位置2, 位置3]
```

因为语言模型的任务是"预测下一个 token"，所以位置 0 的输出应该预测位置 1 的 label，位置 1 预测位置 2，以此类推。`ignore_index=-100` 用于忽略 padding 位置。

**继承关系**：
- `PreTrainedModel`：HuggingFace 的基类，提供 `save_pretrained`、`from_pretrained` 等方法
- `GenerationMixin`：提供 `generate` 方法，支持 greedy、beam search、sampling 等生成策略

### 9.3 完整数据流总结

把所有东西串起来，一次完整的前向传播是这样的：

```
"你好世界"
    ↓ 分词器
[Token IDs: 34, 567, 89, 12]
    ↓ Embedding (共享权重)
[4 个 512 维向量]
    ↓ Dropout
[4 个 512 维向量]
    ↓ Block 0
    │   ├ RMSNorm → Attention (Q/K/V投影 → RoPE → GQA → Softmax → 加权求和) → 残差
    │   └ RMSNorm → FFN (gate_proj → SiLU → * up_proj → down_proj) → 残差
    ↓ Block 1
    │   └ ... (同上)
    ↓ ... (重复 N 次)
    ↓ Block N-1
    │   └ ...
    ↓ RMSNorm (最终归一化)
[4 个 512 维向量]
    ↓ LM Head (共享权重, 512→6400)
[4 个 6400 维 logits]
    ↓ 取最后一个位置的 logits
[6400 维向量]
    ↓ softmax + 采样
下一个 Token ID
```

---

## 10. 模型尺寸对照表

MiniMind 提供了三种模型配置，从"可以在笔记本上训练"到"需要好一点的显卡"：

| 配置 | 参数量 | hidden_size | num_hidden_layers | num_attention_heads | num_key_value_heads | head_dim | intermediate_size |
|------|--------|-------------|-------------------|---------------------|---------------------|----------|-------------------|
| **Small** | 26M | 512 | 8 | 8 | 2 | 64 | 1408 |
| **Base** | 104M | 768 | 16 | 8 | 2 | 96 | 2048 |
| **MoE** | 145M | 768 + MoE | 16 | 8 | 2 | 96 | 2048 |

参数量的粗略计算（以 Small 为例）：

```
Embedding:        6400 × 512                          = 3,276,800  (与 LM Head 共享)
每个 Block:
  Attention:      512×512 + 512×128 + 512×128 + 512×512 = 655,360
  RMSNorm × 2:   512 × 2                              = 1,024
  FFN:            512×1408 + 1408×512 + 512×1408       = 2,162,688
  Block 合计:     ≈ 2,819,072

8 个 Block:       2,819,072 × 8                        = 22,552,576
最终 RMSNorm:     512                                  = 512
LM Head:          (共享，不额外算)

总计:             ≈ 25,829,888 ≈ 26M
```

MoE 模型在 Base 的基础上，将每层的 FFN 替换为 MoE（4 个路由专家 + 1 个共享专家 = 5 个 FFN），所以参数量从 104M 增加到 145M。但由于每个 token 只激活 2 个路由专家 + 1 个共享专家（共 3 个），实际计算量仅增加 50%，远小于参数量的增加。

---

## 总结

这一章我们从头到尾拆解了 MiniMind 的模型架构，让我们回顾核心组件：

| 组件 | 作用 | 关键技术 |
|------|------|----------|
| Embedding | Token ID → 向量 | 权重共享 (Weight Tying) |
| RMSNorm | 归一化，稳定训练 | Pre-Norm，比 LayerNorm 更快 |
| Attention | 信息交换，让 token 互相"看到" | GQA、KV Cache、Flash Attention |
| RoPE | 位置编码 | 旋转位置编码、YaRN 长度外推 |
| FFN (SwiGLU) | 信息加工，非线性变换 | 门控机制、8/3 中间维度 |
| MoE | 增大容量不增加计算 | Top-2 路由、共享专家、负载均衡 |
| 残差连接 | 保证梯度流通 | 深度网络的必备组件 |

所有这些技术的代码实现不到 470 行，这正是 MiniMind 项目的魅力——**用最少的代码，实现最完整的现代 LLM 架构**。

下一章我们将学习如何用这个架构进行预训练——让模型从"一无所知"变成"会说话"。

> [上一章：分词器](02-tokenizer.md) | [下一章：预训练](04-pretrain.md)
