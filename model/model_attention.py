# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                       RoPE 位置编码 + 注意力层
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple

from model.model_config import MiniMindConfig


# ==================== RoPE 位置编码 ====================
# RoPE（旋转位置编码）：让模型知道每个 token 在句子中的位置。
# 通过旋转向量的方式编码位置信息，比传统的位置编码更好用，支持外推到更长序列。
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    """
    预计算 RoPE 需要的 cos 和 sin 值。

    参数:
        dim: 每个注意力头的维度
        end: 最大序列长度
        rope_base: 基础频率（越大，位置编码变化越慢，适合长文本）
        rope_scaling: YaRN 外推配置（推理时处理超长文本）
    返回:
        freqs_cos, freqs_sin: 预计算的余弦和正弦值，形状 [end, dim]
    """
    # 计算每个维度对应的频率（低维变化快，高维变化慢）
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0

    # YaRN 外推：让模型能处理比训练时更长的文本
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            # YaRN 公式：对不同维度的频率做不同程度的缩放
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            # ramp: 线性插值系数，低维不缩放，高维按 factor 缩放
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    # 生成位置 × 频率的矩阵，然后算 cos 和 sin
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # [end, dim//2]
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor  # [end, dim]
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor  # [end, dim]
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    把位置信息"旋转"到 Q 和 K 向量中。

    简单理解：给每个 token 的 Q/K 乘上一个跟位置有关的旋转矩阵，
    这样模型算注意力时就能自动感知 token 之间的相对距离。
    """
    def rotate_half(x):
        # 把向量的前半和后半交换并取负，构造旋转操作
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    GQA（分组查询注意力）的辅助函数：把少量的 KV 头复制扩展到和 Q 头一样多。

    比如 Q 有 8 个头，KV 只有 2 个头，这里就把每个 KV 头复制 4 次。
    这样做的好处是省显存（KV 少），但不影响注意力计算。
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x  # Q 和 KV 头数一样，不需要复制
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


# ==================== 注意力层 ====================
# 注意力是 Transformer 的核心：让每个 token 能"看到"其他 token 并获取相关信息。
# 自回归模型（GPT 类）用因果注意力，即每个 token 只能看到它前面的 token。
class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        # GQA: Q 头多，KV 头少，省显存
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads       # Q 的头数
        self.n_local_kv_heads = self.num_key_value_heads    # KV 的头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 每个 KV 头要复制几次
        self.head_dim = args.hidden_size // args.num_attention_heads  # 每个头的维度

        # 四个线性变换层：Q/K/V 投影 + 输出投影
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)    # 注意力权重的 dropout
        self.resid_dropout = nn.Dropout(args.dropout)   # 输出的 dropout
        self.dropout = args.dropout
        # 检测是否可以用 Flash Attention（PyTorch >= 2.0 自带，更快更省显存）
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

    def forward(self,
                x: torch.Tensor,                       # 输入，形状 [batch, seq_len, hidden_size]
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # RoPE 的 cos 和 sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # 推理时的 KV 缓存
                use_cache=False,                        # 是否缓存 KV（推理时用）
                attention_mask: Optional[torch.Tensor] = None):  # 注意力掩码
        bsz, seq_len, _ = x.shape

        # 第一步：把输入分别投影成 Q、K、V
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 拆成多个头：[batch, seq_len, num_heads, head_dim]
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 第二步：给 Q 和 K 加上旋转位置编码（RoPE）
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # 第三步：KV 缓存（推理加速用）
        # 推理时每次只算一个新 token，把之前的 KV 拼接上，避免重复计算
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # 转置成 [batch, num_heads, seq_len, head_dim]，方便做矩阵乘法
        # GQA：把 KV 头复制到和 Q 头一样多
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        # 第四步：计算注意力
        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            # 快速路径：用 Flash Attention，更快更省显存
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # 慢速路径：手动算注意力（推理 or 有特殊 mask 时）
            # 计算 Q × K^T / sqrt(head_dim)，得到注意力分数
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 加上因果掩码：上三角设为 -inf，让每个 token 只能看到前面的 token
            scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)

            # 如果有额外的 attention_mask（比如 pad token 的掩码）
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            # softmax 归一化，得到注意力权重（每行和为 1）
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            # 用注意力权重对 V 加权求和，得到输出
            output = scores @ xv

        # 第五步：把多头的结果拼回去，通过输出投影
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
