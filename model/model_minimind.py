# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                      MiniMindModel + MiniMindForCausalLM
#
# 最终组装：把 N 层 Block 堆起来，加上 lm_head，构成完整的因果语言模型
#
# 文件结构：
#   model_config.py    → MiniMindConfig（模型配置/设计图纸）
#   model_norm.py      → RMSNorm（归一化层）
#   model_attention.py → RoPE + Attention（位置编码 + 注意力层）
#   model_ffn.py       → FeedForward + MoE（前馈网络 + 混合专家）
#   model_block.py     → MiniMindBlock（单层 Transformer）
#   model_minimind.py  → MiniMindModel + MiniMindForCausalLM（本文件，最终模型）
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from model.model_config import MiniMindConfig
from model.model_norm import RMSNorm
from model.model_attention import precompute_freqs_cis
from model.model_block import MiniMindBlock
from model.model_ffn import MOEFeedForward


# ==================== MiniMind 主模型（Transformer 主干） ====================
# 把所有 Transformer Block 堆叠起来，组成完整的语言模型主干。
#
# 数据流向：
#   token ID → Embedding（查词表变成向量） → N 层 Transformer Block → RMSNorm → 输出向量
class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        # 词嵌入层：把 token ID 转成 hidden_size 维的向量
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        # 堆叠 N 层 Transformer Block
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        # 最后的归一化层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算 RoPE 位置编码（只算一次，后续直接查表）
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)  # 注册为 buffer，不参与训练
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,       # 输入 token ID，[batch, seq_len]
                attention_mask: Optional[torch.Tensor] = None,   # 注意力掩码
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,  # KV 缓存
                use_cache: bool = False,                         # 是否启用 KV 缓存
                **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        # 如果有 KV 缓存，说明是推理续写，起始位置要偏移
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # token ID → 向量
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 取出当前位置的 RoPE cos/sin
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        # 逐层通过 Transformer Block
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        # 最后一层归一化
        hidden_states = self.norm(hidden_states)

        # 收集所有 MoE 层的辅助损失
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss


# ==================== 因果语言模型（最终模型） ====================
# 在 MiniMindModel 的输出上加一个线性层（lm_head），把向量映射回词表大小，
# 用来预测下一个 token 是什么。
#
# 完整流程：
#   token ID → MiniMindModel → hidden_states → lm_head → logits（每个词的概率分布）
#
# 继承了 HuggingFace 的 PreTrainedModel 和 GenerationMixin，
# 可以直接用 model.generate() 来生成文本。
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)        # Transformer 主干
        # lm_head：把 hidden_size 维向量映射到 vocab_size 维（预测下一个词）
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 权重绑定：让 lm_head 和 embed_tokens 共享同一套权重（减少参数量）
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,       # 输入 token ID
                attention_mask: Optional[torch.Tensor] = None,   # 注意力掩码
                labels: Optional[torch.Tensor] = None,           # 训练标签（就是右移一位的 input_ids）
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,   # 只保留最后几个位置的 logits（省显存）
                **args):
        # 通过 Transformer 主干得到隐藏状态
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        # 只取需要的位置，映射到词表大小
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])  # [batch, seq_len, vocab_size]

        # 如果提供了 labels，计算交叉熵损失
        loss = None
        if labels is not None:
            # 预测下一个 token：用位置 i 的输出预测位置 i+1 的 token
            shift_logits = logits[..., :-1, :].contiguous()    # 去掉最后一个位置
            shift_labels = labels[..., 1:].contiguous()        # 去掉第一个位置
            # 交叉熵损失，ignore_index=-100 表示忽略 pad token
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        # 打包返回结果
        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        output.aux_loss = aux_loss  # MoE 的辅助损失
        return output
