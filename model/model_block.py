# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                        Transformer Block（单层）
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from torch import nn

from model.model_config import MiniMindConfig
from model.model_norm import RMSNorm
from model.model_attention import Attention
from model.model_ffn import FeedForward, MOEFeedForward


# ==================== Transformer Block ====================
# 一个完整的 Transformer 层 = 注意力 + FFN，中间用残差连接和归一化。
#
# 数据流向（Pre-Norm 结构）：
#   输入 → RMSNorm → 注意力 → 残差相加 → RMSNorm → FFN → 残差相加 → 输出
#
# 残差连接：把输入直接加到输出上，让梯度更容易流过深层网络。
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)              # 注意力层

        self.layer_id = layer_id                        # 第几层（从 0 开始）
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)         # 注意力前的归一化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # FFN 前的归一化
        # 根据配置选择普通 FFN 或 MoE FFN
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # 注意力 + 残差
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual  # 残差连接

        # FFN + 残差
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value
