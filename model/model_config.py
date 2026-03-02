# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    """
    MiniMind 模型的配置类，控制模型的所有超参数。

    简单理解：这个类就是模型的"设计图纸"，决定了模型有多大、多少层、多少注意力头等。
    改这里的参数就能造出不同大小的模型。
    """
    model_type = "minimind"

    def __init__(
            self,
            # ===== 基础参数 =====
            dropout: float = 0.0,              # 随机丢弃比例，防止过拟合（0 表示不丢弃）
            bos_token_id: int = 1,             # 句子开头的特殊 token ID
            eos_token_id: int = 2,             # 句子结尾的特殊 token ID
            hidden_act: str = 'silu',          # 激活函数类型（silu 是目前主流 LLM 的选择）
            hidden_size: int = 512,            # 隐藏层维度，模型的"宽度"，越大模型越强但越慢
            intermediate_size: int = None,     # FFN 中间层维度，None 时自动算（约 hidden_size 的 2.7 倍）
            max_position_embeddings: int = 32768,  # 模型能处理的最大序列长度（token 数）
            num_attention_heads: int = 8,      # 注意力头数，多头注意力让模型同时关注不同位置
            num_hidden_layers: int = 8,        # Transformer 层数，模型的"深度"，越多越强但越慢
            num_key_value_heads: int = 2,      # KV 头数（GQA），比注意力头数少可以省显存
            vocab_size: int = 6400,            # 词表大小，即模型认识多少个不同的 token
            rms_norm_eps: float = 1e-05,       # RMSNorm 的极小值，防止除以零
            rope_theta: int = 1000000.0,       # RoPE 位置编码的基频，影响模型处理长文本的能力
            inference_rope_scaling: bool = False,  # 推理时是否启用 RoPE 外推（处理超长文本）
            flash_attn: bool = True,           # 是否用 Flash Attention 加速（更快更省显存）

            # ===== MoE（混合专家）参数 =====
            # MoE 核心思想：不是所有参数都参与每次计算，而是由"门控"选几个"专家"来处理
            # use_moe=False 时下面这些参数都不生效
            use_moe: bool = False,             # 是否启用 MoE（False=普通模型，True=MoE 模型）
            num_experts_per_tok: int = 2,      # 每个 token 选几个专家来处理（越多越准但越慢）
            n_routed_experts: int = 4,         # 总共有几个可选专家
            n_shared_experts: int = 1,         # 共享专家数（所有 token 都会经过的专家）
            scoring_func: str = 'softmax',     # 门控打分函数（决定选哪些专家）
            aux_loss_alpha: float = 0.01,      # 辅助损失权重，鼓励各专家被均匀使用
            seq_aux: bool = True,              # 是否在序列级别算辅助损失
            norm_topk_prob: bool = True,       # 是否对选中专家的权重做归一化
            **kwargs
    ):
        super().__init__(**kwargs)
        # ===== 基础参数赋值 =====
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # YaRN RoPE 外推配置：让模型在推理时处理比训练时更长的文本
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,                          # 外推倍数
            "original_max_position_embeddings": 2048,  # 原始训练长度
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn

        # ===== MoE 参数赋值 =====
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
