# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                        前馈网络（FFN）+ MoE
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

from model.model_config import MiniMindConfig


# ==================== 前馈网络（FFN） ====================
# Transformer 的另一个核心组件，在注意力之后做非线性变换。
# 可以理解为：注意力负责"看哪些信息"，FFN 负责"怎么处理这些信息"。
# 结构：先升维（gate_proj + up_proj），中间用激活函数，再降维（down_proj）。
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # 如果没指定中间层维度，按 hidden_size 的 8/3 倍算，并对齐到 64 的倍数
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)  # 门控投影
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)  # 降维投影
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)    # 升维投影
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]  # 激活函数（默认 SiLU）

    def forward(self, x):
        # SwiGLU 结构：act(gate(x)) * up(x)，然后降维
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


# ==================== MoE 门控 ====================
# MoE 的"调度员"：决定每个 token 应该由哪几个专家来处理。
# 原理：用一个线性层给每个专家打分，选分数最高的 top_k 个专家。
class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok       # 每个 token 选几个专家
        self.n_routed_experts = config.n_routed_experts  # 专家总数

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha  # 辅助损失权重
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        # 门控权重矩阵：[专家数, hidden_size]，用来给每个专家打分
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # Kaiming 初始化

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        # 展平成 [batch*seq_len, hidden_size]
        hidden_states = hidden_states.view(-1, h)

        # 计算每个 token 对每个专家的分数
        logits = F.linear(hidden_states, self.weight, None)  # [batch*seq_len, 专家数]
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)  # 归一化成概率
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # 选出分数最高的 top_k 个专家
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 如果选了多个专家，对权重做归一化（权重和为 1）
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # 计算辅助损失（只在训练时）
        # 目的：鼓励所有专家被均匀使用，防止某些专家被"冷落"
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                # 序列级辅助损失：在每个样本内统计专家使用频率
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # token 级辅助损失：在整个 batch 内统计
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()  # 推理时辅助损失为 0

        return topk_idx, topk_weight, aux_loss


# ==================== MoE 前馈网络 ====================
# 多个 FFN "专家" + 门控 + 共享专家。
# 每个 token 不会经过所有专家，只经过门控选出的 top_k 个，所以计算量不会随专家数线性增长。
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 创建多个专家，每个专家就是一个普通的 FFN
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)  # 门控网络，决定用哪些专家
        # 共享专家：所有 token 都会经过，提供基础能力
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x          # 保存原始输入（给共享专家用）
        orig_shape = x.shape   # [batch, seq_len, hidden_size]
        bsz, seq_len, _ = x.shape

        # 第一步：门控选出每个 token 该用哪些专家
        topk_idx, topk_weight, aux_loss = self.gate(x)

        # 第二步：把 token 分配给各个专家处理
        x = x.view(-1, x.shape[-1])              # 展平成 [batch*seq_len, hidden_size]
        flat_topk_idx = topk_idx.view(-1)         # 展平专家索引

        if self.training:
            # 训练时：每个 token 复制 top_k 份，分别送给选中的专家
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)
            for i, expert in enumerate(self.experts):
                # 找出分配给第 i 个专家的 token，送进去计算
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else:
                    # 没有 token 分配给这个专家，但仍需让梯度流过（防止参数不更新）
                    y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
            # 按门控权重加权求和：多个专家的输出 × 各自权重，合并成最终结果
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 推理时：用更高效的批量推理方式
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        # 第三步：加上共享专家的输出
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)

        self.aux_loss = aux_loss  # 保存辅助损失，训练时会加到总 loss 里
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        推理时的高效 MoE 计算：把属于同一个专家的 token 凑在一起批量处理。

        比训练时的逐专家循环更高效，因为避免了 repeat_interleave 的内存开销。
        """
        expert_cache = torch.zeros_like(x)
        # 按专家索引排序，把同一专家的 token 放在一起
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok

        # 遍历每个专家，批量处理分配给它的所有 token
        # 例如 tokens_per_expert = [6, 15, 20, 26] 表示：
        #   专家 0 处理前 6 个，专家 1 处理第 7~15 个，以此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue  # 这个专家没有被分配到任何 token
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 乘以门控权重
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 累加到结果中（同一个 token 可能被多个专家处理）
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache
