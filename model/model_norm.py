# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             归一化层
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import torch
from torch import nn


# ==================== RMSNorm ====================
# 归一化层，作用类似 LayerNorm，但更简单更快。
# 让每一层的输入保持稳定的数值范围，防止训练过程中数值爆炸或消失。
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps                                  # 防止除以零的极小值
        self.weight = nn.Parameter(torch.ones(dim))     # 可学习的缩放参数

    def _norm(self, x):
        # 计算 RMS（均方根）并归一化：x / sqrt(mean(x^2))
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 先转 float32 算归一化（精度更高），再转回原始类型，最后乘以可学习权重
        return self.weight * self._norm(x.float()).type_as(x)
