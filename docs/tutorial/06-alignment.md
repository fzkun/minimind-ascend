# 第六章：对齐训练 — 让模型学会"做好吃的菜"

[上一章：监督微调](05-sft.md) | [下一章：进阶技术](07-advanced.md)

---

## 1. 为什么需要对齐？

经过 SFT（监督微调）之后，模型已经学会了"回答问题"这件事。但如果你多测试几轮就会发现，回答的质量参差不齐：有时候回答得不错，有时候却跑偏了、说了废话、甚至给出有害的内容。

这就像一个刚学会做菜的厨师——他知道怎么把菜做熟，但做出来的味道时好时坏。**对齐训练的目标，就是让模型从"会做菜"进化到"会做好吃的菜"。**

具体来说，对齐要让模型的回答：

- **更有用**：回答切题、信息丰富
- **更安全**：拒绝有害请求、避免偏见
- **更准确**：减少胡编乱造（幻觉）
- **更符合人类偏好**：语气自然、格式清晰

对齐训练的核心思想是：**用"偏好信号"告诉模型，哪种回答好、哪种回答差，让模型学会区分并倾向于生成好的回答。**

MiniMind 实现了四种对齐方法，从简单到复杂依次为：

```
DPO（离线偏好）──→ SPO（单样本RL）──→ GRPO（组内比较RL）──→ PPO（完整RL）
    最简单                                                      最复杂
```

下面我们逐一讲解。

---

## 2. DPO — 直接偏好优化

### 2.1 核心思想

DPO（Direct Preference Optimization）的思路非常直觉：

> 给模型看同一个问题的两个回答——一个好的（chosen）、一个差的（rejected），让模型学会偏向好的回答。

这就像老师批改作文：把范文和差作文放在一起，学生一对比就知道该怎么写了。

### 2.2 偏好数据格式

DPO 需要特殊的偏好数据。每条数据包含一个问题的两个回答版本：

```json
{
  "chosen": [
    {"role": "user", "content": "什么是人工智能？"},
    {"role": "assistant", "content": "人工智能是计算机科学的一个分支，旨在创建能够模拟人类智能的系统..."}
  ],
  "rejected": [
    {"role": "user", "content": "什么是人工智能？"},
    {"role": "assistant", "content": "人工智能就是电脑变聪明了。"}
  ]
}
```

数据集类 `DPODataset` 会把 chosen 和 rejected 分别编码，然后在训练时拼接成一个 batch：

```
batch 的前半部分 = chosen 样本
batch 的后半部分 = rejected 样本
```

### 2.3 参考模型的作用："不忘初心"

DPO 不是简单地让 chosen 概率变高、rejected 概率变低。如果这样做，模型可能会走极端——把某些回答的概率推到极高，同时把自己学过的有用知识也破坏了。

为了防止这一点，DPO 引入了**参考模型（ref_model）**：一个冻结的、不参与训练的模型副本（就是训练开始时的模型快照）。

```
参考模型的意义：
- 训练模型说："我要让 chosen 比 rejected 的概率更高！"
- 参考模型说："别太离谱，跟你原来的样子差太多的话要扣分。"
```

这个机制在代码中体现得很清楚：

```python
# trainer/train_dpo.py 第 193-196 行
# 初始化参考模型（ref_model冻结）
ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
ref_model.eval()
ref_model.requires_grad_(False)  # 冻结，不参与训练
```

### 2.4 DPO 损失函数详解

这是 DPO 的核心，我们逐行来看：

```python
# trainer/train_dpo.py 第 33-51 行
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # 1. 计算每个序列的平均 log prob（用 mask 去掉 padding）
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()

    # 2. 把 chosen 和 rejected 分开（batch 前半是 chosen，后半是 rejected）
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    # 3. 计算策略模型和参考模型的 log ratio 之差
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    logits = pi_logratios - ref_logratios

    # 4. 用 sigmoid 损失让 chosen 的"相对概率"更高
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()
```

用大白话解释这个公式：

```
          策略模型觉得 chosen 比 rejected 好多少
logits = ─────────────────────────────────────── - 参考模型觉得 chosen 比 rejected 好多少
```

- 如果策略模型比参考模型**更偏向 chosen**，logits > 0，loss 较小（好！）
- 如果策略模型比参考模型**更偏向 rejected**，logits < 0，loss 较大（要惩罚！）

### 2.5 关键超参数

| 参数 | 值 | 说明 |
|------|----|------|
| `beta` | 0.1 | 偏好强度。越大越严格地区分好坏，太大会导致训练不稳 |
| `learning_rate` | 4e-8 | 极低的学习率！比 SFT 低了几十倍，防止遗忘已学知识 |
| `from_weight` | `full_sft` | 基于 SFT 模型继续训练 |

> 注意学习率只有 4e-8，这是 DPO 的一大特点。因为 DPO 本质上是在"微调偏好"，如果学习率太高，模型会忘记 SFT 阶段学到的能力。

### 2.6 训练流程

```
                  x_chosen ──┐
                              ├──→ 拼接 ──→ ref_model（冻结）──→ ref_log_probs ─┐
                  x_rejected ─┘                                                   │
                                                                                  ├──→ dpo_loss
                  x_chosen ──┐                                                    │
                              ├──→ 拼接 ──→ policy_model（训练）──→ policy_log_probs ─┘
                  x_rejected ─┘
```

---

## 3. PPO — 近端策略优化

### 3.1 核心思想

PPO（Proximal Policy Optimization）是最"正统"的强化学习对齐方法。与 DPO 不同，PPO 不需要事先准备好"好回答 vs 坏回答"的数据，而是让模型**自己生成回答**，然后用**奖励模型打分**，再根据分数更新模型。

这就像一个自学的厨师：自己做菜，请评委打分，根据分数改进厨艺。

### 3.2 Actor-Critic 架构

PPO 是四种方法中最复杂的，因为它需要**5 个模型**同时工作：

```
┌─────────────────────────────────────────────────────────────┐
│                    PPO 训练架构                               │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  Actor 模型   │    │  Old Actor   │    │  Reference   │   │
│  │  (要训练的)   │    │  (Actor旧版) │    │  (冻结不动)  │   │
│  │              │    │              │    │              │    │
│  │  生成回答     │    │  计算旧概率   │    │  KL惩罚基准  │   │
│  │  计算新概率   │    │  重要性采样   │    │  防止偏离    │   │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘   │
│         │                   │                   │           │
│         └─────────┬─────────┘                   │           │
│                   │                             │           │
│                   ▼                             │           │
│         ┌──────────────────┐                    │           │
│         │  PPO 裁剪损失     │◄───────────────────┘           │
│         │  policy_loss      │                               │
│         └──────────────────┘                                │
│                                                              │
│  ┌──────────────┐    ┌──────────────────────┐               │
│  │  Critic 模型  │    │  Reward Model（外部） │               │
│  │  (要训练的)   │    │  (冻结不动)           │               │
│  │              │    │                      │               │
│  │  估计价值V    │    │  给回答打分 R         │               │
│  └──────┬───────┘    └──────────┬───────────┘               │
│         │                      │                            │
│         ▼                      ▼                            │
│    value_loss          advantages = R - V                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

各模型的角色：

1. **Actor（策略模型）**：就是我们要训练的 LLM，负责生成回答
2. **Old Actor**：Actor 的历史副本，每隔 N 步从 Actor 复制一次，用于计算"重要性采样比率"
3. **Critic（价值模型）**：估计当前状态的价值 V，为计算优势函数服务
4. **Reference（参考模型）**：冻结的初始模型，用于 KL 散度惩罚，与 DPO 中的 ref_model 作用相同
5. **Reward Model（奖励模型）**：外部预训练的打分模型（MiniMind 使用 `internlm2-1_8b-reward`）

### 3.3 Critic 模型的实现

Critic 模型复用了 MiniMind 的 LLM 架构，只是把输出层从"词表大小"换成了"单个数值"：

```python
# trainer/train_ppo.py 第 29-41 行
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        # 替换lm_head为输出单一价值的线性层
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # 使用基础模型获取隐藏状态
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        # 使用value_head获取价值估计
        values = self.value_head(hidden_states).squeeze(-1)
        return values
```

直觉理解：Actor 负责"做事"，Critic 负责"评价"。Actor 做完一件事，Critic 说"你做得比预期好/差"，这个差值就是**优势（advantage）**。

### 3.4 PPO 裁剪损失

PPO 的核心创新是**裁剪（clipping）**机制，防止策略更新幅度过大：

```python
# trainer/train_ppo.py 第 169-174 行
ratio = torch.exp(actor_logp - old_logp)                          # 新策略/旧策略的概率比
surr1 = ratio * advantages                                        # 原始目标
surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon,
                     1.0 + args.clip_epsilon) * advantages         # 裁剪后的目标
policy_loss = -torch.min(surr1, surr2).mean()                     # 取较小值（保守更新）
value_loss = F.mse_loss(values, rewards)                           # Critic 的损失
loss = policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref + aux_loss
```

这段代码的逻辑可以用一张图来理解：

```
        policy_loss
            ^
            |
            |      /
            |     /  surr1（无裁剪）
            |    /
            |   /────── surr2（裁剪后，ratio被限制在 [0.9, 1.1]）
            |  /
            | /
   ─────────┼─────────────→ ratio
          1.0
   (ratio=1 表示新旧策略一样)
```

当 ratio 偏离 1.0 太远时（说明新策略和旧策略差别太大），裁剪机制会"刹车"，防止更新过猛。

### 3.5 训练流程

```
Prompt ──→ Actor 生成回答 ──→ Reward Model 打分(R)
                    │
                    ├──→ Critic 估值(V) ──→ advantage = R - V
                    │
                    ├──→ Actor 计算新概率 ──┐
                    │                       ├──→ ratio ──→ PPO裁剪损失
                    └──→ Old Actor 计算旧概率 ┘
                    │
                    └──→ Reference 计算KL惩罚
```

### 3.6 Old Actor 的更新

Old Actor 不是每步都更新的，而是每隔一定步数从 Actor 复制：

```python
# trainer/train_ppo.py 第 222-227 行
if (step + 1) % args.update_old_actor_freq == 0:
    raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
    raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
    state_dict = raw_actor.state_dict()
    old_actor_model.load_state_dict({k: v.detach().cpu() for k, v in state_dict.items()})
    old_actor_model.to(args.device)
```

默认 `update_old_actor_freq=4`，即每 4 步更新一次。

### 3.7 关键超参数

| 参数 | 值 | 说明 |
|------|----|------|
| `clip_epsilon` | 0.1 | PPO 裁剪范围 [1-0.1, 1+0.1] = [0.9, 1.1] |
| `vf_coef` | 0.5 | Critic 损失的权重 |
| `kl_coef` | 0.02 | KL 散度惩罚的权重 |
| `learning_rate` | 8e-8 | Actor 学习率 |
| `critic_learning_rate` | 8e-8 | Critic 学习率 |
| `update_old_actor_freq` | 4 | 每 4 步同步 Old Actor |

---

## 4. GRPO — 组内相对策略优化

### 4.1 核心思想

GRPO（Group Relative Policy Optimization）的核心创新是：**对每个 prompt 生成 N 个回答，在组内比较来确定优势。**

PPO 需要一个 Critic 模型来估计"这个回答有多好"，但训练 Critic 本身就很困难。GRPO 的巧妙之处在于——**用同组其他回答作为参照，完全省掉了 Critic。**

打个比方：
- PPO：请一个裁判（Critic）给每道菜打绝对分数
- GRPO：做 8 道菜，互相比较排名，排名靠前的就是好的

### 4.2 生成与组内比较

每个 prompt 生成 `num_generations=8` 个回答，然后用奖励模型分别打分：

```
Prompt: "什么是黑洞？"
    │
    ├──→ 回答1: "黑洞是..." (得分: 2.1)
    ├──→ 回答2: "嗯这个..." (得分: -0.5)
    ├──→ 回答3: "黑洞是..." (得分: 1.8)   8个回答
    ├──→ 回答4: "不知道..." (得分: -1.2)   组内比较
    ├──→ 回答5: "黑洞是..." (得分: 2.3)   ──→ 标准化
    ├──→ 回答6: "黑洞就..." (得分: 0.7)
    ├──→ 回答7: "黑洞是..." (得分: 1.5)
    └──→ 回答8: "额..."    (得分: -0.8)
```

### 4.3 组内标准化计算优势

```python
# trainer/train_grpo.py 第 133-137 行
grouped_rewards = rewards.view(-1, args.num_generations)                      # [B, 8]
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*8]
std_r = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)    # [B*8]
advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)       # 组内标准化
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)     # 全局归一化
```

这段代码做的事情：

1. 把奖励按组（每组 8 个）分开
2. 计算每组的均值和标准差
3. 对每个回答的奖励做组内标准化：`(reward - 组均值) / 组标准差`
4. 得到的 advantage > 0 表示"比组内平均水平好"，< 0 表示"比平均水平差"

### 4.4 GRPO 损失函数

```python
# trainer/train_grpo.py 第 144-148 行
kl_div = ref_per_token_logps - per_token_logps
per_token_kl = torch.exp(kl_div) - kl_div - 1                                    # KL散度（逐token）
per_token_loss = -(torch.exp(per_token_logps - per_token_logps.detach())
                   * advantages.unsqueeze(1)
                   - args.beta * per_token_kl)                                     # 策略梯度 + KL惩罚
policy_loss = ((per_token_loss * completion_mask).sum(dim=1)
               / completion_mask.sum(dim=1)).mean()                                # 按token平均后取batch均值
```

这里的 `torch.exp(per_token_logps - per_token_logps.detach())` 是一个小技巧：在数值上等于 1（因为 detach 后自己除自己），但梯度不为零，可以正确反向传播。这等效于直接对 log_prob 求梯度乘以 advantage。

### 4.5 GRPO 的架构（比 PPO 简洁很多）

```
┌───────────────────────────────────────────────────┐
│                 GRPO 训练架构                       │
│                                                    │
│  ┌──────────────┐         ┌──────────────┐        │
│  │  Policy 模型  │         │  Reference   │        │
│  │  (要训练的)   │         │  (冻结不动)  │        │
│  │              │         │              │        │
│  │  生成N个回答  │         │  KL惩罚基准  │        │
│  │  计算log_prob│         │              │        │
│  └──────┬───────┘         └──────┬───────┘        │
│         │                        │                │
│         ▼                        ▼                │
│    per_token_logps         ref_per_token_logps    │
│         │                        │                │
│         └─────────┬──────────────┘                │
│                   ▼                               │
│         ┌──────────────────┐                      │
│         │  组内标准化优势    │                      │
│         │  + KL惩罚损失     │                      │
│         └──────────────────┘                      │
│                                                    │
│  ┌──────────────────────┐                         │
│  │  Reward Model（外部） │                         │
│  │  给每个回答打分        │                         │
│  └──────────────────────┘                         │
│                                                    │
│  没有 Critic！没有 Old Actor！                     │
└───────────────────────────────────────────────────┘
```

### 4.6 关键超参数

| 参数 | 值 | 说明 |
|------|----|------|
| `num_generations` | 8 | 每个 prompt 生成 8 个回答 |
| `beta` | 0.02 | KL 惩罚系数 |
| `learning_rate` | 8e-8 | 学习率 |
| `max_gen_len` | 1536 | 每个回答的最大生成长度 |

> 注意：因为每个 prompt 要生成 8 个回答，GRPO 的显存占用和计算量比 DPO/SPO 大很多。batch_size 通常需要设得较小。

---

## 5. SPO — 自适应基线优化

### 5.1 核心思想

SPO（Self-Play Optimization，自适应基线优化）是四种方法中**最简洁的 RL 方案**：

- 每个 prompt 只生成 **1 个**回答（不像 GRPO 要生成 8 个）
- 不需要 Critic 模型（不像 PPO）
- 用**自适应基线**代替 GRPO 的组内比较

核心问题在于：只有 1 个回答，怎么判断好坏？

答案是：**用一个自适应基线（baseline）作为参照。** 如果回答得分高于基线，说明做得好；低于基线，说明做得差。这个基线会随着训练自动调整。

### 5.2 AutoAdaptiveValueTracker：贝叶斯自适应基线

这是 SPO 的核心组件——一个用贝叶斯方法维护的自适应基线：

```python
# trainer/train_spo.py 第 27-66 行
class AutoAdaptiveValueTracker:
    """SPO自适应价值追踪器"""
    def __init__(self, rho_mode='kl', rho_const=0.9, D_half=0.06,
                 clip_lower=0.5, clip_upper=0.96):
        self.rho_mode = rho_mode
        self.rho_const = rho_const
        self.D_half = D_half
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper
        N_init = 1.0 / (1.0 - self.clip_lower)
        self.alpha = 0.5 * N_init         # Beta分布参数 alpha
        self.beta = 0.5 * N_init          # Beta分布参数 beta

    def get_baselines(self, batch_size):
        baseline = self.alpha / (self.alpha + self.beta)    # Beta分布的均值
        return torch.full((batch_size,), baseline, dtype=torch.float32)

    def compute_rho(self, cur_mean_logprob):
        """根据策略变化程度动态调整遗忘因子 rho"""
        if self.old_mean_logprob is None:
            return self.rho_const
        kl = abs(self.old_mean_logprob - cur_mean_logprob)
        rho = 2 ** (-kl / self.D_half)                     # KL越大，rho越小（忘得越多）
        return max(min(rho, self.clip_upper), self.clip_lower)

    def update(self, rewards, cur_logprobs=None, response_masks=None):
        """用新的奖励观测更新 Beta 分布参数"""
        # ...计算 rho...
        scale = 3.0
        normalized_rewards = (rewards + scale) / (2 * scale)  # 归一化到 [0, 1]
        avg_normalized_reward = normalized_rewards.mean().item()
        self.alpha = rho * self.alpha + avg_normalized_reward   # 衰减旧记忆 + 加入新观测
        self.beta = rho * self.beta + (1 - avg_normalized_reward)
        return rho
```

用大白话解释：

- **alpha 和 beta** 是 Beta 分布的两个参数，`alpha/(alpha+beta)` 就是基线值
- **rho（遗忘因子）**：策略变化大时 rho 变小，让基线更快地适应新策略
- 每一步用新的奖励更新 alpha 和 beta，基线值会平滑地跟随训练变化

这就像一个"经验丰富的评委"——见过足够多的回答后，自然知道什么水平算"及格线"，而且会随着模型进步不断抬高标准。

### 5.3 优势计算

```python
# trainer/train_spo.py 第 169-177 行
baselines = value_tracker.get_baselines(len(prompts)).to(args.device)  # [B]

scale = 3.0
unnormalized_baselines = baselines * (2 * scale) - scale               # 反归一化到 [-3, 3]
advantages = rewards - unnormalized_baselines                          # 优势 = 奖励 - 基线

# 裁剪防止梯度爆炸，不再做 batch 内归一化（因为 baseline 已经提供了跨 batch 的稳定基线）
advantages = advantages.clamp(-5.0, 5.0)
```

### 5.4 KL 惩罚方式

SPO 的 KL 惩罚方式与 GRPO 相同：

```python
# trainer/train_spo.py 第 184-188 行
kl_div = ref_per_token_logps - per_token_logps
per_token_kl = torch.exp(kl_div) - kl_div - 1                   # 近似KL散度
per_token_loss = -per_token_logps * advantages.unsqueeze(1) + args.beta * per_token_kl
policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
```

### 5.5 关键超参数

| 参数 | 值 | 说明 |
|------|----|------|
| `learning_rate` | 1e-7 | 四种方法中最高的学习率 |
| `beta` | 0.02 | KL 惩罚系数 |
| `accumulation_steps` | 4 | 梯度累积（补偿小 batch_size） |
| `rho_const` | 0.9 | 默认遗忘因子 |
| `D_half` | 0.06 | KL 达到多少时 rho 减半 |

---

## 6. 四种对齐方法对比

### 6.1 总览表

| 特性 | DPO | PPO | GRPO | SPO |
|------|-----|-----|------|-----|
| **数据需求** | 偏好对（chosen/rejected） | Prompt + 奖励模型 | Prompt + 奖励模型 | Prompt + 奖励模型 |
| **模型数量** | 2（policy + ref） | 5（actor + old_actor + critic + ref + reward） | 3（policy + ref + reward） | 3（policy + ref + reward） |
| **生成采样** | 不需要 | 每 prompt 1 次 | 每 prompt N 次（默认8） | 每 prompt 1 次 |
| **复杂度** | 低 | 高 | 中 | 低 |
| **学习率** | 4e-8 | 8e-8 | 8e-8 | 1e-7 |
| **显存占用** | 较低 | 很高（5个模型） | 中等 | 较低 |
| **训练速度** | 最快 | 最慢 | 较慢（N倍生成） | 较快 |
| **优势来源** | 隐式（chosen vs rejected） | Critic 估值 | 组内标准化 | 自适应基线 |

### 6.2 如何选择？

```
需要离线偏好数据？ ──→ 是 ──→ DPO（最简单，最稳定）
       │
       否（有奖励模型）
       │
       ├──→ 显存充足？ ──→ 是 ──→ PPO（最经典，效果上限高）
       │
       ├──→ 希望简洁高效？ ──→ SPO（只需1次生成，有自适应基线）
       │
       └──→ 希望组内比较？ ──→ GRPO（DeepSeek 提出，不需要Critic）
```

### 6.3 数据集差异

DPO 使用 `DPODataset`（偏好对数据），其余三种使用 `RLAIFDataset`（只需 prompt）：

```
DPO 数据 (dataset/dpo.jsonl):
┌─────────────────────────────────┐
│ {                               │
│   "chosen": [...对话...],       │  <-- 好的回答
│   "rejected": [...对话...]      │  <-- 差的回答
│ }                               │
└─────────────────────────────────┘

PPO/GRPO/SPO 数据 (dataset/rlaif-mini.jsonl):
┌─────────────────────────────────┐
│ {                               │
│   "conversations": [            │
│     {"content": "问题"},        │  <-- 只需要 prompt
│     {"content": "回答"}         │  <-- 模型自己生成，不用这个
│   ]                             │
│ }                               │
└─────────────────────────────────┘
```

---

## 7. 实操指南

### 7.1 运行训练

```bash
# DPO — 最简单，适合入门
python trainer/train_dpo.py --epochs 1 --batch_size 4

# PPO — 最复杂，需要外部奖励模型
# 注意：需要先下载 internlm2-1_8b-reward 模型
python trainer/train_ppo.py --epochs 1 --batch_size 2

# GRPO — 每个 prompt 生成 8 个回答
python trainer/train_grpo.py --epochs 1 --batch_size 2

# SPO — 最简洁的 RL 方案
python trainer/train_spo.py --epochs 1 --batch_size 2
```

### 7.2 训练之后

对齐训练完成后，模型权重保存在 `out/` 目录下：

```
out/
├── dpo_512.pth           # DPO 训练结果
├── ppo_actor_512.pth     # PPO 训练结果
├── grpo_512.pth          # GRPO 训练结果
└── spo_512.pth           # SPO 训练结果
```

用对齐后的模型进行推理：

```bash
# 使用 DPO 对齐后的模型
python eval_llm.py --weight dpo --device cuda:0

# 使用 GRPO 对齐后的模型
python eval_llm.py --weight grpo --device cuda:0
```

### 7.3 训练建议

1. **先跑 DPO**：最简单，不需要奖励模型，适合验证流程
2. **注意学习率**：对齐训练的学习率极低（e-7 ~ e-8 量级），这是正常的
3. **显存不够？** 降低 batch_size，增加 accumulation_steps 来补偿
4. **PPO 的奖励模型**：需要单独下载 `internlm2-1_8b-reward`，占用额外显存
5. **GRPO 的 num_generations**：默认 8，显存不够可以降到 4，但效果可能下降

### 7.4 观察训练效果

训练日志中的关键指标：

| 指标 | DPO | PPO | GRPO/SPO | 期望趋势 |
|------|-----|-----|----------|----------|
| `dpo_loss` | 有 | - | - | 下降 |
| `policy_loss` / `actor_loss` | - | 有 | 有 | 下降 |
| `reward` | - | 有 | 有 | 上升 |
| `kl` / `kl_ref` | - | 有 | - | 保持较低 |
| `value_loss` / `critic_loss` | - | 有 | - | 下降 |
| `baseline` | - | - | SPO 有 | 逐渐上升 |

---

## 8. 总结

对齐训练是 LLM 训练流水线的最后一步，也是让模型从"能用"变成"好用"的关键。四种方法各有特色：

- **DPO**：离线、简单、稳定，适合有偏好数据的场景
- **PPO**：最经典的 RL 方法，效果上限高但实现复杂
- **GRPO**：巧妙地用组内比较省掉了 Critic，是 PPO 的轻量替代
- **SPO**：用自适应基线进一步简化，每个 prompt 只需一次生成

回到我们的比喻：SFT 让厨师学会了做菜，对齐让厨师学会了做好吃的菜。至于用哪种方法训练——请评委打分（PPO）、做 8 道菜比较（GRPO）、跟自己的"平均水平"比（SPO）、还是直接看范文和反面教材（DPO）——都能达到目标，只是路径不同。

---

[上一章：监督微调](05-sft.md) | [下一章：进阶技术](07-advanced.md)
