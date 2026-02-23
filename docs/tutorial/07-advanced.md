# 07 - 进阶技术

> LoRA、知识蒸馏、推理训练、部署 —— 让你的模型更强、更轻、更聪明。

[上一章：对齐训练](06-alignment.md) | [返回目录](README.md)

---

前面几章我们走完了"预训练 -> SFT -> 对齐"的标准流程。这一章介绍几种**锦上添花**的进阶技术：

1. **LoRA** —— 用极少参数实现高效微调
2. **知识蒸馏** —— 让大模型把"功力"传给小模型
3. **推理训练** —— 教模型学会"先想后答"
4. **推理部署** —— 把训好的模型跑起来、对外提供服务

---

## 1. LoRA —— 低秩适配

### 1.1 核心思想："只调几个旋钮"

想象你买了一台出厂调好的钢琴。全参数微调（Full SFT）相当于把每根弦都重新拧一遍；而 LoRA 相当于**只在关键位置加几个微调旋钮**，不碰原来的弦。

好处显而易见：
- **训练快**：需要更新的参数极少
- **省显存**：优化器状态只跟踪 LoRA 参数
- **易切换**：同一个底座模型可以搭配不同的 LoRA "插件"（身份认知、医疗问答等）
- **不破坏原始能力**：原始权重完全不动

### 1.2 原理：低秩分解 W + BA

LoRA 的数学思想很简单。对于模型中一个 `d x d` 的权重矩阵 `W`，我们不直接修改它，而是在旁边加一个**低秩分解矩阵** `BA`：

```
输出 = W(x) + B(A(x))
```

其中：
- **A**：`(d, r)` 的矩阵，将输入从 `d` 维压缩到 `r` 维（高斯初始化）
- **B**：`(r, d)` 的矩阵，将 `r` 维映射回 `d` 维（全零初始化）
- **r**（rank）：通常取 4 或 8，远小于 `d`

因为 B 全零初始化，训练开始时 `BA = 0`，模型行为和原始模型完全一致。随着训练推进，`BA` 学到一个小的"修正量"。

> **参数量对比**：原始方阵 `d x d`，LoRA 只有 `d x r + r x d = 2dr`。当 `d=512, r=8` 时，LoRA 参数只有原始的 `2 x 8 / 512 = 3.1%`。实际中 LoRA 参数占全部参数约 **0.5%**。

### 1.3 代码实现

LoRA 模块定义在 `model/model_lora.py` 中，只有十几行代码：

```python
# model/model_lora.py:6-18
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank                                      # 低秩的 rank
        self.A = nn.Linear(in_features, rank, bias=False)     # 降维：d -> r
        self.B = nn.Linear(rank, out_features, bias=False)    # 升维：r -> d
        # A 高斯初始化 —— 保证初始有信号
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # B 全零初始化 —— 保证初始 BA = 0，不改变原始模型
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))
```

把 LoRA 挂载到模型上的逻辑也很巧妙——只给**方阵 Linear 层**（即 `in_features == out_features` 的层）添加 LoRA，这些通常是注意力层中的 Q/K/V/O 投影矩阵：

```python
# model/model_lora.py:21-32
def apply_lora(model, rank=8):
    for name, module in model.named_modules():
        # 只对方阵 Linear 层添加 LoRA
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 新的 forward：原始输出 + LoRA 修正
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora
```

### 1.4 训练：冻结原始参数，只训练 LoRA

训练时的关键步骤是**冻结所有非 LoRA 参数**，只让 LoRA 参数参与梯度更新：

```python
# trainer/train_lora.py:146-152
lora_params = []
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True     # LoRA 参数：可训练
        lora_params.append(param)
    else:
        param.requires_grad = False    # 原始参数：冻结
```

优化器也只跟踪 LoRA 参数：

```python
# trainer/train_lora.py:161
optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
```

这样一来，优化器状态（动量、方差等）只需要为 0.5% 的参数分配内存，显存节省非常明显。

### 1.5 保存与加载

LoRA 权重的保存和加载也是独立的——只保存带 `lora` 前缀的参数：

```python
# model/model_lora.py:45-53 (save_lora)
def save_lora(model, path):
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {f'{clean_name}.lora.{k}': v
                          for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)
```

保存的文件非常小（几百 KB），可以方便地分发和切换。

### 1.6 实操

```bash
# 基于 full_sft 权重进行 LoRA 微调
python trainer/train_lora.py --epochs 50 --batch_size 32

# 默认使用 lora_identity.jsonl 数据（身份认知微调）
# 保存路径: out/lora/lora_identity_512.pth
```

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `--lora_name` | `lora_identity` | LoRA 权重名称 |
| `--from_weight` | `full_sft` | 基于哪个底座模型 |
| `--data_path` | `dataset/lora_identity.jsonl` | 训练数据 |
| `--epochs` | 50 | 因为参数少，需要多轮训练 |
| `--learning_rate` | 1e-4 | 比全参数微调大一个数量级 |

---

## 2. 知识蒸馏

### 2.1 核心思想："大模型教小模型"

知识蒸馏就像一个经验丰富的老师（大模型）带一个新手学生（小模型）。与其让学生自己从头看书（只学标准答案），不如让老师把自己对每道题的"理解"也传授给学生。

这个"理解"就是老师模型输出的**概率分布**。比如对于"北京是中国的___"这个问题：

| token | 标准答案 | 老师的理解 |
|-------|----------|------------|
| 首都 | 100% | 85% |
| 中心 | 0% | 8% |
| 核心 | 0% | 4% |
| 城市 | 0% | 3% |

标准答案只告诉学生"答案是首都"，但老师的概率分布还传递了"中心、核心也有点关系"这种**暗知识**（dark knowledge）。

### 2.2 温度参数：让分布更"柔软"

直接用原始 softmax 输出，概率往往非常集中（首都 99.9%，其他趋近于 0）。为了让暗知识更容易被学到，我们引入**温度参数 T**：

```
softmax(logits / T)
```

- **T = 1**：正常的 softmax，分布尖锐
- **T > 1**：分布变平滑，低概率的 token 也能获得可观的概率
- **T < 1**：分布更尖锐（很少使用）

MiniMind 默认 `temperature=1.5`，让分布足够平滑又不至于丢失太多信息。

### 2.3 蒸馏损失函数

蒸馏的核心是让学生模型的输出分布**逼近**老师的输出分布。MiniMind 使用 KL 散度来衡量两个分布的差异：

```python
# trainer/train_distillation.py:24-35
def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    # 老师的概率分布（不需要梯度）
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    # 学生的 log 概率分布
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # KL 散度：衡量两个分布的差异
    kl = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)

    # 乘以 T^2 补偿温度缩放带来的梯度缩小
    return (temperature ** 2) * kl
```

这里有一个容易忽略的细节：`temperature ** 2` 的乘法。因为除以 T 会让梯度缩小 T 倍（链式法则），而 KL 散度里有两次 softmax/log_softmax，所以要乘 T^2 来补偿。

### 2.4 总损失：CE + 蒸馏

训练时的总损失是两部分的加权和：

```
总损失 = alpha * CE_loss + (1 - alpha) * distill_loss
```

- **CE_loss**：学生和标准答案之间的交叉熵（学"对的答案"）
- **distill_loss**：学生和老师之间的 KL 散度（学"老师的理解"）
- **alpha = 0.5**：两者各占一半

```python
# trainer/train_distillation.py:89-90
# 总损失 = alpha * CE + (1-alpha) * Distill
loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps
```

### 2.5 教师与学生的配置

| 角色 | hidden_size | num_layers | 参数量 |
|------|-------------|------------|--------|
| 教师模型 | 768 | 16 | ~104M |
| 学生模型 | 512 | 8 | ~26M |

教师模型在训练全程**冻结**，设为 eval 模式，不计算梯度：

```python
# trainer/train_distillation.py:42-43
teacher_model.eval()
teacher_model.requires_grad_(False)
```

因为两个模型的词表大小可能不同（hidden_size 不同不影响词表，但以防万一），代码中还做了对齐：

```python
# trainer/train_distillation.py:63
teacher_logits = teacher_logits[..., :vocab_size_student]
```

### 2.6 实操

```bash
# 知识蒸馏：用 768 维的大模型教 512 维的小模型
python trainer/train_distillation.py --epochs 6 --batch_size 32
```

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `--student_hidden_size` | 512 | 学生模型维度 |
| `--teacher_hidden_size` | 768 | 教师模型维度 |
| `--from_student_weight` | `full_sft` | 学生的初始权重 |
| `--from_teacher_weight` | `full_sft` | 教师的权重 |
| `--alpha` | 0.5 | CE 损失权重 |
| `--temperature` | 1.5 | 蒸馏温度 |
| `--save_weight` | `full_dist` | 输出权重名 |

---

## 3. 推理训练（Reasoning / 思维链）

### 3.1 核心思想：让模型学会"先想后答"

人类解决复杂问题时，通常先在脑子里过一遍推理过程，然后才给出答案。推理训练就是教模型也这样做——先生成一段**思考过程**，再给出**最终答案**。

这和 OpenAI o1/DeepSeek R1 的思路一致，只不过 MiniMind 用了最简单的方式来实现。

### 3.2 特殊标签

推理训练使用两组特殊标签来标记思考过程和最终答案：

```
<think>这里是模型的思考过程...</think>
<answer>这里是最终答案</answer>
```

训练数据（`r1_mix_1024.jsonl`）中每条样本都包含这种格式的推理过程。

### 3.3 特殊 token 加权：10 倍权重

推理训练中最关键的技巧是**对特殊标签给予更高的权重**。模型必须学会在正确的位置生成 `<think>`、`</think>`、`<answer>`、`</answer>` 这些标签，否则输出格式就会混乱。

```python
# trainer/train_reason.py:24-27
start_of_think_ids = tokenizer('<think>').input_ids
end_of_think_ids = tokenizer('</think>').input_ids
start_of_answer_ids = tokenizer('<answer>').input_ids
end_of_answer_ids = tokenizer('</answer>').input_ids
```

```python
# trainer/train_reason.py:45-51
# 找到标签 token 的位置
sp_ids = torch.isin(shift_labels.view(-1),
                    torch.tensor(start_of_think_ids + end_of_think_ids
                                 + start_of_answer_ids + end_of_answer_ids
                                 ).to(args.device))
loss_mask_flat = loss_mask.view(-1)
# 给标签 token 10 倍权重
loss_mask_flat[sp_ids] = 10
```

这意味着如果模型在 `<think>` 标签的位置预测错误，损失会被放大 10 倍。这样模型能很快学会**输出格式**，不会把思考过程和答案混在一起。

### 3.4 训练流程

推理训练通常在 DPO/对齐之后进行：

1. 加载对齐后的权重（默认 `--from_weight dpo`）
2. 使用包含推理过程的数据集 `r1_mix_1024.jsonl`
3. 最大序列长度设为 720（推理过程比普通对话更长）
4. 学习率较低（1e-6），避免破坏已有能力

### 3.5 实操

```bash
# 推理训练
python trainer/train_reason.py --epochs 1 --batch_size 8
```

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `--from_weight` | `dpo` | 基于对齐后的模型 |
| `--data_path` | `dataset/r1_mix_1024.jsonl` | 推理数据 |
| `--max_seq_len` | 720 | 较长的截断长度 |
| `--learning_rate` | 1e-6 | 较低的学习率 |
| `--save_weight` | `reason` | 输出权重名 |

---

## 4. 推理部署

模型训练好了，接下来就是让它"开口说话"。MiniMind 提供两种部署方式。

### 4.1 命令行交互：eval_llm.py

`eval_llm.py` 是最直接的推理方式，支持自动测试和手动对话两种模式。

**基本用法**：

```bash
# 使用 full_sft 权重
python eval_llm.py --weight full_sft --device cuda:0

# 使用推理模型（会启用 <think>...</think> 格式）
python eval_llm.py --weight reason --device cuda:0

# 加载 LoRA 权重
python eval_llm.py --lora_weight lora_identity --device cuda:0
```

启动后会提示选择模式：

```
[0] 自动测试      ← 用预设的 8 个问题自动测试
[1] 手动输入      ← 自己打字对话
```

**核心功能**：

| 功能 | 参数 | 说明 |
|------|------|------|
| 多轮对话 | `--historys 4` | 保留最近 4 轮历史（需为偶数） |
| 流式输出 | 默认启用 | 使用 `TextStreamer` 逐 token 输出 |
| 速度统计 | `--show_speed 1` | 显示 tokens/s 生成速度 |
| 温度控制 | `--temperature 0.85` | 控制生成的随机性 |
| Top-p 采样 | `--top_p 0.85` | Nucleus 采样阈值 |
| RoPE 外推 | `--inference_rope_scaling` | 启用 4 倍位置外推 |

**推理模型的特殊处理**：

当 `--weight reason` 时，`eval_llm.py` 会自动启用思维链模板：

```python
# eval_llm.py:74
if args.weight == 'reason': templates["enable_thinking"] = True
```

这样模型会先生成 `<think>...</think>` 的思考过程，再输出 `<answer>...</answer>` 的最终答案。

### 4.2 OpenAI 兼容 API：serve_openai_api.py

如果你想让其他应用（比如 ChatBox、Open WebUI 等）调用你的模型，可以启动 OpenAI 兼容的 API 服务：

```bash
python scripts/serve_openai_api.py
```

服务默认监听 `0.0.0.0:8998`，提供 `/v1/chat/completions` 接口，同时支持**流式**和**非流式**响应。

**请求示例**（使用 curl）：

```bash
curl http://localhost:8998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimind",
    "messages": [{"role": "user", "content": "你好"}],
    "temperature": 0.7,
    "stream": false
  }'
```

**支持的参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | string | - | 模型名称（任意值均可） |
| `messages` | list | - | 对话历史 |
| `temperature` | float | 0.7 | 生成温度 |
| `top_p` | float | 0.92 | 采样阈值 |
| `max_tokens` | int | 8192 | 最大生成长度 |
| `stream` | bool | false | 是否流式输出 |

API 服务同样支持 LoRA 权重加载：

```bash
python scripts/serve_openai_api.py --weight full_sft --lora_weight lora_identity
```

---

## 5. 完整训练流程总结

### 5.1 全景图

```
预训练 ──→ 监督微调 ──→ 对齐训练 ──→ 推理训练 ──→ 部署
  │           │            │             │           │
  │           │            │             │           ├── eval_llm.py (命令行)
  │           │            │             │           └── serve_openai_api.py (API)
  │           │            │             │
  │           │            │             └── train_reason.py
  │           │            │
  │           │            ├── train_dpo.py    (离线偏好)
  │           │            ├── train_ppo.py    (在线强化)
  │           │            ├── train_grpo.py   (组内相对)
  │           │            └── train_spo.py    (自适应基线)
  │           │
  │           ├── train_full_sft.py  (全参数微调)
  │           └── train_lora.py      (LoRA 微调，二选一)
  │
  └── train_pretrain.py
```

### 5.2 可选分支

上面的流程不是唯一的路径。有些技术是**独立的分支**，可以根据需要灵活组合：

```
                    ┌── train_lora.py ──────────────────┐
                    │   (替代 Full SFT，轻量微调)        │
                    │                                    │
预训练 ──→ Full SFT ──→ DPO ──→ Reason ──→ 部署        │
                    │                                    │
                    └── train_distillation.py ──→ 部署   │
                        (大模型→小模型，独立于对齐流程)   │
                                                         │
                                    eval_llm.py ←────────┘
                                    (加载 LoRA 插件推理)
```

**关键路径说明**：

| 路径 | 说明 | 输出权重 |
|------|------|----------|
| pretrain -> full_sft -> dpo -> reason | 完整标准流程 | `reason_512.pth` |
| pretrain -> full_sft -> lora | 轻量微调路径 | `lora/lora_identity_512.pth` |
| pretrain -> full_sft -> distillation | 模型压缩路径 | `full_dist_512.pth` |
| pretrain -> full_sft -> ppo/grpo/spo | 其他对齐方法 | `ppo_actor/grpo/spo_512.pth` |

### 5.3 各阶段一键命令

```bash
# 1. 预训练
python trainer/train_pretrain.py --epochs 1 --batch_size 32

# 2. 监督微调（二选一）
python trainer/train_full_sft.py --epochs 2 --batch_size 16    # 全参数
python trainer/train_lora.py --epochs 50 --batch_size 32       # LoRA

# 3. 对齐（四选一或多选）
python trainer/train_dpo.py --epochs 1 --batch_size 4
python trainer/train_ppo.py --epochs 1 --batch_size 2
python trainer/train_grpo.py --epochs 1 --batch_size 2
python trainer/train_spo.py --epochs 1 --batch_size 2

# 4. 可选：知识蒸馏
python trainer/train_distillation.py --epochs 6 --batch_size 32

# 5. 可选：推理训练
python trainer/train_reason.py --epochs 1 --batch_size 8

# 6. 推理/部署
python eval_llm.py --weight full_sft --device cuda:0
python scripts/serve_openai_api.py
```

### 5.4 检查点命名规则

所有权重保存在 `out/` 目录下，命名格式统一：

```
out/
├── pretrain_512.pth          # 预训练权重
├── full_sft_512.pth          # SFT 权重
├── dpo_512.pth               # DPO 对齐权重
├── ppo_actor_512.pth         # PPO 对齐权重
├── grpo_512.pth              # GRPO 对齐权重
├── spo_512.pth               # SPO 对齐权重
├── full_dist_512.pth         # 蒸馏权重
├── reason_512.pth            # 推理训练权重
├── full_sft_768.pth          # 大模型的权重（hidden_size=768）
├── full_sft_768_moe.pth      # MoE 模型权重（带 _moe 后缀）
└── lora/
    └── lora_identity_512.pth # LoRA 权重（独立保存）
```

格式为 `{阶段名}_{hidden_size}[_moe].pth`，简洁明了。

---

## 小结

| 技术 | 核心价值 | 一句话总结 |
|------|----------|------------|
| LoRA | 高效微调 | 冻结 99.5% 参数，只训练低秩旁路 |
| 知识蒸馏 | 模型压缩 | 大模型的概率分布包含"暗知识"，小模型能学到更多 |
| 推理训练 | 思维链能力 | 用特殊标签和加权损失，教模型先想后答 |
| 部署 | 落地应用 | 命令行交互 + OpenAI 兼容 API，即插即用 |

到这里，MiniMind 的全部核心技术就介绍完了。从分词器到模型架构，从预训练到对齐，从 LoRA 到推理训练——整个 LLM 的"小宇宙"都在这几千行 PyTorch 代码里。

希望这套教程能帮你**真正理解** LLM 背后的原理，而不只是会调 API。

---

[上一章：对齐训练](06-alignment.md) | [返回目录](README.md)
