# 05 - 监督微调（Supervised Fine-Tuning）

> [上一章：预训练](04-pretrain.md) | [下一章：对齐训练](06-alignment.md)

预训练完成后，模型已经"会说话"了，但它还不会"好好回答问题"。本章我们学习如何通过监督微调（SFT），让模型学会按照指令进行对话。

---

## 1. 预训练 vs SFT 的区别

预训练和 SFT 做的事情截然不同，虽然底层都是"预测下一个 token"：

| 维度 | 预训练（Pretrain） | 监督微调（SFT） |
|------|-------------------|-----------------|
| 目标 | 学语言 | 学对话 |
| 数据 | 大量无标注文本 | 高质量对话数据 |
| 学到什么 | 语言的统计规律、世界知识 | 如何按照指令格式回答问题 |
| 学习率 | 较大（5e-4） | 很小（1e-6） |
| 起点 | 随机初始化 | 加载预训练权重 |

用一个直观的类比来理解：

```
预训练 = 小孩学说话
    - 听大量语言输入（爸爸妈妈说话、电视、绘本……）
    - 学会语言的基本规律：语法、词汇、常识
    - 结果：能说出通顺的句子，但不知道该怎么"回答问题"

SFT = 上学后学会回答老师的问题
    - 老师提问，学生回答，老师纠正
    - 学会"问→答"的模式：理解问题意图，组织有条理的回答
    - 结果：给定一个问题，能给出有用的回答
```

如果你直接拿预训练模型对话，它往往会"续写"而不是"回答"。比如你输入"什么是机器学习？"，预训练模型可能会续写成"什么是机器学习？什么是深度学习？什么是……"，而不是给你一个回答。SFT 就是教它学会"当用户问问题时，我应该给出回答"这个模式。

---

## 2. 对话数据格式

### 2.1 JSONL 原始格式

SFT 使用的训练数据是 JSONL 格式，每行一个 JSON 对象，包含 `conversations` 字段，其中记录了多轮对话：

```json
{
  "conversations": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮你的吗？"}
  ]
}
```

多轮对话的例子：

```json
{
  "conversations": [
    {"role": "user", "content": "帮我写一首关于春天的诗"},
    {"role": "assistant", "content": "春风拂柳绿如烟，细雨润花红满天。"},
    {"role": "user", "content": "能再加两句吗？"},
    {"role": "assistant", "content": "莺啼燕舞桃源里，人间四月好流年。"}
  ]
}
```

### 2.2 Chat Template 格式化

原始的 JSON 对话不能直接喂给模型。模型需要的是一串连续的 token，所以我们需要用 **Chat Template** 把对话格式化成一个带有特殊标记的字符串。

MiniMind 使用的 Chat Template 会将上面的对话转换成：

```
<s>user
你好</s>
<s>assistant
你好！有什么可以帮你的吗？</s>
```

其中：
- `<s>` 是 BOS（Begin of Sentence）标记，表示一个角色发言的开始
- `</s>` 是 EOS（End of Sentence）标记，表示发言结束
- `user` 和 `assistant` 标识说话的角色

这种格式化由 tokenizer 的 `apply_chat_template` 方法自动完成：

```python
# dataset/lm_dataset.py 第 64-72 行 (SFTDataset.create_chat_prompt)
def create_chat_prompt(self, conversations):
    messages = conversations.copy()
    tools = conversations[0]["functions"] if (
        conversations and conversations[0]["role"] == "system"
        and conversations[0].get("functions")
    ) else None
    return self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools
    )
```

---

## 3. 选择性损失掩码（核心概念）

这是 SFT 与预训练**最关键的区别**。

### 3.1 为什么需要掩码？

在预训练阶段，每一个 token 都参与损失计算，因为我们想让模型学会预测任意位置的下一个词。

但在 SFT 阶段，情况不同了：

- **用户的问题**是"已知条件"，我们不需要模型去学如何"生成"用户的问题
- **助手的回答**才是模型需要学习的部分

因此，SFT 只在 **assistant 回答部分** 计算损失，用户问题部分的标签被设为 `-100`（PyTorch 中 `CrossEntropyLoss` 会自动忽略标签为 -100 的位置）。

### 3.2 掩码效果示意

用一个 ASCII 图来展示哪些 token 计算损失、哪些被掩码：

```
Token序列:
┌─────┬──────┬────┬─────┬─────┬──────────┬────┬─────┬─────────┬──────┬──────────────┬──────┬─────┐
│ <s> │ user │ \n │  你 │  好 │   </s>   │ \n │ <s> │assistant│  \n  │ 你好！有什么 │</s>  │ \n  │
│     │      │    │     │     │          │    │     │         │      │ 可以帮你的吗？│      │     │
└─────┴──────┴────┴─────┴─────┴──────────┴────┴─────┴─────────┴──────┴──────────────┴──────┴─────┘

Labels:
┌─────┬──────┬────┬─────┬─────┬──────────┬────┬─────┬─────────┬──────┬──────────────┬──────┬─────┐
│-100 │ -100 │-100│-100 │-100 │   -100   │-100│-100 │  -100   │ -100 │  计算损失!   │计算! │计算!│
└─────┴──────┴────┴─────┴─────┴──────────┴────┴─────┴─────────┴──────┴──────────────┴──────┴─────┘
  ^                                                                    ^
  |____ 用户问题部分：全部掩码 (-100)                                    |____ 助手回答部分：设为真实标签
        模型不需要学"怎么提问"                                                  模型只学"怎么回答"
```

### 3.3 代码实现

掩码的核心逻辑在 `SFTDataset` 中。首先，在初始化时定义了 assistant 回答的起止标记：

```python
# dataset/lm_dataset.py 第 58-59 行 (SFTDataset.__init__)
self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
```

- `bos_id` 对应 `<s>assistant\n` 这段 token 序列 -- assistant 回答的开始标志
- `eos_id` 对应 `</s>\n` 这段 token 序列 -- assistant 回答的结束标志

然后，`generate_labels` 方法扫描整个 input_ids 序列，只在 `bos_id` 和 `eos_id` 之间的区域设置真实标签：

```python
# dataset/lm_dataset.py 第 74-90 行 (SFTDataset.generate_labels)
def generate_labels(self, input_ids):
    labels = [-100] * len(input_ids)  # 默认全部不计算损失
    i = 0
    while i < len(input_ids):
        if input_ids[i:i + len(self.bos_id)] == self.bos_id:
            start = i + len(self.bos_id)       # 跳过 "<s>assistant\n" 本身
            end = start
            while end < len(input_ids):
                if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                    break                       # 找到 "</s>\n"，回答结束
                end += 1
            # 只有 assistant 回答部分设为真实标签
            for j in range(start, min(end + len(self.eos_id), self.max_length)):
                labels[j] = input_ids[j]
            i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return labels
```

算法流程可以用这个图来理解：

```
input_ids: [... user问题 ... <s>assistant\n  回答内容  </s>\n ... user问题 ... <s>assistant\n  回答内容  </s>\n ...]
                              ^-- bos_id      ^start     ^-- eos_id
                              匹配到 bos_id    从这里开始   匹配到 eos_id
                              ，跳过它本身      设置 labels  ，包含它本身

labels:    [-100 ... -100 ... -100 -100 -100  回答内容     </s>\n -100 ... -100 -100 -100  回答内容  </s>\n ...]
```

关键细节：
- `start = i + len(self.bos_id)` -- 跳过 `<s>assistant\n` 标记本身，不让模型去"学习"如何生成这段标记
- `end + len(self.eos_id)` -- 包含 `</s>\n` 结束标记，让模型学会在何时停止生成
- 多轮对话中，每个 assistant 回答段都会被独立标记

### 3.4 多轮对话的掩码

对于多轮对话，掩码逻辑自然地处理了多个回答段：

```
Token:  <s>user\n 第一个问题 </s>\n <s>assistant\n 第一个回答 </s>\n <s>user\n 第二个问题 </s>\n <s>assistant\n 第二个回答 </s>\n
Label:  --------- -100 ----------- -------------- 计算损失!  ------- --------- -100 ----------- -------------- 计算损失!  -------
```

代码中的 `while` 循环会反复扫描，找到所有的 `bos_id...eos_id` 区间，确保每一轮 assistant 的回答都被标记为需要计算损失。

---

## 4. 随机添加 System Prompt

在实际使用中，对话通常会包含一个 system prompt（系统提示），用来设定 AI 助手的角色和行为。但训练数据中不是每条都有 system prompt。

MiniMind 采取了一个巧妙的策略：以 **20% 的概率** 随机给对话添加一个 system prompt：

```python
# dataset/lm_dataset.py 第 8-24 行 (pre_processing_chat)
def pre_processing_chat(conversations, add_system_ratio=0.2):
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    if conversations and conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations
```

这个设计有几个好处：

1. **让模型适应有/无 system prompt 的场景** -- 推理时用户可能给 system prompt，也可能不给，模型都要能正常工作
2. **中英文混合的系统提示** -- 提升模型对双语指令的鲁棒性
3. **概率控制** -- 20% 的概率不会让 system prompt 占据太多训练信号，避免模型过度依赖 system prompt

加了 system prompt 后，Chat Template 格式化的结果变成：

```
<s>system
你是minimind，一个小巧但有用的语言模型。</s>
<s>user
你好</s>
<s>assistant
你好！有什么可以帮你的吗？</s>
```

注意：system prompt 部分同样会被损失掩码标记为 `-100`，不参与损失计算。模型只需要"理解"system prompt 的内容，而不需要"学会生成"它。

---

## 5. 学习率差异

预训练和 SFT 的学习率差了 **500 倍**：

| 阶段 | 默认学习率 | 数量级 |
|------|-----------|--------|
| 预训练 | `5e-4`（0.0005） | 较大 |
| SFT | `1e-6`（0.000001） | 很小 |

为什么 SFT 的学习率要这么小？

### 5.1 灾难性遗忘（Catastrophic Forgetting）

预训练阶段花了大量时间让模型学会语言能力和世界知识，这些知识都编码在模型参数中。如果 SFT 阶段用太大的学习率，参数会发生剧烈变化，导致预训练学到的知识被覆盖 -- 这就是"灾难性遗忘"。

```
学习率太大:
  预训练知识: ████████████ (100%)
  SFT 后:    ██░░░░░░░░░░ (20%)  <-- 知识被大量破坏!

学习率合适:
  预训练知识: ████████████ (100%)
  SFT 后:    ██████████░░ (85%)  <-- 保留大部分知识，同时学会了对话
```

### 5.2 站在巨人的肩膀上

SFT 不需要"学新知识"（知识已经在预训练中学到了），只需要学会一个"格式"：把已有的知识用问答的方式表达出来。这是一个相对小的调整，所以小学习率就足够了。

打个比方：预训练是让一个人学会中文和各种知识，SFT 是教这个人"考试的答题格式"。学答题格式不需要重新学知识，只需要微调表达方式。

### 5.3 从代码中看学习率

预训练脚本 `train_pretrain.py` 第 79 行：

```python
parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
```

SFT 脚本 `train_full_sft.py` 第 80 行：

```python
parser.add_argument("--learning_rate", type=float, default=1e-6, help="初始学习率")
```

两个脚本都使用余弦退火学习率调度（cosine annealing），初始值不同但调度策略一致：

```python
# trainer/trainer_utils.py 第 48-49 行 (get_lr)
def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
```

---

## 6. 从预训练权重开始 SFT

SFT 不是从零开始的，它必须加载预训练阶段保存的权重。这通过 `--from_weight` 参数控制。

### 6.1 参数默认值对比

```python
# train_pretrain.py: 从零开始，不加载任何权重
parser.add_argument('--from_weight', default='none', type=str)

# train_full_sft.py: 默认加载预训练权重
parser.add_argument('--from_weight', default='pretrain', type=str)
```

### 6.2 权重加载过程

`init_model` 函数负责加载权重：

```python
# trainer/trainer_utils.py 第 139-151 行 (init_model)
def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model',
               save_dir='../out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight != 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer
```

当 `from_weight='pretrain'` 时，函数会从 `out/pretrain_512.pth` 加载预训练权重。`strict=False` 表示允许部分键不匹配（例如模型架构有微小变化时不会报错）。

### 6.3 整个权重传递链

```
预训练 (train_pretrain.py)
  │
  │  保存: out/pretrain_512.pth
  │
  ▼
SFT (train_full_sft.py)         --from_weight pretrain
  │
  │  加载: out/pretrain_512.pth
  │  保存: out/full_sft_512.pth
  │
  ▼
对齐训练 (train_dpo.py 等)      --from_weight full_sft
  │
  │  加载: out/full_sft_512.pth
  │  保存: out/dpo_512.pth
  ▼
```

每个阶段都站在上一个阶段的基础上继续训练，形成完整的训练流水线。

---

## 7. 实操

### 7.1 训练命令

确保已经完成预训练（`out/pretrain_512.pth` 存在），然后运行：

```bash
python trainer/train_full_sft.py --epochs 2 --batch_size 16
```

### 7.2 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--from_weight` | `pretrain` | 加载哪个阶段的权重（`pretrain` / `full_sft` / `none`） |
| `--save_weight` | `full_sft` | 保存权重的前缀名 |
| `--learning_rate` | `1e-6` | 初始学习率，SFT 阶段应保持很小 |
| `--data_path` | `dataset/sft_mini_512.jsonl` | SFT 训练数据路径 |
| `--epochs` | `2` | 训练轮数 |
| `--batch_size` | `16` | 批大小 |
| `--max_seq_len` | `340` | 最大序列长度（中文 1 token 约 1.5-1.7 字符） |
| `--hidden_size` | `512` | 隐藏层维度，需与预训练模型一致 |
| `--num_hidden_layers` | `8` | 隐藏层数量，需与预训练模型一致 |
| `--accumulation_steps` | `1` | 梯度累积步数 |
| `--use_moe` | `0` | 是否使用 MoE 架构（0=否，1=是） |
| `--from_resume` | `0` | 是否从断点续训（0=否，1=是） |

### 7.3 多卡训练

使用 DDP 进行多卡并行训练：

```bash
torchrun --nproc_per_node 4 trainer/train_full_sft.py --epochs 2 --batch_size 16
```

在 Ascend NPU 上训练：

```bash
bash scripts/run_train_npu.sh full_sft --epochs 2 --batch_size 16
```

### 7.4 验证 SFT 效果

训练完成后，使用推理脚本验证效果：

```bash
python eval_llm.py --weight full_sft --device cuda:0
```

你可以对比预训练模型和 SFT 模型的输出差异：

```bash
# 预训练模型：可能会续写，而不是回答
python eval_llm.py --weight pretrain --device cuda:0

# SFT 模型：应该会按照对话格式给出回答
python eval_llm.py --weight full_sft --device cuda:0
```

### 7.5 SFT 训练脚本的完整流程

SFT 脚本 `train_full_sft.py` 的执行流程（与预训练脚本结构一致）：

```
1. 解析参数
2. init_distributed_mode()        → 分布式环境初始化
3. 配置混合精度 (autocast + GradScaler)
4. init_model(from_weight='pretrain')  → 加载预训练权重  ★ 关键区别
5. 创建 SFTDataset                     → 对话数据 + 损失掩码  ★ 关键区别
6. 训练循环（梯度累积、日志、周期保存）
7. dist.destroy_process_group()   → 清理
```

与预训练脚本相比，核心区别只有两个（标记了 ★ 的部分）：
1. 加载预训练权重而非从头开始
2. 使用 `SFTDataset` 而非 `PretrainDataset`，数据处理中包含损失掩码

---

## 本章小结

| 概念 | 要点 |
|------|------|
| SFT 的目标 | 在预训练基础上，教模型学会"对话"格式 |
| 对话数据 | JSONL 格式，用 Chat Template 格式化成带特殊标记的字符串 |
| 选择性损失掩码 | 只在 assistant 回答部分计算损失，用户问题部分标签设为 -100 |
| System Prompt | 20% 概率随机添加，提升模型对不同场景的适应性 |
| 学习率 | 比预训练小 500 倍（1e-6 vs 5e-4），防止灾难性遗忘 |
| 权重加载 | 通过 `--from_weight pretrain` 加载预训练权重 |

> [上一章：预训练](04-pretrain.md) | [下一章：对齐训练](06-alignment.md)
