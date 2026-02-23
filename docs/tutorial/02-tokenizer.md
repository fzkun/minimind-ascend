# 02 - 分词器（Tokenizer）

> [上一章：LLM 入门](01-introduction.md) | [下一章：模型架构](03-model-architecture.md)

在上一章中，我们了解了大语言模型的核心思想：**预测下一个词**。但这里有一个根本性的问题——计算机不认识"文字"，它只能处理数字。那么，文字是如何变成数字的？这就是**分词器（Tokenizer）**要解决的问题。

---

## 1. 为什么需要分词器

### 计算机只懂数字

无论是 GPU 上的矩阵乘法，还是神经网络中的权重更新，底层操作的都是数字（准确说是浮点数和整数）。我们输入给模型的"你好，世界"，必须先转换成一串数字，模型才能处理。

分词器就是这座桥梁：

```
文本 → 分词器编码（encode）→ Token ID 序列 → 模型处理 → Token ID 序列 → 分词器解码（decode）→ 文本
```

举一个直觉性的例子：

| 步骤 | 内容 |
|------|------|
| 原始文本 | `"机器学习很有趣"` |
| 分词 | `["机器", "学习", "很", "有趣"]` |
| 编码为 ID | `[1024, 2048, 356, 4721]`（示意数字） |
| 模型处理 | 对数字序列做计算，输出下一个 token 的概率分布 |
| 解码回文本 | 将模型输出的 ID 转换回人类可读的文字 |

### 为什么不直接按字符编码？

最简单的方案是把每个字符映射成一个数字（类似 ASCII）。但这样做有两个问题：

1. **序列太长**：一句话可能包含几十个字符，但只有几个"词"。序列越长，模型的计算量和内存消耗越大（Transformer 的注意力机制是 O(n^2) 复杂度）。
2. **没有语义单元**："学"和"习"单独出现时几乎没有意义，"学习"作为一个整体才有含义。

反过来，如果把每个词都当成一个 token（像英文那样按空格分词），词表会非常庞大（几十万甚至上百万），模型的 Embedding 层也会变得巨大。

所以，现代 LLM 普遍采用一种**折中方案**：**子词分词（Subword Tokenization）**。它的粒度介于字符和词之间——常见的词保持完整，罕见的词拆分成更小的片段。

---

## 2. BPE 分词算法

MiniMind 使用的分词算法是 **BPE（Byte Pair Encoding，字节对编码）**。这也是 GPT 系列、LLaMA 等主流模型采用的算法。

### 核心思想：从字符级别逐步合并高频对

BPE 的训练过程非常直觉。我们用一个简单的例子来说明。

**假设我们的训练文本是**：`"aabaabaab"`

#### 第 0 步：从字符级别开始

把文本拆成最小单位——字符：

```
a a b a a b a a b
```

初始词表：`{a, b}`（共 2 个 token）

#### 第 1 步：统计所有相邻字符对的出现频率

```
(a, a) → 出现 3 次
(a, b) → 出现 3 次
(b, a) → 出现 2 次
```

最高频的是 `(a, a)`，出现 3 次。**合并它**，创建新 token `aa`：

```
aa b aa b aa b
```

词表更新：`{a, b, aa}`（共 3 个 token）

#### 第 2 步：再次统计

```
(aa, b) → 出现 3 次
(b, aa) → 出现 2 次
```

最高频是 `(aa, b)`，合并为 `aab`：

```
aab aab aab
```

词表更新：`{a, b, aa, aab}`（共 4 个 token）

#### 重复，直到词表达到目标大小

每一轮合并都会产生一个新 token。我们可以预先设定词表大小（例如 6400），算法会持续合并直到达到目标。

### BPE 的优势

| 特性 | 说明 |
|------|------|
| **无未知词** | 任何文本都可以分词——最坏情况下退化为字符/字节级别 |
| **高频词保持完整** | "的"、"the" 等常见词作为单个 token |
| **罕见词优雅拆分** | "transformer" 可能被拆成 "trans" + "former" |
| **可控词表大小** | 通过设定合并轮数来精确控制 |

### 在 MiniMind 中的体现

打开 `model/tokenizer.json`，可以看到两个关键部分：

```json
// model/tokenizer.json（部分）
{
  "model": {
    "type": "BPE",
    "vocab": {
      "<|endoftext|>": 0,
      "<|im_start|>": 1,
      "<|im_end|>": 2,
      "!": 3,
      "\"": 4,
      ...
      "Ġnetworks": 6399
    },
    "merges": [
      ["Ġ", "t"],
      ["Ġ", "a"],
      ["i", "n"],
      ["h", "e"],
      ...
    ]
  }
}
```

- **`vocab`**：最终的词表，共 6400 个 token（ID 0 到 6399）
- **`merges`**：BPE 训练过程中产生的合并规则列表，按优先级排序

> **注意**：词表中的 `Ġ` 符号代表空格。这是 Byte-Level BPE 的编码方式——空格被映射为特殊字符 `Ġ`，这样空格也能参与子词合并，不再是硬性的分词边界。

---

## 3. MiniMind 的分词器实战

### 3.1 词表大小：6400

MiniMind 的词表只有 **6400** 个 token。相比之下：

| 模型 | 词表大小 |
|------|----------|
| GPT-2 | 50,257 |
| LLaMA-2 | 32,000 |
| Qwen-2 | 151,643 |
| **MiniMind** | **6,400** |

为什么这么小？因为 MiniMind 是一个教学用的轻量模型。更小的词表意味着：
- Embedding 层参数更少（`vocab_size * hidden_size = 6400 * 512 = 3.3M` 参数）
- 输出层（lm_head）参数也同比缩小
- 训练更快，模型更小

代价是同样的文本需要更多的 token 来表示（分词粒度更细），但对于一个 26M 参数的教学模型来说，这是合理的取舍。

### 3.2 三个特殊 Token

MiniMind 定义了 3 个特殊 token，它们占据词表最前面的位置：

| Token ID | Token 文本 | 配置中的角色 | 作用 |
|----------|-----------|-------------|------|
| 0 | `<\|endoftext\|>` | `pad_token`、`unk_token` | **填充 token**：将不等长的序列填充到统一长度；同时兼任未知 token |
| 1 | `<\|im_start\|>` | `bos_token` | **序列开始标记**：标记一轮对话中某个角色发言的开始 |
| 2 | `<\|im_end\|>` | `eos_token` | **序列结束标记**：标记一轮发言的结束，也是模型生成时的停止信号 |

这些定义来自 `model/tokenizer_config.json`：

```json
// model/tokenizer_config.json（关键字段摘录）
{
  "bos_token": "<|im_start|>",
  "eos_token": "<|im_end|>",
  "pad_token": "<|endoftext|>",
  "unk_token": "<|endoftext|>",
  "tokenizer_class": "PreTrainedTokenizerFast",
  "model_max_length": 32768,
  "add_bos_token": false,
  "add_eos_token": false
}
```

几个值得注意的细节：

- **`add_bos_token: false`、`add_eos_token: false`**：分词器默认不会自动添加 BOS/EOS token。这意味着在代码中需要手动添加（后面会看到 `PretrainDataset` 就是这样做的）。
- **`pad_token` 和 `unk_token` 共用同一个 token**：ID 0 的 `<|endoftext|>` 身兼二职。这是一种常见做法——在 MiniMind 的小词表下，遇到 "未知" token 的概率较低（BPE 可以回退到字节级），所以用同一个 token 即可。
- **`model_max_length: 32768`**：分词器声明模型最大支持 32768 个 token（配合 YaRN 位置编码外推）。

### 3.3 编码与解码示例

以下是一个概念性的示例，展示分词器的工作过程：

```
输入文本: "你好"

编码过程（tokenizer.encode）:
  "你好" → [4513, 2898]     # 两个 token ID

解码过程（tokenizer.decode）:
  [4513, 2898] → "你好"     # 还原回文本
```

对于预训练数据，还需要手动加上特殊 token：

```
原始 token: [4513, 2898]
加上 BOS:   [1, 4513, 2898]           # 1 = <|im_start|>
加上 EOS:   [1, 4513, 2898, 2]        # 2 = <|im_end|>
填充到固定长度: [1, 4513, 2898, 2, 0, 0, 0, ...]  # 0 = <|endoftext|> (PAD)
```

---

## 4. Chat Template：多轮对话的格式化

### 4.1 问题：模型只看到一个 token 序列

LLM 在训练和推理时，输入都是一个**扁平的 token 序列**。但人类的对话是有结构的——谁说了什么、先后顺序、系统指令等等。我们需要一种**格式约定**，把结构化的对话"压平"成模型能理解的文本。

这就是 **Chat Template** 的作用。

### 4.2 MiniMind 的对话格式

MiniMind 采用的是 **ChatML** 格式（与 Qwen 系列一致），结构如下：

```
<|im_start|>system
你是一个有用的助手<|im_end|>
<|im_start|>user
什么是机器学习？<|im_end|>
<|im_start|>assistant
机器学习是人工智能的一个分支...<|im_end|>
```

解析这个格式：

| 片段 | 含义 |
|------|------|
| `<\|im_start\|>system\n` | 系统消息的开头标记 |
| `你是一个有用的助手` | 系统消息的内容 |
| `<\|im_end\|>\n` | 系统消息的结束标记 |
| `<\|im_start\|>user\n` | 用户消息的开头标记 |
| `什么是机器学习？` | 用户的问题 |
| `<\|im_end\|>\n` | 用户消息的结束标记 |
| `<\|im_start\|>assistant\n` | 助手回复的开头标记 |
| `机器学习是人工智能的一个分支...` | 助手的回复内容 |
| `<\|im_end\|>\n` | 助手回复的结束标记 |

每一轮对话都被 `<|im_start|>角色名\n...内容...<|im_end|>\n` 包裹。这样模型就能从 token 序列中分辨出"谁在说话"以及"每段话的边界"。

### 4.3 多轮对话示例

两轮对话在 Chat Template 下会被展开为：

```
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
1+1等于几？<|im_end|>
<|im_start|>assistant
1+1等于2。<|im_end|>
<|im_start|>user
那2+2呢？<|im_end|>
<|im_start|>assistant
2+2等于4。<|im_end|>
```

整段文本会被分词器编码成一个 token ID 序列，送入模型。模型在训练时只需要预测 **assistant 回复部分**的 token，其他部分（system、user）的 loss 会被掩码掉。这一点我们在第 05 章（监督微调）会详细讨论。

### 4.4 Chat Template 的定义

Chat Template 以 [Jinja2 模板](https://jinja.palletsprojects.com/) 的形式保存在 `model/tokenizer_config.json` 的 `chat_template` 字段中。虽然模板本身的语法看起来有些复杂，但它的核心逻辑是：

```
对于每条消息 message:
    如果是 system 或 user:
        输出 "<|im_start|>" + role + "\n" + content + "<|im_end|>\n"
    如果是 assistant:
        输出 "<|im_start|>assistant\n" + content + "<|im_end|>\n"

如果 add_generation_prompt 为 True:
    在末尾追加 "<|im_start|>assistant\n"  （提示模型开始生成）
```

在代码中，我们不需要手动拼接这些格式字符串。Hugging Face 的 `tokenizer.apply_chat_template()` 方法会自动应用模板：

```python
# 输入：结构化的消息列表
messages = [
    {"role": "system", "content": "你是一个有用的助手"},
    {"role": "user", "content": "什么是机器学习？"},
    {"role": "assistant", "content": "机器学习是人工智能的一个分支..."}
]

# 输出：格式化后的纯文本字符串
text = tokenizer.apply_chat_template(messages, tokenize=False)
# text = "<|im_start|>system\n你是一个有用的助手<|im_end|>\n<|im_start|>user\n什么是机器学习？<|im_end|>\n<|im_start|>assistant\n机器学习是人工智能的一个分支...<|im_end|>\n"
```

---

## 5. 对照源码

现在我们来看分词器在 MiniMind 训练代码中的实际使用方式。

### 5.1 预训练阶段：`PretrainDataset`

```python
# dataset/lm_dataset.py 第 31-49 行
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __getitem__(self, index):
        sample = self.samples[index]
        # 第 1 步：编码文本，不自动添加特殊 token
        tokens = self.tokenizer(
            str(sample['text']),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True
        ).input_ids

        # 第 2 步：手动添加 BOS（id=1）和 EOS（id=2）
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        # 第 3 步：用 PAD（id=0）填充到固定长度
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # 第 4 步：构建标签，PAD 位置设为 -100（PyTorch 会忽略这些位置的损失）
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return input_ids, labels
```

逐步拆解这段代码中分词器的使用：

| 步骤 | 代码 | 说明 |
|------|------|------|
| 编码 | `self.tokenizer(text, add_special_tokens=False, ...)` | 把文本变成 token ID 列表。`add_special_tokens=False` 表示不自动加 BOS/EOS |
| 截断 | `max_length=self.max_length - 2, truncation=True` | 预留 2 个位置给 BOS 和 EOS |
| 加 BOS/EOS | `[bos_token_id] + tokens + [eos_token_id]` | 手动在序列首尾添加特殊 token |
| 填充 | `+ [pad_token_id] * (max_length - len(tokens))` | 用 PAD token 将所有样本对齐到相同长度 |
| 损失掩码 | `labels[input_ids == pad_token_id] = -100` | PAD 位置不计算损失（-100 是 PyTorch CrossEntropyLoss 的忽略值） |

假设 `max_length=10`，一条文本编码后得到 `[4513, 2898, 1567]`，则最终的数据是：

```
input_ids: [1, 4513, 2898, 1567, 2, 0, 0, 0, 0, 0]
            │   ──实际内容──   │  ───PAD填充───
           BOS               EOS

labels:    [1, 4513, 2898, 1567, 2, -100, -100, -100, -100, -100]
                                     ─────不计算损失─────
```

### 5.2 SFT 阶段：`SFTDataset`

SFT（监督微调）阶段需要处理多轮对话数据。这时就要用到 Chat Template：

```python
# dataset/lm_dataset.py 第 64-72 行
def create_chat_prompt(self, conversations):
    messages = conversations.copy()
    # 检查是否包含 function calling 的工具定义
    tools = conversations[0]["functions"] if (
        conversations and
        conversations[0]["role"] == "system" and
        conversations[0].get("functions")
    ) else None

    # 调用 Chat Template，把消息列表转为格式化文本
    return self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,             # 返回字符串，不返回 token ID
        add_generation_prompt=False, # 不在末尾加 "<|im_start|>assistant\n"
        tools=tools                  # 可选：传入工具定义
    )
```

这里的关键是 `apply_chat_template` 方法：

- **`tokenize=False`**：只返回格式化后的文本字符串，不直接编码成 token ID。后续代码会单独调用 `self.tokenizer(prompt)` 来编码。
- **`add_generation_prompt=False`**：SFT 训练时，数据中已经包含了 assistant 的回复，不需要在末尾追加生成提示。
- **`tools`**：如果对话数据中包含函数调用的定义，会传入 Chat Template 以生成包含工具说明的系统提示。

### 5.3 在 SFTDataset.__getitem__ 中的完整流程

```python
# dataset/lm_dataset.py 第 92-105 行
def __getitem__(self, index):
    sample = self.samples[index]
    # 1. 可选地添加随机系统提示
    conversations = pre_processing_chat(sample['conversations'])
    # 2. 用 Chat Template 格式化
    prompt = self.create_chat_prompt(conversations)
    # 3. 后处理（处理空 think 标签）
    prompt = post_processing_chat(prompt)
    # 4. 编码并截断
    input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
    # 5. 填充
    input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
    # 6. 生成标签（只在 assistant 回复部分计算损失）
    labels = self.generate_labels(input_ids)
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
```

注意第 4 步：SFT 阶段直接调用 `self.tokenizer(prompt)`，没有传 `add_special_tokens=False`。因为 Chat Template 已经在文本中包含了 `<|im_start|>` 和 `<|im_end|>` 等特殊 token，分词器会直接将它们识别为特殊 token 并编码为对应的 ID。

### 5.4 `generate_labels`：选择性损失掩码

```python
# dataset/lm_dataset.py 第 74-90 行
def generate_labels(self, input_ids):
    labels = [-100] * len(input_ids)  # 默认全部忽略
    i = 0
    while i < len(input_ids):
        # 找到 "<|im_start|>assistant\n" 的位置
        if input_ids[i:i + len(self.bos_id)] == self.bos_id:
            start = i + len(self.bos_id)
            end = start
            # 找到对应的 "<|im_end|>\n"
            while end < len(input_ids):
                if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                    break
                end += 1
            # 只在 assistant 的回复内容 + EOS 上设置标签
            for j in range(start, min(end + len(self.eos_id), self.max_length)):
                labels[j] = input_ids[j]
            i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return labels
```

这段代码扫描 token 序列，只在 `<|im_start|>assistant\n` 和 `<|im_end|>\n` 之间的部分（即 assistant 的回复）设置有效标签。其余位置保持 `-100`，不参与损失计算。

用一个示意图来表示：

```
<|im_start|>system\n你是助手<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n你好！很高兴见到你<|im_end|>\n
[──────────────── labels = -100（忽略）────────────────────][──── labels = 真实 ID（计算损失）────]
```

这样做的目的是：**模型只学习"如何回复"，不学习"如何提问"**。

---

## 6. 小结

本章我们学习了分词器的核心概念和在 MiniMind 中的具体实现：

| 主题 | 要点 |
|------|------|
| **分词器的作用** | 将文本转换为数字序列（Token ID），让模型能够处理 |
| **BPE 算法** | 从字符级别出发，反复合并最高频的相邻对，直到词表达到目标大小 |
| **MiniMind 词表** | 6400 个 token，基于 Byte-Level BPE，3 个特殊 token |
| **特殊 Token** | ID 0 = PAD/UNK，ID 1 = BOS（`<\|im_start\|>`），ID 2 = EOS（`<\|im_end\|>`） |
| **Chat Template** | 将结构化的多轮对话格式化为模型可理解的 ChatML 格式文本 |
| **代码实践** | `PretrainDataset` 手动添加 BOS/EOS + 填充；`SFTDataset` 通过 Chat Template 格式化后只对 assistant 部分计算损失 |

### 涉及的源码文件

| 文件 | 作用 |
|------|------|
| `model/tokenizer.json` | BPE 词表和合并规则 |
| `model/tokenizer_config.json` | 特殊 token 定义、Chat Template 模板 |
| `dataset/lm_dataset.py` | 数据集类中分词器的实际调用方式 |

---

> [上一章：LLM 入门](01-introduction.md) | [下一章：模型架构](03-model-architecture.md)
