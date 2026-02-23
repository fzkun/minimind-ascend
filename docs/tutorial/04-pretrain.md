# 第四章：预训练（Pre-training）

> [上一章：模型架构](03-model-architecture.md) | [下一章：监督微调](05-sft.md)

---

## 1. 预训练的目标

预训练的核心目标只有一个：**让模型学会语言的统计规律**——给定前文，预测下一个词。

这个过程就像一个人读了一万本书之后，自然就知道"今天天气"后面大概率跟"不错"，而不是"椅子"。模型不需要任何人工标注，只需要大量的原始文本，通过海量阅读来内化语言的模式。

预训练完成后，模型就具备了：
- **语法感知**：知道"我吃了一个"后面应该跟名词
- **常识知识**：知道"水的沸点是"后面应该跟"100度"
- **逻辑连贯**：能生成通顺、有意义的句子

但这时模型还不会"对话"——它只会续写文本。要让它变成一个聊天助手，还需要后续的监督微调（SFT）和对齐训练。

---

## 2. 数据格式

### 2.1 原始数据

预训练数据是最简单的格式——每行一个 JSON 对象，只有一个 `text` 字段：

```jsonl
{"text": "长江是亚洲第一长河，发源于青藏高原唐古拉山脉..."}
{"text": "机器学习是人工智能的一个分支，它通过数据驱动的方式..."}
{"text": "Python是一种解释型、面向对象的编程语言..."}
```

### 2.2 数据集类

`PretrainDataset` 负责把原始文本转换成模型能消费的 token 序列。

```python
# dataset/lm_dataset.py (第 31-49 行)
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __getitem__(self, index):
        sample = self.samples[index]
        # 1) 分词，不加特殊 token，截断到 max_length - 2（留位置给 BOS/EOS）
        tokens = self.tokenizer(
            str(sample['text']),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True
        ).input_ids
        # 2) 头尾分别加上 BOS 和 EOS
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # 3) 右侧补 PAD 到固定长度
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # 4) labels = input_ids 的副本，PAD 位置标记为 -100（不计算损失）
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels
```

整个流程用 ASCII 图表示：

```
原始文本: "今天天气不错"

         分词
          |
          v
tokens: [今天, 天气, 不错]       （纯文本 token）

       加特殊标记
          |
          v
tokens: [BOS, 今天, 天气, 不错, EOS]

        补 PAD
          |
          v
input_ids: [BOS, 今天, 天气, 不错, EOS, PAD, PAD, PAD]
labels:    [BOS, 今天, 天气, 不错, EOS, -100, -100, -100]
                                          ^
                                   PAD 位置不计算损失
```

**为什么要加 BOS 和 EOS？**
- `BOS`（Begin Of Sequence）：告诉模型"一段新文本开始了"
- `EOS`（End Of Sequence）：告诉模型"文本到此结束"，模型学会在合适的时候停止生成

---

## 3. 因果语言建模损失（Causal LM Loss）

### 3.1 核心思想

因果语言建模的训练信号非常直觉：**把 input 向右移一位得到 label，让模型预测每个位置的下一个词**。

```
位置:       0      1      2      3      4
input:    [BOS]  [今天]  [天气]  [不错]  [EOS]
label:    [今天]  [天气]  [不错]  [EOS]    X

模型在位置 0 看到 [BOS]，要预测"今天"
模型在位置 1 看到 [今天]，要预测"天气"
模型在位置 2 看到 [天气]，要预测"不错"
模型在位置 3 看到 [不错]，要预测 [EOS]
```

注意：由于因果注意力掩码的存在，模型在每个位置只能看到当前位置及之前的 token，看不到未来的内容。这正是"因果"二字的含义——预测只能基于过去，不能偷看未来。

### 3.2 代码实现

```python
# model/model_minimind.py (第 461-464 行)
loss = None
if labels is not None:
    shift_logits = logits[..., :-1, :].contiguous()   # 去掉最后一个位置的预测
    shift_labels = labels[..., 1:].contiguous()        # 去掉第一个位置的标签
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),  # [batch*seq_len, vocab_size]
        shift_labels.view(-1),                          # [batch*seq_len]
        ignore_index=-100                               # 忽略 PAD 位置
    )
```

逐行解释：

| 操作 | 含义 |
|------|------|
| `logits[..., :-1, :]` | 取位置 0 到 N-2 的预测（因为位置 N-1 没有"下一个词"可预测） |
| `labels[..., 1:]` | 取位置 1 到 N-1 的标签（即每个位置的"下一个词"） |
| `cross_entropy` | 计算预测分布与真实标签之间的交叉熵损失 |
| `ignore_index=-100` | 标签为 -100 的位置不参与损失计算（PAD 位置） |

---

## 4. 混合精度训练（Mixed Precision Training）

### 4.1 为什么需要混合精度？

默认情况下，模型参数和计算都使用 float32（32 位浮点数）。混合精度训练的核心思想是：**前向传播和反向传播用半精度（float16/bfloat16）加速计算，但参数更新仍用 float32 保证精度**。

好处：
- **速度快**：半精度计算量减半，GPU tensor core 可以充分利用
- **显存省**：激活值占用减半，可以用更大的 batch size

### 4.2 autocast 上下文

```python
# trainer/train_pretrain.py (第 110-122 行)
if "npu" in args.device:
    device_type = "npu"
elif "cuda" in args.device:
    device_type = "cuda"
else:
    device_type = "cpu"
dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

if device_type == "cpu":
    autocast_ctx = nullcontext()                          # CPU 不支持混合精度
elif device_type == "npu":
    autocast_ctx = torch.npu.amp.autocast(dtype=dtype)    # NPU 专用
else:
    autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)   # CUDA 标准方式
```

`autocast` 会自动决定哪些算子用半精度、哪些保留全精度。例如矩阵乘法用半精度加速，而 loss 计算保留全精度防止精度损失。

### 4.3 GradScaler

```python
# trainer/train_pretrain.py (第 143-146 行)
if is_npu_available():
    scaler = torch.npu.amp.GradScaler(enabled=(args.dtype == 'float16'))
else:
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
```

**GradScaler 的作用**：float16 的表示范围很小（最小正数约 6e-8），反向传播时梯度很容易"下溢"变成 0。GradScaler 在反向传播前把 loss 放大（scale up），计算完梯度后再缩小回来，从而避免梯度下溢。

> 注意：`bfloat16` 的指数位与 float32 一样，不存在下溢问题，所以使用 bfloat16 时 GradScaler 是禁用的（`enabled=False`）。

---

## 5. 梯度累积（Gradient Accumulation）

### 5.1 为什么需要梯度累积？

大 batch size 能让训练更稳定，但显存是有限的。梯度累积的思路是：**连续做多次前向+反向，把梯度"攒起来"，再一次性更新参数**。效果等价于使用更大的 batch size。

```
假设 batch_size=32, accumulation_steps=8
有效 batch = 32 x 8 = 256
```

### 5.2 代码实现

```python
# trainer/train_pretrain.py (第 34-46 行)
with autocast_ctx:
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss
    loss = loss / args.accumulation_steps       # (1) 先除以累积步数，保证梯度量级正确

scaler.scale(loss).backward()                   # (2) 每步都做反向传播，梯度会自动累加

if (step + 1) % args.accumulation_steps == 0:   # (3) 每累积够 N 步，才真正更新一次
    scaler.unscale_(optimizer)                   #     还原梯度的真实量级
    torch.nn.utils.clip_grad_norm_(              #     裁剪梯度，防止梯度爆炸
        model.parameters(), args.grad_clip
    )
    scaler.step(optimizer)                       #     用优化器更新参数
    scaler.update()                              #     调整 scaler 的缩放因子
    optimizer.zero_grad(set_to_none=True)         #     清空梯度，开始下一轮累积
```

整个过程用图来理解：

```
step 1: forward -> backward -> 梯度累加到 .grad
step 2: forward -> backward -> 梯度继续累加
  ...
step 8: forward -> backward -> 梯度继续累加
         |
         v
    unscale -> clip_grad -> optimizer.step -> zero_grad
         |
         v
    参数更新完毕，开始下一个 8 步周期
```

> `loss / accumulation_steps` 这一步很关键：因为梯度是累加的，如果不除的话，8 步累积的梯度就是单步的 8 倍，相当于把学习率放大了 8 倍。除以 8 后，累积效果就等价于用 8 倍大的 batch 做一次正常训练。

---

## 6. 学习率调度（Learning Rate Schedule）

### 6.1 Warmup + Cosine 衰减

```python
# trainer/trainer_utils.py (第 48-49 行)
def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
```

这个公式实现了一个 **cosine 退火** 调度器。我们来拆解它的行为：

| 训练进度 (`current_step / total_steps`) | `cos(...)` 值 | 系数 `0.1 + 0.45 * (1 + cos)` | 实际学习率 |
|:---:|:---:|:---:|:---:|
| 0%（刚开始） | 1.0 | 0.1 + 0.45 * 2 = **1.0** | `lr` |
| 50%（中间） | 0.0 | 0.1 + 0.45 * 1 = **0.55** | `0.55 * lr` |
| 100%（结束） | -1.0 | 0.1 + 0.45 * 0 = **0.10** | `0.10 * lr` |

学习率变化曲线：

```
lr  |*
    | **
    |   ***
    |      ****
    |          *****
    |               *******
    |                      **********
0.1*lr |                                ***
    +-----------------------------------------> step
    0%               50%              100%
```

**设计意图**：
- 训练初期用较大的学习率快速探索参数空间
- 训练后期用较小的学习率精细调整，避免在最优解附近"跳来跳去"
- 最终衰减到初始值的 10%，而不是 0，避免训练末期完全停止学习

### 6.2 学习率的应用方式

```python
# trainer/train_pretrain.py (第 28-30 行)
lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
for param_group in optimizer.param_groups:
    param_group['lr'] = lr
```

每个 step 都手动计算并设置当前学习率，而不是使用 PyTorch 内置的 `lr_scheduler`。这种方式更灵活，也更直观。

---

## 7. DDP 分布式训练

### 7.1 数据并行的工作方式

DDP（DistributedDataParallel）是 PyTorch 的多卡数据并行方案。核心思想：

```
                    训练数据
                      |
           +----------+----------+
           |          |          |
         GPU 0      GPU 1     GPU 2      <-- 每张卡拿到不同的数据子集
           |          |          |
        前向传播    前向传播   前向传播
           |          |          |
        反向传播    反向传播   反向传播
           |          |          |
        梯度 g0     梯度 g1    梯度 g2
           |          |          |
           +-----AllReduce-------+        <-- 所有卡的梯度求平均
           |          |          |
     g_avg = (g0+g1+g2)/3    (各卡同步)
           |          |          |
        参数更新    参数更新   参数更新    <-- 所有卡用相同的平均梯度更新
```

**关键点**：每张卡持有完整的模型副本，处理不同的数据，梯度通过 AllReduce 同步后保证所有卡的参数始终一致。

### 7.2 初始化代码

```python
# trainer/trainer_utils.py (第 52-64 行)
def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非 DDP 模式，单卡训练

    if _NPU_AVAILABLE:
        dist.init_process_group(backend="hccl")     # 华为昇腾 NPU 用 HCCL 后端
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.npu.set_device(local_rank)
    else:
        dist.init_process_group(backend="nccl")      # NVIDIA GPU 用 NCCL 后端
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    return local_rank
```

| 通信后端 | 硬件 | 说明 |
|----------|------|------|
| NCCL | NVIDIA GPU | GPU 间高速通信的标准库 |
| HCCL | 华为昇腾 NPU | 昇腾芯片的集合通信库 |

### 7.3 DDP 包装模型

```python
# trainer/train_pretrain.py (第 158-161 行)
if dist.is_initialized():
    model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
    model = DistributedDataParallel(model, device_ids=[local_rank])
```

`freqs_cos` 和 `freqs_sin` 是 RoPE 位置编码的预计算值，所有卡上都一样，不需要同步，因此标记为忽略。

---

## 8. 断点续训机制（Checkpoint & Resume）

### 8.1 为什么需要断点续训？

预训练可能需要跑几个小时甚至几天。如果训练中途因为断电、OOM 或其他原因中断，从头开始代价太大。断点续训机制让训练可以从上次中断的位置继续。

### 8.2 保存 Checkpoint

```python
# trainer/trainer_utils.py (第 80-126 行) - lm_checkpoint 函数（保存模式）
# 当 model is not None 时执行保存

# 保存内容包括：
resume_data = {
    'model': state_dict,                      # 模型权重
    'optimizer': optimizer.state_dict(),       # 优化器状态（动量、自适应学习率等）
    'epoch': epoch,                            # 当前 epoch
    'step': step,                              # 当前 step
    'world_size': dist.get_world_size(),       # GPU 数量
    'wandb_id': wandb_id                       # wandb 实验 ID（用于续接日志）
}
```

MiniMind 会保存两种文件：

| 文件 | 路径 | 内容 | 用途 |
|------|------|------|------|
| 模型权重 | `out/{weight}_{hidden_size}.pth` | 仅模型 state_dict（半精度） | 推理和下游任务加载 |
| 续训快照 | `checkpoints/{weight}_{hidden_size}_resume.pth` | 模型 + 优化器 + 进度 | 断点续训 |

> 使用 `.tmp` + `os.replace` 的原子写入策略，即使写入过程中程序崩溃，也不会损坏已有的 checkpoint。

### 8.3 恢复 Checkpoint

```python
# trainer/trainer_utils.py (第 127-136 行) - lm_checkpoint 函数（加载模式）
# 当 model is None 时执行加载

if os.path.exists(resume_path):
    ckp_data = torch.load(resume_path, map_location='cpu')
    # 自动处理 GPU 数量变化
    saved_ws = ckp_data.get('world_size', 1)
    current_ws = dist.get_world_size() if dist.is_initialized() else 1
    if saved_ws != current_ws:
        ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
        Logger(f'GPU数量变化({saved_ws}->{current_ws})，step已自动转换为{ckp_data["step"]}')
    return ckp_data
```

**跨 GPU 数量续训**：假设原来用 4 卡训练到了 step 1000，现在改用 2 卡继续。由于卡数减半，每步处理的数据量也减半，所以需要把 step 数翻倍（1000 * 4 / 2 = 2000）才能跳过相同数量的数据。

### 8.4 SkipBatchSampler

恢复训练时，不能重复训练已经看过的数据。`SkipBatchSampler` 负责跳过已训练的 batch：

```python
# trainer/trainer_utils.py (第 154-177 行)
class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1           # 跳过前 N 个 batch
                    batch = []
                    continue
                yield batch                # 从第 N+1 个 batch 开始正常产出
                batch = []
```

```
假设上次训练到 step 500，恢复时：

batch 1   -> 跳过
batch 2   -> 跳过
  ...
batch 500 -> 跳过
batch 501 -> 开始正常训练  <-- 从这里继续
batch 502 -> 正常训练
  ...
```

---

## 9. 训练脚本结构（9 步模板）

MiniMind 的所有 9 个训练脚本（pretrain、full_sft、lora、dpo、ppo、grpo、spo、distillation、reason）都遵循相同的结构。以 `train_pretrain.py` 为例：

```python
# trainer/train_pretrain.py

# ========== 1. 初始化环境和随机种子 ==========
local_rank = init_distributed_mode()
setup_seed(42 + rank_offset)

# ========== 2. 配置目录、模型参数、检查 ckp ==========
os.makedirs(args.save_dir, exist_ok=True)
lm_config = MiniMindConfig(hidden_size=..., num_hidden_layers=..., use_moe=...)
ckp_data = lm_checkpoint(lm_config, weight=...) if args.from_resume else None

# ========== 3. 设置混合精度 ==========
autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)

# ========== 4. 配 wandb ==========
wandb.init(project=..., name=..., id=wandb_id, resume=...)

# ========== 5. 定义模型、数据、优化器 ==========
model, tokenizer = init_model(lm_config, args.from_weight)
train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
scaler = torch.cuda.amp.GradScaler(enabled=...)
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

# ========== 6. 从 ckp 恢复状态 ==========
if ckp_data:
    model.load_state_dict(ckp_data['model'])
    optimizer.load_state_dict(ckp_data['optimizer'])
    start_epoch, start_step = ckp_data['epoch'], ckp_data['step']

# ========== 7. DDP 包模型 ==========
if dist.is_initialized():
    model = DistributedDataParallel(model, device_ids=[local_rank])

# ========== 8. 开始训练 ==========
for epoch in range(start_epoch, args.epochs):
    train_epoch(epoch, loader, ...)

# ========== 9. 清理分布进程 ==========
if dist.is_initialized():
    dist.destroy_process_group()
```

**为什么统一结构很重要？** 当你理解了一个脚本的结构，就理解了所有 9 个脚本。它们的区别仅在于：数据集类不同、损失函数不同、训练循环内的逻辑不同。外围的环境搭建、断点续训、分布式设置都是完全一样的。

---

## 10. 实操指南

### 10.1 单卡训练

```bash
python trainer/train_pretrain.py --epochs 1 --batch_size 32
```

### 10.2 多卡训练（4 卡 DDP）

```bash
torchrun --nproc_per_node 4 trainer/train_pretrain.py --epochs 1 --batch_size 32
```

### 10.3 昇腾 NPU 训练

```bash
bash scripts/run_train_npu.sh pretrain --epochs 1 --batch_size 32
```

### 10.4 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 1 | 训练轮数，建议 1 轮快速验证或 2-6 轮充分训练 |
| `--batch_size` | 32 | 每张卡的 batch size |
| `--learning_rate` | 5e-4 | 初始学习率 |
| `--accumulation_steps` | 8 | 梯度累积步数，有效 batch = batch_size x accumulation_steps |
| `--grad_clip` | 1.0 | 梯度裁剪阈值，防止梯度爆炸 |
| `--dtype` | bfloat16 (CUDA) / float16 (NPU) | 混合精度类型 |
| `--hidden_size` | 512 | 模型隐藏层维度（对应 26M 参数量） |
| `--num_hidden_layers` | 8 | Transformer 层数 |
| `--max_seq_len` | 340 | 最大序列长度（中文 1 token 约 1.5-1.7 字符） |
| `--use_moe` | 0 | 是否使用 MoE 架构（0=否，1=是） |
| `--data_path` | `dataset/pretrain_hq.jsonl` | 预训练数据路径 |
| `--from_weight` | none | 基于哪个权重继续训练，none 则从头开始 |
| `--from_resume` | 0 | 是否启用断点续训（0=否，1=是） |
| `--save_interval` | 1000 | 每隔多少步保存一次 checkpoint |
| `--log_interval` | 100 | 每隔多少步打印一次日志 |
| `--use_wandb` | False | 是否启用 wandb 实验追踪 |
| `--use_compile` | 0 | 是否使用 torch.compile 加速（NPU 不支持） |

### 10.5 训练产出

训练完成后，你会在以下位置找到产出文件：

```
out/
  pretrain_512.pth          <-- 模型权重（用于推理和下游任务）

checkpoints/
  pretrain_512_resume.pth   <-- 续训快照（用于断点续训）
```

如果使用了 MoE 架构（`--use_moe 1`），文件名会带上 `_moe` 后缀：

```
out/
  pretrain_512_moe.pth
```

### 10.6 常见问题

**Q: 显存不够怎么办？**
- 减小 `--batch_size`（比如从 32 降到 16）
- 增大 `--accumulation_steps`（比如从 8 改为 16），保持有效 batch 不变
- 减小 `--max_seq_len`

**Q: 训练中断了怎么恢复？**
- 加上 `--from_resume 1` 参数重新启动训练脚本即可
- 程序会自动检测 `checkpoints/` 目录下的续训文件并恢复

**Q: 换了 GPU 数量能续训吗？**
- 可以。step 数会自动按 GPU 数量比例转换

---

## 小结

本章介绍了 MiniMind 预训练的完整流程：

1. **数据**：JSONL 格式的原始文本，经过分词、加特殊标记、补 PAD 后送入模型
2. **目标**：因果语言建模——预测每个位置的下一个词
3. **加速**：混合精度训练（autocast + GradScaler）
4. **大 batch**：梯度累积模拟更大的 batch size
5. **调度**：cosine 学习率衰减
6. **多卡**：DDP 数据并行（NCCL/HCCL）
7. **容错**：断点续训 + SkipBatchSampler 跳过已训练数据
8. **结构**：9 步模板，所有训练脚本通用

预训练完成后，模型已经学会了语言的基本规律，能够生成通顺的文本。但它还只是一个"续写机器"，不懂得如何与人对话。下一章，我们将通过监督微调（SFT）教会它如何当一个合格的聊天助手。

> [上一章：模型架构](03-model-architecture.md) | [下一章：监督微调](05-sft.md)
