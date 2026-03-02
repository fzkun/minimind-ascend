# Mac 本地复现指南

在 Mac（Apple Silicon M1/M2/M3/M4）上从零跑通 MiniMind 的完整流程。

## 1. 环境搭建

```bash
# 克隆项目
git clone http://10.8.9.81:3000/llm/minimind.git
cd minimind
git checkout trainer/mac

# 用 uv 创建虚拟环境（Python 3.11）
uv venv --python 3.11 .venv
source .venv/bin/activate

# 安装依赖（阿里云加速）
uv pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## 2. PyCharm 配置

1. 用 PyCharm 打开项目
2. `Settings → Project → Python Interpreter → Add Interpreter → Existing`
3. 选择 `.venv/bin/python`
4. 右上角运行配置会自动出现以下选项：
   - **`debug_pretrain`** — 预训练调试
   - **`debug_sft`** — SFT 微调调试
   - **`test_pretrain`** — 测试预训练模型（续写）
   - **`test_sft`** — 测试 SFT 模型（对话）

## 3. 预训练调试

### 命令行方式

```bash
# 在 trainer/ 目录下运行
cd trainer
python train_pretrain.py \
    --data_path ../dataset/pretrain_hq_debug.jsonl \
    --batch_size 2 \
    --accumulation_steps 1 \
    --num_workers 0 \
    --epochs 1 \
    --log_interval 1 \
    --save_interval 50 \
    --hidden_size 512 \
    --num_hidden_layers 2 \
    --max_seq_len 128 \
    --device cpu
```

### PyCharm 方式

右上角选 **`debug_pretrain`**，直接点 Debug 按钮即可打断点。

### 参数说明

| 参数 | 调试值 | 正式训练值 | 说明 |
|------|--------|-----------|------|
| `--data_path` | `pretrain_hq_debug.jsonl` | `pretrain_hq.jsonl` | 调试用 100 条，正式 140 万条 |
| `--batch_size` | 2 | 32 | 调试用极小 batch |
| `--accumulation_steps` | 1 | 8 | 调试不累积梯度 |
| `--num_workers` | 0 | 8 | **0 才能在 PyCharm 打断点** |
| `--num_hidden_layers` | 2 | 8 | 调试用 2 层，跑得快 |
| `--max_seq_len` | 128 | 340 | 调试缩短序列 |
| `--device` | cpu | cuda:0 | Mac 用 cpu |
| `--log_interval` | 1 | 100 | 调试每步都打印 |
| `--save_interval` | 50 | 1000 | 调试减少保存频率 |

## 4. 测试预训练模型

预训练模型只会**续写**（接着你的开头往下写），不会回答问题。

### 命令行方式

```bash
# 在项目根目录运行（自动从权重文件推断层数，不用手动指定）
python scripts/test_pretrain.py --device cpu

# 自定义续写开头
python scripts/test_pretrain.py --device cpu --prompt "从前有座山"

# 控制生成长度
python scripts/test_pretrain.py --device cpu --max_new_tokens 100
```

### PyCharm 方式

右上角选 **`test_pretrain`**，点运行即可。

### 输出示例

调试用的 2 层小模型 + 100 条数据，续写效果很差是正常的：

```
正在加载模型...
从权重自动检测到 2 层
已加载权重: out/pretrain_512.pth
模型参数量: 8.92M

============================================================
预训练模型续写测试（模型会接着你的开头往下写）
============================================================

输入: 中国的首都是
续写: ，和。
----------------------------------------
输入: 从前有座山，山上有座庙
续写: 。
----------------------------------------
```

> 正式训练（8 层 + 140 万条数据 + GPU）后，续写效果会好很多。

## 5. SFT 微调调试

SFT（Supervised Fine-Tuning）让模型从"续写"变成"回答问题"。

### 命令行方式

```bash
# 在 trainer/ 目录下运行
cd trainer
python train_full_sft.py \
    --data_path ../dataset/sft_mini_512_debug.jsonl \
    --from_weight none \
    --batch_size 2 \
    --accumulation_steps 1 \
    --num_workers 0 \
    --epochs 1 \
    --log_interval 1 \
    --save_interval 50 \
    --hidden_size 512 \
    --num_hidden_layers 2 \
    --max_seq_len 128 \
    --device cpu
```

### PyCharm 方式

右上角选 **`debug_sft`**，直接点 Debug 按钮即可打断点。

### 参数说明

| 参数 | 调试值 | 正式训练值 | 说明 |
|------|--------|-----------|------|
| `--data_path` | `sft_mini_512_debug.jsonl` | `sft_mini_512.jsonl` | 调试用 100 条，正式 120 万条 |
| `--from_weight` | none | pretrain | **none=随机初始化**，不需要先跑预训练 |
| `--batch_size` | 2 | 32 | 调试用极小 batch |
| `--accumulation_steps` | 1 | 8 | 调试不累积梯度 |
| `--num_workers` | 0 | 8 | **0 才能在 PyCharm 打断点** |
| `--num_hidden_layers` | 2 | 8 | 调试用 2 层，跑得快 |
| `--max_seq_len` | 128 | 512 | 调试缩短序列 |
| `--device` | cpu | cuda:0 | Mac 用 cpu |

> `--from_weight none` 表示不加载预训练权重，直接从随机初始化开始 SFT。调试时这样最方便，正式训练应该用 `--from_weight pretrain` 加载预训练好的权重。

## 6. 测试 SFT 模型

SFT 模型能**理解指令并回答问题**（区别于预训练模型只会续写）。

### 命令行方式

```bash
# 在项目根目录运行（自动模式，跑内置问题）
python scripts/test_sft.py --device cpu

# 手动对话模式
python scripts/test_sft.py --device cpu --mode chat

# 控制生成长度
python scripts/test_sft.py --device cpu --max_new_tokens 200
```

### PyCharm 方式

右上角选 **`test_sft`**，点运行即可。

### 输出示例

调试用的 2 层小模型 + 100 条数据，对话效果很差是正常的：

```
正在加载模型...
从权重自动检测到 2 层
已加载权重: out/full_sft_512.pth
模型参数量: 8.92M

============================================================
SFT 模型对话测试
============================================================

你: 你好，请介绍一下你自己
AI: （乱码或重复输出）
[50 tokens, 12.3 tokens/s]
----------------------------------------
```

> 正式训练（8 层 + 120 万条数据 + GPU + 加载预训练权重）后，对话效果会好很多。

## 7. 完整流程

```
预训练 → SFT 微调 → 测试对话
  ↓         ↓          ↓
debug_pretrain → debug_sft → test_sft
```

1. 先跑 `debug_pretrain` 训练预训练模型（学会"语言"）
2. 再跑 `debug_sft` 微调模型（学会"回答问题"）
3. 最后用 `test_sft` 测试对话效果

> 调试模式下可以跳过预训练，直接跑 SFT（`--from_weight none`）。

## 8. 注意事项

- Mac 没有 CUDA，只能用 `--device cpu`，速度较慢但功能完整
- `num_workers=0` 是 PyCharm 断点调试的关键，否则 DataLoader 子进程里的断点不会触发
- 训练产出的权重保存在 `out/` 目录
- 预训练完成后可以接着跑 SFT（`train_full_sft.py`），模型才能"回答问题"
