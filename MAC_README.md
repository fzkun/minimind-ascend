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
4. 右上角运行配置会自动出现 **`debug_pretrain`** 和 **`test_pretrain`**

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

## 5. 注意事项

- Mac 没有 CUDA，只能用 `--device cpu`，速度较慢但功能完整
- `num_workers=0` 是 PyCharm 断点调试的关键，否则 DataLoader 子进程里的断点不会触发
- 训练产出的权重保存在 `out/` 目录
- 预训练完成后可以接着跑 SFT（`train_full_sft.py`），模型才能"回答问题"
