"""
测试预训练模型的文本续写能力。

预训练模型只会"接着写"，不会"回答问题"（那是 SFT 之后的能力）。
所以这里给一段开头，让模型续写后面的内容。

用法：
    # 在项目根目录运行
    python scripts/test_pretrain.py

    # 自定义参数
    python scripts/test_pretrain.py --prompt "从前有座山" --max_new_tokens 100
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoTokenizer
from model.model_config import MiniMindConfig
from model.model_minimind import MiniMindForCausalLM
import argparse


def main():
    parser = argparse.ArgumentParser(description="测试预训练模型的续写能力")
    parser.add_argument('--weight_dir', default='out', type=str, help="权重目录")
    parser.add_argument('--tokenizer_dir', default='model', type=str, help="分词器目录")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度（要和训练时一致）")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="层数（要和训练时一致）")
    parser.add_argument('--device', default='cpu', type=str, help="运行设备（Mac 用 cpu）")
    parser.add_argument('--max_new_tokens', default=50, type=int, help="最多生成多少个 token")
    parser.add_argument('--temperature', default=0.85, type=float, help="温度，越大越随机")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus 采样阈值")
    parser.add_argument('--prompt', default=None, type=str, help="自定义续写开头")
    args = parser.parse_args()

    # ---------- 1. 加载模型和分词器 ----------
    print("正在加载模型...")
    weight_path = f'{args.weight_dir}/pretrain_{args.hidden_size}.pth'

    # 自动从权重文件推断层数，省得手动对齐
    num_layers = args.num_hidden_layers
    if os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location=args.device)
        layer_ids = {int(k.split('layers.')[1].split('.')[0]) for k in state_dict if 'layers.' in k}
        num_layers = len(layer_ids)
        print(f"从权重自动检测到 {num_layers} 层")
    else:
        state_dict = None
        print(f"警告: 未找到权重 {weight_path}，使用随机初始化（输出无意义）")

    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=num_layers,
    )
    model = MiniMindForCausalLM(config)

    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
        print(f"已加载权重: {weight_path}")

    model = model.eval().to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {total_params:.2f}M")

    # ---------- 2. 准备测试 prompt ----------
    # 预训练模型只会续写，所以给一些开头让它接着写
    prompts = [
        "中国的首都是",
        "人工智能的发展",
        "从前有座山，山上有座庙",
        "今天天气很好",
    ]

    if args.prompt:
        prompts = [args.prompt]

    # ---------- 3. 逐条生成 ----------
    print("\n" + "=" * 60)
    print("预训练模型续写测试（模型会接着你的开头往下写）")
    print("=" * 60)

    for prompt in prompts:
        # 预训练模型的输入格式：bos_token + 文本
        input_text = tokenizer.bos_token + prompt
        inputs = tokenizer(input_text, return_tensors="pt").to(args.device)

        with torch.no_grad():
            generated_ids = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_p=args.top_p,
                temperature=args.temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # 只取新生成的部分
        new_tokens = generated_ids[0][len(inputs["input_ids"][0]):]
        output = tokenizer.decode(new_tokens, skip_special_tokens=True)

        print(f"\n输入: {prompt}")
        print(f"续写: {output}")
        print("-" * 40)


if __name__ == "__main__":
    main()
