"""
测试 SPO（Self-Play Optimization）模型的推理能力。

SPO 模型默认以 reasoning 模式训练，输出包含 <think>/<answer> 标签。

用法：
    python scripts/test_spo.py
    python scripts/test_spo.py --mode chat
    python scripts/test_spo.py --device cpu --max_new_tokens 500
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
from transformers import AutoTokenizer
from model.model_config import MiniMindConfig
from model.model_minimind import MiniMindForCausalLM
from trainer.trainer_utils import setup_seed
import argparse


def main():
    parser = argparse.ArgumentParser(description="测试 SPO 模型的推理能力")
    parser.add_argument('--weight_dir', default='out', type=str, help="权重目录")
    parser.add_argument('--weight_name', default='spo', type=str, help="权重名称前缀")
    parser.add_argument('--tokenizer_dir', default='model', type=str, help="分词器目录")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--device', default='cpu', type=str, help="运行设备（Mac 用 cpu）")
    parser.add_argument('--max_new_tokens', default=500, type=int, help="最多生成多少个 token（推理链较长，默认500）")
    parser.add_argument('--temperature', default=0.85, type=float, help="温度，越大越随机")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus 采样阈值")
    parser.add_argument('--mode', default='auto', choices=['auto', 'chat'], help="auto=自动跑内置问题，chat=手动对话")
    args = parser.parse_args()

    # ---------- 1. 加载模型和分词器 ----------
    print("正在加载模型...")
    weight_path = f'{args.weight_dir}/{args.weight_name}_{args.hidden_size}.pth'

    if os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location=args.device)
        layer_ids = {int(k.split('layers.')[1].split('.')[0]) for k in state_dict if 'layers.' in k}
        num_layers = len(layer_ids)
        print(f"从权重自动检测到 {num_layers} 层")
    else:
        state_dict = None
        num_layers = 2
        print(f"警告: 未找到权重 {weight_path}，使用随机初始化（输出无意义）")

    config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=num_layers)
    model = MiniMindForCausalLM(config)

    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
        print(f"已加载权重: {weight_path}")

    model = model.eval().to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {total_params:.2f}M")

    # ---------- 2. 准备测试问题（偏推理类） ----------
    test_prompts = [
        "请用一段话描述阿里巴巴集团的企业文化。",
        "1+1等于几？请一步步思考。",
        "如果今天是周三，那三天后是周几？",
        "一个房间里有3只猫，每只猫看到2只猫，这可能吗？为什么？",
    ]

    # ---------- 3. 对话 ----------
    print("\n" + "=" * 60)
    print("SPO 模型推理测试（观察 <think> 和 <answer> 标签）")
    print("=" * 60)

    if args.mode == 'chat':
        prompts = iter(lambda: input('\n你: '), '')
    else:
        prompts = test_prompts

    for prompt in prompts:
        if args.mode == 'auto':
            print(f"\n你: {prompt}")

        setup_seed(2026)

        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(args.device)

        st = time.time()
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
                repetition_penalty=1.0,
            )

        new_tokens = generated_ids[0][len(inputs["input_ids"][0]):]
        output = tokenizer.decode(new_tokens, skip_special_tokens=False)
        output = output.replace(tokenizer.eos_token, '').strip()
        gen_count = len(new_tokens)
        speed = gen_count / (time.time() - st)

        print(f"AI: {output}")
        print(f"[{gen_count} tokens, {speed:.1f} tokens/s]")
        print("-" * 40)


if __name__ == "__main__":
    main()
