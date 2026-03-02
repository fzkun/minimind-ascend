"""
测试知识蒸馏学生模型的对话能力。

学生模型通过大模型（教师）指导训练，混合 CE Loss 和 KL 散度。

用法：
    python scripts/test_distill.py
    python scripts/test_distill.py --mode chat
    python scripts/test_distill.py --device cpu --max_new_tokens 200
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
    parser = argparse.ArgumentParser(description="测试知识蒸馏学生模型的对话能力")
    parser.add_argument('--weight_dir', default='out', type=str, help="权重目录")
    parser.add_argument('--weight_name', default='full_dist', type=str, help="权重名称前缀")
    parser.add_argument('--tokenizer_dir', default='model', type=str, help="分词器目录")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--device', default='cpu', type=str, help="运行设备（Mac 用 cpu）")
    parser.add_argument('--max_new_tokens', default=200, type=int, help="最多生成多少个 token")
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

    # ---------- 2. 准备测试问题 ----------
    test_prompts = [
        "你好，请介绍一下你自己",
        "什么是人工智能？",
        "请用Python写一个计算斐波那契数列的函数",
        "为什么天空是蓝色的？",
        "推荐一些中国的美食",
    ]

    # ---------- 3. 对话 ----------
    print("\n" + "=" * 60)
    print("知识蒸馏学生模型对话测试")
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
        output = tokenizer.decode(new_tokens, skip_special_tokens=True)
        gen_count = len(new_tokens)
        speed = gen_count / (time.time() - st)

        print(f"AI: {output}")
        print(f"[{gen_count} tokens, {speed:.1f} tokens/s]")
        print("-" * 40)


if __name__ == "__main__":
    main()
