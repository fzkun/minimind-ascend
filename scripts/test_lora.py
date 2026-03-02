"""
测试 LoRA 微调模型的对话能力。

LoRA 在基座模型上叠加轻量适配器，不修改原始权重。
推理时需要：加载基座权重 → 注入 LoRA → 加载 LoRA 权重。

用法：
    # 在项目根目录运行
    python scripts/test_lora.py

    # 指定 LoRA 权重名称（如 lora_medical）
    python scripts/test_lora.py --lora_name lora_medical

    # 手动对话
    python scripts/test_lora.py --mode chat
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
from transformers import AutoTokenizer
from model.model_config import MiniMindConfig
from model.model_minimind import MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora
from trainer.trainer_utils import setup_seed
import argparse


def main():
    parser = argparse.ArgumentParser(description="测试 LoRA 微调模型的对话能力")
    parser.add_argument('--weight_dir', default='out', type=str, help="基座权重目录")
    parser.add_argument('--base_weight', default='full_sft', type=str, help="基座权重名称（默认 full_sft）")
    parser.add_argument('--lora_dir', default='out/lora', type=str, help="LoRA 权重目录")
    parser.add_argument('--lora_name', default='lora_identity', type=str, help="LoRA 权重名称")
    parser.add_argument('--tokenizer_dir', default='model', type=str, help="分词器目录")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--device', default='cpu', type=str, help="运行设备（Mac 用 cpu）")
    parser.add_argument('--max_new_tokens', default=200, type=int, help="最多生成多少个 token")
    parser.add_argument('--temperature', default=0.85, type=float, help="温度，越大越随机")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus 采样阈值")
    parser.add_argument('--mode', default='auto', choices=['auto', 'chat'], help="auto=自动跑内置问题，chat=手动对话")
    args = parser.parse_args()

    # ---------- 1. 加载基座模型 ----------
    print("正在加载基座模型...")
    base_path = f'{args.weight_dir}/{args.base_weight}_{args.hidden_size}.pth'

    if os.path.exists(base_path):
        state_dict = torch.load(base_path, map_location=args.device)
        layer_ids = {int(k.split('layers.')[1].split('.')[0]) for k in state_dict if 'layers.' in k}
        num_layers = len(layer_ids)
        print(f"从权重自动检测到 {num_layers} 层")
    else:
        state_dict = None
        num_layers = 2
        print(f"警告: 未找到基座权重 {base_path}，使用随机初始化（输出无意义）")

    config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=num_layers)
    model = MiniMindForCausalLM(config)

    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
        print(f"已加载基座权重: {base_path}")

    # ---------- 2. 注入并加载 LoRA ----------
    apply_lora(model)
    lora_path = f'{args.lora_dir}/{args.lora_name}_{args.hidden_size}.pth'
    if os.path.exists(lora_path):
        load_lora(model, lora_path)
        print(f"已加载 LoRA 权重: {lora_path}")
    else:
        print(f"警告: 未找到 LoRA 权重 {lora_path}，LoRA 层为随机初始化")

    model = model.eval().to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    lora_params = sum(p.numel() for n, p in model.named_parameters() if 'lora' in n) / 1e6
    print(f"模型参数量: {total_params:.2f}M (LoRA: {lora_params:.2f}M)")

    # ---------- 3. 准备测试问题 ----------
    test_prompts = [
        "你好，请介绍一下你自己",
        "你是谁开发的？",
        "你叫什么名字？",
        "你能做什么？",
    ]

    # ---------- 4. 对话 ----------
    print("\n" + "=" * 60)
    print(f"LoRA 模型对话测试 ({args.lora_name})")
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
