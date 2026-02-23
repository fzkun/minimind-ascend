"""将 MiniMind .pth 权重转换为 HuggingFace LlamaForCausalLM 格式，供 vLLM 加载"""
import os
import sys
import json
import shutil
import argparse
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from safetensors.torch import save_file


def convert(args):
    # 1. 构建模型并加载权重
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=False,
    )
    model = MiniMindForCausalLM(config)
    pth = os.path.join(args.save_dir, f"{args.weight}_{args.hidden_size}.pth")
    print(f"加载权重: {pth}")
    state_dict = torch.load(pth, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {params_m:.2f}M")

    # 2. 计算 intermediate_size（与模型内部逻辑一致）
    intermediate_size = config.intermediate_size
    if intermediate_size is None:
        intermediate_size = int(config.hidden_size * 8 / 3)
        intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

    # 3. 创建输出目录
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # 4. 写 config.json (LlamaForCausalLM 格式)
    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "hidden_act": "silu",
        "max_position_embeddings": config.max_position_embeddings,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
        "tie_word_embeddings": True,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "torch_dtype": "float16",
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(hf_config, f, indent=2)
    print(f"写入 config.json")

    # 5. 保存权重为 safetensors
    sd = model.half().state_dict()
    # 由于 tie_word_embeddings=True，去掉 lm_head.weight（它与 embed_tokens 共享）
    if "lm_head.weight" in sd:
        del sd["lm_head.weight"]
    save_file(sd, os.path.join(out_dir, "model.safetensors"))
    print(f"写入 model.safetensors")

    # 6. 复制 tokenizer 文件
    tokenizer_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
    for fname in os.listdir(tokenizer_dir):
        if fname.startswith("tokenizer") or fname == "special_tokens_map.json":
            src = os.path.join(tokenizer_dir, fname)
            dst = os.path.join(out_dir, fname)
            shutil.copy2(src, dst)
            print(f"复制 {fname}")

    print(f"\n转换完成! 输出目录: {out_dir}")
    print(f"可通过 vllm serve {out_dir} 启动服务")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="out", help="pth 权重所在目录")
    parser.add_argument("--weight", default="full_sft", help="权重前缀")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--output_dir", default="out/minimind-hf", help="HF 格式输出目录")
    args = parser.parse_args()
    convert(args)
