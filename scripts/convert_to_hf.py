"""将 MiniMind .pth 权重转换为 HuggingFace 格式，供 vLLM 加载

Dense 模型 → LlamaForCausalLM 格式
MoE 模型  → Qwen2MoeForCausalLM 格式（完整保留共享专家）
"""
import os
import sys
import json
import shutil
import argparse
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from safetensors.torch import save_file


def _compute_intermediate_size(hidden_size, intermediate_size):
    """计算 intermediate_size（与模型内部逻辑一致）"""
    if intermediate_size is not None:
        return intermediate_size
    size = int(hidden_size * 8 / 3)
    return 64 * ((size + 64 - 1) // 64)


def _copy_tokenizer(out_dir):
    """复制 tokenizer 文件到输出目录"""
    tokenizer_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
    for fname in os.listdir(tokenizer_dir):
        if fname.startswith("tokenizer") or fname == "special_tokens_map.json":
            src = os.path.join(tokenizer_dir, fname)
            dst = os.path.join(out_dir, fname)
            shutil.copy2(src, dst)
            print(f"复制 {fname}")


def _remap_moe_state_dict(sd, config):
    """将 MiniMind MoE 权重名映射为 Qwen2MoE 格式

    映射规则:
      mlp.gate.weight                        → mlp.gate.weight（不变）
      mlp.experts.{j}.gate/up/down_proj      → mlp.experts.{j}.gate/up/down_proj（不变）
      mlp.shared_experts.0.gate_proj         → mlp.shared_expert.gate_proj（去掉复数和索引）
      mlp.shared_experts.0.up_proj           → mlp.shared_expert.up_proj
      mlp.shared_experts.0.down_proj         → mlp.shared_expert.down_proj（×2 补偿 sigmoid gate）

    补偿原理:
      MiniMind 的共享专家无条件应用: output += shared_expert(x)
      Qwen2MoE 有 sigmoid gate:    output += sigmoid(gate(x)) * shared_expert(x)
      设 gate 权重为零 → sigmoid(0) = 0.5 → 需要将 shared_expert 的 down_proj 乘以 2
      最终: 0.5 * (2 * down_proj)(silu(gate_proj(x)) * up_proj(x)) = shared_expert(x)
    """
    new_sd = {}
    for key, val in sd.items():
        new_key = key.replace("mlp.shared_experts.0.", "mlp.shared_expert.")
        # 补偿 sigmoid(0) = 0.5：将共享专家的 down_proj 权重乘以 2
        if "mlp.shared_expert.down_proj" in new_key:
            val = val.float() * 2.0
            val = val.half()
        new_sd[new_key] = val

    # 为每层添加 shared_expert_gate（全零权重 → sigmoid(0) = 0.5）
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = hidden_size // num_heads
    for i in range(num_layers):
        gate_key = f"model.layers.{i}.mlp.shared_expert_gate.weight"
        new_sd[gate_key] = torch.zeros(1, hidden_size, dtype=torch.float16)
        # Qwen2MoE attention 默认有 bias，MiniMind 没有，补零 bias
        new_sd[f"model.layers.{i}.self_attn.q_proj.bias"] = torch.zeros(num_heads * head_dim, dtype=torch.float16)
        new_sd[f"model.layers.{i}.self_attn.k_proj.bias"] = torch.zeros(num_kv_heads * head_dim, dtype=torch.float16)
        new_sd[f"model.layers.{i}.self_attn.v_proj.bias"] = torch.zeros(num_kv_heads * head_dim, dtype=torch.float16)
    print(f"共享专家: 添加 {num_layers} 个 sigmoid gate（全零）+ down_proj ×2 补偿")
    print(f"Attention: 添加 {num_layers * 3} 个零 bias（q/k/v_proj）")

    return new_sd


def convert(args):
    use_moe = bool(args.use_moe)
    moe_suffix = "_moe" if use_moe else ""

    # 1. 构建模型并加载权重
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=use_moe,
    )
    model = MiniMindForCausalLM(config)
    pth = os.path.join(args.save_dir, f"{args.weight}_{args.hidden_size}{moe_suffix}.pth")
    print(f"加载权重: {pth}")
    state_dict = torch.load(pth, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {params_m:.2f}M")

    intermediate_size = _compute_intermediate_size(config.hidden_size, config.intermediate_size)

    # 2. 创建输出目录
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # 3. 构建 config.json
    hf_config = {
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

    if use_moe:
        hf_config["architectures"] = ["Qwen2MoeForCausalLM"]
        hf_config["model_type"] = "qwen2_moe"
        hf_config["num_experts"] = config.n_routed_experts
        hf_config["num_experts_per_tok"] = config.num_experts_per_tok
        hf_config["moe_intermediate_size"] = intermediate_size
        hf_config["shared_expert_intermediate_size"] = intermediate_size
        hf_config["decoder_sparse_step"] = 1
        hf_config["norm_topk_prob"] = config.norm_topk_prob
        hf_config["router_aux_loss_coef"] = config.aux_loss_alpha
    else:
        hf_config["architectures"] = ["LlamaForCausalLM"]
        hf_config["model_type"] = "llama"

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(hf_config, f, indent=2)
    print(f"写入 config.json ({hf_config['model_type']})")

    # 4. 保存权重为 safetensors
    sd = model.half().state_dict()
    if "lm_head.weight" in sd:
        del sd["lm_head.weight"]
    if use_moe:
        sd = _remap_moe_state_dict(sd, config)
    save_file(sd, os.path.join(out_dir, "model.safetensors"))
    print(f"写入 model.safetensors")

    # 5. 复制 tokenizer 文件
    _copy_tokenizer(out_dir)

    print(f"\n转换完成! 输出目录: {out_dir}")
    print(f"可通过 vllm serve {out_dir} 启动服务")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="out", help="pth 权重所在目录")
    parser.add_argument("--weight", default="full_sft", help="权重前缀")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, default=0, help="是否为 MoE 模型")
    parser.add_argument("--output_dir", default="out/minimind-hf", help="HF 格式输出目录")
    args = parser.parse_args()
    convert(args)
