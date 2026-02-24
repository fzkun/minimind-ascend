import os
import sys
import json
import re
import argparse

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora

warnings.filterwarnings('ignore')


# ==================== 工具注册表与模拟 ====================

TOOL_REGISTRY = {
    "get_weather": {
        "definition": {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"}
                    },
                    "required": ["city"]
                }
            }
        },
        "mock": lambda args: f'{args.get("city", "未知")}：晴，气温25°C，湿度40%'
    },
    "calculate": {
        "definition": {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "计算数学表达式的结果",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "数学表达式"}
                    },
                    "required": ["expression"]
                }
            }
        },
        "mock": lambda args: str(eval(args.get("expression", "0"))) if args.get("expression") else "0"
    },
    "search": {
        "definition": {
            "type": "function",
            "function": {
                "name": "search",
                "description": "搜索互联网获取相关信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"}
                    },
                    "required": ["query"]
                }
            }
        },
        "mock": lambda args: f'搜索"{args.get("query", "")}"的结果：找到3条相关信息。'
    },
    "get_time": {
        "definition": {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "获取当前时间或指定时区的时间",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {"type": "string", "description": "时区，如Asia/Shanghai", "default": "Asia/Shanghai"}
                    },
                    "required": []
                }
            }
        },
        "mock": lambda args: f'当前时间（{args.get("timezone", "Asia/Shanghai")}）：2026-02-23 14:30:00'
    },
}


# ==================== 测试用例 ====================

TEST_CASES = [
    {
        "name": "天气查询",
        "query": "北京今天天气怎么样？",
        "expected_tool": "get_weather",
        "expected_has_call": True,
    },
    {
        "name": "数学计算",
        "query": "帮我算一下 123 * 456 等于多少？",
        "expected_tool": "calculate",
        "expected_has_call": True,
    },
    {
        "name": "时间查询",
        "query": "现在几点了？",
        "expected_tool": "get_time",
        "expected_has_call": True,
    },
    {
        "name": "搜索查询",
        "query": "帮我搜一下Python最新版本是什么？",
        "expected_tool": "search",
        "expected_has_call": True,
    },
    {
        "name": "普通对话（不需要工具）",
        "query": "你好，请介绍一下你自己。",
        "expected_tool": None,
        "expected_has_call": False,
    },
    {
        "name": "多参数天气查询",
        "query": "上海的天气如何？",
        "expected_tool": "get_weather",
        "expected_has_call": True,
    },
]


# ==================== 核心函数 ====================

def parse_tool_calls(text):
    """从模型输出中解析<tool_call>标签"""
    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    calls = []
    for match in matches:
        try:
            call_data = json.loads(match.strip())
            name = call_data.get('name', '')
            arguments = call_data.get('arguments', {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            calls.append({"name": name, "arguments": arguments})
        except json.JSONDecodeError:
            calls.append({"name": "", "arguments": {}, "raw": match.strip(), "parse_error": True})
    return calls


def simulate_tool(name, arguments):
    """模拟执行工具，返回结果字符串"""
    if name in TOOL_REGISTRY:
        try:
            return TOOL_REGISTRY[name]["mock"](arguments)
        except Exception as e:
            return f'工具执行错误: {e}'
    return f'未知工具: {name}'


def get_tool_definitions():
    """获取所有工具定义列表"""
    return [info["definition"] for info in TOOL_REGISTRY.values()]


def generate_response(model, tokenizer, messages, tools=None, device='cpu', max_new_tokens=512, temperature=0.1, top_p=0.9):
    """生成模型响应"""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, tools=tools
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def generate_with_tools(model, tokenizer, messages, tools, device='cpu', max_rounds=3, **kwargs):
    """
    带工具调用的完整生成循环：
    生成 → 检测tool_call → 模拟执行 → 注入结果 → 再次生成
    """
    current_messages = messages.copy()
    all_tool_calls = []

    for round_i in range(max_rounds):
        response = generate_response(model, tokenizer, current_messages, tools=tools, device=device, **kwargs)
        calls = parse_tool_calls(response)

        if not calls:
            # 没有工具调用，返回最终回复
            return response, all_tool_calls

        # 有工具调用
        all_tool_calls.extend(calls)
        current_messages.append({"role": "assistant", "content": response})

        # 模拟执行每个工具调用并注入结果
        for call in calls:
            result = simulate_tool(call["name"], call["arguments"])
            current_messages.append({"role": "tool", "content": result})

    # 最后一轮生成最终回答
    response = generate_response(model, tokenizer, current_messages, tools=tools, device=device, **kwargs)
    return response, all_tool_calls


# ==================== 模型加载 ====================

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
        ))
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')
    return model.eval().to(args.device), tokenizer


# ==================== 自动评估模式 ====================

def run_auto_eval(model, tokenizer, device):
    """运行预设测试用例，输出评估指标"""
    tools = get_tool_definitions()
    system_msg = {"role": "system", "content": "你是一个有用的助手。", "functions": tools}

    results = []
    print('\n' + '=' * 60)
    print('自动评估模式')
    print('=' * 60)

    for tc in TEST_CASES:
        print(f'\n--- 测试: {tc["name"]} ---')
        print(f'用户: {tc["query"]}')

        messages = [system_msg, {"role": "user", "content": tc["query"]}]
        response, tool_calls = generate_with_tools(model, tokenizer, messages, tools, device=device)

        has_call = len(tool_calls) > 0
        selected_tool = tool_calls[0]["name"] if tool_calls else None
        json_valid = all("parse_error" not in c for c in tool_calls) if tool_calls else True
        params_complete = all(
            isinstance(c.get("arguments"), dict) and len(c["arguments"]) > 0
            for c in tool_calls
        ) if tool_calls else True

        result = {
            "name": tc["name"],
            "query": tc["query"],
            "response": response,
            "tool_calls": tool_calls,
            "has_call": has_call,
            "expected_has_call": tc["expected_has_call"],
            "selected_tool": selected_tool,
            "expected_tool": tc["expected_tool"],
            "json_valid": json_valid,
            "params_complete": params_complete,
        }
        results.append(result)

        # 打印结果
        if tool_calls:
            for c in tool_calls:
                print(f'工具调用: {c["name"]}({json.dumps(c.get("arguments", {}), ensure_ascii=False)})')
        print(f'最终回复: {response[:200]}{"..." if len(response) > 200 else ""}')

    # 汇总指标
    print('\n' + '=' * 60)
    print('评估指标汇总')
    print('=' * 60)

    # 需要调用工具的用例
    should_call = [r for r in results if r["expected_has_call"]]
    # 不需要调用工具的用例
    should_not_call = [r for r in results if not r["expected_has_call"]]

    # 1. 工具调用检测率（应该调用时确实调用了）
    detect_correct = sum(1 for r in should_call if r["has_call"])
    detect_rate = detect_correct / len(should_call) if should_call else 0
    print(f'工具调用检测率:   {detect_correct}/{len(should_call)} = {detect_rate:.1%}')

    # 2. 工具选择正确率
    select_correct = sum(1 for r in should_call if r["selected_tool"] == r["expected_tool"])
    select_rate = select_correct / len(should_call) if should_call else 0
    print(f'工具选择正确率:   {select_correct}/{len(should_call)} = {select_rate:.1%}')

    # 3. JSON有效率
    json_valid_count = sum(1 for r in should_call if r["has_call"] and r["json_valid"])
    called_count = sum(1 for r in should_call if r["has_call"])
    json_rate = json_valid_count / called_count if called_count else 0
    print(f'JSON有效率:       {json_valid_count}/{called_count} = {json_rate:.1%}')

    # 4. 参数完整率
    params_ok = sum(1 for r in should_call if r["has_call"] and r["params_complete"])
    params_rate = params_ok / called_count if called_count else 0
    print(f'参数完整率:       {params_ok}/{called_count} = {params_rate:.1%}')

    # 5. 无需工具调用正确率（不应调用时确实没调用）
    no_call_correct = sum(1 for r in should_not_call if not r["has_call"])
    no_call_rate = no_call_correct / len(should_not_call) if should_not_call else 0
    print(f'无需工具正确率:   {no_call_correct}/{len(should_not_call)} = {no_call_rate:.1%}')

    print('=' * 60)


# ==================== 交互评估模式 ====================

def run_interactive(model, tokenizer, device):
    """交互模式: 用户输入查询，自动拦截工具调用并模拟执行"""
    tools = get_tool_definitions()
    system_msg = {"role": "system", "content": "你是一个有用的助手。", "functions": tools}

    print('\n' + '=' * 60)
    print('交互模式（输入quit退出）')
    print(f'已注册工具: {", ".join(TOOL_REGISTRY.keys())}')
    print('=' * 60)

    conversation = [system_msg]

    while True:
        try:
            query = input('\n用户: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n退出')
            break

        if query.lower() in ('quit', 'exit', 'q'):
            print('退出')
            break

        if query == '/clear':
            conversation = [system_msg]
            print('对话已清空')
            continue

        if not query:
            continue

        conversation.append({"role": "user", "content": query})
        response, tool_calls = generate_with_tools(
            model, tokenizer, conversation, tools, device=device
        )

        if tool_calls:
            print(f'\n[工具调用]')
            for c in tool_calls:
                result = simulate_tool(c["name"], c["arguments"])
                print(f'  {c["name"]}({json.dumps(c.get("arguments", {}), ensure_ascii=False)}) → {result}')

        print(f'\n助手: {response}')
        conversation.append({"role": "assistant", "content": response})


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="MiniMind工具调用评估")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='tool_sft', type=str, help="权重名称前缀")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--mode', default='auto', choices=['auto', 'interactive'], help="评估模式（auto=自动测试, interactive=交互式）")
    args = parser.parse_args()

    model, tokenizer = init_model(args)

    if args.mode == 'auto':
        run_auto_eval(model, tokenizer, args.device)
    else:
        run_interactive(model, tokenizer, args.device)


if __name__ == '__main__':
    main()
