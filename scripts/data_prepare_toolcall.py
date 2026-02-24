import os
import sys
import json
import argparse
import random

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import load_dataset
from transformers import AutoTokenizer


# ==================== 数据源转换函数 ====================

def convert_hermes(dataset, tokenizer, max_length, max_samples):
    """
    NousResearch/hermes-function-calling-v1
    字段: conversations (list of {from, value}), tools (str, JSON array)
    转换: from/value → role/content, human→user, gpt→assistant
    tools字段解析后挂到system消息的functions键
    注意: system消息中内嵌了工具描述文本（很长），需要替换为简短内容
    """
    ROLE_MAP = {'human': 'user', 'gpt': 'assistant', 'tool': 'tool'}
    results = []
    skipped = 0
    for sample in dataset:
        if len(results) >= max_samples:
            break
        conversations = sample.get('conversations', [])
        tools_str = sample.get('tools', '')
        if not conversations:
            skipped += 1
            continue

        # 解析工具定义（tools是JSON字符串）
        functions = None
        if tools_str and isinstance(tools_str, str) and tools_str.strip():
            try:
                tools_raw = json.loads(tools_str)
                if isinstance(tools_raw, list) and len(tools_raw) > 0:
                    functions = []
                    for t in tools_raw:
                        if isinstance(t, dict) and 'type' in t and 'function' in t:
                            functions.append(t)
                        elif isinstance(t, dict) and 'name' in t:
                            functions.append({"type": "function", "function": t})
            except (json.JSONDecodeError, TypeError):
                pass
        elif isinstance(tools_str, list) and len(tools_str) > 0:
            # datasets库可能已自动解析为list
            functions = []
            for t in tools_str:
                if isinstance(t, dict) and 'type' in t and 'function' in t:
                    functions.append(t)
                elif isinstance(t, dict) and 'name' in t:
                    functions.append({"type": "function", "function": t})

        # 转换对话
        msgs = []
        for turn in conversations:
            role = ROLE_MAP.get(turn.get('from', ''), turn.get('from', ''))
            content = turn.get('value', '')
            if role == 'system':
                # system消息内嵌了冗长的工具说明文本，替换为简短内容
                # 工具定义已从tools字段解析，不需要system文本中的
                short_content = 'You are a helpful assistant.'
                msg = {'role': 'system', 'content': short_content}
                if functions:
                    msg['functions'] = functions
                    functions = None  # 只挂一次
                msgs.append(msg)
            else:
                msgs.append({'role': role, 'content': content})

        # 如果有functions但没有system消息，插入一条
        if functions:
            msgs.insert(0, {'role': 'system', 'content': 'You are a helpful assistant.', 'functions': functions})

        # 过滤：至少有一条user和一条assistant
        roles = [m['role'] for m in msgs]
        if 'user' not in roles or 'assistant' not in roles:
            skipped += 1
            continue

        # 长度检查
        if not check_token_length(tokenizer, msgs, max_length):
            skipped += 1
            continue

        results.append({'conversations': msgs})

    print(f'[hermes] 转换完成: {len(results)} 条, 跳过: {skipped} 条')
    return results


def convert_glaive(dataset, tokenizer, max_length, max_samples):
    """
    hiyouga/glaive-function-calling-v2-sharegpt
    字段: conversations (list of {from, value})
    from可能是: human, gpt, function_call, observation
    function_call → assistant + <tool_call>包裹
    observation → tool
    从function_call消息中提取工具名和参数，自动构建工具定义挂到system.functions
    """
    results = []
    skipped = 0
    for sample in dataset:
        if len(results) >= max_samples:
            break
        conversations = sample.get('conversations', [])
        if not conversations:
            skipped += 1
            continue

        msgs = []
        functions = None
        # 先扫描所有function_call消息，提取工具定义
        extracted_tools = {}
        for turn in conversations:
            if turn.get('from') == 'function_call':
                try:
                    call_data = json.loads(turn.get('value', ''))
                    name = call_data.get('name', call_data.get('function', ''))
                    arguments = call_data.get('arguments', call_data.get('parameters', {}))
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    if name and name not in extracted_tools:
                        # 从参数反推properties
                        properties = {}
                        if isinstance(arguments, dict):
                            for k, v in arguments.items():
                                if isinstance(v, bool):
                                    properties[k] = {"type": "boolean", "description": k}
                                elif isinstance(v, int):
                                    properties[k] = {"type": "integer", "description": k}
                                elif isinstance(v, float):
                                    properties[k] = {"type": "number", "description": k}
                                else:
                                    properties[k] = {"type": "string", "description": k}
                        extracted_tools[name] = {
                            "type": "function",
                            "function": {
                                "name": name,
                                "description": name.replace('_', ' '),
                                "parameters": {
                                    "type": "object",
                                    "properties": properties,
                                    "required": list(properties.keys())
                                }
                            }
                        }
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass

        if extracted_tools:
            functions = list(extracted_tools.values())

        has_system = False
        for turn in conversations:
            role = turn.get('from', '')
            content = turn.get('value', '')

            if role == 'system':
                has_system = True
                clean_content = content.split('\n\nAvailable functions:')[0].split('\n\nYou have access to')[0].strip()
                if not clean_content:
                    clean_content = 'You are a helpful assistant.'
                msg = {'role': 'system', 'content': clean_content}
                if functions:
                    msg['functions'] = functions
                msgs.append(msg)

            elif role == 'human':
                msgs.append({'role': 'user', 'content': content})

            elif role == 'gpt':
                msgs.append({'role': 'assistant', 'content': content})

            elif role == 'function_call':
                tool_call_content = format_tool_call(content)
                msgs.append({'role': 'assistant', 'content': tool_call_content})

            elif role == 'observation':
                msgs.append({'role': 'tool', 'content': content})

        # 如果没有system消息但有工具定义，插入一条
        if not has_system and functions:
            msgs.insert(0, {'role': 'system', 'content': 'You are a helpful assistant.', 'functions': functions})

        # 过滤
        roles = [m['role'] for m in msgs]
        if 'user' not in roles or 'assistant' not in roles:
            skipped += 1
            continue

        if not check_token_length(tokenizer, msgs, max_length):
            skipped += 1
            continue

        results.append({'conversations': msgs})

    print(f'[glaive] 转换完成: {len(results)} 条, 跳过: {skipped} 条')
    return results


def convert_xlam(dataset, tokenizer, max_length, max_samples):
    """
    Salesforce/xlam-function-calling-60k
    字段: query (str), answers (str, JSON), tools (str, JSON array)
    单轮格式: query→user, answers→assistant(<tool_call>包裹), tools→system.functions
    """
    results = []
    skipped = 0
    for sample in dataset:
        if len(results) >= max_samples:
            break

        query = sample.get('query', '')
        answers_str = sample.get('answers', '')
        tools_str = sample.get('tools', '')

        if not query or not answers_str:
            skipped += 1
            continue

        # 解析工具定义
        functions = None
        if tools_str and tools_str.strip():
            try:
                tools_raw = json.loads(tools_str)
                if isinstance(tools_raw, list):
                    functions = []
                    for t in tools_raw:
                        if isinstance(t, dict):
                            if 'type' in t and 'function' in t:
                                functions.append(t)
                            elif 'name' in t:
                                functions.append({"type": "function", "function": t})
            except (json.JSONDecodeError, TypeError):
                pass

        # 解析答案为tool_call格式
        try:
            answers = json.loads(answers_str)
        except (json.JSONDecodeError, TypeError):
            skipped += 1
            continue

        if not isinstance(answers, list) or len(answers) == 0:
            skipped += 1
            continue

        # 构建assistant回复
        tool_call_parts = []
        for ans in answers:
            name = ans.get('name', '')
            arguments = ans.get('arguments', {})
            if name:
                call_json = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
                tool_call_parts.append(f'<tool_call>\n{call_json}\n</tool_call>')

        if not tool_call_parts:
            skipped += 1
            continue

        assistant_content = '\n'.join(tool_call_parts)

        msgs = []
        if functions:
            msgs.append({'role': 'system', 'content': '你是一个有用的助手。', 'functions': functions})
        msgs.append({'role': 'user', 'content': query})
        msgs.append({'role': 'assistant', 'content': assistant_content})

        if not check_token_length(tokenizer, msgs, max_length):
            skipped += 1
            continue

        results.append({'conversations': msgs})

    print(f'[xlam] 转换完成: {len(results)} 条, 跳过: {skipped} 条')
    return results


# ==================== 辅助函数 ====================

def render_tools_into_system(messages):
    """
    将functions字段预渲染到system消息内容中，然后移除functions字段。
    这样JSONL中不再有嵌套的functions dict，避免datasets库schema推断失败。
    渲染格式与chat template的tools参数一致。
    """
    if not messages or messages[0].get('role') != 'system' or not messages[0].get('functions'):
        return messages
    functions = messages[0]['functions']
    system_content = messages[0]['content']
    # 构建与chat template一致的工具说明文本
    tools_text = '\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n'
    tools_text += 'You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n'
    for func in functions:
        tools_text += json.dumps(func, ensure_ascii=False) + '\n'
    tools_text += '</tools>\n\n'
    tools_text += 'For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n'
    tools_text += '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
    # 合并到system内容
    new_system = {'role': 'system', 'content': system_content + tools_text}
    return [new_system] + messages[1:]


def check_token_length(tokenizer, messages, max_length):
    """用tokenizer估算token数，超过max_length则跳过"""
    try:
        tools = None
        if messages and messages[0]['role'] == 'system' and messages[0].get('functions'):
            tools = messages[0]['functions']
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens) <= max_length
    except Exception:
        # 退回到简单估算
        text = ''.join(m.get('content', '') for m in messages)
        return len(text) // 2 <= max_length


def extract_functions_from_system(text):
    """从glaive的system文本中提取工具定义JSON"""
    functions = []
    # 尝试多种模式提取
    # 模式1: 查找JSON数组
    import re
    json_patterns = re.findall(r'\{[^{}]*"name"\s*:\s*"[^"]*"[^{}]*"parameters"\s*:\s*\{[^}]*\}[^{}]*\}', text, re.DOTALL)
    for jp in json_patterns:
        try:
            obj = json.loads(jp)
            if 'name' in obj:
                functions.append({"type": "function", "function": obj})
        except json.JSONDecodeError:
            continue

    if functions:
        return functions

    # 模式2: 尝试查找整个JSON块
    for match in re.finditer(r'\[[\s\S]*?\]', text):
        try:
            arr = json.loads(match.group())
            if isinstance(arr, list):
                for item in arr:
                    if isinstance(item, dict) and 'name' in item:
                        functions.append({"type": "function", "function": item})
                if functions:
                    return functions
        except json.JSONDecodeError:
            continue

    return functions if functions else None


def format_tool_call(content):
    """将function_call内容包裹为<tool_call>格式"""
    content = content.strip()
    # 尝试解析为JSON并规范化
    try:
        call_data = json.loads(content)
        if isinstance(call_data, dict):
            name = call_data.get('name', call_data.get('function', ''))
            arguments = call_data.get('arguments', call_data.get('parameters', {}))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            call_json = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
            return f'<tool_call>\n{call_json}\n</tool_call>'
    except json.JSONDecodeError:
        pass
    # 无法解析则直接包裹
    return f'<tool_call>\n{content}\n</tool_call>'


def mix_sft_data(tool_data, sft_path, mix_ratio):
    """混入原有SFT数据防止遗忘"""
    if not os.path.exists(sft_path):
        print(f'警告: SFT数据文件不存在: {sft_path}, 跳过混合')
        return tool_data

    sft_samples = []
    with open(sft_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    sft_samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # 按比例采样SFT数据
    n_sft = int(len(tool_data) * mix_ratio)
    if n_sft > len(sft_samples):
        n_sft = len(sft_samples)
    sampled_sft = random.sample(sft_samples, n_sft)

    mixed = tool_data + sampled_sft
    random.shuffle(mixed)
    print(f'混合完成: 工具数据 {len(tool_data)} 条 + SFT数据 {n_sft} 条 = 共 {len(mixed)} 条')
    return mixed


# ==================== 数据源配置 ====================

SOURCES = {
    'hermes': {
        'hf_id': 'NousResearch/hermes-function-calling-v1',
        'converter': convert_hermes,
    },
    'glaive': {
        'hf_id': 'hiyouga/glaive-function-calling-v2-sharegpt',
        'converter': convert_glaive,
    },
    'xlam': {
        'hf_id': 'Salesforce/xlam-function-calling-60k',
        'converter': convert_xlam,
    },
}


def main():
    parser = argparse.ArgumentParser(description="下载并转换工具调用数据集为MiniMind格式")
    parser.add_argument('--source', type=str, default='hermes', choices=list(SOURCES.keys()),
                        help="数据源（hermes/glaive/xlam）")
    parser.add_argument('--max_length', type=int, default=512,
                        help="最大token长度，超过则跳过")
    parser.add_argument('--max_samples', type=int, default=10000,
                        help="每个数据源最大样本数")
    parser.add_argument('--output', type=str, default='../dataset/sft_tool_call.jsonl',
                        help="输出文件路径")
    parser.add_argument('--tokenizer_path', type=str, default='../model',
                        help="tokenizer路径")
    parser.add_argument('--mix_ratio', type=float, default=0.0,
                        help="混入原有SFT数据的比例（相对于工具数据条数），0表示不混合")
    parser.add_argument('--sft_path', type=str, default='../dataset/sft_mini_512.jsonl',
                        help="原有SFT数据路径（用于混合）")
    parser.add_argument('--seed', type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)

    # 加载tokenizer
    print(f'加载tokenizer: {args.tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # 下载并加载数据集
    source_cfg = SOURCES[args.source]
    hf_id = source_cfg['hf_id']
    print(f'下载数据集: {hf_id}')
    dataset = load_dataset(hf_id, split='train')
    print(f'数据集大小: {len(dataset)} 条')

    # 转换
    converter = source_cfg['converter']
    results = converter(dataset, tokenizer, args.max_length, args.max_samples)

    if not results:
        print('错误: 转换后无有效数据')
        return

    # 混入SFT数据
    if args.mix_ratio > 0:
        results = mix_sft_data(results, args.sft_path, args.mix_ratio)

    # 预渲染functions到system消息中，避免datasets库schema推断问题
    for sample in results:
        sample['conversations'] = render_tools_into_system(sample['conversations'])

    # 写入输出文件
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for sample in results:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f'输出文件: {args.output}, 共 {len(results)} 条')

    # 打印几条样例
    print('\n===== 样例数据 =====')
    for i, sample in enumerate(results[:3]):
        print(f'\n--- 样例 {i+1} ---')
        for msg in sample['conversations']:
            role = msg['role']
            content = msg['content'][:100] + ('...' if len(msg['content']) > 100 else '')
            print(f'  {role}: {content}')


if __name__ == '__main__':
    main()
