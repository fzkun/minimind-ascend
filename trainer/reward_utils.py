"""
奖励模型工具：支持 local / mock / api 三种模式

用法:
    from trainer.reward_utils import init_reward_model

    # argparse 里加参数（见 add_reward_args）
    reward_model, reward_tokenizer = init_reward_model(args)

    # 之后 calculate_rewards() 里照常调用:
    score = reward_model.get_score(reward_tokenizer, chat_messages)
"""

import random
import re


class MockRewardModel:
    """基于内容长度的伪随机分数，零依赖，用于调试训练循环"""

    def __init__(self, seed=42):
        self.rng = random.Random(seed)

    def get_score(self, tokenizer, chat_messages) -> float:
        # 基础随机分数 [-1, 1]
        base = self.rng.uniform(-1.0, 1.0)
        # 根据 assistant 回复长度给轻微偏置：越长越高（鼓励生成），上限 0.5
        assistant_text = ""
        for msg in chat_messages:
            if msg.get("role") == "assistant":
                assistant_text += msg.get("content", "")
        length_bias = min(len(assistant_text) / 200.0, 0.5)
        return round(base + length_bias, 4)


class APIRewardModel:
    """调用 OpenAI 兼容 API 打分（GLM-4.7 / GPT 等）"""

    SYSTEM_PROMPT = (
        "你是一个回答质量评估专家。请评估以下对话中 assistant 的回答质量。\n"
        "评分标准（-3 到 3 分）：\n"
        "- 3分：回答完美，准确、全面、有帮助\n"
        "- 2分：回答很好，基本正确且有用\n"
        "- 1分：回答尚可，部分正确\n"
        "- 0分：回答一般，没什么价值\n"
        "- -1分：回答较差，有明显错误\n"
        "- -2分：回答很差，大部分错误\n"
        "- -3分：完全错误或有害\n"
        "请只返回一个数字分数（如 1.5、-0.5），不要返回其他内容。"
    )

    def __init__(self, api_url, api_key, model):
        from openai import OpenAI
        self.client = OpenAI(base_url=api_url, api_key=api_key)
        self.model = model

    def get_score(self, tokenizer, chat_messages) -> float:
        # 把 chat_messages 格式化成可读文本
        conversation = ""
        for msg in chat_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            conversation += f"[{role}]: {content}\n"

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"请评估以下对话：\n\n{conversation}"},
                ],
                max_tokens=16,
                temperature=0.1,
            )
            text = resp.choices[0].message.content.strip()
            # 提取数字（支持负数、小数）
            match = re.search(r'-?\d+\.?\d*', text)
            if match:
                score = float(match.group())
                return max(min(score, 3.0), -3.0)
        except Exception as e:
            print(f"[APIRewardModel] 打分失败: {e}，返回 0.0")
        return 0.0


def add_reward_args(parser):
    """给 argparse 添加 reward 相关参数（3 个训练脚本统一调用）"""
    parser.add_argument("--reward_mode", default="local", choices=["local", "mock", "api"],
                        help="奖励模型模式: local=本地HF模型, mock=随机分数, api=API打分")
    parser.add_argument("--reward_api_url", default="http://192.168.0.81:4000/v1",
                        help="API 奖励模型的 base URL")
    parser.add_argument("--reward_api_key", default="sk-fsl123456",
                        help="API 奖励模型的 key")
    parser.add_argument("--reward_api_model", default="GLM-4.7",
                        help="API 奖励模型的模型名")


def init_reward_model(args):
    """
    根据 args.reward_mode 初始化奖励模型。

    返回: (reward_model, reward_tokenizer)
        - mock/api 模式下 reward_tokenizer = None
        - 三种模式都实现 get_score(tokenizer, chat_messages) -> float
    """
    if args.reward_mode == "mock":
        print("[Reward] 使用 Mock 奖励模型（随机分数，仅调试用）")
        return MockRewardModel(), None

    if args.reward_mode == "api":
        print(f"[Reward] 使用 API 奖励模型: {args.reward_api_url} / {args.reward_api_model}")
        return APIRewardModel(args.reward_api_url, args.reward_api_key, args.reward_api_model), None

    # local 模式：加载 HF 本地模型
    import torch
    from transformers import AutoModel, AutoTokenizer
    print(f"[Reward] 加载本地奖励模型: {args.reward_model_path}")
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
    return reward_model, reward_tokenizer
