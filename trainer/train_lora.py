import os
import sys

# LoRA 微调入口（实际逻辑在 train.py）
# 用法: python train_lora.py --lora_name lora_identity --batch_size 2 --device cpu
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trainer.train import main

if __name__ == "__main__":
    main(task='lora')
