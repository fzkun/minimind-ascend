import os
import sys

# 推理蒸馏入口（实际逻辑在 train.py）
# 用法: python train_reason.py --batch_size 2 --device cpu
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trainer.train import main

if __name__ == "__main__":
    main(task='reason')
