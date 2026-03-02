import os
import sys

# 预训练入口（实际逻辑在 train.py）
# 用法: python train_pretrain.py --batch_size 2 --device cpu
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trainer.train import main

if __name__ == "__main__":
    main(task='pretrain')
