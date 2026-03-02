import os
import sys

# 让 Python 能找到项目根目录下的其他模块（model/、dataset/ 等）
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_config import MiniMindConfig
from dataset.lm_dataset import PretrainDataset, SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')

# 各训练任务的默认参数差异
TASK_DEFAULTS = {
    'pretrain': {
        'dataset_class': PretrainDataset,
        'learning_rate': 5e-4,
        'epochs': 1,
        'batch_size': 32,
        'accumulation_steps': 8,
        'save_weight': 'pretrain',
        'from_weight': 'none',
        'data_path': '../dataset/pretrain_hq.jsonl',
        'wandb_project': 'MiniMind-Pretrain',
        'save_dir': '../out',
        'log_interval': 100,
        'save_interval': 1000,
        'max_seq_len': 340,
    },
    'sft': {
        'dataset_class': SFTDataset,
        'learning_rate': 1e-6,
        'epochs': 2,
        'batch_size': 16,
        'accumulation_steps': 1,
        'save_weight': 'full_sft',
        'from_weight': 'pretrain',
        'data_path': '../dataset/sft_mini_512.jsonl',
        'wandb_project': 'MiniMind-Full-SFT',
        'save_dir': '../out',
        'log_interval': 100,
        'save_interval': 1000,
        'max_seq_len': 340,
    },
    'lora': {
        'dataset_class': SFTDataset,
        'learning_rate': 1e-4,
        'epochs': 50,
        'batch_size': 32,
        'accumulation_steps': 1,
        'save_weight': 'lora_identity',
        'from_weight': 'full_sft',
        'data_path': '../dataset/lora_identity.jsonl',
        'wandb_project': 'MiniMind-LoRA',
        'save_dir': '../out/lora',
        'log_interval': 10,
        'save_interval': 1000,
        'max_seq_len': 340,
    },
    'reason': {
        'dataset_class': SFTDataset,
        'learning_rate': 1e-6,
        'epochs': 1,
        'batch_size': 8,
        'accumulation_steps': 1,
        'save_weight': 'reason',
        'from_weight': 'dpo',
        'data_path': '../dataset/r1_mix_1024.jsonl',
        'wandb_project': 'MiniMind-Reasoning',
        'save_dir': '../out',
        'log_interval': 100,
        'save_interval': 100,
        'max_seq_len': 720,
    },
}

# LoRA 相关（仅在 task=lora 时设置）
save_lora = None
lora_params = None


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """训练一个 epoch 的核心函数"""
    # reason 任务：预先编码特殊 token，用于 loss 加权
    if args.task == 'reason':
        start_of_think_ids = tokenizer('<think>').input_ids
        end_of_think_ids = tokenizer('</think>').input_ids
        start_of_answer_ids = tokenizer('<answer>').input_ids
        end_of_answer_ids = tokenizer('</answer>').input_ids
        loss_fct = nn.CrossEntropyLoss(reduction='none')

    start_time = time.time()

    # 遍历数据，每次取一个 batch 的 input_ids 和 labels
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 把数据搬到训练设备上（CPU / GPU）
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        # 按余弦策略动态调整学习率（开始大、中间小、结尾更小）
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ---------- 前向传播：把数据喂给模型，算出 loss ----------
        with autocast_ctx:  # 混合精度上下文（CPU 下无效，自动跳过）
            if args.task == 'reason':
                # 推理蒸馏：自定义 loss，对 <think>/<answer> 等特殊 token 加权 ×10
                res = model(input_ids)
                shift_logits = res.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.size())
                loss_mask = (shift_labels != -100).float()
                sp_ids = torch.isin(shift_labels.view(-1),
                                    torch.tensor(start_of_think_ids + end_of_think_ids
                                                 + start_of_answer_ids + end_of_answer_ids
                                                 ).to(args.device))
                loss_mask_flat = loss_mask.view(-1)
                loss_mask_sum = loss_mask_flat.sum()
                loss_mask_flat[sp_ids] = 10
                loss_mask = loss_mask_flat.view(shift_labels.size())
                logits_loss = (loss * loss_mask).sum() / loss_mask_sum
                loss = logits_loss + res.aux_loss
                loss = loss / args.accumulation_steps
            else:
                # 标准前向：pretrain / sft / lora
                res = model(input_ids, labels=labels)
                # loss = 主损失 + MoE 辅助损失（非 MoE 时 aux_loss=0）
                loss = res.loss + res.aux_loss
                # 梯度累积：把 loss 平摊到多步，等效于更大的 batch_size
                loss = loss / args.accumulation_steps

        # ---------- 反向传播：计算梯度 ----------
        scaler.scale(loss).backward()

        # 每累积够 accumulation_steps 步，才真正更新一次权重
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            # LoRA 只裁剪 LoRA 参数的梯度
            clip_params = lora_params if args.task == 'lora' else model.parameters()
            torch.nn.utils.clip_grad_norm_(clip_params, args.grad_clip)

            scaler.step(optimizer)   # 用梯度更新模型权重
            scaler.update()

            optimizer.zero_grad(set_to_none=True)  # 清空梯度，准备下一轮

        # ---------- 打印日志 ----------
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            # reason 任务直接使用 logits_loss；其他任务用 total_loss - aux_loss
            current_logits_loss = logits_loss.item() if args.task == 'reason' else current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # ---------- 定期保存模型权重（断点续训用） ----------
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()  # 切到评估模式（关闭 dropout 等）
            if args.task == 'lora':
                # LoRA 只保存 LoRA 权重
                lora_save_path = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}.pth'
                save_lora(model, lora_save_path)
                lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            else:
                # 保存完整模型权重
                moe_suffix = '_moe' if lm_config.use_moe else ''
                ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
                # 如果是 DDP 多卡训练，要取出原始模型
                raw_model = model.module if isinstance(model, DistributedDataParallel) else model
                raw_model = getattr(raw_model, '_orig_mod', raw_model)
                state_dict = raw_model.state_dict()
                # 转成 float16 再存，节省磁盘空间
                torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
                lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
                del state_dict
            model.train()  # 切回训练模式

        # 及时释放不再需要的变量，节省内存
        del input_ids, labels, res, loss


def main(task=None):
    global args, optimizer, autocast_ctx, scaler, lm_config, model, tokenizer
    global save_lora, lora_params

    # ==================== 命令行参数 ====================
    # 运行时可以通过 --参数名 值 来覆盖默认值
    # 例如: python train.py --task pretrain --batch_size 4 --device cpu
    parser = argparse.ArgumentParser(description="MiniMind Training (Pretrain / SFT / LoRA / Reason)")
    parser.add_argument('--task', default=task or 'pretrain', choices=['pretrain', 'sft', 'lora', 'reason'],
                        help="训练任务类型（pretrain=预训练，sft=监督微调，lora=LoRA微调，reason=推理蒸馏）")
    # 以下参数的默认值取决于 task，这里先设为 None，解析后根据 task 填充
    parser.add_argument("--save_dir", type=str, default=None, help="模型保存目录")
    parser.add_argument('--save_weight', default=None, type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=None, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="初始学习率")
    parser.add_argument("--accumulation_steps", type=int, default=None, help="梯度累积步数")
    parser.add_argument("--log_interval", type=int, default=None, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=None, help="模型保存间隔")
    parser.add_argument('--max_seq_len', default=None, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument("--data_path", type=str, default=None, help="训练数据路径")
    parser.add_argument('--from_weight', default=None, type=str, help="基于哪个权重训练（none=从头开始，pretrain=预训练权重）")
    parser.add_argument("--wandb_project", type=str, default=None, help="wandb项目名")
    # 以下参数所有 task 共享同一默认值
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    # LoRA 专用参数
    parser.add_argument("--lora_name", type=str, default=None, help="LoRA 权重名称（仅 --task lora 有效，覆盖 --save_weight）")
    args = parser.parse_args()

    # 用 task 默认值填充用户未指定的参数
    defaults = TASK_DEFAULTS[args.task]
    for key, value in defaults.items():
        if key == 'dataset_class':
            continue
        if getattr(args, key) is None:
            setattr(args, key, value)

    # LoRA: --lora_name 覆盖 --save_weight（兼容原 train_lora.py 的用法）
    if args.lora_name:
        args.save_weight = args.lora_name

    # ========== 1. 初始化环境和随机种子 ==========
    # 检测是否多卡分布式训练（单机调试时会跳过）
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 固定随机种子，保证每次训练结果可复现
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查断点 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # 根据命令行参数创建模型配置（控制模型大小）
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 如果开启了续训，尝试加载之前保存的断点
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None

    # ========== 3. 设置混合精度 ==========
    # 混合精度：用半精度（float16/bfloat16）加速训练、省显存
    # CPU 上不支持混合精度，所以用 nullcontext() 跳过
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配置训练监控（可选） ==========
    # wandb/swanlab 用来可视化训练曲线，调试时一般不开
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        task_name = {'pretrain': 'Pretrain', 'sft': 'Full-SFT', 'lora': f'LoRA-{args.save_weight}', 'reason': 'Reasoning'}[args.task]
        wandb_run_name = f"MiniMind-{task_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 创建模型、数据集、优化器 ==========
    # 初始化模型和分词器
    # from_weight='pretrain' → 加载预训练好的权重作为起点
    # from_weight='none' → 随机初始化（预训练或调试用）
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)  # PyTorch 2.0 编译加速，调试时不建议开
        Logger('torch.compile enabled')

    # LoRA: 应用 LoRA 层，冻结非 LoRA 参数
    if args.task == 'lora':
        from model.model_lora import save_lora as _save_lora, apply_lora
        save_lora = _save_lora
        apply_lora(model)
        total_params = sum(p.numel() for p in model.parameters())
        lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
        Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
        Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
        Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")
        lora_params = []
        for name, param in model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
                lora_params.append(param)
            else:
                param.requires_grad = False

    # 加载训练数据集（pretrain=纯文本，sft/lora/reason=问答对格式）
    DatasetClass = defaults['dataset_class']
    train_ds = DatasetClass(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 多卡训练时用 DistributedSampler 把数据分给不同 GPU
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # GradScaler 配合混合精度用，CPU / bfloat16 下自动禁用
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # AdamW 优化器（LoRA 只优化 LoRA 参数）
    opt_params = lora_params if args.task == 'lora' else model.parameters()
    optimizer = optim.AdamW(opt_params, lr=args.learning_rate)

    # ========== 6. 从断点恢复（续训） ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        strict = (args.task != 'lora')  # LoRA 需要 strict=False
        model.load_state_dict(ckp_data['model'], strict=strict)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 多卡并行（单机调试会跳过） ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        # 每个 epoch 重新打乱数据顺序
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 如果是续训，跳过已经训练过的 batch
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # DataLoader：负责把数据集按 batch 喂给模型
        # num_workers=0 表示单线程加载（调试时方便打断点）
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # ========== 9. 训练结束，清理 ==========
    if dist.is_initialized(): dist.destroy_process_group()


if __name__ == "__main__":
    main()
