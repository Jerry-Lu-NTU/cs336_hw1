"""
CS336 Transformer LM 训练脚本
- 命令行控制全部超参数
- memmap 加载海量数据
- checkpoint 防止中断丢失进度
- WandB 监控训练健康度
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import wandb

from cs336_basics.model import TransformerLM
from cs336_basics.optim import AdamW, gradient_clipping, get_lr_cosine_schedule
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.data import run_get_batch


def parse_args():
    parser = argparse.ArgumentParser(description="CS336 Transformer LM Training")

    # 数据
    parser.add_argument("--train_data", type=str, required=True, help="训练数据路径 (.npy memmap)")
    parser.add_argument("--val_data", type=str, default=None, help="验证数据路径 (.npy memmap)")
    parser.add_argument("--val_batches", type=int, default=10, help="每次验证采样的 batch 数")

    # 模型架构
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=None, help="FFN 中间维度，默认 8/3 * d_model 对齐到 256")
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # 训练超参数
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--max_iters", type=int, required=True)
    parser.add_argument("--max_lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--cosine_cycle_iters", type=int, default=None, help="cosine decay 迭代数，默认等于 max_iters")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.95])
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # 精度
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])

    # Checkpoint
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--ckpt_interval", type=int, default=500, help="每 N 步保存一次 checkpoint")
    parser.add_argument("--resume_from", type=str, default=None, help="从 checkpoint 恢复训练")
    parser.add_argument("--val_interval", type=int, default=500, help="每 N 步验证一次")

    # WandB
    parser.add_argument("--wandb_project", type=str, default="cs336-basics")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true", help="禁用 WandB")

    # 日志
    parser.add_argument("--log_interval", type=int, default=10, help="每 N 步打印日志")

    # 设备
    parser.add_argument("--device", type=str, default=None, help="设备，自动检测")
    parser.add_argument("--compile", action="store_true", help="使用 torch.compile 加速")

    return parser.parse_args()


def compute_d_ff(d_model: int) -> int:
    raw = int((8 / 3) * d_model)
    return 64 * math.ceil(raw / 64)


@torch.no_grad()
def estimate_loss(model, dataset, batch_size, context_length, device, val_batches, ctx):
    model.eval()
    losses = []
    for _ in range(val_batches):
        x, y = run_get_batch(dataset, batch_size, context_length, device)
        with ctx:
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    args = parse_args()

    # 默认值
    if args.d_ff is None:
        args.d_ff = compute_d_ff(args.d_model)
    if args.cosine_cycle_iters is None:
        args.cosine_cycle_iters = args.max_iters
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    # ---- WandB ----
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=vars(args),
        )

    # ---- 加载数据 (memmap) ----
    print(f"Loading train data from {args.train_data} ...")
    train_dataset = np.memmap(args.train_data, dtype=np.uint16, mode="r")
    print(f"  Train tokens: {len(train_dataset):,}")

    val_dataset = None
    if args.val_data:
        print(f"Loading val data from {args.val_data} ...")
        val_dataset = np.memmap(args.val_data, dtype=np.uint16, mode="r")
        print(f"  Val tokens: {len(val_dataset):,}")

    # ---- 模型 ----
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device=device, dtype=dtype)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    # ---- 优化器 ----
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # ---- 恢复训练 ----
    start_iter = 0
    if args.resume_from:
        print(f"Resuming from {args.resume_from} ...")
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(f"  Resumed at iteration {start_iter}")

    # ---- torch.compile ----
    if args.compile:
        model = torch.compile(model)

    # ---- 混合精度 ----
    ctx = torch.amp.autocast(device_type=device, dtype=dtype) if device == "cuda" else torch.amp.autocast(device_type="cpu", dtype=dtype)
    scaler = torch.amp.GradScaler(device) if device == "cuda" and dtype == torch.float16 else None

    # ---- 创建 checkpoint 目录 ----
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ---- 训练循环 ----
    print(f"\nStarting training from iter {start_iter} to {args.max_iters} ...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Context length: {args.context_length}")
    print(f"  Max LR: {args.max_lr}, Min LR: {args.min_lr}")
    print(f"  Warmup: {args.warmup_iters}, Cosine cycle: {args.cosine_cycle_iters}")
    print(f"  Grad clip: {args.grad_clip}")
    print(f"  dtype: {args.dtype}")
    print(f"  Device: {device}")
    print()

    model.train()
    t0 = time.time()

    for it in range(start_iter, args.max_iters):
        # 学习率调度
        lr = get_lr_cosine_schedule(
            it, args.max_lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 采样 batch
        x, y = run_get_batch(train_dataset, args.batch_size, args.context_length, device)

        # 前向传播
        with ctx:
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )

        # 反向传播
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_clipping(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            gradient_clipping(model.parameters(), args.grad_clip)
            optimizer.step()

        # ---- 日志 ----
        if it % args.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            tokens_per_sec = args.batch_size * args.context_length / dt if dt > 0 else 0
            log_dict = {
                "train/loss": loss.item(),
                "train/lr": lr,
                "train/iter": it,
                "train/tokens_per_sec": tokens_per_sec,
                "train/iter_time_ms": dt * 1000,
            }
            print(
                f"iter {it:>6d} | loss {loss.item():.4f} | lr {lr:.2e} | "
                f"{tokens_per_sec:.0f} tok/s | {dt*1000:.1f} ms/iter"
            )
            if not args.no_wandb:
                wandb.log(log_dict, step=it)

        # ---- 验证 ----
        if val_dataset is not None and it % args.val_interval == 0 and it > 0:
            val_loss = estimate_loss(
                model, val_dataset, args.batch_size, args.context_length,
                device, args.val_batches, ctx
            )
            print(f"  [val] iter {it} | val_loss {val_loss:.4f}")
            if not args.no_wandb:
                wandb.log({"val/loss": val_loss}, step=it)

        # ---- Checkpoint ----
        if (it + 1) % args.ckpt_interval == 0 or it == args.max_iters - 1:
            ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{it+1}.pt")
            save_checkpoint(model, optimizer, it + 1, ckpt_path)
            print(f"  [ckpt] saved to {ckpt_path}")

            # 同时保存 config
            config_path = os.path.join(args.ckpt_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(vars(args), f, indent=2, default=str)

    print("\nTraining complete!")
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
