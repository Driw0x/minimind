import os
import sys
import time
import argparse

import torch
from torch import optim
from torch.utils.data import DataLoader

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import get_device, init_model, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile MiniMind DirectML training bottlenecks."
    )

    parser.add_argument("--device", type=str, default="directml:1")
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset/pretrain_t2t_mini.jsonl",
    )

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=340)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)

    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--accumulation_steps", type=int, default=8)

    parser.add_argument("--warmup_steps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--disable_foreach", action="store_true",)

    parser.add_argument("--model_dtype", choices=["float32", "float16"], default="float32",)

    return parser.parse_args()


def force_sync(tensor):
    """
    DirectML does not expose a CUDA-style synchronize API.

    Copying a scalar result back to CPU forces completion of queued
    work sufficiently for coarse profiling measurements.
    """
    tensor.detach().float().mean().cpu().item()


def main():
    args = parse_args()

    if args.steps <= args.warmup_steps:
        raise ValueError("--steps must be greater than --warmup_steps.")

    if args.accumulation_steps <= 0:
        raise ValueError("--accumulation_steps must be greater than 0.")

    setup_seed(42)

    device = get_device(args.device)

    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=False,
    )

    model, tokenizer = init_model(
        config,
        from_weight="none",
        tokenizer_path="model",
        device=device,
    )

    if args.model_dtype == "float16":
        model = model.half()
        
    dataset = PretrainDataset(
        args.data_path,
        tokenizer,
        max_length=args.max_seq_len,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        foreach=False if args.disable_foreach else None,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    totals = {
        "transfer": 0.0,
        "forward": 0.0,
        "backward": 0.0,
        "grad_clip": 0.0,
        "optimizer": 0.0,
    }

    measured_steps = 0
    optimizer_steps = 0

    print()
    print("=" * 70)
    print("MiniMind DirectML bottleneck profiler")
    print("=" * 70)
    print(f"Device:                 {args.device} -> {device}")
    print(f"Batch size:             {args.batch_size}")
    print(f"Max sequence length:    {args.max_seq_len}")
    print(f"Accumulation steps:     {args.accumulation_steps}")
    print(f"Warmup steps:           {args.warmup_steps}")
    print(f"Total steps:            {args.steps}")
    print("=" * 70)
    print()

    for step, (input_ids, labels) in enumerate(loader, start=1):
        if step > args.steps:
            break

        t0 = time.perf_counter()

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        force_sync(input_ids)
        transfer_time = time.perf_counter() - t0

        t0 = time.perf_counter()

        result = model(input_ids, labels=labels)
        loss = result.loss + result.aux_loss

        force_sync(loss)
        forward_time = time.perf_counter() - t0

        scaled_loss = loss / args.accumulation_steps

        t0 = time.perf_counter()

        scaled_loss.backward()

        # Read one gradient to force completion of backward work.
        for parameter in model.parameters():
            if parameter.grad is not None:
                force_sync(parameter.grad)
                break

        backward_time = time.perf_counter() - t0

        grad_clip_time = 0.0
        optimizer_time = 0.0

        if step % args.accumulation_steps == 0:
            t0 = time.perf_counter()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )

            for parameter in model.parameters():
                if parameter.grad is not None:
                    force_sync(parameter.grad)
                    break

            grad_clip_time = time.perf_counter() - t0

            t0 = time.perf_counter()

            optimizer.step()

            # Force completion after optimizer step.
            force_sync(next(model.parameters()))

            optimizer_time = time.perf_counter() - t0

            optimizer.zero_grad(set_to_none=True)

        if step <= args.warmup_steps:
            print(f"Warmup [{step}/{args.warmup_steps}]")
            continue
        
        if step % args.accumulation_steps == 0:
            optimizer_steps += 1

        totals["transfer"] += transfer_time
        totals["forward"] += forward_time
        totals["backward"] += backward_time
        totals["grad_clip"] += grad_clip_time
        totals["optimizer"] += optimizer_time

        measured_steps += 1

        print(
            f"Step [{step}/{args.steps}] "
            f"transfer={transfer_time:.3f}s "
            f"forward={forward_time:.3f}s "
            f"backward={backward_time:.3f}s "
            f"clip={grad_clip_time:.3f}s "
            f"optimizer={optimizer_time:.3f}s"
        )

    if measured_steps == 0:
        raise RuntimeError("No profiling steps were measured.")

    print()
    print("=" * 70)
    print("Profiling results")
    print("=" * 70)

    print(
        f"Average transfer time:   "
        f"{totals['transfer'] / measured_steps:.3f} s"
    )
    print(
        f"Average forward time:    "
        f"{totals['forward'] / measured_steps:.3f} s"
    )
    print(
        f"Average backward time:   "
        f"{totals['backward'] / measured_steps:.3f} s"
    )
    print(
        f"Average grad clip / step:"
        f" {totals['grad_clip'] / measured_steps:.3f} s"
    )

    if optimizer_steps > 0:
        print(
            f"Average optimizer time:  "
            f"{totals['optimizer'] / optimizer_steps:.3f} s"
        )

    total_profiled = sum(totals.values())

    print()
    print(f"Measured steps:           {measured_steps}")
    print(f"Optimizer steps:          {optimizer_steps}")
    print(f"Total profiled time:      {total_profiled:.3f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()