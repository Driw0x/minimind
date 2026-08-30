import os
import sys
import time
import argparse

import torch
from torch import optim
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import get_device, init_model, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark real MiniMind pretraining performance."
    )

    parser.add_argument("--device", type=str, default="directml:1")
    parser.add_argument("--data_path", type=str, default="dataset/pretrain_t2t_mini.jsonl",)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=340)

    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)

    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--adam_eps", type=float, default=1e-8,)

    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)

    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--accumulation_steps", type=int, default=8,)

    parser.add_argument("--model_dtype", choices=["float32", "float16"], default="float32",)
    parser.add_argument("--loss_scale", type=float, default=1.0,)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.steps <= args.warmup_steps:
        raise ValueError("--steps must be greater than --warmup_steps.")

    if args.accumulation_steps <= 0:
        raise ValueError("--accumulation_steps must be greater than 0.")

    if args.loss_scale <= 0:
        raise ValueError("--loss_scale must be greater than 0.")

    if args.adam_eps <= 0:
        raise ValueError("--adam_eps must be greater than 0.")

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
        eps=args.adam_eps,
    )

    model.train()

    measured_time = 0.0
    measured_samples = 0
    measured_effective_tokens = 0
    measured_padded_tokens = 0
    measured_steps = 0

    print()
    print("=" * 60)
    print("MiniMind real training benchmark")
    print("=" * 60)
    print(f"Device:             {args.device} -> {device}")
    print(f"Model dtype:        {args.model_dtype}")
    print(f"Loss scale:         {args.loss_scale}")
    print(f"AdamW epsilon:      {args.adam_eps}")
    print(f"Dataset samples:    {len(dataset)}")
    print(f"Batch size:         {args.batch_size}")
    print(f"Max sequence len:   {args.max_seq_len}")
    print(f"Accumulation steps: {args.accumulation_steps}")
    print(f"Warmup steps:       {args.warmup_steps}")
    print(f"Total steps:        {args.steps}")
    print("=" * 60)
    print()

    optimizer.zero_grad(set_to_none=True)

    for step, (input_ids, labels) in enumerate(loader, start=1):
        if step > args.steps:
            break

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        start = time.perf_counter()

        result = model(input_ids, labels=labels)
        loss = result.loss + result.aux_loss

        loss_value = loss.detach().float().cpu().item()

        if not torch.isfinite(torch.tensor(loss_value)).item():
            raise RuntimeError(
                f"Non-finite loss detected at step {step}: {loss_value}"
            )

        scaled_loss = (
            loss / args.accumulation_steps
        ) * args.loss_scale

        scaled_loss.backward()

        if step % args.accumulation_steps == 0:
            if args.loss_scale != 1.0:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(args.loss_scale)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )

            grad_norm_value = grad_norm.detach().float().cpu().item()

            if not torch.isfinite(torch.tensor(grad_norm_value)).item():
                raise RuntimeError(
                    f"Non-finite gradient norm detected at step {step}: "
                    f"{grad_norm_value}"
                )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        elapsed = time.perf_counter() - start

        if step <= args.warmup_steps:
            print(
                f"Warmup [{step}/{args.warmup_steps}] "
                f"{elapsed:.3f}s "
                f"loss={loss_value:.4f}"
            )
            continue

        batch_samples = input_ids.size(0)

        # labels == -100 corresponds to padding ignored by the loss.
        effective_tokens = (labels != -100).sum().item()
        padded_tokens = input_ids.numel()

        measured_time += elapsed
        measured_samples += batch_samples
        measured_effective_tokens += effective_tokens
        measured_padded_tokens += padded_tokens
        measured_steps += 1

        print(
            f"Step [{step}/{args.steps}] "
            f"time={elapsed:.3f}s "
            f"loss={loss_value:.4f}"
        )

    if measured_steps == 0:
        raise RuntimeError("No benchmark steps were measured.")

    average_step_time = measured_time / measured_steps
    samples_per_second = measured_samples / measured_time
    effective_tokens_per_second = measured_effective_tokens / measured_time
    padded_tokens_per_second = measured_padded_tokens / measured_time

    steps_per_epoch = (
        len(dataset) + args.batch_size - 1
    ) // args.batch_size

    estimated_epoch_seconds = steps_per_epoch * average_step_time
    estimated_epoch_hours = estimated_epoch_seconds / 3600
    estimated_epoch_days = estimated_epoch_hours / 24

    print()
    print("=" * 60)
    print("Benchmark results")
    print("=" * 60)
    print(f"Measured steps:              {measured_steps}")
    print(f"Average iteration time:      {average_step_time:.3f} s")
    print(f"Samples / second:            {samples_per_second:.2f}")
    print(f"Effective tokens / second:   {effective_tokens_per_second:.2f}")
    print(f"Padded tokens / second:      {padded_tokens_per_second:.2f}")
    print(f"Steps / epoch:               {steps_per_epoch}")
    print(f"Estimated epoch duration:    {estimated_epoch_hours:.2f} h")
    print(f"Estimated epoch duration:    {estimated_epoch_days:.2f} days")
    print("=" * 60)


if __name__ == "__main__":
    main()