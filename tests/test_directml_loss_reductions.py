import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

TESTS_DIR = Path(__file__).resolve().parent
ROOT = TESTS_DIR.parent
TRAINER_DIR = ROOT / "trainer"

sys.path.insert(0, str(ROOT))
os.chdir(TRAINER_DIR)

from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import get_device, init_model, setup_seed


def main():
    parser = argparse.ArgumentParser(
        description="Compare DirectML cross-entropy reduction strategies on the same logits."
    )
    parser.add_argument("--device", default="directml:1")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--weight", default="pretrain")
    parser.add_argument("--data_path", default="../dataset/pretrain_t2t_mini.jsonl")
    parser.add_argument("--max_seq_len", type=int, default=340)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_seed(args.seed)
    device = get_device(args.device)

    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )

    model, tokenizer = init_model(
        config,
        args.weight,
        device=device,
        dtype=args.dtype,
    )
    model.eval()

    dataset = PretrainDataset(
        args.data_path,
        tokenizer,
        max_length=args.max_seq_len,
    )

    start = args.sample_index
    end = min(start + args.batch_size, len(dataset))
    if start < 0 or start >= len(dataset):
        raise ValueError(f"sample_index must be in [0, {len(dataset) - 1}]")
    if end <= start:
        raise RuntimeError("Empty batch.")

    samples = [dataset[i] for i in range(start, end)]
    input_ids = torch.stack([sample[0] for sample in samples]).to(device)
    labels = torch.stack([sample[1] for sample in samples]).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    logits = outputs.logits[:, :-1, :].contiguous()
    targets = labels[:, 1:].contiguous()

    x = logits.view(-1, logits.size(-1))
    y = targets.view(-1)
    valid = y != -100

    valid_count = int(valid.sum().item())
    total_count = y.numel()

    if valid_count == 0:
        raise RuntimeError("No valid target tokens in selected batch.")

    # A: current DirectML strategy: reduction=sum, then divide by valid tokens.
    dml_sum_raw = F.cross_entropy(
        x,
        y,
        ignore_index=-100,
        reduction="sum",
    )
    loss_sum_valid = dml_sum_raw.float() / valid_count

    # B: per-token CE on DirectML, then FP32 accumulation/mean over valid tokens.
    dml_none = F.cross_entropy(
        x,
        y,
        ignore_index=-100,
        reduction="none",
    )
    loss_none_mean = dml_none[valid].float().mean()

    # C: independent CPU FP32 reference using exactly the same logits/targets.
    x_cpu = x.detach().float().cpu()
    y_cpu = y.detach().cpu()
    loss_cpu = F.cross_entropy(
        x_cpu,
        y_cpu,
        ignore_index=-100,
        reduction="mean",
    )

    # Optional: ask the model's labels path for the exact current training loss.
    with torch.no_grad():
        outputs_with_labels = model(
            input_ids=input_ids,
            labels=labels,
        )
    model_loss = outputs_with_labels.loss.detach().float().cpu()

    print("=" * 70)
    print("DirectML cross-entropy reduction comparison")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Dtype: {args.dtype}")
    print(f"Weight: {args.weight}")
    print(f"Samples: {start} -> {end - 1}")
    print(f"Batch size: {end - start}")
    print(f"Max sequence length: {args.max_seq_len}")
    print(f"Total shifted positions: {total_count}")
    print(f"Valid target tokens: {valid_count}")
    print(f"Valid fraction: {valid_count / total_count:.4%}")
    print("-" * 70)
    print(f"Model outputs.loss:                  {model_loss.item():.8f}")
    print(f"A - DML sum / valid:                {loss_sum_valid.item():.8f}")
    print(f"B - DML none -> FP32 valid mean:    {loss_none_mean.item():.8f}")
    print(f"C - CPU FP32 mean reference:        {loss_cpu.item():.8f}")
    print("-" * 70)
    print(f"|A - C|: {abs(loss_sum_valid.item() - loss_cpu.item()):.10f}")
    print(f"|B - C|: {abs(loss_none_mean.item() - loss_cpu.item()):.10f}")
    print(f"|model - C|: {abs(model_loss.item() - loss_cpu.item()):.10f}")
    print("=" * 70)

    tol = 1e-2
    if abs(loss_none_mean.item() - loss_cpu.item()) <= tol:
        print("B matches CPU reference within tolerance.")
    else:
        print("WARNING: B does not match CPU reference within tolerance.")

    if abs(loss_sum_valid.item() - loss_cpu.item()) <= tol:
        print("A matches CPU reference within tolerance.")
    else:
        print("A does NOT match CPU reference within tolerance.")

    if abs(model_loss.item() - loss_cpu.item()) <= tol:
        print("Current model loss matches CPU reference within tolerance.")
    else:
        print("Current model loss does NOT match CPU reference within tolerance.")


if __name__ == "__main__":
    main()
