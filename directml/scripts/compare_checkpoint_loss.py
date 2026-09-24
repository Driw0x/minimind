import argparse
import csv
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from transformers import AutoTokenizer


def parse_checkpoint(spec: str):
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            "Expected LABEL=PATH, e.g. DirectML=out/pretrain_768.pth"
        )

    label, path = spec.split("=", 1)
    label = label.strip()
    path = Path(path.strip())

    if not path.is_absolute():
        path = ROOT_DIR / path

    return label, path


@torch.inference_mode()
def evaluate_checkpoint(
    checkpoint_path,
    loader,
    device,
    hidden_size,
    num_hidden_layers,
    use_moe,
):
    config = MiniMindConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        use_moe=bool(use_moe),
    )

    model = MiniMindForCausalLM(config)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    # Match eval_llm.py: inference in FP16, but compute CE reduction in FP32.
    model = model.half().eval().to(device)

    total_nll = 0.0
    total_tokens = 0

    for batch_index, (input_ids, labels) in enumerate(loader, start=1):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        outputs = model(input_ids)

        logits = outputs.logits[..., :-1, :].float()
        targets = labels[..., 1:].contiguous()

        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )

        valid_tokens = (targets != -100).sum()

        total_nll += loss_sum.item()
        total_tokens += valid_tokens.item()

        if batch_index % 20 == 0 or batch_index == len(loader):
            print(
                f"  batch {batch_index:4d}/{len(loader)} | "
                f"tokens={total_tokens:,}"
            )

    mean_loss = total_nll / total_tokens
    perplexity = math.exp(mean_loss) if mean_loss < 50 else float("inf")

    del model
    torch.cuda.empty_cache()

    return {
        "loss": mean_loss,
        "perplexity": perplexity,
        "tokens": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare MiniMind checkpoints on the exact same fixed subset "
            "under one PyTorch/ROCm environment."
        )
    )

    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help=(
            "Checkpoint to evaluate. Repeat for multiple models. "
            "Example: --checkpoint DirectML=out/pretrain_768.pth "
            "--checkpoint ROCm=out/pretrain_rocm_768.pth"
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Evaluation device. ROCm uses the PyTorch CUDA API.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("dataset/pretrain_t2t_mini.jsonl"),
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path("model"),
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=340)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--use-moe", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/checkpoint_eval.csv"),
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("No ROCm/CUDA device available.")

    data_path = args.data_path
    tokenizer_path = args.tokenizer_path
    output_path = args.output

    if not data_path.is_absolute():
        data_path = ROOT_DIR / data_path
    if not tokenizer_path.is_absolute():
        tokenizer_path = ROOT_DIR / tokenizer_path
    if not output_path.is_absolute():
        output_path = ROOT_DIR / output_path

    checkpoints = [parse_checkpoint(spec) for spec in args.checkpoint]

    for label, path in checkpoints:
        if not path.exists():
            raise FileNotFoundError(f"{label}: checkpoint not found: {path}")

    print("=" * 72)
    print("MiniMind checkpoint evaluation")
    print("=" * 72)
    print(f"PyTorch : {torch.__version__}")
    print(f"HIP     : {torch.version.hip}")
    print(f"Device  : {torch.cuda.get_device_name(0)}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    dataset = PretrainDataset(
        str(data_path),
        tokenizer,
        max_length=args.max_seq_len,
    )

    sample_count = min(args.samples, len(dataset))

    # One fixed random subset, reused for every checkpoint.
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), sample_count)

    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(f"Dataset samples : {len(dataset):,}")
    print(f"Eval samples    : {sample_count:,}")
    print(f"Max seq len     : {args.max_seq_len}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Subset seed     : {args.seed}")
    print()

    results = []

    for label, checkpoint_path in checkpoints:
        print("-" * 72)
        print(f"{label}: {checkpoint_path}")
        print("-" * 72)

        metrics = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            loader=loader,
            device=args.device,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=args.use_moe,
        )

        row = {
            "model": label,
            "checkpoint": str(checkpoint_path),
            "samples": sample_count,
            "tokens": metrics["tokens"],
            "loss": metrics["loss"],
            "perplexity": metrics["perplexity"],
        }
        results.append(row)

        print(
            f"\n{label}: "
            f"loss={metrics['loss']:.6f}, "
            f"perplexity={metrics['perplexity']:.4f}, "
            f"tokens={metrics['tokens']:,}\n"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "checkpoint",
                "samples",
                "tokens",
                "loss",
                "perplexity",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("=" * 72)
    print("Summary")
    print("=" * 72)

    for row in results:
        print(
            f"{row['model']:12s} "
            f"loss={row['loss']:.6f}  "
            f"ppl={row['perplexity']:.4f}"
        )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
