import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import get_device


FIXED_INDICES = [
    233478,
    52451,
    576778,
    513575,
    468106,
    292632,
    214947,
    1143716,
]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one fixed MiniMind batch on DirectML or ROCm."
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--use-moe", type=int, choices=[0, 1], default=0)
    parser.add_argument("--max-seq-len", type=int, default=340)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint
    data_path = args.data_path
    tokenizer_path = args.tokenizer_path

    if not checkpoint.is_absolute():
        checkpoint = ROOT_DIR / checkpoint
    if not data_path.is_absolute():
        data_path = ROOT_DIR / data_path
    if not tokenizer_path.is_absolute():
        tokenizer_path = ROOT_DIR / tokenizer_path

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    device = get_device(args.device)

    print("=" * 72)
    print(f"Backend    : {args.label}")
    print(f"Device arg : {args.device}")
    print(f"Device     : {device}")
    print(f"Checkpoint : {checkpoint}")
    print(f"Indices    : {FIXED_INDICES}")
    print("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    dataset = PretrainDataset(
        str(data_path),
        tokenizer,
        max_length=args.max_seq_len,
    )

    if max(FIXED_INDICES) >= len(dataset):
        raise RuntimeError(
            f"Dataset too small for fixed indices: {len(dataset)} samples"
        )

    loader = DataLoader(
        Subset(dataset, FIXED_INDICES),
        batch_size=len(FIXED_INDICES),
        shuffle=False,
        num_workers=0,
    )

    input_ids, labels = next(iter(loader))
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        flash_attn=True,
    )

    model = MiniMindForCausalLM(config)

    try:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    model = model.float().eval().to(device)

    # torch.inference_mode() can fail on DirectML/privateuseone because
    # some model buffers are inspected/recreated during forward.
    # no_grad() disables autograd without creating inference tensors.
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)

    backend_loss = float(outputs.loss.detach().float().cpu().item())

    logits_cpu = outputs.logits[..., :-1, :].detach().float().cpu()
    targets_cpu = labels[..., 1:].detach().cpu()

    cpu_reference_loss = float(
        F.cross_entropy(
            logits_cpu.reshape(-1, logits_cpu.size(-1)),
            targets_cpu.reshape(-1),
            ignore_index=-100,
            reduction="mean",
        ).item()
    )

    valid_tokens = int((targets_cpu != -100).sum().item())

    finite = torch.isfinite(logits_cpu)

    result = {
        "label": args.label,
        "device_arg": args.device,
        "checkpoint": str(checkpoint),
        "indices": FIXED_INDICES,
        "valid_tokens": valid_tokens,
        "backend_loss": backend_loss,
        "cpu_reference_loss": cpu_reference_loss,
        "loss_difference": backend_loss - cpu_reference_loss,
        "logits_mean": float(logits_cpu.mean().item()),
        "logits_std": float(logits_cpu.std().item()),
        "logits_abs_max": float(logits_cpu.abs().max().item()),
        "logits_finite_ratio": float(finite.float().mean().item()),
    }

    print()
    print(f"Valid tokens       : {valid_tokens:,}")
    print(f"Backend loss       : {backend_loss:.6f}")
    print(f"CPU reference loss : {cpu_reference_loss:.6f}")
    print(f"Difference         : {backend_loss - cpu_reference_loss:+.6f}")
    print()
    print("Logits:")
    print(f"  mean         : {result['logits_mean']:.6f}")
    print(f"  std          : {result['logits_std']:.6f}")
    print(f"  abs max      : {result['logits_abs_max']:.6f}")
    print(f"  finite ratio : {result['logits_finite_ratio']:.6f}")

    output = args.output
    if output is None:
        safe_label = args.label.lower().replace(" ", "_")
        output = ROOT_DIR / "benchmarks" / f"single_batch_{safe_label}.json"
    elif not output.is_absolute():
        output = ROOT_DIR / output

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
