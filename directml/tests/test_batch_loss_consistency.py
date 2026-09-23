import argparse
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import get_device, prepare_model_precision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=340)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--weight", default="pretrain")
    parser.add_argument("--dataset", default="dataset/pretrain_t2t_mini.jsonl")
    parser.add_argument("--tokenizer", default="model")
    args = parser.parse_args()

    device = get_device(args.device)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }

    dtype = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        local_files_only=True
    )

    dataset = PretrainDataset(
        args.dataset,
        tokenizer,
        max_length=args.max_seq_len
    )

    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )

    model = MiniMindForCausalLM(config)
    model = prepare_model_precision(model, device, dtype)

    weight_path = os.path.join(
        ROOT,
        "out",
        f"{args.weight}_{args.hidden_size}.pth"
    )

    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    start = args.sample_index
    end = min(start + args.num_samples, len(dataset))

    total_loss_sum = 0.0
    total_valid_tokens = 0
    batch_losses = []

    print("=" * 70)
    print("Batch loss analysis")
    print("=" * 70)
    print(f"Checkpoint: {weight_path}")
    print(f"Samples: {start} -> {end - 1}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max sequence length: {args.max_seq_len}")
    print()

    with torch.no_grad():
        for batch_start in range(start, end, args.batch_size):
            batch_end = min(batch_start + args.batch_size, end)

            samples = [dataset[i] for i in range(batch_start, batch_end)]

            input_ids = torch.stack(
                [sample[0] for sample in samples]
            ).to(device)

            labels = torch.stack(
                [sample[1] for sample in samples]
            ).to(device)

            outputs = model(
                input_ids=input_ids,
                labels=labels
            )

            model_loss = outputs.loss.float().item()

            logits = outputs.logits.float()
            shifted_logits = logits[..., :-1, :].contiguous()
            shifted_labels = labels[..., 1:].contiguous()

            valid_tokens = (shifted_labels != -100).sum().item()

            loss_sum = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_labels.view(-1),
                ignore_index=-100,
                reduction="sum"
            ).item()

            manual_loss = loss_sum / valid_tokens

            batch_losses.append(manual_loss)
            total_loss_sum += loss_sum
            total_valid_tokens += valid_tokens

            print(
                f"{batch_start:6d}-{batch_end - 1:<6d} "
                f"tokens={valid_tokens:5d} "
                f"model={model_loss:.4f} "
                f"manual={manual_loss:.4f}"
            )

    token_weighted_loss = total_loss_sum / total_valid_tokens
    mean_batch_loss = sum(batch_losses) / len(batch_losses)

    print()
    print("=" * 70)
    print("Aggregate")
    print("=" * 70)
    print(f"Valid tokens: {total_valid_tokens}")
    print(f"Mean batch loss: {mean_batch_loss:.4f}")
    print(f"Token-weighted global loss: {token_weighted_loss:.4f}")
    print(f"Minimum batch loss: {min(batch_losses):.4f}")
    print(f"Maximum batch loss: {max(batch_losses):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()