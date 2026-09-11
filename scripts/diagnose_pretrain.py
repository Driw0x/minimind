import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

TRAINER_DIR = Path(__file__).resolve().parent
ROOT = TRAINER_DIR.parent
sys.path.insert(0, str(ROOT))
os.chdir(TRAINER_DIR)

from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import get_device, init_model, setup_seed


def main():
    parser = argparse.ArgumentParser(description="MiniMind pretrain checkpoint diagnostic")
    parser.add_argument("--device", default="directml:1")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--weight", default="pretrain")
    parser.add_argument("--data_path", default="../dataset/pretrain_t2t_mini.jsonl")
    parser.add_argument("--max_seq_len", type=int, default=340)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_seed(args.seed)
    device = get_device(args.device)

    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )

    model, tokenizer = init_model(
        config,
        args.weight,
        device=device,
        dtype=args.dtype
    )

    model.eval()

    dataset = PretrainDataset(
        args.data_path,
        tokenizer,
        max_length=args.max_seq_len
    )

    start_index = args.sample_index
    end_index = min(start_index + args.num_samples, len(dataset))

    total_train_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_repeat = 0
    total_true_repeat = 0
    total_both = 0
    total_correct_only = 0
    total_repeat_only = 0
    total_entropy = 0.0

    print("============================================================")
    print(" MiniMind pretrain diagnostic")
    print("============================================================")
    print(f"Device: {device}")
    print(f"Dtype: {args.dtype}")
    print(f"Weight: {args.weight}")
    print(f"Samples: {start_index} -> {end_index - 1}")
    print(f"Count: {end_index - start_index}")
    print("============================================================")

    for index in range(start_index, end_index):
        input_ids, labels = dataset[index]

        input_ids = input_ids.unsqueeze(0).to(device)
        labels_device = labels.unsqueeze(0).to(device)

        targets = labels[1:].cpu()
        previous_tokens = input_ids[0, :-1].detach().cpu()
        mask = targets != -100

        targets = targets[mask]
        previous_tokens = previous_tokens[mask]

        if targets.numel() == 0:
            continue

        token_count = targets.numel()

        with torch.no_grad():
            train_outputs = model(
                input_ids=input_ids,
                labels=labels_device
            )

            diagnostic_outputs = model(
                input_ids=input_ids
            )

        train_loss = train_outputs.loss.detach().float().item()

        logits = diagnostic_outputs.logits[0, :-1].detach().float().cpu()
        logits = logits[mask]

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        predictions = logits.argmax(dim=-1)

        correct_mask = predictions == targets
        repeat_mask = predictions == previous_tokens

        sample_correct = correct_mask.sum().item()
        sample_repeat = repeat_mask.sum().item()
        sample_true_repeat = (targets == previous_tokens).sum().item()
        sample_both = (correct_mask & repeat_mask).sum().item()
        sample_correct_only = (correct_mask & ~repeat_mask).sum().item()
        sample_repeat_only = (repeat_mask & ~correct_mask).sum().item()
        sample_entropy = -(probs * log_probs).sum(dim=-1).sum().item()

        total_train_loss += train_loss * token_count
        total_tokens += token_count
        total_correct += sample_correct
        total_repeat += sample_repeat
        total_true_repeat += sample_true_repeat
        total_both += sample_both
        total_correct_only += sample_correct_only
        total_repeat_only += sample_repeat_only
        total_entropy += sample_entropy

        print(
            f"[{index}] "
            f"tokens={token_count} "
            f"train_loss={train_loss:.4f} "
            f"top1={sample_correct / token_count:.2%} "
            f"repeat={sample_repeat / token_count:.2%} "
            f"true_repeat={sample_true_repeat / token_count:.2%} "
            f"both={sample_both / token_count:.2%} "
            f"correct_only={sample_correct_only / token_count:.2%} "
            f"repeat_only={sample_repeat_only / token_count:.2%}"
        )

    if total_tokens == 0:
        raise RuntimeError("No valid tokens found.")

    mean_train_loss = total_train_loss / total_tokens
    perplexity = math.exp(min(mean_train_loss, 20))
    top1_accuracy = total_correct / total_tokens
    repeat_rate = total_repeat / total_tokens
    true_repeat_rate = total_true_repeat / total_tokens
    both_rate = total_both / total_tokens
    correct_only_rate = total_correct_only / total_tokens
    repeat_only_rate = total_repeat_only / total_tokens
    mean_entropy = total_entropy / total_tokens

    print("============================================================")
    print(" Aggregate results")
    print("============================================================")
    print(f"Valid tokens: {total_tokens}")
    print(f"Train-path mean loss: {mean_train_loss:.4f}")
    print(f"Train-path perplexity: {perplexity:.2f}")
    print(f"Top-1 accuracy: {top1_accuracy:.2%}")
    print(f"Top-1 repeat rate: {repeat_rate:.2%}")
    print(f"True-data repeat rate: {true_repeat_rate:.2%}")
    print(f"Top-1 correct and repeat: {both_rate:.2%}")
    print(f"Top-1 correct only: {correct_only_rate:.2%}")
    print(f"Top-1 repeat only: {repeat_only_rate:.2%}")
    print(f"Mean entropy: {mean_entropy:.4f}")
    print(f"Uniform baseline loss: {math.log(config.vocab_size):.4f}")
    print("============================================================")


if __name__ == "__main__":
    main()