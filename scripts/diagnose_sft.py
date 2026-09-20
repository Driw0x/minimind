import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT))
os.chdir(SCRIPT_DIR)

from dataset.lm_dataset import SFTDataset
from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import get_device, init_model, setup_seed


def main():
    parser = argparse.ArgumentParser(description="MiniMind Full SFT checkpoint diagnostic")
    parser.add_argument("--device", default="directml:1")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--weight", default="full_sft")
    parser.add_argument("--data_path", default="../dataset/sft_t2t_mini.jsonl")
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
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

    dataset = SFTDataset(
        args.data_path,
        tokenizer,
        max_length=args.max_seq_len,
    )

    start_index = args.sample_index
    end_index = min(start_index + args.num_samples, len(dataset))

    total_train_loss = 0.0
    total_manual_loss = 0.0
    total_batches = 0
    total_tokens = 0
    total_correct = 0
    total_repeat = 0
    total_true_repeat = 0
    total_both = 0
    total_correct_only = 0
    total_repeat_only = 0
    total_entropy = 0.0

    print("============================================================")
    print(" MiniMind Full SFT diagnostic")
    print("============================================================")
    print(f"Device: {device}")
    print(f"Dtype: {args.dtype}")
    print(f"Weight: {args.weight}")
    print(f"Samples: {start_index} -> {end_index - 1}")
    print(f"Count: {end_index - start_index}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max sequence length: {args.max_seq_len}")
    print("============================================================")

    for batch_start in range(start_index, end_index, args.batch_size):
        batch_end = min(batch_start + args.batch_size, end_index)
        samples = [dataset[i] for i in range(batch_start, batch_end)]

        input_ids = torch.stack([sample[0] for sample in samples]).to(device)
        labels = torch.stack([sample[1] for sample in samples]).to(device)

        shifted_targets = labels[:, 1:]
        valid_mask = shifted_targets != -100
        token_count = valid_mask.sum().item()

        if token_count == 0:
            print(f"[{batch_start}:{batch_end}] no valid assistant tokens")
            continue

        with torch.no_grad():
            outputs = model(input_ids, labels=labels)

        train_loss = outputs.loss.detach().float().cpu().item()

        logits = outputs.logits[:, :-1, :].detach().float().cpu()
        targets = shifted_targets.detach().cpu()
        previous_tokens = input_ids[:, :-1].detach().cpu()
        mask = valid_mask.detach().cpu()

        valid_logits = logits[mask]
        valid_targets = targets[mask]
        valid_previous = previous_tokens[mask]

        manual_token_losses = F.cross_entropy(
            valid_logits,
            valid_targets,
            reduction="none",
        )
        manual_loss = manual_token_losses.mean().item()

        log_probs = F.log_softmax(valid_logits, dim=-1)
        probs = log_probs.exp()
        predictions = valid_logits.argmax(dim=-1)

        correct_mask = predictions == valid_targets
        repeat_mask = predictions == valid_previous

        sample_correct = correct_mask.sum().item()
        sample_repeat = repeat_mask.sum().item()
        sample_true_repeat = (valid_targets == valid_previous).sum().item()
        sample_both = (correct_mask & repeat_mask).sum().item()
        sample_correct_only = (correct_mask & ~repeat_mask).sum().item()
        sample_repeat_only = (repeat_mask & ~correct_mask).sum().item()
        sample_entropy = -(probs * log_probs).sum(dim=-1).sum().item()

        # outputs.loss is a mean over valid tokens in THIS batch.  Keep a
        # batch-mean aggregate for an apples-to-apples comparison with the
        # manual batch mean.  Separately retain the token-weighted manual
        # global loss as a dataset-quality metric.
        total_train_loss += train_loss
        total_manual_loss += manual_loss
        total_batches += 1
        total_tokens += token_count
        total_correct += sample_correct
        total_repeat += sample_repeat
        total_true_repeat += sample_true_repeat
        total_both += sample_both
        total_correct_only += sample_correct_only
        total_repeat_only += sample_repeat_only
        total_entropy += sample_entropy

        print(
            f"[{batch_start}:{batch_end}] "
            f"tokens={token_count} "
            f"train_loss={train_loss:.4f} "
            f"manual_loss={manual_loss:.4f} "
            f"diff={abs(train_loss - manual_loss):.6f} "
            f"top1={sample_correct / token_count:.2%} "
            f"repeat={sample_repeat / token_count:.2%}"
        )

        del input_ids, labels, outputs, logits, targets, previous_tokens
        del valid_logits, valid_targets, valid_previous, manual_token_losses

    if total_tokens == 0:
        raise RuntimeError("No valid SFT assistant tokens found.")

    mean_train_loss = total_train_loss / total_batches
    mean_manual_loss = total_manual_loss / total_batches
    loss_difference = abs(mean_train_loss - mean_manual_loss)

    # Recompute the true token-weighted global CE independently. This is
    # deliberately distinct from averaging per-batch model losses.
    global_manual_loss_sum = 0.0
    global_manual_tokens = 0
    with torch.no_grad():
        for batch_start in range(start_index, end_index, args.batch_size):
            batch_end = min(batch_start + args.batch_size, end_index)
            samples = [dataset[i] for i in range(batch_start, batch_end)]
            input_ids = torch.stack([sample[0] for sample in samples]).to(device)
            labels = torch.stack([sample[1] for sample in samples]).to(device)
            outputs = model(input_ids)
            logits = outputs.logits[:, :-1, :].detach().float().cpu()
            targets = labels[:, 1:].detach().cpu()
            mask = targets != -100
            if mask.any():
                losses = F.cross_entropy(logits[mask], targets[mask], reduction="none")
                global_manual_loss_sum += losses.sum().item()
                global_manual_tokens += mask.sum().item()

    global_manual_loss = global_manual_loss_sum / global_manual_tokens
    perplexity = math.exp(min(global_manual_loss, 20))
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
    print(f"Valid assistant tokens: {total_tokens}")
    print(f"Mean batch train-path loss: {mean_train_loss:.4f}")
    print(f"Mean batch manual loss: {mean_manual_loss:.4f}")
    print(f"Mean-batch loss difference: {loss_difference:.10f}")
    print(f"Token-weighted global manual loss: {global_manual_loss:.4f}")
    print(f"Manual perplexity: {perplexity:.2f}")
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
