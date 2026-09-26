# scripts/diagnose_validation.py

import argparse
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from dataset.lm_dataset import PretrainDataset, SFTDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


ROOT = Path(__file__).resolve().parent.parent


EXTERNAL_PROMPTS = [
    "用两句话解释为什么天空看起来是蓝色的。",
    "请用Python写一个函数，返回斐波那契数列的第n项，并说明fibonacci(10)的结果。",
    "一个篮子里有12个苹果，拿走5个，又放进去3个，现在有多少个苹果？请简短说明。",
    "请简要解释植物光合作用中光反应和卡尔文循环分别做什么。",
    "严格使用三个要点解释机器学习，不要写第四个要点。",
    "比较RAM和SSD的作用，它们为什么不能互相替代？",
    "如果明天下大雨并且必须步行20分钟出门，给出三条实用建议。",
    "找出下面代码的问题并修复：\n"
    "def add(a, b):\n"
    "    result = a + b\n"
    "    print(result)\n"
    "    return a - b",
]


def resolve_project_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def resolve_weight(spec, args):
    """Resolve either a direct path or a short weight name such as pretrain/full_sft."""
    path = Path(spec)

    if not path.is_absolute():
        direct_path = ROOT / path
        if direct_path.exists():
            return spec, direct_path

    if path.exists():
        return path.stem, path.resolve()

    suffix = "_moe" if args.use_moe else ""
    candidate = ROOT / args.out_dir / f"{spec}_{args.hidden_size}{suffix}.pth"

    if candidate.exists():
        return spec, candidate

    raise FileNotFoundError(
        f"Weight not found: {spec}\n"
        f"Also tried: {candidate}"
    )


def file_info(path):
    path = Path(path)
    return {
        "exists": path.exists(),
        "path": str(path),
        "size_mb": path.stat().st_size / 1024**2 if path.exists() else None,
    }


def infer_resume_checkpoint(weight_path, args):
    """Infer the resume checkpoint from the weight filename when possible."""
    candidate = ROOT / args.checkpoints_dir / f"{Path(weight_path).stem}_resume.pth"
    return candidate


def inspect_resume_checkpoint(path):
    path = Path(path)

    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
        }

    checkpoint = torch.load(path, map_location="cpu")
    optimizer = checkpoint.get("optimizer", {}) if isinstance(checkpoint, dict) else {}
    param_groups = optimizer.get("param_groups", []) if isinstance(optimizer, dict) else []

    saved_lr = param_groups[0].get("lr") if param_groups else None

    return {
        "exists": True,
        "path": str(path),
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "step": checkpoint.get("step") if isinstance(checkpoint, dict) else None,
        "world_size": checkpoint.get("world_size") if isinstance(checkpoint, dict) else None,
        "saved_lr": saved_lr,
    }


def fixed_indices(dataset_length, sample_count, seed):
    n = min(sample_count, dataset_length)
    return random.Random(seed).sample(range(dataset_length), n)


def dataset_stats(dataset, pad_token_id, sample_count, seed):
    indices = fixed_indices(len(dataset), sample_count, seed)

    random.seed(seed)
    torch.manual_seed(seed)

    input_tokens = 0
    supervised_tokens = 0

    for index in indices:
        input_ids, labels = dataset[index]
        input_tokens += int((input_ids != pad_token_id).sum().item())
        supervised_tokens += int((labels != -100).sum().item())

    count = max(len(indices), 1)

    return {
        "total_samples": len(dataset),
        "samples_analyzed": len(indices),
        "avg_input_tokens": input_tokens / count,
        "avg_supervised_tokens": supervised_tokens / count,
    }


def load_model(weight_path, args, device):
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )

    model = MiniMindForCausalLM(config)
    state_dict = torch.load(weight_path, map_location="cpu")

    if (
        isinstance(state_dict, dict)
        and "model" in state_dict
        and isinstance(state_dict["model"], dict)
    ):
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_loss(model, dataset, indices, device, batch_size, seed):
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    random.seed(seed)
    torch.manual_seed(seed)

    total_loss = 0.0
    total_tokens = 0

    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        output = model(input_ids, labels=labels)
        valid_tokens = int((labels != -100).sum().item())

        total_loss += float(output.loss.detach().float().cpu()) * valid_tokens
        total_tokens += valid_tokens

    loss = total_loss / max(total_tokens, 1)

    return {
        "loss": loss,
        "perplexity": math.exp(min(loss, 20)),
        "tokens": total_tokens,
    }


def print_loss_result(model_name, dataset_name, result):
    print(
        f"{model_name:20s} "
        f"on {dataset_name:10s} "
        f"| loss={result['loss']:.4f} "
        f"| ppl={result['perplexity']:.2f} "
        f"| tokens={result['tokens']}"
    )


def repetition_ratio(text, n=3):
    chars = [c for c in text if not c.isspace()]

    if len(chars) < n:
        return 0.0

    ngrams = [tuple(chars[i:i + n]) for i in range(len(chars) - n + 1)]
    if not ngrams:
        return 0.0

    return 1.0 - len(set(ngrams)) / len(ngrams)


def build_prompt(tokenizer, question):
    messages = [{"role": "user", "content": question}]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return question


@torch.no_grad()
def generate_answer(model, tokenizer, question, device, max_new_tokens):
    prompt = build_prompt(tokenizer, question)
    encoded = tokenizer(prompt, return_tensors="pt")

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")

    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    generation_args = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
        "eos_token_id": tokenizer.eos_token_id,
    }

    if attention_mask is not None:
        generation_args["attention_mask"] = attention_mask

    output = model.generate(**generation_args)
    generated_ids = output[0, input_ids.shape[1]:]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    return answer, len(generated_ids)


def load_local_datasets(tokenizer, args):
    pretrain_path = resolve_project_path(args.pretrain_data)
    sft_path = resolve_project_path(args.sft_data)

    if not pretrain_path.exists():
        raise FileNotFoundError(pretrain_path)
    if not sft_path.exists():
        raise FileNotFoundError(sft_path)

    pretrain_dataset = PretrainDataset(
        str(pretrain_path),
        tokenizer,
        max_length=args.pretrain_max_seq_len,
    )

    sft_dataset = SFTDataset(
        str(sft_path),
        tokenizer,
        max_length=args.sft_max_seq_len,
    )

    return pretrain_path, sft_path, pretrain_dataset, sft_dataset


def run_training_audit(weights, tokenizer, args):
    print("\n========================================")
    print(" TRAINING AUDIT")
    print("========================================")

    pretrain_path, sft_path, pretrain_dataset, sft_dataset = load_local_datasets(
        tokenizer,
        args,
    )

    print("\n--- FILES ---")
    for label, path in [
        ("Pretrain dataset", pretrain_path),
        ("SFT dataset", sft_path),
        *[(f"Weight: {name}", path) for name, path in weights],
    ]:
        info = file_info(path)
        if info["exists"]:
            print(f"{label}: {info['size_mb']:.2f} MB ({info['path']})")
        else:
            print(f"{label}: MISSING ({info['path']})")

    print("\n--- CHECKPOINTS ---")
    for name, weight_path in weights:
        resume_path = infer_resume_checkpoint(weight_path, args)
        info = inspect_resume_checkpoint(resume_path)

        print(f"\n{name}:")
        for key, value in info.items():
            print(f"  {key}: {value}")

    print(
        "\nNote: saved_lr is the optimizer learning rate stored in the resume "
        "checkpoint, not necessarily the initial learning rate."
    )

    print("\n--- TOKENIZER ---")
    print(f"vocab size: {len(tokenizer)}")
    print(f"pad token: {tokenizer.pad_token_id}")
    print(f"bos token: {tokenizer.bos_token_id}")
    print(f"eos token: {tokenizer.eos_token_id}")

    print("\n--- DATASETS ---")
    pretrain_stats = dataset_stats(
        pretrain_dataset,
        tokenizer.pad_token_id,
        args.stat_samples,
        args.seed,
    )
    sft_stats = dataset_stats(
        sft_dataset,
        tokenizer.pad_token_id,
        args.stat_samples,
        args.seed,
    )

    print("Pretrain:")
    for key, value in pretrain_stats.items():
        print(f"  {key}: {value}")

    print("\nSFT:")
    for key, value in sft_stats.items():
        print(f"  {key}: {value}")

    return pretrain_dataset, sft_dataset


def run_local_test(weights, tokenizer, args, device, datasets=None):
    print("\n========================================")
    print(" LOCAL DATA VALIDATION")
    print("========================================")

    print(
        "\nThis mode evaluates random samples from the local training datasets. "
        "It measures training fit and forgetting, not true generalization."
    )

    if datasets is None:
        _, _, pretrain_dataset, sft_dataset = load_local_datasets(tokenizer, args)
    else:
        pretrain_dataset, sft_dataset = datasets

    print(f"\nPretrain samples: {len(pretrain_dataset):,}")
    print(f"SFT samples     : {len(sft_dataset):,}")

    pretrain_indices = fixed_indices(
        len(pretrain_dataset),
        args.eval_samples,
        args.seed,
    )
    sft_indices = fixed_indices(
        len(sft_dataset),
        args.eval_samples,
        args.seed,
    )

    print(
        f"Samples evaluated: {len(pretrain_indices)} pretrain / "
        f"{len(sft_indices)} SFT"
    )
    print(f"Seed: {args.seed}")

    results = {}

    for name, path in weights:
        print(f"\nLoading {name}")
        print(f"  {path}")

        model = load_model(path, args, device)
        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params / 1e6:.2f}M")

        pretrain_result = evaluate_loss(
            model=model,
            dataset=pretrain_dataset,
            indices=pretrain_indices,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed,
        )

        sft_result = evaluate_loss(
            model=model,
            dataset=sft_dataset,
            indices=sft_indices,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed,
        )

        results[name] = {
            "pretrain": pretrain_result,
            "sft": sft_result,
        }

        print_loss_result(name, "pretrain", pretrain_result)
        print_loss_result(name, "SFT", sft_result)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n========================================")
    print(" LOCAL COMPARISON")
    print("========================================")

    baseline_name = weights[0][0]
    baseline = results[baseline_name]
    print(f"\nBaseline: {baseline_name}")

    for name, _ in weights[1:]:
        delta_pretrain = (
            results[name]["pretrain"]["loss"]
            - baseline["pretrain"]["loss"]
        )
        delta_sft = (
            results[name]["sft"]["loss"]
            - baseline["sft"]["loss"]
        )

        print(f"\n{name} vs {baseline_name}")
        print(f"  pretrain loss delta: {delta_pretrain:+.4f}")
        print(f"  SFT loss delta     : {delta_sft:+.4f}")

        if delta_pretrain > 0:
            print("  -> higher pretrain loss: possible forgetting")
        elif delta_pretrain < 0:
            print("  -> lower pretrain loss")
        else:
            print("  -> unchanged pretrain loss")

        if delta_sft < 0:
            print("  -> lower SFT loss: stronger SFT fit")
        elif delta_sft > 0:
            print("  -> higher SFT loss")
        else:
            print("  -> unchanged SFT loss")

    return results


def run_external_test(weights, tokenizer, args, device):
    print("\n========================================")
    print(" EXTERNAL PROMPT VALIDATION")
    print("========================================")

    print(
        "\nThis benchmark is separate from the training data loaders. "
        "Generation uses deterministic greedy decoding."
    )

    answers = {}

    for name, path in weights:
        print("\n========================================")
        print(f" MODEL: {name}")
        print("========================================")
        print(f"Weight: {path}")

        model = load_model(path, args, device)
        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params / 1e6:.2f}M")

        answers[name] = []

        for i, question in enumerate(EXTERNAL_PROMPTS, start=1):
            answer, token_count = generate_answer(
                model=model,
                tokenizer=tokenizer,
                question=question,
                device=device,
                max_new_tokens=args.max_new_tokens,
            )

            repetition = repetition_ratio(answer)
            answers[name].append(
                {
                    "answer": answer,
                    "tokens": token_count,
                    "repetition": repetition,
                }
            )

            print(f"\n[{i}] USER: {question}")
            print(f"ASSISTANT: {answer}")
            print(
                f"[tokens={token_count} | repetition={repetition:.3f}]"
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n========================================")
    print(" EXTERNAL SUMMARY")
    print("========================================")

    for name, _ in weights:
        model_answers = answers[name]
        avg_tokens = sum(x["tokens"] for x in model_answers) / len(model_answers)
        avg_repetition = (
            sum(x["repetition"] for x in model_answers) / len(model_answers)
        )

        print(f"\n{name}")
        print(f"  average generated tokens: {avg_tokens:.1f}")
        print(f"  average repetition      : {avg_repetition:.3f}")

    print(
        "\nRepetition is only a degeneration indicator. "
        "Factual accuracy, reasoning, code correctness, and instruction following "
        "must still be reviewed manually."
    )

    return answers


def choose_mode(cli_mode):
    if cli_mode:
        return cli_mode

    print("\nChoose validation mode:")
    print("[0] Local data validation + training audit")
    print("[1] External prompt validation")
    print("[2] Full validation")

    mapping = {
        "0": "local",
        "1": "external",
        "2": "full",
    }

    while True:
        choice = input("\nChoice: ").strip()
        if choice in mapping:
            return mapping[choice]
        print("Enter 0, 1, or 2.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare MiniMind checkpoints on local losses, checkpoint metadata, "
            "dataset statistics, and an external deterministic prompt benchmark."
        )
    )

    parser.add_argument(
        "--weights",
        nargs="+",
        required=True,
        help="Weights to compare, e.g. --weights pretrain full_sft",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "external", "full"],
        default=None,
        help="Skip the interactive mode selector.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, choices=[0, 1], default=0)

    parser.add_argument("--out_dir", default="out")
    parser.add_argument("--checkpoints_dir", default="checkpoints")
    parser.add_argument("--tokenizer", default="model")

    parser.add_argument(
        "--pretrain_data",
        default="dataset/pretrain_t2t_mini.jsonl",
    )
    parser.add_argument(
        "--sft_data",
        default="dataset/sft_t2t_mini.jsonl",
    )
    parser.add_argument("--pretrain_max_seq_len", type=int, default=340)
    parser.add_argument("--sft_max_seq_len", type=int, default=768)

    parser.add_argument("--eval_samples", type=int, default=256)
    parser.add_argument("--stat_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer_path = resolve_project_path(args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    weights = [resolve_weight(spec, args) for spec in args.weights]
    mode = choose_mode(args.mode)

    print("\n========================================")
    print(" MINIMIND VALIDATION")
    print("========================================")
    print(f"Device: {device}")
    print(f"Mode  : {mode}")

    print("\nWeights:")
    for name, path in weights:
        print(f"  {name}: {path}")

    if mode == "external":
        run_external_test(weights, tokenizer, args, device)
        return

    pretrain_dataset, sft_dataset = run_training_audit(
        weights,
        tokenizer,
        args,
    )

    run_local_test(
        weights,
        tokenizer,
        args,
        device,
        datasets=(pretrain_dataset, sft_dataset),
    )

    if mode == "full":
        run_external_test(weights, tokenizer, args, device)


if __name__ == "__main__":
    main()
