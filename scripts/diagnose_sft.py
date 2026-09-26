# scripts/diagnose_sft.py

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent


def normalize(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def preview(text, length=220):
    text = normalize(text)
    return text if len(text) <= length else text[:length] + "..."


def text_hash(text):
    return hashlib.blake2b(
        normalize(text).encode("utf-8"),
        digest_size=8,
    ).digest()


def repetition_ratio(text, n=3):
    chars = [c for c in str(text) if not c.isspace()]

    if len(chars) < n:
        return 0.0

    ngrams = [tuple(chars[i:i + n]) for i in range(len(chars) - n + 1)]
    if not ngrams:
        return 0.0

    return 1.0 - len(set(ngrams)) / len(ngrams)


def percentile(values, p):
    return float(np.percentile(values, p)) if values else 0.0


def print_distribution(name, values):
    if not values:
        print(f"{name}: no data")
        return

    print(
        f"{name:28s} "
        f"mean={np.mean(values):8.2f} "
        f"p50={percentile(values, 50):8.2f} "
        f"p90={percentile(values, 90):8.2f} "
        f"p95={percentile(values, 95):8.2f} "
        f"p99={percentile(values, 99):8.2f} "
        f"max={max(values):8.2f}"
    )


def extract_text(conversations, role):
    return "\n".join(
        normalize(message.get("content"))
        for message in conversations
        if message.get("role") == role
        and normalize(message.get("content"))
    )


def has_tools(conversations):
    return any(
        message.get("role") == "tool"
        or message.get("tools")
        or message.get("tool_calls")
        for message in conversations
    )


def suspicious_role_order(conversations):
    roles = [message.get("role", "<missing>") for message in conversations]

    return any(
        roles[i] == roles[i - 1]
        and roles[i] not in {"system", "tool"}
        for i in range(1, len(roles))
    )


def make_chat_prompt(tokenizer, conversations):
    messages = []
    tools = None

    for raw_message in conversations:
        message = dict(raw_message)

        if message.get("role") == "system" and message.get("tools"):
            raw_tools = message["tools"]
            try:
                tools = json.loads(raw_tools) if isinstance(raw_tools, str) else raw_tools
            except Exception:
                tools = None

        if isinstance(message.get("tool_calls"), str):
            try:
                message["tool_calls"] = json.loads(message["tool_calls"])
            except Exception:
                pass

        messages.append(message)

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools,
    )


def duplicate_count(counter):
    return sum(count - 1 for count in counter.values() if count > 1)


class BlockStats:
    def __init__(self, start_line, sample_size, seed):
        self.start_line = start_line
        self.sample_size = sample_size
        self.rng = random.Random(seed + start_line)

        self.count = 0
        self.valid = 0
        self.tools = 0
        self.single_turn = 0
        self.multi_turn = 0
        self.role_suspect = 0

        self.prompt_hashes = Counter()
        self.answer_hashes = Counter()
        self.sample = []

    def add(self, line_number, sample):
        self.count += 1
        conversations = sample.get("conversations")

        if not isinstance(conversations, list) or not conversations:
            return

        self.valid += 1

        if has_tools(conversations):
            self.tools += 1

        user_count = sum(message.get("role") == "user" for message in conversations)
        assistant_count = sum(
            message.get("role") == "assistant" for message in conversations
        )

        if user_count == 1 and assistant_count == 1:
            self.single_turn += 1
        else:
            self.multi_turn += 1

        if suspicious_role_order(conversations):
            self.role_suspect += 1

        user_text = extract_text(conversations, "user")
        assistant_text = extract_text(conversations, "assistant")

        if user_text:
            self.prompt_hashes[text_hash(user_text)] += 1
        if assistant_text:
            self.answer_hashes[text_hash(assistant_text)] += 1

        entry = (line_number, conversations)

        if len(self.sample) < self.sample_size:
            self.sample.append(entry)
        else:
            position = self.rng.randint(1, self.valid)
            if position <= self.sample_size:
                self.sample[position - 1] = entry


def analyze_samples(samples, tokenizer, max_seq_len, title, seed):
    print("\n========================================")
    print(f" {title}")
    print("========================================")

    if not samples:
        print("No samples.")
        return

    conversation_lengths = []
    full_tokens = []
    assistant_tokens = []
    user_tokens = []
    repetition_values = []

    truncated_count = 0
    empty_assistant_count = 0
    no_assistant_count = 0
    malformed_role_order = 0

    prompts = Counter()
    answers = Counter()
    role_patterns = Counter()
    suspicious = []

    for sample_id, sample in samples:
        conversations = sample.get("conversations", [])
        if not isinstance(conversations, list):
            continue

        conversation_lengths.append(len(conversations))

        roles = tuple(
            message.get("role", "<missing>")
            for message in conversations
        )
        role_patterns[roles] += 1

        if suspicious_role_order(conversations):
            malformed_role_order += 1

        assistant_messages = [
            message
            for message in conversations
            if message.get("role") == "assistant"
        ]

        if not assistant_messages:
            no_assistant_count += 1

        for message in assistant_messages:
            content = normalize(message.get("content"))
            tool_calls = message.get("tool_calls")
            if not content and not tool_calls:
                empty_assistant_count += 1

        user_text = extract_text(conversations, "user")
        assistant_text = extract_text(conversations, "assistant")

        if user_text:
            prompts[user_text] += 1
            user_tokens.append(
                len(tokenizer(user_text, add_special_tokens=False).input_ids)
            )

        if assistant_text:
            answers[assistant_text] += 1
            assistant_tokens.append(
                len(tokenizer(assistant_text, add_special_tokens=False).input_ids)
            )

            rep = repetition_ratio(assistant_text)
            repetition_values.append(rep)

            if rep >= 0.60:
                suspicious.append(
                    (
                        "HIGH_REPETITION",
                        sample_id,
                        rep,
                        user_text,
                        assistant_text,
                    )
                )

        try:
            prompt = make_chat_prompt(tokenizer, conversations)
            token_count = len(
                tokenizer(prompt, add_special_tokens=True).input_ids
            )
            full_tokens.append(token_count)

            if token_count > max_seq_len:
                truncated_count += 1
                suspicious.append(
                    (
                        "TRUNCATED",
                        sample_id,
                        token_count,
                        user_text,
                        assistant_text,
                    )
                )
        except Exception as exc:
            suspicious.append(
                (
                    "TEMPLATE_ERROR",
                    sample_id,
                    str(exc),
                    user_text,
                    assistant_text,
                )
            )

    count = len(samples)

    print(f"\nSamples analyzed: {count:,}")

    print("\n--- LENGTHS ---")
    print_distribution("messages / conversation", conversation_lengths)
    print_distribution("user tokens", user_tokens)
    print_distribution("assistant tokens", assistant_tokens)
    print_distribution("conversation tokens", full_tokens)

    print(f"\nMax sequence length: {max_seq_len}")
    print(
        f"Truncated conversations: {truncated_count:,} / {count:,} "
        f"({100 * truncated_count / max(count, 1):.2f}%)"
    )

    print("\n--- STRUCTURE ---")
    print(f"No assistant message : {no_assistant_count:,}")
    print(f"Empty assistant       : {empty_assistant_count:,}")
    print(f"Suspicious role order : {malformed_role_order:,}")

    print("\nMost common role patterns:")
    for pattern, occurrences in role_patterns.most_common(10):
        print(f"  {occurrences:6d}  {' -> '.join(pattern)}")

    print("\n--- REPETITION ---")
    print_distribution("assistant repetition", repetition_values)

    high_rep = sum(value >= 0.60 for value in repetition_values)
    if repetition_values:
        print(
            f"\nResponses with repetition >= 0.60: "
            f"{high_rep:,} / {len(repetition_values):,} "
            f"({100 * high_rep / len(repetition_values):.2f}%)"
        )

    print("\n--- DUPLICATION ---")
    print(f"Duplicate prompts   : {duplicate_count(prompts):,}")
    print(f"Duplicate responses : {duplicate_count(answers):,}")

    print("\nMost repeated exact responses:")
    shown = 0
    for answer, occurrences in answers.most_common():
        if occurrences <= 1:
            break

        print(f"\n  x{occurrences}")
        print(f"  {preview(answer)}")
        shown += 1

        if shown >= 5:
            break

    if shown == 0:
        print("  none in this sample")

    print("\n--- SUSPICIOUS EXAMPLES ---")
    random.Random(seed).shuffle(suspicious)

    for issue, sample_id, value, user, assistant in suspicious[:10]:
        print(f"\n[{issue}] sample={sample_id} value={value}")
        print("USER:")
        print(preview(user))
        print("ASSISTANT:")
        print(preview(assistant))


def analyze_block_token_sample(stats, tokenizer, max_seq_len):
    conversation_tokens = []
    assistant_tokens = []
    repetition = []

    truncated = 0
    template_errors = 0

    for _, conversations in stats.sample:
        assistant_text = extract_text(conversations, "assistant")

        if assistant_text:
            assistant_tokens.append(
                len(tokenizer(assistant_text, add_special_tokens=False).input_ids)
            )
            repetition.append(repetition_ratio(assistant_text))

        try:
            prompt = make_chat_prompt(tokenizer, conversations)
            token_count = len(
                tokenizer(prompt, add_special_tokens=True).input_ids
            )
            conversation_tokens.append(token_count)

            if token_count > max_seq_len:
                truncated += 1
        except Exception:
            template_errors += 1

    return {
        "sample_n": len(stats.sample),
        "conv_mean": float(np.mean(conversation_tokens)) if conversation_tokens else 0.0,
        "conv_p50": percentile(conversation_tokens, 50),
        "conv_p95": percentile(conversation_tokens, 95),
        "assistant_mean": float(np.mean(assistant_tokens)) if assistant_tokens else 0.0,
        "truncated_pct": (
            100 * truncated / len(conversation_tokens)
            if conversation_tokens
            else 0.0
        ),
        "rep_mean": float(np.mean(repetition)) if repetition else 0.0,
        "rep_p95": percentile(repetition, 95),
        "rep_high_pct": (
            100 * sum(value >= 0.60 for value in repetition) / len(repetition)
            if repetition
            else 0.0
        ),
        "template_errors": template_errors,
    }


def print_block(index, stats, token_stats):
    start = stats.start_line
    end = start + stats.count - 1
    valid = max(stats.valid, 1)

    print("\n" + "=" * 72)
    print(f" BLOCK {index} | lines {start:,} - {end:,}")
    print("=" * 72)

    print(f"Samples                 : {stats.valid:,}")
    print(f"Single-turn             : {100 * stats.single_turn / valid:6.2f}%")
    print(f"Multi-turn              : {100 * stats.multi_turn / valid:6.2f}%")
    print(f"Tools                   : {100 * stats.tools / valid:6.2f}%")
    print(f"Suspicious role order   : {stats.role_suspect:,}")
    print(f"Duplicate prompts       : {duplicate_count(stats.prompt_hashes):,}")
    print(f"Duplicate responses     : {duplicate_count(stats.answer_hashes):,}")

    print(f"\nToken sample            : {token_stats['sample_n']:,}")
    print(f"Conversation tokens mean: {token_stats['conv_mean']:.1f}")
    print(f"Conversation tokens p50 : {token_stats['conv_p50']:.1f}")
    print(f"Conversation tokens p95 : {token_stats['conv_p95']:.1f}")
    print(f"Assistant tokens mean   : {token_stats['assistant_mean']:.1f}")
    print(f"Truncated > max_seq     : {token_stats['truncated_pct']:.2f}%")
    print(f"Repetition mean         : {token_stats['rep_mean']:.3f}")
    print(f"Repetition p95          : {token_stats['rep_p95']:.3f}")
    print(f"Repetition >= 0.60      : {token_stats['rep_high_pct']:.2f}%")


def print_block_summary(blocks, tokenizer, max_seq_len):
    print("\n========================================")
    print(" BLOCK ANALYSIS")
    print("========================================")

    summary = []

    for index, stats in enumerate(blocks, start=1):
        token_stats = analyze_block_token_sample(
            stats,
            tokenizer,
            max_seq_len,
        )
        print_block(index, stats, token_stats)

        summary.append(
            {
                "block": index,
                "start": stats.start_line,
                "end": stats.start_line + stats.count - 1,
                "samples": stats.valid,
                "single": 100 * stats.single_turn / max(stats.valid, 1),
                "tools": 100 * stats.tools / max(stats.valid, 1),
                "tokens": token_stats["conv_mean"],
                "truncated": token_stats["truncated_pct"],
                "repetition": token_stats["rep_mean"],
            }
        )

    print("\n" + "=" * 100)
    print(" BLOCK SUMMARY")
    print("=" * 100)
    print(
        f"{'Block':<7}"
        f"{'Lines':<22}"
        f"{'Single%':>9}"
        f"{'Tools%':>9}"
        f"{'Tokens':>10}"
        f"{'Trunc%':>9}"
        f"{'Rep':>9}"
    )
    print("-" * 100)

    for item in summary:
        line_range = f"{item['start']:,}-{item['end']:,}"
        print(
            f"{item['block']:<7}"
            f"{line_range:<22}"
            f"{item['single']:>8.2f}%"
            f"{item['tools']:>8.2f}%"
            f"{item['tokens']:>10.1f}"
            f"{item['truncated']:>8.2f}%"
            f"{item['repetition']:>9.3f}"
        )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Audit MiniMind SFT JSONL quality, compare first/random samples, "
            "and analyze source distribution by file blocks."
        )
    )

    parser.add_argument("--data", default="dataset/sft_t2t_mini.jsonl")
    parser.add_argument("--tokenizer", default="model")
    parser.add_argument(
        "--mode",
        choices=["dataset", "blocks", "all"],
        default="all",
        help="Run detailed dataset analysis, block analysis, or both.",
    )
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--first_samples", type=int, default=5000)
    parser.add_argument("--block_size", type=int, default=100_000)
    parser.add_argument("--sample_per_block", type=int, default=2000)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path

    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.is_absolute():
        tokenizer_path = ROOT / tokenizer_path

    if not data_path.exists():
        raise FileNotFoundError(data_path)

    print("\n========================================")
    print(" MINIMIND SFT DATA DIAGNOSTIC")
    print("========================================")
    print(f"Dataset          : {data_path}")
    print(f"Tokenizer        : {tokenizer_path}")
    print(f"Mode             : {args.mode}")
    print(f"Max sequence len : {args.max_seq_len}")
    print(f"Block size       : {args.block_size:,}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print("Scanning JSONL...")

    rng = random.Random(args.seed)
    first_samples = []
    random_samples = []
    blocks = []

    total_lines = 0
    valid_lines = 0
    invalid_json = 0
    missing_conversations = 0
    empty_conversations = 0
    tool_conversations = 0
    global_roles = Counter()

    current_block = BlockStats(
        start_line=1,
        sample_size=args.sample_per_block,
        seed=args.seed,
    )

    with open(data_path, "r", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            if current_block.count >= args.block_size:
                blocks.append(current_block)
                current_block = BlockStats(
                    start_line=line_number,
                    sample_size=args.sample_per_block,
                    seed=args.seed,
                )

            total_lines += 1

            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                current_block.count += 1
                continue

            current_block.add(line_number, sample)

            conversations = sample.get("conversations")
            if not isinstance(conversations, list):
                missing_conversations += 1
                continue

            if not conversations:
                empty_conversations += 1
                continue

            valid_lines += 1

            if has_tools(conversations):
                tool_conversations += 1

            for message in conversations:
                global_roles[message.get("role", "<missing>")] += 1

            entry = (line_number, sample)

            if len(first_samples) < args.first_samples:
                first_samples.append(entry)

            if len(random_samples) < args.samples:
                random_samples.append(entry)
            else:
                position = rng.randint(1, valid_lines)
                if position <= args.samples:
                    random_samples[position - 1] = entry

    if current_block.count:
        blocks.append(current_block)

    print("\n========================================")
    print(" DATASET GLOBAL")
    print("========================================")
    print(f"Total lines               : {total_lines:,}")
    print(f"Valid lines               : {valid_lines:,}")
    print(f"Invalid JSON              : {invalid_json:,}")
    print(f"Missing conversations     : {missing_conversations:,}")
    print(f"Empty conversations       : {empty_conversations:,}")
    print(f"Conversations with tools  : {tool_conversations:,}")

    if valid_lines:
        print(f"Tool conversation share   : {100 * tool_conversations / valid_lines:.2f}%")

    print("\nRoles:")
    for role, count in global_roles.most_common():
        print(f"  {role:12s}: {count:,}")

    if args.mode in {"dataset", "all"}:
        analyze_samples(
            first_samples,
            tokenizer,
            args.max_seq_len,
            f"FIRST {len(first_samples):,} SAMPLES",
            args.seed,
        )

        analyze_samples(
            random_samples,
            tokenizer,
            args.max_seq_len,
            f"{len(random_samples):,} RANDOM SAMPLES",
            args.seed,
        )

    if args.mode in {"blocks", "all"}:
        print_block_summary(
            blocks,
            tokenizer,
            args.max_seq_len,
        )

    print("\n========================================")
    print(" INTERPRETATION GUIDE")
    print("========================================")
    print(
        """
- Large FIRST vs RANDOM differences indicate an ordered or heterogeneous source file.
- High truncation means many conversations exceed the configured training context length.
- High repetition can teach repetitive answer patterns.
- High duplicate counts indicate source/template over-representation.
- Abrupt block changes in Single%, Tools%, token length, or truncation indicate source boundaries.
- Physical file ordering is not a training-order problem when the trainer shuffles indices.
"""
    )


if __name__ == "__main__":
    main()
