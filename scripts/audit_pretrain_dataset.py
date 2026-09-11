import argparse
import json
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max_seq_len", type=int, default=340)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    total = 0
    empty = 0
    truncated = 0
    literal_unk = 0
    url_like = 0
    html_like = 0

    token_lengths = []
    exact_counts = Counter()
    prefixes = Counter()

    url_re = re.compile(r"https?://|www\.", re.IGNORECASE)
    html_re = re.compile(r"<[^>]+>|&[a-zA-Z]+;")

    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            if args.max_samples and total >= args.max_samples:
                break

            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = str(sample.get("text", ""))
            total += 1

            if not text.strip():
                empty += 1

            exact_counts[text] += 1
            prefixes[text[:30]] += 1

            if "<UNK>" in text:
                literal_unk += 1

            if url_re.search(text):
                url_like += 1

            if html_re.search(text):
                html_like += 1

            ids = tokenizer(
                text,
                add_special_tokens=False,
                truncation=False
            ).input_ids

            token_lengths.append(len(ids))

            if len(ids) > args.max_seq_len - 2:
                truncated += 1

    if total == 0:
        print("No valid samples found.")
        return

    token_lengths.sort()

    def percentile(p):
        index = int((len(token_lengths) - 1) * p)
        return token_lengths[index]

    unique = len(exact_counts)
    duplicate_samples = total - unique

    print("=" * 60)
    print("Pretrain dataset audit")
    print("=" * 60)
    print(f"Samples: {total}")
    print(f"Unique texts: {unique}")
    print(f"Duplicate samples: {duplicate_samples} ({duplicate_samples / total * 100:.2f}%)")
    print(f"Empty texts: {empty} ({empty / total * 100:.2f}%)")
    print(f"Literal <UNK>: {literal_unk} ({literal_unk / total * 100:.2f}%)")
    print(f"URL-like: {url_like} ({url_like / total * 100:.2f}%)")
    print(f"HTML-like: {html_like} ({html_like / total * 100:.2f}%)")
    print()
    print(f"Max sequence length: {args.max_seq_len}")
    print(f"Content limit: {args.max_seq_len - 2}")
    print(f"Truncated samples: {truncated} ({truncated / total * 100:.2f}%)")
    print()
    print("Token length distribution:")
    print(f"Min: {token_lengths[0]}")
    print(f"P50: {percentile(0.50)}")
    print(f"P75: {percentile(0.75)}")
    print(f"P90: {percentile(0.90)}")
    print(f"P95: {percentile(0.95)}")
    print(f"P99: {percentile(0.99)}")
    print(f"Max: {token_lengths[-1]}")
    print()
    print("Most duplicated texts:")
    for text, count in exact_counts.most_common(10):
        if count <= 1:
            break
        preview = text[:120].replace("\n", " ")
        print(f"{count}x | {preview}")
    print()
    print("Most common prefixes:")
    for prefix, count in prefixes.most_common(10):
        preview = prefix.replace("\n", " ")
        print(f"{count}x | {preview}")


if __name__ == "__main__":
    main()