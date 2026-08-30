from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def take_valid(path: Path, validator: Callable[[dict], bool], count: int):
    rows = []
    if path.exists():
        for row in read_jsonl(path):
            if validator(row):
                rows.append(row)
                if len(rows) >= count:
                    break
    return rows


def valid_pretrain(x):
    return isinstance(x.get("text"), str) and bool(x["text"].strip())


def valid_sft(x):
    conv = x.get("conversations")
    return (
        isinstance(conv, list)
        and len(conv) >= 2
        and all(isinstance(m, dict) and m.get("role") and "content" in m for m in conv)
    )


def valid_dpo(x):
    return (
        isinstance(x.get("chosen"), list)
        and isinstance(x.get("rejected"), list)
        and len(x["chosen"]) >= 2
        and len(x["rejected"]) >= 2
    )


def valid_rlaif(x):
    conv = x.get("conversations")
    return isinstance(conv, list) and len(conv) >= 2


def valid_agent(x):
    conv = x.get("conversations")
    return (
        isinstance(conv, list)
        and len(conv) >= 2
        and isinstance(x.get("gt"), list)
    )


FALLBACKS = {
    "pretrain": [
        {"text": "MiniMind is a compact language model used for training experiments."},
        {"text": "DirectML allows PyTorch workloads to run on compatible Windows GPUs."},
    ],
    "sft": [
        {
            "conversations": [
                {"role": "user", "content": "What is 2 + 2?", "reasoning_content": "", "tools": "", "tool_calls": ""},
                {"role": "assistant", "content": "4", "reasoning_content": "", "tools": "", "tool_calls": ""},
            ]
        },
        {
            "conversations": [
                {"role": "user", "content": "Say hello.", "reasoning_content": "", "tools": "", "tool_calls": ""},
                {"role": "assistant", "content": "Hello!", "reasoning_content": "", "tools": "", "tool_calls": ""},
            ]
        },
    ],
    "dpo": [
        {
            "chosen": [
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "2 + 2 = 4."},
            ],
            "rejected": [
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "2 + 2 = 5."},
            ],
        }
    ],
    "rlaif": [
        {
            "conversations": [
                {"role": "user", "content": "Give a short greeting."},
                {"role": "assistant", "content": "Hello!"},
            ]
        }
    ],
    "agent": [
        {
            "conversations": [
                {
                    "role": "system",
                    "content": "You can use tools.",
                    "tools": json.dumps(
                        [
                            {
                                "type": "function",
                                "function": {
                                    "name": "calculate_math",
                                    "description": "Calculate a mathematical expression",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {"expression": {"type": "string"}},
                                        "required": ["expression"],
                                    },
                                },
                            }
                        ],
                        ensure_ascii=False,
                    ),
                },
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ],
            "gt": ["4"],
        }
    ],
}


def normalize_sft(rows):
    # SFTDataset declares a fixed Arrow schema. Make optional fields explicit.
    normalized = []
    for row in rows:
        convs = []
        for msg in row["conversations"]:
            msg = dict(msg)
            convs.append(
                {
                    "role": str(msg.get("role", "")),
                    "content": str(msg.get("content", "")),
                    "reasoning_content": str(msg.get("reasoning_content") or ""),
                    "tools": msg.get("tools") if isinstance(msg.get("tools"), str) else (
                        json.dumps(msg.get("tools"), ensure_ascii=False) if msg.get("tools") else ""
                    ),
                    "tool_calls": msg.get("tool_calls") if isinstance(msg.get("tool_calls"), str) else (
                        json.dumps(msg.get("tool_calls"), ensure_ascii=False) if msg.get("tool_calls") else ""
                    ),
                }
            )
        normalized.append({"conversations": convs})
    return normalized


def main():
    parser = argparse.ArgumentParser(description="Generate tiny JSONL fixtures from MiniMind datasets.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("tests/fixtures"))
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args()

    specs = {
        "pretrain": ("pretrain_t2t_mini.jsonl", valid_pretrain, "tiny_pretrain.jsonl"),
        "sft": ("sft_t2t_mini.jsonl", valid_sft, "tiny_sft.jsonl"),
        "dpo": ("dpo.jsonl", valid_dpo, "tiny_dpo.jsonl"),
        "rlaif": ("rlaif.jsonl", valid_rlaif, "tiny_rlaif.jsonl"),
        "agent": ("agent_rl.jsonl", valid_agent, "tiny_agent_rl.jsonl"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, (source_name, validator, out_name) in specs.items():
        source = args.dataset_dir / source_name
        rows = take_valid(source, validator, args.samples)
        source_kind = str(source)

        if not rows:
            rows = FALLBACKS[name][: args.samples]
            source_kind = "built-in fallback"

        if name == "sft":
            rows = normalize_sft(rows)

        out = args.output_dir / out_name
        write_jsonl(out, rows)
        print(f"[OK] {out} <- {source_kind} ({len(rows)} sample(s))")


if __name__ == "__main__":
    main()
