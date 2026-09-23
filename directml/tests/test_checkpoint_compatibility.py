from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from eval_llm import init_model
from trainer.trainer_utils import get_device


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "out"

CHECKPOINTS = [
    ("pretrain", None),
    ("full_sft", None),
    ("dpo", None),
    ("grpo", None),
    ("ppo_actor", None),
    ("agent", None),
    ("full_dist", None),
    ("full_sft", "lora_identity"),
]


@pytest.mark.parametrize("weight,lora_weight", CHECKPOINTS)
def test_checkpoint_compatibility(weight, lora_weight):
    hidden_size = 64
    num_hidden_layers = 2

    checkpoint = OUT_DIR / f"{weight}_{hidden_size}.pth"

    if not checkpoint.exists():
        pytest.skip(f"Checkpoint not found: {checkpoint}")

    if lora_weight:
        lora_checkpoint = OUT_DIR / f"{lora_weight}_{hidden_size}.pth"

        if not lora_checkpoint.exists():
            pytest.skip(f"LoRA checkpoint not found: {lora_checkpoint}")

    device = get_device("directml")

    args = SimpleNamespace(
        load_from="model",
        save_dir="out",
        weight=weight,
        lora_weight=lora_weight or "None",
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        use_moe=0,
        inference_rope_scaling=False,
        device=device,
    )

    model, tokenizer = init_model(args)

    if weight == "pretrain":
        prompt = tokenizer.bos_token + "Hello"
    else:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            tokenize=False,
            add_generation_prompt=True,
            open_thinking=False,
        )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = (
        output.shape[1] - inputs["input_ids"].shape[1]
    )

    assert generated_tokens > 0