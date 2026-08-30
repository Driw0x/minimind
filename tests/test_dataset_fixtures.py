from pathlib import Path

import torch

from dataset.lm_dataset import (
    PretrainDataset,
    SFTDataset,
    DPODataset,
    RLAIFDataset,
    AgentRLDataset,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_pretrain_fixture(tokenizer):
    ds = PretrainDataset(
        [str(FIXTURES / "tiny_pretrain.jsonl")],
        tokenizer,
        max_length=64
    )
    x, y = ds[0]
    assert x.shape == y.shape == (64,)
    assert x.dtype == torch.long
    assert (y != -100).any()


def test_sft_fixture(tokenizer):
    ds = SFTDataset(
        str(FIXTURES / "tiny_sft.jsonl"),
        tokenizer,
        max_length=64
    )
    x, y = ds[0]
    assert x.shape == y.shape == (64,)
    assert x.dtype == y.dtype == torch.long
    assert (y != -100).any()


def test_dpo_fixture(tokenizer):
    ds = DPODataset(
        str(FIXTURES / "tiny_dpo.jsonl"),
        tokenizer,
        max_length=64
    )
    sample = ds[0]
    assert set(sample) == {
        "x_chosen", "y_chosen", "mask_chosen",
        "x_rejected", "y_rejected", "mask_rejected",
    }
    assert sample["x_chosen"].shape == (63,)
    assert sample["x_rejected"].shape == (63,)
    assert sample["mask_chosen"].sum() > 0
    assert sample["mask_rejected"].sum() > 0


def test_rlaif_fixture(tokenizer):
    ds = RLAIFDataset(
        str(FIXTURES / "tiny_rlaif.jsonl"),
        tokenizer,
        max_length=64,
        thinking_ratio=0.0
    )
    sample = ds[0]
    assert isinstance(sample["prompt"], str)
    assert sample["prompt"]
    assert sample["answer"] == ""


def test_agent_fixture(tokenizer):
    ds = AgentRLDataset(FIXTURES / "tiny_agent_rl.jsonl", tokenizer, max_length=64)
    sample = ds[0]
    assert isinstance(sample["messages"], list)
    assert isinstance(sample["gt"], list)
