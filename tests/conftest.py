from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def repo_root():
    return ROOT


@pytest.fixture(scope="session")
def tokenizer(repo_root):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(repo_root / "model", trust_remote_code=True)


@pytest.fixture(scope="session")
def directml_device():
    try:
        import torch_directml
        return torch_directml.device()
    except Exception as exc:
        pytest.skip(f"DirectML unavailable: {exc}")


@pytest.fixture()
def tiny_config():
    from model.model_minimind import MiniMindConfig
    return MiniMindConfig(
        hidden_size=64,
        num_hidden_layers=2,
        max_seq_len=64,
        use_moe=False,
    )
