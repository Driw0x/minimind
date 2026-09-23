import os

import pytest
import torch

from trainer.trainer_utils import (
    clear_device_cache,
    get_device,
    init_distributed_mode,
    is_directml_device,
    setup_seed,
)


def test_get_device_auto():
    device = get_device("auto")

    assert device is not None
    assert device.type in {"privateuseone", "cuda", "cpu"}


def test_get_device_directml():
    device = get_device("directml")

    assert device.type == "privateuseone"


def test_is_directml_device():
    device = get_device("directml")

    assert is_directml_device(device)


def test_setup_seed():
    setup_seed(42)

    tensor_a = torch.rand(4)

    setup_seed(42)

    tensor_b = torch.rand(4)

    assert torch.equal(tensor_a, tensor_b)


def test_clear_device_cache():
    device = get_device("directml")

    clear_device_cache(device)


def test_init_distributed_mode_without_rank(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    local_rank = init_distributed_mode("directml")

    assert local_rank == 0


def test_directml_distributed_is_rejected(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")

    with pytest.raises(
        RuntimeError,
        match="Distributed training is not currently supported with DirectML",
    ):
        init_distributed_mode("directml")
        