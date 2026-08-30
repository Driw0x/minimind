import pytest
import torch
import torch_directml

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


@pytest.fixture(scope="module")
def dml_device():
    """Return the DirectML device used by the tests."""
    return torch_directml.device()


@pytest.fixture(scope="module")
def model(dml_device):
    """Create a small MiniMind model for DirectML smoke tests."""
    config = MiniMindConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=6400,
    )

    model = MiniMindForCausalLM(config)
    return model.to(dml_device)


@pytest.fixture
def input_ids(dml_device):
    """Create a small random input tensor directly on DirectML."""
    return torch.randint(
        low=0,
        high=6400,
        size=(1, 16),
        dtype=torch.long,
        device=dml_device,
    )


def test_directml_device(dml_device):
    assert dml_device is not None
    assert str(dml_device) == "privateuseone:0"


def test_model_on_directml(model):
    model_device = next(model.parameters()).device

    assert model_device.type == "privateuseone"


def test_input_on_directml(input_ids):
    assert input_ids.device.type == "privateuseone"
    assert input_ids.shape == (1, 16)


def test_forward_pass(model, input_ids):
    outputs = model(input_ids)

    logits = outputs.logits

    assert logits is not None
    assert logits.shape == (1, 16, 6400)
    assert logits.device.type == "privateuseone"
    assert torch.isfinite(logits).all().item()


def test_backward_pass(model, input_ids):
    model.zero_grad()

    outputs = model(input_ids)

    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    loss_fn = torch.nn.CrossEntropyLoss()

    loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )

    assert torch.isfinite(loss).item()

    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]

    assert gradients


def test_optimizer_step(model, input_ids):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    optimizer.zero_grad()

    outputs = model(input_ids)

    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    loss_fn = torch.nn.CrossEntropyLoss()

    loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )

    assert torch.isfinite(loss).item()

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    assert True