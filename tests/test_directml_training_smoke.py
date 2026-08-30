from __future__ import annotations

import torch
import torch.nn.functional as F

from dataset.lm_dataset import PretrainDataset, DPODataset
from model.model_minimind import MiniMindForCausalLM
from trainer.train_distillation import distillation_loss


FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


def _device_name(device):
    return str(device).lower()


def test_pretrain_forward_backward_optimizer_on_directml(tokenizer, tiny_config, directml_device):
    model = MiniMindForCausalLM(tiny_config).to(directml_device)
    model.train()

    ds = PretrainDataset(
        [str(FIXTURES / "tiny_pretrain.jsonl")],
        tokenizer,
        max_length=64
    )
    input_ids, labels = ds[0]
    input_ids = input_ids.unsqueeze(0).to(directml_device)
    labels = labels.unsqueeze(0).to(directml_device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)

    out = model(input_ids)
    logits = out.logits[:, :-1, :]
    targets = labels[:, 1:]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
    )

    assert torch.isfinite(loss).item()
    loss.backward()

    grad_found = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert grad_found

    optimizer.step()


def _sequence_logp(model, x, y, mask):
    logits = model(x).logits
    token_logps = F.log_softmax(logits, dim=-1).gather(
        2, y.unsqueeze(-1)
    ).squeeze(-1)
    mask = mask.float()
    return (token_logps * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)


def test_dpo_forward_backward_on_directml(tokenizer, tiny_config, directml_device):
    policy = MiniMindForCausalLM(tiny_config).to(directml_device)
    reference = MiniMindForCausalLM(tiny_config).to(directml_device)
    reference.load_state_dict(policy.state_dict())
    reference.eval().requires_grad_(False)

    ds = DPODataset(
        str(FIXTURES / "tiny_dpo.jsonl"),
        tokenizer,
        max_length=64
    )
    s = ds[0]

    xc = s["x_chosen"].unsqueeze(0).to(directml_device)
    yc = s["y_chosen"].unsqueeze(0).to(directml_device)
    mc = s["mask_chosen"].unsqueeze(0).to(directml_device)
    xr = s["x_rejected"].unsqueeze(0).to(directml_device)
    yr = s["y_rejected"].unsqueeze(0).to(directml_device)
    mr = s["mask_rejected"].unsqueeze(0).to(directml_device)

    pi_c = _sequence_logp(policy, xc, yc, mc)
    pi_r = _sequence_logp(policy, xr, yr, mr)
    with torch.no_grad():
        ref_c = _sequence_logp(reference, xc, yc, mc)
        ref_r = _sequence_logp(reference, xr, yr, mr)

    beta = 0.1
    logits = beta * ((pi_c - pi_r) - (ref_c - ref_r))
    loss = -F.logsigmoid(logits).mean()

    assert torch.isfinite(loss).item()
    loss.backward()
    assert any(p.grad is not None for p in policy.parameters() if p.requires_grad)


def test_distillation_loss_backward_on_directml(tiny_config, directml_device):
    student = MiniMindForCausalLM(tiny_config).to(directml_device)
    teacher = MiniMindForCausalLM(tiny_config).to(directml_device)
    teacher.eval().requires_grad_(False)

    vocab = tiny_config.vocab_size
    input_ids = torch.randint(
        low=0,
        high=min(vocab, 128),
        size=(1, 16),
        dtype=torch.long,
        device=directml_device,
    )

    student_logits = student(input_ids).logits[:, :-1, :]
    with torch.no_grad():
        teacher_logits = teacher(input_ids).logits[:, :-1, :]

    loss = distillation_loss(
        student_logits.reshape(-1, student_logits.size(-1)),
        teacher_logits.reshape(-1, teacher_logits.size(-1)),
        temperature=1.5,
    )

    assert torch.isfinite(loss).item()
    loss.backward()
    assert any(p.grad is not None for p in student.parameters() if p.requires_grad)
