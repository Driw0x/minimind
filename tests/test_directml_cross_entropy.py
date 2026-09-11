import torch
import torch.nn.functional as F
import torch_directml


def test(device):
    logits = torch.randn(8, 339, 6400, device=device)
    targets = torch.randint(0, 6400, (8, 339), device=device)

    targets[:, 200:] = -100

    x = logits.view(-1, logits.size(-1))
    y = targets.view(-1)

    mean_loss = F.cross_entropy(x, y, ignore_index=-100)
    sum_loss = F.cross_entropy(x, y, ignore_index=-100, reduction="sum")
    valid = (y != -100).sum()

    correct_loss = sum_loss / valid

    print(f"Device: {device}")
    print(f"Total positions: {y.numel()}")
    print(f"Valid positions: {valid.item()}")
    print(f"CE mean: {mean_loss.item():.6f}")
    print(f"CE sum / valid: {correct_loss.item():.6f}")
    print()


test(torch.device("cpu"))
test(torch_directml.device(1))