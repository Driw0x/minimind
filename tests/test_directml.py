import torch
import torch_directml

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def main():
    device = torch_directml.device()

    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")

    config = MiniMindConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
    )

    model = MiniMindForCausalLM(config).to(device)

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (1, 16),
        device=device,
    )

    labels = torch.randint(
        0,
        config.vocab_size,
        (1, 16),
        device=device,
    )

    print(f"Input device: {input_ids.device}")
    print(f"Model device: {next(model.parameters()).device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    output = model(
        input_ids=input_ids,
        labels=labels,
    )

    print(f"Logits shape: {output.logits.shape}")
    print(f"Logits device: {output.logits.device}")
    print(f"Loss: {output.loss.item()}")

    output.loss.backward()

    print("DirectML forward pass: OK")
    print("DirectML backward pass: OK")

    optimizer.step()

    print("DirectML optimizer step: OK")

    optimizer.zero_grad()

    print("DirectML zero_grad: OK")
    print("DirectML training step: OK")


if __name__ == "__main__":
    main()