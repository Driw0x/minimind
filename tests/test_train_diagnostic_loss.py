import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import get_device, prepare_model_precision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=340)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--weight", default="pretrain")
    parser.add_argument("--dataset", default="dataset/pretrain_t2t_mini.jsonl")
    parser.add_argument("--tokenizer", default="model")
    args = parser.parse_args()

    device = get_device(args.device)
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    dtype = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        local_files_only=True
    )

    dataset = PretrainDataset(
        args.dataset,
        tokenizer,
        max_length=args.max_seq_len
    )

    indices = list(
        range(
            args.sample_index,
            min(args.sample_index + args.batch_size, len(dataset))
        )
    )

    samples = [dataset[i] for i in indices]
    input_ids = torch.stack([sample[0] for sample in samples]).to(device)
    labels = torch.stack([sample[1] for sample in samples]).to(device)

    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )

    model = MiniMindForCausalLM(config)
    model = prepare_model_precision(model, device, dtype)

    weight_path = os.path.join(
        ROOT,
        "out",
        f"{args.weight}_{args.hidden_size}.pth"
    )

    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        model_loss = outputs.loss.float()

        logits = outputs.logits.float()
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()

        manual_loss = F.cross_entropy(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
            ignore_index=-100
        )

    difference = abs(model_loss.item() - manual_loss.item())

    print("=" * 60)
    print("Train vs diagnostic loss comparison")
    print("=" * 60)
    print(f"Checkpoint: {weight_path}")
    print(f"Samples: {indices[0]} -> {indices[-1]}")
    print(f"Batch size: {len(indices)}")
    print(f"Max sequence length: {args.max_seq_len}")
    print(f"Valid tokens: {(shifted_labels != -100).sum().item()}")
    print()
    print(f"Model loss:  {model_loss.item():.8f}")
    print(f"Manual loss: {manual_loss.item():.8f}")
    print(f"Difference:  {difference:.10f}")
    print()
    print("MATCH" if difference < 1e-5 else "MISMATCH")


if __name__ == "__main__":
    main()