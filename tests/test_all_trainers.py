import os
import subprocess
import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
TRAINER_DIR = ROOT_DIR / "trainer"
OUT_DIR = ROOT_DIR / "out"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

PYTHON = sys.executable
TEST_PREFIX = "test_directml_"


TRAINER_TESTS = [
    (
        "Dense Pretrain",
        [
            "train_pretrain.py",
            "--device", "directml:1",
            "--dtype", "float16",
            "--hidden_size", "768",
            "--num_hidden_layers", "8",
            "--use_moe", "0",
            "--batch_size", "2",
            "--max_seq_len", "128",
            "--accumulation_steps", "8",
            "--directml_loss_scale", "1024",
            "--directml_adam_eps", "1e-4",
            "--max_steps", "9",
            "--log_interval", "1",
            "--save_interval", "8",
            "--from_weight", "none",
            "--save_weight", f"{TEST_PREFIX}pretrain",
        ],
    ),
    (
        "Dense Full SFT",
        [
            "train_full_sft.py",
            "--device", "directml:1",
            "--dtype", "float16",
            "--hidden_size", "768",
            "--num_hidden_layers", "8",
            "--use_moe", "0",
            "--batch_size", "2",
            "--max_seq_len", "128",
            "--accumulation_steps", "8",
            "--directml_loss_scale", "1024",
            "--directml_adam_eps", "1e-4",
            "--max_steps", "9",
            "--log_interval", "1",
            "--save_interval", "8",
            "--from_weight", f"{TEST_PREFIX}pretrain",
            "--save_weight", f"{TEST_PREFIX}full_sft",
        ],
    ),
    (
        "MoE Pretrain",
        [
            "train_pretrain.py",
            "--device", "directml:1",
            "--dtype", "float16",
            "--hidden_size", "768",
            "--num_hidden_layers", "8",
            "--use_moe", "1",
            "--batch_size", "1",
            "--max_seq_len", "64",
            "--accumulation_steps", "8",
            "--directml_loss_scale", "1024",
            "--directml_adam_eps", "1e-4",
            "--max_steps", "9",
            "--log_interval", "1",
            "--save_interval", "8",
            "--from_weight", "none",
            "--save_weight", f"{TEST_PREFIX}pretrain",
        ],
    ),
    (
        "MoE Full SFT",
        [
            "train_full_sft.py",
            "--device", "directml:1",
            "--dtype", "float16",
            "--hidden_size", "768",
            "--num_hidden_layers", "8",
            "--use_moe", "1",
            "--batch_size", "1",
            "--max_seq_len", "64",
            "--accumulation_steps", "8",
            "--directml_loss_scale", "1024",
            "--directml_adam_eps", "1e-4",
            "--max_steps", "9",
            "--log_interval", "1",
            "--save_interval", "8",
            "--from_weight", f"{TEST_PREFIX}pretrain",
            "--save_weight", f"{TEST_PREFIX}full_sft",
        ],
    ),
    (
        "LoRA",
        [
            "train_lora.py",
            "--device", "directml:1",
            "--dtype", "float16",
            "--hidden_size", "768",
            "--num_hidden_layers", "8",
            "--use_moe", "0",
            "--batch_size", "2",
            "--max_seq_len", "128",
            "--accumulation_steps", "8",
            "--directml_loss_scale", "1024",
            "--directml_adam_eps", "1e-4",
            "--max_steps", "9",
            "--log_interval", "1",
            "--save_interval", "8",
            "--from_weight", f"{TEST_PREFIX}full_sft",
            "--lora_name", f"{TEST_PREFIX}lora",
        ],
    ),
    (
        "Distillation (Dense student <- MoE teacher)",
        [
            "train_distillation.py",
            "--device", "directml:1",
            "--dtype", "float16",
            "--batch_size", "1",
            "--max_seq_len", "64",
            "--accumulation_steps", "8",
            "--directml_loss_scale", "1024",
            "--directml_adam_eps", "1e-4",
            "--max_steps", "9",
            "--log_interval", "1",
            "--save_interval", "8",

            "--student_hidden_size", "768",
            "--student_num_layers", "8",
            "--student_use_moe", "0",
            "--from_student_weight", f"{TEST_PREFIX}full_sft",

            "--teacher_hidden_size", "768",
            "--teacher_num_layers", "8",
            "--teacher_use_moe", "1",
            "--from_teacher_weight", f"{TEST_PREFIX}full_sft",

            "--save_weight", f"{TEST_PREFIX}full_dist",
        ],
    ),
    (
        "GRPO",
        [
            "train_grpo.py",
            "--device", "directml:1",
            "--dtype", "float16",
            "--hidden_size", "768",
            "--num_hidden_layers", "8",
            "--use_moe", "0",
            "--batch_size", "1",
            "--num_generations", "2",
            "--max_seq_len", "64",
            "--max_gen_len", "32",
            "--accumulation_steps", "1",
            "--directml_loss_scale", "1024",
            "--directml_adam_eps", "1e-4",
            "--max_steps", "2",
            "--log_interval", "1",
            "--save_interval", "8",
            "--from_weight", f"{TEST_PREFIX}full_sft",
            "--save_weight", f"{TEST_PREFIX}grpo",
        ],
    ),
    (
        "Agent",
        [
            "train_agent.py",
            "--device", "directml:1",
            "--dtype", "float16",
            "--hidden_size", "768",
            "--num_hidden_layers", "8",
            "--use_moe", "0",
            "--batch_size", "1",
            "--num_generations", "2",
            "--max_seq_len", "64",
            "--max_gen_len", "32",
            "--max_total_len", "128",
            "--accumulation_steps", "1",
            "--directml_loss_scale", "1024",
            "--directml_adam_eps", "1e-4",
            "--max_steps", "2",
            "--log_interval", "1",
            "--save_interval", "8",
            "--from_weight", f"{TEST_PREFIX}full_sft",
            "--save_weight", f"{TEST_PREFIX}agent",
        ],
    ),
    (
        "PPO",
        [
            "train_ppo.py",
            "--device", "directml:1",
            "--dtype", "float16",
            "--hidden_size", "768",
            "--num_hidden_layers", "8",
            "--use_moe", "0",
            "--batch_size", "1",
            "--mini_batch_size", "1",
            "--max_seq_len", "64",
            "--max_gen_len", "32",
            "--ppo_update_iters", "1",
            "--accumulation_steps", "1",
            "--directml_loss_scale", "1024",
            "--directml_adam_eps", "1e-4",
            "--max_steps", "2",
            "--log_interval", "1",
            "--save_interval", "8",
            "--from_weight", f"{TEST_PREFIX}full_sft",
            "--save_weight", f"{TEST_PREFIX}ppo_actor",
        ],
    ),
]


def cleanup_test_artifacts() -> None:
    removed = []

    for directory in (OUT_DIR, CHECKPOINT_DIR):
        if not directory.exists():
            continue

        for path in directory.glob(f"{TEST_PREFIX}*"):
            if path.is_file():
                path.unlink()
                removed.append(path)

    if removed:
        print()
        print("Cleaned test artifacts:")
        for path in removed:
            print(f"  - {path.relative_to(ROOT_DIR)}")


def run_trainer_test(name: str, args: list[str]) -> None:
    print()
    print("=" * 60)
    print(f" Testing {name}")
    print("=" * 60)

    trainer_path = TRAINER_DIR / args[0]

    if not trainer_path.exists():
        raise FileNotFoundError(f"Trainer not found: {trainer_path}")

    command = [PYTHON, *args]

    result = subprocess.run(
        command,
        cwd=TRAINER_DIR,
        env=os.environ.copy(),
        check=False,
    )

    if result.returncode != 0:
        print()
        print(f"[FAIL] {name}")
        raise RuntimeError(
            f"{name} failed with exit code {result.returncode}"
        )

    print()
    print(f"[PASS] {name}")


def main() -> int:
    print("=" * 60)
    print(" MiniMind DirectML trainer smoke tests")
    print("=" * 60)
    print(f"Python: {PYTHON}")
    print(f"Trainer directory: {TRAINER_DIR}")
    print(f"Tests: {len(TRAINER_TESTS)}")
    print()
    print("Dense path: Pretrain -> Full SFT")
    print("MoE path:   Pretrain -> Full SFT")
    print("Distill:    Dense student <- MoE teacher")

    passed = 0

    # Remove leftovers from a previously interrupted smoke run.
    cleanup_test_artifacts()

    try:
        for name, args in TRAINER_TESTS:
            run_trainer_test(name, args)
            passed += 1

    except Exception as exc:
        print()
        print("=" * 60)
        print(" Trainer smoke tests stopped")
        print("=" * 60)
        print(f"Passed: {passed}/{len(TRAINER_TESTS)}")
        print(f"Error: {exc}")
        return 1

    finally:
        # Only files generated with the dedicated smoke-test prefix
        # are removed. Real model weights and benchmark files remain.
        cleanup_test_artifacts()

    print()
    print("=" * 60)
    print(" All trainer smoke tests passed")
    print("=" * 60)
    print(f"Passed: {passed}/{len(TRAINER_TESTS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

@pytest.mark.slow
def test_all_trainers():
    assert main() == 0