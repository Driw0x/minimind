import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ============================================================================
# Paths
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
TRAINER_DIR = ROOT_DIR / "trainer"

LOG_DIR = ROOT_DIR / "logs" / "rocm"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
OUT_DIR = ROOT_DIR / "out"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# LOG_FILE = LOG_DIR / "pretrain_768.log"

# # Use a distinct save_weight so ROCm never overwrites DirectML artifacts.
# SAVE_WEIGHT = "pretrain_rocm"

# RESUME_CHECKPOINT = CHECKPOINT_DIR / f"{SAVE_WEIGHT}_768_resume.pth"
# MODEL_CHECKPOINT = CHECKPOINT_DIR / f"{SAVE_WEIGHT}_768.pth"
# OUTPUT_MODEL = OUT_DIR / f"{SAVE_WEIGHT}_768.pth"

LOG_FILE = LOG_DIR / "pretrain_768_bf16.log"

# Use a distinct save_weight so ROCm never overwrites DirectML artifacts.
SAVE_WEIGHT = "pretrain_rocm_bf16"

RESUME_CHECKPOINT = CHECKPOINT_DIR / f"{SAVE_WEIGHT}_768_resume.pth"
MODEL_CHECKPOINT = CHECKPOINT_DIR / f"{SAVE_WEIGHT}_768.pth"
OUTPUT_MODEL = OUT_DIR / f"{SAVE_WEIGHT}_768.pth"


# ============================================================================
# Arguments
# ============================================================================

parser = argparse.ArgumentParser(
    description="Run MiniMind ROCm pretraining with automatic resume and logs."
)

parser.add_argument(
    "--fresh",
    action="store_true",
    help="Delete previous ROCm pretrain checkpoint/logs and restart from scratch.",
)

args = parser.parse_args()


# ============================================================================
# Fresh start
# ============================================================================

if args.fresh:
    files_to_delete = [
        RESUME_CHECKPOINT,
        MODEL_CHECKPOINT,
        OUTPUT_MODEL,
        LOG_FILE,
    ]

    for path in files_to_delete:
        if path.exists():
            path.unlink()
            print(f"Supprimé : {path}")


# ============================================================================
# Detect resume checkpoint
# ============================================================================

resume = RESUME_CHECKPOINT.exists()

if resume:
    mode = "RESUME"
    from_resume = "1"
else:
    mode = "FRESH"
    from_resume = "0"


# ============================================================================
# Training command
# ============================================================================

command = [
    sys.executable,
    "-u",
    "train_pretrain.py",

    # ROCm uses the PyTorch CUDA API.
    "--device", "cuda:0",
    "--dtype", "float16",

    # One complete epoch for DirectML / ROCm comparison.
    "--epochs", "2",

    "--save_weight", SAVE_WEIGHT,

    "--from_weight", "none",
    "--from_resume", from_resume,

    "--use_compile", "0",
]

# ============================================================================
# Header
# ============================================================================

separator = "=" * 80

header = (
    f"\n{separator}\n"
    f"MiniMind ROCm Pretrain\n"
    f"Mode       : {mode}\n"
    f"Date       : {datetime.now().isoformat(timespec='seconds')}\n"
    f"Checkpoint : {RESUME_CHECKPOINT}\n"
    f"Log        : {LOG_FILE}\n"
    f"{separator}\n"
)

print(header)

if resume:
    print("Reprise depuis le dernier checkpoint sauvegardé.\n")
else:
    print("Aucun checkpoint trouvé : entraînement depuis zéro.\n")


# ============================================================================
# Run training + save logs
# ============================================================================

with LOG_FILE.open("a", encoding="utf-8") as log_file:
    log_file.write(header)
    log_file.write(f"Commande : {' '.join(command)}\n\n")
    log_file.flush()

    process = subprocess.Popen(
        command,
        cwd=TRAINER_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    return_code = 1

    try:
        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="", flush=True)

            log_file.write(line)
            log_file.flush()

        return_code = process.wait()

    except KeyboardInterrupt:
        message = (
            "\nInterruption demandée.\n"
            "Le prochain lancement reprendra depuis le dernier checkpoint sauvegardé.\n"
        )

        print(message)

        log_file.write(message)
        log_file.flush()

        process.terminate()

        try:
            return_code = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait()

    finally:
        footer = (
            f"\n{separator}\n"
            f"Fin        : {datetime.now().isoformat(timespec='seconds')}\n"
            f"Exit code  : {return_code}\n"
            f"{separator}\n"
        )

        print(footer)

        log_file.write(footer)
        log_file.flush()


sys.exit(return_code)