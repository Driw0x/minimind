$ErrorActionPreference = "Stop"

# ============================================================
# MiniMind DirectML - Sequential training pipeline
# Run from repository root:
#   .\train_all.ps1
# ============================================================

$Python = ".\.venv\Scripts\python.exe"

$Device = "directml"
$HiddenSize = 768
$NumHiddenLayers = 8
$UseMoe = 0
$SaveDir = "../out"

function Run-Training {
    param (
        [string]$Name,
        [string]$Script,
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " $Name"
    Write-Host "============================================================"
    Write-Host ""

    & $Python $Script @Arguments

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "[FAILED] $Name"
        Write-Host "Exit code: $LASTEXITCODE"
        exit $LASTEXITCODE
    }

    Write-Host ""
    Write-Host "[OK] $Name completed"
}


# ------------------------------------------------------------
# 1. Pretraining
# ------------------------------------------------------------

Run-Training `
    "1/8 - Pretraining" `
    "trainer/train_pretrain.py" `
    @(
        "--device", $Device,
        "--hidden_size", $HiddenSize,
        "--num_hidden_layers", $NumHiddenLayers,
        "--use_moe", $UseMoe,
        "--save_dir", $SaveDir,
        "--save_weight", "pretrain",
        "--from_weight", "none",
        "--use_compile", "0"
    )


# ------------------------------------------------------------
# 2. Full SFT
# pretrain -> full_sft
# ------------------------------------------------------------

Run-Training `
    "2/8 - Full SFT" `
    "trainer/train_full_sft.py" `
    @(
        "--device", $Device,
        "--hidden_size", $HiddenSize,
        "--num_hidden_layers", $NumHiddenLayers,
        "--use_moe", $UseMoe,
        "--save_dir", $SaveDir,
        "--save_weight", "full_sft",
        "--from_weight", "pretrain",
        "--use_compile", "0"
    )


# ------------------------------------------------------------
# 3. LoRA
# full_sft -> LoRA adapter
# ------------------------------------------------------------

Run-Training `
    "3/8 - LoRA" `
    "trainer/train_lora.py" `
    @(
        "--device", $Device,
        "--hidden_size", $HiddenSize,
        "--num_hidden_layers", $NumHiddenLayers,
        "--use_moe", $UseMoe,
        "--save_dir", $SaveDir,
        "--from_weight", "full_sft",
        "--use_compile", "0"
    )


# ------------------------------------------------------------
# 4. DPO
# full_sft -> dpo
# ------------------------------------------------------------

Run-Training `
    "4/8 - DPO" `
    "trainer/train_dpo.py" `
    @(
        "--device", $Device,
        "--hidden_size", $HiddenSize,
        "--num_hidden_layers", $NumHiddenLayers,
        "--use_moe", $UseMoe,
        "--save_dir", $SaveDir,
        "--save_weight", "dpo",
        "--from_weight", "full_sft",
        "--use_compile", "0"
    )


# ------------------------------------------------------------
# 5. GRPO
# full_sft -> grpo
#
# Reward model is automatically kept on CPU when DirectML
# is selected.
# ------------------------------------------------------------

Run-Training `
    "5/8 - GRPO" `
    "trainer/train_grpo.py" `
    @(
        "--device", $Device,
        "--hidden_size", $HiddenSize,
        "--num_hidden_layers", $NumHiddenLayers,
        "--use_moe", $UseMoe,
        "--save_dir", $SaveDir,
        "--save_weight", "grpo",
        "--from_weight", "full_sft",
        "--use_compile", "0"
    )


# ------------------------------------------------------------
# 6. PPO
# full_sft -> ppo_actor
# ------------------------------------------------------------

Run-Training `
    "6/8 - PPO" `
    "trainer/train_ppo.py" `
    @(
        "--device", $Device,
        "--hidden_size", $HiddenSize,
        "--num_hidden_layers", $NumHiddenLayers,
        "--use_moe", $UseMoe,
        "--save_dir", $SaveDir,
        "--save_weight", "ppo_actor",
        "--from_weight", "full_sft",
        "--use_compile", "0"
    )


# ------------------------------------------------------------
# 7. Agent RL
# ------------------------------------------------------------

Run-Training `
    "7/8 - Agent RL" `
    "trainer/train_agent.py" `
    @(
        "--device", $Device,
        "--hidden_size", $HiddenSize,
        "--num_hidden_layers", $NumHiddenLayers,
        "--use_moe", $UseMoe,
        "--save_dir", $SaveDir
    )


# ------------------------------------------------------------
# 8. Distillation
# Uncomment / adapt script name if you want it in the
# automatic full run.
# ------------------------------------------------------------

# Run-Training `
#     "8/8 - Distillation" `
#     "trainer/train_distill_reason.py" `
#     @(
#         "--device", $Device,
#         "--hidden_size", $HiddenSize,
#         "--num_hidden_layers", $NumHiddenLayers,
#         "--use_moe", $UseMoe,
#         "--save_dir", $SaveDir,
#         "--use_compile", "0"
#     )


Write-Host ""
Write-Host "============================================================"
Write-Host " ALL TRAINING STAGES COMPLETED SUCCESSFULLY"
Write-Host "============================================================"