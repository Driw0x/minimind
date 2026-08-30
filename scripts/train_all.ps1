$ErrorActionPreference = "Stop"

# ============================================================
# MiniMind - Sequential DirectML training
#
# Run from repository root:
#   .\scripts\train_all.ps1
#
# out/         -> model weights
# checkpoints/ -> training resume checkpoints
# ============================================================

# Repository root = parent of scripts/
$Root = Split-Path -Parent $PSScriptRoot

$Python = Join-Path $Root ".venv\Scripts\python.exe"
$TrainerDir = Join-Path $Root "trainer"

# ------------------------------------------------------------
# Global DirectML configuration
# ------------------------------------------------------------

$Device = "directml:1"
$HiddenSize = 768
$NumHiddenLayers = 8
$UseMoe = 0

$BatchSize = 8
$MaxSeqLen = 340

$UseCompile = 0


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

    Push-Location $TrainerDir

    try {
        & $Python $Script @Arguments

        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "[FAILED] $Name"
            Write-Host "Exit code: $LASTEXITCODE"
            exit $LASTEXITCODE
        }
    }
    finally {
        Pop-Location
    }

    Write-Host ""
    Write-Host "[OK] $Name completed successfully."
}


# ============================================================
# 1. PRETRAIN
# ============================================================

Run-Training `
    "1 - Pretrain" `
    "train_pretrain.py" `
    @(
        "--device", $Device,
        "--hidden_size", $HiddenSize,
        "--num_hidden_layers", $NumHiddenLayers,
        "--use_moe", $UseMoe,
        "--batch_size", $BatchSize,
        "--max_seq_len", $MaxSeqLen,
        "--from_weight", "none",
        "--use_compile", $UseCompile
    )


# ============================================================
# 2. FULL SFT
# ============================================================

Run-Training `
    "2 - Full SFT" `
    "train_full_sft.py" `
    @(
        "--device", $Device,
        "--hidden_size", $HiddenSize,
        "--num_hidden_layers", $NumHiddenLayers,
        "--use_moe", $UseMoe,
        "--batch_size", $BatchSize,
        "--max_seq_len", $MaxSeqLen,
        "--from_weight", "pretrain",
        "--use_compile", $UseCompile
    )


# ============================================================
# 3. LoRA
# ============================================================

Run-Training `
    "3 - LoRA" `
    "train_lora.py" `
    @(
        "--device", $Device,
        "--hidden_size", $HiddenSize,
        "--num_hidden_layers", $NumHiddenLayers,
        "--use_moe", $UseMoe,
        "--batch_size", $BatchSize,
        "--max_seq_len", $MaxSeqLen,
        "--from_weight", "full_sft",
        "--use_compile", $UseCompile
    )


# ============================================================
# 4. DPO
# ============================================================

Run-Training `
    "4 - DPO" `
    "train_dpo.py" `
    @(
        "--device", $Device,
        "--hidden_size", $HiddenSize,
        "--num_hidden_layers", $NumHiddenLayers,
        "--use_moe", $UseMoe,
        "--batch_size", $BatchSize,
        "--max_seq_len", $MaxSeqLen,
        "--from_weight", "full_sft",
        "--use_compile", $UseCompile
    )


# ============================================================
# 5. GRPO
# ============================================================

Run-Training `
    "5 - GRPO" `
    "train_grpo.py" `
    @(
        "--device", $Device,
        "--hidden_size", $HiddenSize,
        "--num_hidden_layers", $NumHiddenLayers,
        "--use_moe", $UseMoe,
        "--batch_size", $BatchSize,
        "--max_seq_len", $MaxSeqLen,
        "--from_weight", "full_sft",
        "--use_compile", $UseCompile
    )


# ============================================================
# 6. PPO
# ============================================================

Run-Training `
    "6 - PPO" `
    "train_ppo.py" `
    @(
        "--device", $Device,
        "--hidden_size", $HiddenSize,
        "--num_hidden_layers", $NumHiddenLayers,
        "--use_moe", $UseMoe,
        "--batch_size", $BatchSize,
        "--max_seq_len", $MaxSeqLen,
        "--from_weight", "full_sft",
        "--use_compile", $UseCompile
    )


# ============================================================
# 7. AGENT
# ============================================================

Run-Training `
    "7 - Agent" `
    "train_agent.py" `
    @(
        "--device", $Device,
        "--hidden_size", $HiddenSize,
        "--num_hidden_layers", $NumHiddenLayers,
        "--use_moe", $UseMoe,
        "--batch_size", $BatchSize,
        "--max_seq_len", $MaxSeqLen,
        "--use_compile", $UseCompile
    )


Write-Host ""
Write-Host "============================================================"
Write-Host " ALL TRAININGS COMPLETED SUCCESSFULLY"
Write-Host "============================================================"