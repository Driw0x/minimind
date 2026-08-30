$ErrorActionPreference = "Stop"

# ============================================================
# MiniMind - Full sequential DirectML training
#
# Run from repository root:
#   .\scripts\train_all.ps1
#
# This script trains both Dense and MoE branches so that the
# default distillation workflow can use:
#
#   Dense Full SFT -> student
#   MoE Full SFT   -> teacher
#
# No --max_steps is used: every stage runs its full training.
# ============================================================

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$TrainerDir = Join-Path $Root "trainer"

if (-not (Test-Path $Python)) {
    throw "Python virtual environment not found: $Python"
}

if (-not (Test-Path $TrainerDir)) {
    throw "Trainer directory not found: $TrainerDir"
}


# ============================================================
# Global configuration
# ============================================================

$Device = "directml:1"
$DType = "float16"

$HiddenSize = 768
$NumHiddenLayers = 8
$MaxSeqLen = 340

$DirectMLLossScale = 1024
$DirectMLAdamEps = "1e-4"

$UseCompile = 0


# ------------------------------------------------------------
# Dense configuration
#
# M4 validated configuration.
# Effective batch size = 8 * 8 = 64.
# ------------------------------------------------------------

$DenseBatchSize = 8
$DenseAccumulationSteps = 8


# ------------------------------------------------------------
# MoE configuration
#
# The MoE model is significantly larger than the Dense model.
# Keep a conservative micro-batch while preserving the same
# effective batch size:
#
#   1 * 64 = 64
#
# This configuration has not yet been benchmarked as extensively
# as the Dense M4 configuration.
# ------------------------------------------------------------

$MoeBatchSize = 1
$MoeAccumulationSteps = 64


# ------------------------------------------------------------
# Distillation configuration
#
# Both Dense student and MoE teacher are resident in memory.
# Use the conservative MoE micro-batch configuration.
# ------------------------------------------------------------

$DistillationBatchSize = 1
$DistillationAccumulationSteps = 64


function Run-Training {
    param (
        [string]$Name,
        [string]$Script,
        [string[]]$PythonArgs
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " $Name"
    Write-Host "============================================================"
    Write-Host ""

    $ScriptPath = Join-Path $TrainerDir $Script

    if (-not (Test-Path $ScriptPath)) {
        throw "Trainer not found: $ScriptPath"
    }

    Push-Location $TrainerDir

    try {
        & $Python $Script @PythonArgs

        if ($LASTEXITCODE -ne 0) {
            throw "$Name failed with exit code $LASTEXITCODE"
        }

        Write-Host ""
        Write-Host "[OK] $Name completed successfully."
    }
    finally {
        Pop-Location
    }
}


# ============================================================
# 1. DENSE PRETRAIN
#
# Produces:
#   ../out/pretrain_768.pth
# ============================================================

Run-Training `
    "1 - Dense Pretrain" `
    "train_pretrain.py" `
    @(
        "--device", $Device,
        "--dtype", $DType,
        "--hidden_size", $HiddenSize,
        "--num_hidden_layers", $NumHiddenLayers,
        "--use_moe", 0,
        "--batch_size", $DenseBatchSize,
        "--max_seq_len", $MaxSeqLen,
        "--accumulation_steps", $DenseAccumulationSteps,
        "--directml_loss_scale", $DirectMLLossScale,
        "--directml_adam_eps", $DirectMLAdamEps,
        "--from_weight", "none",
        "--use_compile", $UseCompile
    )


# ============================================================
# 2. DENSE FULL SFT
#
# Loads:
#   ../out/pretrain_768.pth
#
# Produces:
#   ../out/full_sft_768.pth
# ============================================================

Run-Training `
    "2 - Dense Full SFT" `
    "train_full_sft.py" `
    @(
        "--device", $Device,
        "--dtype", $DType,
        "--hidden_size", $HiddenSize,
        "--num_hidden_layers", $NumHiddenLayers,
        "--use_moe", 0,
        "--batch_size", $DenseBatchSize,
        "--max_seq_len", $MaxSeqLen,
        "--accumulation_steps", $DenseAccumulationSteps,
        "--directml_loss_scale", $DirectMLLossScale,
        "--directml_adam_eps", $DirectMLAdamEps,
        "--from_weight", "pretrain",
        "--use_compile", $UseCompile
    )


# ============================================================
# 3. MoE PRETRAIN
#
# Produces:
#   ../out/pretrain_768_moe.pth
# ============================================================

Run-Training `
    "3 - MoE Pretrain" `
    "train_pretrain.py" `
    @(
        "--device", $Device,
        "--dtype", $DType,
        "--hidden_size", $HiddenSize,
        "--num_hidden_layers", $NumHiddenLayers,
        "--use_moe", 1,
        "--batch_size", $MoeBatchSize,
        "--max_seq_len", $MaxSeqLen,
        "--accumulation_steps", $MoeAccumulationSteps,
        "--directml_loss_scale", $DirectMLLossScale,
        "--directml_adam_eps", $DirectMLAdamEps,
        "--from_weight", "none",
        "--use_compile", $UseCompile
    )


# ============================================================
# 4. MoE FULL SFT
#
# Loads:
#   ../out/pretrain_768_moe.pth
#
# Produces:
#   ../out/full_sft_768_moe.pth
# ============================================================

Run-Training `
    "4 - MoE Full SFT" `
    "train_full_sft.py" `
    @(
        "--device", $Device,
        "--dtype", $DType,
        "--hidden_size", $HiddenSize,
        "--num_hidden_layers", $NumHiddenLayers,
        "--use_moe", 1,
        "--batch_size", $MoeBatchSize,
        "--max_seq_len", $MaxSeqLen,
        "--accumulation_steps", $MoeAccumulationSteps,
        "--directml_loss_scale", $DirectMLLossScale,
        "--directml_adam_eps", $DirectMLAdamEps,
        "--from_weight", "pretrain",
        "--use_compile", $UseCompile
    )


# ============================================================
# 5. LoRA - Dense branch
#
# Loads:
#   ../out/full_sft_768.pth
# ============================================================

Run-Training `
    "5 - LoRA" `
    "train_lora.py" `
    @(
        "--device", $Device,
        "--dtype", $DType,
        "--hidden_size", $HiddenSize,
        "--num_hidden_layers", $NumHiddenLayers,
        "--use_moe", 0,
        "--batch_size", $DenseBatchSize,
        "--max_seq_len", $MaxSeqLen,
        "--accumulation_steps", $DenseAccumulationSteps,
        "--directml_loss_scale", $DirectMLLossScale,
        "--directml_adam_eps", $DirectMLAdamEps,
        "--from_weight", "full_sft",
        "--use_compile", $UseCompile
    )


# ============================================================
# 6. DPO - Dense branch
#
# Kept on the Dense Full SFT model.
#
# Note:
# The actual train_dpo.py CLI must remain compatible with these
# arguments. DirectML FP16-specific optimizer options are not added
# here until the real DPO trainer is validated with the M4 helpers.
# ============================================================

Run-Training `
    "6 - DPO" `
    "train_dpo.py" `
    @(
        "--device", $Device,
        "--dtype", $DType,
        "--hidden_size", $HiddenSize,
        "--num_hidden_layers", $NumHiddenLayers,
        "--use_moe", 0,
        "--batch_size", $DenseBatchSize,
        "--max_seq_len", $MaxSeqLen,
        "--accumulation_steps", $DenseAccumulationSteps,
        "--from_weight", "full_sft",
        "--use_compile", $UseCompile
    )


# ============================================================
# 7. DISTILLATION
#
# Student:
#   Dense Full SFT -> ../out/full_sft_768.pth
#
# Teacher:
#   MoE Full SFT   -> ../out/full_sft_768_moe.pth
#
# This matches the default upstream distillation architecture:
# Dense student distilled from a MoE teacher.
# ============================================================

Run-Training `
    "7 - Distillation (Dense student <- MoE teacher)" `
    "train_distillation.py" `
    @(
        "--device", $Device,
        "--dtype", $DType,
        "--batch_size", $DistillationBatchSize,
        "--max_seq_len", $MaxSeqLen,
        "--accumulation_steps", $DistillationAccumulationSteps,
        "--directml_loss_scale", $DirectMLLossScale,
        "--directml_adam_eps", $DirectMLAdamEps,

        "--student_hidden_size", $HiddenSize,
        "--student_num_layers", $NumHiddenLayers,
        "--student_use_moe", 0,
        "--from_student_weight", "full_sft",

        "--teacher_hidden_size", $HiddenSize,
        "--teacher_num_layers", $NumHiddenLayers,
        "--teacher_use_moe", 1,
        "--from_teacher_weight", "full_sft",

        "--use_compile", $UseCompile
    )


# ============================================================
# 8. GRPO - Dense branch
# ============================================================

Run-Training `
    "8 - GRPO" `
    "train_grpo.py" `
    @(
        "--device", $Device,
        "--dtype", $DType,
        "--hidden_size", $HiddenSize,
        "--num_hidden_layers", $NumHiddenLayers,
        "--use_moe", 0,
        "--batch_size", $DenseBatchSize,
        "--max_seq_len", $MaxSeqLen,
        "--accumulation_steps", $DenseAccumulationSteps,
        "--directml_loss_scale", $DirectMLLossScale,
        "--directml_adam_eps", $DirectMLAdamEps,
        "--from_weight", "full_sft",
        "--use_compile", $UseCompile
    )


# ============================================================
# 9. PPO - Dense branch
# ============================================================

Run-Training `
    "9 - PPO" `
    "train_ppo.py" `
    @(
        "--device", $Device,
        "--dtype", $DType,
        "--hidden_size", $HiddenSize,
        "--num_hidden_layers", $NumHiddenLayers,
        "--use_moe", 0,
        "--batch_size", $DenseBatchSize,
        "--max_seq_len", $MaxSeqLen,
        "--accumulation_steps", $DenseAccumulationSteps,
        "--directml_loss_scale", $DirectMLLossScale,
        "--directml_adam_eps", $DirectMLAdamEps,
        "--from_weight", "full_sft",
        "--use_compile", $UseCompile
    )


# ============================================================
# 10. AGENT - Dense branch
# ============================================================

Run-Training `
    "10 - Agent" `
    "train_agent.py" `
    @(
        "--device", $Device,
        "--dtype", $DType,
        "--hidden_size", $HiddenSize,
        "--num_hidden_layers", $NumHiddenLayers,
        "--use_moe", 0,
        "--batch_size", $DenseBatchSize,
        "--max_seq_len", $MaxSeqLen,
        "--accumulation_steps", $DenseAccumulationSteps,
        "--directml_loss_scale", $DirectMLLossScale,
        "--directml_adam_eps", $DirectMLAdamEps,
        "--from_weight", "full_sft",
        "--use_compile", $UseCompile
    )


Write-Host ""
Write-Host "============================================================"
Write-Host " ALL TRAININGS COMPLETED SUCCESSFULLY"
Write-Host "============================================================"
Write-Host ""
Write-Host "Dense checkpoints:"
Write-Host "  pretrain_768.pth"
Write-Host "  full_sft_768.pth"
Write-Host ""
Write-Host "MoE checkpoints:"
Write-Host "  pretrain_768_moe.pth"
Write-Host "  full_sft_768_moe.pth"
Write-Host ""