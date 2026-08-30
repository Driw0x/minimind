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

$DirectMLLossScale = 1024
$DirectMLAdamEps = "1e-4"
$SaveInterval = 100

$UseCompile = 0


# ------------------------------------------------------------
# Trainer configurations
#
# Matches docs/training_commands.md:
#
# Trainer        Batch   MaxSeqLen   Accumulation
# Pretrain       32      340         8
# Full SFT       16      768         1
# LoRA           32      340         1
# DPO             4     1024         1
# Distillation   32      340         1
# GRPO            2      768         1
# PPO             2      768         1
# Agent           2     1024         1
#
# IMPORTANT:
# These are the trainer-specific target values documented for the
# full workflow. The sustained DirectML pretraining baseline remains
# 8 / 340 / accumulation 8 and should not be confused with this table.
# ------------------------------------------------------------

# Pretrain
$PretrainBatchSize = 32
$PretrainMaxSeqLen = 340
$PretrainAccumulationSteps = 8

# Full SFT
$FullSftBatchSize = 16
$FullSftMaxSeqLen = 768
$FullSftAccumulationSteps = 1

# LoRA
$LoraBatchSize = 32
$LoraMaxSeqLen = 340
$LoraAccumulationSteps = 1

# DPO (upstream trainer defaults)
$DpoBatchSize = 4
$DpoMaxSeqLen = 1024
$DpoAccumulationSteps = 1

# Distillation
$DistillationBatchSize = 32
$DistillationMaxSeqLen = 340
$DistillationAccumulationSteps = 1

# GRPO
$GrpoBatchSize = 2
$GrpoMaxSeqLen = 768
$GrpoAccumulationSteps = 1

# PPO
$PpoBatchSize = 2
$PpoMaxSeqLen = 768
$PpoAccumulationSteps = 1

# Agent
$AgentBatchSize = 2
$AgentMaxSeqLen = 1024
$AgentAccumulationSteps = 1


# ------------------------------------------------------------
# MoE DirectML override
#
# Matches docs/training_commands.md:
#
# Trainer          Batch   MaxSeqLen   Accumulation
# MoE Pretrain       1      340         64
# MoE Full SFT       1      768         64
#
# These values are a DirectML safety override, not upstream defaults.
# ------------------------------------------------------------

$MoePretrainBatchSize = 1
$MoePretrainMaxSeqLen = 340
$MoePretrainAccumulationSteps = 64

$MoeFullSftBatchSize = 1
$MoeFullSftMaxSeqLen = 768
$MoeFullSftAccumulationSteps = 64

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
        "--batch_size", $PretrainBatchSize,
        "--max_seq_len", $PretrainMaxSeqLen,
        "--accumulation_steps", $PretrainAccumulationSteps,
        "--save_interval", $SaveInterval,
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
        "--batch_size", $FullSftBatchSize,
        "--max_seq_len", $FullSftMaxSeqLen,
        "--accumulation_steps", $FullSftAccumulationSteps,
        "--save_interval", $SaveInterval,
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
        "--batch_size", $MoePretrainBatchSize,
        "--max_seq_len", $MoePretrainMaxSeqLen,
        "--accumulation_steps", $MoePretrainAccumulationSteps,
        "--save_interval", $SaveInterval,
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
        "--batch_size", $MoeFullSftBatchSize,
        "--max_seq_len", $MoeFullSftMaxSeqLen,
        "--accumulation_steps", $MoeFullSftAccumulationSteps,
        "--save_interval", $SaveInterval,
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
        "--batch_size", $LoraBatchSize,
        "--max_seq_len", $LoraMaxSeqLen,
        "--accumulation_steps", $LoraAccumulationSteps,
        "--save_interval", $SaveInterval,
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
        "--batch_size", $DpoBatchSize,
        "--max_seq_len", $DpoMaxSeqLen,
        "--accumulation_steps", $DpoAccumulationSteps,
        "--save_interval", $SaveInterval,
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
        "--max_seq_len", $DistillationMaxSeqLen,
        "--accumulation_steps", $DistillationAccumulationSteps,
        "--save_interval", $SaveInterval,
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
        "--batch_size", $GrpoBatchSize,
        "--max_seq_len", $GrpoMaxSeqLen,
        "--accumulation_steps", $GrpoAccumulationSteps,
        "--save_interval", $SaveInterval,
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
        "--batch_size", $PpoBatchSize,
        "--max_seq_len", $PpoMaxSeqLen,
        "--accumulation_steps", $PpoAccumulationSteps,
        "--save_interval", $SaveInterval,
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
        "--batch_size", $AgentBatchSize,
        "--max_seq_len", $AgentMaxSeqLen,
        "--accumulation_steps", $AgentAccumulationSteps,
        "--save_interval", $SaveInterval,
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
