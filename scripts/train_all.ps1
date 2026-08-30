param(
    [int]$BatchSize = 2,
    [int]$MaxSeqLen = 256
)

$ErrorActionPreference = "Stop"

$OriginalLocation = Get-Location
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$TrainerDir = Join-Path $ProjectRoot "trainer"

$Python = "..\.venv\Scripts\python.exe"
$SaveDir = "../out"

function Run-Training {
    param(
        [string]$Name,
        [string]$Script,
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "========================================"
    Write-Host "Training: $Name"
    Write-Host "Batch size: $BatchSize"
    Write-Host "Max seq len: $MaxSeqLen"
    Write-Host "========================================"
    Write-Host ""

    & $Python $Script @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }

    Write-Host ""
    Write-Host "$Name completed successfully."
}

try {
    Set-Location $TrainerDir

    Write-Host ""
    Write-Host "========================================"
    Write-Host "MiniMind DirectML training pipeline"
    Write-Host "========================================"
    Write-Host "Device      : DirectML"
    Write-Host "Batch size  : $BatchSize"
    Write-Host "Max seq len : $MaxSeqLen"
    Write-Host "Save dir    : $SaveDir"
    Write-Host "========================================"

    # ---------------------------------------------------------
    # 1. Pretrain
    # ---------------------------------------------------------

    Run-Training `
        -Name "Pretrain" `
        -Script "train_pretrain.py" `
        -Arguments @(
            "--device", "directml",
            "--hidden_size", "768",
            "--num_hidden_layers", "8",
            "--use_moe", "0",
            "--batch_size", "$BatchSize",
            "--max_seq_len", "$MaxSeqLen",
            "--num_workers", "0",
            "--save_dir", "$SaveDir",
            "--save_weight", "pretrain",
            "--from_weight", "none",
            "--use_compile", "0"
        )

    # ---------------------------------------------------------
    # 2. Full SFT
    # ---------------------------------------------------------

    Run-Training `
        -Name "Full SFT" `
        -Script "train_full_sft.py" `
        -Arguments @(
            "--device", "directml",
            "--hidden_size", "768",
            "--num_hidden_layers", "8",
            "--use_moe", "0",
            "--batch_size", "$BatchSize",
            "--max_seq_len", "$MaxSeqLen",
            "--num_workers", "0",
            "--save_dir", "$SaveDir",
            "--save_weight", "full_sft",
            "--from_weight", "pretrain",
            "--use_compile", "0"
        )

    # ---------------------------------------------------------
    # 3. LoRA
    # ---------------------------------------------------------

    Run-Training `
        -Name "LoRA" `
        -Script "train_lora.py" `
        -Arguments @(
            "--device", "directml",
            "--hidden_size", "768",
            "--num_hidden_layers", "8",
            "--use_moe", "0",
            "--batch_size", "$BatchSize",
            "--max_seq_len", "$MaxSeqLen",
            "--num_workers", "0",
            "--save_dir", "$SaveDir",
            "--from_weight", "full_sft",
            "--use_compile", "0"
        )

    # ---------------------------------------------------------
    # 4. DPO
    # ---------------------------------------------------------

    Run-Training `
        -Name "DPO" `
        -Script "train_dpo.py" `
        -Arguments @(
            "--device", "directml",
            "--hidden_size", "768",
            "--num_hidden_layers", "8",
            "--use_moe", "0",
            "--batch_size", "$BatchSize",
            "--max_seq_len", "$MaxSeqLen",
            "--num_workers", "0",
            "--save_dir", "$SaveDir",
            "--save_weight", "dpo",
            "--from_weight", "full_sft",
            "--use_compile", "0"
        )

    # ---------------------------------------------------------
    # 5. GRPO
    # ---------------------------------------------------------

    Run-Training `
        -Name "GRPO" `
        -Script "train_grpo.py" `
        -Arguments @(
            "--device", "directml",
            "--hidden_size", "768",
            "--num_hidden_layers", "8",
            "--use_moe", "0",
            "--batch_size", "$BatchSize",
            "--max_seq_len", "$MaxSeqLen",
            "--num_workers", "0",
            "--save_dir", "$SaveDir",
            "--save_weight", "grpo",
            "--from_weight", "full_sft",
            "--use_compile", "0"
        )

    # ---------------------------------------------------------
    # 6. PPO
    # ---------------------------------------------------------

    Run-Training `
        -Name "PPO" `
        -Script "train_ppo.py" `
        -Arguments @(
            "--device", "directml",
            "--hidden_size", "768",
            "--num_hidden_layers", "8",
            "--use_moe", "0",
            "--batch_size", "$BatchSize",
            "--max_seq_len", "$MaxSeqLen",
            "--num_workers", "0",
            "--save_dir", "$SaveDir",
            "--save_weight", "ppo",
            "--from_weight", "full_sft",
            "--use_compile", "0"
        )

    # ---------------------------------------------------------
    # 7. Agent RL
    # ---------------------------------------------------------

    Run-Training `
        -Name "Agent RL" `
        -Script "train_agent.py" `
        -Arguments @(
            "--device", "directml",
            "--hidden_size", "768",
            "--num_hidden_layers", "8",
            "--use_moe", "0",
            "--batch_size", "$BatchSize",
            "--max_seq_len", "$MaxSeqLen",
            "--num_workers", "0",
            "--save_dir", "$SaveDir",
            "--save_weight", "agent",
            "--from_weight", "full_sft",
            "--use_compile", "0"
        )

    # ---------------------------------------------------------
    # Distillation
    # ---------------------------------------------------------
    #
    # Intentionally disabled for now.
    #
    # Run-Training `
    #     -Name "Distillation" `
    #     -Script "train_distillation.py" `
    #     -Arguments @(
    #         ...
    #     )

    Write-Host ""
    Write-Host "========================================"
    Write-Host "All training stages completed."
    Write-Host "========================================"
}
catch {
    Write-Host ""
    Write-Host "========================================"
    Write-Host "Training pipeline stopped."
    Write-Host "========================================"
    Write-Host $_.Exception.Message

    exit 1
}
finally {
    Set-Location $OriginalLocation
}