$ErrorActionPreference = "Stop"

$steps = 1000

$commonArgs = @(

    "--device", "directml:1",

    "--dtype", "float16",

    "--epochs", "2",

    "--batch_size", "32",

    "--learning_rate", "5e-4",

    "--num_workers", "8",

    "--accumulation_steps", "8",

    "--grad_clip", "1.0",

    "--log_interval", "100",

    "--save_interval", "100000",

    "--hidden_size", "768",

    "--num_hidden_layers", "8",

    "--max_seq_len", "340",

    "--use_moe", "0",

    "--directml_loss_scale", "1024",

    "--directml_adam_eps", "1e-4",

    "--from_weight", "none",

    "--from_resume", "0",

    "--use_compile", "0",

    "--max_steps", "$steps"

)

$rootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$trainerDir = Join-Path $rootDir "trainer"

Push-Location $trainerDir

try {

    Write-Host ""

    Write-Host "============================================================"

    Write-Host " Stable trainer benchmark"

    Write-Host "============================================================"

    $stableTimer = [System.Diagnostics.Stopwatch]::StartNew()

    python train_pretrain.py @commonArgs --save_weight bench_stable

    if ($LASTEXITCODE -ne 0) {

        throw "Stable trainer failed with exit code $LASTEXITCODE"

    }

    $stableTimer.Stop()

    $stableSeconds = $stableTimer.Elapsed.TotalSeconds


    Write-Host ""

    Write-Host "============================================================"

    Write-Host " Log-Sync trainer benchmark"

    Write-Host "============================================================"

    $logSyncTimer = [System.Diagnostics.Stopwatch]::StartNew()

    python train_pretrain_log_sync.py @commonArgs --save_weight bench_log_sync

    if ($LASTEXITCODE -ne 0) {

        throw "Log-Sync trainer failed with exit code $LASTEXITCODE"

    }

    $logSyncTimer.Stop()

    $logSyncSeconds = $logSyncTimer.Elapsed.TotalSeconds

}
finally {

    Pop-Location

}


$stableStep = $stableSeconds / $steps

$logSyncStep = $logSyncSeconds / $steps

$stableStepsPerSec = $steps / $stableSeconds

$logSyncStepsPerSec = $steps / $logSyncSeconds

$gainPercent = (1 - ($logSyncSeconds / $stableSeconds)) * 100

$speedup = $stableSeconds / $logSyncSeconds


Write-Host ""

Write-Host "============================================================"

Write-Host " Benchmark results"

Write-Host "============================================================"

Write-Host ("Steps:                    {0}" -f $steps)

Write-Host ""

Write-Host ("Stable total:             {0:N2} s" -f $stableSeconds)

Write-Host ("Log-Sync total:           {0:N2} s" -f $logSyncSeconds)

Write-Host ""

Write-Host ("Stable time / step:       {0:N4} s" -f $stableStep)

Write-Host ("Log-Sync time / step:     {0:N4} s" -f $logSyncStep)

Write-Host ""

Write-Host ("Stable steps / second:    {0:N4}" -f $stableStepsPerSec)

Write-Host ("Log-Sync steps / second:  {0:N4}" -f $logSyncStepsPerSec)

Write-Host ""

Write-Host ("Speedup:                  {0:N3}x" -f $speedup)

Write-Host ("Time reduction:           {0:N2} %" -f $gainPercent)

Write-Host "============================================================"