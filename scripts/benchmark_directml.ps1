$ErrorActionPreference = "Stop"

$OriginalLocation = Get-Location
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$TrainerDir = Join-Path $ProjectRoot "trainer"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

$Device = "directml:1"
$TimeoutSeconds = 300
$MaxSteps = 9

$Batches = @(1, 2, 4, 8, 16, 32)

$SequencesByBatch = @{
    1  = @(64, 128, 256, 340)
    2  = @(128, 256, 340)
    4  = @(128, 256, 340)
    8  = @(128, 256, 340)
    16 = @(128, 256, 340)
    32 = @(128, 256, 340)
}

$Results = @()
$StopBenchmark = $false

try {
    Set-Location $TrainerDir

    foreach ($Batch in $Batches) {

        if ($StopBenchmark) {
            break
        }

        $Sequences = $SequencesByBatch[$Batch]

        foreach ($Seq in $Sequences) {

            Write-Host ""
            Write-Host "========================================"
            Write-Host "Testing DirectML FP16"
            Write-Host "device             = $Device"
            Write-Host "batch_size         = $Batch"
            Write-Host "max_seq_len        = $Seq"
            Write-Host "dtype              = float16"
            Write-Host "accumulation_steps = 8"
            Write-Host "loss_scale         = 1024"
            Write-Host "adam_eps           = 1e-4"
            Write-Host "max_steps          = $MaxSteps"
            Write-Host "timeout            = $TimeoutSeconds sec"
            Write-Host "========================================"

            $Arguments = @(
                "train_pretrain.py",
                "--device", "$Device",
                "--dtype", "float16",
                "--hidden_size", "768",
                "--num_hidden_layers", "8",
                "--use_moe", "0",
                "--batch_size", "$Batch",
                "--max_seq_len", "$Seq",
                "--num_workers", "0",
                "--accumulation_steps", "8",
                "--directml_loss_scale", "1024",
                "--directml_adam_eps", "1e-4",
                "--save_dir", "../out",
                "--save_weight", "benchmark",
                "--from_weight", "none",
                "--use_compile", "0",
                "--epochs", "1",
                "--max_steps", "$MaxSteps",
                "--log_interval", "1"
            )

            $Job = Start-Job `
                -ArgumentList $Python, $Arguments, $TrainerDir `
                -ScriptBlock {

                    param(
                        $PythonPath,
                        $PythonArgs,
                        $WorkingDirectory
                    )

                    Set-Location $WorkingDirectory

                    & $PythonPath @PythonArgs

                    $Code = $LASTEXITCODE

                    [PSCustomObject]@{
                        Marker   = "BENCHMARK_EXIT"
                        ExitCode = $Code
                    }
                }

            $Completed = Wait-Job $Job -Timeout $TimeoutSeconds

            if ($null -eq $Completed) {

                Write-Host ""
                Write-Host "Result: TIMEOUT"

                Stop-Job $Job
                Remove-Job $Job -Force

                $Status = "TIMEOUT"
                $ExitCode = "-"
            }
            else {

                # Receive stderr without letting a Python error
                # terminate the PowerShell benchmark.
                $PreviousErrorActionPreference = $ErrorActionPreference
                $ErrorActionPreference = "Continue"

                $Output = Receive-Job $Job -ErrorAction Continue 2>&1

                $ErrorActionPreference = $PreviousErrorActionPreference

                foreach ($Line in $Output) {

                    if (
                        $Line -isnot [PSCustomObject] -or
                        $Line.Marker -ne "BENCHMARK_EXIT"
                    ) {
                        Write-Host $Line
                    }
                }

                $ExitResult = $Output |
                    Where-Object {
                        $_.Marker -eq "BENCHMARK_EXIT"
                    } |
                    Select-Object -Last 1

                if ($null -eq $ExitResult) {

                    $ExitCode = "-"
                    $Status = "FAIL"

                    Write-Host ""
                    Write-Host "Result: FAIL"
                    Write-Host "No exit code returned"
                }
                else {

                    $ExitCode = $ExitResult.ExitCode

                    if ($ExitCode -eq 0) {

                        Write-Host ""
                        Write-Host "Result: PASS"

                        $Status = "PASS"
                    }
                    else {

                        Write-Host ""
                        Write-Host "Result: FAIL"
                        Write-Host "Exit code: $ExitCode"

                        $Status = "FAIL"
                    }
                }

                Remove-Job $Job -Force
            }

            $Results += [PSCustomObject]@{
                BatchSize = $Batch
                SeqLen    = $Seq
                Status    = $Status
                ExitCode  = $ExitCode
            }

            # -------------------------------------------------
            # Failure on the smallest sequence for this batch:
            # stop testing larger batches.
            # -------------------------------------------------

            if (
                $Status -ne "PASS" -and
                $Seq -eq $Sequences[0]
            ) {

                Write-Host ""
                Write-Host "Smallest sequence failed for batch $Batch."
                Write-Host "Stopping benchmark before larger batches."

                $StopBenchmark = $true
                break
            }

            # -------------------------------------------------
            # Failure on a larger sequence:
            # skip remaining sequence lengths for this batch,
            # then continue with the next batch.
            # -------------------------------------------------

            if ($Status -ne "PASS") {

                Write-Host ""
                Write-Host "Configuration $Batch / $Seq did not pass."
                Write-Host "Skipping larger sequence lengths for batch $Batch."

                break
            }

            Start-Sleep -Seconds 3
        }
    }
}
finally {

    Set-Location $OriginalLocation
}

Write-Host ""
Write-Host "========================================"
Write-Host "DirectML FP16 compatibility results"
Write-Host "Device:             $Device"
Write-Host "Model dtype:        float16"
Write-Host "Loss scale:         1024"
Write-Host "AdamW epsilon:      1e-4"
Write-Host "Accumulation steps: 8"
Write-Host "Validation steps:   $MaxSteps"
Write-Host "========================================"
Write-Host ""

$Results | Format-Table -AutoSize