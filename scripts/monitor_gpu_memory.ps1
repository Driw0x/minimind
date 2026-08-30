param(
    [int]$IntervalMs = 1000,
    [string]$OutputFile = "out/gpu_memory_benchmark.csv"
)

$ErrorActionPreference = "Stop"

# Resolve output path from the project root.
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$OutputPath = Join-Path $ProjectRoot $OutputFile
$OutputDirectory = Split-Path -Parent $OutputPath

if (-not (Test-Path $OutputDirectory)) {
    New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
}

Write-Host ""
Write-Host "============================================================"
Write-Host "DirectML GPU memory monitor"
Write-Host "============================================================"
Write-Host "Sampling interval: $IntervalMs ms"
Write-Host "Output:            $OutputPath"
Write-Host ""
Write-Host "Waiting for a Python process..."
Write-Host "Press Ctrl+C to stop monitoring."
Write-Host "============================================================"
Write-Host ""

$Samples = @()

try {
    while ($true) {

        $PythonProcesses = Get-Process python -ErrorAction SilentlyContinue

        if (-not $PythonProcesses) {
            Start-Sleep -Milliseconds $IntervalMs
            continue
        }

        $PythonPids = $PythonProcesses.Id

        $Counters = Get-Counter '\GPU Process Memory(*)\Dedicated Usage' `
            -ErrorAction SilentlyContinue

        if ($null -eq $Counters) {
            Start-Sleep -Milliseconds $IntervalMs
            continue
        }

        $TotalBytes = 0

        foreach ($Sample in $Counters.CounterSamples) {

            # Windows GPU counter instance names contain the process PID:
            # pid_1234_...
            if ($Sample.InstanceName -match 'pid_(\d+)_') {

                $PidFromCounter = [int]$Matches[1]

                if ($PythonPids -contains $PidFromCounter) {
                    $TotalBytes += [double]$Sample.CookedValue
                }
            }
        }

        $MemoryMB = $TotalBytes / 1MB

        $Entry = [PSCustomObject]@{
            Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
            DedicatedMemoryMB = [math]::Round($MemoryMB, 2)
        }

        $Samples += $Entry

        Write-Host (
            "`rDedicated GPU memory: {0,10:N2} MB" -f $MemoryMB
        ) -NoNewline

        Start-Sleep -Milliseconds $IntervalMs
    }
}
finally {

    Write-Host ""
    Write-Host ""

    if ($Samples.Count -eq 0) {
        Write-Host "No GPU memory samples were collected."
        exit
    }

    $Samples | Export-Csv `
        -Path $OutputPath `
        -NoTypeInformation `
        -Encoding UTF8

    $Values = $Samples.DedicatedMemoryMB

    $Minimum = ($Values | Measure-Object -Minimum).Minimum
    $Maximum = ($Values | Measure-Object -Maximum).Maximum
    $Average = ($Values | Measure-Object -Average).Average

    Write-Host "============================================================"
    Write-Host "GPU memory results"
    Write-Host "============================================================"
    Write-Host ("Samples:                  {0}" -f $Samples.Count)
    Write-Host ("Minimum dedicated memory: {0:N2} MB" -f $Minimum)
    Write-Host ("Maximum dedicated memory: {0:N2} MB" -f $Maximum)
    Write-Host ("Average dedicated memory: {0:N2} MB" -f $Average)
    Write-Host ""
    Write-Host "Results saved to:"
    Write-Host $OutputPath
    Write-Host "============================================================"
}