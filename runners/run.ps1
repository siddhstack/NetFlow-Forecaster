param(
    [ValidateSet("synthetic", "kaggle", "kaggle_opt", "dataset_opt", "public_benchmark", "simulate", "live", "deploy", "destroy", "train", "visualize", "benchmark")]
    [string]$Mode = "synthetic",
    [int]$Samples = 720,
    [int]$Interval = 1,
    [int]$Epochs = 130,
    [int]$LIpn = -1,
    [double]$TargetQuality = 90,
    [int]$MaxAttempts = 24,
    [double]$MaxMinutes = -1,
    [bool]$AutoBenchmark = $true,
    [bool]$Learn = $true,
    [switch]$SkipInstall,
    [switch]$Help
)

if ($Help) {
    Write-Host @"
NetFlow-Forecaster Runner

USAGE:
    .\runners\run.ps1 -Mode <mode> [options]

MODES:
    synthetic              Generate synthetic data and train (default)
    kaggle                 Download and train on Kaggle dataset
    public_benchmark       Download CICIDS2017 and train
    benchmark              Search for best candidate (with meta-policy)
    train                  Train a single model on existing data
    visualize              Generate dashboards from completed run

OPTIONS:
    -Samples <int>         Synthetic data samples (default: 720)
    -Epochs <int>          Training epochs (default: 130)
    -TargetQuality <double> Quality gate for benchmark (default: 90)
    -MaxAttempts <int>     Max attempts in benchmark search (default: 24)
    -SkipInstall           Skip pip install
    -Help                  Show this message

EXAMPLES:
    .\runners\run.ps1 -Mode synthetic -Epochs 60
    .\runners\run.ps1 -Mode benchmark -TargetQuality 90
    .\runners\run.ps1 -Mode public_benchmark -Samples 5000
"@
    exit 0
}

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

function Get-PythonCommand {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -notlike "*WindowsApps*") {
        return "python"
    }

    $cmd = Get-Command py -ErrorAction SilentlyContinue
    if ($cmd) {
        return "py"
    }

    throw "Python was not found. Install Python 3.10+, then reopen this terminal."
}

function Write-Status {
    param([string]$Message)
    $timestamp = (Get-Date).ToString("HH:mm:ss")
    Write-Host "[$timestamp] $Message" -ForegroundColor Cyan
}

function Write-Error-Status {
    param([string]$Message)
    $timestamp = (Get-Date).ToString("HH:mm:ss")
    Write-Host "[$timestamp] ERROR: $Message" -ForegroundColor Red
}

$Python = Get-PythonCommand
Write-Status "Using Python: $($Python)"
Write-Status "Mode: $Mode | Epochs: $Epochs | Samples: $Samples"

$ArgsList = @(
    (Join-Path $ScriptDir "run.py"),
    $Mode,
    "--samples", $Samples,
    "--interval", $Interval,
    "--epochs", $Epochs,
    "--l-ipn", $LIpn,
    "--target-quality", $TargetQuality,
    "--max-attempts", $MaxAttempts
)

if ($MaxMinutes -ge 0) {
    $ArgsList += @("--max-minutes", $MaxMinutes)
}

if ($AutoBenchmark) {
    $ArgsList += "--auto-benchmark"
} else {
    $ArgsList += "--no-auto-benchmark"
}

if ($Learn) {
    $ArgsList += "--learn"
} else {
    $ArgsList += "--no-learn"
}

if ($SkipInstall) {
    $ArgsList += "--skip-install"
}

Push-Location $ProjectDir
try {
    & $Python @ArgsList
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Write-Error-Status "Process exited with code $exitCode"
        exit $exitCode
    }
    Write-Status "Completed successfully (exit code 0)"
} catch {
    Write-Error-Status "Fatal error: $_"
    exit 1
} finally {
    Pop-Location
}
