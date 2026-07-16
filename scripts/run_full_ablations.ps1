<#!
.SYNOPSIS
Runs the documented full-scale ablation studies sequentially.

.DESCRIPTION
The studies share the same training resources, so this script deliberately
runs candidate selection before spike-loss training.  It writes all detailed
artifacts under runs/full_ablations and publishes only completed summaries to
docs/results.
#>

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$runRoot = Join-Path $repo 'runs/full_ablations'
$docsResults = Join-Path $repo 'docs/results'
New-Item -ItemType Directory -Force $runRoot, $docsResults | Out-Null

function Invoke-Ablation([string[]] $Arguments) {
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Ablation command failed with exit code ${LASTEXITCODE}: python $($Arguments -join ' ')"
    }
}

Invoke-Ablation @(
    'ml/ablation_selection.py', '--data', 'ml/telemetry.csv',
    '--output-dir', (Join-Path $runRoot 'selection')
)
Copy-Item (Join-Path $runRoot 'selection/ablation_selection_telemetry.csv') $docsResults -Force
Copy-Item (Join-Path $runRoot 'selection/ablation_selection_summary.json') $docsResults -Force

Invoke-Ablation @(
    'ml/ablation_spike_loss.py', '--data', 'ml/telemetry.csv',
    '--output-dir', (Join-Path $runRoot 'spike_loss')
)
Copy-Item (Join-Path $runRoot 'spike_loss/ablation_spike_loss_summary.csv') $docsResults -Force
Copy-Item (Join-Path $runRoot 'spike_loss/ablation_spike_loss_summary.json') $docsResults -Force
