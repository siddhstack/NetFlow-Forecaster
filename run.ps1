param(
    [ValidateSet("synthetic", "kaggle", "kaggle_opt", "dataset_opt", "simulate", "live", "deploy", "destroy", "train", "visualize")]
    [string]$Mode = "synthetic",
    [int]$Samples = 720,
    [int]$Interval = 1,
    [int]$Epochs = 130,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$MlDir = Join-Path $ProjectDir "ml"
$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunDir = Join-Path (Join-Path $ProjectDir "runs") "$($RunStamp)_$Mode"
$DataFile = Join-Path (Join-Path $RunDir "raw_data") "telemetry.csv"

function Log-Step($Message) {
    Write-Host ""
    Write-Host "[$(Get-Date -Format HH:mm:ss)] $Message" -ForegroundColor Cyan
}

function Get-PythonCommand {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -notlike "*WindowsApps*") {
        return "python"
    }

    $cmd = Get-Command py -ErrorAction SilentlyContinue
    if ($cmd) {
        return "py"
    }

    throw "Python was not found. Run .\scripts\setup_windows.ps1 -SkipDocker -SkipWsl, then reopen this terminal."
}

function ConvertTo-WslPath($Path) {
    $result = & wsl.exe wslpath -a $Path 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $result) {
        throw "WSL is not ready. Run .\scripts\setup_windows.ps1, open Ubuntu once, then run scripts/setup_wsl_containerlab.sh inside Ubuntu."
    }
    return ($result | Select-Object -First 1)
}

function Invoke-WslBash($Command) {
    & wsl.exe bash -lc $Command
    if ($LASTEXITCODE -ne 0) {
        throw "WSL command failed: $Command"
    }
}

function Test-WindowsDockerLab {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        return $false
    }
    $names = & docker ps --format "{{.Names}}" 2>$null
    if ($LASTEXITCODE -ne 0) {
        return $false
    }
    return ($names -match "^clab-ai-traffic-lab-").Count -gt 0
}

function Test-WslDockerLab {
    $names = & wsl.exe -d Ubuntu -- bash -lc "docker ps --format '{{.Names}}'" 2>$null
    if ($LASTEXITCODE -ne 0) {
        return $false
    }
    return ($names -match "^clab-ai-traffic-lab-").Count -gt 0
}

function Invoke-ContainerLab($Action) {
    if (Get-Command containerlab -ErrorAction SilentlyContinue) {
        Push-Location (Join-Path $ProjectDir "containerlab")
        & containerlab $Action -t topology.clab.yml
        Pop-Location
        if ($LASTEXITCODE -ne 0) {
            throw "containerlab $Action failed."
        }
        return
    }

    $wslProject = ConvertTo-WslPath $ProjectDir
    Invoke-WslBash "cd '$wslProject/containerlab' && sudo containerlab $Action -t topology.clab.yml"
}

function Invoke-ModelPipeline($Python, $InputCsv, $OutputDir) {
    Log-Step "Training LSTM"
    Push-Location $MlDir
    & $Python enhanced_train.py --data $InputCsv --epochs $Epochs --output-dir $OutputDir
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        throw "Model training failed."
    }

    Log-Step "Building dashboard"
    & $Python visualize.py --data (Join-Path (Join-Path $OutputDir "raw_data") "telemetry.csv") --output-dir $OutputDir
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        throw "Dashboard generation failed."
    }

    Log-Step "Evaluating model"
    & $Python evaluate_model.py --run-dir $OutputDir
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        throw "Model evaluation failed."
    }

    Log-Step "Exporting readable model report"
    & $Python export_model_report.py --run-dir $OutputDir
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        throw "Readable model export failed."
    }
    Pop-Location

    Log-Step "Cleaning empty run folders"
    Push-Location $ProjectDir
    & $Python scripts\cleanup_runs.py
    Pop-Location
}

function New-RunFolder($RunDir) {
    New-Item -ItemType Directory -Path (Join-Path $RunDir "raw_data") -Force | Out-Null
}

function Show-RunArtifacts($RunDir) {
    Write-Host "Artifacts:"
    Write-Host "  $(Join-Path (Join-Path $RunDir 'raw_data') 'telemetry.csv')"
    Write-Host "  $(Join-Path (Join-Path $RunDir 'images') 'traffic_prediction_dashboard.png')"
    Write-Host "  $(Join-Path (Join-Path $RunDir 'images') 'model_evaluation_dashboard.png')"
    Write-Host "  $(Join-Path (Join-Path $RunDir 'json') 'evaluation_summary.json')"
    Write-Host "  $(Join-Path (Join-Path $RunDir 'json') 'model_metadata.json')"

    $lstmModel = Join-Path (Join-Path $RunDir "model") "lstm_model.pth"
    $datasetModel = Join-Path (Join-Path $RunDir "model") "dataset_model.joblib"
    if (Test-Path $lstmModel) {
        Write-Host "  $(Join-Path (Join-Path $RunDir 'model') 'model_readable_report.md')"
        Write-Host "Binary model weights:"
        Write-Host "  $lstmModel"
    } elseif (Test-Path $datasetModel) {
        Write-Host "Dataset model artifact:"
        Write-Host "  $datasetModel"
    }
    Write-Host "Run folder:"
    Write-Host "  $RunDir"
}

function Invoke-KaggleTelemetry($Python, $OutputCsv, $Rows) {
    Push-Location $MlDir
    & $Python load_kaggle_data.py --rows $Rows --output $OutputCsv
    $windowsExit = $LASTEXITCODE
    Pop-Location
    if ($windowsExit -eq 0) {
        return
    }

    Log-Step "Windows Kaggle load failed; trying WSL virtual environment"
    $wslProject = ConvertTo-WslPath $ProjectDir
    $wslOutputCsv = ConvertTo-WslPath $OutputCsv
    Invoke-WslBash "cd '$wslProject' && if [ ! -x .venv-kaggle/bin/python ]; then python3 -m venv .venv-kaggle; fi && .venv-kaggle/bin/python -c 'import kagglehub' 2>/dev/null || .venv-kaggle/bin/python -m pip install 'kagglehub[pandas-datasets]' && .venv-kaggle/bin/python ml/load_kaggle_data.py --rows $Rows --output '$wslOutputCsv'"
}

$NeedsPython = $Mode -in @("synthetic", "kaggle", "kaggle_opt", "dataset_opt", "simulate", "live", "train", "visualize")
if ($NeedsPython) {
    $Python = Get-PythonCommand
    if (-not $SkipInstall) {
        Log-Step "Installing Python dependencies"
        & $Python -m pip install -r (Join-Path $ProjectDir "requirements.txt")
    }
}

switch ($Mode) {
    "synthetic" {
        New-RunFolder $RunDir
        Log-Step "Generating synthetic telemetry"
        Push-Location $MlDir
        & $Python generate_data.py --hours $Samples --output $DataFile --seed 7
        Pop-Location
        Invoke-ModelPipeline $Python $DataFile $RunDir
    }

    "kaggle" {
        New-RunFolder $RunDir
        Log-Step "Loading Kaggle network telemetry"
        Invoke-KaggleTelemetry $Python $DataFile $Samples
        Invoke-ModelPipeline $Python $DataFile $RunDir
    }

    { $_ -in @("kaggle_opt", "dataset_opt") } {
        New-RunFolder $RunDir
        if ($Mode -eq "kaggle_opt") {
            Log-Step "Loading Kaggle network telemetry"
            Invoke-KaggleTelemetry $Python $DataFile $Samples
        } else {
            $sourceData = Join-Path $ProjectDir "ml\telemetry.csv"
            if (-not (Test-Path $sourceData)) {
                throw "dataset_opt needs a standard telemetry CSV at ml\telemetry.csv, or use kaggle_opt for Kaggle data."
            }
            Copy-Item $sourceData $DataFile -Force
        }

        Log-Step "Training dataset spike-aware model"
        Push-Location $MlDir
        & $Python train_dataset_model.py --data $DataFile --output-dir $RunDir --lookback 24 --spike-std 1.2 --spike-oversample 0
        if ($LASTEXITCODE -ne 0) {
            Pop-Location
            throw "Dataset optimized training failed."
        }

        Log-Step "Building dashboard"
        & $Python visualize.py --data (Join-Path (Join-Path $RunDir "raw_data") "telemetry.csv") --output-dir $RunDir
        if ($LASTEXITCODE -ne 0) {
            Pop-Location
            throw "Dashboard generation failed."
        }

        Log-Step "Evaluating model"
        & $Python evaluate_model.py --run-dir $RunDir
        if ($LASTEXITCODE -ne 0) {
            Pop-Location
            throw "Model evaluation failed."
        }
        Pop-Location

        Log-Step "Cleaning empty run folders"
        Push-Location $ProjectDir
        & $Python scripts\cleanup_runs.py
        Pop-Location
    }

    "simulate" {
        New-RunFolder $RunDir
        Log-Step "Collecting simulated telemetry with collector"
        Push-Location $ProjectDir
        & $Python scripts\collect_telemetry.py --mode simulate --samples $Samples --interval $Interval --output $DataFile
        if ($LASTEXITCODE -ne 0) {
            Pop-Location
            throw "Simulated telemetry collection failed."
        }
        Pop-Location
        Invoke-ModelPipeline $Python $DataFile $RunDir
    }

    "live" {
        New-RunFolder $RunDir
        Log-Step "Collecting live ContainerLab telemetry"
        if (Test-WindowsDockerLab) {
            Push-Location $ProjectDir
            & $Python scripts\collect_telemetry.py --mode live --samples $Samples --interval $Interval --output $DataFile
            if ($LASTEXITCODE -ne 0) {
                Pop-Location
                throw "Live telemetry collection failed."
            }
            Pop-Location
        } elseif (Test-WslDockerLab) {
            $wslProject = ConvertTo-WslPath $ProjectDir
            $wslDataFile = ConvertTo-WslPath $DataFile
            Invoke-WslBash "cd '$wslProject' && python3 scripts/collect_telemetry.py --mode live --samples $Samples --interval $Interval --output '$wslDataFile'"
        } else {
            throw "No running ContainerLab containers found. Run .\run.ps1 deploy first."
        }
        Invoke-ModelPipeline $Python $DataFile $RunDir
    }

    "deploy" {
        Log-Step "Deploying ContainerLab topology"
        Invoke-ContainerLab "deploy"
        return
    }

    "destroy" {
        Log-Step "Destroying ContainerLab topology"
        Invoke-ContainerLab "destroy"
        return
    }

    "train" {
        Log-Step "Training LSTM from existing telemetry"
        $ExistingData = Join-Path $MlDir "telemetry.csv"
        New-RunFolder $RunDir
        Invoke-ModelPipeline $Python $ExistingData $RunDir
    }

    "visualize" {
        Log-Step "Building dashboard from existing artifacts"
        $LatestRun = Get-ChildItem (Join-Path $ProjectDir "runs") -Directory -ErrorAction SilentlyContinue |
            Where-Object {
                (Test-Path (Join-Path $_.FullName "results\predictions.csv")) -and
                (Test-Path (Join-Path $_.FullName "results\actuals.csv")) -and
                (Test-Path (Join-Path $_.FullName "results\train_losses.csv"))
            } |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if (-not $LatestRun) {
            throw "No run folders with readable CSV artifacts found. Run .\run.ps1 synthetic first."
        }
        $RunDir = $LatestRun.FullName
        $DataFile = Join-Path (Join-Path $RunDir "raw_data") "telemetry.csv"
        Push-Location $MlDir
        & $Python visualize.py --data $DataFile --output-dir $RunDir
        Pop-Location
    }
}

Log-Step "Done"
Show-RunArtifacts $RunDir
