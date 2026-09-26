[CmdletBinding()]
param(
    [switch]$SetupCpuDemo,
    [switch]$ExportModels,
    [string]$Device = "Snapdragon X Elite CRD"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $ProjectRoot

try {
    if ($SetupCpuDemo) {
        $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
        if (-not (Test-Path $Python)) {
            if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
                throw "Install Python 3.10-3.13 or create a .venv before running CPU demo setup."
            }
            & py -3 -m venv .venv
            if ($LASTEXITCODE -ne 0) {
                throw "Could not create the CPU demo virtual environment."
            }
        }

        & $Python -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)"
        if ($LASTEXITCODE -ne 0) {
            throw "The CPU demo requires Python 3.10 through 3.13."
        }

        & $Python -m pip install --upgrade pip
        if ($LASTEXITCODE -ne 0) { throw "Could not upgrade pip." }
        & $Python -m pip install -e ".[demo]"
        if ($LASTEXITCODE -ne 0) { throw "CPU demo dependency installation failed." }
        & $Python -m pip install "numpy>=1.24,<2.0" "onnx>=1.12,<2.0" "onnxslim>=0.1.82"
        if ($LASTEXITCODE -ne 0) { throw "ONNX export helper installation failed." }

        $YoloCli = Join-Path $ProjectRoot ".venv\Scripts\yolo.exe"
        Push-Location (Join-Path $ProjectRoot "models")
        try {
            & $YoloCli export model=yolov8n.pt format=onnx nms=False
            if ($LASTEXITCODE -ne 0) { throw "Ultralytics YOLOv8n ONNX export failed." }
        }
        finally {
            Pop-Location
        }

        Write-Host "CPU demo model ready at $(Join-Path $ProjectRoot 'models\yolov8n.onnx')"
        Write-Host 'Set $env:NETRA_DETECTOR_MODEL to that path before starting Netra.'
        return
    }

    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "The Python launcher 'py' was not found. Install 64-bit Python 3.10 and enable the launcher."
    }

    & py -3.10 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)"
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.10 is required. Install 64-bit Python 3.10, then rerun this script."
    }

    $Python = Join-Path $ProjectRoot ".venv310\Scripts\python.exe"
    if (-not (Test-Path $Python)) {
        & py -3.10 -m venv .venv310
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create the Python 3.10 virtual environment."
        }
    }

    $Architecture = & $Python -c "import platform; print(platform.machine())"
    if ($Architecture -notin @("AMD64", "x86_64")) {
        throw "qai_hub_models currently requires AMD64 Python on Windows. Run setup/export from a 64-bit x64 Python host; deploy artifacts to the Snapdragon PC separately."
    }

    & $Python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "Could not upgrade pip."
    }

    & $Python -m pip install -e ".[export]"
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed. Check Python 3.10 and network access, then retry."
    }

    $HubCli = Join-Path $ProjectRoot ".venv310\Scripts\qai-hub.exe"
    $ModelsCli = Join-Path $ProjectRoot ".venv310\Scripts\qai-hub-models.exe"

    if (-not [string]::IsNullOrWhiteSpace($env:QAI_HUB_TOKEN)) {
        & $HubCli configure --api_token $env:QAI_HUB_TOKEN
        if ($LASTEXITCODE -ne 0) {
            throw "AI Hub token configuration failed. Confirm QAI_HUB_TOKEN is valid."
        }
    }
    else {
        Write-Warning "QAI_HUB_TOKEN is not set. Local dependencies were installed, but AI Hub exports require a token."
    }

    if ($ExportModels) {
        if ([string]::IsNullOrWhiteSpace($env:QAI_HUB_TOKEN)) {
            throw "Set QAI_HUB_TOKEN in this PowerShell session before exporting models."
        }

        & $ModelsCli install yolov8_det
        if ($LASTEXITCODE -ne 0) { throw "Could not install YOLOv8 export dependencies." }
        & $ModelsCli export yolov8_det --runtime onnx --precision w8a8 --device $Device
        if ($LASTEXITCODE -ne 0) { throw "YOLOv8 export failed." }

        & $ModelsCli install whisper_tiny
        if ($LASTEXITCODE -ne 0) { throw "Could not install Whisper Tiny export dependencies." }
        & $ModelsCli export whisper_tiny --runtime qnn_context_binary --precision float --device $Device
        if ($LASTEXITCODE -ne 0) { throw "Whisper Tiny export failed." }

        & $ModelsCli fetch Qwen3-VL-2B-Instruct --runtime geniex_llamacpp --precision q4_0
        if ($LASTEXITCODE -ne 0) { throw "Qwen3-VL model download failed." }

        Write-Host "Export/download commands completed. Keep the returned asset paths for the runtime integration phase."
    }
    else {
        Write-Host "Setup complete. Set QAI_HUB_TOKEN and rerun with -ExportModels when ready to export/download model assets."
    }
}
finally {
    Pop-Location
}
