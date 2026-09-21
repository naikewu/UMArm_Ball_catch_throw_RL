param(
    [switch]$RecreateVenv
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $ProjectRoot ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"

if ($RecreateVenv -and (Test-Path -LiteralPath $Venv)) {
    Remove-Item -LiteralPath $Venv -Recurse -Force
}

if (-not (Test-Path -LiteralPath $Python)) {
    $Bootstrap = Get-Command py -ErrorAction SilentlyContinue
    if ($null -eq $Bootstrap) {
        throw "No workspace venv exists and Python launcher 'py' was not found. Install Python 3.12+ first."
    }
    & py -3.12 -m venv $Venv
}

& $Python -m pip install --upgrade pip
& $Python -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
& $Python -c "import mujoco, numpy, scipy, torch; print('ready:', 'mujoco', mujoco.__version__, 'torch', torch.__version__)"
