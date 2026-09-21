param(
    [ValidateSet('Collect','Fit')][string]$Mode = 'Collect',
    [string]$OutputDir = 'teacher_runs/v21_release_calibration/campaign_v1',
    [string]$BCCheckpoint = 'teacher_runs/v15_improved/bc_v1_formal/bc_best.pt',
    [int]$EpisodesPerCell = 10,
    [int]$Workers = 8,
    [int]$Seed = 21100001,
    [double]$RidgeLambda = 0.25
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Missing Python: $python" }
$canModules = Join-Path $PSScriptRoot '../umarm-mjx-rl-catch-and-place-pivot/mjx_experiments/umarm_can'
if (-not (Test-Path -LiteralPath $canModules)) { throw "Missing CAN control modules: $canModules" }
$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = @($canModules, $workspaceRoot, $previousPythonPath) -join [IO.Path]::PathSeparator
Push-Location $PSScriptRoot
try {
    $arguments = @('-u','-m','teacher_rl.release_calibration_rl')
    if ($Mode -eq 'Collect') {
        $arguments += @('collect','--init',$BCCheckpoint,'--out',$OutputDir,
            '--episodes-per-cell',$EpisodesPerCell,'--workers',$Workers,'--seed',$Seed)
    }
    else {
        $arguments += @('fit','--init',$BCCheckpoint,'--out',$OutputDir,'--ridge-lambda',$RidgeLambda)
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "V21 $Mode failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
