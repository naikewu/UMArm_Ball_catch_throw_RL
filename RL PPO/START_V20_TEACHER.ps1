param(
    [ValidateSet('Screen','Validate')][string]$Mode = 'Screen',
    [string]$OutputDir = 'teacher_runs/v20_trajectory_release/screen_v1',
    [string]$BCCheckpoint = 'teacher_runs/v15_improved/bc_v1_formal/bc_best.pt',
    [int]$Episodes = 0,
    [int]$Workers = 8,
    [int]$Seed = 0
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
    if ($Episodes -eq 0) { $Episodes = if ($Mode -eq 'Screen') { 20 } else { 80 } }
    if ($Seed -eq 0) { $Seed = if ($Mode -eq 'Screen') { 20100001 } else { 20200001 } }
    $arguments = @('-u','-m','teacher_rl.trajectory_release_rl')
    if ($Mode -eq 'Screen') {
        $arguments += @('screen','--init',$BCCheckpoint,'--out',$OutputDir,'--episodes',$Episodes,
            '--workers',$Workers,'--seed',$Seed)
    }
    else {
        if ($Episodes -lt 80) { throw 'Validate requires at least 80 independent scenarios.' }
        $validationOut = Join-Path $OutputDir "validation_$Episodes"
        $arguments += @('validate','--init',$BCCheckpoint,'--screen',$OutputDir,'--out',$validationOut,
            '--episodes',$Episodes,'--workers',$Workers,'--seed',$Seed)
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "V20 $Mode failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
