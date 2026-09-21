param(
    [ValidateSet('Pilot','Collect','Fit')][string]$Mode = 'Pilot',
    [string]$OutputDir = '',
    [string]$BCCheckpoint = 'teacher_runs/v15_improved/bc_v1_formal/bc_best.pt',
    [int]$EpisodesPerCell = 0,
    [int]$Workers = 8,
    [int]$Seed = 0,
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
    if (-not $OutputDir) {
        $OutputDir = if ($Mode -eq 'Pilot') {
            'teacher_runs/v22_trajectory_envelope/pilot_v2'
        } else {
            'teacher_runs/v22_trajectory_envelope/campaign_v1'
        }
    }
    if ($EpisodesPerCell -eq 0) { $EpisodesPerCell = if ($Mode -eq 'Pilot') { 1 } else { 8 } }
    if ($Seed -eq 0) { $Seed = if ($Mode -eq 'Pilot') { 24110001 } else { 24200001 } }
    $arguments = @('-u','-m','teacher_rl.trajectory_envelope_rl')
    if ($Mode -in @('Pilot','Collect')) {
        $arguments += @('collect','--init',$BCCheckpoint,'--out',$OutputDir,
            '--episodes-per-cell',$EpisodesPerCell,'--workers',$Workers,'--seed',$Seed)
    }
    else {
        $arguments += @('fit','--init',$BCCheckpoint,'--out',$OutputDir,
            '--ridge-lambda',$RidgeLambda)
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "V22 $Mode failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
