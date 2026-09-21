param(
    [int]$TotalSteps = 100000,
    [int]$RolloutSteps = 512,
    [double[]]$TargetOffset = @(0.0, 0.80, 0.0),
    [double]$TargetRadius = 0.06,
    [int]$MppiSamples = 16,
    [int]$MppiHorizon = 8,
    [string]$OutDir = "runs/catch_throw_v9_rendezvous",
    [ValidateRange(1, 5)]
    [int]$StartStage = 1,
    [ValidateRange(1, 3)]
    [int]$StartStage2Level = 1,
    [int]$StageMinUpdates = 20,
    [double]$StageSuccessThreshold = 0.80,
    [int]$CurriculumEvalEvery = 10,
    [int]$CurriculumEvalEpisodes = 20,
    [switch]$FixedStage
)

$ErrorActionPreference = "Stop"
if ($TargetOffset.Count -ne 3) {
    throw "TargetOffset must contain exactly X, Y, Z metres."
}
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Run .\setup.ps1 first."
}

Push-Location $PSScriptRoot
try {
    $Arguments = @(
        "-m", "rl_ppo.train_catch_throw",
        "--total-steps", $TotalSteps,
        "--rollout-steps", $RolloutSteps,
        "--target-offset", $TargetOffset[0], $TargetOffset[1], $TargetOffset[2],
        "--target-radius", $TargetRadius,
        "--mppi-samples", $MppiSamples,
        "--mppi-horizon", $MppiHorizon,
        "--out-dir", $OutDir,
        "--start-stage", $StartStage,
        "--start-stage2-level", $StartStage2Level,
        "--stage-min-updates", $StageMinUpdates,
        "--stage-success-threshold", $StageSuccessThreshold,
        "--curriculum-eval-every", $CurriculumEvalEvery,
        "--curriculum-eval-episodes", $CurriculumEvalEpisodes
    )
    if ($FixedStage) {
        $Arguments += "--fixed-stage"
    }
    & $Python @Arguments
}
finally {
    Pop-Location
}
