param(
    [int]$TotalSteps = 120000,
    [int]$MppiSamples = 16,
    [int]$MppiHorizon = 8,
    [string]$OutDir = "runs/catch_throw_v9_rendezvous_full_curriculum"
)

$ErrorActionPreference = "Stop"
$TrainingScript = Join-Path $PSScriptRoot "run_physical_training.ps1"
& $TrainingScript `
    -TotalSteps $TotalSteps `
    -RolloutSteps 512 `
    -TargetOffset @(0.0, 0.80, 0.0) `
    -TargetRadius 0.06 `
    -MppiSamples $MppiSamples `
    -MppiHorizon $MppiHorizon `
    -StartStage 1 `
    -StartStage2Level 1 `
    -StageMinUpdates 20 `
    -StageSuccessThreshold 0.80 `
    -CurriculumEvalEvery 5 `
    -CurriculumEvalEpisodes 20 `
    -OutDir $OutDir
