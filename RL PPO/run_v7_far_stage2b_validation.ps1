param(
    [int]$TotalSteps = 5000,
    [int]$MppiSamples = 16,
    [int]$MppiHorizon = 8,
    [string]$OutDir = "runs/catch_throw_v7_far_stage2b_validation"
)

$ErrorActionPreference = "Stop"
$TrainingScript = Join-Path $PSScriptRoot "run_physical_training.ps1"
& $TrainingScript `
    -TotalSteps $TotalSteps `
    -MppiSamples $MppiSamples `
    -MppiHorizon $MppiHorizon `
    -StartStage 2 `
    -StartStage2Level 2 `
    -FixedStage `
    -CurriculumEvalEvery 10 `
    -CurriculumEvalEpisodes 20 `
    -OutDir $OutDir
