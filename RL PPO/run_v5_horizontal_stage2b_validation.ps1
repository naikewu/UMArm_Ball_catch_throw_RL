param(
    [int]$TotalSteps = 5000,
    [int]$MppiSamples = 16,
    [int]$MppiHorizon = 8
)

$ErrorActionPreference = "Stop"
Write-Warning "v5 used a near-launch early curriculum. Forwarding to the v6 far-launch validation."
$ValidationScript = Join-Path $PSScriptRoot "run_v6_far_stage2b_validation.ps1"
& $ValidationScript `
    -TotalSteps $TotalSteps `
    -MppiSamples $MppiSamples `
    -MppiHorizon $MppiHorizon
