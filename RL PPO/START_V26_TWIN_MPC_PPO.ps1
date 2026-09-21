param(
    [ValidateSet('verify','oracle','mpc_pilot','hybrid_validate','collect')]
    [string]$Mode = 'verify',
    [int]$Workers = 8,
    [switch]$Smoke
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$Python = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found: $Python"
}

$Root = if ($Smoke) { 'teacher_runs/v26_twin_mpc_ppo_smoke' } else { 'teacher_runs/v26_twin_mpc_ppo' }

function Invoke-V26Stage {
    param([string[]]$Arguments)
    & $Python -u -m teacher_rl.twin_mpc_ppo @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "V26 stage failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

switch ($Mode) {
    'verify' {
        Invoke-V26Stage @('verify', '--out', "$Root/stage0_verify")
    }
    'oracle' {
        # The shared scenario manifest needs at least five episodes so its
        # validation partition is non-empty, including in smoke mode.
        $Episodes = if ($Smoke) { 5 } else { 200 }
        $Arguments = @('oracle', '--verification', "$Root/stage0_verify/verification.json",
            '--out', "$Root/stage1_oracle", '--episodes', "$Episodes", '--workers', "$Workers")
        if ($Smoke) { $Arguments += '--smoke' }
        Invoke-V26Stage $Arguments
    }
    'mpc_pilot' {
        $Episodes = if ($Smoke) { 5 } else { 100 }
        $MpcOut = if ($Smoke) { "$Root/stage2_causal_mpc_pilot_v3" } else { "$Root/stage2_causal_mpc_pilot" }
        $MpcOracle = if ($Smoke) { 'teacher_runs/v26_twin_mpc_ppo/stage1_oracle/oracle_report.json' } else { "$Root/stage1_oracle/oracle_report.json" }
        $MpcArguments = @('--oracle', $MpcOracle,
            '--out', $MpcOut, '--episodes', "$Episodes",
            '--workers', "$Workers")
        if ($Smoke) { $MpcArguments += @('--seed', '33900201', '--design-seed', '20261301') }
        & $Python -u -m teacher_rl.twin_causal_mpc @MpcArguments
        if ($LASTEXITCODE -ne 0) { throw "V26 causal MPC pilot failed with exit code $LASTEXITCODE" }
    }
    'hybrid_validate' {
        $Episodes = if ($Smoke) { 5 } else { 100 }
        $HybridOut = "$Root/stage3_hybrid_validation"
        $HybridArguments = @(
            '--oracle', 'teacher_runs/v26_twin_mpc_ppo/stage1_oracle/oracle_report.json',
            '--development', 'teacher_runs/v26_twin_mpc_ppo/stage2_causal_mpc_pilot/pilot_report.json',
            '--out', $HybridOut, '--episodes', "$Episodes", '--workers', "$Workers")
        if ($Smoke) { $HybridArguments += @('--seed', '33900301', '--design-seed', '20261303') }
        & $Python -u -m teacher_rl.twin_hybrid_gate @HybridArguments
        if ($LASTEXITCODE -ne 0) { throw "V26 hybrid validation failed with exit code $LASTEXITCODE" }
    }
    'collect' {
        $Episodes = if ($Smoke) { 5 } else { 1300 }
        $Contexts = if ($Smoke) { 1 } else { 1000 }
        $CollectionOut = "$Root/stage4_dataset"
        $CollectionArguments = @(
            '--qualification', 'teacher_runs/v26_twin_mpc_ppo/stage3_hybrid_validation/validation_report.json',
            '--out', $CollectionOut, '--episodes', "$Episodes", '--contexts', "$Contexts",
            '--workers', "$Workers")
        if ($Smoke) {
            $CollectionArguments += @('--seed', '33900401', '--design-seed', '20261305', '--smoke')
        }
        & $Python -u -m teacher_rl.twin_dataset_collection @CollectionArguments
        if ($LASTEXITCODE -ne 0) { throw "V26 dataset collection failed with exit code $LASTEXITCODE" }
    }
}
