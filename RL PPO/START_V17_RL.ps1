param(
    [ValidateSet('Pilot', 'Sensitivity', 'Train', 'Resume', 'Evaluate', 'Smoke')]
    [string]$Mode = 'Pilot',
    [string]$BCCheckpoint = 'teacher_runs/v15_improved/bc_v1_formal/bc_best.pt',
    [string]$PilotDir = 'teacher_runs/v17_buffered/pilot_formal',
    [string]$OutputDir = 'teacher_runs/v17_buffered/rl_formal',
    [string]$Config = '',
    [int]$Episodes = 40,
    [int]$Workers = 8,
    [ValidateSet(1, 2, 3)][int]$PressureAuthority = 1,
    [ValidateSet('match','catch_pressure','catch_kp','catch_kd','blend','buffer_pressure','buffer_kd','all')]
    [string]$Dimension = 'catch_pressure',
    [int]$PilotSeed = 6100001,
    [int]$Updates = 100,
    [int]$EpisodesPerUpdate = 32,
    [int]$EvalEpisodes = 80,
    [int]$EvalEvery = 10,
    [int]$Seed = 7100001,
    [int]$EvalSeed = 7200001,
    [int]$TestSeed = 7300001
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Missing Python: $python" }
Push-Location $PSScriptRoot
try {
    $arguments = @('-u', '-m', 'teacher_rl.buffered_rl')
    if ($Mode -in @('Pilot', 'Sensitivity')) {
        if ($Mode -eq 'Sensitivity' -and -not $PSBoundParameters.ContainsKey('PilotDir')) {
            $PilotDir = "teacher_runs/v17_buffered/sensitivity_${PressureAuthority}psi_$Dimension"
        }
        if ($Mode -eq 'Sensitivity' -and -not $PSBoundParameters.ContainsKey('Episodes')) { $Episodes = 10 }
        $arguments += @('pilot', '--init', $BCCheckpoint, '--out', $PilotDir,
            '--episodes', $Episodes, '--workers', $Workers, '--seed', $PilotSeed,
            '--pressure-authority', $PressureAuthority)
        if ($Mode -eq 'Sensitivity') { $arguments += @('--sensitivity', '--dimension', $Dimension) }
    }
    elseif ($Mode -eq 'Evaluate') {
        if (-not $PSBoundParameters.ContainsKey('Episodes')) { $Episodes = 200 }
        $arguments += @('evaluate', '--init', $BCCheckpoint,
            '--checkpoint', (Join-Path $OutputDir 'ppo_best.pt'),
            '--out', (Join-Path $OutputDir "evaluation_$Episodes"),
            '--episodes', $Episodes, '--workers', $Workers, '--seed', $TestSeed)
    }
    else {
        if (-not $Config) {
            $selection = if ($Mode -eq 'Smoke') { 'candidate_config.json' } else { 'selected_config.json' }
            $Config = Join-Path $PilotDir $selection
        }
        if (-not (Test-Path -LiteralPath $Config)) {
            throw "A passing Pilot is required before training. Missing: $Config"
        }
        if ($Mode -eq 'Smoke') {
            if (-not $PSBoundParameters.ContainsKey('Updates')) { $Updates = 1 }
            if (-not $PSBoundParameters.ContainsKey('EpisodesPerUpdate')) { $EpisodesPerUpdate = 5 }
            if (-not $PSBoundParameters.ContainsKey('EvalEpisodes')) { $EvalEpisodes = 5 }
            if (-not $PSBoundParameters.ContainsKey('EvalEvery')) { $EvalEvery = 1 }
            if (-not $PSBoundParameters.ContainsKey('OutputDir')) { $OutputDir = 'teacher_runs/v17_buffered/rl_smoke' }
            if (-not $PSBoundParameters.ContainsKey('Seed')) { $Seed = 8100001 }
            if (-not $PSBoundParameters.ContainsKey('EvalSeed')) { $EvalSeed = 8200001 }
        }
        $arguments += @('train', '--init', $BCCheckpoint, '--config', $Config,
            '--out', $OutputDir, '--updates', $Updates, '--episodes-per-update', $EpisodesPerUpdate,
            '--workers', $Workers, '--eval-episodes', $EvalEpisodes, '--eval-every', $EvalEvery,
            '--seed', $Seed, '--eval-seed', $EvalSeed)
        if ($Mode -eq 'Resume') { $arguments += '--resume' }
        if ($Mode -eq 'Smoke') { $arguments += '--smoke' }
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "V17 $Mode failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }
