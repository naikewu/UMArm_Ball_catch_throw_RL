param(
    [ValidateSet('Pilot', 'Train', 'Resume', 'Evaluate', 'Smoke')]
    [string]$Mode = 'Pilot',
    [string]$BCCheckpoint = 'teacher_runs/v15_improved/bc_v1_formal/bc_best.pt',
    [string]$PilotDir = 'teacher_runs/v16_continuous/pilot_formal',
    [string]$OutputDir = 'teacher_runs/v16_continuous/rl_formal',
    [string]$Config = '',
    [int]$Episodes = 40,
    [int]$Workers = 8,
    [int]$Updates = 100,
    [int]$EpisodesPerUpdate = 32,
    [int]$EvalEpisodes = 80,
    [int]$EvalEvery = 10,
    [int]$Seed = 5100001,
    [int]$EvalSeed = 5200001,
    [int]$TestSeed = 5300001
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Missing Python: $python" }
Push-Location $PSScriptRoot
try {
    $arguments = @('-u', '-m', 'teacher_rl.continuous_rl')
    if ($Mode -eq 'Pilot') {
        $arguments += @('pilot', '--init', $BCCheckpoint, '--out', $PilotDir,
            '--episodes', $Episodes, '--workers', $Workers)
    }
    elseif ($Mode -eq 'Evaluate') {
        if (-not $PSBoundParameters.ContainsKey('Episodes')) { $Episodes = 200 }
        $arguments += @('evaluate', '--init', $BCCheckpoint,
            '--checkpoint', (Join-Path $OutputDir 'ppo_best.pt'),
            '--out', (Join-Path $OutputDir "evaluation_$Episodes"),
            '--episodes', $Episodes, '--workers', $Workers, '--seed', $TestSeed)
    }
    else {
        if (-not $Config) { $Config = Join-Path $PilotDir 'selected_config.json' }
        if (-not (Test-Path -LiteralPath $Config)) {
            throw "A passing Pilot is required before training. Missing: $Config"
        }
        if ($Mode -eq 'Smoke') {
            if (-not $PSBoundParameters.ContainsKey('Updates')) { $Updates = 2 }
            if (-not $PSBoundParameters.ContainsKey('EpisodesPerUpdate')) { $EpisodesPerUpdate = 8 }
            if (-not $PSBoundParameters.ContainsKey('EvalEpisodes')) { $EvalEpisodes = 10 }
            if (-not $PSBoundParameters.ContainsKey('EvalEvery')) { $EvalEvery = 1 }
            if (-not $PSBoundParameters.ContainsKey('OutputDir')) { $OutputDir = 'teacher_runs/v16_continuous/rl_smoke' }
            if (-not $PSBoundParameters.ContainsKey('Seed')) { $Seed = 9100001 }
            if (-not $PSBoundParameters.ContainsKey('EvalSeed')) { $EvalSeed = 9200001 }
        }
        $arguments += @('train', '--init', $BCCheckpoint, '--config', $Config,
            '--out', $OutputDir, '--updates', $Updates, '--episodes-per-update', $EpisodesPerUpdate,
            '--workers', $Workers, '--eval-episodes', $EvalEpisodes, '--eval-every', $EvalEvery,
            '--seed', $Seed, '--eval-seed', $EvalSeed)
        if ($Mode -eq 'Resume') { $arguments += '--resume' }
        if ($Mode -eq 'Smoke') { $arguments += '--smoke' }
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "V16 $Mode failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }
