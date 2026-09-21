param(
    [ValidateSet('Train', 'Resume', 'Evaluate')]
    [string]$Mode = 'Train',
    [string]$DataDir = 'teacher_runs/v15_improved/dataset_1000_formal',
    [string]$BCCheckpoint = 'teacher_runs/v15_improved/bc_v1_formal/bc_best.pt',
    [string]$OutputDir = 'teacher_runs/v15_improved/rl_v1_formal',
    [string]$Checkpoint = '',
    [int]$Updates = 100,
    [int]$EpisodesPerUpdate = 8,
    [int]$Workers = 8,
    [int]$Threads = 4,
    [int]$EvalEvery = 10,
    [int]$EvalEpisodes = 80,
    [int]$Episodes = 200,
    [int]$Seed = 100001,
    [int]$EvalSeed = 2000001,
    [int]$TestSeed = 3000001,
    [int]$DesignSeed = 20260930,
    [double]$LearningRate = 3e-5,
    [double]$CriticLearningRate = 3e-4,
    [double]$QualityWeight = 1.0,
    [double]$FocusFraction = 0.25
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Python environment missing: $python" }
Push-Location $PSScriptRoot
try {
    $culture = [System.Globalization.CultureInfo]::InvariantCulture
    if ($Mode -eq 'Evaluate') {
        if (-not $Checkpoint) { $Checkpoint = Join-Path $OutputDir 'ppo_best.pt' }
        $arguments = @('-u', '-m', 'teacher_rl.improved_rl', 'evaluate',
            '--init', $BCCheckpoint, '--checkpoint', $Checkpoint,
            '--episodes', $Episodes, '--workers', $Workers, '--seed', $TestSeed,
            '--out', (Join-Path $OutputDir "evaluation_$Episodes"))
    }
    else {
        $arguments = @('-u', '-m', 'teacher_rl.improved_rl', 'train',
            '--data', $DataDir, '--init', $BCCheckpoint, '--out', $OutputDir,
            '--updates', $Updates, '--episodes-per-update', $EpisodesPerUpdate,
            '--workers', $Workers, '--threads', $Threads, '--eval-every', $EvalEvery,
            '--eval-episodes', $EvalEpisodes, '--seed', $Seed, '--eval-seed', $EvalSeed,
            '--design-seed', $DesignSeed, '--lr', $LearningRate.ToString($culture),
            '--critic-lr', $CriticLearningRate.ToString($culture),
            '--quality-weight', $QualityWeight.ToString($culture),
            '--focus-fraction', $FocusFraction.ToString($culture))
        if ($Mode -eq 'Resume') {
            $arguments += @('--resume', (Join-Path $OutputDir 'ppo_latest.pt'))
        }
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "V15 RL $Mode failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }
