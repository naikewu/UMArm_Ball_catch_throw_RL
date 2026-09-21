param(
    [ValidateSet('Collect', 'BC', 'Train', 'Resume', 'Evaluate', 'DAgger')]
    [string]$Mode = 'Train',
    [int]$Updates = 100,
    [int]$Episodes = 0,
    [int]$Workers = 4,
    [string]$RunDir = 'teacher_runs/ppo_full',
    [string]$DataDir = 'teacher_runs/dataset_v1'
)

$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'
Push-Location $PSScriptRoot
try {
    $bc = 'teacher_runs/bc/bc_best.pt'
    switch ($Mode) {
        'Collect' {
            if ($Episodes -eq 0) { $Episodes = 96 }
            $cli = @('collect', '--episodes', $Episodes, '--seed', 401, '--workers', $Workers, '--out', $DataDir)
        }
        'BC' { $cli = @('bc', '--data', $DataDir, '--out', 'teacher_runs/bc', '--epochs', 60) }
        'Train' { $cli = @('ppo', '--data', $DataDir, '--init', $bc, '--out', $RunDir, '--updates', $Updates) }
        'Resume' {
            $cli = @('ppo', '--data', $DataDir, '--init', $bc, '--out', $RunDir,
                '--resume', (Join-Path $RunDir 'ppo_latest.pt'), '--updates', $Updates)
        }
        'Evaluate' {
            if ($Episodes -eq 0) { $Episodes = 6 }
            $checkpoint = Join-Path $RunDir 'ppo_best.pt'
            if (-not (Test-Path -LiteralPath $checkpoint)) { $checkpoint = $bc }
            $cli = @('evaluate', '--checkpoint', $checkpoint, '--episodes', $Episodes,
                '--seed', 91001, '--out', (Join-Path $RunDir 'heldout_evaluation.json'))
        }
        'DAgger' {
            if ($Episodes -eq 0) { $Episodes = 24 }
            $cli = @('dagger', '--checkpoint', $bc, '--episodes', $Episodes, '--seed', 2001,
                '--workers', $Workers, '--out', $DataDir, '--beta', 0.5)
        }
    }
    & $python -u -m teacher_rl @cli
    if ($LASTEXITCODE -ne 0) { throw "teacher_rl failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }
