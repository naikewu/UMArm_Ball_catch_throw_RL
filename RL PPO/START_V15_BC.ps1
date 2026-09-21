param(
    [ValidateSet('Train', 'Evaluate')]
    [string]$Mode = 'Train',
    [string]$DataDir = 'teacher_runs/v15_improved/dataset_1000_formal',
    [string]$OutputDir = 'teacher_runs/v15_improved/bc_v1_formal',
    [int]$Epochs = 60,
    [int]$Batch = 256,
    [double]$LearningRate = 3e-4,
    [int]$Threads = 4,
    [int]$Workers = 8,
    [int]$Episodes = 200,
    [int]$Seed = 92001,
    [int]$DesignSeed = 20260928
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'
Push-Location $PSScriptRoot
try {
    if ($Mode -eq 'Train') {
        $arguments = @('-u', '-m', 'teacher_rl.improved_bc', 'train',
            '--data', $DataDir, '--out', $OutputDir,
            '--epochs', $Epochs, '--batch', $Batch,
            '--lr', $LearningRate, '--threads', $Threads)
    }
    else {
        $arguments = @('-u', '-m', 'teacher_rl.improved_bc', 'evaluate',
            '--checkpoint', (Join-Path $OutputDir 'bc_best.pt'),
            '--episodes', $Episodes, '--seed', $Seed,
            '--design-seed', $DesignSeed, '--workers', $Workers,
            '--out', (Join-Path $OutputDir 'evaluation_200'))
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "V15 BC $Mode failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }
