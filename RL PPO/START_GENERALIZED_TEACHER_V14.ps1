param(
    [ValidateSet('Verify', 'Collect')]
    [string]$Mode = 'Verify',
    [int]$Workers = 8,
    [int]$Episodes = 0,
    [string]$RootDir = 'teacher_runs/v14_generalized'
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'
Push-Location $PSScriptRoot
try {
    switch ($Mode) {
        'Verify' {
            if ($Episodes -eq 0) { $Episodes = 24 }
            $cli = @('verify-batch', '--workers', $Workers, '--episodes', $Episodes,
                '--out', (Join-Path $RootDir 'verification'))
        }
        'Collect' {
            if ($Episodes -eq 0) { $Episodes = 256 }
            $cli = @('collect', '--workers', $Workers, '--episodes', $Episodes,
                '--out', (Join-Path $RootDir 'dataset_pilot'))
        }
    }
    & $python -u -m teacher_rl.generalized_teacher @cli
    if ($LASTEXITCODE -ne 0) { throw "V14 failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }
