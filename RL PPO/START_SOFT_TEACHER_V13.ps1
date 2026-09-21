param(
    [ValidateSet('Search', 'Collect', 'BC', 'Evaluate')]
    [string]$Mode = 'Search',
    [int]$Workers = 4,
    [int]$Episodes = 0,
    [string]$RootDir = 'teacher_runs/v13'
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'
Push-Location $PSScriptRoot
try {
    $recipe = Join-Path $RootDir 'search/selected_recipe.json'
    switch ($Mode) {
        'Search' { $cli = @('search', '--workers', $Workers, '--out', (Join-Path $RootDir 'search')) }
        'Collect' {
            if ($Episodes -eq 0) { $Episodes = 256 }
            $selection = Get-Content -LiteralPath $recipe -Raw | ConvertFrom-Json
            if (-not $selection.improved) { Write-Warning 'Search did not validate an improved teacher; collection will use baseline.' }
            $cli = @('collect', '--recipe', $recipe, '--workers', $Workers, '--episodes', $Episodes,
                '--out', (Join-Path $RootDir 'dataset'))
        }
        'BC' { $cli = @('bc', '--data', (Join-Path $RootDir 'dataset'), '--out', (Join-Path $RootDir 'bc')) }
        'Evaluate' {
            if ($Episodes -eq 0) { $Episodes = 96 }
            $cli = @('evaluate', '--recipe', $recipe, '--workers', $Workers, '--episodes', $Episodes,
                '--checkpoint', (Join-Path $RootDir 'bc/bc_best.pt'), '--out', (Join-Path $RootDir 'evaluation'))
        }
    }
    & $python -u -m teacher_rl.soft_teacher @cli
    if ($LASTEXITCODE -ne 0) { throw "V13 failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }
