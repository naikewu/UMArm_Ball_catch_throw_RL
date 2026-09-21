param(
    [ValidateSet('Search', 'Audit', 'Collect')]
    [string]$Mode = 'Search',
    [int]$Workers = 8,
    [int]$ScreenEpisodes = 32,
    [int]$ValidationEpisodes = 64,
    [int]$Top = 3,
    [int]$Episodes = 1000,
    [int]$Seed = 0,
    [int]$ValidationSeed = 0,
    [int]$DesignSeed = 0,
    [string]$Recipe = 'teacher_runs/v15_improved/search/selected_recipe.json',
    [string]$RootDir = ''
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'
Push-Location $PSScriptRoot
try {
    if ($Mode -eq 'Search') {
        if (-not $RootDir) { $RootDir = 'teacher_runs/v15_improved/search' }
        $arguments = @('-u', '-m', 'teacher_rl.improved_teacher', 'search',
            '--workers', $Workers, '--screen-episodes', $ScreenEpisodes,
            '--validation-episodes', $ValidationEpisodes, '--top', $Top,
            '--out', $RootDir)
        if ($Seed -gt 0) { $arguments += @('--seed', $Seed) }
        if ($ValidationSeed -gt 0) { $arguments += @('--validation-seed', $ValidationSeed) }
        if ($DesignSeed -gt 0) { $arguments += @('--design-seed', $DesignSeed) }
    }
    elseif ($Mode -eq 'Audit') {
        if (-not $RootDir) { $RootDir = 'teacher_runs/v15_improved/audit' }
        $arguments = @('-u', '-m', 'teacher_rl.improved_teacher', 'audit',
            '--workers', $Workers, '--episodes', $Episodes, '--recipe', $Recipe,
            '--out', $RootDir)
        if ($Seed -gt 0) { $arguments += @('--seed', $Seed) }
        if ($DesignSeed -gt 0) { $arguments += @('--design-seed', $DesignSeed) }
    }
    else {
        if (-not $RootDir) { $RootDir = 'teacher_runs/v15_improved/dataset_1000' }
        $arguments = @('-u', '-m', 'teacher_rl.reachable_collection',
            '--workers', $Workers, '--episodes', $Episodes, '--recipe', $Recipe,
            '--out', $RootDir)
        if ($Seed -gt 0) { $arguments += @('--seed', $Seed) }
        if ($DesignSeed -gt 0) { $arguments += @('--design-seed', $DesignSeed) }
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "V15 $Mode failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }
