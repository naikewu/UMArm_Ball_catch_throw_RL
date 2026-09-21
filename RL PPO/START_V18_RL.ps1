param(
    [ValidateSet('Explore','Resume','Evaluate','Status')][string]$Mode = 'Explore',
    [ValidateSet('Candidate','Latest','Accepted')][string]$Checkpoint = 'Candidate',
    [ValidateSet('catch','throw','joint')][string]$Stage = 'throw',
    [string]$OutputDir = 'teacher_runs/v18_online/explore_throw',
    [string]$BCCheckpoint = 'teacher_runs/v15_improved/bc_v1_formal/bc_best.pt',
    [int]$Updates = 10,
    [int]$EpisodesPerUpdate = 16,
    [int]$EvalEpisodes = 20,
    [int]$EvalEvery = 5,
    [int]$Workers = 8,
    [int]$Seed = 18100001,
    [int]$EvalSeed = 18200001,
    [int]$TestSeed = 18300001,
    [int]$Episodes = 40
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Missing Python: $python" }
Push-Location $PSScriptRoot
try {
    if ($Mode -eq 'Status') {
        & $python -u -m teacher_rl.exploration_status --out $OutputDir --init $BCCheckpoint
        if ($LASTEXITCODE -ne 0) { throw "V18 Status failed with exit code $LASTEXITCODE" }
        return
    }
    $arguments = @('-u','-m','teacher_rl.exploration_rl')
    if ($Mode -eq 'Evaluate') {
        $checkpointName = $Checkpoint.ToLowerInvariant()
        $checkpointPath = Join-Path $OutputDir "ppo_$checkpointName.pt"
        if (-not (Test-Path -LiteralPath $checkpointPath)) { throw "Missing checkpoint: $checkpointPath" }
        $summaryPath = Join-Path $OutputDir 'training_summary.json'
        if ($Checkpoint -eq 'Candidate' -and (Test-Path -LiteralPath $summaryPath)) {
            $summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($summary.best_update -eq 0) {
                Write-Warning 'Candidate is update 0 (zero residual), not a learned improvement. Use -Checkpoint Latest to evaluate the trained model.'
            }
        }
        $evaluationName = if ($Checkpoint -eq 'Candidate') { "evaluation_$Episodes" } else { "evaluation_${checkpointName}_$Episodes" }
        $arguments += @('evaluate','--init',$BCCheckpoint,'--checkpoint',$checkpointPath,
            '--out',(Join-Path $OutputDir $evaluationName),'--episodes',$Episodes,
            '--workers',$Workers,'--seed',$TestSeed)
    }
    else {
        $arguments += @('explore','--init',$BCCheckpoint,'--out',$OutputDir,'--stage',$Stage,
            '--updates',$Updates,'--episodes-per-update',$EpisodesPerUpdate,'--eval-episodes',$EvalEpisodes,
            '--eval-every',$EvalEvery,'--workers',$Workers,'--seed',$Seed,'--eval-seed',$EvalSeed)
        if ($Mode -eq 'Resume') { $arguments += '--resume' }
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "V18 $Mode failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }
