param(
    [ValidateSet('Pretrain','Smoke','Train','Resume','Evaluate','All')]
    [string]$Mode = 'Smoke',
    [string]$RunDir = 'teacher_runs/v15_anchor_dynamic_ppo_v2',
    [int]$Workers = 8,
    [int]$Updates = 100,
    [int]$EpisodesPerUpdate = 32
)
$ErrorActionPreference = 'Stop'
$anchoredPython = Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'
$anchoredWorkspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$anchoredCanModules = Join-Path $anchoredWorkspace 'umarm-mjx-rl-catch-and-place-pivot/mjx_experiments/umarm_can'
$anchoredPreviousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = @($PSScriptRoot, $anchoredCanModules, $anchoredWorkspace, $anchoredPreviousPythonPath) -join [IO.Path]::PathSeparator

function Invoke-AnchoredPPO([string[]]$Arguments) {
    & $anchoredPython '-u' '-m' 'teacher_rl.anchored_dynamic_rl' @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "V15-anchored PPO stage failed (exit $LASTEXITCODE)."
    }
}

Push-Location $PSScriptRoot
try {
    $pretrain = "$RunDir/pretrain/policy_init.pt"
    $formal = "$RunDir/formal"
    $smoke = "$RunDir/smoke_guarded"
    $stages = if ($Mode -eq 'All') { @('Pretrain','Smoke','Train','Evaluate') } else { @($Mode) }
    foreach ($stage in $stages) {
        switch ($stage) {
            'Pretrain' {
                Invoke-AnchoredPPO @('pretrain','--out',$pretrain)
            }
            'Smoke' {
                if (-not (Test-Path $pretrain)) { Invoke-AnchoredPPO @('pretrain','--out',$pretrain) }
                if (Test-Path "$smoke/training_summary.json") {
                    Write-Host 'Smoke training is already complete; keeping the recorded result.'
                }
                else {
                    Invoke-AnchoredPPO @('train','--pretrain',$pretrain,
                        '--out',$smoke,'--updates','1','--episodes-per-update','5',
                        '--eval-every','1','--eval-episodes','5','--workers',"$Workers",
                        '--seed','31200001','--eval-seed','31300001',
                        '--design-seed','20261104','--random-seed','20261104','--smoke')
                }
            }
            'Train' {
                if (-not (Test-Path $pretrain)) { Invoke-AnchoredPPO @('pretrain','--out',$pretrain) }
                Invoke-AnchoredPPO @('train','--pretrain',$pretrain,'--out',$formal,
                    '--updates',"$Updates",'--episodes-per-update',"$EpisodesPerUpdate",
                    '--eval-every','5','--eval-episodes','40','--workers',"$Workers")
            }
            'Resume' {
                Invoke-AnchoredPPO @('train','--pretrain',$pretrain,'--out',$formal,
                    '--resume',"$formal/ppo_latest.pt",'--updates',"$Updates",
                    '--episodes-per-update',"$EpisodesPerUpdate",'--eval-every','5',
                    '--eval-episodes','40','--workers',"$Workers")
            }
            'Evaluate' {
                if (-not (Test-Path "$formal/ppo_best.pt")) {
                    throw 'No policy has passed paired validation yet; ppo_best.pt was not published.'
                }
                Invoke-AnchoredPPO @('evaluate','--checkpoint',"$formal/ppo_best.pt",
                    '--out',"$formal/evaluation_80",'--episodes','80','--workers',"$Workers")
            }
        }
    }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $anchoredPreviousPythonPath
}
