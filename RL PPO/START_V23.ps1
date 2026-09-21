param(
    [ValidateSet('Sweep','Fit','Pilot','Screen','Validate','Collect','BC','BCEvaluate','PPO','PPOTest','All')]
    [string]$Mode = 'Pilot',
    [string]$RunDir = 'teacher_runs/v23_contextual',
    [int]$Workers = 8,
    [int]$Updates = 100
)
$ErrorActionPreference = 'Stop'
$v23Python = Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'
$v23Workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$v23CanModules = Join-Path $v23Workspace 'umarm-mjx-rl-catch-and-place-pivot/mjx_experiments/umarm_can'
$v23PreviousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = @($v23CanModules, $v23Workspace, $v23PreviousPythonPath) -join [IO.Path]::PathSeparator
function Invoke-V23([string[]]$Arguments) {
    & $v23Python '-u' '-m' 'teacher_rl.contextual_training' @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "V23 stage did not pass (exit $LASTEXITCODE). Read gate_result.json before continuing."
    }
}
Push-Location $PSScriptRoot
try {
    $v23Teacher = "$RunDir/teacher_v1/teacher.pt"
    $v23BC = "$RunDir/bc_v1/bc_best.pt"
    $v23Stages = if ($Mode -eq 'All') { @('Sweep','Fit','Pilot','Screen','Validate','Collect','BC','BCEvaluate','PPO','PPOTest') } else { @($Mode) }
    foreach ($v23Stage in $v23Stages) {
        switch ($v23Stage) {
            'Sweep' {
                & $v23Python '-u' '-m' 'teacher_rl.contextual_rl' 'sweep' '--out' "$RunDir/sweep_v1" '--episodes' '20' '--workers' $Workers
                if ($LASTEXITCODE -ne 0) { throw "V23 sweep failed: $LASTEXITCODE" }
            }
            'Fit' { Invoke-V23 @('fit','--data',"$RunDir/sweep_v1",'--out',"$RunDir/teacher_v1") }
            'Pilot' { Invoke-V23 @('gate','--stage','pilot','--checkpoint',$v23Teacher,'--out',"$RunDir/pilot_v1",'--seed','26200001','--design-seed','20261012','--workers',"$Workers") }
            'Screen' { Invoke-V23 @('gate','--stage','screen','--checkpoint',$v23Teacher,'--prerequisite',"$RunDir/pilot_v1",'--out',"$RunDir/screen_v1",'--seed','26300001','--design-seed','20261013','--workers',"$Workers") }
            'Validate' { Invoke-V23 @('gate','--stage','validate','--checkpoint',$v23Teacher,'--prerequisite',"$RunDir/screen_v1",'--out',"$RunDir/validation_v1",'--episodes','80','--seed','26400001','--design-seed','20261014','--workers',"$Workers") }
            'Collect' { Invoke-V23 @('collect','--checkpoint',$v23Teacher,'--prerequisite',"$RunDir/validation_v1",'--out',"$RunDir/dataset_v1",'--episodes','1000','--seed','26500001','--design-seed','20261015','--workers',"$Workers") }
            'BC' { Invoke-V23 @('bc','--data',"$RunDir/dataset_v1",'--out',"$RunDir/bc_v1",'--epochs','100') }
            'BCEvaluate' { Invoke-V23 @('gate','--stage','bc-evaluate','--checkpoint',$v23BC,'--out',"$RunDir/bc_evaluation_v1",'--episodes','80','--seed','26600001','--design-seed','20261016','--workers',"$Workers") }
            'PPO' { Invoke-V23 @('ppo','--checkpoint',$v23BC,'--prerequisite',"$RunDir/bc_evaluation_v1",'--out',"$RunDir/ppo_v1",'--updates',"$Updates",'--episodes','16','--seed','26700001','--validation-seed','26800001','--design-seed','20261017','--workers',"$Workers") }
            'PPOTest' { Invoke-V23 @('gate','--stage','ppo-test','--checkpoint',"$RunDir/ppo_v1/ppo_candidate.pt",'--out',"$RunDir/ppo_test_v1",'--episodes','80','--seed','26900001','--design-seed','20261019','--workers',"$Workers") }
        }
    }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $v23PreviousPythonPath
}
