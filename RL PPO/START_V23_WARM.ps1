param(
    [ValidateSet('Sweep','Fit','Pilot','Screen','Validate','Collect','BC','BCEvaluate','PPO','PPOTest','All')]
    [string]$Mode = 'Pilot',
    [string]$RunDir = 'teacher_runs/v23_contextual',
    [string]$Calibration = 'teacher_runs/v23_contextual/warm_kernel_v1/release_calibration_v23_kernel.json',
    [int]$Workers = 12,
    [int]$Updates = 100
)
$ErrorActionPreference = 'Stop'
$v23Python = Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'
$v23Workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$v23CanModules = Join-Path $v23Workspace 'umarm-mjx-rl-catch-and-place-pivot/mjx_experiments/umarm_can'
$v23PreviousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = @($v23CanModules, $v23Workspace, $v23PreviousPythonPath) -join [IO.Path]::PathSeparator
function Invoke-V23Warm([string[]]$Arguments) {
    & $v23Python '-u' '-m' 'teacher_rl.contextual_training' @Arguments '--calibration' $Calibration
    if ($LASTEXITCODE -ne 0) {
        throw "V23 stage did not pass (exit $LASTEXITCODE). Details are saved in gate_result.json."
    }
}
Push-Location $PSScriptRoot
try {
    $v23Teacher = "$RunDir/warm_teacher_v2/teacher.pt"
    $v23BC = "$RunDir/warm_bc_v2/bc_best.pt"
    $v23Stages = if ($Mode -eq 'All') { @('Sweep','Fit','Pilot','Screen','Validate','Collect','BC','BCEvaluate','PPO','PPOTest') } else { @($Mode) }
    foreach ($v23Stage in $v23Stages) {
        switch ($v23Stage) {
            'Sweep' {
                & $v23Python '-u' '-m' 'teacher_rl.contextual_warm_sweep' '--calibration' $Calibration '--out' "$RunDir/warm_sweep_v1" '--episodes' '40' '--workers' $Workers
                if ($LASTEXITCODE -ne 0) { throw "V23 warm sweep failed: $LASTEXITCODE" }
                & $v23Python '-u' '-m' 'teacher_rl.contextual_warm_sweep' '--calibration' $Calibration '--out' "$RunDir/warm_sweep_increment_v2" '--episodes' '80' '--workers' $Workers '--seed' '28200001' '--design-seed' '20261101'
                if ($LASTEXITCODE -ne 0) { throw "V23 warm incremental sweep failed: $LASTEXITCODE" }
            }
            'Fit' { Invoke-V23Warm @('fit-outcome','--data',"$RunDir/warm_sweep_v1", "$RunDir/warm_sweep_increment_v2",'--out',"$RunDir/warm_teacher_v2") }
            'Pilot' { Invoke-V23Warm @('gate','--stage','pilot','--checkpoint',$v23Teacher,'--out',"$RunDir/warm_pilot_v2",'--seed','28300001','--design-seed','20261102','--workers',"$Workers") }
            'Screen' { Invoke-V23Warm @('gate','--stage','screen','--checkpoint',$v23Teacher,'--prerequisite',"$RunDir/warm_pilot_v2",'--out',"$RunDir/warm_screen_v2",'--seed','28400001','--design-seed','20261103','--workers',"$Workers") }
            'Validate' { Invoke-V23Warm @('gate','--stage','validate','--checkpoint',$v23Teacher,'--prerequisite',"$RunDir/warm_screen_v2",'--out',"$RunDir/warm_validation_v2",'--episodes','80','--seed','28500001','--design-seed','20261104','--workers',"$Workers") }
            'Collect' { Invoke-V23Warm @('collect','--checkpoint',$v23Teacher,'--prerequisite',"$RunDir/warm_validation_v2",'--out',"$RunDir/warm_dataset_v2",'--episodes','1000','--seed','28600001','--design-seed','20261105','--workers',"$Workers") }
            'BC' { Invoke-V23Warm @('bc','--data',"$RunDir/warm_dataset_v2",'--out',"$RunDir/warm_bc_v2",'--epochs','100') }
            'BCEvaluate' { Invoke-V23Warm @('gate','--stage','bc-evaluate','--checkpoint',$v23BC,'--out',"$RunDir/warm_bc_evaluation_v2",'--episodes','80','--seed','28700001','--design-seed','20261106','--workers',"$Workers") }
            'PPO' { Invoke-V23Warm @('ppo','--checkpoint',$v23BC,'--prerequisite',"$RunDir/warm_bc_evaluation_v2",'--out',"$RunDir/warm_ppo_v2",'--updates',"$Updates",'--episodes','16','--seed','28800001','--validation-seed','28900001','--design-seed','20261107','--workers',"$Workers") }
            'PPOTest' { Invoke-V23Warm @('gate','--stage','ppo-test','--checkpoint',"$RunDir/warm_ppo_v2/ppo_candidate.pt",'--out',"$RunDir/warm_ppo_test_v2",'--episodes','80','--seed','29000001','--design-seed','20261108','--workers',"$Workers") }
        }
    }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $v23PreviousPythonPath
}
