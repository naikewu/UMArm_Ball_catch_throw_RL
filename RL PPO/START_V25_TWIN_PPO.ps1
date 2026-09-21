param(
    [ValidateSet('verify','preview','pretrain','train','gate','screen','evaluate','all')]
    [string]$Mode = 'verify',
    [int]$Workers = 8,
    [switch]$Resume,
    [switch]$Smoke
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$Python = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found: $Python"
}

$Root = if ($Smoke) { 'teacher_runs/v25_twin_residual_smoke' } else { 'teacher_runs/v25_twin_residual' }
$Candidate = if ($Smoke) { "$Root/stage3_candidate/candidate_best.pt" } else { "$Root/stage3_candidate/candidate_qualified.pt" }

function Invoke-V25Stage {
    param([string[]]$Arguments)
    & $Python -u -m teacher_rl.twin_residual_rl @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "V25 stage failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Run-Verify {
    Invoke-V25Stage @('verify', '--out', "$Root/stage1_verify")
}

function Run-Preview {
    $Episodes = if ($Smoke) { 5 } else { 48 }
    $Horizon = if ($Smoke) { '.5' } else { '2.0' }
    Invoke-V25Stage @('preview', '--verification', "$Root/stage1_verify/verification.json",
        '--out', "$Root/stage1_preview", '--episodes', "$Episodes", '--workers', "$Workers",
        '--preview-horizon', $Horizon)
}

function Run-Pretrain {
    $Epochs = if ($Smoke) { 2 } else { 300 }
    $Minimum = if ($Smoke) { 1 } else { 100 }
    $Hidden = if ($Smoke) { 16 } else { 32 }
    Invoke-V25Stage @('pretrain', '--preview', "$Root/stage1_preview/preview_dataset.json",
        '--out', "$Root/stage2_pretrain/policy_init.pt", '--epochs', "$Epochs",
        '--minimum-preview-decisions', "$Minimum", '--hidden', "$Hidden")
}

function Run-Train {
    $Arguments = @('train', '--pretrain', "$Root/stage2_pretrain/policy_init.pt",
        '--out', "$Root/stage3_candidate", '--workers', "$Workers")
    if ($Smoke) {
        $Arguments += @('--updates','1','--episodes-per-update','5','--eval-every','1',
            '--eval-episodes','5','--update-epochs','1','--minibatch-size','16',
            '--early-stop-patience','0','--smoke')
    }
    if ($Resume) { $Arguments += '--resume' }
    Invoke-V25Stage $Arguments
}

function Run-Gate {
    $Arguments = @('gate', '--candidate', $Candidate,
        '--training-dir', "$Root/stage3_candidate", '--out', "$Root/stage4_gate/gate.pt",
        '--workers', "$Workers")
    if ($Smoke) {
        $Arguments += @('--episodes','10','--ensemble','2','--epochs','2','--batch-size','4',
            '--hidden','8','--smoke')
    }
    Invoke-V25Stage $Arguments
}

function Run-Screen {
    $Arguments = @('screen', '--candidate', $Candidate, '--gate', "$Root/stage4_gate/gate.pt",
        '--out', "$Root/stage5_screen", '--workers', "$Workers")
    if ($Smoke) { $Arguments += @('--episodes','5','--smoke') }
    Invoke-V25Stage $Arguments
}

function Run-Evaluate {
    if ($Smoke) { throw 'The independent 80-scenario evaluation cannot run in smoke mode.' }
    Invoke-V25Stage @('evaluate', '--candidate', $Candidate, '--gate', "$Root/stage4_gate/gate.pt",
        '--screen-pass', "$Root/stage5_screen/screen_passed.json",
        '--out', "$Root/stage5_evaluation_80", '--episodes', '80', '--workers', "$Workers")
}

switch ($Mode) {
    'verify'   { Run-Verify }
    'preview'  { Run-Preview }
    'pretrain' { Run-Pretrain }
    'train'    { Run-Train }
    'gate'     { Run-Gate }
    'screen'   { Run-Screen }
    'evaluate' { Run-Evaluate }
    'all' {
        Run-Verify
        Run-Preview
        Run-Pretrain
        Run-Train
        Run-Gate
        Run-Screen
        if (-not $Smoke) { Run-Evaluate }
    }
}
