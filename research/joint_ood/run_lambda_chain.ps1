# Penalty-strength sweep for spread.py: lambda 0.3 / 3 / 10 for every penalty (lambda 1, and ens 5,
# already exist in spread_grid and spread_pair). Two stages, pushed after each.
# Start detached:  Start-Process pwsh -ArgumentList "-NoProfile -File research/joint_ood/run_lambda_chain.ps1" -WindowStyle Hidden
# Stop with:       pwsh -File research/joint_ood/stop_runs.ps1   (finished cells are kept; rerun this script to resume)
param([int]$Workers = 5)
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent | Split-Path -Parent)
$py = "C:\Users\gengl\venvs\torch-cu124\Scripts\python.exe"
$J = "research/joint_ood"
$log = "$J/lambda_chain.log"
function Say($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Add-Content $log }

function Push($what, $out) {
    & $py "$J/summarize_spread.py" (Get-ChildItem "$out/*.json").FullName > "$out/summary.txt" 2>&1
    & $py "$J/analyze_spread.py" (Get-ChildItem "$out/*.json").FullName > "$out/analysis.txt" 2>&1
    git add "$out/*.json" "$out/summary.txt" "$out/analysis.txt" $log
    git commit -q -m "spread lambda sweep: $what`n`nCo-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
    git pull -q --rebase --autostash origin claude/project-thread-j063fs
    git push -q -u origin claude/project-thread-j063fs
    Say "pushed $what (git exit $LASTEXITCODE)"
}

function Stage($what, $out, $gridArgs) {
    New-Item -ItemType Directory -Force $out | Out-Null
    # references depend only on (n, seed), so reuse the main grid's
    Copy-Item "$J/spread_grid/ref_*.json" $out -Force
    Say "start $what"
    # launcher options must come before any '--' (everything after it goes to spread.py)
    $p = Start-Process -PassThru -Wait -WindowStyle Hidden -FilePath $py -ArgumentList "$J/run_spread_grid.py --workers $Workers --out $out $gridArgs" -RedirectStandardOutput "$out/runner.log" -RedirectStandardError "$out/runner.err"
    $fails = (Select-String -Path "$out/runner.log" -Pattern FAIL).Count
    Say "end $what exit=$($p.ExitCode) FAIL=$fails"
    Push $what $out
}

$common = "--ns 4,8 --seeds 3 --seed_start 0 --data assignment --archs additive,joint,attention"
Say "chain start (pid $PID, workers $Workers)"
Stage "ens/agent_knn/joint_knn lambda 0.3,3,10" "$J/spread_lam" "$common --configs ens:0.3,ens:3,ens:10,agent_knn:0.3,agent_knn:3,agent_knn:10,joint_knn:0.3,joint_knn:3,joint_knn:10"
Stage "pair_knn lambda 0.3,3,10" "$J/spread_lam_pair" "$common --configs pair_knn:0.3,pair_knn:3,pair_knn:10"
Say "chain done"
