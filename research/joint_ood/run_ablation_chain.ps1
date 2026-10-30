# Runs the spread ablations one after another once the seeds 3-4 launcher has exited.
# Start detached:  Start-Process pwsh -ArgumentList "-File research/joint_ood/run_ablation_chain.ps1" -WindowStyle Hidden
param([int]$WaitPid = 42812)
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent | Split-Path -Parent)
$py = "C:\Users\gengl\venvs\torch-cu124\Scripts\python.exe"
$J = "research/joint_ood"
$log = "$J/ablation_chain.log"
function Say($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Add-Content $log }

function Push($what, $out) {
    & $py "$J/summarize_spread.py" (Get-ChildItem "$out/*.json").FullName > "$out/summary.txt" 2>&1
    & $py "$J/analyze_spread.py" (Get-ChildItem "$out/*.json").FullName > "$out/analysis.txt" 2>&1
    git add "$out/*.json" "$out/summary.txt" "$out/analysis.txt" "$J/ablation_chain.log"
    git commit -q -m "spread ablation: $what`n`nCo-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
    git pull -q --rebase origin claude/project-thread-j063fs
    git push -q -u origin claude/project-thread-j063fs
    Say "pushed $what (git exit $LASTEXITCODE)"
}

function Stage($what, $out, $gridArgs) {
    New-Item -ItemType Directory -Force $out | Out-Null
    # references depend only on (n, seed), so reuse the main grid's
    Copy-Item "$J/spread_grid/ref_*.json" $out -Force
    Say "start $what"
    $p = Start-Process -PassThru -Wait -WindowStyle Hidden -FilePath $py -ArgumentList "$J/run_spread_grid.py --workers 5 --out $out $gridArgs" -RedirectStandardOutput "$out/runner.log" -RedirectStandardError "$out/runner.err"
    $fails = (Select-String -Path "$out/runner.log" -Pattern FAIL).Count
    Say "end $what exit=$($p.ExitCode) FAIL=$fails"
    Push $what $out
}

Say "waiting for pid $WaitPid"
# pid 0 is the Idle process on Windows and always exists, so 0 means "don't wait"
while ($WaitPid -gt 0 -and (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue)) { Start-Sleep 60 }
# never overlap spread.py jobs left running by an earlier launcher (they finish and save on their own)
while (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'spread\.py' -and $_.CommandLine -notmatch 'run_spread_grid' }) { Start-Sleep 60 }
Say "chain start"
# horizon stages first (cheap); pair_knn is ~10x slower per policy at n=8, so it runs last
Stage "horizon 5" "$J/abl_h5" "--ns 4,8 --seeds 3 --data assignment --configs none:0,ens:1,joint_knn:1 -- --horizon 5"
Stage "horizon 10" "$J/abl_h10" "--ns 4,8 --seeds 3 --data assignment --configs none:0,ens:1,joint_knn:1 -- --horizon 10"
Stage "pair_knn on main grid" "$J/spread_pair" "--ns 2,4,8 --seeds 5 --seed_start 0 --configs pair_knn:1"
Say "chain done"
