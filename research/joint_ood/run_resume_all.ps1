# Finishes the spread seeds 3-4 grid, pushes it, then runs the ablation chain (which pushes after
# each stage). Meant to run from Task Scheduler (task "nba-joint-ood") so no Claude session owns it.
# Stop with:  pwsh -File research/joint_ood/stop_runs.ps1
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent | Split-Path -Parent)
$py = "C:\Users\gengl\venvs\torch-cu124\Scripts\python.exe"
$J = "research/joint_ood"
$log = "$J/resume_all.log"
function Say($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Add-Content $log }

Say "start (pid $PID)"
$G = "$J/spread_grid"
$p = Start-Process -PassThru -Wait -WindowStyle Hidden -FilePath $py `
    -ArgumentList "$J/run_spread_grid.py --ns 2,4,8 --seeds 2 --seed_start 3 --workers 5 --out $G" `
    -RedirectStandardOutput "$G/runner_s34_resume.log" -RedirectStandardError "$G/runner_s34_resume.err"
Say "seeds 3-4 exit=$($p.ExitCode) FAIL=$((Select-String -Path "$G/runner_s34_resume.log" -Pattern FAIL).Count)"

$files = (Get-ChildItem "$G/*.json").FullName
& $py "$J/summarize_spread.py" $files > "$J/spread_main_summary.txt" 2>&1
& $py "$J/analyze_spread.py" $files > "$J/spread_main_analysis.txt" 2>&1
git add "$G/*.json" "$J/spread_main_summary.txt" "$J/spread_main_analysis.txt" $log
git commit -q -m "spread seeds 3-4 complete (5 seeds total)`n`nCo-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
git pull -q --rebase origin claude/project-thread-j063fs
git push -q -u origin claude/project-thread-j063fs
Say "pushed seeds 3-4 (git exit $LASTEXITCODE)"

& "C:\Program Files\PowerShell\7\pwsh.exe" -NoProfile -File "$J/run_ablation_chain.ps1" -WaitPid 0
Say "chain finished"
