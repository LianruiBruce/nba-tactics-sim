# Stops the "nba-joint-ood" scheduled task and every process it started (wrapper, chain, launchers,
# spread.py jobs). Finished cells stay on disk; rerunning the task resumes and skips them.
Stop-ScheduledTask -TaskName "nba-joint-ood" -ErrorAction SilentlyContinue
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -match 'run_resume_all|run_ablation_chain|run_spread_grid|spread\.py' -and $_.ProcessId -ne $PID } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; "stopped $($_.ProcessId)" }
