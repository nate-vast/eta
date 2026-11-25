---
name: eta
description: Estimate how long the working agent needs to finish its current task. Use for /eta, time-remaining questions, and /eta stats.
---

# /eta

For `stats long`, run `python3 ~/.eta/eta.py stats --min-remaining-min 5` (use the requested threshold if supplied). For `stats`, run `python3 ~/.eta/eta.py stats`. For a task that is finished, say so. If blocked on a user decision, identify it.

For a running task, privately choose your own best estimate of **total remaining wall-clock minutes to verified completion** before looking up times or past results. Include waits, retries and verification. Fractional minutes work. Keep that estimate private as a scoring baseline. Use measured mode for every user-facing ETA:

```bash
python3 ~/.eta/eta.py estimate --measured --agent-min <minutes> --model "<working model>" --reference "<remaining activity, stage and scale>" --agent <harness> --task-id "<unique task identity>"
```

Run from the project directory. Use `claude`, `codex` or `pi` for a known harness. For another harness omit `--agent` and include its name in `--model`. Reuse the same task identity across check-ins for one job. Use a fresh identity for each new job, including repeated jobs in one chat. A session ID plus the originating message ID works. Keep the reference label consistent across comparable jobs, e.g. `implement one small Python function and pass acceptance tests`. If a recurring job’s label is unfamiliar, `references --agent <harness> --model "<working model>"` lists previously used labels without revealing timings. Reuse a label only when the remaining stage and scale match.

- When a runner reports progress for **all remaining variable work**, pass `--batch <completed units> <total units> <elapsed seconds for those units>`. Use a measured count and wall time from comparably sized units. For multiple concurrent workers use their aggregate completed count over the same elapsed wall time.
- Use `--wait-min` only for known additional, nonoverlapping waiting time. The agent's own total already includes it. An uncertain download or queue estimate belongs in the agent's total.
- For a running test or build, pass `--command "<exact shell command>" --elapsed-s <seconds since this attempt started> --tail-min <minutes after success>`. The script reads successful runs from this project's local transcripts, including runs where nobody asked for an ETA. For GitHub Actions use `--ci-job "<job URL>" --tail-min <minutes after success>`. The script reads the running step and compares past runs at the same point in that step, including work after it. Use measured post-success work for the tail. If that work cannot be timed, report the job ETA separately and state that whole-task timing is unavailable. Use zero tail only when the successful job completes the request.
- While working on a task with an open forecast, issue a new forecast on an observed queue exit, retry, overrun of the reported range, or material scope change. Keep the task ID, lock a fresh whole-task estimate, and pass `--event started|retry|overrun|scope-change`. Reset elapsed time on retry. On failure, reassess the fix and rerun before giving a whole-task ETA. Keep all issued IDs for closure at the same finish.
- Without a progress counter, the script uses confirmed outcomes from comparable tasks in this project and model. With three comparable finishes it uses their median observed duration. When the script reports insufficient evidence, say “No measured ETA yet,” identify the next observable checkpoint, and keep the ID so completion supplies timing history.

Post the finish time, remaining duration, rough range and basis as soon as the command returns. Use a progress message while continuing the task. Keep each printed prediction ID. At verified completion, run `python3 ~/.eta/eta.py done --id <id> [<another id> ...]` immediately. A blocked, interrupted or failed task stays open. If the launcher is missing, run `scripts/eta.py` beside this skill file.
