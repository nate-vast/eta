# eta

Ask `/eta` during a task to get a remaining duration, finish time, rough range and the evidence behind them. The agent posts the result and continues working. Claude Code uses `/eta`, Codex supports `$eta` and pi uses `/skill:eta`.

## How the forecast works

The working agent gives its own total remaining time in private before seeing the skill's forecast. The script records both estimates at one timestamp.

- **Measured progress.** When the remaining work is a homogeneous batch, the script uses the actual time per completed unit and the number still to go.
- **Previous command and CI runtimes.** An exact local shell command reads successful executions in up to 40 recent project transcripts from the selected harness. A GitHub Actions job URL retrieves successful jobs matching repository, workflow, branch, event and job name via the authenticated `gh` CLI. Completed attempt metadata is cached locally. For Actions, the skill first looks for five prior successes that were still in the same numbered, named step after the same time in that step. It measures from that point through the end of each historical job and accepts the agent’s guess if it falls inside the middle half of those durations. For an outside guess, a step that historically starts at least 80% of the way through the job (median) uses the nearest quartile. Earlier steps use the elapsed-time median, or the matching step median when elapsed-time history is sparse. When step history is sparse, it uses previous jobs that lasted longer than the current elapsed time. The fallback uses their median remaining duration. Both paths add the explicitly estimated work after success. Failed or cancelled jobs need a new whole-task assessment.
- **Comparable finishes.** For recurring work, the last 40 eligible completed tasks from the same project directory, harness, model and remaining-work label provide a comparison. With at least three distinct tasks, the script scores three options against earlier outcomes. These are the working agent’s estimate, a blend with the historical median, and a correction for the agent’s observed timing bias. Each earlier task is left out when predicting its own outcome. The option with the lowest total absolute error supplies the forecast. Known nonoverlapping waits are removed before fitting and added back at their current duration.
- **Cold start.** With insufficient comparable history or progress data, the output identifies the agent's own estimate as the fallback. The range is descriptive. The trials cover short coding tasks and measurable batches. See [paired accuracy trials](verify/PAIRED.md).

At completion, the agent closes the specific forecast IDs after verifying the requested outcome. Giving up, starting another turn, and transcript inactivity leave paired forecasts open. `/eta stats` reports mean absolute error for both methods, wins, ties, losses, and open forecasts. The legacy call-count method has a separate track record.

## Install

```bash
git clone https://github.com/nate-vast/eta ~/projects/eta
~/projects/eta/install.sh
```

The installer links the shared skill and `~/.eta/eta.py`. It also supports `--copy`, `--dir PATH` and `--uninstall`. Python 3.9+ is required. Run from a consistent project directory. Set `ETA_HOME` to use an isolated history.

On POSIX, ETA restricts its default history directory to its owner and stores history files with mode 0600. A custom `ETA_HOME` keeps its existing directory permissions. The installer preserves unrecognized launcher and helper files. If upgrading an older unmarked `--copy` installation, move those files aside and run the installer again.

## CLI

```bash
python3 ~/.eta/eta.py estimate --agent-min MINUTES --model MODEL --agent codex \
    --reference "remaining stage and scale" --task-id UNIQUE_JOB
python3 ~/.eta/eta.py done --id PREDICTION_ID
python3 ~/.eta/eta.py stats
```

Use `references --model MODEL --agent codex` to look up existing remaining-work labels for this scope. Use `--batch DONE TOTAL SECONDS` for measured progress covering all remaining variable work. `--wait-min` adds a known, nonoverlapping wait. `--json` exposes both predictions in seconds. Multiple forecasts for the same job share a task ID and can be closed together with `done --id ID1 ID2`.

For a running command use `--command "make test" --elapsed-s 120 --tail-min 0`. For GitHub Actions use `--ci-job JOB_URL --tail-min MINUTES`. Supply the time needed after success, including verification or deployment. Change the event to `started`, `retry`, `overrun` or `scope-change` when the agent observes a transition and issues another forecast for the same task. No service runs in the background. Each forecast is scored from its own issue time. See [the longer-task benchmark](verify/COMMANDS.md).

Existing `measure`, `estimate --calls`, `estimate --minutes` and session-based `done` commands still support the previous transcript workflow. Legacy estimates apply their correction to variable work and add known waits afterward. For the open-model invocation tests, see [verification results](verify/RESULTS.md).

Run the test suite with `python3 -m unittest discover -s tests -v`.
